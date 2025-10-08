# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import numpy as np
import sys
import torch
import subprocess
import os

# Auto-detect backend
_is_amd_backend = True
try:
    rt_path = "libamdhip64.so"
    gpu_runtime = ctypes.cdll.LoadLibrary(rt_path)
except OSError:
    try:
        rt_path = "libcudart.so"
        gpu_runtime = ctypes.cdll.LoadLibrary(rt_path)
        _is_amd_backend = False
    except OSError:
        rt_path = "libamdhip64.so"
        gpu_runtime = ctypes.cdll.LoadLibrary(rt_path)


def gpu_try(err):
    if err != 0:
        if _is_amd_backend:
            gpu_runtime.hipGetErrorString.restype = ctypes.c_char_p
            error_string = gpu_runtime.hipGetErrorString(ctypes.c_int(err)).decode("utf-8")
            raise RuntimeError(f"HIP error code {err}: {error_string}")
        else:
            gpu_runtime.cudaGetErrorString.restype = ctypes.c_char_p
            error_string = gpu_runtime.cudaGetErrorString(ctypes.c_int(err)).decode("utf-8")
            raise RuntimeError(f"CUDA error code {err}: {error_string}")


def get_ipc_handle_size():
    """Return the IPC handle size for the current backend."""
    return 64 if _is_amd_backend else 128


class gpuIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * get_ipc_handle_size())]


def open_ipc_handle(ipc_handle_data, rank):
    ptr = ctypes.c_void_p()
    handle_size = get_ipc_handle_size()

    if _is_amd_backend:
        hipIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)
        gpu_runtime.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            gpuIpcMemHandle_t,
            ctypes.c_uint,
        ]
    else:
        gpu_runtime.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            gpuIpcMemHandle_t,
            ctypes.c_uint,
        ]
        cudaIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)

    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != handle_size:
            raise ValueError(f"ipc_handle_data must be a {handle_size}-element uint8 numpy array")
        ipc_handle_bytes = ipc_handle_data.tobytes()
        ipc_handle_data = (ctypes.c_char * handle_size).from_buffer_copy(ipc_handle_bytes)
    else:
        raise TypeError(f"ipc_handle_data must be a numpy.ndarray of dtype uint8 with {handle_size} elements")

    raw_memory = ctypes.create_string_buffer(handle_size)
    ctypes.memset(raw_memory, 0x00, handle_size)
    ipc_handle_struct = gpuIpcMemHandle_t.from_buffer(raw_memory)
    ipc_handle_data_bytes = bytes(ipc_handle_data)
    ctypes.memmove(raw_memory, ipc_handle_data_bytes, handle_size)

    if _is_amd_backend:
        gpu_try(
            gpu_runtime.hipIpcOpenMemHandle(
                ctypes.byref(ptr),
                ipc_handle_struct,
                hipIpcMemLazyEnablePeerAccess,
            )
        )
    else:
        gpu_try(
            gpu_runtime.cudaIpcOpenMemHandle(
                ctypes.byref(ptr),
                ipc_handle_struct,
                cudaIpcMemLazyEnablePeerAccess,
            )
        )

    return ptr.value


def get_ipc_handle(ptr, rank):
    ipc_handle = gpuIpcMemHandle_t()
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipIpcGetMemHandle(ctypes.byref(ipc_handle), ptr))
    else:
        gpu_try(gpu_runtime.cudaIpcGetMemHandle(ctypes.byref(ipc_handle), ptr))
    return ipc_handle


def count_devices():
    device_count = ctypes.c_int()
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipGetDeviceCount(ctypes.byref(device_count)))
    else:
        gpu_try(gpu_runtime.cudaGetDeviceCount(ctypes.byref(device_count)))
    return device_count.value


def set_device(gpu_id):
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipSetDevice(gpu_id))
    else:
        gpu_try(gpu_runtime.cudaSetDevice(gpu_id))


def get_device_id():
    device_id = ctypes.c_int()
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipGetDevice(ctypes.byref(device_id)))
    else:
        gpu_try(gpu_runtime.cudaGetDevice(ctypes.byref(device_id)))
    return device_id.value


def get_cu_count(device_id=None):
    if device_id is None:
        device_id = get_device_id()

    cu_count = ctypes.c_int()

    if _is_amd_backend:
        hipDeviceAttributeMultiprocessorCount = 63
        gpu_try(
            gpu_runtime.hipDeviceGetAttribute(ctypes.byref(cu_count), hipDeviceAttributeMultiprocessorCount, device_id)
        )
    else:
        cudaDevAttrMultiProcessorCount = 16
        gpu_try(gpu_runtime.cudaDeviceGetAttribute(ctypes.byref(cu_count), cudaDevAttrMultiProcessorCount, device_id))

    return cu_count.value


def get_rocm_version():
    if not _is_amd_backend:
        # Not applicable for CUDA
        return (-1, -1)

    major, minor = -1, -1

    # Try hipconfig --path first
    try:
        result = subprocess.run(["hipconfig", "--path"], capture_output=True, text=True, check=True)
        rocm_path = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Then look for $ROCM_PATH environment variable
        rocm_path = os.environ.get("ROCM_PATH")
        if not rocm_path:
            # Finally, try default location
            rocm_path = "/opt/rocm"

    # Try to read version from .info/version file
    try:
        version_file_path = os.path.join(rocm_path, ".info", "version")
        with open(version_file_path, "r") as version_file:
            version = version_file.readline().strip()
            major = int(version.split(".")[0])
            minor = int(version.split(".")[1])
    except (FileNotFoundError, IOError, ValueError, IndexError):
        # If we can't read the version file, return -1, -1
        pass

    return (major, minor)


def get_wall_clock_rate(device_id):
    wall_clock_rate = ctypes.c_int()

    if _is_amd_backend:
        hipDeviceAttributeWallClockRate = 10017
        status = gpu_runtime.hipDeviceGetAttribute(
            ctypes.byref(wall_clock_rate), hipDeviceAttributeWallClockRate, device_id
        )
    else:
        cudaDevAttrClockRate = 13
        status = gpu_runtime.cudaDeviceGetAttribute(ctypes.byref(wall_clock_rate), cudaDevAttrClockRate, device_id)

    gpu_try(status)
    return wall_clock_rate.value


def get_arch_string(device_id=None):
    if device_id is None:
        device_id = get_device_id()

    if _is_amd_backend:
        arch_full = torch.cuda.get_device_properties(device_id).gcnArchName
        arch_name = arch_full.split(":")[0]
        return arch_name
    else:
        # For CUDA, return compute capability
        props = torch.cuda.get_device_properties(device_id)
        return f"sm_{props.major}{props.minor}"


def get_num_xcc(device_id=None):
    if device_id is None:
        device_id = get_device_id()

    if not _is_amd_backend:
        # XCC is AMD-specific, return 1 for CUDA
        return 1

    rocm_major, _ = get_rocm_version()
    if rocm_major < 7:
        return 8
    hipDeviceAttributeNumberOfXccs = 10018
    xcc_count = ctypes.c_int()
    gpu_try(gpu_runtime.hipDeviceGetAttribute(ctypes.byref(xcc_count), hipDeviceAttributeNumberOfXccs, device_id))
    return xcc_count.value


def malloc_fine_grained(size):
    ptr = ctypes.c_void_p()

    if _is_amd_backend:
        hipDeviceMallocFinegrained = 0x1
        gpu_try(gpu_runtime.hipExtMallocWithFlags(ctypes.byref(ptr), size, hipDeviceMallocFinegrained))
    else:
        # CUDA doesn't have direct equivalent, use regular malloc
        gpu_try(gpu_runtime.cudaMalloc(ctypes.byref(ptr), size))

    return ptr


def hip_malloc(size):
    ptr = ctypes.c_void_p()
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipMalloc(ctypes.byref(ptr), size))
    else:
        gpu_try(gpu_runtime.cudaMalloc(ctypes.byref(ptr), size))
    return ptr


def hip_free(ptr):
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipFree(ptr))
    else:
        gpu_try(gpu_runtime.cudaFree(ptr))
