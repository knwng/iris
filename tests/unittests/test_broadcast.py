# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import numpy as np
import pytest
import iris


@pytest.mark.parametrize(
    "value,expected",
    [
        (42, 42),
        (3.14159, 3.14159),
        (True, True),
        (False, False),
        ("Hello, Iris!", "Hello, Iris!"),
        ({"key": "value", "num": 42}, {"key": "value", "num": 42}),
    ],
)
def test_broadcast_scalar(value, expected):
    """Test broadcasting scalar values (int, float, bool, string, dict)."""
    shmem = iris.iris(1 << 20)
    rank = shmem.get_rank()

    val = value if rank == 0 else None
    result = shmem.broadcast(val, source_rank=0)

    if isinstance(expected, float):
        assert abs(result - expected) < 1e-6
    else:
        assert result == expected


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.int32,
        torch.int64,
    ],
)
def test_broadcast_tensor_dtype(dtype):
    """Test broadcasting tensors with different dtypes."""
    shmem = iris.iris(1 << 20)
    rank = shmem.get_rank()

    value = torch.arange(10, dtype=dtype) if rank == 0 else None
    result = shmem.broadcast(value, source_rank=0)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.arange(10))


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (10, 20),
        (5, 10, 15),
    ],
)
def test_broadcast_tensor_shape(shape):
    """Test broadcasting tensors with different shapes."""
    shmem = iris.iris(1 << 25)
    rank = shmem.get_rank()

    value = torch.randn(shape) if rank == 0 else None
    result = shmem.broadcast(value, source_rank=0)

    assert isinstance(result, np.ndarray)
    assert result.shape == shape
