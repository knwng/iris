# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch


def validate_gemm(A, B, C, shmem, atol=1):
    expected = A @ B
    diff_mask = ~torch.isclose(C, expected, atol=atol)
    breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

    if not torch.allclose(C, expected, atol=atol):
        max_diff = (C - expected).abs().max().item()
        shmem.info(f"Max absolute difference: {max_diff}")
        for idx in breaking_indices:
            idx = tuple(idx.tolist())
            computed_val = C[idx]
            expected_val = expected[idx]
            shmem.error(f"Mismatch at index {idx}: C={computed_val}, expected={expected_val}")
            break
        return False

    return True
