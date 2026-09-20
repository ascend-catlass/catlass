# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import pytest
import torch

import torch_catlass
from common import only_on_3510

pytestmark = only_on_3510

FP16_RTOL = 2.0**-9
FP16_ATOL = 2.0**-9
REQUIRED_MATCHED_RATIO = 0.99
FP16_MAX_ABS_ERROR = 0.1
FP16_MIN_NORMAL = 2.0**-14
FP16_MIN_SUBNORMAL = 2.0**-24
FP16_FRACTION_BITS = 10
FP16_MAX_ULP_ERROR = 32.0

SHAPE_SEED_M_FACTOR = 1_000_003
SHAPE_SEED_J_FACTOR = 1_009
SHAPE_SEED_N_FACTOR = 31

FULL_LOAD_A_MAX_K = 2048
DEFAULT_INPUT_RANGE = 5.0
LARGE_K_INPUT_RANGE = 0.1

# Cover both FullLoad Gather modes, a narrow-N tail, and the arbitrary-shape SIMT fallback.
TEST_CASES = [
    (400, 128, 128, 128),
    (500, 256, 256, 768),
    (400, 128, 256, 2048),
    (2, 1, 3, 65537),
]


def _assert_fp16_precision(actual: torch.Tensor, expected: torch.Tensor, shape: tuple[int, int, int, int]) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_error = (actual_f - expected_f).abs()
    mixed_tolerance = FP16_ATOL + FP16_RTOL * expected_f.abs()
    matched_ratio = (abs_error <= mixed_tolerance).float().mean().item()

    abs_expected = expected_f.abs()
    normal_exponent = torch.floor(torch.log2(abs_expected.clamp_min(FP16_MIN_NORMAL)))
    ulp = torch.where(
        abs_expected < FP16_MIN_NORMAL,
        torch.full_like(abs_expected, FP16_MIN_SUBNORMAL),
        torch.pow(torch.full_like(abs_expected, 2.0), normal_exponent - FP16_FRACTION_BITS),
    )
    hard_limit = torch.maximum(torch.full_like(abs_error, FP16_MAX_ABS_ERROR), FP16_MAX_ULP_ERROR * ulp)
    hard_limit_ok = bool(torch.all(abs_error <= hard_limit).item())

    assert matched_ratio >= REQUIRED_MATCHED_RATIO and hard_limit_ok, (
        f"shape={shape}, matched_ratio={matched_ratio}, "
        f"max_abs_error={abs_error.max().item()}, "
        f"max_hard_limit={hard_limit.max().item()}"
    )


@pytest.mark.parametrize("m,j,n,k", TEST_CASES)
def test_ascend950_matmul_gather_scatter(m: int, j: int, n: int, k: int) -> None:
    shape_seed = m * SHAPE_SEED_M_FACTOR + j * SHAPE_SEED_J_FACTOR + n * SHAPE_SEED_N_FACTOR + k
    generator = torch.Generator().manual_seed(shape_seed)
    indices_cpu = torch.randperm(m, generator=generator)[:j]
    indices = indices_cpu.to(device="npu", dtype=torch.int32)
    indices_long = indices_cpu.to(device="npu", dtype=torch.int64)
    input_range = LARGE_K_INPUT_RANGE if k > FULL_LOAD_A_MAX_K else DEFAULT_INPUT_RANGE
    a = torch.empty((m, k), dtype=torch.float16, device="npu").uniform_(-input_range, input_range)
    b = torch.empty((k, n), dtype=torch.float16, device="npu").uniform_(-input_range, input_range)

    actual = torch_catlass.ascend950_matmul_gather_scatter(a, b, indices)
    gathered = torch.index_select(a, 0, indices_long)
    expected = torch.zeros((m, n), dtype=torch.float16, device="npu")
    expected.index_copy_(0, indices_long, torch.matmul(gathered, b))

    assert actual.shape == (m, n)
    assert actual.dtype == torch.float16
    assert actual.device.type == "npu"
    _assert_fp16_precision(actual, expected, (m, j, n, k))

    selected = torch.zeros(m, dtype=torch.bool)
    selected[indices_cpu.to(torch.int64)] = True
    unselected = torch.nonzero(~selected, as_tuple=False).flatten().to(device="npu", dtype=torch.int64)
    assert torch.count_nonzero(torch.index_select(actual, 0, unselected)).item() == 0
