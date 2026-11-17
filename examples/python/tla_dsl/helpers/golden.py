# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import overload

import torch

# HF32 mixed tolerance thresholds.
_HF32_RTOL = 2.0**-9
_HF32_ATOL = 2.0**-10
_HF32_REQUIRED_MATCHED_RATIO = 0.99
_HF32_MAX_ABS_ERROR_LIMIT = 1e-1

# dtype -> (rtol, atol)
_ERROR_THRESHOLD_MAP = {
    torch.float32: (1.2e-4, 1.2e-4),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (7.8e-3, 7.8e-3),
}


def tolerance(
    expected: torch.Tensor,
    k: int,
    *,
    bf16: bool,
    rtol: float | None = None,
    floor: float | None = None,
) -> torch.Tensor:
    """Return the element-wise pass threshold for the matmul precision standard."""
    if rtol is None:
        if bf16:
            rtol = (1.0 / 128.0) if k < 2048 else (1.0 / 64.0)
        else:
            rtol = (1.0 / 256.0) if k < 2048 else (1.0 / 128.0)
    floor = 1.0 if floor is None else floor
    return rtol * torch.maximum(torch.full_like(expected, floor), expected.abs())


def _compare_hf32(
    result: torch.Tensor,
    expected: torch.Tensor,
) -> bool:
    """Compare result against an HF32-semantic golden."""
    result = result.float()
    expected = expected.float()
    diff = (result - expected).abs()
    matched = diff <= _HF32_ATOL + _HF32_RTOL * expected.abs()
    matched_ratio = matched.float().mean().item()
    max_abs_error = diff.max().item()
    ulp = torch.finfo(expected.dtype).eps * (
        2.0 ** torch.floor(torch.log2(expected.abs().max()))
    )
    max_abs_error_limit = max(_HF32_MAX_ABS_ERROR_LIMIT, 32.0 * ulp.item())
    return bool(
        matched_ratio >= _HF32_REQUIRED_MATCHED_RATIO
        and max_abs_error <= max_abs_error_limit
    )


@overload
def compare(
    result: torch.Tensor,
    expected: torch.Tensor,
    k: int,
    *,
    rtol: float | None = None,
    floor: float | None = None,
) -> bool: ...


@overload
def compare(
    result: torch.Tensor,
    expected: torch.Tensor,
    *,
    enable_hf32: bool = False,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> bool: ...


def compare(
    result: torch.Tensor,
    expected: torch.Tensor,
    k: int | None = None,
    *,
    rtol: float | None = None,
    floor: float | None = None,
    atol: float | None = None,
    enable_hf32: bool = False,
) -> bool:
    """Compare ``result`` against ``expected`` with given threshold.

    Three call forms are supported:

    1. Accumulative precision standard (``k`` given), for matmul-like operators.
    2. Generic element-wise check.
    3. HF32 mixed tolerance (``enable_hf32=True``).
    """
    if enable_hf32:
        # Use mixed tolerance for HF32.
        return _compare_hf32(result, expected)
    if k is not None and isinstance(k, int):
        # Single precision standard.
        is_bf16 = result.dtype == torch.bfloat16
        result, expected = result.float(), expected.float()
        return bool(
            (
                (result - expected).abs()
                <= tolerance(expected, k, bf16=is_bf16, rtol=rtol, floor=floor)
            ).all()
        )
    if result.dtype != expected.dtype:
        raise TypeError("the data type between the golden and the result do not match")
    if result.dtype in (torch.float32, torch.float16, torch.bfloat16):
        # Generic element-wise precision check.
        _rtol, _atol = _ERROR_THRESHOLD_MAP[result.dtype]
        return bool(
            torch.isclose(
                result,
                expected,
                rtol=_rtol if rtol is None else rtol,
                atol=_atol if atol is None else atol,
            ).all()
        )
    return bool(result.eq(expected).all())


_SMALL_VALUE_STANDARDS: dict[torch.dtype, tuple[float, float]] = {
    torch.float16: (2.0**-11, 2.0**-16),
    torch.bfloat16: (2.0**-8, 2.0**-16),
    torch.float32: (2.0**-14, 2.0**-30),
}


@dataclass(frozen=True)
class ErrorMetrics:
    """Error-ratio metrics of an NPU result against a high-precision golden.

    Torch port of ``Catlass::golden::ErrorMetrics`` /
    ``ComputeErrorMetrics`` in ``examples/common/golden/compare_data.hpp``.

    - ``mare_ratio``: max absolute relative error, ``npu / cpu``
    - ``mere_ratio``: mean absolute relative error, ``npu / cpu``
    - ``rmse_ratio``: root mean squared error, ``npu / cpu``
    - ``small_cnt_ratio``: small-value error count, ``npu / cpu``

    The three relative-error ratios always cover every sample: they are the
    primary judgement, so a value above its threshold means the small-value
    review was triggered, never that it passed. ``small_cnt_ratio`` covers the
    samples below the small-value threshold and stays 0.0 for result dtypes
    without a standard.

    A ``passed`` result with ``mare_ratio > mare_threshold`` therefore means the
    mare excess came from the small-value range and that range stayed within
    its own error budget.
    """

    passed: bool
    mare_ratio: float
    mere_ratio: float
    rmse_ratio: float
    small_cnt_ratio: float


def _relative_error_stats(
    value: torch.Tensor, golden: torch.Tensor, eps: float
) -> tuple[float, float, float]:
    """Return (mare, mere, rmse) of ``value`` against ``golden`` (f64)."""
    if value.numel() == 0:
        # Every sample landed in the other value range. Zeros keep a subset
        # that is entirely small-value from failing relative-error criteria it
        # is no longer being judged by.
        return 0.0, 0.0, 0.0
    diff = (value - golden).abs()
    relative = diff / (golden.abs() + eps)
    return (
        relative.max().item(),
        relative.mean().item(),
        torch.sqrt((diff * diff).mean()).item(),
    )


def _metric_ratios(
    npu: torch.Tensor, cpu: torch.Tensor, ref: torch.Tensor, eps: float
) -> tuple[float, float, float]:
    """Return the (mare, mere, rmse) ``npu / cpu`` ratios over one subset."""
    mare_c, mere_c, rmse_c = _relative_error_stats(npu, ref, eps)
    mare_cpu, mere_cpu, rmse_cpu = _relative_error_stats(cpu, ref, eps)
    return (
        _error_ratio(mare_c, mare_cpu),
        _error_ratio(mere_c, mere_cpu),
        _error_ratio(rmse_c, rmse_cpu),
    )


def _error_ratio(device_error: float, cpu_error: float) -> float:
    """Ratio of a device error to the CPU baseline error.

    Mirrors ``errorRatio`` in ``compare_data.hpp``: an exactly-right CPU
    baseline only passes a device that is also exact and reports ``inf``
    otherwise. Clamping the denominator to ``eps`` instead turns zero-baseline
    rounding noise into large but finite ratios, which read like real
    measurements and hide the fact that the denominator was empty.
    """
    if cpu_error > 0:
        return device_error / cpu_error
    return 0.0 if device_error == 0 else math.inf


def _small_value_standard(
    dtype: torch.dtype,
    threshold: float | None,
    error: float | None,
) -> tuple[float, float] | None:
    """Resolve the small-value ``(threshold, error)`` pair for ``dtype``."""
    standard = _SMALL_VALUE_STANDARDS.get(dtype)
    if standard is None:
        # Dtypes outside the table have no small-value criterion; opting in
        # needs both an explicit threshold and an explicit error.
        if threshold is None or error is None:
            return None
        return threshold, error
    table_threshold, table_error = standard
    return (
        table_threshold if threshold is None else threshold,
        table_error if error is None else error,
    )


def _small_value_error_count(
    actual: torch.Tensor,
    golden: torch.Tensor,
    threshold: float,
    error: float,
) -> int:
    """Count ``|golden| < threshold`` samples failing the absolute-error bound."""
    small = golden.abs() < threshold
    return int((small & ((actual - golden).abs() > error)).sum().item())


def compute_error_metrics(
    npu_result: torch.Tensor,
    cpu_result: torch.Tensor,
    golden: torch.Tensor,
    mare_threshold: float = 5.0,
    mere_threshold: float = 1.5,
    rmse_threshold: float = 1.5,
    small_ratio_threshold: float = 2.0,
    small_val_threshold: float | None = None,
    small_val_error: float | None = None,
    eps: float = 1e-7,
) -> ErrorMetrics:
    """Compare an NPU result and a same-precision CPU result against a
    high-precision golden reference, mirroring ``ComputeErrorMetrics`` in
    ``examples/common/golden/compare_data.hpp`` and adding the small-value
    pass standard.

    ``npu_result`` is the device output, ``cpu_result`` the same-precision CPU
    result, and ``golden`` the high-precision reference.

    Every sample is judged by the mare/mere/rmse ratios (``npu`` against
    ``cpu``); an exact CPU baseline (ratio denominator 0) only passes an exact
    device. A relative error is meaningless for a sample whose golden is near
    zero, though: the denominator collapses and a few ULP of rounding read as a
    large relative deviation.

    mare is a *max*, so a handful of such samples can fail the whole tensor on
    their own while mere and rmse stay flat. The small-value standard is
    therefore a review of the mare failure, not a culling step applied up
    front:

    1. All three ratios over every sample pass -> passed, and the small-value
       range is never consulted. ``small_cnt_ratio`` is still reported.
    2. mare exceeds its threshold -> re-examine ``|golden| < threshold`` with an
       absolute error count instead of a ratio::

         ErrorCount(actual) = count(|golden| < threshold
                                    and |actual - golden| > error)
         small_cnt_ratio = ErrorCount(npu) / max(ErrorCount(cpu), 1)

       The failure is excused only if the large-value subset passes all three
       ratios *and* ``small_cnt_ratio <= small_ratio_threshold``. The returned
       ratios still describe every sample, so a passing result with
       ``mare_ratio > mare_threshold`` says the excess was confined to the
       small-value range.
    3. mere or rmse alone exceed their thresholds -> failed. The small-value
       range cannot excuse them: an error count is not comparable to a mean.

    ``small_val_threshold`` / ``small_val_error`` default to the result-dtype
    table (f16: 2^-11 / 2^-16, bf16: 2^-8 / 2^-16, f32: 2^-14 / 2^-30); dtypes
    outside the table skip the whole small-value standard.
    """
    npu_tensor = torch.as_tensor(npu_result)
    cpu_tensor = torch.as_tensor(cpu_result)
    npu = npu_tensor.to(torch.float64).reshape(-1)
    cpu = cpu_tensor.to(torch.float64).reshape(-1)
    ref = torch.as_tensor(golden).to(torch.float64).reshape(-1)

    n = ref.numel()
    if n == 0 or npu.numel() != n or cpu.numel() != n:
        return ErrorMetrics(False, 0.0, 0.0, 0.0, 0.0)

    # Primary judgement: every sample, all three ratios.
    mare_ratio, mere_ratio, rmse_ratio = _metric_ratios(npu, cpu, ref, eps)
    ratios_pass = bool(
        mare_ratio <= mare_threshold
        and mere_ratio <= mere_threshold
        and rmse_ratio <= rmse_threshold
    )

    standard = _small_value_standard(
        npu_tensor.dtype, small_val_threshold, small_val_error
    )
    if standard is None:
        # No small-value criterion for this dtype: the ratios decide alone.
        small_cnt_ratio = 0.0
        passed = ratios_pass
    else:
        threshold, error = standard
        small_cnt_ratio = _small_value_error_count(npu, ref, threshold, error) / max(
            _small_value_error_count(cpu, ref, threshold, error), 1
        )
        if ratios_pass or mare_ratio <= mare_threshold:
            # Either everything passed, or mare passed and mere/rmse did not.
            # The latter is a mean-level failure an error count cannot excuse.
            passed = ratios_pass
        else:
            # mare is the only metric that failed. Re-examine it on the
            # large-value subset, where a relative error is well defined, and
            # hold the small-value range to its own absolute criterion. The
            # returned ratios stay full-sample: overwriting them with the
            # subset's would make a triggered review read as a clean pass.
            large = ref.abs() >= threshold
            large_mare, large_mere, large_rmse = _metric_ratios(
                npu[large], cpu[large], ref[large], eps
            )
            passed = bool(
                large_mare <= mare_threshold
                and large_mere <= mere_threshold
                and large_rmse <= rmse_threshold
                and small_cnt_ratio <= small_ratio_threshold
            )
    return ErrorMetrics(
        passed=passed,
        mare_ratio=mare_ratio,
        mere_ratio=mere_ratio,
        rmse_ratio=rmse_ratio,
        small_cnt_ratio=float(small_cnt_ratio),
    )
