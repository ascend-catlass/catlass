# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

import catlass.tla as tla


@tla.kernel
def _relu_ops(dtype: tla.Constexpr[type]) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src = tla.full(-4.0, dtype)
            slopes = tla.full(0.25, dtype)
            mask = tla.create_mask(pattern=tla.mask.VL16, dtype=dtype)
            tla.prelu(src, slopes, mask=mask)
            tla.leaky_relu(src, dtype(0.1))
            tla.leaky_relu(src, 0.1)


@pytest.mark.parametrize(
    ("dtype", "lanes", "element"),
    [(tla.Float16, 128, "f16"), (tla.Float32, 64, "f32")],
)
def test_relu_ops_emit_matching_vector_and_scalar_types(dtype, lanes, element):
    mlir = _relu_ops.dump_mlir(type_args=(dtype,))
    vector = f"!tla.vector<{lanes}x{element}>"
    prelu = next(line for line in mlir.splitlines() if "tla.prelu " in line)
    leaky_ops = [line for line in mlir.splitlines() if "tla.leaky_relu " in line]
    assert f": {vector}, {vector} mask !tla.mask<{lanes}> -> {vector}" in prelu
    assert len(leaky_ops) == 2
    assert all(f": {vector}, {element} -> {vector}" in line for line in leaky_ops)


@tla.kernel
def _invalid_relu(
    op: tla.Constexpr[str],
    dtype: tla.Constexpr[type],
    slope_dtype: tla.Constexpr[type],
    mask_dtype: tla.Constexpr[type],
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src = tla.full(-4.0, dtype)
            mask = tla.create_mask(pattern=tla.mask.VL16, dtype=mask_dtype)
            if tla.const_expr(op == "prelu"):
                tla.prelu(src, tla.full(0.25, slope_dtype), mask=mask)
            else:
                tla.leaky_relu(src, slope_dtype(0.1), mask=mask)


@pytest.mark.parametrize(
    ("op", "dtype", "slope_dtype", "mask_dtype", "error"),
    [
        (
            "prelu",
            tla.BFloat16,
            tla.BFloat16,
            tla.BFloat16,
            "src must have Float16 or Float32",
        ),
        ("prelu", tla.Float16, tla.Float32, tla.Float16, "rhs has element type f32"),
        (
            "leaky_relu",
            tla.Int32,
            tla.Float32,
            tla.Int32,
            "src must have Float16 or Float32",
        ),
        (
            "leaky_relu",
            tla.Float16,
            tla.Int32,
            tla.Float16,
            "slope must be a Float16 or Float32 scalar",
        ),
        (
            "leaky_relu",
            tla.Float32,
            tla.Float16,
            tla.Float32,
            "slope must match the src element type f32, got f16",
        ),
        (
            "leaky_relu",
            tla.Float16,
            tla.Float32,
            tla.Float16,
            "slope must match the src element type f16, got f32",
        ),
        ("leaky_relu", tla.Float32, tla.Float32, tla.Float16, "mask.*lanes"),
    ],
)
def test_relu_ops_reject_invalid_types(op, dtype, slope_dtype, mask_dtype, error):
    with pytest.raises(tla.TlaCoreAPIError, match=error):
        _invalid_relu.dump_mlir(type_args=(op, dtype, slope_dtype, mask_dtype))


@tla.kernel
def _leaky_relu_loaded_slope(
    dtype: tla.Constexpr[type], slope_dtype: tla.Constexpr[type]
) -> None:
    with tla.vector():
        ptr = tla.allocate(16, slope_dtype, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(16), tla.make_stride(1))
        slopes = tla.make_tensor(ptr, layout)
        with tla.vec.func(mode="simd"):
            tla.leaky_relu(tla.full(-1.0, dtype), slopes[0])


def test_leaky_relu_rejects_mismatched_loaded_slope():
    with pytest.raises(
        tla.TlaCoreAPIError, match="slope must match the src element type f32, got f16"
    ):
        _leaky_relu_loaded_slope.dump_mlir(type_args=(tla.Float32, tla.Float16))


@tla.kernel
def _plain_relu(
    dtype: tla.Constexpr[type],
    mask_dtype: tla.Constexpr[type],
    lanes: tla.Constexpr[int],
) -> None:
    with tla.vector():
        ptr = tla.allocate(lanes, dtype, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(lanes), tla.make_stride(1))
        tile = tla.make_tensor(ptr, layout)
        with tla.vec.func(mode="simd"):
            src = tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=mask_dtype)
            tla.relu(src, mask=mask)


@pytest.mark.parametrize(
    ("dtype", "lanes", "element"),
    [
        (tla.Float16, 128, "f16"),
        (tla.Float32, 64, "f32"),
        (tla.Int32, 64, "i32"),
        (tla.Int64, 32, "i64"),
    ],
)
def test_plain_relu_emits_matching_vector_type(dtype, lanes, element):
    ir = _plain_relu.dump_mlir(type_args=(dtype, dtype, lanes))
    vector = f"!tla.vector<{lanes}x{element}>"
    line = next(line for line in ir.splitlines() if "tla.relu " in line)
    assert f": {vector} mask !tla.mask<{lanes}> -> {vector}" in line


@pytest.mark.parametrize("dtype", [tla.BFloat16, tla.Int16, tla.UInt32, tla.UInt64])
def test_plain_relu_rejects_unsupported_types(dtype):
    with pytest.raises(tla.TlaCoreAPIError, match="requires f16, f32, i32, or i64"):
        _plain_relu.dump_mlir(type_args=(dtype, dtype, 2048 // dtype.width))
