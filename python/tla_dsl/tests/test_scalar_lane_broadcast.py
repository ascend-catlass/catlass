from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    extent: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def scalar_lane_full(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            out = reg + tla.full(0.0, tla.Float32)
            dst_tile.store(out)


@tla.kernel
def full_int_literal(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load() + tla.full(1, tla.Int32))


@tla.kernel
def full_on_lhs(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.full(0.0, tla.Float32) + src_tile.load())


@tla.kernel
def full_rejects_runtime_scalar_type_mismatch(src: tla.Tensor, dst: tla.Tensor) -> None:
    # block_idx() is an i32 scalar SSA; an f32 result requires an f32 scalar.
    idx = tla.arch.block_idx()
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load() + tla.full(idx, tla.Float32))


@tla.kernel
def full_accepts_runtime_scalar_i64(src: tla.Tensor, flag: tla.Int64) -> None:
    # An Int64 kernel scalar argument hits the runtime-scalar path; it lowers
    # through the linked bitcode helper instead of an AVE 64-bit broadcast.
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.full(flag, tla.Int64)


@tla.kernel
def full_rejects_simt_mode(src: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=1):
            tla.full(0, tla.Int32)


@tla.kernel
def full_accepts_runtime_scalar(src: tla.Tensor, dst: tla.Tensor) -> None:
    idx = tla.arch.block_idx()
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load() + tla.full(idx, tla.Int32))


@tla.kernel
def full_string_dtype_is_rejected(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load() + tla.full(0.0, "f32"))


def test_scalar_lane_full_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(scalar_lane_full, type_args=(_ub_tensor(), _ub_tensor()))

    assert "tla.full" in mlir
    assert "tla.add" in mlir


def test_full_int_literal_uses_vector_dtype(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        full_int_literal,
        type_args=(_ub_tensor(tla.Int32), _ub_tensor(tla.Int32)),
    )

    assert "value = 1 : i32" in mlir


def test_full_on_lhs_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(full_on_lhs, type_args=(_ub_tensor(), _ub_tensor()))

    assert "tla.full" in mlir
    assert "tla.add" in mlir


def test_full_rejects_runtime_scalar_type_mismatch(compiler_tlair: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="does not match result element type"):
        compiler_tlair(
            full_rejects_runtime_scalar_type_mismatch,
            type_args=(_ub_tensor(), _ub_tensor()),
        )


def test_full_accepts_runtime_scalar_i64(compiler_tlair: Any) -> None:
    # i64 broadcasts through the linked bitcode helper: the AVE 64-bit
    # broadcast ABI cannot legalize a <32xi64> result.
    mlir = compiler_tlair(
        full_accepts_runtime_scalar_i64,
        type_args=(_ub_tensor(tla.Int64), tla.Int64(7)),
    )

    assert "tla.full" in mlir
    assert "!tla.vector<32xi64>" in mlir


def test_full_rejects_simt_mode(compiler_tlair: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="only supported in tla.vec.func"):
        compiler_tlair(full_rejects_simt_mode, type_args=(_ub_tensor(tla.Int32),))


def test_full_accepts_runtime_scalar(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        full_accepts_runtime_scalar,
        type_args=(_ub_tensor(tla.Int32), _ub_tensor(tla.Int32)),
    )

    assert "tla.full" in mlir
    assert "tla.add" in mlir


def test_full_rejects_invalid_dtype(compiler_tlair: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="not a str"):
        compiler_tlair(
            full_string_dtype_is_rejected,
            type_args=(_ub_tensor(), _ub_tensor()),
        )


# --- tla.full source and mask constraints ------------------------------------
# A vector source must be a *one-lane* fragment (a VectorSSA.reduce result). A
# full-width vector lowers to the same AVE vector_broadcast, which keeps lane 0
# and drops the rest, so it is rejected rather than silently truncated. The
# optional predicate is checked against the result the op is about to build.


@tla.kernel
def full_rejects_full_width_vector(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.full(src_tile.load(), tla.Float32))


@tla.kernel
def full_accepts_one_lane_fragment(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            m = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            reduced = src_tile.load().reduce(tla.ReductionOp.ADD, mask=m)
            dst_tile.store(tla.full(reduced, tla.Float32))


@tla.kernel
def full_rejects_mismatched_mask(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            # an f32 predicate (64 lanes) against an f16 result (128 lanes)
            m = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            dst_tile.store(tla.full(1.0, tla.Float16, mask=m))


def test_full_rejects_full_width_vector_source() -> None:
    with pytest.raises(Exception, match="one-lane"):
        full_rejects_full_width_vector.dump_mlir(
            type_args=(_ub_tensor(), _ub_tensor())
        )


def test_full_accepts_one_lane_fragment() -> None:
    mlir = full_accepts_one_lane_fragment.dump_mlir(
        type_args=(_ub_tensor(), _ub_tensor())
    )
    assert "tla.full" in mlir


def test_full_rejects_mask_that_does_not_match_the_result() -> None:
    with pytest.raises(Exception, match="predicate lanes"):
        full_rejects_mismatched_mask.dump_mlir(
            type_args=(_ub_tensor(), _ub_tensor())
        )


# --- tla.full element-type matrix -------------------------------------------
# Element types mirror AscendC Duplicate. Bool/hif8/e8m0/fp4x2 broadcast an i8
# carrier byte, unsigned broadcasts the same-width signless integer, and fp8
# broadcasts natively. i64/u64 lower through linked bitcode helpers.


def _make_full_literal_kernel(dtype: type[tla.Numeric], value: Any, extent: int):
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        dst_tile = tla.tile_view(dst, tla.make_shape(extent), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst_tile.store(tla.full(value, dtype))

    return kernel


_FULL_LITERAL_CASES = (
    ("bool", tla.Bool, True, 256, "i8", "!tla.vector<256xi1>"),
    ("i8", tla.Int8, -3, 256, "i8", "!tla.vector<256xi8>"),
    ("u8", tla.UInt8, 200, 256, "i8", "!tla.vector<256xui8>"),
    ("i16", tla.Int16, -300, 128, "i16", "!tla.vector<128xi16>"),
    ("u16", tla.UInt16, 60000, 128, "i16", "!tla.vector<128xui16>"),
    ("i32", tla.Int32, -70000, 64, "i32", "!tla.vector<64xi32>"),
    ("u32", tla.UInt32, 4000000000, 64, "i32", "!tla.vector<64xui32>"),
    ("i64", tla.Int64, -(2**40), 32, "i64", "!tla.vector<32xi64>"),
    ("u64", tla.UInt64, 2**63 + 5, 32, "i64", "!tla.vector<32xui64>"),
    ("f16", tla.Float16, 1.5, 128, "f16", "!tla.vector<128xf16>"),
    ("bf16", tla.BFloat16, 1.5, 128, "bf16", "!tla.vector<128xbf16>"),
    ("f32", tla.Float32, 1.5, 64, "f32", "!tla.vector<64xf32>"),
    ("f8e4m3fn", tla.Float8E4M3FN, 1.5, 256, "f8E4M3FN", "!tla.vector<256xf8E4M3FN>"),
    ("f8e5m2", tla.Float8E5M2, 1.5, 256, "f8E5M2", "!tla.vector<256xf8E5M2>"),
    ("f8e8m0", tla.Float8E8M0, 2.0, 256, "i8", "!tla.vector<256x!tla.f8e8m0>"),
    ("hif8", tla.HiFloat8, 0x38, 256, "i8", "!tla.vector<256x!tla.hif8>"),
    (
        "f4e2m1x2",
        tla.Float4E2M1x2,
        0x21,
        256,
        "i8",
        "!tla.vector<256x!tla.f4e2m1x2>",
    ),
    (
        "f4e1m2x2",
        tla.Float4E1M2x2,
        0x21,
        256,
        "i8",
        "!tla.vector<256x!tla.f4e1m2x2>",
    ),
)


@pytest.mark.parametrize(
    "dtype, value, extent, carrier, result_type",
    [case[1:] for case in _FULL_LITERAL_CASES],
    ids=[case[0] for case in _FULL_LITERAL_CASES],
)
def test_full_literal_element_type_matrix(
    compiler_tlair: Any,
    dtype: type[tla.Numeric],
    value: Any,
    extent: int,
    carrier: str,
    result_type: str,
) -> None:
    mlir = compiler_tlair(
        _make_full_literal_kernel(dtype, value, extent),
        type_args=(_ub_tensor(dtype, extent),),
    )

    assert f"({carrier}) -> {result_type}" in mlir


def _make_full_runtime_kernel(
    dtype: type[tla.Numeric], extent: int, mask_lanes: int | None = None
):
    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        dst_tile = tla.tile_view(dst, tla.make_shape(extent), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                scale = src[0]
                if mask_lanes is None:
                    dst_tile.store(tla.full(scale, dtype))
                else:
                    mask, _ = tla.update_mask(mask_lanes, dtype)
                    dst_tile.store(tla.full(scale, dtype, mask=mask))

    return kernel


# The tensor read keeps the TLA element type through the frontend; the SIMD
# lowering maps it onto the broadcast carrier.
_FULL_RUNTIME_CASES = (
    ("bool", tla.Bool, 256, "i1", "!tla.vector<256xi1>"),
    ("u8", tla.UInt8, 256, "ui8", "!tla.vector<256xui8>"),
    ("u16", tla.UInt16, 128, "ui16", "!tla.vector<128xui16>"),
    ("u32", tla.UInt32, 64, "ui32", "!tla.vector<64xui32>"),
    ("u64", tla.UInt64, 32, "ui64", "!tla.vector<32xui64>"),
    ("f8e4m3fn", tla.Float8E4M3FN, 256, "f8E4M3FN", "!tla.vector<256xf8E4M3FN>"),
    ("f8e5m2", tla.Float8E5M2, 256, "f8E5M2", "!tla.vector<256xf8E5M2>"),
    ("f8e8m0", tla.Float8E8M0, 256, "!tla.f8e8m0", "!tla.vector<256x!tla.f8e8m0>"),
    ("hif8", tla.HiFloat8, 256, "!tla.hif8", "!tla.vector<256x!tla.hif8>"),
    (
        "f4e2m1x2",
        tla.Float4E2M1x2,
        256,
        "!tla.f4e2m1x2",
        "!tla.vector<256x!tla.f4e2m1x2>",
    ),
    (
        "f4e1m2x2",
        tla.Float4E1M2x2,
        256,
        "!tla.f4e1m2x2",
        "!tla.vector<256x!tla.f4e1m2x2>",
    ),
)


@pytest.mark.parametrize(
    "dtype, extent, operand_type, result_type",
    [case[1:] for case in _FULL_RUNTIME_CASES],
    ids=[case[0] for case in _FULL_RUNTIME_CASES],
)
def test_full_runtime_scalar_element_type_matrix(
    compiler_tlair: Any,
    dtype: type[tla.Numeric],
    extent: int,
    operand_type: str,
    result_type: str,
) -> None:
    mlir = compiler_tlair(
        _make_full_runtime_kernel(dtype, extent),
        type_args=(_ub_tensor(dtype, extent), _ub_tensor(dtype, extent)),
    )

    assert f"({operand_type}) -> {result_type}" in mlir


@tla.kernel
def full_rejects_float64(src: tla.Tensor, dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.full(0.0, tla.Float64))


@tla.kernel
def full_rejects_logical_fp4(src: tla.Tensor, dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.full(0x2, tla.Float4E2M1))


@tla.kernel
def full_rejects_fp4_float_pair(src: tla.Tensor, dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.full(0.5, tla.Float4E2M1x2))


@tla.kernel
def full_rejects_hif8_float(src: tla.Tensor, dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.full(1.0, tla.HiFloat8))


@tla.kernel
def full_rejects_e8m0_non_power_of_two(src: tla.Tensor, dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.full(3.0, tla.Float8E8M0))


@tla.kernel
def full_rejects_u8_out_of_range(src: tla.Tensor, dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.full(256, tla.UInt8))


def test_full_rejects_float64(compiler_tlair: Any) -> None:
    # f64 is storage-only and complex has no TLA vector type: both stay out of
    # the Duplicate matrix.
    with pytest.raises(tla.TlaCoreAPIError, match="unsupported vector element dtype"):
        compiler_tlair(
            full_rejects_float64, type_args=(_ub_tensor(), _ub_tensor())
        )


def test_full_rejects_logical_fp4(compiler_tlair: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="unsupported vector element dtype"):
        compiler_tlair(
            full_rejects_logical_fp4, type_args=(_ub_tensor(), _ub_tensor())
        )


def test_full_rejects_fp4_float_pair(compiler_tlair: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="packed pair byte"):
        compiler_tlair(
            full_rejects_fp4_float_pair,
            type_args=(_ub_tensor(tla.Float4E2M1x2, 256), _ub_tensor(tla.Float4E2M1x2, 256)),
        )


def test_full_rejects_hif8_float(compiler_tlair: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="explicit encoded byte"):
        compiler_tlair(
            full_rejects_hif8_float,
            type_args=(_ub_tensor(tla.HiFloat8, 256), _ub_tensor(tla.HiFloat8, 256)),
        )


def test_full_rejects_e8m0_non_power_of_two(compiler_tlair: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="E8M0"):
        compiler_tlair(
            full_rejects_e8m0_non_power_of_two,
            type_args=(_ub_tensor(tla.Float8E8M0, 256), _ub_tensor(tla.Float8E8M0, 256)),
        )


def test_full_rejects_u8_out_of_range(compiler_tlair: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="out of range for u8"):
        compiler_tlair(
            full_rejects_u8_out_of_range,
            type_args=(_ub_tensor(tla.UInt8, 256), _ub_tensor(tla.UInt8, 256)),
        )


@tla.kernel
def full_rejects_i64_fragment(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(1), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.full(src_tile.load(), tla.Int64))


def test_full_rejects_i64_fragment(compiler_tlair: Any) -> None:
    # A 64-bit fragment needs a 64-bit tla.reduce, which is not supported yet;
    # reject it here instead of reaching the device compiler.
    with pytest.raises(tla.TlaCoreAPIError, match="64-bit fragment source"):
        compiler_tlair(
            full_rejects_i64_fragment,
            type_args=(_ub_tensor(tla.Int64, 1), _ub_tensor(tla.Int64, 32)),
        )


@pytest.mark.parametrize(
    "value, encoded",
    [(0.5, 126), (1.0, 127), (2.0, -128), (4.0, -127)],
)
def test_full_e8m0_accepts_every_representable_power_of_two(
    compiler_tlair: Any, value: float, encoded: int
) -> None:
    # E8M0 codes 0..254 are 2**(code-127), so the valid literals start at
    # 2**-127; the byte carrier holds the code as a signless i8.
    mlir = compiler_tlair(
        _make_full_literal_kernel(tla.Float8E8M0, value, 256),
        type_args=(_ub_tensor(tla.Float8E8M0, 256),),
    )

    assert f"value = {encoded} : i8" in mlir


@pytest.mark.parametrize("dtype", (tla.Float8E4M3FN, tla.Float8E5M2))
def test_full_rejects_inexact_fp8(
    compiler_tlair: Any, dtype: type[tla.Numeric]
) -> None:
    # 0.1 is not representable in either 8-bit float format; the frontend
    # rejects it instead of rounding silently.
    with pytest.raises(tla.TlaCoreAPIError, match="not exactly representable"):
        compiler_tlair(
            _make_full_literal_kernel(dtype, 0.1, 256),
            type_args=(_ub_tensor(dtype, 256),),
        )


@pytest.mark.parametrize("dtype", (tla.Float8E4M3FN, tla.Float8E5M2))
def test_full_accepts_exact_fp8(
    compiler_tlair: Any, dtype: type[tla.Numeric]
) -> None:
    mlir = compiler_tlair(
        _make_full_literal_kernel(dtype, 0.125, 256),
        type_args=(_ub_tensor(dtype, 256),),
    )

    assert "tla.full" in mlir
