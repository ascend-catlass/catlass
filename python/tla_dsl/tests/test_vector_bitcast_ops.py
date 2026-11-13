from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass._mlir import ir as mlir_ir
from catlass.base_dsl import BaseDSL
from catlass.core_api import TlaCoreAPIError, VectorSSA
from catlass.tla.runtime import make_fake_tensor


def _lower(kernel, type_args):
    return BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=type_args,
        location=kernel.decorator_location,
    )


def _mlir_token(dtype) -> str:
    """The element-type spelling MLIR actually prints for `dtype`.

    Derived from the type system rather than written out by hand: MLIR's
    spelling differs from the DSL dtype token in ways that are easy to get
    wrong (`u32` prints as `ui32`, `f8e4m3fn` as `f8E4M3FN`), and a hardcoded
    guess turns a passing op into a failing test.
    """
    with mlir_ir.Context():
        return str(dtype.mlir_type())


def _ub(dtype, lanes):
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def _to_bits(
    src: tla.Tensor,
    dst: tla.Tensor,
    bits_type: tla.Constexpr[type],
    lanes: tla.Constexpr[int],
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            bits = src_tile.load().bitcast(bits_type)
            assert isinstance(bits, VectorSSA)
            dst_tile.store(bits)


@pytest.mark.parametrize(
    ("src_dtype", "dst_dtype", "lanes"),
    [
        (tla.Float32, tla.Int32, 64),
        (tla.Int32, tla.Float32, 64),
        (tla.Float32, tla.UInt32, 64),
        (tla.Float16, tla.Int16, 128),
        (tla.BFloat16, tla.Int16, 128),
        (tla.Float16, tla.BFloat16, 128),
        (tla.Int8, tla.UInt8, 256),
        (tla.Int64, tla.UInt64, 32),
    ],
)
def test_same_width_bitcast_emits_op_and_keeps_lanes(
    src_dtype, dst_dtype, lanes
) -> None:
    lowered = _lower(
        _to_bits, (_ub(src_dtype, lanes), _ub(dst_dtype, lanes), dst_dtype, lanes)
    )
    assert lowered.module.operation.verify()
    (line,) = [line for line in lowered.asm().splitlines() if "tla.bitcast " in line]
    assert f"!tla.vector<{lanes}x{_mlir_token(src_dtype)}>" in line
    assert f"!tla.vector<{lanes}x{_mlir_token(dst_dtype)}>" in line


@tla.kernel
def _exponent(src: tla.Tensor, dst: tla.Tensor, lanes: tla.Constexpr[int]) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
            bits = src_tile.load().bitcast(tla.Int32)
            shifted = tla.shift_right(bits, 23, mask=mask)
            dst_tile.store(tla.bitwise_and(shifted, tla.full(0xFF, tla.Int32)))


def test_exponent_extraction_reaches_the_bit_pattern() -> None:
    lowered = _lower(_exponent, (_ub(tla.Float32, 64), _ub(tla.Int32, 64), 64))
    assert lowered.module.operation.verify()
    asm = lowered.asm()
    assert "tla.bitcast" in asm
    assert "tla.shift_rights" in asm
    assert "tla.bitwise_and" in asm


@tla.kernel
def _same_type(src: tla.Tensor, dst: tla.Tensor, lanes: tla.Constexpr[int]) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            same = value.bitcast(tla.Float32)
            assert same is value
            dst_tile.store(same)


def test_bitcast_to_the_source_type_is_a_no_op() -> None:
    lowered = _lower(_same_type, (_ub(tla.Float32, 64), _ub(tla.Float32, 64), 64))
    assert lowered.module.operation.verify()
    assert "tla.bitcast" not in lowered.asm()


@tla.kernel
def _bad(
    src: tla.Tensor,
    dst: tla.Tensor,
    dst_type: tla.Constexpr[type],
    lanes: tla.Constexpr[int],
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load().bitcast(dst_type))


@pytest.mark.parametrize(
    ("dst_type", "message"),
    [
        (tla.Bool, "unsupported destination element type"),
        # f8e8m0 is a TLA dialect type, not a builtin float, so
        # !tla.vector<Nxf8e8m0> cannot be constructed. It must be refused here
        # rather than failing deep in the vector-region lowering with an
        # "Execution-mode lowering failed in tla.vec.func region-body" that
        # says nothing about the element type the caller actually named.
        (tla.Float8E8M0, "unsupported destination element type"),
    ],
    ids=["bool", "f8e8m0"],
)
def test_non_register_types_are_rejected(dst_type, message) -> None:
    with pytest.raises(TlaCoreAPIError, match=message):
        _ = _lower(_bad, (_ub(tla.Float32, 64), _ub(tla.Float32, 64), dst_type, 64))


def test_bitcast_requires_a_concrete_numeric_target() -> None:
    with pytest.raises(
        TlaCoreAPIError, match="expected a concrete Numeric element type"
    ):
        _ = _lower(_bad, (_ub(tla.Float32, 64), _ub(tla.Float32, 64), int, 64))


@tla.kernel
def _partial_lanes(
    src: tla.Tensor,
    dst: tla.Tensor,
    lanes: tla.Constexpr[int],
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load().bitcast(tla.Int32))


def test_bitcast_preserves_a_partial_lane_count() -> None:
    """A tile narrower than the register keeps its valid_lanes on both sides."""
    lowered = _lower(_partial_lanes, (_ub(tla.Float32, 64), _ub(tla.Int32, 64), 7))
    assert lowered.module.operation.verify()
    (line,) = [line for line in lowered.asm().splitlines() if "tla.bitcast " in line]
    assert "!tla.vector<7xf32>" in line
    assert "!tla.vector<7xi32>" in line


@pytest.mark.parametrize(
    "source, dest, src_lanes, dst_lanes",
    [
        (tla.Float32, tla.BFloat16, 64, 128),
        (tla.BFloat16, tla.Int32, 128, 64),
        (tla.Int8, tla.UInt32, 256, 64),
        (tla.Float32, tla.UInt8, 7, 28),
        (tla.Float32, tla.Int64, 64, 32),
    ],
)
def test_register_view_rescales_valid_lanes(source, dest, src_lanes, dst_lanes):
    lowered = _lower(
        _to_bits, (_ub(source, src_lanes), _ub(dest, dst_lanes), dest, src_lanes)
    )
    assert lowered.module.operation.verify()
    (line,) = [line for line in lowered.asm().splitlines() if "tla.bitcast " in line]
    assert f"!tla.vector<{src_lanes}x{_mlir_token(source)}>" in line
    assert f"!tla.vector<{dst_lanes}x{_mlir_token(dest)}>" in line


def test_register_view_rejects_incomplete_destination_element():
    with pytest.raises(TlaCoreAPIError, match="not divisible"):
        _ = _lower(_bad, (_ub(tla.Int8, 3), _ub(tla.Int32, 1), tla.Int32, 3))


@pytest.mark.parametrize(
    ("src_dtype", "dst_dtype"),
    [
        (tla.Int8, tla.Float8E4M3FN),
        (tla.Float8E4M3FN, tla.Int8),
        (tla.Float8E4M3FN, tla.Float8E5M2),
    ],
)
def test_fp8_element_types_are_reinterpretable(src_dtype, dst_dtype) -> None:
    lowered = _lower(
        _to_bits, (_ub(src_dtype, 256), _ub(dst_dtype, 256), dst_dtype, 256)
    )
    assert lowered.module.operation.verify()
    (line,) = [line for line in lowered.asm().splitlines() if "tla.bitcast " in line]
    assert f"!tla.vector<256x{_mlir_token(src_dtype)}>" in line
    assert f"!tla.vector<256x{_mlir_token(dst_dtype)}>" in line
