from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.execution_lowering import UnsupportedExecutionLowering
from catlass.tla.runtime import make_fake_tensor


def _tensor(dtype: type[tla.Numeric], lanes: int) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def _mixed_exp_sub_kernel(
    lhs: tla.Tensor, rhs: tla.Tensor, even_dst: tla.Tensor, odd_dst: tla.Tensor
) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(128), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(128), tla.make_coord(0))
    even_tile = tla.tile_view(even_dst, tla.make_shape(64), tla.make_coord(0))
    odd_tile = tla.tile_view(odd_dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            lhs_reg = lhs_tile.load()
            rhs_reg = rhs_tile.load()
            half_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            even = tla.experimental.exp_sub(
                lhs_reg, rhs_reg, reg_slot=tla.params.RegSlot.ZERO, mask=half_mask
            )
            odd = tla.experimental.exp_sub(
                lhs_reg, rhs_reg, reg_slot=tla.params.RegSlot.ONE, mask=half_mask
            )
            even_tile.store(even)
            odd_tile.store(odd)


def test_mixed_exp_sub_emits_both_parts_and_half_mask() -> None:
    mlir = _mixed_exp_sub_kernel.dump_mlir(
        type_args=(
            _tensor(tla.Float16, 128),
            _tensor(tla.Float16, 128),
            _tensor(tla.Float32, 64),
            _tensor(tla.Float32, 64),
        )
    )
    lines = [line for line in mlir.splitlines() if "tla.exp_sub" in line]
    assert len(lines) == 2
    assert "part 0" in lines[0]
    assert "part 1" in lines[1]
    assert all("!tla.mask<128>" in line for line in lines)
    assert all("!tla.vector<64xf32>" in line for line in lines)


@tla.kernel
def _mixed_exp_sub_from_narrow_cast() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            full_float = tla.full(1.0, dtype=tla.Float32)
            float_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            half_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            narrowed = full_float.to(
                tla.Float16,
                tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.NOSAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                ),
                mask=float_mask,
            )
            tla.experimental.exp_sub(
                narrowed, narrowed, reg_slot=tla.params.RegSlot.ZERO, mask=half_mask
            )
            tla.experimental.exp_sub(narrowed, narrowed)
            tla.experimental.exp_sub(narrowed, narrowed, mask=None)


def test_mixed_exp_sub_accepts_unknown_valid_lanes_from_half_cast() -> None:
    mlir = _mixed_exp_sub_from_narrow_cast.dump_mlir(type_args=())
    _assert_optional_masks_match(mlir, 128)
    line = next(line for line in mlir.splitlines() if "tla.exp_sub" in line)
    assert "!tla.vector<?xf16>" in line
    assert "!tla.vector<64xf32>" in line


@tla.kernel
def _invalid_mixed_exp_sub(
    dtype: tla.Constexpr[type],
    mask_dtype: tla.Constexpr[type],
    slot: tla.Constexpr[object],
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = tla.full(1, dtype)
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=mask_dtype)
            tla.experimental.exp_sub(value, value, reg_slot=slot, mask=mask)


@pytest.mark.parametrize(
    ("dtype", "mask_dtype", "slot", "message"),
    [
        (tla.BFloat16, tla.BFloat16, tla.params.RegSlot.ZERO, "matching f16 or f32"),
        (tla.Float32, tla.Float16, tla.params.RegSlot.ZERO, "64-lane input-domain"),
        (tla.Float16, tla.Float32, tla.params.RegSlot.ZERO, "128-lane input-domain"),
        (
            tla.Float16,
            tla.Float16,
            tla.params.RegSlot.TWO,
            "RegSlot.ZERO or RegSlot.ONE",
        ),
    ],
)
def test_mixed_exp_sub_rejects_invalid_types_and_slot(
    dtype: type[tla.Numeric], mask_dtype: type[tla.Numeric], slot: object, message: str
) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=message):
        _invalid_mixed_exp_sub.dump_mlir(type_args=(dtype, mask_dtype, slot))


@tla.kernel
def _float_exp_sub_kernel(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    dst: tla.Tensor,
    slot: tla.Constexpr[object],
) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            result = tla.experimental.exp_sub(
                lhs_tile.load(), rhs_tile.load(), reg_slot=slot, mask=mask
            )
            dst_tile.store(result)


@pytest.mark.parametrize("slot", [tla.params.RegSlot.ZERO, tla.params.RegSlot.ONE])
def test_float_exp_sub_emits_full_width_float_result(slot) -> None:
    mlir = _float_exp_sub_kernel.dump_mlir(
        type_args=(
            _tensor(tla.Float32, 64),
            _tensor(tla.Float32, 64),
            _tensor(tla.Float32, 64),
            slot,
        )
    )
    line = next(line for line in mlir.splitlines() if "tla.exp_sub" in line)
    assert "!tla.mask<64>" in line
    assert line.count("!tla.vector<64xf32>") == 3


@tla.kernel
def _mismatched_exp_sub_kernel() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            half = tla.full(1.0, dtype=tla.Float16)
            float_value = tla.full(1.0, dtype=tla.Float32)
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            tla.experimental.exp_sub(
                half, float_value, reg_slot=tla.params.RegSlot.ZERO, mask=mask
            )


def test_exp_sub_rejects_mixed_input_types() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="matching f16 or f32"):
        _mismatched_exp_sub_kernel.dump_mlir(type_args=())


@tla.kernel
def _default_exp_sub_kernel(dtype: tla.Constexpr[type]) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = tla.full(1.0, dtype=dtype)
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype)
            tla.experimental.exp_sub(value, value, mask=mask)
            tla.experimental.exp_sub(
                value, value, reg_slot=tla.params.RegSlot.ZERO, mask=mask
            )


@pytest.mark.parametrize("dtype", [tla.Float16, tla.Float32])
def test_exp_sub_default_slot_matches_explicit_zero(dtype) -> None:
    mlir = _default_exp_sub_kernel.dump_mlir(type_args=(dtype,))
    lines = [line for line in mlir.splitlines() if "tla.exp_sub" in line]
    assert len(lines) == 2
    assert "part 0" in lines[0]
    assert lines[0].partition("=")[2] == lines[1].partition("=")[2]


def _assert_optional_masks_match(mlir: str, lanes: int) -> None:
    operations = [line for line in mlir.splitlines() if "tla.exp_sub" in line]
    assert len(operations) == 3
    producers = {
        line.partition("=")[0].strip(): line.partition("=")[2]
        for line in mlir.splitlines()
        if "tla.create_mask" in line
    }
    masks = []
    bodies = []
    for operation in operations:
        mask = operation.split(" mask ", 1)[1].split()[0]
        assert f"!tla.mask<{lanes}>" in producers[mask]
        masks.append(producers[mask])
        bodies.append(
            operation.partition("=")[2].replace(f" mask {mask} ", " mask %full_mask ")
        )
    assert masks[0] == masks[1] == masks[2]
    assert bodies[0] == bodies[1] == bodies[2]


@tla.kernel
def _optional_mask_exp_sub_kernel(
    dtype: tla.Constexpr[type], slot: tla.Constexpr[object]
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = tla.full(1.0, dtype=dtype)
            full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype)
            tla.experimental.exp_sub(value, value, reg_slot=slot, mask=full_mask)
            tla.experimental.exp_sub(value, value, reg_slot=slot)
            tla.experimental.exp_sub(value, value, reg_slot=slot, mask=None)


@pytest.mark.parametrize("dtype", [tla.Float16, tla.Float32])
@pytest.mark.parametrize("slot", [tla.params.RegSlot.ZERO, tla.params.RegSlot.ONE])
def test_exp_sub_optional_mask_matches_explicit_full_mask(dtype, slot) -> None:
    mlir = _optional_mask_exp_sub_kernel.dump_mlir(type_args=(dtype, slot))
    _assert_optional_masks_match(mlir, 128 if dtype is tla.Float16 else 64)


@pytest.mark.parametrize("bad_mask", [0, False, "ALL"])
def test_exp_sub_rejects_invalid_explicit_masks(bad_mask) -> None:
    @tla.kernel
    def kernel():
        with tla.vector():
            with tla.vec.func(mode="simd"):
                value = tla.full(1.0, dtype=tla.Float16)
                tla.experimental.exp_sub(value, value, mask=bad_mask)

    with pytest.raises(tla.TlaCoreAPIError, match="mask"):
        kernel.dump_mlir()


@tla.kernel
def _static_exp_sub_kernel(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    mask_mode: tla.Constexpr[str],
    slot: tla.Constexpr[object],
    dtype: tla.Constexpr[type],
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            a = lhs.load()
            b = rhs.load()
            if tla.const_expr(mask_mode == "omitted"):
                tla.experimental.exp_sub(a, b, reg_slot=slot)
            elif tla.const_expr(mask_mode == "none"):
                tla.experimental.exp_sub(a, b, reg_slot=slot, mask=None)
            else:
                full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype)
                tla.experimental.exp_sub(a, b, reg_slot=slot, mask=full_mask)


@pytest.mark.parametrize(
    "dtype,lhs_lanes,rhs_lanes",
    [
        (tla.Float16, 1, 1),
        (tla.Float16, 3, 3),
        (tla.Float16, 64, 64),
        (tla.Float16, 65, 128),
        (tla.Float16, 127, 65),
        (tla.Float32, 1, 1),
        (tla.Float32, 3, 3),
        (tla.Float32, 32, 32),
        (tla.Float32, 33, 64),
        (tla.Float32, 63, 33),
    ],
)
@pytest.mark.parametrize("slot", [tla.params.RegSlot.ZERO, tla.params.RegSlot.ONE])
@pytest.mark.parametrize("mask_mode", ["omitted", "none", "explicit"])
def test_exp_sub_accepts_static_input_prefixes(
    dtype, lhs_lanes, rhs_lanes, slot, mask_mode
) -> None:
    mlir = _static_exp_sub_kernel.dump_mlir(
        type_args=(
            _tensor(dtype, lhs_lanes),
            _tensor(dtype, rhs_lanes),
            mask_mode,
            slot,
            dtype,
        )
    )
    element = "f16" if dtype is tla.Float16 else "f32"
    capacity = 128 if dtype is tla.Float16 else 64
    line = next(line for line in mlir.splitlines() if "tla.exp_sub" in line)
    assert f"!tla.vector<{lhs_lanes}x{element}>" in line
    assert f"!tla.vector<{rhs_lanes}x{element}>" in line
    assert f"!tla.mask<{capacity}>" in line
    assert line.endswith("-> !tla.vector<64xf32>")


@tla.kernel
def _static_exp_sub_with_unknown_rhs(lhs: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            full_float = tla.full(1.0, dtype=tla.Float32)
            unknown_half = full_float.to(
                tla.Float16,
                tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.NOSAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                ),
            )
            tla.experimental.exp_sub(lhs.load(), unknown_half)


def test_exp_sub_accepts_static_prefix_with_unknown_rhs() -> None:
    mlir = _static_exp_sub_with_unknown_rhs.dump_mlir(
        type_args=(_tensor(tla.Float16, 3),)
    )
    line = next(line for line in mlir.splitlines() if "tla.exp_sub" in line)
    assert "!tla.vector<3xf16>, !tla.vector<?xf16>" in line
    assert line.endswith("-> !tla.vector<64xf32>")


@pytest.mark.parametrize(
    "dtype,lanes,message",
    [
        (tla.Float16, 0, "valid_lanes must be positive"),
        (tla.Float16, 129, "valid_lanes must be <= 128"),
        (tla.Float32, 65, "valid_lanes must be <= 64"),
    ],
)
def test_exp_sub_rejects_invalid_input_shapes(dtype, lanes, message) -> None:
    with pytest.raises(UnsupportedExecutionLowering, match=message):
        _static_exp_sub_kernel.dump_mlir(
            type_args=(
                _tensor(dtype, lanes),
                _tensor(dtype, lanes),
                "none",
                tla.params.RegSlot.ZERO,
                dtype,
            )
        )
