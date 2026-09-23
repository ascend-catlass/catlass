from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.base_dsl import BaseDSL
from catlass.core_api import MaskSSA
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def _mask_interleave(
    src: tla.Tensor,
    dst0: tla.Tensor,
    dst1: tla.Tensor,
    dtype: tla.Constexpr[type],
    lanes: tla.Constexpr[int],
    op_name: tla.Constexpr[str],
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
    dst0_tile = tla.tile_view(dst0, tla.make_shape(lanes), tla.make_coord(0))
    dst1_tile = tla.tile_view(dst1, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            first = tla.create_mask(pattern=tla.mask.H, dtype=dtype)
            second = tla.create_mask(pattern=tla.mask.Q, dtype=dtype)
            if tla.const_expr(op_name == "interleave"):
                out0, out1 = tla.interleave(first, second)
            else:
                out0, out1 = tla.deinterleave(first, second)
            assert isinstance(out0, MaskSSA)
            assert isinstance(out1, MaskSSA)
            combined = tla.bitwise_xor(out0, out1)
            value = src_tile.load()
            zero = tla.sub(value, value)
            dst0_tile.store(tla.where(out0, value, zero), mask=combined)
            dst1_tile.store(tla.where(out1, value, zero))


@pytest.mark.parametrize("op_name", ["interleave", "deinterleave"])
@pytest.mark.parametrize(
    ("dtype", "lanes"),
    [(tla.Int8, 256), (tla.Float16, 128), (tla.Float32, 64)],
    ids=["b8", "b16", "b32"],
)
def test_mask_interleave_preserves_types_and_both_results(op_name, dtype, lanes) -> None:
    tensor = make_fake_tensor(
        dtype, (lanes,), (1,), addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,), layout_tag=tla.arch.RowMajor,
    )
    lowered = BaseDSL()._lower(
        _mask_interleave.fn, kind=_mask_interleave.kind,
        options=dict(_mask_interleave.options),
        type_args=(tensor, tensor, tensor, dtype, lanes, op_name),
        location=_mask_interleave.decorator_location,
    )
    assert lowered.module.operation.verify()
    asm = lowered.asm()
    line, = [line for line in asm.splitlines() if f"tla.{op_name} " in line]
    assert line.count(f"!tla.mask<{lanes}>") == 4
    assert "tla.bitwise_xor" in asm
    assert asm.count("tla.where") == 2


@tla.kernel
def _mixed_interleave(reverse: tla.Constexpr[bool], op_name: tla.Constexpr[str]) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            vector = tla.full(1.0, tla.Float32)
            if tla.const_expr(op_name == "interleave"):
                if tla.const_expr(reverse):
                    tla.interleave(vector, mask)
                else:
                    tla.interleave(mask, vector)
            else:
                if tla.const_expr(reverse):
                    tla.deinterleave(vector, mask)
                else:
                    tla.deinterleave(mask, vector)


@pytest.mark.parametrize("op_name", ["interleave", "deinterleave"])
@pytest.mark.parametrize("reverse", [False, True], ids=["mask-vector", "vector-mask"])
def test_mask_interleave_rejects_mixed_categories(op_name, reverse) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="must both be MaskSSA.*both be VectorSSA"):
        _mixed_interleave.dump_mlir(type_args=(reverse, op_name))


@tla.kernel
def _invalid_mask_types(
    dtype0: tla.Constexpr[type], dtype1: tla.Constexpr[type], op_name: tla.Constexpr[str]
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            first = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype0)
            second = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype1)
            if tla.const_expr(op_name == "interleave"):
                tla.interleave(first, second)
            else:
                tla.deinterleave(first, second)


@pytest.mark.parametrize("op_name", ["interleave", "deinterleave"])
@pytest.mark.parametrize(
    ("dtype0", "dtype1", "message"),
    [
        (tla.Float32, tla.Float16, "same type"),
        (tla.Int64, tla.Int64, "b64 is unsupported"),
    ],
    ids=["different-lanes", "b64"],
)
def test_mask_interleave_rejects_invalid_types(op_name, dtype0, dtype1, message) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=message):
        _invalid_mask_types.dump_mlir(type_args=(dtype0, dtype1, op_name))


@pytest.mark.parametrize("op_name", ["interleave", "deinterleave"])
def test_mask_interleave_requires_vec_func(op_name) -> None:
    import catlass.runtime as runtime
    from catlass._mlir import ir

    # Supply a typed argument directly: variables created inside vec.func
    # are local to its staged body and cannot be referenced outside it.
    with runtime._eager_capture():
        module = ir.Module.parse(
            "module { func.func @mask_arg(%arg: !tla.mask<64>) { return } }"
        )
        argument = module.body.operations[0].regions[0].blocks[0].arguments[0]
        mask = MaskSSA(argument)
        with pytest.raises(tla.TlaCoreAPIError, match="vec.func"):
            getattr(tla, op_name)(mask, mask)
