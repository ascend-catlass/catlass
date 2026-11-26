import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _tensor(dtype, lanes):
    return make_fake_tensor(
        dtype, (lanes,), (1,), addrspace=tla.AddressSpace.ub, origin_shape=(lanes,)
    )


@tla.kernel
def _apply(
    dst: tla.Tensor,
    factor: tla.Tensor,
    bias: tla.Tensor,
    mask_dtype: tla.Constexpr,
    mode: tla.Constexpr,
):
    with tla.vector():
        with tla.vec.func(mode=mode):
            if tla.const_expr(mode == "simt"):
                tla.arch.inplace_mul_add(tla.Float32(1), tla.Float32(2), tla.Float32(3))
            else:
                a, b, c = dst.load(), factor.load(), bias.load()
                if tla.const_expr(mask_dtype is None):
                    result = tla.arch.inplace_mul_add(a, b, c)
                else:
                    active = tla.create_mask(pattern=tla.mask.H, dtype=mask_dtype)
                    result = tla.arch.inplace_mul_add(a, b, c, mask=active)
                dst.store(result)


@pytest.mark.parametrize("dtype", [tla.Float16, tla.BFloat16, tla.Float32])
@pytest.mark.parametrize("masked", [False, True])
def test_inplace_mul_add_emits_one_operation(dtype, masked):
    tile = _tensor(dtype, 13)
    text = _apply.dump_mlir(
        type_args=(tile, tile, tile, dtype if masked else None, "simd")
    )
    (op,) = [line for line in text.splitlines() if "tla.inplace_mul_add" in line]
    assert (" mask " in op) == masked


@pytest.mark.parametrize(
    ("factor_type", "bias_len", "mask_type", "mode", "error"),
    [
        (tla.Float16, 64, None, "simd", "same type and valid lane count"),
        (tla.Float32, 13, None, "simd", "same type and valid lane count"),
        (tla.Float32, 64, tla.Float16, "simd", "mask"),
        (tla.Float32, 64, None, "simt", "mode='simd'"),
    ],
)
def test_inplace_mul_add_rejects_invalid_operands(
    factor_type, bias_len, mask_type, mode, error
):
    with pytest.raises(
        tla.TlaCoreAPIError, match=rf"tla\.arch\.inplace_mul_add.*{error}"
    ):
        _ = _apply.dump_mlir(
            type_args=(
                _tensor(tla.Float32, 64),
                _tensor(factor_type, 64),
                _tensor(tla.Float32, bias_len),
                mask_type,
                mode,
            )
        )


def test_inplace_mul_add_rejects_integer_registers():
    tile = _tensor(tla.Int32, 64)
    with pytest.raises(tla.TlaCoreAPIError, match="Float16, BFloat16 or Float32"):
        _ = _apply.dump_mlir(type_args=(tile, tile, tile, None, "simd"))
