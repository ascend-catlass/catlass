import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def ub(dtype, lanes=64):
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def addc_chain(a: tla.Tensor, b: tla.Tensor, out: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x, y = a.load(), b.load()
            zero = tla.create_mask(pattern=tla.mask.ALLF, dtype=tla.Int32)
            partial, carry = tla.addc(x, y, zero)
            total, _ = tla.addc(partial, y, carry)
            out.store(total)


@tla.kernel
def mull_pair(a: tla.Tensor, b: tla.Tensor, lo_out: tla.Tensor, hi_out: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x, y = a.load(), b.load()
            lo, hi = tla.mull(x, y)
            lo_out.store(lo)
            hi_out.store(hi)


@pytest.mark.parametrize("dtype", [tla.Int32, tla.UInt32])
def test_addc_returns_chainable_values(dtype):
    mlir = addc_chain.dump_mlir(type_args=tuple(ub(dtype) for _ in range(3)))
    assert mlir.count('"tla.addc"') == 2
    assert "!tla.mask<64>" in mlir


@pytest.mark.parametrize("dtype", [tla.Int32, tla.UInt32])
def test_mull_returns_both_halves(dtype):
    mlir = mull_pair.dump_mlir(type_args=tuple(ub(dtype) for _ in range(4)))
    assert mlir.count('"tla.mull"') == 1
    assert mlir.count("tla.store") == 2


@tla.kernel
def bad_addc_carry(a: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x = a.load()
            carry = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
            tla.addc(x, x, carry)


def test_addc_rejects_b16_carry():
    with pytest.raises(tla.TlaCoreAPIError, match="mask.*64"):
        bad_addc_carry.dump_mlir(type_args=(ub(tla.Int32),))


@tla.kernel
def subc_chain(a: tla.Tensor, b: tla.Tensor, out: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x, y = a.load(), b.load()
            active = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt32)
            partial, carry = tla.subc(x, y, active)
            diff, _ = tla.subc(partial, y, carry, mask=active)
            out.store(diff)


@pytest.mark.parametrize("dtype", [tla.Int32, tla.UInt32])
def test_subc_returns_chainable_values(dtype):
    mlir = subc_chain.dump_mlir(type_args=tuple(ub(dtype) for _ in range(3)))
    assert mlir.count('"tla.subc"') == 2
    assert "!tla.mask<64>" in mlir


@tla.kernel
def bad_subc_carry(a: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x = a.load()
            carry = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
            tla.subc(x, x, carry)


def test_subc_rejects_b16_carry():
    with pytest.raises(tla.TlaCoreAPIError, match="mask.*64"):
        bad_subc_carry.dump_mlir(type_args=(ub(tla.Int32),))


KERNEL_CASES = [
    pytest.param(addc_chain, 3, id="addc"),
    pytest.param(mull_pair, 4, id="mull"),
    pytest.param(subc_chain, 3, id="subc"),
]


@pytest.mark.parametrize("kernel,num_args", KERNEL_CASES)
@pytest.mark.parametrize("dtype,lanes", [(tla.Int16, 64), (tla.UInt32, 32)])
def test_b32_arithmetic_rejects_unsupported_registers(kernel, num_args, dtype, lanes):
    with pytest.raises(tla.TlaCoreAPIError, match="full 64-lane"):
        kernel.dump_mlir(type_args=tuple(ub(dtype, lanes) for _ in range(num_args)))


@pytest.mark.parametrize("kernel,num_args", KERNEL_CASES)
def test_b32_arithmetic_rejects_mixed_signedness(kernel, num_args):
    type_args = (ub(tla.Int32), ub(tla.UInt32), *(ub(tla.Int32) for _ in range(num_args - 2)))
    with pytest.raises(tla.TlaCoreAPIError, match="same VectorSSA type"):
        kernel.dump_mlir(type_args=type_args)
