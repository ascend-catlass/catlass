import pytest

from catlass import tla


@tla.kernel
def bypass_store(
    offset: tla.Int32,
    value: tla.Int64,
    dtype: tla.Constexpr,
    space: tla.Constexpr,
    alignment: tla.Constexpr,
) -> None:
    with tla.vector():
        ptr = tla.make_ptr(dtype, 0, mem_space=space, assumed_align=alignment)
        tla.arch.store_bypass_dcache(ptr=ptr + offset, value=value)


def test_bypass_store_emits_pointer_offset_and_scalar_store():
    ir = bypass_store.dump_mlir(
        type_args=(tla.Int32(0), tla.Int64(0), tla.Int32, tla.AddressSpace.gm, 4)
    )
    assert "tla.ptr_add" in ir
    assert "tla.ptrtoint" in ir
    assert "arith.trunci" in ir
    assert "tla.arch.store_bypass_dcache" in ir


@pytest.mark.parametrize(
    "dtype,space,alignment,diagnostic",
    (
        (tla.Int32, tla.AddressSpace.ub, 4, "ptr must point to GM"),
        (tla.UInt32, tla.AddressSpace.gm, 4, "ptr must point to an Int32 element"),
        (tla.Int32, tla.AddressSpace.gm, 1, "ptr must be naturally aligned"),
    ),
)
def test_bypass_store_rejects_invalid_pointer(dtype, space, alignment, diagnostic):
    with pytest.raises(tla.TlaCoreAPIError, match=diagnostic):
        bypass_store.dump_mlir(
            type_args=(tla.Int32(0), tla.Int64(0), dtype, space, alignment)
        )
