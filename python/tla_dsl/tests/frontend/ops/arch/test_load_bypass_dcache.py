import pytest

from catlass import tla


@tla.kernel
def bypass_read(
    dtype: tla.Constexpr, space: tla.Constexpr, alignment: tla.Constexpr
) -> None:
    with tla.cube():
        addr = tla.make_ptr(dtype, 0, mem_space=space, assumed_align=alignment)
        tla.arch.load_bypass_dcache(ptr=addr + 1)


@tla.kernel
def bypass_read_without_core() -> None:
    tla.arch.load_bypass_dcache(tla.make_ptr(tla.Int32, 0))


@tla.kernel
def bypass_read_in_vec_func(mode: tla.Constexpr) -> None:
    with tla.vector():  # noqa: SIM117 - TLA lowers each region separately.
        with tla.vec.func(mode=mode):
            tla.arch.load_bypass_dcache(tla.make_ptr(tla.Int32, 0))


@pytest.mark.parametrize("width", (8, 16, 32, 64))
@pytest.mark.parametrize("signed", (True, False))
def test_bypass_read_preserves_pointee_type(width, signed):
    dtype = getattr(tla, f"{'Int' if signed else 'UInt'}{width}")
    ir_type = f"{'i' if signed else 'ui'}{width}"
    mlir = bypass_read.dump_mlir(type_args=(dtype, tla.AddressSpace.gm, width // 8))
    read = next(
        line for line in mlir.splitlines() if "tla.arch.load_bypass_dcache" in line
    )
    assert read.endswith(f": {ir_type}")


@pytest.mark.parametrize(
    "dtype,space,alignment,diagnostic",
    (
        (tla.Int32, tla.AddressSpace.ub, 4, "ptr must point to GM"),
        (tla.Float32, tla.AddressSpace.gm, 4, "ptr must point to an 8-"),
        (tla.Int32, tla.AddressSpace.gm, 1, "ptr must be naturally aligned"),
    ),
)
def test_bypass_read_rejects_invalid_address(dtype, space, alignment, diagnostic):
    with pytest.raises(tla.TlaCoreAPIError, match=diagnostic):
        bypass_read.dump_mlir(type_args=(dtype, space, alignment))


def test_bypass_read_requires_core_region():
    with pytest.raises(tla.TlaCoreAPIError, match="must be nested inside tla.cube"):
        bypass_read_without_core.dump_mlir()


@pytest.mark.parametrize("mode", ("simd", "simt"))
def test_bypass_read_rejects_vec_func(mode):
    with pytest.raises(tla.TlaCoreAPIError, match="is a scalar operation"):
        bypass_read_in_vec_func.dump_mlir(type_args=(mode,))


@tla.kernel
def bypass_read_in_mixed_regions() -> None:
    with tla.cube():  # noqa: SIM117 - The nested regions are the invalid case.
        with tla.vector():
            tla.arch.load_bypass_dcache(tla.make_ptr(tla.Int32, 0))


@tla.kernel
def bypass_read_after_byte_offset(first: tla.Constexpr, second: tla.Constexpr) -> None:
    with tla.cube():
        ptr = tla.make_ptr(tla.Int32, 0, assumed_align=4)
        byte_ptr = tla.recast_ptr(ptr, dtype=tla.UInt8) + first
        if second:
            byte_ptr = byte_ptr + second
        tla.arch.load_bypass_dcache(tla.recast_ptr(byte_ptr, dtype=tla.Int32))


def test_bypass_read_rejects_mixed_regions():
    with pytest.raises(tla.TlaCoreAPIError, match="mixed cube/vector placement"):
        bypass_read_in_mixed_regions.dump_mlir()


def test_bypass_read_preserves_alignment_through_byte_offset():
    with pytest.raises(tla.TlaCoreAPIError, match="ptr must be naturally aligned"):
        bypass_read_after_byte_offset.dump_mlir(type_args=(1, 0))
    for offsets in ((4, 0), (1, 3)):
        ir = bypass_read_after_byte_offset.dump_mlir(type_args=offsets)
        assert "tla.arch.load_bypass_dcache" in ir
