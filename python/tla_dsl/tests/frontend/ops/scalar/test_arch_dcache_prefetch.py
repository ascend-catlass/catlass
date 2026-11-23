import pytest

from catlass import tla
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def _request(src: tla.Tensor, offset: tla.Constexpr):
    with tla.vector():
        tla.arch.dcache_prefetch(src.ptr, offset)


@tla.kernel
def _vf(src: tla.Tensor, mode: tla.Constexpr):
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode=mode):
            tla.arch.dcache_prefetch(src.ptr)


@pytest.mark.parametrize("offset", [False, 0.5, 1 << 63, -(1 << 63) - 1])
def test_invalid_offset(offset):
    src = make_fake_tensor(tla.Int32, (64,), (1,))
    with pytest.raises(Exception, match="arch.dcache_prefetch.*byte_offset"):
        _request.dump_mlir(type_args=(src, offset))


@pytest.mark.parametrize("mode", ["simt", "simd"])
def test_rejects_vector_function(mode):
    src = make_fake_tensor(tla.Int32, (64,), (1,))
    with pytest.raises(Exception, match="arch.dcache_prefetch.*scalar"):
        _vf.dump_mlir(type_args=(src, mode))


@tla.kernel
def _dynamic(src: tla.Tensor, offset: tla.Int16):
    with tla.cube():
        tla.arch.dcache_prefetch(src.ptr, offset)


@tla.kernel
def _tensor_as_address(src: tla.Tensor):
    with tla.vector():
        tla.arch.dcache_prefetch(src)


def test_signed_byte_offset():
    src = make_fake_tensor(tla.Int32, (64,), (1,))
    assert "tla.arch.dcache_prefetch" in _dynamic.dump_mlir(
        type_args=(src, tla.Int16(-2))
    )
    for offset in (-(1 << 63), 0, (1 << 63) - 1):
        assert "tla.arch.dcache_prefetch" in _request.dump_mlir(type_args=(src, offset))


def test_rejects_implicit_tensor_address():
    src = make_fake_tensor(tla.Int32, (64,), (1,))
    with pytest.raises(Exception, match="typed GM pointer"):
        _tensor_as_address.dump_mlir(type_args=(src,))


def test_rejects_ub_pointer():
    src = make_fake_tensor(tla.Int32, (64,), (1,), addrspace=tla.AddressSpace.ub)
    with pytest.raises(Exception, match="typed GM pointer"):
        _request.dump_mlir(type_args=(src, 0))


@tla.kernel
def _repeat(src: tla.Tensor):
    with tla.vector():
        tla.arch.dcache_prefetch(src.ptr)
        tla.arch.dcache_prefetch(src.ptr)


def test_cse_preserves_both_requests():
    from catlass._mlir import passmanager
    from catlass.base_dsl import BaseDSL

    src = make_fake_tensor(tla.Int32, (64,), (1,))
    module = (
        BaseDSL()
        ._lower(
            _repeat.fn,
            kind=_repeat.kind,
            options=dict(_repeat.options),
            type_args=(src,),
            location=_repeat.decorator_location,
        )
        .module
    )
    with module.context:
        passmanager.PassManager.parse("builtin.module(canonicalize,cse)").run(
            module.operation
        )
    assert str(module).count("tla.arch.dcache_prefetch") == 2


@tla.kernel
def _nested_tile(src: tla.Tensor):
    with tla.vector():
        parent = tla.get_tile(src, tla.make_coord(1, 0), tla.make_shape(2, 16))
        child = tla.get_tile(parent, tla.make_coord(1, 0), tla.make_shape(1, 16))
        tla.arch.dcache_prefetch(src.ptr, byte_offset=128)
        tla.arch.dcache_prefetch(child.ptr, byte_offset=128)
        tile = tla.tile_view(src, tla.make_shape(1, 16), tla.make_coord(2, 0))
        alias = tla.make_tensor(
            tile.ptr, tla.make_layout(tla.make_shape(1, 16), tla.make_stride(16, 1))
        )
        tla.arch.dcache_prefetch(alias.ptr, byte_offset=128)


def test_nested_get_tile_uses_same_backing_pointer():
    from catlass.compiler_bridge import lower_tlair_module_to_mlir
    from catlass.base_dsl import BaseDSL

    src = make_fake_tensor(tla.Int32, (4, 16), (16, 1), layout_tag=tla.arch.RowMajor)
    module = (
        BaseDSL()
        ._lower(
            _nested_tile.fn,
            kind=_nested_tile.kind,
            options=dict(_nested_tile.options),
            type_args=(src,),
            location=_nested_tile.decorator_location,
        )
        .module
    )
    lowered = lower_tlair_module_to_mlir(module).lowered_mlir
    calls = [
        line.partition(" loc(")[0].strip()
        for line in lowered.splitlines()
        if "call @_mlir_ciface_tla_dcache_prefetch_aiv(" in line
    ]
    assert len(calls) == 3
    assert len(set(calls)) == 1
