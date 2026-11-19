"""SIMT warp reduction types, attributes, and frontend diagnostics."""

import pytest

from catlass import tla
from catlass.tla.runtime import make_fake_tensor

TYPES = (
    tla.Int8,
    tla.Int16,
    tla.Int32,
    tla.UInt8,
    tla.UInt16,
    tla.UInt32,
    tla.Float16,
    tla.Float32,
)


def _reduce(value, kind):
    result = tla.arch.warp_reduce_sync(value, kind)
    assert isinstance(result, type(value))
    return result


@tla.kernel
def _reduce_kernel(out: tla.Tensor, dtype: tla.Constexpr, kind: tla.Constexpr):
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode="simt", thread_block_dim=48):
            out[0] = _reduce(dtype(1), kind).to(tla.Float32)


@tla.kernel
def _default(out: tla.Tensor):
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode="simt", thread_block_dim=32):
            out[0] = tla.arch.warp_reduce_sync(1)


@tla.kernel
def _wrong_region(mode: tla.Constexpr):
    with tla.vector():
        if mode == "vector":
            _ = tla.arch.warp_reduce_sync(1)
        else:
            with tla.vec.func(mode="simd"):
                _ = tla.arch.warp_reduce_sync(1)


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("kind", ("add", "max", "min"))
def test_reduce_type_and_kind(dtype, kind):
    out = make_fake_tensor(tla.Float32, (1,), (1,))
    text = _reduce_kernel.dump_mlir(type_args=(out, dtype, kind))
    assert f'tla.arch.warp_reduce "{kind}"' in text
    assert ("isUnsigned" in text) == (dtype in (tla.UInt8, tla.UInt16, tla.UInt32))


def test_reduce_default_and_literal():
    out = make_fake_tensor(tla.Int32, (1,), (1,))
    assert 'tla.arch.warp_reduce "add"' in _default.dump_mlir(type_args=(out,))


@pytest.mark.parametrize(
    "dtype",
    (
        tla.Int64,
        tla.UInt64,
        tla.BFloat16,
        tla.Bool,
        tla.Float8E4M3FN,
        tla.Float8E5M2,
        tla.Float8E8M0,
        tla.Float4E2M1,
        tla.Float4E1M2,
    ),
)
def test_reduce_unsupported_type(dtype):
    out = make_fake_tensor(tla.Float32, (1,), (1,))
    with pytest.raises(
        Exception, match="arch.warp_reduce_sync.*unsupported value type"
    ):
        _reduce_kernel.dump_mlir(type_args=(out, dtype, "add"))


@pytest.mark.parametrize("kind", ("sum", "ADD", "", None, 1))
def test_reduce_invalid_kind(kind):
    out = make_fake_tensor(tla.Float32, (1,), (1,))
    with pytest.raises(Exception, match="arch.warp_reduce_sync.*kind"):
        _reduce_kernel.dump_mlir(type_args=(out, tla.Int32, kind))


@pytest.mark.parametrize("mode", ("vector", "simd"))
def test_reduce_requires_simt(mode):
    with pytest.raises(Exception, match="arch.warp_reduce_sync.*simt"):
        _wrong_region.dump_mlir(type_args=(mode,))


def test_reduce_requires_kernel():
    with pytest.raises(Exception, match="arch.warp_reduce_sync"):
        tla.arch.warp_reduce_sync(1)


@tla.kernel
def _runtime_kind(src: tla.Tensor):
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode="simt", thread_block_dim=32):
            _ = tla.arch.warp_reduce_sync(1, src[0])


def test_reduce_rejects_runtime_kind():
    src = make_fake_tensor(tla.Int32, (1,), (1,))
    with pytest.raises(Exception, match="arch.warp_reduce_sync.*kind"):
        _runtime_kind.dump_mlir(type_args=(src,))


@tla.kernel
def _divergent_reduce(out: tla.Tensor, kind: tla.Constexpr):
    with tla.vector():  # noqa: SIM117 - preserve DSL region nesting
        with tla.vec.func(mode="simt", thread_block_dim=32):
            tid, _, _ = tla.arch.thread_idx()
            one = tla.Int32(1)
            out[tid] = tla.arch.warp_reduce_sync(one, kind)
            if tid % 2 == 0:
                out[32 + tid] = tla.arch.warp_reduce_sync(one, kind)
            out[64 + tid] = tla.arch.warp_reduce_sync(one, kind)


@pytest.mark.parametrize("kind", ("add", "max", "min"))
@pytest.mark.parametrize(
    "pipeline", ("builtin.module(cse)", "builtin.module(canonicalize,cse)")
)
def test_reduce_preserves_active_set_before_lowering(kind, pipeline):
    from catlass._mlir import passmanager
    from catlass.base_dsl import BaseDSL

    out = make_fake_tensor(tla.Int32, (96,), (1,))
    lowered = BaseDSL()._lower(
        _divergent_reduce.fn,
        kind=_divergent_reduce.kind,
        options=dict(_divergent_reduce.options),
        type_args=(out, kind),
        location=_divergent_reduce.decorator_location,
    )
    module = lowered.module
    with module.context:
        passmanager.PassManager.parse(pipeline).run(module.operation)
    parents = []

    def visit(op):
        if op.name == "tla.arch.warp_reduce":
            parents.append(op.parent.name)
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)

    visit(module.operation)
    assert parents == ["tla.vec.func", "scf.if", "tla.vec.func"]
    assert module.operation.verify()
