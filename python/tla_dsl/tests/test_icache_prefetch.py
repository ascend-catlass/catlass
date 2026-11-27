import pytest

from catlass import tla
from catlass.tla.params import ICachePrefetchStatus


@tla.kernel
def _vector_prefetch(units: tla.Constexpr):
    with tla.vector():
        tla.arch.icache_prefetch(units)
        status = tla.arch.get_icache_prefetch_status()
        if status == ICachePrefetchStatus.COMPLETE:
            tla.arch.icache_prefetch(2)


@tla.kernel
def _cube_prefetch(units: tla.Constexpr):
    with tla.cube():
        tla.arch.icache_prefetch(units)


@tla.kernel
def _vf_status(mode: tla.Constexpr):
    with tla.vector():
        with tla.vec.func(mode=mode, thread_block_dim=32 if mode == "simt" else None):
            _ = tla.arch.get_icache_prefetch_status()


@pytest.mark.parametrize(
    "kernel, units, core",
    [
        (_vector_prefetch, 0, "aiv"),
        (_vector_prefetch, 7, "aiv"),
        (_cube_prefetch, 0, "aic"),
        (_cube_prefetch, 15, "aic"),
    ],
)
def test_scalar_icache_frontend(kernel, units, core):
    from catlass._mlir import ir

    with ir.Context() as ctx:
        from catlass import _tla_type_bridge, compiler_bridge

        _tla_type_bridge.load_tla_dialect(ctx)
        module = ir.Module.parse(kernel.dump_mlir(type_args=(units,)))
        assert module.operation.verify()
        result = compiler_bridge.lower_tlair_module_to_mlir(module)
        assert result.icache_prefetch_cores == {core}


@pytest.mark.parametrize(
    "kernel, units",
    [
        (_vector_prefetch, -1),
        (_vector_prefetch, True),
        (_vector_prefetch, 8),
        (_cube_prefetch, -1),
        (_cube_prefetch, 16),
    ],
)
def test_prefetch_rejects_invalid_units(kernel, units):
    with pytest.raises(Exception, match="prefetch_units"):
        _ = kernel.dump_mlir(type_args=(units,))


@pytest.mark.parametrize("mode", ["simd", "simt"])
def test_status_rejects_vf(mode):
    with pytest.raises(Exception, match="outside tla.vec.func"):
        _ = _vf_status.dump_mlir(type_args=(mode,))


@tla.kernel
def _uncategorized_status():
    _ = tla.arch.get_icache_prefetch_status()


def test_status_requires_explicit_scalar_region():
    with pytest.raises(Exception, match="scalar vector or cube"):
        _uncategorized_status.dump_mlir()


@tla.kernel
def _poll_status():
    with tla.vector():
        status = tla.arch.get_icache_prefetch_status()
        retries = tla.Int64(0)
        incomplete = tla.params.ICachePrefetchStatus.INCOMPLETE
        while retries < tla.Int64(4) and status == incomplete:
            status = tla.arch.get_icache_prefetch_status()
            retries = retries + tla.Int64(1)
        if status == tla.Int64(0):
            tla.arch.icache_prefetch(2)


def test_status_supports_loop_carried_values_and_numeric_comparison():
    from catlass import _tla_type_bridge
    from catlass._mlir import ir

    def operations(op):
        yield op
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from operations(child.operation)

    with ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        module = ir.Module.parse(_poll_status.dump_mlir())
        assert module.operation.verify()
        ops = list(operations(module.operation))
        (loop,) = (op for op in ops if op.name == "scf.while")
        compared_results = [
            operand
            for op in ops
            if op.name == "arith.cmpi"
            for operand in op.operands
            if operand in loop.results
        ]
        assert compared_results
        assert all(
            result.type == ir.IntegerType.get_signless(64)
            for result in compared_results
        )


@tla.kernel
def _mixed_conditional_prefetch():
    with tla.cube():
        tla.arch.icache_prefetch(15)
    with tla.vector():
        status = tla.arch.get_icache_prefetch_status()
        complete = tla.params.ICachePrefetchStatus.COMPLETE
        if tla.arch.block_idx() == 0 and status == complete:
            tla.arch.icache_prefetch(7)


def test_mixed_conditional_prefetch_routes_to_both_issuers():
    from catlass import _tla_type_bridge, compiler_bridge
    from catlass._mlir import ir

    with ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        module = ir.Module.parse(_mixed_conditional_prefetch.dump_mlir())
        result = compiler_bridge.lower_tlair_module_to_mlir(module)
        assert result.icache_prefetch_cores == {"aic", "aiv"}
