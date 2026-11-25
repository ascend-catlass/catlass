from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass import _tla_type_bridge
from catlass._mlir import ir
from catlass.execution_lowering import TlaLoweringError


@tla.kernel
def trap_in_vector():
    with tla.vector():
        tla.trap()


@tla.kernel
def trap_in_cube():
    with tla.cube():
        tla.trap()


@tla.kernel
def trap_in_simd():
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.trap()


@tla.kernel
def trap_in_simt():
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=32):
            tla.trap()


def _operations(kernel):
    with ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        module = ir.Module.parse(kernel.dump_mlir())
        result = []

        def visit(op):
            result.append(op)
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations:
                        visit(child.operation)

        visit(module.operation)
        return [
            (
                op.name,
                [str(r.type) for r in op.results],
                {a.name for a in op.attributes},
            )
            for op in result
        ]


@pytest.mark.parametrize("kernel", [trap_in_vector, trap_in_cube])
def test_trap_emits_a_resultless_side_effecting_op(kernel):
    traps = [entry for entry in _operations(kernel) if entry[0] == "tla.trap"]
    assert len(traps) == 1
    _, results, attributes = traps[0]
    assert results == []
    assert attributes == set()


@pytest.mark.parametrize("kernel", [trap_in_simd, trap_in_simt])
def test_trap_is_rejected_inside_vec_func(kernel):
    """Only scalar placement is supported; a vector function has no lowering."""

    with pytest.raises(
        (tla.TlaCoreAPIError, TlaLoweringError, tla.TlaIRNotExecutableError),
        match=r"requires scalar code outside tla\.vec\.func",
    ):
        kernel.dump_mlir()


def test_trap_is_callable_and_exported():
    assert tla.trap is not None


def test_trap_rejects_host_execution():
    with pytest.raises((tla.TlaIRNotExecutableError, TlaLoweringError)):
        tla.trap()
