"""Behavioral frontend and verifier coverage for the active-lane query."""

import pytest

from catlass import tla
from catlass._mlir import ir
from catlass.base_dsl import BaseDSL
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def _query(out: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=(48, 1, 1)):
            value = tla.arch.activemask()
            out[0] = value.to(tla.UInt64)


@tla.kernel
def _wrong_region(where: tla.Constexpr[str]):
    if where == "kernel":
        tla.arch.activemask()
    with tla.vector():
        if where == "vector":
            tla.arch.activemask()
        else:
            with tla.vec.func(mode="simd"):
                tla.arch.activemask()


def test_activemask_lowering_and_unsigned_result():
    out = make_fake_tensor(tla.UInt64, (1,), (1,))
    lowered = BaseDSL()._lower(
        _query.fn,
        kind=_query.kind,
        options=dict(_query.options),
        type_args=(out,),
        location=_query.decorator_location,
    )
    assert lowered.module.operation.verify()
    ops = []

    def visit(op):
        ops.append(op)
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)

    visit(lowered.module.operation)
    queries = [op for op in ops if op.name == "tla.arch.activemask"]
    assert len(queries) == 1
    assert queries[0].results[0].type.width == 32
    casts = [op for op in ops if op.name == "tla.simt_cast"]
    assert len(casts) == 1
    assert ir.StringAttr(casts[0].attributes["kind"]).value == "extui"
    assert casts[0].results[0].type.width == 64


@pytest.mark.parametrize("where", ["kernel", "vector", "simd"])
def test_activemask_rejects_wrong_scope(where):
    with pytest.raises(Exception, match=r"arch\.activemask"):
        _wrong_region.dump_mlir(type_args=(where,))


def test_activemask_rejects_host():
    with pytest.raises(Exception, match=r"arch\.activemask"):
        tla.arch.activemask()


def test_cse_preserves_queries_in_distinct_active_sets():
    from catlass import _tla_type_bridge
    from catlass._mlir import passmanager

    with ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        module = ir.Module.parse("""
          module {
            func.func private @consume(i32)
            func.func @query(%condition: i1) {
              "tla.vector"() ({
                "tla.vec.func"() ({
                  %outer = tla.arch.activemask -> i32
                  func.call @consume(%outer) : (i32) -> ()
                  scf.if %condition {
                    %branch = tla.arch.activemask -> i32
                    func.call @consume(%branch) : (i32) -> ()
                  }
                }) {mode = "simt", thread_block_dim = array<i64: 32, 1, 1>} : () -> ()
              }) : () -> ()
              return
            }
          }
        """)
        assert module.operation.verify()
        pm = passmanager.PassManager.parse("builtin.module(cse)")
        pm.run(module.operation)
        function = module.body.operations[1]
        vector = function.regions[0].blocks[0].operations[0]
        simt = vector.regions[0].blocks[0].operations[0]
        body = simt.regions[0].blocks[0]
        outer = body.operations[0].operation
        branch = body.operations[2].operation
        inner = branch.regions[0].blocks[0].operations[0].operation
        assert outer.name == "tla.arch.activemask"
        assert inner.name == "tla.arch.activemask"
        assert inner.results[0] != outer.results[0]
