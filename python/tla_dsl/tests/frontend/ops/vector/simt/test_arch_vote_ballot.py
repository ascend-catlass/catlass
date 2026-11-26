"""Warp vote predicate conversion, result types, scope, and CSE behavior."""

import pytest

from catlass import tla, _tla_type_bridge
from catlass._mlir import ir, passmanager
from catlass.base_dsl import BaseDSL
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def _query(
    out: tla.Tensor, dtype: tla.Constexpr, value: tla.Constexpr, vote: tla.Constexpr
):
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=48):
            out[0] = vote(dtype(value)).to(tla.UInt64)


@tla.kernel
def _literal(out: tla.Tensor, value: tla.Constexpr, vote: tla.Constexpr):
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=48):
            out[0] = vote(value).to(tla.UInt64)


@tla.kernel
def _wrong_scope(scope: tla.Constexpr, vote: tla.Constexpr):
    if scope == "kernel":
        vote(True)
    with tla.vector():
        if scope == "vector":
            vote(True)
        else:
            with tla.vec.func(mode="simd"):
                vote(True)


@tla.kernel
def _tensor_predicate(out: tla.Tensor, vote: tla.Constexpr):
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=32):
            vote(out)


def _lower(kernel, args):
    return BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=args,
        location=kernel.decorator_location,
    )


def _ops(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _ops(child.operation)


@pytest.fixture(params=["ballot", "all", "any"])
def vote_kind(request):
    return request.param


def _vote(kind):
    return getattr(tla.arch, f"vote_{kind}_sync")


@pytest.mark.parametrize(
    "dtype",
    [
        tla.Bool,
        tla.Int8,
        tla.Int16,
        tla.Int32,
        tla.Int64,
        tla.UInt8,
        tla.UInt16,
        tla.UInt32,
        tla.UInt64,
    ],
)
@pytest.mark.parametrize("value", [0, 1])
def test_predicate_types_and_unsigned_result(dtype, value, vote_kind):
    out = make_fake_tensor(tla.UInt64, (1,), (1,))
    lowered = _lower(_query, (out, dtype, value, _vote(vote_kind)))
    assert lowered.module.operation.verify()
    ops = list(_ops(lowered.module.operation))
    ballot = [op for op in ops if op.name == f"tla.arch.vote_{vote_kind}"]
    assert len(ballot) == 1
    assert ir.IntegerType(ballot[0].operands[0].type).width == 1
    assert ir.IntegerType(ballot[0].results[0].type).width == (
        32 if vote_kind == "ballot" else 1
    )
    widening = [
        op
        for op in ops
        if op.name == "tla.simt_cast" and ir.IntegerType(op.results[0].type).width == 64
    ]
    assert len(widening) == 1
    assert ir.StringAttr(widening[0].attributes["kind"]).value == "extui"


@pytest.mark.parametrize("value", [False, True, 0, -1, 1 << 40, 1 << 100])
def test_python_constants(value, vote_kind):
    out = make_fake_tensor(tla.UInt64, (1,), (1,))
    lowered = _lower(_literal, (out, value, _vote(vote_kind)))
    assert lowered.module.operation.verify()
    ballot = next(
        op
        for op in _ops(lowered.module.operation)
        if op.name == f"tla.arch.vote_{vote_kind}"
    )
    constant = ballot.operands[0].owner
    assert bool(ir.IntegerAttr(constant.attributes["value"]).value) == bool(value)


@pytest.mark.parametrize(
    "dtype", [tla.Float16, tla.BFloat16, tla.Float32, tla.Float8E4M3FN]
)
def test_reject_floating_predicate(dtype, vote_kind):
    out = make_fake_tensor(tla.UInt64, (1,), (1,))
    with pytest.raises(
        Exception, match=f"arch.vote_{vote_kind}_sync.*Bool or integer scalar"
    ):
        _ = _lower(_query, (out, dtype, 1, _vote(vote_kind)))


@pytest.mark.parametrize("value", [1.0, "predicate", (1,), None])
def test_reject_non_integer_literals(value, vote_kind):
    out = make_fake_tensor(tla.UInt64, (1,), (1,))
    with pytest.raises(
        Exception, match=f"arch.vote_{vote_kind}_sync.*Bool or integer scalar"
    ):
        _ = _lower(_literal, (out, value, _vote(vote_kind)))


def test_reject_tensor_predicate(vote_kind):
    out = make_fake_tensor(tla.UInt64, (1,), (1,))
    with pytest.raises(
        Exception, match=f"arch.vote_{vote_kind}_sync.*Bool or integer scalar"
    ):
        _ = _lower(_tensor_predicate, (out, _vote(vote_kind)))


@pytest.mark.parametrize("scope", ["kernel", "vector", "simd"])
def test_requires_simt(scope, vote_kind):
    with pytest.raises(Exception, match=f"arch.vote_{vote_kind}_sync.*simt"):
        _ = _lower(_wrong_scope, (scope, _vote(vote_kind)))


def test_reject_host(vote_kind):
    with pytest.raises(Exception, match=f"arch.vote_{vote_kind}_sync"):
        _vote(vote_kind)(True)


def test_cse_preserves_distinct_active_sets(vote_kind):
    with ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        module = ir.Module.parse(
            """
          module {
            func.func private @consume(i32)
            func.func @ballot(%condition: i1) {
              "tla.vector"() ({
                "tla.vec.func"() ({
                  %true = arith.constant true
                  %outer = tla.arch.vote_ballot %true : i32
                  func.call @consume(%outer) : (i32) -> ()
                  scf.if %condition {
                    %inner = tla.arch.vote_ballot %true : i32
                    func.call @consume(%inner) : (i32) -> ()
                  }
                }) {mode = "simt", thread_block_dim = array<i64: 32, 1, 1>} : () -> ()
              }) : () -> ()
              return
            }
          }
        """.replace("vote_ballot", f"vote_{vote_kind}").replace(
                "i32", "i32" if vote_kind == "ballot" else "i1"
            )
        )
        assert module.operation.verify()
        passmanager.PassManager.parse("builtin.module(cse)").run(module.operation)
        ballots = []
        for op in _ops(module.operation):
            if op.name == f"tla.arch.vote_{vote_kind}":
                ballots.append(op)
        assert len(ballots) == 2
        assert ballots[0].results[0] != ballots[1].results[0]
        assert ballots[0].parent != ballots[1].parent
