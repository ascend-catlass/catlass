from __future__ import annotations

import subprocess
import sys

import pytest

import catlass.tla as tla
from catlass._mlir import ir


@tla.kernel
def scoped() -> None:
    with tla.vector():
        with tla.prof_scope():
            tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def explicit() -> None:
    with tla.cube():
        tla.prof_start()
        tla.pipe_barrier(tla.pipes.ALL)
        tla.prof_end()


def _op_names(kernel):
    # Inspect the actual staged operation tree, not source text.
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        # A supplied signature forces staging instead of reusing cached IR.
        module = ir.Module.parse(kernel.dump_mlir(type_args=[]))
        names = []

        def visit(op):
            names.append(op.operation.name)
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations:
                        visit(child)

        visit(module.operation)
        return names


@pytest.mark.parametrize("kernel", [scoped, explicit])
def test_pair_preserves_body_order(kernel):
    names = _op_names(kernel)
    assert names.count("tla.prof_start") == names.count("tla.prof_end") == 1
    assert names.index("tla.prof_start") < names.index("tla.pipe_barrier")
    assert names.index("tla.pipe_barrier") < names.index("tla.prof_end")
    # Each new lowering owns its own state.
    assert _op_names(kernel) == names


@tla.kernel
def nested() -> None:
    with tla.prof_scope():
        with tla.prof_scope():
            pass


@tla.kernel
def duplicate() -> None:
    tla.prof_start()
    tla.prof_end()
    tla.prof_start()


@tla.kernel
def reversed_pair() -> None:
    tla.prof_end()


@tla.kernel
def missing_end() -> None:
    tla.prof_start()


@tla.kernel
def duplicate_end() -> None:
    tla.prof_start()
    tla.prof_end()
    tla.prof_end()


@pytest.mark.parametrize(
    "kernel, message",
    [
        (nested, "only one"),
        (duplicate, "only one"),
        (reversed_pair, "unmatched"),
        (missing_end, "missing prof_end"),
        (duplicate_end, "unmatched"),
    ],
)
def test_obvious_misuse(kernel, message):
    with pytest.raises(tla.TlaCoreAPIError, match=message):
        kernel.dump_mlir()


@pytest.mark.parametrize("operation", [tla.prof_start, tla.prof_end])
def test_requires_staging(operation):
    with pytest.raises(tla.TlaIRNotExecutableError):
        operation()


@pytest.mark.parametrize("symbol", ["prof_start", "prof_end"])
def test_missing_generated_binding_fails_at_import(symbol):
    # Isolate reload side effects from the other frontend tests.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
import catlass.core_api as api
delattr(api._tla_ops_gen, sys.argv[1])
try:
    importlib.reload(api)
except RuntimeError as error:
    if f"Generated Tla binding is missing `{sys.argv[1]}`" not in str(error):
        raise
else:
    raise RuntimeError("missing generated binding was accepted")
""",
            symbol,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
