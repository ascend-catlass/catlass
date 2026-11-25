from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import catlass.tla as tla
from catlass._mlir import ir
from catlass import _tla_type_bridge
from catlass.execution_lowering import TlaLoweringError


@tla.kernel
def raw_counter():
    with tla.vector():
        a = tla.arch.get_system_cycle()
        b = tla.arch.get_system_cycle()
        tla.make_coord(a, b)


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
                {a.name: a.attr for a in op.attributes},
            )
            for op in result
        ]


def test_raw_read_emits_distinct_i64_operations():
    reads = [
        types
        for name, types, _ in _operations(raw_counter)
        if name == "tla.arch.get_system_cycle"
    ]
    assert reads == [["i64"], ["i64"]]


@tla.kernel
def wrong_scope():
    tla.arch.get_system_cycle()


@tla.kernel
def in_vec_func():
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.arch.get_system_cycle()


@pytest.mark.parametrize(
    ("kernel", "diagnostic"),
    [
        (wrong_scope, "must be nested inside tla.cube"),
        (in_vec_func, "must execute on Scalar outside tla.vec.func"),
    ],
)
def test_invalid_counter_usage_fails_before_launch(kernel, diagnostic):
    with pytest.raises(
        (tla.TlaCoreAPIError, TlaLoweringError, tla.TlaIRNotExecutableError),
        match=diagnostic,
    ):
        kernel.dump_mlir()


def test_raw_read_rejects_host_execution():
    with pytest.raises((tla.TlaIRNotExecutableError, TlaLoweringError)):
        tla.arch.get_system_cycle()


def _counter_example_path():
    return (
        Path(__file__).resolve().parents[3]
        / "examples/python/tla_dsl/tla/kernel/system_counter/system_counter.py"
    )


@pytest.fixture
def counter_example():
    spec = importlib.util.spec_from_file_location(
        "system_counter_example_validation", _counter_example_path()
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_accepts_valid_counter_records(counter_example):
    counter_example.validate_counter_sample([100, 120, 20, -1, -1, -1, -1, -1])
    counter_example.validate_counter_sample([100, 100, 0, -1, -1, -1, -1, -1])


@pytest.mark.parametrize(
    "row",
    [
        [100, 120, 20, -1, -1, -1, 0, -1],
        [100, 120, 19, -1, -1, -1, -1, -1],
        [100, 120, 20, -1, -1, -1, -1],
    ],
)
def test_example_rejects_invalid_counter_records(counter_example, row):
    with pytest.raises(RuntimeError):
        counter_example.validate_counter_sample(row)


def test_example_checks_remain_enabled_under_python_optimization():
    script = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('counter_example', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.validate_counter_sample([100, 120, 20, -1, -1, -1, -1, -1])
for row in ([100, 120, 20, -1, -1, -1, 0, -1],
            [100, 120, 19, -1, -1, -1, -1, -1],
            [100, 120, 20, -1, -1, -1, -1]):
    try:
        module.validate_counter_sample(row)
    except RuntimeError:
        continue
    raise SystemExit('Invalid record was accepted under Python -O')
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", script, str(_counter_example_path())],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_example_rejects_constant_counter_for_workload(counter_example):
    samples = [[0, 0, 0, -1, -1, -1, -1, -1]]
    with pytest.raises(RuntimeError, match="System counter did not advance"):
        counter_example.summarize_counter_samples(32, samples)
    assert (
        counter_example.summarize_counter_samples(0, samples)["elapsed_ticks_median"]
        == 0
    )
