from __future__ import annotations

from collections import deque
import importlib
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


@pytest.fixture
def shift_example(monkeypatch):
    repo = Path(__file__).resolve().parents[7]
    monkeypatch.syspath_prepend(
        str(repo / "examples/python/tla_dsl/dsl_tutorials/vector_simd_ops")
    )
    example = importlib.import_module("shift_op")
    for name in (
        "VECTOR_ELE",
        "VL_ELE",
        "LOOPS",
        "SHIFT_BITS",
        "_KERNEL_DTYPE",
        "_AMOUNT_DTYPE",
        "_IS_VECTOR_SHIFT",
        "_MASK_PATTERN",
        "_IS_64_BIT",
        "_KERNEL_SHAPE",
        "_SHIFT_OP",
        "_VV64_TILE_ELE",
    ):
        monkeypatch.setattr(example, name, getattr(example, name))
    return example


def _assert_event_protocol(ir: str) -> None:
    """Explore every ordering allowed by three independent FIFO pipelines.

    A wait consumes one notification. A repeated set before consumption, an
    unconsumed final notification, or a state with no enabled pipe is illegal.
    Copies and computation may delay a pipe but cannot add event ordering.
    """
    pipes = ("mte2", "vector", "mte3")
    queues = [[] for _ in pipes]
    flags = {}
    for value, dest, source in re.findall(
        r'(%\w+) = tla.flag "[^"]+" \{dst_pipe = #tla.pipe<(\w+)>, src_pipe = #tla.pipe<(\w+)>\}',
        ir,
    ):
        flags[value] = (len(flags), source, dest)
    for action, value in re.findall(r"tla\.(set_flag|wait_flag) (%\w+)", ir):
        bit, source, dest = flags[value]
        is_set = action == "set_flag"
        queues[pipes.index(source if is_set else dest)].append((is_set, 1 << bit))
    assert all(queues), "expected events on all three pipelines"

    initial = ((0, 0, 0), 0)
    pending = deque([initial])
    visited = {initial}
    while pending:
        positions, bits = pending.popleft()
        if all(pos == len(queue) for pos, queue in zip(positions, queues, strict=True)):
            assert bits == 0, "unconsumed final event"
            continue
        enabled = False
        for pipe, queue in enumerate(queues):
            pos = positions[pipe]
            if pos == len(queue):
                continue
            is_set, bit = queue[pos]
            if is_set:
                assert not bits & bit, "event set again before its wait consumed it"
            elif not bits & bit:
                continue
            enabled = True
            next_positions = list(positions)
            next_positions[pipe] += 1
            state = (tuple(next_positions), bits | bit if is_set else bits & ~bit)
            if state not in visited:
                visited.add(state)
                pending.append(state)
        assert enabled, "event protocol deadlock"


def _assert_tiled_pipeline(ir: str, tiles: int) -> None:
    assert ir.count("tla.copy ") == 3 * tiles, "expected three tile copies per block"
    allocations = re.findall(r"tla.alloc_ptr\s*\{size_bytes = (\d+)\}", ir)
    assert [int(n) for n in allocations] == [16384] * 3, (
        "expected fixed two-slot UB buffers"
    )
    ready_counts = []
    for value in re.findall(
        r'(%\w+) = tla.flag "[^"]+" \{dst_pipe = #tla.pipe<vector>, src_pipe = #tla.pipe<mte2>\}',
        ir,
    ):
        count = len(re.findall(r"tla.set_flag " + re.escape(value) + r" :", ir))
        if count:
            ready_counts.append(count)
    assert sorted(ready_counts) == [tiles // 2, (tiles + 1) // 2]
    if tiles >= 3:
        assert max(ready_counts) >= 2, "expected a reused readiness event"


@pytest.mark.parametrize("dtype", (tla.Int64, tla.UInt64))
@pytest.mark.parametrize("op", ("shift_left", "shift_right"))
@pytest.mark.parametrize("tiles", (2, 3, 4, 8, 11))
def test_vv64_pipeline_event_reuse(shift_example, dtype, op, tiles):
    n = tiles * 1024
    shift_example._set_kernel_config(op, "u64" if dtype is tla.UInt64 else "i64", (n,))

    def tensor(element):
        return make_fake_tensor(
            element,
            (n,),
            (1,),
            addrspace=tla.AddressSpace.gm,
            origin_shape=(n,),
            layout_tag=tla.arch.RowMajor,
        )

    data = tensor(dtype)
    counts = tensor(tla.Int64)
    ir = shift_example.shift_op.dump_mlir(type_args=(data, counts, data))
    _assert_tiled_pipeline(ir, tiles)
    _assert_event_protocol(ir)


def test_pipeline_check_rejects_ordinary_path(shift_example, monkeypatch):
    monkeypatch.setattr(shift_example, "_VV64_TILE_ELE", 1 << 20)
    shift_example._set_kernel_config("shift_right", "u64", (4096,))

    def tensor(dtype):
        return make_fake_tensor(
            dtype,
            (4096,),
            (1,),
            addrspace=tla.AddressSpace.gm,
            origin_shape=(4096,),
            layout_tag=tla.arch.RowMajor,
        )

    data, counts = tensor(tla.UInt64), tensor(tla.Int64)
    ir = shift_example.shift_op.dump_mlir(type_args=(data, counts, data))
    with pytest.raises(AssertionError, match="tile copies"):
        _assert_tiled_pipeline(ir, 4)


@pytest.mark.parametrize("dtype", ("i64", "u64"))
@pytest.mark.parametrize(
    "op", ("shift_left", "shift_right", "shift_lefts", "shift_rights")
)
def test_b64_example_tail_uses_full_registers(shift_example, dtype, op):
    n = 400
    shift_example._set_kernel_config(op, dtype, (n,))

    def tensor(element):
        return make_fake_tensor(
            element,
            (n,),
            (1,),
            addrspace=tla.AddressSpace.gm,
            origin_shape=(n,),
            layout_tag=tla.arch.RowMajor,
        )

    data = tensor(tla.Int64 if dtype == "i64" else tla.UInt64)
    ir = shift_example.shift_op.dump_mlir(type_args=(data, tensor(tla.Int64), data))
    shift_line = next(line for line in ir.splitlines() if f"tla.{op} " in line)
    element_type = "i64" if dtype == "i64" else "ui64"
    assert f"!tla.vector<32x{element_type}>" in shift_line
    assert "tla.update_mask" in ir


@pytest.mark.parametrize("dtype", (tla.Int64, tla.UInt64))
@pytest.mark.parametrize("n", (11264, 32769))
def test_large_vv64_uses_bounded_ub(shift_example, dtype, n):
    shift_example._set_kernel_config(
        "shift_right", "u64" if dtype is tla.UInt64 else "i64", (n,)
    )

    def tensor(element):
        return make_fake_tensor(
            element,
            (n,),
            (1,),
            addrspace=tla.AddressSpace.gm,
            origin_shape=(n,),
            layout_tag=tla.arch.RowMajor,
        )

    data, counts = tensor(dtype), tensor(tla.Int64)
    ir = shift_example.shift_op.dump_mlir(type_args=(data, counts, data))
    _assert_tiled_pipeline(ir, (n + 1023) // 1024)


@pytest.mark.parametrize("dtype", ("i64", "u64"))
@pytest.mark.parametrize("op", ("shift_left", "shift_right"))
@pytest.mark.parametrize("n", (32, 4096))
def test_vv64_example_covers_count_boundaries(shift_example, monkeypatch, dtype, op, n):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.Tensor, "npu", lambda tensor: tensor, raising=False)
    _, counts = shift_example._make_inputs(
        SimpleNamespace(op=op, shape=(n,)), dtype, torch
    )
    values = counts.tolist()[:32]
    assert {0, 1, 31, 32, 63, 64, 65, 128, 32767} <= set(values)
    assert max(counts.tolist()) <= 32767
