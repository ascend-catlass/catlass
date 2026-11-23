# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import catlass.tla as tla


def _ensure_bisheng_path() -> None:
    """Make the CANN BiSheng compiler visible to hivmc-a5."""
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        return
    bisheng_bin = Path(ascend_home) / "tools" / "bisheng_compiler" / "bin"
    if bisheng_bin.is_dir():
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if str(bisheng_bin) not in path_entries:
            os.environ["PATH"] = (
                str(bisheng_bin) + os.pathsep + os.environ.get("PATH", "")
            )


_ensure_bisheng_path()


@tla.kernel
def chunk_local_cumsum_kernel(mem_g: tla.Tensor, mem_o: tla.Tensor) -> None:
    """Accumulate the 64 lanes of each packed task in fp32 in CATLASS."""
    mtx_ub = tla.mutex("clc_ub", id=0)

    g_ptr = tla.recast_ptr(
        _allocate_bytes(INPUT_TASK_ELEMS * INPUT_BYTES, 256, tla.AddressSpace.ub),
        dtype=INPUT_DTYPE,
    )
    out_ptr = tla.recast_ptr(
        _allocate_bytes(OUTPUT_TASK_ELEMS * F32_BYTES, 256, tla.AddressSpace.ub),
        dtype=tla.Float32,
    )
    acc_ptr = tla.recast_ptr(
        _allocate_bytes(DIM * F32_BYTES, 256, tla.AddressSpace.ub),
        dtype=tla.Float32,
    )

    with tla.vector():
        for task_id in tla.range(tla.arch.block_idx(), TASKS, tla.arch.block_num()):
            input_base = task_id * INPUT_TASK_ELEMS
            output_base = task_id * OUTPUT_TASK_ELEMS
            g_gm = tla.make_tensor(
                mem_g.ptr + input_base, _flat_layout(INPUT_TASK_ELEMS)
            )
            out_gm = tla.make_tensor(
                mem_o.ptr + output_base, _flat_layout(OUTPUT_TASK_ELEMS)
            )
            g_ub = tla.make_tensor_like(g_ptr, g_gm, tla.arch.RowMajor)
            out_ub = tla.make_tensor_like(out_ptr, out_gm, tla.arch.RowMajor)
            acc_gm = tla.make_tensor(mem_o.ptr + output_base, _flat_layout(DIM))
            acc_ub = tla.make_tensor_like(acc_ptr, acc_gm, tla.arch.RowMajor)

            mtx_ub.lock(pipe=tla.arch.MTE2)
            tla.copy(g_ub, g_gm)
            mtx_ub.unlock(pipe=tla.arch.MTE2)
            mtx_ub.lock(pipe=tla.arch.VECTOR)
            # BT=128 uses two vector functions sharing the UB accumulator.
            for row_base in tla.range_constexpr(0, CHUNK, 64):
                with tla.vec.func(mode="simd"):
                    if row_base == 0:
                        acc_ub.store(tla.full(0.0, tla.Float32))
                    for row in tla.range_constexpr(row_base, row_base + 64, 1):
                        g_row = tla.tile_view(
                            g_ub, tla.make_shape(INPUT_DIM), tla.make_coord(row)
                        )
                        out_row = tla.tile_view(
                            out_ub, tla.make_shape(DIM), tla.make_coord(row)
                        )
                        prefix = acc_ub.load() + g_row.load().to(
                            tla.Float32, _CAST_WIDEN
                        )
                        acc_ub.store(prefix)
                        out_row.store(prefix * SCALE_VALUE if HAS_SCALE else prefix)
            mtx_ub.unlock(pipe=tla.arch.VECTOR)
            mtx_ub.lock(pipe=tla.arch.MTE3)
            tla.copy(out_gm, out_ub)
            mtx_ub.unlock(pipe=tla.arch.MTE3)

        mtx_ub.lock(pipe=tla.arch.MTE2)
        tla.pipe_barrier(tla.pipes.ALL)


DIM = 64
INPUT_DIM = 128
INPUT_BYTES = 2
F32_BYTES = 4

TASKS = 1
CHUNK = 64
INPUT_TASK_ELEMS = CHUNK * INPUT_DIM
OUTPUT_TASK_ELEMS = CHUNK * DIM
INPUT_DTYPE = tla.Float16
INPUT_DTYPE_NAME = "float16"
HAS_SCALE = False
SCALE_VALUE = 1.0

_CAST_WIDEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)


def _allocate_bytes(size_bytes: int, alignment: int, mem_scope: Any) -> Any:
    return tla.allocate(size_bytes, tla.UInt8, mem_scope, alignment)


def _flat_layout(elements: int):
    return tla.make_layout(
        tla.make_shape(elements), tla.make_stride(1), layoutTag=tla.arch.RowMajor
    )


def configure_problem(
    num_tasks: int,
    chunk_size: int,
    input_dtype: str,
    scale: float | None,
    dim: int = 64,
) -> None:
    """Configure the one kernel's compile-time specialization."""
    global \
        DIM, \
        INPUT_DIM, \
        TASKS, \
        CHUNK, \
        INPUT_TASK_ELEMS, \
        OUTPUT_TASK_ELEMS, \
        INPUT_DTYPE, \
        INPUT_DTYPE_NAME
    global HAS_SCALE, SCALE_VALUE
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}")
    if chunk_size not in (64, 128):
        raise ValueError(f"chunk_size must be 64 or 128, got {chunk_size}")
    if input_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"unsupported input dtype: {input_dtype}")
    if dim not in (64, 128):
        raise ValueError(f"unsupported D: {dim}")
    DIM = dim
    INPUT_DIM = 2 * DIM
    TASKS = num_tasks
    CHUNK = chunk_size
    INPUT_TASK_ELEMS = chunk_size * INPUT_DIM
    OUTPUT_TASK_ELEMS = chunk_size * DIM
    INPUT_DTYPE_NAME = input_dtype
    INPUT_DTYPE = tla.Float16 if input_dtype == "float16" else tla.BFloat16
    HAS_SCALE = scale is not None
    SCALE_VALUE = 1.0 if scale is None else float(scale)


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache"
DEFAULT_PROFILE_DIR = DEMO_DIR / "prof"
DEFAULT_UUID = "clc-001-fp16"
KERNEL = chunk_local_cumsum_kernel


@dataclass(frozen=True)
class WorkloadSpec:
    uuid: str
    mode: str
    input_dtype: str
    output_dtype: str
    g_shape: tuple[int, ...]
    batch: int
    sequence_length: int
    heads: int
    dim: int
    chunk_size: int
    scale: float | None
    cu_seqlens: tuple[int, ...] | None
    num_chunks: int
    num_tasks: int
    elements: int


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Accept the checked-in JSONL payload despite its requested .json name."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    return (
        json.loads(text)
        if stripped.startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )


def load_case(path: str | Path, uuid: str) -> dict[str, Any]:
    for case in load_cases(path):
        if case["uuid"] == uuid:
            return case
    raise ValueError(f"unknown workload uuid: {uuid}")


def _cu_values(spec: Any, total: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in spec["value"])
    if len(values) < 2 or values[0] != 0 or values[-1] != total:
        raise ValueError(f"cu_seqlens must span [0, {total}], got {values}")
    if any(end <= start for start, end in zip(values, values[1:])):
        raise ValueError(f"cu_seqlens must be strictly increasing: {values}")
    return values


def _parse_workload(case: dict[str, Any]) -> WorkloadSpec:
    values = case["inputs"]
    g_shape = tuple(int(value) for value in values["g"]["shape"])
    input_dtype = str(values["g"]["dtype"])
    output_dtype = str(values["output_dtype"]["value"])
    chunk_size = int(values["chunk_size"]["value"])
    scale_value = values.get("scale")
    scale = None if scale_value is None else float(scale_value["value"])
    if input_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"unsupported input dtype: {input_dtype}")
    if output_dtype not in ("float16", "float32"):
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    if chunk_size not in (64, 128):
        raise ValueError(f"unsupported chunk size: {chunk_size}")
    if bool(values["reverse"]["value"]):
        raise ValueError("reverse cumsum is not among the required 12 workloads")
    if bool(values["head_first"]["value"]):
        raise ValueError("head_first is not among the required 12 workloads")
    if values.get("chunk_indices") is not None:
        raise ValueError("explicit chunk_indices are not supported")

    cu_spec = values.get("cu_seqlens")
    mode = "bsnd" if cu_spec is None else "tnd"
    if mode == "bsnd":
        if len(g_shape) != 4:
            raise ValueError(f"BSND g must have rank 4, got {g_shape}")
        batch, sequence_length, heads, dim = g_shape
        cu_seqlens = None
        if sequence_length % chunk_size:
            raise ValueError("BSND sequence length must be chunk aligned")
        num_chunks = batch * sequence_length // chunk_size
    else:
        if len(g_shape) != 3:
            raise ValueError(f"TND g must have rank 3, got {g_shape}")
        sequence_length, heads, dim = g_shape
        batch = 1
        cu_seqlens = _cu_values(cu_spec, sequence_length)
        lengths = tuple(end - start for start, end in zip(cu_seqlens, cu_seqlens[1:]))
        if any(length % chunk_size for length in lengths):
            raise ValueError("required TND sequence boundaries must be chunk aligned")
        num_chunks = sum(length // chunk_size for length in lengths)
    if dim not in (64, 128):
        raise ValueError(f"D must be 64 or 128, got {dim}")
    elements = 1
    for size in g_shape:
        elements *= size
    if elements != num_chunks * heads * chunk_size * dim:
        raise ValueError("packed task stream does not cover input")
    return WorkloadSpec(
        uuid=str(case["uuid"]),
        mode=mode,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        g_shape=g_shape,
        batch=batch,
        sequence_length=sequence_length,
        heads=heads,
        dim=dim,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        num_chunks=num_chunks,
        num_tasks=num_chunks * heads,
        elements=elements,
    )


def _torch_dtype(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _tla_dtype(name: str) -> Any:
    return tla.Float16 if name == "float16" else tla.BFloat16


def _meta_tensor(elements: int, dtype: Any) -> Any:
    return tla.make_fake_tensor(dtype, (elements,), (1,), layout_tag=tla.arch.RowMajor)


def _type_args(spec: WorkloadSpec) -> tuple[Any, Any]:
    configure_problem(
        spec.num_tasks, spec.chunk_size, spec.input_dtype, spec.scale, spec.dim
    )
    return (
        _meta_tensor(
            spec.num_tasks * spec.chunk_size * INPUT_DIM, _tla_dtype(spec.input_dtype)
        ),
        _meta_tensor(spec.elements, tla.Float32),
    )


def _runtime_kwargs(
    cache_dir: str | Path, force_recompile: bool, cache: bool = True
) -> dict[str, Any]:
    os.environ["CATLASS_DSL_CACHE"] = "1" if cache else "0"
    os.environ.pop("CATLASS_DSL_CACHE_DIR", None)
    os.environ["CATLASS_DSL_FORCE_RECOMPILE"] = "1" if force_recompile else "0"
    return {
        "arch_scope": "aiv.c310",
        "cache": cache,
        "cache_dir": str(Path(cache_dir).expanduser().resolve()),
        "force_recompile": force_recompile,
    }


def _case_cache_dir(base: str | Path, spec: WorkloadSpec) -> Path:
    scale = "noscale" if spec.scale is None else f"scale{spec.scale:g}"
    return (
        Path(base)
        / f"{spec.uuid}-{spec.mode}-{spec.input_dtype}-bt{spec.chunk_size}-{scale}-tasks{spec.num_tasks}"
    )


def _require_torch_npu(device: int) -> Any:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    return torch


def make_inputs(torch: Any, spec: WorkloadSpec, seed: int) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (
        torch.randn(spec.g_shape, generator=generator, dtype=torch.float32)
        .to(_torch_dtype(torch, spec.input_dtype))
        .to("npu")
    )


def _pack_tasks(torch: Any, spec: WorkloadSpec, g: Any) -> Any:
    tasks: list[Any] = []
    if spec.mode == "bsnd":
        for batch in range(spec.batch):
            for start in range(0, spec.sequence_length, spec.chunk_size):
                for head in range(spec.heads):
                    tasks.append(g[batch, start : start + spec.chunk_size, head, :])
    else:
        if spec.cu_seqlens is None:
            raise ValueError("TND mode requires cu_seqlens")
        for sequence_start, sequence_end in zip(spec.cu_seqlens, spec.cu_seqlens[1:]):
            for start in range(sequence_start, sequence_end, spec.chunk_size):
                for head in range(spec.heads):
                    tasks.append(g[start : start + spec.chunk_size, head, :])
    dense = torch.stack(tasks, dim=0).contiguous()
    # CATLASS f16/bf16-to-f32 vector conversion is 128 input lanes to 64
    # fp32 lanes.  Keep values in the even lanes, matching the established
    # KDA vector-kernel packing convention.
    packed = torch.zeros(
        dense.shape[:-1] + (2 * spec.dim,), dtype=dense.dtype, device=dense.device
    )
    packed[..., 0::2] = dense
    return packed.contiguous()


def _unpack_tasks(torch: Any, spec: WorkloadSpec, packed: Any) -> Any:
    out = torch.empty(spec.g_shape, dtype=packed.dtype, device=packed.device)
    task = 0
    if spec.mode == "bsnd":
        for batch in range(spec.batch):
            for start in range(0, spec.sequence_length, spec.chunk_size):
                for head in range(spec.heads):
                    out[batch, start : start + spec.chunk_size, head, :] = packed[task]
                    task += 1
    else:
        if spec.cu_seqlens is None:
            raise ValueError("TND mode requires cu_seqlens")
        for sequence_start, sequence_end in zip(spec.cu_seqlens, spec.cu_seqlens[1:]):
            for start in range(sequence_start, sequence_end, spec.chunk_size):
                for head in range(spec.heads):
                    out[start : start + spec.chunk_size, head, :] = packed[task]
                    task += 1
    return out


def torch_reference(torch: Any, spec: WorkloadSpec, g: Any) -> Any:
    """Oracle only: the candidate never calls this function or Torch cumsum."""
    out = torch.empty(spec.g_shape, dtype=torch.float32, device=g.device)
    if spec.mode == "bsnd":
        out = (
            g.float()
            .reshape(
                spec.batch,
                spec.sequence_length // spec.chunk_size,
                spec.chunk_size,
                spec.heads,
                spec.dim,
            )
            .cumsum(2)
            .reshape(spec.g_shape)
        )
    else:
        if spec.cu_seqlens is None:
            raise ValueError("TND mode requires cu_seqlens")
        for sequence_start, sequence_end in zip(spec.cu_seqlens, spec.cu_seqlens[1:]):
            out[sequence_start:sequence_end] = (
                g[sequence_start:sequence_end]
                .float()
                .reshape(
                    (sequence_end - sequence_start) // spec.chunk_size,
                    spec.chunk_size,
                    spec.heads,
                    spec.dim,
                )
                .cumsum(1)
                .reshape(sequence_end - sequence_start, spec.heads, spec.dim)
            )
    return (out if spec.scale is None else out * spec.scale).to(
        _torch_dtype(torch, spec.output_dtype)
    )


def _to_tla(dev_tensor: Any) -> Any:
    return tla.from_dlpack(
        dev_tensor.reshape(-1).contiguous(), layout_tag=tla.arch.RowMajor
    )


def chunk_local_cumsum(
    g: Any,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: Any = None,
    head_first: bool = False,
    output_dtype: Any = None,
    chunk_indices: Any = None,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_recompile: bool = False,
    cache: bool = True,
    block: int = 160,
) -> Any:
    """Run a supported input through the CATLASS cumsum kernel."""
    import torch

    if reverse or head_first or chunk_indices is not None:
        raise ValueError(
            "reverse, head_first and explicit chunk_indices are unsupported"
        )
    if g.shape[-1] == 128:
        return torch.cat(
            [
                chunk_local_cumsum(
                    g[..., start : start + 64],
                    chunk_size,
                    reverse,
                    scale,
                    cu_seqlens,
                    head_first,
                    output_dtype,
                    chunk_indices,
                    cache_dir=cache_dir,
                    force_recompile=force_recompile,
                    cache=cache,
                    block=block,
                )
                for start in (0, 64)
            ],
            dim=-1,
        )
    dtype = "float16" if g.dtype == torch.float16 else "bfloat16"
    output = "float16" if output_dtype in (torch.float16, "float16") else "float32"
    if cu_seqlens is None:
        batch, length, heads, dim = tuple(g.shape)
        mode = "bsnd"
        cu_values = None
        num_chunks = batch * length // chunk_size
    else:
        length, heads, dim = tuple(g.shape)
        batch = 1
        mode = "tnd"
        cu_values = tuple(int(x) for x in cu_seqlens.detach().cpu().tolist())
        num_chunks = sum(
            (end - start) // chunk_size for start, end in zip(cu_values, cu_values[1:])
        )
    spec = WorkloadSpec(
        f"runtime-{mode}",
        mode,
        dtype,
        output,
        tuple(g.shape),
        batch,
        length,
        heads,
        dim,
        chunk_size,
        scale,
        cu_values,
        num_chunks,
        num_chunks * heads,
        g.numel(),
    )
    packed_g = _pack_tasks(torch, spec, g)
    packed_out = torch.empty(
        (spec.num_tasks, spec.chunk_size, spec.dim),
        dtype=torch.float32,
        device=g.device,
    )
    configure_problem(
        spec.num_tasks, spec.chunk_size, spec.input_dtype, spec.scale, spec.dim
    )
    artifact = tla.compile(
        KERNEL,
        *_type_args(spec),
        **_runtime_kwargs(_case_cache_dir(cache_dir, spec), force_recompile, cache),
    )
    artifact(_to_tla(packed_g), _to_tla(packed_out), block_num=block)
    torch.npu.synchronize()
    return _unpack_tasks(torch, spec, packed_out).to(_torch_dtype(torch, output))


def _compare(torch: Any, spec: WorkloadSpec, actual: Any, expected: Any) -> bool:
    diff = (actual.float() - expected.float()).abs()
    atol = rtol = 5e-2 if spec.input_dtype == "bfloat16" else 2e-2
    passed = bool(torch.all(diff <= atol + rtol * expected.float().abs()).item())
    print(
        f"accuracy uuid={spec.uuid} pass={passed} max_abs={diff.max().item():.6g} atol={atol} rtol={rtol}"
    )
    return passed


def _compare_declared(
    torch: Any,
    actual: Any,
    expected: Any,
    tolerance: dict[str, Any],
) -> bool:
    actual_f = actual.float()
    expected_f = expected.float()
    finite = torch.isfinite(actual_f) & torch.isfinite(expected_f)
    diff = (actual_f - expected_f).abs()
    close = finite & (
        diff
        <= float(tolerance["max_atol"])
        + float(tolerance["max_rtol"]) * expected_f.abs()
    )
    matched_ratio = float(close.sum()) / close.numel()
    cap_ok = bool((diff[finite] <= float(tolerance["max_error_cap"])).all())
    passed = (
        bool(finite.all())
        and cap_ok
        and matched_ratio >= float(tolerance["required_matched_ratio"])
    )
    print(
        f"accuracy pass={passed} max_abs={float(diff.max())} "
        f"matched_ratio={matched_ratio:.9f} cap_ok={cap_ok} "
        f"required_matched_ratio={tolerance['required_matched_ratio']}"
    )
    return passed


def run_validation(
    args: argparse.Namespace, uuid: str, manage_runtime: bool = True
) -> int:
    case = getattr(args, "direct_case", None) or load_case(args.workload, uuid)
    spec = _parse_workload(case)
    try:
        torch = _require_torch_npu(args.device)
        g = make_inputs(torch, spec, args.seed)
        expected = torch_reference(torch, spec, g)
        actual = chunk_local_cumsum(
            g,
            spec.chunk_size,
            scale=spec.scale,
            cu_seqlens=None
            if spec.cu_seqlens is None
            else torch.tensor(spec.cu_seqlens, device=g.device),
            output_dtype=_torch_dtype(torch, spec.output_dtype),
            cache_dir=args.cache_dir,
            force_recompile=args.force_recompile,
            cache=not args.no_cache,
            block=args.block,
        )
        actual_cpu = actual.cpu()
        expected_cpu = expected.cpu()
        tolerance = case.get("tolerance")
        output_ok = (
            _compare_declared(torch, actual_cpu, expected_cpu, tolerance)
            if tolerance is not None
            else _compare(torch, spec, actual_cpu, expected_cpu)
        )
        passed = bool(torch.isfinite(actual.float()).all()) and output_ok
        print(f"validation uuid={uuid} pass={passed}")
        return 0 if passed else 1

    finally:
        pass


def _export_case_reports(
    profile_dir: Path, cases: list[dict[str, Any]], report_dir: Path
) -> None:
    """Split the single msprof trace by ordered kernel invocations.

    ``--run-all-workloads`` launches exactly one candidate kernel for each
    workload in JSON order.  Thus the matching rows in op_summary can be
    assigned to cases by their stable call order without needing 12 PROF runs.
    """
    profs = sorted(profile_dir.rglob("PROF_*"), key=lambda path: path.stat().st_mtime)
    if not profs:
        raise RuntimeError(f"no PROF_* directory found under {profile_dir}")
    prof_dir = profs[-1]
    summaries = sorted(
        (prof_dir / "mindstudio_profiler_output").glob("op_summary_*.csv")
    )
    if not summaries:
        raise RuntimeError(f"no op_summary CSV found in {prof_dir}")
    summary_path = summaries[-1]
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("Op Name") == "chunk_local_cumsum_kernel"
        ]
    if len(rows) < len(cases):
        raise RuntimeError(
            f"expected {len(cases)} chunk_local_cumsum_kernel rows in "
            f"{summary_path}, found {len(rows)}"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, str]] = []
    for index, (case, row) in enumerate(zip(cases, rows), start=1):
        uuid = str(case["uuid"])
        csv_path = report_dir / f"{uuid}.csv"
        json_path = report_dir / f"{uuid}.json"
        enriched = {
            "case_index": str(index),
            "uuid": uuid,
            "kernel": "chunk_local_cumsum_kernel",
            "prof_dir": str(prof_dir),
            **row,
        }
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(enriched))
            writer.writeheader()
            writer.writerow(enriched)
        json_path.write_text(
            json.dumps(
                {
                    "case_index": index,
                    "uuid": uuid,
                    "kernel": "chunk_local_cumsum_kernel",
                    "prof_dir": str(prof_dir),
                    "op_summary": row,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        reports.append({"uuid": uuid, "csv": str(csv_path), "json": str(json_path)})
    (report_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prof_dir": str(prof_dir),
                "op_summary": str(summary_path),
                "reports": reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _print_case_performance_table(cases, rows)
    print(f"case_reports written={len(reports)} dir={report_dir}")


def _print_case_performance_table(
    cases: list[dict[str, Any]], rows: list[dict[str, str]]
) -> None:
    """Print the same one-row-per-case data persisted in the report files."""
    headers = [
        "#",
        "uuid",
        "mode",
        "dtype",
        "shape",
        "BT",
        "tasks",
        "duration(us)",
        "aicore(us)",
        "aiv(us)",
        "Block",
        "MixBlock",
    ]
    metric_names = [
        "Task Duration(us)",
        "aicore_time(us)",
        "aiv_time(us)",
        "Block Num",
        "Mix Block Num",
    ]
    formatted: list[dict[str, str]] = []
    for index, (case, row) in enumerate(zip(cases, rows), start=1):
        spec = _parse_workload(case)
        values = [
            str(index),
            spec.uuid,
            spec.mode,
            spec.input_dtype,
            "x".join(str(size) for size in spec.g_shape),
            str(spec.chunk_size),
            str(spec.num_tasks),
            *(str(row.get(name) or "N/A") for name in metric_names),
        ]
        formatted.append(dict(zip(headers, values)))
    widths = {
        header: max(len(header), *(len(row[header]) for row in formatted))
        for header in headers
    }
    print("\nmsprof chunk_local_cumsum case performance")
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in formatted:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def profile_workloads(args: argparse.Namespace) -> int:
    """Run all 12 cases under one msprof process and emit per-case reports."""
    if args.workload is None:
        raise ValueError("--profile-all-workloads requires an explicit --workload file")
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    workload = Path(args.workload).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cases = load_cases(workload)
    command = [
        "msprof",
        f"--output={profile_dir}",
        sys.executable,
        str(Path(__file__).resolve()),
        "--run",
        "--run-all-workloads",
        "--workload",
        str(workload),
        "--cache-dir",
        str(cache_dir),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--block",
        str(args.block),
    ]
    if args.force_recompile:
        command.append("--force-recompile")
    if args.no_cache:
        command.append("--no-cache")
    env = os.environ.copy()
    tla_root = str(Path(tla.__file__).resolve().parents[2])
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{tla_root}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else tla_root
    )
    print(f"profile_start cases={len(cases)} output={profile_dir}")
    result = subprocess.run(command, cwd=profile_dir, env=env, check=False)
    if result.returncode:
        print(f"profile_failed returncode={result.returncode}")
        return result.returncode
    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else profile_dir / "reports"
    )
    _export_case_reports(profile_dir, cases, report_dir)
    print(f"profile_summary passed={len(cases)}/{len(cases)} prof_dir={profile_dir}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", nargs="?", type=int, help="BSND batch size")
    parser.add_argument(
        "sequence_length", nargs="?", type=int, help="sequence length T"
    )
    parser.add_argument("heads", nargs="?", type=int, help="number of heads H")
    parser.add_argument(
        "dim", nargs="?", type=int, help="feature dimension D, 64 or 128"
    )
    parser.add_argument(
        "device_pos", nargs="?", type=int, default=0, help="NPU device ID (default: 0)"
    )
    parser.add_argument("--workload", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--case-uuid", "--uuid", default=DEFAULT_UUID, help=argparse.SUPPRESS
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-all-workloads", action="store_true")
    parser.add_argument("--profile-all-workloads", action="store_true")
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--block", type=int, default=160)
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR))
    parser.add_argument("--report-dir")
    parser.add_argument("--mode", choices=("bsnd", "tnd"), default="bsnd")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--output-dtype", choices=("float16", "float32"), default="float32"
    )
    parser.add_argument("--scale", type=float)
    parser.add_argument("--cu-seqlens", help="TND cumulative lengths, comma-separated")
    return parser


def _direct_case(args: argparse.Namespace) -> dict[str, Any] | None:
    values = (args.batch, args.sequence_length, args.heads, args.dim)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("direct mode requires: B T H D [device]")
    if args.mode == "bsnd":
        shape = [args.batch, args.sequence_length, args.heads, args.dim]
        cu_seqlens = None
    else:
        shape = [args.sequence_length, args.heads, args.dim]
        if not args.cu_seqlens:
            raise ValueError("TND mode requires --cu-seqlens, for example 0,16384")
        cu_seqlens = [int(value) for value in args.cu_seqlens.split(",")]
    case = {
        "uuid": f"direct-{args.mode}-{args.dtype}-bt{args.chunk_size}",
        "inputs": {
            "g": {"shape": shape, "dtype": args.dtype},
            "chunk_size": {"value": args.chunk_size},
            "reverse": {"value": False},
            "scale": None if args.scale is None else {"value": args.scale},
            "cu_seqlens": None if cu_seqlens is None else {"value": cu_seqlens},
            "head_first": {"value": False},
            "output_dtype": {"value": args.output_dtype},
            "chunk_indices": None,
        },
    }
    args.device = args.device_pos
    args.direct_case = case
    return case


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    direct_case = _direct_case(args)
    if args.profile_all_workloads:
        if direct_case is not None:
            raise ValueError(
                "--profile-all-workloads cannot be combined with direct parameters"
            )
        return profile_workloads(args)
    if direct_case is not None:
        if args.run_all_workloads:
            raise ValueError(
                "direct mode accepts one workload; remove --run-all-workloads"
            )
        return run_validation(args, direct_case["uuid"], manage_runtime=False)
    if args.workload is None:
        raise ValueError(
            "provide B T H D [device], or explicitly pass --workload for legacy mode"
        )
    uuids = (
        [case["uuid"] for case in load_cases(args.workload)]
        if args.run_all_workloads
        else [args.case_uuid]
    )
    results = [run_validation(args, uuid, manage_runtime=False) for uuid in uuids]
    passed = sum(result == 0 for result in results)
    print(f"all_workloads passed={passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
