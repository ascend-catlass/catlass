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

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse
import re

import catlass.tla as tla

M_DIM = 32
N_DIM = 32
K_DIM = 32
VECTOR_TILE_M = 16
VECTOR_TILE_N = 32
VECTOR_REG_TILE_M = 2
UB_TILE_BYTES = VECTOR_TILE_M * VECTOR_TILE_N * 4
L1_STAGE_BYTES = 256 * 1024
L0A_BYTES = 32 * 32 * 4
L0B_BYTES = 32 * 32 * 4
L0C_BYTES = 32 * 32 * 4

# Minimal dynamic-GM smoke for mix kernel: M/N stay 32 (AIV tile split), vary K.
DESCRIPTION = "Basic Mixed Cube+Vector add."
_PRINT_RECORD = re.compile(
    r"^tla\.print dtype=float32 position=(?P<position>[A-Z0-9]+) "
    r"subblock=(?P<subblock>[01]) "
    r"shape=\[(?P<shape>[0-9,]+)\] count=(?P<count>[0-9]+) "
    r"values=\[(?P<values>.*)\]$"
)

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def basic_mixed(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
    addend: tla.Tensor,
) -> None:
    m = lhs.origin_shape[0]
    k = lhs.origin_shape[1]
    n = rhs.origin_shape[1]

    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l1_loaded = tla.flag("l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    ub_load_ready = tla.flag("ub_load_ready", tla.arch.VECTOR, tla.arch.MTE2)
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    fix_done = tla.cross_flag("fix_done")

    l1a_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_BYTES // 4, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_BYTES // 4, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_BYTES // 4, tla.Float32, tla.AddressSpace.l0c, 1024)

    c_ub_ptr = tla.allocate(UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256)
    addend_ub_ptr = tla.allocate(
        UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256
    )
    result_ub_ptr = tla.allocate(
        UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256
    )

    with tla.cube():
        gm_a = tla.tile_view(lhs, tla.make_shape(m, k), tla.make_coord(0, 0))
        gm_b = tla.tile_view(rhs, tla.make_shape(k, n), tla.make_coord(0, 0))
        gm_c = tla.tile_view(out, tla.make_shape(m, n), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)
        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)

        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        l1_a_l0 = tla.tile_view(l1_a, tla.make_shape(m, k), tla.make_coord(0, 0))
        l1_b_l0 = tla.tile_view(l1_b, tla.make_shape(k, n), tla.make_coord(0, 0))
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.copy(l0_a, l1_a_l0)
        tla.copy(l0_b, l1_b_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)

        ub_c = tla.make_tensor_like(c_ub_ptr, l0_c, tla.arch.RowMajor)
        tla.copy(
            ub_c,
            l0_c,
            tla.params.CopyL0C2DstParams(
                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
            ),
        )

        tla.cross_core_set_flag(fix_done, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()

        gm_result = tla.tile_view(
            out,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(vec_idx, 0),
        )
        gm_addend = tla.tile_view(
            addend,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(vec_idx, 0),
        )
        ub_result = tla.make_tensor_like(result_ub_ptr, gm_result, tla.arch.RowMajor)
        ub_addend = tla.make_tensor_like(addend_ub_ptr, gm_addend, tla.arch.RowMajor)

        tla.set_flag(ub_load_ready)
        tla.wait_flag(ub_load_ready)
        tla.copy(ub_addend, gm_addend)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        ub_c = tla.make_tensor_like(c_ub_ptr, gm_result, tla.arch.RowMajor)
        tla.cross_core_wait_flag(fix_done, tla.arch.VECTOR)

        for row_tile_idx in tla.range(0, VECTOR_TILE_M // VECTOR_REG_TILE_M, 1):
            with tla.vec.func(mode="simd"):
                c_chunk = tla.tile_view(
                    ub_c,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0),
                )
                addend_chunk = tla.tile_view(
                    ub_addend,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0),
                )
                result_chunk = tla.tile_view(
                    ub_result,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0),
                )
                result_chunk.store(c_chunk.load() + addend_chunk.load())

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_result, ub_result)

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def _verify_mixed_print_output(output: str) -> list[str]:
    records = [line for line in output.splitlines() if line.startswith("tla.print ")]
    if len(records) != 2:
        raise tla.TlaExecutionError(
            "mixed tensor tla.print validation failed: "
            f"expected two AIV records, got {records!r}"
        )
    records_by_subblock: dict[int, str] = {}
    for record in records:
        match = _PRINT_RECORD.fullmatch(record)
        if match is None:
            raise tla.TlaExecutionError(
                f"mixed tensor tla.print validation failed: malformed record {record!r}"
            )
        shape = tuple(int(extent) for extent in match.group("shape").split(","))
        try:
            values = [
                float(value.strip()) for value in match.group("values").split(",")
            ]
        except ValueError as exc:
            raise tla.TlaExecutionError(
                f"mixed tensor tla.print validation failed: malformed record {record!r}"
            ) from exc
        subblock = int(match.group("subblock"))
        if subblock in records_by_subblock:
            raise tla.TlaExecutionError(
                "mixed tensor tla.print validation failed: "
                f"duplicate subblock={subblock}"
            )
        records_by_subblock[subblock] = record
        if (
            match.group("position") != "UB"
            or shape != (VECTOR_TILE_M, VECTOR_TILE_N)
            or int(match.group("count")) != 16
            or values != [3.0] * 16
        ):
            raise tla.TlaExecutionError(
                "mixed tensor tla.print validation failed: expected position=UB, "
                f"shape=[{VECTOR_TILE_M},{VECTOR_TILE_N}], count=16, and sixteen "
                f"3.0 values; got {record!r}"
            )
    if set(records_by_subblock) != {0, 1}:
        raise tla.TlaExecutionError(
            "mixed tensor tla.print validation failed: expected records from "
            f"subblocks 0 and 1, got {sorted(records_by_subblock)}"
        )
    return records


def golden(lhs, rhs, addend):
    import torch

    return lhs.to(torch.float32) @ rhs.to(torch.float32) + addend.to(torch.float32)


def prepare_npu(buf, layout: str):
    storage = buf.contiguous() if layout == "row" else buf.permute(1, 0).contiguous()
    return storage.npu()


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu

    from helpers import (
        get_block_num,
        create_tla_tensor,
        compare,
    )

    mi, ni, ki = int(args.m), int(args.n), int(args.k)

    torch.npu.set_device(args.device)
    print(f"--- mnk=({mi},{ni},{ki}) ---")
    lhs = torch.rand(mi, ki, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    rhs = torch.rand(ki, ni, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    addend = torch.full((mi, ni), 3.0, dtype=torch.float32, device="cpu")
    out = torch.full((mi, ni), args.sentinel, dtype=torch.float32, device="cpu")
    ref = golden(lhs, rhs, addend)

    lhs = prepare_npu(lhs, args.layout_a)
    rhs = prepare_npu(rhs, args.layout_b)
    out = prepare_npu(out, "row")
    addend = prepare_npu(addend, "row")
    a_tensor = create_tla_tensor(lhs, args.layout_a)
    b_tensor = create_tla_tensor(rhs, args.layout_b)
    c_tensor = create_tla_tensor(out, "row")
    d_tensor = create_tla_tensor(addend, "row")

    artifact = tla.compile(
        basic_mixed, a_tensor, b_tensor, c_tensor, d_tensor, options="--npu-arch 3510"
    )
    block_num = get_block_num(args.block_num, args.device, kind="mix")
    artifact(a_tensor, b_tensor, c_tensor, d_tensor, block_num=block_num)
    torch.npu.synchronize()

    passed = compare(out.detach().cpu(), ref, ki)
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--layout-a", choices=("row", "col"), default="row")
    parser.add_argument("--layout-b", choices=("row", "col"), default="row")
    parser.add_argument("--block-num", type=int, default=-1)
    parser.add_argument("--sentinel", type=float, default=-9.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
