# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E accuracy: GM→L1 bool / hif8 (ND2NZ RowMajor→zN + VectorLayout).

Compares L1 debug-print C0 words against a host reference:
- VectorLayout: payload bytes are identity (flat 32B C0).
- ND2NZ: ``host_nd2nz`` of the RowMajor source (1-byte elements, C0=32).
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DSL_EXAMPLE_PATH = str(_REPO_ROOT / "examples" / "python" / "tla_dsl")
if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

from catlass import tla

_C0_BYTES = 32
_C0_U32 = 8
# Two C0 column blocks of 1-byte elements: 16 rows × 64 cols = 1024 bytes = 32 C0s.
# 16×32 would be a single column block, so RowMajor and ND2NZ byte streams match
# and a missing layout transform would still pass.
_ND_ROWS = 16
_ND_COLS = 64
_ND_C0_COUNT = (_ND_ROWS * _ND_COLS) // _C0_BYTES
_ND_C0_PER_COL_BLOCK = (_ND_ROWS * _C0_BYTES) // _C0_BYTES  # 16 C0s per 32-col block

CASES = (
    "vector-bool",
    "vector-hif8",
    "nd2nz-bool",
    "nd2nz-hif8",
    "nd2nz-i8",  # control: already-registered 1-byte ND2NZ
)


@tla.kernel
def vector_gm_to_l1(
    gm_src: tla.Tensor,
    dtype: tla.Constexpr,
    n_elems: tla.Constexpr,
) -> None:
    # Byte-addressed L1 storage (same pattern as fill_op); logical dtype via recast.
    l1_bytes = tla.allocate(n_elems, tla.Int8, tla.AddressSpace.l1, 512)
    l1_ptr = tla.recast_ptr(l1_bytes, dtype=dtype)
    copied = tla.flag("vl_copied", tla.arch.MTE2, tla.arch.MTE1)
    with tla.cube():
        l1 = tla.make_tensor_like(l1_ptr, gm_src, tla.arch.VectorLayout)
        tla.copy(l1, gm_src)
        tla.set_flag(copied)
        tla.wait_flag(copied)
        print_ptr = tla.recast_ptr(l1_bytes, dtype=tla.UInt32)
        printed = tla.make_tensor(
            print_ptr,
            tla.make_layout_with_tag((1, _C0_U32), tla.UInt32, tla.arch.zN),
        )
        tla.print(printed, _C0_U32)


@tla.kernel
def nd2nz_gm_to_l1(
    gm_src: tla.Tensor,
    dtype: tla.Constexpr,
    rows: tla.Constexpr,
    cols: tla.Constexpr,
) -> None:
    n_elems = rows * cols
    l1_bytes = tla.allocate(n_elems, tla.Int8, tla.AddressSpace.l1, 512)
    l1_ptr = tla.recast_ptr(l1_bytes, dtype=dtype)
    copied = tla.flag("nd_copied", tla.arch.MTE2, tla.arch.MTE1)
    with tla.cube():
        l1 = tla.make_tensor(
            l1_ptr, tla.make_layout_with_tag((rows, cols), dtype, tla.arch.zN)
        )
        tla.copy(l1, gm_src)
        tla.set_flag(copied)
        tla.wait_flag(copied)
        for c0_idx in tla.range_constexpr(_ND_C0_COUNT):
            print_ptr = tla.recast_ptr(l1_bytes + c0_idx * _C0_BYTES, dtype=tla.UInt32)
            printed = tla.make_tensor(
                print_ptr,
                tla.make_layout_with_tag((1, _C0_U32), tla.UInt32, tla.arch.zN),
            )
            tla.print(printed, _C0_U32)


def _parse_print_words(output: str, expect_records: int) -> list[int]:
    records: list[str] = []
    for line in output.splitlines():
        if line.startswith("tla.print "):
            records.append(line.strip())
    if len(records) != expect_records:
        raise AssertionError(
            f"expected {expect_records} L1 print records, got {len(records)}: {records}"
        )
    words: list[int] = []
    for rec in records:
        match = re.fullmatch(
            r"tla\.print (?:call=\d+ block=\d+ )?"
            r"dtype=u32 (?:position=\S+ )?"
            r"shape=\[1,8\] count=8 values=\[([0-9, ]+)\]",
            rec,
        )
        if match is None:
            raise AssertionError(f"malformed L1 record: {rec}")
        chunk = [int(v.strip()) for v in match[1].split(",")]
        if len(chunk) != _C0_U32:
            raise AssertionError(f"incomplete C0: {chunk}")
        words.extend(chunk)
    return words


def _words_from_bytes(payload: bytes) -> list[int]:
    if len(payload) % _C0_BYTES != 0:
        raise ValueError(f"payload length {len(payload)} not multiple of {_C0_BYTES}")
    return list(struct.unpack(f"<{len(payload) // 4}I", payload))


def _vector_payload(kind: str) -> bytes:
    if kind == "bool":
        # AscendC bool: 0 / nonzero; use 0x00 / 0x01 pattern across one C0.
        return bytes((i % 2) for i in range(_C0_BYTES))
    if kind == "hif8":
        return bytes((0x10 + i) & 0xFF for i in range(_C0_BYTES))
    raise ValueError(kind)


def _nd_payload(kind: str) -> bytes:
    if kind == "bool":
        return bytes(
            ((r * _ND_COLS + c) % 3 != 0)
            for r in range(_ND_ROWS)
            for c in range(_ND_COLS)
        )
    if kind == "hif8":
        return bytes(
            ((r * 17 + c * 3) & 0xFF) for r in range(_ND_ROWS) for c in range(_ND_COLS)
        )
    if kind == "i8":
        return bytes(
            ((r * 13 + c) & 0xFF) for r in range(_ND_ROWS) for c in range(_ND_COLS)
        )
    raise ValueError(kind)


def _bind_vector(payload: bytes, kind: str, device: int):
    import torch
    import torch_npu
    from helpers import create_tla_tensor

    del torch_npu
    if kind == "bool":
        buf = torch.tensor(list(payload), dtype=torch.bool).to(f"npu:{device}")
        return create_tla_tensor(buf, "vector"), tla.Bool, _C0_BYTES
    if kind == "hif8":
        buf = (
            torch.frombuffer(bytearray(payload), dtype=torch.int8)
            .clone()
            .to(f"npu:{device}")
        )
        return (
            create_tla_tensor(buf, "vector", element_type=tla.HiFloat8),
            tla.HiFloat8,
            _C0_BYTES,
        )
    raise ValueError(kind)


def _bind_nd_rowmajor(payload: bytes, kind: str, device: int):
    import torch
    import torch_npu
    from helpers import create_tla_tensor, host_nd2nz

    del torch_npu
    flat = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    if kind == "bool":
        nd = flat.view(_ND_ROWS, _ND_COLS).to(torch.bool)
        gm = create_tla_tensor(nd.to(f"npu:{device}"), "row")
        dtype = tla.Bool
        # host_nd2nz needs a numeric tensor with element_size(); use uint8 0/1.
        ref_src = flat.view(_ND_ROWS, _ND_COLS).to(torch.uint8)
    elif kind == "hif8":
        nd = flat.view(torch.int8).view(_ND_ROWS, _ND_COLS)
        gm = create_tla_tensor(nd.to(f"npu:{device}"), "row", element_type=tla.HiFloat8)
        dtype = tla.HiFloat8
        ref_src = flat.view(_ND_ROWS, _ND_COLS)
    else:
        nd = flat.view(torch.int8).view(_ND_ROWS, _ND_COLS)
        gm = create_tla_tensor(nd.to(f"npu:{device}"), "row")
        dtype = tla.Int8
        ref_src = flat.view(_ND_ROWS, _ND_COLS)
    expected = host_nd2nz(ref_src, "zN").cpu().view(torch.uint8).numpy().tobytes()
    return gm, dtype, expected


def run_case(case: str, device: int) -> None:
    from catlass.base_dsl.runtime.ascend_debug_fifo import _debug_bus_support_error

    reason = _debug_bus_support_error()
    if reason:
        raise SystemExit(f"SKIP: L1 debug readback unavailable: {reason}")

    import torch
    import torch_npu

    del torch_npu
    torch.npu.set_device(device)

    if case.startswith("vector-"):
        kind = case.split("-", 1)[1]
        payload = _vector_payload(kind)
        gm, dtype, n_elems = _bind_vector(payload, kind, device)
        expected = _words_from_bytes(payload)
        artifact = tla.compile(
            vector_gm_to_l1, gm, dtype, n_elems, options="--npu-arch 3510"
        )
        captured = StringIO()
        with redirect_stdout(captured):
            artifact(gm, block_num=1)
            torch.npu.synchronize()
        words = _parse_print_words(captured.getvalue(), 1)
        if words != expected:
            raise AssertionError(
                f"{case}: L1 {words} != expected {expected} payload={payload.hex()}"
            )
        print(f"PASS {case} words={[hex(w) for w in words]}")
        return

    kind = case.split("-", 1)[1]
    payload = _nd_payload(kind)
    gm, dtype, expected_bytes = _bind_nd_rowmajor(payload, kind, device)
    expected = _words_from_bytes(expected_bytes)
    artifact = tla.compile(
        nd2nz_gm_to_l1,
        gm,
        dtype,
        _ND_ROWS,
        _ND_COLS,
        options="--npu-arch 3510",
    )
    captured = StringIO()
    with redirect_stdout(captured):
        artifact(gm, block_num=1)
        torch.npu.synchronize()
    words = _parse_print_words(captured.getvalue(), _ND_C0_COUNT)
    if words != expected:
        got_b = struct.pack(f"<{len(words)}I", *words)
        raise AssertionError(
            f"{case}: L1 mismatch\n  got={got_b.hex()}\n  exp={expected_bytes.hex()}"
        )
    # Two column blocks must differ from a pure RowMajor flatten of the source.
    rowmajor_words = _words_from_bytes(payload)
    if words == rowmajor_words:
        raise AssertionError(
            f"{case}: L1 matches RowMajor flatten; ND2NZ column-block reorder missing"
        )
    block0 = words[: _ND_C0_PER_COL_BLOCK * _C0_U32]
    block1 = words[_ND_C0_PER_COL_BLOCK * _C0_U32 :]
    if block0 == block1:
        raise AssertionError(f"{case}: column blocks unexpectedly identical")
    print(f"PASS {case} c0s={_ND_C0_COUNT} bytes={len(expected_bytes)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    args = parser.parse_args()
    names = CASES if args.case == "all" else (args.case,)
    for name in names:
        run_case(name, args.device)
    print(f"GM→L1 bool/hif8 accuracy: {len(names)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
