# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E: Host ``from_dlpack`` NZFamily (4-leaf) bind → GM zN/nZ → L1 same-layout copy.

Demonstrates the Host-side 4D / NZFamily path end to end:

1. Pack an ND tile on the Host with ``host_nd2nz``.
2. Export a four-dimensional view of the packed buffer and bind it directly
   via ``from_dlpack(view_4d, layout_tag=tag)``. Cross-check the schema-v4 13
   launch fields against binding the same flat buffer with the tag and origin.
3. Cube-copy the prepacked GM tile into an L1 tile with the same fractal tag
   (plain block move, no ND→NZ reorder).
4. Read every L1 C0 back through ``tla.print`` and compare to the packed
   Host bytes.

Battery cases: ``host-dlpack-zn-gm-l1``, ``host-dlpack-nz-gm-l1``.
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
    # append (not insert(0)) to avoid shadowing installed packages (G.PSL.03).
    sys.path.append(_DSL_EXAMPLE_PATH)

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# Aligned f16 tile with multiple fractals along both logical axes.
_ROWS = 32
_COLS = 48
_ELEMS = _ROWS * _COLS
_VIEW_SHAPE = (16, 2, 16, 3)
_VIEW_STRIDES = {"zN": (16, 256, 1, 512), "nZ": (1, 768, 16, 256)}
_C0_BYTES = 32
_C0_U32 = 8
_C0_COUNT = (_ELEMS * 2) // _C0_BYTES  # f16 = 2 bytes


@tla.kernel
def gm_fractal_to_l1_copy(
    gm_src: tla.Tensor,
    layout_tag: tla.Constexpr,
) -> None:
    """GM NZFamily → L1 same tag, then print every L1 C0 as u32."""
    # Byte-addressed L1 base so C0 strides stay 32B (same pattern as fill_op).
    l1_bytes = tla.allocate(_ELEMS * 2, tla.Int8, tla.AddressSpace.l1, 512)
    l1_ptr = tla.recast_ptr(l1_bytes, dtype=tla.Float16)
    copied = tla.flag("gm_fractal_l1_copied", tla.arch.MTE2, tla.arch.MTE1)
    with tla.cube():
        l1 = tla.make_tensor_like(l1_ptr, gm_src, layout_tag)
        tla.copy(l1, gm_src)
        tla.set_flag(copied)
        tla.wait_flag(copied)
        for c0_idx in tla.range_constexpr(_C0_COUNT):
            print_ptr = tla.recast_ptr(l1_bytes + c0_idx * _C0_BYTES, dtype=tla.UInt32)
            printed = tla.make_tensor(
                print_ptr,
                tla.make_layout_with_tag((1, _C0_U32), tla.UInt32, tla.arch.zN),
            )
            tla.print(printed, _C0_U32)


def _nd_pattern():
    """Deterministic f16 ND tile on the Host."""
    import torch

    vals = (torch.arange(_ELEMS, dtype=torch.float32) * 0.25 - 2.0).reshape(
        _ROWS, _COLS
    )
    return vals.half().contiguous()


def _expected_u32_words(packed_1d) -> list[int]:
    import torch

    payload = packed_1d.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    expected_bytes = _C0_COUNT * _C0_BYTES
    if len(payload) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} packed bytes, got {len(payload)}")
    return list(struct.unpack(f"<{_C0_COUNT * _C0_U32}I", payload))


def _parse_print_words(output: str) -> list[int]:
    records = [
        line.strip()
        for line in output.splitlines()
        if line.startswith("tla.print ")
    ]
    if len(records) != _C0_COUNT:
        raise AssertionError(
            f"expected {_C0_COUNT} L1 print records, got {len(records)}: {records[:3]}..."
        )
    words: list[int] = []
    pattern = re.compile(
        r"tla\.print (?:call=\d+ block=\d+ )?"
        r"dtype=u32 (?:position=\S+ )?"
        r"shape=\[1,8\] count=8 values=\[([0-9, ]+)\]"
    )
    for record in records:
        match = pattern.fullmatch(record)
        if match is None:
            raise AssertionError(f"malformed L1 record: {record}")
        chunk = [int(v.strip()) for v in match[1].split(",")]
        if len(chunk) != _C0_U32:
            raise AssertionError(f"incomplete C0: {chunk}")
        words.extend(chunk)
    return words


def _check_binding(tensor, layout: str, origin_shape: tuple[int, int]) -> None:
    """Check the launch descriptor before issuing the copy."""
    fields = tensor.build_memref_launch_fields()
    if len(fields) != 13:
        raise RuntimeError(
            f"expected schema-v4 13-slot launch fields for {layout}, got {len(fields)}"
        )
    if fields[-2:] != origin_shape:
        raise RuntimeError(
            f"origin slots mismatch for {layout}: {fields[-2:]} != {origin_shape}"
        )
    if tensor.layout_tag != layout:
        raise RuntimeError(f"layout_tag={tensor.layout_tag!r}, expected {layout!r}")
    flat_shape = []
    for group in tensor.shape:
        flat_shape.extend(group)
    if len(flat_shape) != 4:
        raise RuntimeError(
            f"{layout} physical shape must have 4 leaves, got {tensor.shape!r}"
        )


def run_one(layout: str, device: int) -> None:
    from catlass.base_dsl.runtime.ascend_debug_fifo import _debug_bus_support_error
    from helpers import host_nd2nz

    reason = _debug_bus_support_error()
    if reason:
        raise SystemExit(f"SKIP: L1 debug readback unavailable: {reason}")

    import torch
    import torch_npu

    del torch_npu  # import required for torch.npu
    torch.npu.set_device(device)

    nd = _nd_pattern()
    packed = host_nd2nz(nd, layout)
    expected_elems = _ELEMS
    if int(packed.numel()) != expected_elems:
        raise RuntimeError(
            f"{layout} packed numel {packed.numel()} != {expected_elems} "
            f"for logical ({_ROWS}, {_COLS})"
        )
    expected = _expected_u32_words(packed)
    packed_npu = packed.npu()
    tag = tla.arch.zN if layout == "zN" else tla.arch.nZ
    origin = (_ROWS, _COLS)
    view_4d = packed_npu.as_strided(size=_VIEW_SHAPE, stride=_VIEW_STRIDES[layout])
    gm = from_dlpack(view_4d, layout_tag=tag).mark_layout_dynamic()
    flat = from_dlpack(packed_npu, layout_tag=tag, origin_shape=origin)
    _check_binding(gm, layout, origin)
    if gm.build_memref_launch_fields() != flat.build_memref_launch_fields():
        raise AssertionError(
            f"{layout}: flat and 4-D export produced different launch fields"
        )

    print(
        f"--- compile layout={layout} shape={gm.shape} "
        f"origin={gm.origin_shape} slots=13 ---",
        flush=True,
    )
    artifact = tla.compile(
        gm_fractal_to_l1_copy,
        gm,
        tag,
        options="--npu-arch 3510",
    )
    captured = StringIO()
    with redirect_stdout(captured):
        artifact(gm, block_num=1)
        torch.npu.synchronize()
    words = _parse_print_words(captured.getvalue())
    if words != expected:
        raise AssertionError(
            f"{layout}: L1 words mismatch "
            f"(got {len(words)} u32, expected {len(expected)})"
        )
    print(
        f"PASS layout={layout} origin=({_ROWS},{_COLS}) "
        f"slots=13 c0s={_C0_COUNT} cache_key={artifact.cache_key}",
        flush=True,
    )
    print(f"kernel.o={artifact.kernel_binary_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--layout",
        choices=("zN", "nZ", "all"),
        default="all",
        help="Prepacked fractal layout bound via from_dlpack (default: all)",
    )
    args = parser.parse_args()
    names = ("zN", "nZ") if args.layout == "all" else (args.layout,)
    for name in names:
        run_one(name, args.device)
    print(f"Host DLPack NZFamily GM→L1: {len(names)} passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
