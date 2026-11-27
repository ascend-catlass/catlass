# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E: flat ``VectorLayout`` GM->L1 copy for continuous-run dtypes.

Exercises the continuous-run path end to end: host bind -> cube ``tla.copy`` ->
L1 debug print readback. One C0 (32 bytes) of known pattern is staged so the
print side can reuse the zN ``u32`` C0 record format used by ``fill_op``.

L1 ``tla.print`` only accepts zN/nZ, so after the VectorLayout copy the same
buffer is re-viewed as a single ``(1, 8)`` ``u32`` zN C0 for verification.

Battery cases: ``vector-layout-gm-l1-{i8,f16,bf16,i64,u64,f64}``.
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

from catlass import tla

# One L1 C0 is 32 bytes -> eight u32 words for print.
_C0_BYTES = 32
_C0_U32 = 8

# dtype -> (tla type, element count for 32 bytes, pack host payload -> bytes)
DTYPE_SPECS = {
    "i8": (tla.Int8, 32),
    "f16": (tla.Float16, 16),
    "bf16": (tla.BFloat16, 16),
    "i64": (tla.Int64, 4),
    "u64": (tla.UInt64, 4),
    "f64": (tla.Float64, 4),
}


@tla.kernel
def vector_layout_gm_to_l1_copy(
    gm_src: tla.Tensor,
    dtype: tla.Constexpr,
    n_elems: tla.Constexpr,
) -> None:
    """GM VectorLayout -> L1 VectorLayout, then print L1 as one u32 C0."""
    l1_ptr = tla.allocate(n_elems, dtype, tla.AddressSpace.l1, 512)
    copied = tla.flag("vl_gm_l1_copied", tla.pipes.MTE2, tla.pipes.MTE1)
    with tla.cube():
        l1 = tla.make_tensor_like(l1_ptr, gm_src, tla.arch.VectorLayout)
        tla.copy(l1, gm_src)
        tla.set_flag(copied)
        tla.wait_flag(copied)
        # Print needs zN/nZ on L1; the flat 32-byte run is one C0 of eight u32.
        print_ptr = tla.recast_ptr(l1_ptr, dtype=tla.UInt32)
        printed = tla.make_tensor(
            print_ptr,
            tla.make_layout_with_tag((1, _C0_U32), tla.UInt32, tla.arch.zN),
        )
        tla.print(printed, _C0_U32)


def _pack_payload(dtype_name: str) -> bytes:
    """Build a deterministic 32-byte pattern for ``dtype_name``."""
    if dtype_name == "i8":
        # Distinct signed bytes across the C0.
        return bytes((i - 16) & 0xFF for i in range(_C0_BYTES))
    if dtype_name == "f16":
        import numpy as np

        vals = np.arange(16, dtype=np.float16) * np.float16(0.5) - np.float16(2.0)
        return vals.tobytes()
    if dtype_name == "bf16":
        import numpy as np

        # numpy has no native bf16; pack via float32 -> truncate to bf16 bits.
        f32 = (np.arange(16, dtype=np.float32) * 0.5) - 2.0
        bits = f32.view(np.uint32)
        # Round-to-nearest-even truncate to bf16 (match Float32->bf16 convention).
        out = bytearray()
        for b in bits:
            b = int(b)
            if (b & 0x7F800000) != 0x7F800000:
                b = (b + 0x00007FFF + ((b >> 16) & 1)) & 0xFFFFFFFF
            out += struct.pack("<H", (b >> 16) & 0xFFFF)
        return bytes(out)
    if dtype_name == "i64":
        vals = (-3, -1, 0, 42)
        return struct.pack("<4q", *vals)
    if dtype_name == "u64":
        vals = (0, 1, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF)
        return struct.pack("<4Q", *vals)
    if dtype_name == "f64":
        vals = (-2.5, -0.0, 1.5, 8.0)
        return struct.pack("<4d", *vals)
    raise ValueError(dtype_name)


def _expected_u32_words(payload: bytes) -> list[int]:
    if len(payload) != _C0_BYTES:
        raise ValueError(f"payload must be {_C0_BYTES} bytes, got {len(payload)}")
    return list(struct.unpack(f"<{_C0_U32}I", payload))


def _parse_print_words(output: str) -> list[int]:
    records = [line.strip() for line in output.splitlines() if line.startswith("tla.print ")]
    if len(records) != 1:
        raise AssertionError(
            f"expected 1 L1 print record, got {len(records)}: {records}"
        )
    # fill_op may emit call=/block=; single-core VectorLayout path often omits them.
    match = re.fullmatch(
        r"tla\.print (?:call=\d+ block=\d+ )?"
        r"dtype=u32 (?:position=\S+ )?"
        r"shape=\[1,8\] count=8 values=\[([0-9, ]+)\]",
        records[0],
    )
    if match is None:
        raise AssertionError(f"malformed L1 record: {records[0]}")
    words = [int(v.strip()) for v in match[1].split(",")]
    if len(words) != _C0_U32:
        raise AssertionError(f"incomplete C0: {words}")
    return words


def _bind_vector_gm(payload: bytes, dtype_name: str, device: int):
    """Put ``payload`` on NPU and bind as VectorLayout with the logical dtype."""
    import torch
    import torch_npu
    from helpers import create_tla_tensor

    del torch_npu  # import required for torch.npu
    tla_dtype, n_elems = DTYPE_SPECS[dtype_name]
    # Storage view: torch carries same-width integer / float buffer; override
    # element_type when DLPack cannot spell the DSL type (u64 / f64 / bf16).
    if dtype_name == "i8":
        buf = torch.frombuffer(bytearray(payload), dtype=torch.int8).clone()
        element_type = None
    elif dtype_name == "f16":
        buf = torch.frombuffer(bytearray(payload), dtype=torch.float16).clone()
        element_type = None
    elif dtype_name == "bf16":
        # DLPack-friendly int16 storage; override pointee as BFloat16.
        buf = torch.frombuffer(bytearray(payload), dtype=torch.int16).clone()
        element_type = tla.BFloat16
    elif dtype_name in ("i64", "u64", "f64"):
        # int64 storage for i64 / u64 / f64 bit patterns; override when needed.
        buf = torch.frombuffer(bytearray(payload), dtype=torch.int64).clone()
        element_type = None if dtype_name == "i64" else tla_dtype
    else:
        raise ValueError(dtype_name)

    if buf.numel() != n_elems:
        raise ValueError(f"payload element count {buf.numel()} != {n_elems}")
    buf = buf.to(device=f"npu:{device}")
    return create_tla_tensor(buf, "vector", element_type=element_type)


def run_one(dtype_name: str, device: int) -> None:
    from catlass.base_dsl.runtime.ascend_debug_fifo import _debug_bus_support_error

    reason = _debug_bus_support_error()
    if reason:
        raise SystemExit(f"SKIP: L1 debug readback unavailable: {reason}")

    import torch
    import torch_npu

    del torch_npu  # import required for torch.npu
    torch.npu.set_device(device)

    tla_dtype, n_elems = DTYPE_SPECS[dtype_name]
    payload = _pack_payload(dtype_name)
    expected = _expected_u32_words(payload)
    gm = _bind_vector_gm(payload, dtype_name, device)

    artifact = tla.compile(
        vector_layout_gm_to_l1_copy,
        gm,
        tla_dtype,
        n_elems,
        options="--npu-arch 3510",
    )
    captured = StringIO()
    with redirect_stdout(captured):
        artifact(gm, block_num=1)
        torch.npu.synchronize()
    words = _parse_print_words(captured.getvalue())
    if words != expected:
        raise AssertionError(
            f"{dtype_name}: L1 words {words} != expected {expected} "
            f"(payload={payload.hex()})"
        )
    print(
        f"PASS dtype={dtype_name} n={n_elems} "
        f"cache_key={artifact.cache_key} words={[hex(w) for w in words]}"
    )
    print(f"kernel.o={artifact.kernel_binary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=(*DTYPE_SPECS, "all"),
        default="all",
        help="Element type for the VectorLayout GM->L1 copy (default: all)",
    )
    args = parser.parse_args()
    names = tuple(DTYPE_SPECS) if args.dtype == "all" else (args.dtype,)
    for name in names:
        run_one(name, args.device)
    print(f"VectorLayout GM->L1: {len(names)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
