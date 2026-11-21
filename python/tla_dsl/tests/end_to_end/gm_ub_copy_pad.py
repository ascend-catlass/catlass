# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device GM→UB CopyPadParams then aligned UB→GM writeback.

GM source is 2×7 Float32 RowMajor. Copy into a 2×8 UB tile with
``right_padding=1`` and a distinctive ``padding_value``, then copy the full
UB tile back to a 2×8 GM destination. Host checks that the first 7 columns
match the source and the padded column equals ``padding_value``.
"""

from __future__ import annotations

import argparse

import catlass.tla as tla

_ROWS = 2
_SRC_COLS = 7
_DST_COLS = 8  # src + right_padding=1; also ELE_NUM_PER_BLK for f32
_PAD_VALUE = 42.0
_ALIGN = 256
_UB_ELEMS = _ROWS * _DST_COLS


@tla.kernel
def gm_ub_pad_then_writeback(gm_src: tla.Tensor, gm_dst: tla.Tensor) -> None:
    ub_ptr = tla.allocate(_UB_ELEMS, tla.Float32, tla.AddressSpace.ub, _ALIGN)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.MTE3)
    with tla.vector():
        ub = tla.make_tensor(
            ub_ptr,
            tla.make_layout(
                tla.make_shape(_ROWS, _DST_COLS),
                tla.make_stride(_DST_COLS, 1),
                origin_shape=tla.make_shape(_ROWS, _DST_COLS),
                layoutTag=tla.arch.RowMajor,
            ),
            tla.make_coord(0, 0),
        )
        tla.copy(
            ub,
            gm_src,
            params=tla.params.CopyPadParams(
                right_padding=1,
                padding_value=_PAD_VALUE,
            ),
        )
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        # Exact UB→GM of the padded tile so host can inspect pad bytes.
        tla.copy(gm_dst, ub)
        tla.pipe_barrier(tla.pipes.ALL)


def run(device: int) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)

    # Distinct payload so a silent zero-fill cannot pass.
    src_host = torch.arange(_ROWS * _SRC_COLS, dtype=torch.float32).reshape(
        _ROWS, _SRC_COLS
    )
    dst_host = torch.full((_ROWS, _DST_COLS), -1.0, dtype=torch.float32)

    src_dev = src_host.to(f"npu:{device}")
    dst_dev = dst_host.to(f"npu:{device}")
    gm_src = tla.from_dlpack(
        src_dev,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(_ROWS, _SRC_COLS),
    )
    gm_dst = tla.from_dlpack(
        dst_dev,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(_ROWS, _DST_COLS),
    )
    artifact = tla.compile(
        gm_ub_pad_then_writeback,
        gm_src,
        gm_dst,
        options="--npu-arch 3510",
    )
    artifact(gm_src, gm_dst, block_num=1)
    torch.npu.synchronize()

    got = dst_dev.cpu()
    payload_ok = torch.equal(got[:, :_SRC_COLS], src_host)
    pad_ok = torch.allclose(
        got[:, _SRC_COLS],
        torch.full((_ROWS,), _PAD_VALUE, dtype=torch.float32),
    )
    passed = bool(payload_ok and pad_ok)
    print(
        f"GM→UB pad1 → UB→GM: rows={_ROWS} src_cols={_SRC_COLS} "
        f"dst_cols={_DST_COLS} pad={_PAD_VALUE} "
        f"payload_ok={payload_ok} pad_ok={pad_ok} passed={passed}"
    )
    if not passed:
        print(f"  got=\n{got}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args().device)


if __name__ == "__main__":
    raise SystemExit(main())
