# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check img2col parent-tile writes and L0B auto-transpose on Ascend 950."""

from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401

from catlass import tla
from catlass.tla.runtime import from_dlpack

CHANNELS = 64


@tla.kernel
def load_tile(
    fmap: tla.Tensor,
    initial: tla.Tensor,
    identity: tla.Tensor,
    out: tla.Tensor,
    dtype: tla.Constexpr,
    readback_dtype: tla.Constexpr,
    accumulator_dtype: tla.Constexpr,
    c0: tla.Constexpr,
    to_l0b: tla.Constexpr,
    pad_bits: tla.Constexpr,
    channels: tla.Constexpr,
    scenario: tla.Constexpr,
) -> None:
    rows = 96 if to_l0b and channels != 8 else 64
    c1 = channels // c0
    fmap_ptr = tla.allocate(rows * channels, dtype, tla.AddressSpace.l1, 512)
    initial_ptr = tla.allocate(rows * channels, dtype, tla.AddressSpace.l1, 512)
    eye_extent = rows if to_l0b else channels
    padded_channels = (channels + 15) // 16 * 16
    eye_ptr = tla.allocate(
        eye_extent * eye_extent, readback_dtype, tla.AddressSpace.l1, 512
    )
    a_ptr = tla.allocate(rows * eye_extent, readback_dtype, tla.AddressSpace.l0a, 512)
    b_ptr = tla.allocate(
        eye_extent * padded_channels, readback_dtype, tla.AddressSpace.l0b, 512
    )
    c_ptr = tla.allocate(
        rows * padded_channels, accumulator_dtype, tla.AddressSpace.l0c, 1024
    )

    inputs_ready = tla.flag("inputs_ready", tla.pipes.MTE2, tla.pipes.MTE1)
    operands_ready = tla.flag("operands_ready", tla.pipes.MTE1, tla.pipes.CUBE)
    result_ready = tla.flag("result_ready", tla.pipes.CUBE, tla.pipes.FIX)

    with tla.cube():
        fmap_layout = tla.make_layout(
            tla.make_shape((rows, 1), (c0, c1)),
            tla.make_stride((c0, rows * c0), (1, rows * c0)),
            origin_shape=tla.make_shape(rows, channels),
            layoutTag=tla.arch.zNUnAlign,
        )
        gm_fmap = tla.make_tensor(tla.recast_ptr(fmap.ptr, dtype=dtype), fmap_layout)
        gm_initial = tla.make_tensor(
            tla.recast_ptr(initial.ptr, dtype=dtype), fmap_layout
        )
        l1_fmap = tla.make_tensor_like(fmap_ptr, gm_fmap, tla.arch.zNUnAlign)
        l1_initial = tla.make_tensor_like(initial_ptr, gm_initial, tla.arch.zNUnAlign)
        l1_eye = tla.make_tensor_like(eye_ptr, identity, tla.arch.zN)
        if tla.const_expr(scenario == "gm_c1_crop"):
            gm_tile = tla.get_tile(
                gm_fmap, tla.make_coord(32, c0), tla.make_shape(32, c0)
            )
            src = tla.make_tensor_like(fmap_ptr, gm_tile, tla.arch.zNUnAlign)
            tla.copy(src, gm_tile)
        elif tla.const_expr(scenario == "l1_c1_dst"):
            gm_tile = tla.get_tile(
                gm_fmap, tla.make_coord(0, c0), tla.make_shape(32, c0)
            )
            src = tla.get_tile(l1_fmap, tla.make_coord(32, c0), tla.make_shape(32, c0))
            tla.copy(src, gm_tile)
        else:
            tla.copy(l1_fmap, gm_fmap)
            src = (
                tla.get_tile(l1_fmap, tla.make_coord(32, 0), tla.make_shape(32, c0))
                if tla.const_expr(scenario == "single_c1_crop")
                else l1_fmap
            )
        tla.copy(l1_initial, gm_initial)
        tla.copy(l1_eye, identity)
        tla.set_flag(inputs_ready)
        tla.wait_flag(inputs_ready)

        params = tla.experimental.Img2colParams(
            fmap_hw=(1, rows),
            filter_hw=(1, 1),
            pad=(1 if scenario == "padding" else 0, 0, 0, 0),
            pad_value=pad_bits,
        )
        if tla.const_expr(to_l0b):
            l0_a = tla.make_tensor_like(a_ptr, identity, tla.arch.zN)
            l0_b = tla.make_tensor_like(b_ptr, out, tla.arch.nZ)
            l0_loaded = tla.make_tensor_like(
                tla.recast_ptr(b_ptr, dtype=dtype), out, tla.arch.nZ
            )
            tla.copy(l0_a, l1_eye)
        else:
            l0_a = tla.make_tensor_like(a_ptr, out, tla.arch.zN)
            l0_b = tla.make_tensor_like(b_ptr, identity, tla.arch.nZ)
            l0_loaded = tla.make_tensor_like(
                tla.recast_ptr(a_ptr, dtype=dtype), out, tla.arch.zN
            )
            tla.copy(l0_b, l1_eye)

        if tla.const_expr(scenario in ("single_c1_crop", "gm_c1_crop", "l1_c1_dst")):
            dst = tla.get_tile(l0_loaded, tla.make_coord(0, 0), tla.make_shape(32, c0))
            tla.experimental.img2col(l0_loaded, l1_initial, params)
            tla.experimental.img2col(
                dst,
                src,
                tla.experimental.Img2colParams(
                    fmap_hw=(1, 32),
                    filter_hw=(1, 1),
                    repeat_load_stride=0,
                    repeat_load_time=1,
                    repeat_load_axis=tla.experimental.height,
                ),
            )
        elif tla.const_expr(scenario == "sub_tile"):
            tla.experimental.img2col(l0_loaded, l1_initial, params)
            m_start = 32 if to_l0b else 16
            k_start = 32 if c0 == 32 else 16
            dst = tla.get_tile(
                l0_loaded,
                tla.make_coord(m_start, k_start),
                tla.make_shape(48 if to_l0b else 32, 32),
            )
            tla.experimental.img2col(dst, l1_fmap, params)
        elif tla.const_expr(scenario != "padding"):
            tla.experimental.img2col(l0_loaded, l1_initial, params)
            repeat_width = scenario == "repeat_width"
            height = 16 if scenario == "repeat_gap" else rows // 2
            stride = 2 if scenario == "repeat_gap" else height // 16
            repeat_time = 0 if scenario == "nop" else 2
            dst = tla.get_tile(
                l0_loaded,
                tla.make_coord(0, 0),
                tla.make_shape(
                    rows if repeat_width else height,
                    channels // 2 if repeat_width else channels,
                ),
            )
            tla.experimental.img2col(
                dst,
                l1_fmap,
                tla.experimental.Img2colParams(
                    fmap_hw=(1, rows),
                    filter_hw=(1, 1),
                    repeat_load_stride=channels // 2 // c0 if repeat_width else stride,
                    repeat_load_time=repeat_time,
                    repeat_load_axis=tla.experimental.width
                    if repeat_width
                    else tla.experimental.height,
                ),
            )
        else:
            tla.experimental.img2col(l0_loaded, l1_fmap, params)

        tla.set_flag(operands_ready)
        tla.wait_flag(operands_ready)
        # Multiply by the identity to read back the entire L0 operand.
        l0_c = tla.make_tensor_like(c_ptr, out, tla.arch.L0Clayout)
        tla.mmad(l0_c, l0_a, l0_b, init_c=True)
        tla.set_flag(result_ready)
        tla.wait_flag(result_ready)
        tla.copy(out, l0_c)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(7)

    # Equal-width views let MMAD read the original bits for integer and FP8 loads
    # even when MMAD has no compute route for the loaded type.
    for dtype, torch_dtype, readback_dtype in (
        (tla.Float16, torch.float16, tla.Float16),
        (tla.BFloat16, torch.bfloat16, tla.BFloat16),
        (tla.Float32, torch.float32, tla.Float32),
        (tla.Int32, torch.float32, tla.Float32),
        (tla.UInt32, torch.float32, tla.Float32),
        (tla.Int8, torch.int8, tla.Int8),
        (tla.UInt8, torch.int8, tla.Int8),
        (tla.HiFloat8, torch.int8, tla.Int8),
        (tla.Float8E4M3FN, torch.int8, tla.Int8),
        (tla.Float8E5M2, torch.int8, tla.Int8),
    ):
        c0 = 32 // torch_dtype.itemsize
        accumulator_dtype = tla.Int32 if c0 == 32 else tla.Float32
        torch_out_dtype = torch.int32 if c0 == 32 else torch.float32
        pad_bytes = torch.tensor([2], dtype=torch_dtype).view(torch.uint8).tolist()
        pad_bits = int.from_bytes(bytes(pad_bytes), "little")
        destinations = [(False, CHANNELS)]
        if c0 != 32:
            destinations.append((True, CHANNELS))
        if dtype is tla.Float32:
            destinations.append((True, 8))
        for to_l0b, channels in destinations:
            rows = 96 if to_l0b and channels != 8 else 64
            identity = torch.eye(
                rows if to_l0b else channels, dtype=torch_dtype, device=device
            )
            fmap = torch.randint(-8, 9, (rows, channels), generator=generator).to(
                torch_dtype
            )
            packed = (
                fmap.reshape(rows, channels // c0, c0)
                .transpose(0, 1)
                .contiguous()
                .reshape(-1, c0)
            )
            initial = torch.full_like(packed, 2).to(device)
            expected = fmap.to(torch_out_dtype)
            out = torch.empty(expected.shape, dtype=torch_out_dtype, device=device)
            tensors = tuple(
                from_dlpack(tensor, layout_tag=tla.arch.RowMajor)
                for tensor in (packed.to(device), initial, identity, out)
            )
            cases = ["padding"]
            if channels == CHANNELS:
                cases.append("sub_tile")
            if dtype is tla.Float16:
                cases.extend(
                    [
                        "nop",
                        "repeat_height",
                        "repeat_width",
                        "repeat_gap",
                        "single_c1_crop",
                        "gm_c1_crop",
                        "l1_c1_dst",
                    ]
                )
            for scenario in cases:
                compiled = tla.compile(
                    load_tile,
                    *tensors,
                    dtype,
                    readback_dtype,
                    accumulator_dtype,
                    c0,
                    to_l0b,
                    pad_bits,
                    channels,
                    scenario,
                    options="--npu-arch 3510",
                )
                compiled(*tensors, block_num=1)
                if scenario in ("single_c1_crop", "gm_c1_crop", "l1_c1_dst"):
                    reference = torch.full_like(expected, 2)
                    src_row = 0 if scenario == "l1_c1_dst" else 32
                    src_col = 0 if scenario == "single_c1_crop" else c0
                    reference[:32, :c0] = expected[
                        src_row : src_row + 32, src_col : src_col + c0
                    ]
                elif scenario == "repeat_gap":
                    reference = torch.full_like(expected, 2)
                    reference[:16] = expected[:16]
                    reference[16:32] = expected[32:48]
                elif scenario == "sub_tile":
                    reference = torch.full_like(expected, 2)
                    k_start = 32 if c0 == 32 else 16
                    window = (
                        slice(32, 80) if to_l0b else slice(16, 48),
                        slice(k_start, k_start + 32),
                    )
                    reference[window] = expected[window]
                elif scenario == "nop":
                    reference = torch.full_like(expected, 2)
                elif scenario in ("repeat_height", "repeat_width"):
                    reference = expected
                else:
                    reference = torch.empty_like(expected)
                    reference[0] = 2
                    reference[1:] = expected[:-1]
                torch.testing.assert_close(out.cpu(), reference, rtol=0, atol=0)
                print(
                    f"{dtype.__name__} {'L0B' if to_l0b else 'L0A'} "
                    f"{rows}x{channels} {scenario}: PASS"
                )


if __name__ == "__main__":
    main()
