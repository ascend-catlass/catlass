"""Check aligned Load/Store distributions against a byte-level reference on Ascend 950."""

from __future__ import annotations

import argparse
from typing import NamedTuple

import numpy as np

from catlass import tla
from catlass.params import (
    BlockLoadParams,
    BlockStoreParams,
    LoadDist,
    MaskLoadDist,
    MaskLoadParams,
    MaskStoreDist,
    MaskStoreParams,
    NormalLoadParams,
    NormalStoreParams,
    PostMode,
    StoreDist,
)
from catlass.tla.runtime import from_dlpack, make_fake_tensor

_SIZE = 2048
_STEP = 544


class Case(NamedTuple):
    name: str
    kind: str
    dtype: str
    dist: str
    unit: int
    post: bool = True
    mask_stride: int = 64


_CASES = (
    Case("brc-b8", "load", "UInt8", LoadDist.DIST_BRC_B8, 1),
    Case("brc-b16", "load", "UInt16", LoadDist.DIST_BRC_B16, 2),
    Case("brc-b32", "load", "Float32", LoadDist.DIST_BRC_B32, 4),
    Case("brc-b8-u32", "load", "UInt32", LoadDist.DIST_BRC_B8, 1, False),
    Case("brc-b16-u32", "load", "UInt32", LoadDist.DIST_BRC_B16, 2, False),
    Case("us-b8", "load", "UInt8", LoadDist.DIST_US_B8, 1),
    Case("us-b16", "load", "UInt16", LoadDist.DIST_US_B16, 2),
    Case("ds-b8", "load", "UInt8", LoadDist.DIST_DS_B8, 1),
    Case("ds-b16", "load", "UInt16", LoadDist.DIST_DS_B16, 2),
    Case("unpack-b8", "load", "UInt8", LoadDist.DIST_UNPACK_B8, 1),
    Case("unpack-b16", "load", "UInt16", LoadDist.DIST_UNPACK_B16, 2),
    Case("unpack-b16-bf16", "load", "BFloat16", LoadDist.DIST_UNPACK_B16, 2),
    Case("unpack-b16-u8", "load", "UInt8", LoadDist.DIST_UNPACK_B16, 2),
    Case("unpack-b16-u8-normal", "load", "UInt8", LoadDist.DIST_UNPACK_B16, 2, False),
    Case("unpack-b32", "load", "UInt32", LoadDist.DIST_UNPACK_B32, 4),
    Case("unpack4-b8", "load", "Int8", LoadDist.DIST_UNPACK4_B8, 1),
    Case("unpack4-b8-u8", "load", "UInt8", LoadDist.DIST_UNPACK4_B8, 1),
    Case("blk", "load", "UInt8", LoadDist.DIST_BLK, 1),
    Case("e2b-b16", "load", "UInt16", LoadDist.DIST_E2B_B16, 2),
    Case("e2b-b16-bf16", "load", "BFloat16", LoadDist.DIST_E2B_B16, 2),
    Case("e2b-b32", "load", "Float32", LoadDist.DIST_E2B_B32, 4),
    Case("e2b-b32-u32", "load", "UInt32", LoadDist.DIST_E2B_B32, 4),
    Case("first-b8", "store", "UInt8", StoreDist.DIST_FIRST_ELEMENT_B8, 1),
    Case("first-b16", "store", "UInt16", StoreDist.DIST_FIRST_ELEMENT_B16, 2),
    Case("first-b32", "store", "Float32", StoreDist.DIST_FIRST_ELEMENT_B32, 4),
    Case("first-b32-u32", "store", "UInt32", StoreDist.DIST_FIRST_ELEMENT_B32, 4),
    Case("pack-b16", "store", "UInt16", StoreDist.DIST_PACK_B16, 2),
    Case("pack-b32-bf16", "store", "BFloat16", StoreDist.DIST_PACK_B32, 4),
    Case("pack-b32", "store", "Float32", StoreDist.DIST_PACK_B32, 4),
    Case("pack-b32-u32", "store", "UInt32", StoreDist.DIST_PACK_B32, 4),
    Case("pack-b64", "store", "UInt64", StoreDist.DIST_PACK_B64, 8),
    Case("pack4-b32", "store", "Int8", StoreDist.DIST_PACK4_B32, 4),
    Case("pack4-b32-u8", "store", "UInt8", StoreDist.DIST_PACK4_B32, 4),
    Case("norm-b8", "store", "UInt8", StoreDist.DIST_NORM, 1),
    Case("norm-b16", "store", "UInt16", StoreDist.DIST_NORM, 2),
    Case("norm-b32", "store", "UInt32", StoreDist.DIST_NORM, 4),
    Case("norm-b8-normal", "store", "UInt8", StoreDist.DIST_NORM, 1, False),
    Case("norm-b16-normal", "store", "UInt16", StoreDist.DIST_NORM, 2, False),
    Case("norm-b32-normal", "store", "UInt32", StoreDist.DIST_NORM, 4, False),
    Case("dintlv-b8-u16", "dual_load", "UInt16", LoadDist.DIST_DINTLV_B8, 1),
    Case(
        "dintlv-b8-u16-normal", "dual_load", "UInt16", LoadDist.DIST_DINTLV_B8, 1, False
    ),
    Case("dintlv-b16-u32", "dual_load", "UInt32", LoadDist.DIST_DINTLV_B16, 2),
    Case(
        "dintlv-b16-u32-normal",
        "dual_load",
        "UInt32",
        LoadDist.DIST_DINTLV_B16,
        2,
        False,
    ),
    Case("intlv-b8-u16", "dual_store", "UInt16", StoreDist.DIST_INTLV_B8, 1, False),
    Case("intlv-b16-u32", "dual_store", "UInt32", StoreDist.DIST_INTLV_B16, 2, False),
    Case("dintlv-b32-u64", "dual_load", "UInt64", LoadDist.DIST_DINTLV_B32, 4, False),
    Case("dintlv-b32-i64", "dual_load", "Int64", LoadDist.DIST_DINTLV_B32, 4, False),
    Case("dintlv-b32-f64", "dual_load", "Float64", LoadDist.DIST_DINTLV_B32, 4, False),
    Case("intlv-b32-u64", "dual_store", "UInt64", StoreDist.DIST_INTLV_B32, 4, False),
    Case("intlv-b32-i64", "dual_store", "Int64", StoreDist.DIST_INTLV_B32, 4, False),
    Case("intlv-b32-f64", "dual_store", "Float64", StoreDist.DIST_INTLV_B32, 4, False),
    Case("dintlv-b8-i4x2", "dual_load", "Int4x2", LoadDist.DIST_DINTLV_B8, 1),
    Case(
        "dintlv-b8-i4x2-normal",
        "dual_load",
        "Int4x2",
        LoadDist.DIST_DINTLV_B8,
        1,
        False,
    ),
    Case("norm-load-i4x2", "load", "Int4x2", LoadDist.DIST_NORM, 1),
    Case("norm-store-i4x2", "store", "Int4x2", StoreDist.DIST_NORM, 1),
    Case("mask-norm-load", "mask_load", "UInt8", MaskLoadDist.DIST_NORM, 1),
    Case(
        "mask-load-zero", "mask_load", "UInt8", MaskLoadDist.DIST_NORM, 1, mask_stride=0
    ),
    Case("mask-us-load", "mask_load", "UInt8", MaskLoadDist.DIST_US, 1),
    Case("mask-ds-load", "mask_load", "UInt8", MaskLoadDist.DIST_DS, 1),
    Case("mask-us-load-u32", "mask_load", "UInt32", MaskLoadDist.DIST_US, 1),
    Case("mask-ds-load-u32", "mask_load", "UInt32", MaskLoadDist.DIST_DS, 1),
    Case("mask-norm-store", "mask_store", "UInt8", MaskStoreDist.DIST_NORM, 1),
    Case(
        "mask-store-zero",
        "mask_store",
        "UInt8",
        MaskStoreDist.DIST_NORM,
        1,
        mask_stride=0,
    ),
    Case("mask-norm-store-u32", "mask_store", "UInt32", MaskStoreDist.DIST_NORM, 1),
    Case("mask-pack-store-u32", "mask_store", "UInt32", MaskStoreDist.DIST_PACK, 1),
    Case("mask-pack-store", "mask_store", "UInt8", MaskStoreDist.DIST_PACK, 1),
    Case(
        "mask-pack-store-normal",
        "mask_store",
        "UInt8",
        MaskStoreDist.DIST_PACK,
        1,
        False,
    ),
    Case("block-load", "block_load", "UInt8", "", 1, False),
    Case("block-load-update", "block_load", "UInt8", "", 1),
    Case("block-store", "block_store", "UInt8", "", 1, False),
    Case("block-store-update", "block_store", "UInt8", "", 1),
    *(
        Case(name + ("-update" if post else ""), "store", dtype, dist, unit, post)
        for name, dtype, dist, unit in (
            ("norm-b8-u8", "UInt8", StoreDist.DIST_NORM_B8, 1),
            ("norm-b8-u16", "UInt16", StoreDist.DIST_NORM_B8, 1),
            ("norm-b8-u32", "UInt32", StoreDist.DIST_NORM_B8, 1),
            ("norm-b8-bf16", "BFloat16", StoreDist.DIST_NORM_B8, 1),
            ("norm-b16-u16", "UInt16", StoreDist.DIST_NORM_B16, 2),
            ("norm-b16-u32", "UInt32", StoreDist.DIST_NORM_B16, 2),
            ("norm-b16-f32", "Float32", StoreDist.DIST_NORM_B16, 2),
            ("norm-b32-u32", "UInt32", StoreDist.DIST_NORM_B32, 4),
            ("norm-b32-u64", "UInt64", StoreDist.DIST_NORM_B32, 4),
        )
        for post in (False, True)
    ),
)


def make_kernel(case: Case):
    kind, dist, post = case.kind, case.dist, case.post
    dtype = getattr(tla, case.dtype)
    width = 1 if dtype is tla.Int4x2 else dtype.width // 8
    register_lanes = 256 // width
    stride = _STEP // width
    mode = PostMode.POST_MODE_UPDATE if post else PostMode.POST_MODE_NORMAL
    load = NormalLoadParams(
        load_dist=dist, post_mode=mode, post_update_stride=stride if post else 0
    )
    store = NormalStoreParams(
        store_dist=dist, post_mode=mode, post_update_stride=stride if post else 0
    )
    mask_stride = case.mask_stride if post else 0
    mask_load = MaskLoadParams(
        load_dist=dist, post_mode=mode, post_update_stride=mask_stride
    )
    mask_store = MaskStoreParams(
        store_dist=dist, post_mode=mode, post_update_stride=mask_stride
    )
    block_load = BlockLoadParams(
        block_stride=2, post_mode=mode, repeat_stride=_STEP // 32 if post else 0
    )
    block_store = BlockStoreParams(
        block_stride=2, post_mode=mode, repeat_stride=_STEP // 32 if post else 0
    )

    @tla.kernel
    def transfer(source: tla.Tensor, initial: tla.Tensor, result: tla.Tensor):
        src_ptr = tla.allocate(_SIZE, tla.UInt8, tla.AddressSpace.ub, 256)
        dst_ptr = tla.allocate(_SIZE, tla.UInt8, tla.AddressSpace.ub, 256)
        src = tla.make_tensor_like(src_ptr, source, tla.arch.RowMajor)
        dst = tla.make_tensor_like(dst_ptr, initial, tla.arch.RowMajor)
        src_typed = tla.recast_ptr(src_ptr, dtype=dtype)
        dst_typed = tla.recast_ptr(dst_ptr, dtype=dtype)
        layout = tla.make_layout(tla.make_shape(register_lanes), tla.make_stride(1))
        ready = tla.flag("in", tla.pipes.MTE2, tla.pipes.VECTOR)
        done = tla.flag("out", tla.pipes.VECTOR, tla.pipes.MTE3)
        with tla.vector():
            tla.copy(src, source)
            tla.copy(dst, initial)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                src_cursor = tla.make_tensor(src_typed, layout, tla.make_coord(0))
                dst_cursor = tla.make_tensor(dst_typed, layout, tla.make_coord(0))
                for i in tla.range_constexpr(2):
                    src_window = tla.make_tensor(
                        src_typed, layout, tla.make_coord(i * stride)
                    )
                    dst_window = tla.make_tensor(
                        dst_typed, layout, tla.make_coord(i * stride)
                    )
                    check = tla.make_tensor(
                        dst_typed, layout, tla.make_coord(i * register_lanes)
                    )
                    if tla.const_expr(kind == "load"):
                        if tla.const_expr(post):
                            reg, src_cursor = src_cursor.load(load)
                        else:
                            reg = src_window.load(load)
                        check.store(reg)
                    elif tla.const_expr(kind == "store"):
                        reg = src_window.load()
                        if tla.const_expr(post):
                            dst_cursor = dst_cursor.store(reg, store)
                        else:
                            dst_window.store(reg, store)
                    elif tla.const_expr(kind == "dual_load"):
                        if tla.const_expr(post):
                            regs, src_cursor = src_cursor.load(load)
                        else:
                            regs = src_window.load(load)
                        low = tla.make_tensor(
                            dst_typed, layout, tla.make_coord(i * 2 * register_lanes)
                        )
                        high = tla.make_tensor(
                            dst_typed,
                            layout,
                            tla.make_coord((i * 2 + 1) * register_lanes),
                        )
                        low.store(regs[0])
                        high.store(regs[1])
                    elif tla.const_expr(kind == "dual_store"):
                        high = tla.make_tensor(
                            src_typed,
                            layout,
                            tla.make_coord(i * stride + register_lanes),
                        )
                        dst_window.store((src_window.load(), high.load()), store)
                    elif tla.const_expr(kind == "mask_load"):
                        mask, src_cursor = src_cursor.load(mask_load)
                        check.store(mask, MaskStoreParams())
                    elif tla.const_expr(kind == "mask_store"):
                        bytes_window = tla.make_tensor(
                            src_ptr, layout, tla.make_coord(i * _STEP)
                        )
                        mask = bytes_window.load(MaskLoadParams())
                        if tla.const_expr(post):
                            dst_cursor = dst_cursor.store(mask, mask_store)
                        else:
                            dst_window.store(mask, mask_store)
                    elif tla.const_expr(kind == "block_load"):
                        pred, _ = tla.update_mask(37, dtype=dtype)
                        if tla.const_expr(post):
                            reg, src_cursor = src_cursor.load(block_load, mask=pred)
                        else:
                            reg = src_window.load(block_load, mask=pred)
                        check.store(reg)
                    else:
                        pred, _ = tla.update_mask(64, dtype=dtype)
                        reg = src_window.load()
                        if tla.const_expr(post):
                            dst_cursor = dst_cursor.store(reg, block_store, mask=pred)
                        else:
                            dst_window.store(reg, block_store, mask=pred)
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(result, dst)
            tla.pipe_barrier(tla.pipes.ALL)

    return transfer


def expected(case: Case, source: np.ndarray) -> np.ndarray:
    out = np.full(_SIZE, 0xA5, dtype=np.uint8)
    unit, dist = case.unit, case.dist
    for i in range(2):
        start = _STEP * i
        data = source[start : start + 512].reshape(-1, unit)
        if case.kind == "load":
            if dist.startswith("brc_"):
                moved = np.tile(data[0], 256 // unit)
            elif dist.startswith("us_"):
                moved = np.repeat(data[: 128 // unit], 2, axis=0)
            elif dist.startswith("ds_"):
                moved = data[::2]
            elif dist.startswith("unpack"):
                expansion = 4 if dist == LoadDist.DIST_UNPACK4_B8 else 2
                moved = np.zeros(
                    (256 // (unit * expansion), unit * expansion), dtype=np.uint8
                )
                moved[:, :unit] = data[: len(moved)]
            elif dist == LoadDist.DIST_BLK:
                moved = np.tile(source[start : start + 32], 8)
            elif dist.startswith("e2b_"):
                moved = np.tile(data[:8], (1, 32 // unit))
            else:
                moved = source[start : start + 256]
            out[i * 256 : (i + 1) * 256] = moved.ravel()
        elif case.kind == "store":
            if dist in (
                StoreDist.DIST_NORM_B8,
                StoreDist.DIST_NORM_B16,
                StoreDist.DIST_NORM_B32,
            ):
                word_bytes = min(4, getattr(tla, case.dtype).width // 8)
                target = out[start : start + 256].reshape(-1, word_bytes)
                target[:, :unit] = source[start : start + 256].reshape(
                    -1, word_bytes
                )[:, :unit]
                continue
            reg = data[: 256 // unit]
            if dist.startswith("first_element_"):
                moved = reg[0]
            elif dist == StoreDist.DIST_PACK4_B32:
                moved = reg[:, 0]
            elif dist.startswith("pack_"):
                moved = reg[:, : unit // 2]
            else:
                moved = reg
            out[start : start + moved.size] = moved.ravel()
        elif case.kind == "dual_load":
            out[i * 512 : (i + 1) * 512] = np.concatenate(
                (data[::2], data[1::2])
            ).ravel()
        elif case.kind == "dual_store":
            lanes = 256 // unit
            dest = out[start : start + 512].reshape(-1, unit)
            dest[::2], dest[1::2] = data[:lanes], data[lanes:]
        elif case.kind == "mask_load":
            offset = i * case.mask_stride
            bits = np.unpackbits(source[offset : offset + 64], bitorder="little")
            if dist == MaskLoadDist.DIST_US:
                bits = np.repeat(bits[:128], 2)
            elif dist == MaskLoadDist.DIST_DS:
                bits = bits[::2]
            out[i * 256 : i * 256 + 32] = np.packbits(bits[:256], bitorder="little")
        elif case.kind == "mask_store":
            bits = np.unpackbits(source[start : start + 32], bitorder="little")
            if dist == MaskStoreDist.DIST_PACK:
                bits = bits[::2]
            moved = np.packbits(bits, bitorder="little")
            offset = i * case.mask_stride if case.post else start
            out[offset : offset + moved.size] = moved
        elif case.kind == "block_load":
            # 37 valid bytes enable two complete DataBlocks.
            out[i * 256 : (i + 1) * 256] = 0
            out[i * 256 : i * 256 + 32] = source[start : start + 32]
            out[i * 256 + 32 : i * 256 + 64] = source[start + 64 : start + 96]
        else:
            out[start : start + 32] = source[start : start + 32]
            out[start + 64 : start + 96] = source[start + 32 : start + 64]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", choices=[case.name for case in _CASES]
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not args.compile_only:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        indices = np.arange(_SIZE, dtype=np.uint32)
        source = ((indices * 37 + 11) ^ (indices >> 3) ^ (indices >> 8)).astype(
            np.uint8
        )
        initial = np.full(_SIZE, 0xA5, dtype=np.uint8)
        inputs = [
            torch.from_numpy(data).to(device)
            for data in (source, initial, initial.copy())
        ]
        views = [from_dlpack(item, layout_tag=tla.arch.RowMajor) for item in inputs]
    fake = make_fake_tensor(
        tla.UInt8,
        (_SIZE,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(_SIZE,),
        layout_tag=tla.arch.RowMajor,
    )
    failures = []
    for case in _CASES:
        if args.case and case.name not in args.case:
            continue
        try:
            compiled = make_kernel(case).compile(
                type_args=(fake, fake, fake), options="--npu-arch 3510"
            )
            if not args.compile_only:
                compiled(*views, block_num=1)
                torch.npu.synchronize(args.device)
                np.testing.assert_array_equal(
                    inputs[2].cpu().numpy(), expected(case, source)
                )
            print(f"PASS {case.name}", flush=True)
        except Exception as exc:
            print(f"FAIL {case.name}: {exc}", flush=True)
            failures.append(case.name)
    if failures:
        raise SystemExit(f"Failed cases: {', '.join(failures)}")


if __name__ == "__main__":
    main()
