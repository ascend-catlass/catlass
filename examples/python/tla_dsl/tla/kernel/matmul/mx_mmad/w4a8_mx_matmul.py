# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""MXFP4 weights and MXFP8 activations with bitcode widening on Ascend 950.

The vector cores widen packed FP4 weights in UB with ``tla.experimental.mx_to``
while the cube consumes the previous FP8 tile from double-buffered L1. The
original E8M0 scales are loaded separately and remain valid because every E2M1
value is exactly representable in E4M3. Run with ``--device`` on Ascend 950.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse
from dataclasses import dataclass

import catlass.tla as tla
import gen_data
from catlass.tla.runtime import from_dlpack

MX_SCALE_GROUP_NUM = 32


@dataclass
class W4A8Tiling:
    tm: tla.Constexpr[int] = 128
    tn: tla.Constexpr[int] = 128
    tk: tla.Constexpr[int] = 128
    credits: tla.Constexpr[int] = 2
    dual: tla.Constexpr[int] = 1


@tla.kernel
def w4a8_mx_matmul(
    gm_a: tla.Tensor,
    gm_b4: tla.Tensor,
    gm_b4_shape: tla.Tensor,
    gm_sa: tla.Tensor,
    gm_sb: tla.Tensor,
    gm_c: tla.Tensor,
    _tiling: tla.Constexpr[W4A8Tiling],
) -> None:
    tm = _tiling.tm
    tn = _tiling.tn
    tk = _tiling.tk
    credits = _tiling.credits
    dual = _tiling.dual
    sk = tk // MX_SCALE_GROUP_NUM
    m = gm_a.origin_shape[0]
    n = gm_b4_shape.origin_shape[1]
    grid_m = (m + tm - 1) // tm
    grid_n = (n + tn - 1) // tn
    total_blocks = grid_m * grid_n
    k_tiles = (gm_a.origin_shape[1] + tk - 1) // tk
    c0 = 0
    c1 = 1
    ub_b4 = tla.allocate(tk * tn // 2, tla.Int8, tla.AddressSpace.ub, 256)
    ub_b8 = tla.allocate(tk * tn, tla.Int8, tla.AddressSpace.ub, 256)
    ub_scale_in = tla.allocate(tn * tk // 32, tla.Int8, tla.AddressSpace.ub, 256)
    l1a = tla.allocate(tm * tk, tla.Float8E4M3FN, tla.AddressSpace.l1, 512)
    l1b = tla.allocate(2 * tk * tn, tla.Int8, tla.AddressSpace.l1, 512)
    l1sa = tla.allocate(tm * sk, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(tm * tk, tla.Float8E4M3FN, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(tk * tn, tla.Float8E4M3FN, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 512)
    b4_in_ub = tla.flag("b4_in_ub", tla.arch.MTE2, tla.arch.VECTOR)
    b8_ready = tla.flag("b8_ready", tla.arch.VECTOR, tla.arch.MTE3)
    ub_free = tla.flag("ub_free", tla.arch.MTE3, tla.arch.MTE2)
    l1b_full = tla.cross_flag("l1b_full", mode=4)
    l1b_free = tla.cross_flag("l1b_free", mode=4)
    a_ready = tla.flag("a_ready", tla.arch.MTE2, tla.arch.MTE1)
    l0_ready = tla.flag("l0_ready", tla.arch.MTE1, tla.arch.CUBE)
    l1a_avail = tla.flag("l1a_avail", tla.arch.MTE1, tla.arch.MTE2)
    l0_avail = tla.flag("l0_avail", tla.arch.CUBE, tla.arch.MTE1)
    ks = 2 if dual else 1
    with tla.vector():
        # The cube loads the actual scales separately; widening preserves them.
        scale_flat = tla.make_tensor(
            ub_scale_in, tla.make_layout(tla.make_shape(256), tla.make_stride(1))
        )
        with tla.vec.func(mode="simd"):
            scale_flat.store(tla.full(127, dtype=tla.Int8))
        if tla.arch.sub_block_idx() == 0:
            kb = tk // 2 // ks
            kn = tk // ks
            tla.set_flag(ub_free)
            vblocks = tla.range(
                tla.arch.block_idx(), total_blocks, tla.arch.block_num()
            )
            for block_linear in vblocks:
                block_col = block_linear % grid_n
                for kk in tla.range(c0, k_tiles, c1):
                    gm_b_t = tla.tile_view(
                        gm_b4,
                        tla.make_shape(tn, kb),
                        tla.make_coord(block_col, kk * ks + 0),
                    )
                    t_ub_b4 = tla.make_tensor(
                        ub_b4,
                        tla.make_layout(
                            tla.make_shape(tn, kb),
                            tla.make_stride(kb, 1),
                            origin_shape=tla.make_shape(tn, kb),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                    t_ub_b8 = tla.make_tensor(
                        ub_b8,
                        tla.make_layout(
                            tla.make_shape(tn, kn),
                            tla.make_stride(kn, 1),
                            origin_shape=tla.make_shape(tn, kn),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                    tla.wait_flag(ub_free)
                    tla.copy(t_ub_b4, gm_b_t)
                    tla.set_flag(b4_in_ub)
                    tla.wait_flag(b4_in_ub)
                    mx_in = tla.make_tensor(
                        tla.recast_ptr(ub_b4, dtype=tla.Float4E2M1),
                        tla.make_layout(tla.make_shape(tn, kn), tla.make_stride(kn, 1)),
                    )
                    mx_out = tla.make_tensor(
                        tla.recast_ptr(ub_b8, dtype=tla.Float8E4M3FN),
                        tla.make_layout(tla.make_shape(tn, kn), tla.make_stride(kn, 1)),
                    )
                    mx_si = tla.make_tensor(
                        ub_scale_in,
                        tla.make_layout(
                            tla.make_shape(tn, kn // 32), tla.make_stride(kn // 32, 1)
                        ),
                    )
                    tla.experimental.mx_to((mx_in, mx_si), (mx_out, mx_si))
                    tla.set_flag(b8_ready)
                    tla.wait_flag(b8_ready)
                    l1b_half = l1b + kk % 2 * (tk * tn) + 0 * (tn * kn)
                    t_l1b = tla.make_tensor_like(l1b_half, t_ub_b8, tla.arch.zN)
                    tla.cross_core_wait_flag(l1b_free, tla.arch.MTE3, aiv_id=0)
                    tla.copy(t_l1b, t_ub_b8)
                    tla.set_flag(ub_free)
                    tla.cross_core_set_flag(l1b_full, tla.arch.MTE3, aiv_id=0)
        if dual:  # noqa: SIM102 -- dual is constexpr, sub_block_idx is runtime.
            if tla.arch.sub_block_idx() == 1:
                kb_b = tk // 2 // ks
                kn_b = tk // ks
                tla.set_flag(ub_free)
                vblocks_b = tla.range(
                    tla.arch.block_idx(), total_blocks, tla.arch.block_num()
                )
                for block_linear_b in vblocks_b:
                    block_col_b = block_linear_b % grid_n
                    for kk_b in tla.range(c0, k_tiles, c1):
                        gm_b_t_b = tla.tile_view(
                            gm_b4,
                            tla.make_shape(tn, kb_b),
                            tla.make_coord(block_col_b, kk_b * ks + 1),
                        )
                        t_ub_b4_b = tla.make_tensor(
                            ub_b4,
                            tla.make_layout(
                                tla.make_shape(tn, kb_b),
                                tla.make_stride(kb_b, 1),
                                origin_shape=tla.make_shape(tn, kb_b),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        t_ub_b8_b = tla.make_tensor(
                            ub_b8,
                            tla.make_layout(
                                tla.make_shape(tn, kn_b),
                                tla.make_stride(kn_b, 1),
                                origin_shape=tla.make_shape(tn, kn_b),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.wait_flag(ub_free)
                        tla.copy(t_ub_b4_b, gm_b_t_b)
                        tla.set_flag(b4_in_ub)
                        tla.wait_flag(b4_in_ub)
                        mx_in_b = tla.make_tensor(
                            tla.recast_ptr(ub_b4, dtype=tla.Float4E2M1),
                            tla.make_layout(
                                tla.make_shape(tn, kn_b), tla.make_stride(kn_b, 1)
                            ),
                        )
                        mx_out_b = tla.make_tensor(
                            tla.recast_ptr(ub_b8, dtype=tla.Float8E4M3FN),
                            tla.make_layout(
                                tla.make_shape(tn, kn_b), tla.make_stride(kn_b, 1)
                            ),
                        )
                        mx_si_b = tla.make_tensor(
                            ub_scale_in,
                            tla.make_layout(
                                tla.make_shape(tn, kn_b // 32),
                                tla.make_stride(kn_b // 32, 1),
                            ),
                        )
                        tla.experimental.mx_to((mx_in_b, mx_si_b), (mx_out_b, mx_si_b))
                        tla.set_flag(b8_ready)
                        tla.wait_flag(b8_ready)
                        l1b_half_b = l1b + kk_b % 2 * (tk * tn) + 1 * (tn * kn_b)
                        t_l1b_b = tla.make_tensor_like(
                            l1b_half_b, t_ub_b8_b, tla.arch.zN
                        )
                        tla.cross_core_wait_flag(l1b_free, tla.arch.MTE3, aiv_id=1)
                        tla.copy(t_l1b_b, t_ub_b8_b)
                        tla.set_flag(ub_free)
                        tla.cross_core_set_flag(l1b_full, tla.arch.MTE3, aiv_id=1)
    # Double-buffered L1 lets the vector core widen the next weight tile
    # while the cube consumes the previous tile.
    with tla.cube():
        tla.cross_core_set_flag(l1b_free, tla.arch.MTE1, aiv_id=0)
        if dual:
            tla.cross_core_set_flag(l1b_free, tla.arch.MTE1, aiv_id=1)
        if credits >= 2:
            tla.cross_core_set_flag(l1b_free, tla.arch.MTE1, aiv_id=0)
            if dual:
                tla.cross_core_set_flag(l1b_free, tla.arch.MTE1, aiv_id=1)
        cblocks = tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_num())
        for block_linear in cblocks:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n
            gm_c_t = tla.tile_view(
                gm_c, tla.make_shape(tm, tn), tla.make_coord(block_row, block_col)
            )
            t_l0c = tla.make_tensor_like(l0c, gm_c_t)
            tla.set_flag(l1a_avail)
            tla.set_flag(l0_avail)
            for kk in tla.range(c0, k_tiles, c1):
                gm_a_t = tla.tile_view(
                    gm_a, tla.make_shape(tm, tk), tla.make_coord(block_row, kk)
                )
                gm_sa_t = tla.tile_view(
                    gm_sa, tla.make_shape(tm, sk), tla.make_coord(block_row, kk)
                )
                gm_sb_t = tla.tile_view(
                    gm_sb, tla.make_shape(sk, tn), tla.make_coord(kk, block_col)
                )
                gm_b_shape = tla.tile_view(
                    gm_b4_shape, tla.make_shape(tk, tn), tla.make_coord(kk, block_col)
                )
                t_l1a = tla.make_tensor_like(l1a, gm_a_t)
                t_l1b = tla.make_tensor_like(
                    tla.recast_ptr(l1b + kk % 2 * (tk * tn), dtype=tla.Float8E4M3FN),
                    gm_b_shape,
                )
                t_l1sa = tla.make_tensor_like(l1sa, gm_sa_t, tla.arch.zZMxScale)
                t_l1sb = tla.make_tensor_like(l1sb, gm_sb_t, tla.arch.nNMxScale)
                tla.wait_flag(l1a_avail)
                tla.copy(t_l1a, gm_a_t)
                tla.copy(t_l1sa, gm_sa_t)
                tla.copy(t_l1sb, gm_sb_t)
                tla.set_flag(a_ready)
                tla.wait_flag(a_ready)
                t_l0a = tla.make_tensor_like(l0a, t_l1a, tla.arch.zN)
                t_l0b = tla.make_tensor_like(l0b, t_l1b, tla.arch.nZ)
                tla.wait_flag(l0_avail)
                tla.cross_core_wait_flag(l1b_full, tla.arch.MTE1, aiv_id=0)
                if dual:
                    tla.cross_core_wait_flag(l1b_full, tla.arch.MTE1, aiv_id=1)
                tla.copy(t_l0a, t_l1a, scale=t_l1sa)
                tla.copy(t_l0b, t_l1b, scale=t_l1sb)
                tla.set_flag(l1a_avail)
                tla.cross_core_set_flag(l1b_free, tla.arch.MTE1, aiv_id=0)
                if dual:
                    tla.cross_core_set_flag(l1b_free, tla.arch.MTE1, aiv_id=1)
                tla.set_flag(l0_ready)
                tla.wait_flag(l0_ready)
                init_c = True if kk == 0 else False  # noqa: SIM210 -- traced TLA control flow.
                unit_flag = 3 if kk == k_tiles - 1 else 2
                tla.mmad_mx(t_l0c, t_l0a, t_l0b, init_c=init_c, unit_flag=unit_flag)
                tla.set_flag(l0_avail)
            tla.wait_flag(l1a_avail)
            tla.wait_flag(l0_avail)
            tla.copy(gm_c_t, t_l0c, tla.params.CopyL0C2DstParams(unit_flag=3))


def main() -> int:
    import torch
    import torch_npu  # noqa: F401 -- register the backend in the device entry

    parser = argparse.ArgumentParser(description="MXFP4 weights and MXFP8 activations")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--block-num", type=int, default=1)
    args = parser.parse_args()
    if args.m % 128 or args.n % 128 or args.k % 256 or args.block_num < 1:
        parser.error(
            "m and n must be multiples of 128, k a multiple of 256, and block-num positive"
        )

    # Reuse the existing MX matmul host reference and quantization helpers.
    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    a = torch.randn(args.m, args.k, dtype=torch.float32) * 0.5
    b = torch.randn(args.k, args.n, dtype=torch.float32) * 0.5
    qa, sa, a_ref = gen_data.quantize_mx_fp8(a, 1, torch.float8_e4m3fn)
    qb, sb, b_ref = gen_data.quantize_mx_fp4(b, 0, "f4e2m1")
    a_npu = qa.contiguous().view(torch.int8).npu()
    b_npu = qb.contiguous().view(torch.int8).npu()
    sa_npu = sa.view(torch.int8).contiguous().npu()
    sb_npu = (
        sb.view(torch.int8)
        .contiguous()
        .transpose(0, 1)
        .contiguous()
        .transpose(0, 1)
        .npu()
    )
    c_npu = torch.zeros(args.m, args.n, dtype=torch.float32, device="npu")
    sk = args.k // MX_SCALE_GROUP_NUM
    tensors = (
        from_dlpack(
            a_npu,
            layout_tag=tla.arch.RowMajor,
            origin_shape=(args.m, args.k),
            element_type=tla.Float8E4M3FN,
        ).mark_layout_dynamic(),
        from_dlpack(
            b_npu, layout_tag=tla.arch.RowMajor, origin_shape=(args.n, args.k // 2)
        ).mark_layout_dynamic(),
        from_dlpack(
            b_npu,
            layout_tag=tla.arch.ColumnMajor,
            origin_shape=(args.k, args.n),
            element_type=tla.Float4E2M1,
        ).mark_layout_dynamic(),
        from_dlpack(
            sa_npu,
            layout_tag=tla.arch.RowMajorMxScaleA,
            origin_shape=(args.m, sk),
            element_type=tla.Float8E8M0,
        ).mark_layout_dynamic(),
        from_dlpack(
            sb_npu,
            layout_tag=tla.arch.ColMajorMxScaleB,
            origin_shape=(sk, args.n),
            element_type=tla.Float8E8M0,
        ).mark_layout_dynamic(),
        from_dlpack(c_npu, layout_tag=tla.arch.RowMajor).mark_layout_dynamic(),
    )
    artifact = tla.compile(
        w4a8_mx_matmul, *tensors, W4A8Tiling(), options="--npu-arch 3510"
    )
    artifact(*tensors, block_num=args.block_num)
    torch.npu.synchronize()
    expected = a_ref @ b_ref
    rel = (c_npu.cpu() - expected).abs().max().item() / max(
        expected.abs().max().item(), 1e-30
    )
    passed = rel < 2e-2
    print(
        f"W4A8 {args.m}x{args.n}x{args.k}: max relative error={rel:.3e}, passed={passed}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
