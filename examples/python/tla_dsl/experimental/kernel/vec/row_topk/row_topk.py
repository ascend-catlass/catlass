# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select row-wise maxima and their leftmost column indices.

One AIV completes each row. The input stays in UB for all selection passes,
including the updates that suppress every occurrence of a selected value.
"""

import catlass.tla as tla

VL_ELE = 64  # FP32 elements per SIMD vector


@tla.kernel
def row_topk_kernel(
    x_addr: tla.Int64,
    val_addr: tla.Int64,
    idx_addr: tla.Int64,
    M: tla.Constexpr[int],
    N: tla.Constexpr[int],
    N_PAD: tla.Constexpr[int],
    TAIL: tla.Constexpr[int],
    K: tla.Constexpr[int],
) -> None:
    vl = 64
    min_value = -3.4028234663852886e38
    missing_index = 3.0e38
    cast_none = tla.params.CastParams()
    gm_x = tla.make_ptr(tla.Float32, x_addr, tla.AddressSpace.gm)
    gm_val = tla.make_ptr(tla.Float32, val_addr, tla.AddressSpace.gm)
    gm_idx = tla.make_ptr(tla.Float32, idx_addr, tla.AddressSpace.gm)

    # The input row and packed outputs have independent lifetimes.
    input_slot = tla.mutex(resource="tk_input", id=0)
    output_slot = tla.mutex(resource="tk_outputs", id=1)

    x_ptr = tla.allocate(N_PAD, tla.Float32, tla.AddressSpace.ub, 256)
    # Pack K results into the leading lanes for one MTE3 copy per output row.
    v_ptr = tla.allocate(vl, tla.Float32, tla.AddressSpace.ub, 256)
    i_ptr = tla.allocate(vl, tla.Float32, tla.AddressSpace.ub, 256)

    # UB rows are padded to a whole 64-lane vector; the GM side stays dense (N).
    row_ub = tla.make_layout(
        tla.make_shape(1, N),
        tla.make_stride(N_PAD, 1),
        origin_shape=tla.make_shape(1, N),
    )
    row_gm = tla.make_layout(
        tla.make_shape(1, N),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(1, N),
    )
    pack_ub = tla.make_layout(
        tla.make_shape(1, K),
        tla.make_stride(1, 1),
        origin_shape=tla.make_shape(1, K),
    )
    pack_gm = tla.make_layout(
        tla.make_shape(1, K),
        tla.make_stride(K, 1),
        origin_shape=tla.make_shape(1, K),
    )
    ub_x = tla.make_tensor(x_ptr, row_ub)
    ub_pv = tla.make_tensor(v_ptr, pack_ub)
    ub_pi = tla.make_tensor(i_ptr, pack_ub)

    n_chunks = N // vl
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    with tla.vector():
        # Assign whole rows to each AIV so all ranks are selected locally.
        for r in tla.range(gid, M, aiv_num):
            row_off = r.to(tla.Int64) * tla.Int64(N)
            out_off = r.to(tla.Int64) * tla.Int64(K)
            gm_row_x = tla.make_tensor(gm_x + row_off, row_gm)
            gm_row_v = tla.make_tensor(gm_val + out_off, pack_gm)
            gm_row_i = tla.make_tensor(gm_idx + out_off, pack_gm)

            input_slot.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_x, gm_row_x)
            input_slot.unlock(pipe=tla.arch.MTE2)

            input_slot.lock(pipe=tla.arch.VECTOR)
            output_slot.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                if tla.const_expr(TAIL != 0):
                    tail_mask, _ = tla.update_mask(TAIL, dtype=tla.Float32)
                else:
                    tail_mask = mask

                lane = tla.arange(base=0, order="increase", dtype=tla.Int32).to(
                    tla.Float32, cast_none, mask
                )
                packed_v = tla.full(0.0, dtype=tla.Float32)
                packed_i = tla.full(0.0, dtype=tla.Float32)

                # Specialize the short rank loop. Keep the column loops dynamic
                # to bound the instruction count and register use for wide rows.
                for kk in tla.range_constexpr(K):
                    # Accumulate full chunks lane-wise before horizontal reduction.
                    # Reduce the tail with its own mask to exclude UB padding.
                    mx = tla.full(min_value, dtype=tla.Float32)
                    for c in tla.range(n_chunks):
                        v1 = tla.tile_view(
                            ub_x, tla.make_shape(1, vl), tla.make_coord(0, c)
                        ).load()
                        mx = tla.max(mx, v1, mask=mask)
                    tmx = tla.full(min_value, dtype=tla.Float32)
                    if tla.const_expr(TAIL != 0):
                        tv = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, vl),
                            tla.make_coord(0, n_chunks),
                        ).load()
                        tmx = tla.max(tmx, tv, mask=tail_mask)
                    max_b = tla.max(
                        tla.full(
                            mx.reduce(tla.ReductionOp.MAX, mask=mask),
                            dtype=tla.Float32,
                        ),
                        tla.full(
                            tmx.reduce(tla.ReductionOp.MAX, mask=tail_mask),
                            dtype=tla.Float32,
                        ),
                        mask=mask,
                    )

                    # Select the smallest column index that matches the maximum.
                    pos = tla.full(missing_index, dtype=tla.Float32)
                    for cp in tla.range(n_chunks):
                        v2 = tla.tile_view(
                            ub_x, tla.make_shape(1, vl), tla.make_coord(0, cp)
                        ).load()
                        lc2 = tla.arange(
                            base=cp * vl, order="increase", dtype=tla.Int32
                        ).to(tla.Float32, cast_none, mask)
                        pos = tla.min(
                            pos,
                            tla.where(
                                tla.cmp(v2, max_b, "eq"),
                                lc2,
                                tla.full(missing_index, dtype=tla.Float32),
                            ),
                            mask=mask,
                        )
                    tpos = tla.full(missing_index, dtype=tla.Float32)
                    if tla.const_expr(TAIL != 0):
                        tv = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, vl),
                            tla.make_coord(0, n_chunks),
                        ).load()
                        tlc = tla.add(
                            lane,
                            tla.full(float(n_chunks * vl), dtype=tla.Float32),
                            mask=tail_mask,
                        )
                        tpos = tla.min(
                            tpos,
                            tla.where(
                                tla.cmp(tv, max_b, "eq"),
                                tlc,
                                tla.full(missing_index, dtype=tla.Float32),
                            ),
                            mask=tail_mask,
                        )
                    idx_b = tla.min(
                        tla.full(
                            pos.reduce(tla.ReductionOp.MIN, mask=mask),
                            dtype=tla.Float32,
                        ),
                        tla.full(
                            tpos.reduce(tla.ReductionOp.MIN, mask=tail_mask),
                            dtype=tla.Float32,
                        ),
                        mask=mask,
                    )

                    # Pack the k-th result into lane kk of the staging vectors.
                    sel = tla.cmp(lane, kk, "eq")
                    packed_v = tla.where(
                        sel,
                        tla.full(
                            max_b.reduce(tla.ReductionOp.MAX, mask=mask),
                            dtype=tla.Float32,
                        ),
                        packed_v,
                    )
                    packed_i = tla.where(
                        sel,
                        tla.full(
                            idx_b.reduce(tla.ReductionOp.MIN, mask=mask),
                            dtype=tla.Float32,
                        ),
                        packed_i,
                    )

                    # Suppress all matching values in UB for the next rank.
                    for cm in tla.range(n_chunks):
                        xc = tla.tile_view(
                            ub_x, tla.make_shape(1, vl), tla.make_coord(0, cm)
                        )
                        v3 = xc.load()
                        xc.store(
                            tla.where(
                                tla.cmp(v3, max_b, "eq"),
                                tla.full(min_value, dtype=tla.Float32),
                                v3,
                            ),
                            mask=mask,
                        )
                    if tla.const_expr(TAIL != 0):
                        xt = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, vl),
                            tla.make_coord(0, n_chunks),
                        )
                        v3t = xt.load()
                        xt.store(
                            tla.where(
                                tla.cmp(v3t, max_b, "eq"),
                                tla.full(min_value, dtype=tla.Float32),
                                v3t,
                            ),
                            mask=tail_mask,
                        )
                    # The next rank reads the same UB row after the vector stores.
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )

                tla.tile_view(ub_pv, tla.make_shape(1, vl), tla.make_coord(0, 0)).store(
                    packed_v
                )
                tla.tile_view(ub_pi, tla.make_shape(1, vl), tla.make_coord(0, 0)).store(
                    packed_i
                )

            output_slot.unlock(pipe=tla.arch.VECTOR)
            input_slot.unlock(pipe=tla.arch.VECTOR)

            # MTE3 holds the result slot while the next row can load into UB.
            output_slot.lock(pipe=tla.arch.MTE3)
            tla.copy(gm_row_v, ub_pv)
            tla.copy(gm_row_i, ub_pi)
            output_slot.unlock(pipe=tla.arch.MTE3)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================

_COMPILED = {}
_K_VALUES = (1, 2, 4, 8)
# Arch 3510 has 248 KiB of UB after compiler reservations. Two vectors hold the
# packed outputs and the remainder fits the row rounded up to VL_ELE.
_MAX_COLUMNS = 248 * 1024 // 4 - 2 * VL_ELE


def _compile_row_topk(m, n, k):
    n_pad = (n + VL_ELE - 1) // VL_ELE * VL_ELE
    # Address arguments are rebound on every call; none of the tensors is cached.
    placeholder = tla.Int64(0)
    return tla.compile(
        row_topk_kernel,
        placeholder,
        placeholder,
        placeholder,
        m,
        n,
        n_pad,
        n % VL_ELE,
        k,
        options="--npu-arch 3510",
    )


def run(x, k=1):
    """Select row-wise maxima and return their leftmost column indices."""
    import torch

    if x.dtype != torch.float32:
        raise ValueError("x must be float32")
    if x.dim() != 2:
        raise ValueError("x must be 2-D [M, N]")
    if type(k) is not int or k not in _K_VALUES:
        raise ValueError("k must be one of {}".format(_K_VALUES))
    m, n = x.shape
    if not VL_ELE <= n <= _MAX_COLUMNS:
        raise ValueError(f"N must be between {VL_ELE} and {_MAX_COLUMNS}, got {n}")
    if m == 0:
        return x.new_empty((0, k)), x.new_empty((0, k), dtype=torch.int32)

    device = x.device
    if device.type != "npu":
        raise ValueError("x must be on an Ascend NPU")

    key = (device, m, n, k)
    with torch.npu.device(device):
        x_contiguous = x.contiguous()
        val = x.new_empty((m, k))
        idx_f = x.new_empty((m, k))
        plan = _COMPILED.get(key)
        if plan is None:
            vector_cores = int(torch.npu.get_device_properties(device).vector_core_num)
            artifact = _compile_row_topk(m, n, k)
            plan = (artifact, vector_cores)
            _COMPILED[key] = plan

        artifact, vector_cores = plan
        stream = torch.npu.current_stream(device)
        # The asynchronous kernel may still be reading the input after return.
        x_contiguous.record_stream(stream)
        artifact(
            tla.Int64(x_contiguous.data_ptr()),
            tla.Int64(val.data_ptr()),
            tla.Int64(idx_f.data_ptr()),
            block_num=vector_cores,
            stream=stream.npu_stream,
        )
        return val, idx_f.to(torch.int32)


if __name__ == "__main__":
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.manual_seed(0)

    for m, n, k in ((64, 512, 1), (37, 1000, 4), (16, 2048, 8)):
        x = torch.randn(m, n, dtype=torch.float32, device="npu")
        val, idx = run(x, k)
        ref_v, ref_i = torch.topk(x.cpu(), k, dim=-1)
        err_v = (val.cpu() - ref_v).abs().max().item()
        # Compare the values gathered back with the reported indices as well, so
        # a wrong index cannot hide behind a matching value.
        gathered = x.cpu().gather(1, idx.cpu().to(torch.int64))
        err_g = (gathered - ref_v).abs().max().item()
        err_i = (idx.cpu() - ref_i.to(torch.int32)).abs().max().item()
        print(
            "shape",
            (m, n),
            "k",
            k,
            "val err",
            err_v,
            "gather err",
            err_g,
            "idx err",
            err_i,
        )
