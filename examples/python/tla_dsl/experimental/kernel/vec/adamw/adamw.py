# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run one AdamW step with FP32 moments and FP32, FP16, or BF16 parameters.

The parameter and both moment updates share one AIV pass over each tile.
"""

import catlass.tla as tla


@tla.kernel
def adamw_kernel(
    addr_p: tla.Int64,
    addr_g: tla.Int64,
    addr_m: tla.Int64,
    addr_v: tla.Int64,
    addr_po: tla.Int64,
    addr_mo: tla.Int64,
    addr_vo: tla.Int64,
    ELEMENTS: tla.Constexpr[int],
    PARAM_DTYPE: tla.Constexpr[type],
    B1: tla.Constexpr[float],
    B2: tla.Constexpr[float],
    BC1: tla.Constexpr[float],
    BC2: tla.Constexpr[float],
    LR: tla.Constexpr[float],
    EPS: tla.Constexpr[float],
    WD: tla.Constexpr[float],
) -> None:
    tile_elems = 1024
    vector_elems = 64
    full_tiles = ELEMENTS // tile_elems
    tail_elems = ELEMENTS % tile_elems
    gm_p = tla.make_ptr(PARAM_DTYPE, addr_p)
    gm_g = tla.make_ptr(PARAM_DTYPE, addr_g)
    gm_m = tla.make_ptr(tla.Float32, addr_m)
    gm_v = tla.make_ptr(tla.Float32, addr_v)
    gm_po = tla.make_ptr(PARAM_DTYPE, addr_po)
    gm_mo = tla.make_ptr(tla.Float32, addr_mo)
    gm_vo = tla.make_ptr(tla.Float32, addr_vo)
    vector_shape = tla.make_shape(vector_elems)
    cast_wide = tla.params.CastParams()
    cast_narrow = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    unpack = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_UNPACK_B16)
    pack = tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32)

    # The parameter and moments reuse their input buffers for writeback.
    # Keep the full UB slot until MTE3 finishes reading the updated tile.
    tile_slot = tla.mutex(resource="aw_tile", id=0)
    p_ptr = tla.allocate(tile_elems, PARAM_DTYPE, tla.AddressSpace.ub, 256)
    g_ptr = tla.allocate(tile_elems, PARAM_DTYPE, tla.AddressSpace.ub, 256)
    m_ptr = tla.allocate(tile_elems, tla.Float32, tla.AddressSpace.ub, 256)
    v_ptr = tla.allocate(tile_elems, tla.Float32, tla.AddressSpace.ub, 256)
    gid = tla.arch.block_idx()
    cores = tla.arch.block_num()

    with tla.vector():
        # Assign tiles cyclically across AIV cores. Specialize the tail so
        # its copies use the real length and need no padding in GM.
        for part in tla.range_constexpr(2):
            if tla.const_expr(part == 0):
                valid = tile_elems
                start = gid
                stop = full_tiles
            else:
                valid = tail_elems
                start = full_tiles + (gid + cores - full_tiles % cores) % cores
                stop = full_tiles + 1

            if tla.const_expr(valid > 0):
                layout = tla.make_layout(tla.make_shape(valid), tla.make_stride(1))
                ub_p = tla.make_tensor(p_ptr, layout)
                ub_g = tla.make_tensor(g_ptr, layout)
                ub_m = tla.make_tensor(m_ptr, layout)
                ub_v = tla.make_tensor(v_ptr, layout)
                chunks = (valid + vector_elems - 1) // vector_elems

                for tile in tla.range(start, stop, cores):
                    off = tile.to(tla.Int64) * tla.Int64(tile_elems)
                    gp = tla.make_tensor(gm_p + off, layout)
                    gg = tla.make_tensor(gm_g + off, layout)
                    gm = tla.make_tensor(gm_m + off, layout)
                    gv = tla.make_tensor(gm_v + off, layout)
                    gpo = tla.make_tensor(gm_po + off, layout)
                    gmo = tla.make_tensor(gm_mo + off, layout)
                    gvo = tla.make_tensor(gm_vo + off, layout)

                    tile_slot.lock(pipe=tla.pipes.MTE2)
                    tla.copy(ub_p, gp)
                    tla.copy(ub_g, gg)
                    tla.copy(ub_m, gm)
                    tla.copy(ub_v, gv)
                    tile_slot.unlock(pipe=tla.pipes.MTE2)

                    tile_slot.lock(pipe=tla.pipes.VECTOR)
                    with tla.vec.func(mode="simd"):
                        full_mask = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        for chunk in tla.range_constexpr(chunks):
                            coord = tla.make_coord(chunk)
                            pc = tla.tile_view(ub_p, vector_shape, coord)
                            gc = tla.tile_view(ub_g, vector_shape, coord)
                            mc = tla.tile_view(ub_m, vector_shape, coord)
                            vc = tla.tile_view(ub_v, vector_shape, coord)
                            if tla.const_expr((chunk + 1) * vector_elems > valid):
                                mask, _ = tla.update_mask(
                                    valid - chunk * vector_elems, dtype=tla.Float32
                                )
                            else:
                                mask = full_mask

                            # Unpack 16-bit lanes for FP32 arithmetic. The
                            # parameter is narrowed once when storing it.
                            if tla.const_expr(PARAM_DTYPE is tla.Float32):
                                p = pc.load()
                                g = gc.load()
                            else:
                                p = pc.load(unpack).to(tla.Float32, cast_wide)
                                g = gc.load(unpack).to(tla.Float32, cast_wide)
                            mom = mc.load()
                            vel = vc.load()

                            mom = tla.add(
                                tla.mul(mom, B1, mask=mask),
                                tla.mul(g, 1.0 - B1, mask=mask),
                                mask=mask,
                            )
                            vel = tla.add(
                                tla.mul(vel, B2, mask=mask),
                                tla.mul(tla.mul(g, g, mask=mask), 1.0 - B2, mask=mask),
                                mask=mask,
                            )
                            # Reuse the updated moments and apply the bias
                            # correction as compile-time reciprocal factors.
                            mhat = tla.mul(mom, 1.0 / BC1, mask=mask)
                            vhat = tla.mul(vel, 1.0 / BC2, mask=mask)
                            update = tla.div(
                                mhat,
                                tla.add(tla.sqrt(vhat, mask=mask), EPS, mask=mask),
                                mask=mask,
                            )
                            p = tla.sub(
                                p,
                                tla.mul(
                                    tla.add(
                                        update, tla.mul(p, WD, mask=mask), mask=mask
                                    ),
                                    LR,
                                    mask=mask,
                                ),
                                mask=mask,
                            )
                            if tla.const_expr(PARAM_DTYPE is tla.Float32):
                                pc.store(p)
                            else:
                                pc.store(p.to(PARAM_DTYPE, cast_narrow), pack)
                            mc.store(mom)
                            vc.store(vel)

                    tile_slot.unlock(pipe=tla.pipes.VECTOR)

                    tile_slot.lock(pipe=tla.pipes.MTE3)
                    tla.copy(gpo, ub_p)
                    tla.copy(gmo, ub_m)
                    tla.copy(gvo, ub_v)
                    tile_slot.unlock(pipe=tla.pipes.MTE3)


# ======================================================================================
# HOST side: input checks, compile and launch.
# ======================================================================================

_COMPILED = {}


def run(
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.01,
    step=1,
):
    """Return new ``(param, exp_avg, exp_avg_sq)`` tensors after one AdamW step."""
    import torch

    if any(tensor.shape != param.shape for tensor in (grad, exp_avg, exp_avg_sq)):
        raise ValueError("param/grad/exp_avg/exp_avg_sq must share a shape")
    device = param.device
    if any(tensor.device != device for tensor in (grad, exp_avg, exp_avg_sq)):
        raise ValueError("param/grad/exp_avg/exp_avg_sq must share a device")
    if (
        param.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or grad.dtype != param.dtype
    ):
        raise ValueError("param and grad must share a float32/float16/bfloat16 dtype")
    if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
        raise ValueError("exp_avg / exp_avg_sq must be float32")

    numel = param.numel()
    if numel == 0:
        return (
            torch.empty_like(param),
            torch.empty_like(exp_avg),
            torch.empty_like(exp_avg_sq),
        )
    if device.type != "npu":
        raise ValueError("inputs must be on an Ascend NPU")

    coefficients = (
        float(beta1),
        float(beta2),
        float(1.0 - beta1**step),
        float(1.0 - beta2**step),
        float(lr),
        float(eps),
        float(weight_decay),
    )
    key = (device, numel, param.dtype, *coefficients)
    with torch.npu.device(device):
        inputs = (param, grad, exp_avg, exp_avg_sq)
        p, g, em, ev = (tensor.reshape(-1).contiguous() for tensor in inputs)
        po = torch.empty_like(p)
        mo = torch.empty_like(em)
        vo = torch.empty_like(ev)
        # Use current device addresses with the cached shape specialization.
        addresses = tuple(tla.Int64(t.data_ptr()) for t in (p, g, em, ev, po, mo, vo))
        plan = _COMPILED.get(key)
        if plan is None:
            vector_cores = int(torch.npu.get_device_properties(device).vector_core_num)
            dtype = {
                torch.float32: tla.Float32,
                torch.float16: tla.Float16,
                torch.bfloat16: tla.BFloat16,
            }[param.dtype]
            artifact = tla.compile(
                adamw_kernel,
                *addresses,
                numel,
                dtype,
                *coefficients,
                options="--npu-arch 3510",
            )
            plan = (artifact, vector_cores)
            _COMPILED[key] = plan

        artifact, vector_cores = plan
        # The executor uses this device's current PyTorch stream. Keep all
        # seven tensors alive here while their addresses are submitted.
        artifact(*addresses, block_num=vector_cores)

    shape = param.shape
    return po.view(shape), mo.view(shape), vo.view(shape)


if __name__ == "__main__":
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.manual_seed(0)

    def ref_adamw(p, g, m, v, lr, b1, b2, eps, wd, step):
        p = p.float()
        m = m.float() * b1 + (1 - b1) * g.float()
        v = v.float() * b2 + (1 - b2) * g.float() * g.float()
        bc1 = 1 - b1**step
        bc2 = 1 - b2**step
        p = p - lr * ((m / bc1) / ((v / bc2).sqrt() + eps) + wd * p)
        return p, m, v

    lr, b1, b2, eps, wd, step = 1e-2, 0.9, 0.999, 1e-8, 0.01, 5
    tol = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        p = torch.randn(37, 1000, dtype=dt, device="npu")
        g = torch.randn(37, 1000, dtype=dt, device="npu")
        m = torch.rand(37, 1000, dtype=torch.float32, device="npu")
        v = torch.rand(37, 1000, dtype=torch.float32, device="npu")
        po, mo, vo = run(p, g, m, v, lr, b1, b2, eps, wd, step)
        rp, rm, rv = ref_adamw(
            p.cpu(), g.cpu(), m.cpu(), v.cpu(), lr, b1, b2, eps, wd, step
        )
        ep = (po.cpu().float() - rp).abs().max().item()
        em = (mo.cpu() - rm).abs().max().item()
        ev = (vo.cpu() - rv).abs().max().item()
        print("adamw", str(dt).split(".")[-1], "param", ep, "m", em, "v", ev)
        if not (ep <= tol[dt] + 2e-4 and em <= 2e-5 and ev <= 2e-5):
            raise RuntimeError(f"AdamW accuracy check failed for {dt}")
