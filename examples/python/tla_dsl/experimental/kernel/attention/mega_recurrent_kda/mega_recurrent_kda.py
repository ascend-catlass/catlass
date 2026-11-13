# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend 950 forward operator. Standard canonical-Conv path launches once."""

import math
import torch
import torch_npu
import catlass.tla as tla
from .kernel.mega_recurrent_kda import mega_kernel

NZ_WEIGHTS = (1, 2, 3, 4, 5, 7)


def _validate(tensors, scale, lower, eps):
    x, wq, wa, wb, wbe, wz, gamma, wo, cw, cs, rs, ci, si, ca, sa, alog, bias = tensors
    if x.ndim != 3 or tuple(x.shape[1:]) != (8, 7168) or wbe.ndim != 2:
        raise ValueError("requires hidden [B,8,7168] and beta weight [N,7168]")
    B = x.shape[0]
    N = wbe.shape[0]
    P = N * 128
    if N not in (6, 12, 24) or B % (96 // N) or not 96 // N <= B <= 32 * (96 // N):
        raise ValueError("requires TP4/8/16 and model batch 1..32")
    shapes = {
        1: (3 * P, 7168),
        2: (128, 7168),
        3: (P, 128),
        4: (N, 7168),
        5: (P, 7168),
        6: (128,),
        7: (7168, P),
        11: (B,),
        12: (B * 8,),
        13: (B,),
        14: (B,),
        15: (N,),
        16: (N, 128),
    }
    for i, shape in shapes.items():
        if tuple(tensors[i].shape) != shape:
            raise ValueError(f"argument {i} requires shape {shape}")
    bf16 = {0, 1, 2, 3, 4, 5, 7, 8, 9}
    fp32 = {6, 10, 15, 16}
    for i, t in enumerate(tensors):
        dtype = (
            torch.bfloat16 if i in bf16 else torch.float32 if i in fp32 else torch.int32
        )
        if (
            t.device.type != "npu"
            or t.device != x.device
            or t.dtype != dtype
            or not t.is_contiguous()
        ):
            raise ValueError(f"argument {i} requires contiguous {dtype} on {x.device}")
        fmt = torch_npu.get_npu_format(t)
        if i in NZ_WEIGHTS:
            if (
                fmt != 29
                or t._base is not None
                or t.storage_offset() != 0
                or isinstance(t, torch.nn.Parameter)
            ):
                raise ValueError(
                    f"argument {i} requires a detached FRACTAL_NZ allocation root"
                )
        elif fmt not in (0, 2):
            raise ValueError(f"argument {i} requires ND storage")
    if tuple(cw.shape) not in ((3, N, 4, 128), (4, 3 * P), (3, 4, 128)):
        raise ValueError("invalid Conv weight layout")
    if cs.ndim != 3 or cs.shape[0] < 1 or cs.shape[1] < 10 or cs.shape[2] != 3 * P:
        raise ValueError("invalid Conv state pool")
    if rs.ndim != 4 or tuple(rs.shape[1:]) != (N, 128, 128) or rs.shape[0] < B * 8:
        raise ValueError("invalid recurrent state pool")
    if not math.isfinite(float(scale)) or not -5 <= float(lower) <= 0:
        raise ValueError("invalid scale or lower_bound")
    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("rms_norm_eps must be positive and finite")
    return B, N, P


def _bind(t, shape):
    return tla.from_dlpack(t, layout_tag=tla.arch.RowMajor, origin_shape=shape)


def _prepare(
    tensors,
    scale,
    lower,
    eps,
    initial_n=128,
    pipelined=False,
    output_n=128,
    dual_m=False,
    output_resident=False,
):
    B, N, P = _validate(tensors, scale, lower, eps)
    M = B * 8
    NB = ((N + 15) // 16) * 16
    data = list(tensors)
    # Optional compatibility preparation; canonical [3,N,4,128] is zero-copy.
    if data[8].ndim == 2:
        data[8] = data[8].view(4, 3, N, 128).permute(1, 2, 0, 3).contiguous()
    elif data[8].ndim == 3:
        data[8] = data[8][:, None].expand(3, N, 4, 128).contiguous()
    L = data[9].shape[1]
    counts = [M * 3 * P, M * 128, M * P, M * NB, M * P, M * P]
    padded = [(v + 255) // 256 * 256 for v in counts]
    workspace = torch.empty(sum(padded), dtype=torch.bfloat16, device=data[0].device)
    views = []
    offset = 0
    shapes = [(M, 3 * P), (M, 128), (M, P), (M, NB), (M, P), (M, P)]
    for count, size, shape in zip(counts, padded, shapes):
        views.append(workspace[offset : offset + count].view(shape))
        offset += size
    output = torch.empty((B, 8, 7168), dtype=torch.bfloat16, device=data[0].device)
    argshapes = [
        (M, 7168),
        (3 * P, 7168),
        (128, 7168),
        (P, 128),
        (NB, 7168),
        (P, 7168),
        (128,),
        (7168, P),
        (3 * N * 4, 128),
        (data[9].shape[0] * L, 3 * P),
        (data[10].shape[0] * N * 128, 128),
        (B,),
        (M,),
        (B,),
        (B,),
        (N,),
        (N * 128,),
    ]
    args = tuple(_bind(t, s) for t, s in zip(data, argshapes))
    args += tuple(_bind(t, s) for t, s in zip(views, shapes)) + (
        _bind(output, (M, 7168)),
    )
    artifact = tla.compile(
        mega_kernel,
        *args,
        N,
        L,
        min(M, 128),
        NB,
        initial_n,
        bool(pipelined),
        output_n,
        bool(dual_m),
        bool(output_resident),
        float(scale),
        float(lower),
        float(eps),
        options="--npu-arch 3510",
    )
    return artifact, args, output, workspace, views, data


@torch.no_grad()
def mega_recurrent_kda(
    hidden_states,
    qkv_projection_weight,
    decay_projection_a_weight,
    decay_projection_b_weight,
    beta_projection_weight,
    output_gate_projection_weight,
    output_norm_weight,
    output_projection_weight,
    conv1d_weight,
    conv_state,
    recurrent_state,
    conv_state_indices,
    ssm_state_indices,
    conv_num_accepted_tokens,
    num_accepted_tokens,
    A_log,
    dt_bias,
    scale,
    lower_bound,
    rms_norm_eps=1e-5,
):
    tensors = (
        hidden_states,
        qkv_projection_weight,
        decay_projection_a_weight,
        decay_projection_b_weight,
        beta_projection_weight,
        output_gate_projection_weight,
        output_norm_weight,
        output_projection_weight,
        conv1d_weight,
        conv_state,
        recurrent_state,
        conv_state_indices,
        ssm_state_indices,
        conv_num_accepted_tokens,
        num_accepted_tokens,
        A_log,
        dt_bias,
    )
    stream = torch.npu.current_stream(hidden_states.device)
    quota = torch.npu.get_stream_limit(stream)
    cores = min(32, quota["cube_core_num"])
    if cores not in (8, 16, 24, 28, 32) or quota["vector_core_num"] < 2 * cores:
        raise ValueError("requires 8/16/24/28/32 AIC and at least twice as many AIV")
    # Retain every allocation in the prepared tuple until the launch completes.
    prepared = _prepare(
        tensors,
        scale,
        lower_bound,
        rms_norm_eps,
        initial_n=384
        if cores in (8, 16, 24, 28) and hidden_states.shape[0] * 8 >= 256
        else 128,
        pipelined=cores in (8, 16, 24, 28) and hidden_states.shape[0] * 8 >= 256,
        output_n=256 if cores == 28 and hidden_states.shape[0] * 8 >= 256 else 128,
        dual_m=cores == 28
        and A_log.numel() == 6
        and hidden_states.shape[0] * 8 >= 2048,
        output_resident=cores == 28
        and A_log.numel() == 6
        and hidden_states.shape[0] * 8 >= 256,
    )
    artifact, args, out = prepared[:3]
    artifact(*args, block_num=cores, stream=stream.npu_stream)
    return out
