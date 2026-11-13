# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public GDN2 forward interface, validation, allocation and compilation cache."""

import math
import numbers
from collections import OrderedDict
from threading import Lock

import torch
import torch_npu  # noqa: F401
from catlass import tla
from .gdn2_fwd import (
    prepare_kernel,
    recurrent_kernel,
)


_WORKSPACE_LAYOUT = (
    ((64, 128, torch.bfloat16),) * 3
    + ((1, 128, torch.float32),)
    + ((64, 64, torch.float32),) * 2
    + ((64, 128, torch.bfloat16),) * 2
)

_ARTIFACT_CACHE = OrderedDict()

_ARTIFACT_CACHE_LIMIT = 32

_ARTIFACT_CACHE_LOCK = Lock()


def _compiled_artifact(kernel, key, bindings, compile_args):
    # Only executable artifacts are retained. Bindings are recreated for every
    # call, including a cache hit, so input addresses never become cache state.
    with _ARTIFACT_CACHE_LOCK:
        artifact = _ARTIFACT_CACHE.get(key)
        if artifact is None:
            artifact = tla.compile(
                kernel, *bindings, *compile_args, options="--npu-arch 3510"
            )
            _ARTIFACT_CACHE[key] = artifact
            if len(_ARTIFACT_CACHE) > _ARTIFACT_CACHE_LIMIT:
                _ARTIFACT_CACHE.popitem(last=False)
        else:
            _ARTIFACT_CACHE.move_to_end(key)
        return artifact


def _workspace_bytes(factors):
    return factors * sum(
        rows * cols * dtype.itemsize for rows, cols, dtype in _WORKSPACE_LAYOUT
    )


def validate_inputs(
    q,
    k,
    v,
    g,
    b,
    w_gate,
    scale,
    initial_state,
    output_final_state,
    chunk_size,
    *,
    device_type="cpu",
):
    tensors = (q, k, v, g, b, w_gate)
    if any(not isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("q, k, v, g, b, w_gate must be tensors")
    if not isinstance(output_final_state, bool):
        raise TypeError("output_final_state must be bool")
    if not isinstance(chunk_size, int) or chunk_size not in (64, 128):
        raise ValueError("chunk_size must be 64 or 128")
    if any(x.ndim != 4 for x in tensors):
        raise ValueError("all inputs must have rank 4 in BNSD layout")
    batch, hk, length, dim = q.shape
    hv = v.shape[1]
    if batch < 1 or length < 1 or hk < 1 or not 1 <= hv <= 96:
        raise ValueError("B, T, HK must be positive and 1 <= HV <= 96")
    if hv % hk or hv // hk not in (1, 2, 4):
        raise ValueError("HV / HK must be 1, 2 or 4")
    if dim != 128 or k.shape != q.shape or v.shape != (batch, hv, length, 128):
        raise ValueError("invalid Q/K/V shape; K=V=128 is required")
    for name, x in (("g", g), ("b", b)):
        if x.shape[1] not in (hk, hv) or x.shape != (batch, x.shape[1], length, 128):
            raise ValueError(f"{name} must have BNSD shape with HK or HV heads")
    if w_gate.shape != v.shape:
        raise ValueError("w_gate shape must equal v shape")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("q, k and v must be BF16")
    if any(x.dtype not in (torch.bfloat16, torch.float32) for x in (g, b, w_gate)):
        raise TypeError("g, b and w_gate must be BF16 or FP32")
    if initial_state is not None:
        if (
            not isinstance(initial_state, torch.Tensor)
            or initial_state.dtype != torch.float32
        ):
            raise TypeError("initial_state must be an FP32 tensor or None")
        if initial_state.shape != (batch, hv, 128, 128):
            raise ValueError("initial_state must have shape [B, HV, K, V]")
        tensors += (initial_state,)
    if any(x.device.type != device_type or x.device != q.device for x in tensors):
        raise ValueError(f"all tensors must be on the same {device_type} device")
    if any(not x.is_contiguous() for x in tensors):
        raise ValueError("all tensors must be contiguous")
    if scale is None:
        scale = 128**-0.5
    if (
        isinstance(scale, bool)
        or not isinstance(scale, numbers.Real)
        or not math.isfinite(scale)
    ):
        raise ValueError("scale must be a finite real scalar or None")
    return float(scale)


def gdn2_fwd(
    q,
    k,
    v,
    g,
    b,
    w_gate,
    scale=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
):
    scale = validate_inputs(
        q,
        k,
        v,
        g,
        b,
        w_gate,
        scale,
        initial_state,
        output_final_state,
        chunk_size,
        device_type="npu",
    )
    batch, hk, length, _ = q.shape
    hv = v.shape[1]
    hf = max(hk, g.shape[1], b.shape[1])
    factors = batch * hf * ((length + 63) // 64)
    # Both supported logical chunks are composed of identical 64-token work units.
    # Use explicit 64-bit element offsets in the device and reject overflow here.
    if any(
        x.numel() > ((1 << 63) - 1) // x.element_size() for x in (q, k, v, g, b, w_gate)
    ):
        raise OverflowError("input byte count exceeds signed 64-bit addressing")
    if (
        _workspace_bytes(factors) > (1 << 63) - 1
        or batch * hv * 128 * 128 * 4 > (1 << 63) - 1
    ):
        raise OverflowError(
            "workspace or state byte count exceeds signed 64-bit addressing"
        )
    with torch.npu.device(q.device):
        cores = torch.npu.get_device_properties(q.device).cube_core_num
        full_heads = batch * hv // cores * cores
        groups = full_heads + 4 * (batch * hv - full_heads)
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        final_state = (
            torch.empty((batch, hv, 128, 128), dtype=torch.float32, device=q.device)
            if output_final_state
            else None
        )
        dummy = (
            torch.empty((1, 128), dtype=torch.float32, device=q.device)
            if initial_state is None or final_state is None
            else None
        )
        state_input = initial_state if initial_state is not None else dummy
        state_output = final_state if final_state is not None else dummy
        # Qg, Kt high/low, end, Aq, inverse, negative-W high/low.
        workspace = [
            torch.empty((factors * rows, cols), dtype=dtype, device=q.device)
            for rows, cols, dtype in _WORKSPACE_LAYOUT
        ]
        prep_inputs = [x.detach().reshape(-1, 128) for x in (q, k, g, b)] + workspace
        state_inputs = (
            [x.detach().reshape(-1, 128) for x in (v, w_gate)]
            + workspace
            + [x.detach().reshape(-1, 128) for x in (state_input, out, state_output)]
        )
        layout = tla.arch.RowMajor
        prep_bindings = [tla.from_dlpack(x, layout_tag=layout) for x in prep_inputs]
        state_bindings = [tla.from_dlpack(x, layout_tag=layout) for x in state_inputs]
        prep_args = (
            length,
            hk,
            hf,
            g.shape[1],
            b.shape[1],
            factors,
            g.dtype == torch.bfloat16,
            b.dtype == torch.bfloat16,
        )
        state_args = (
            length,
            hk,
            hv,
            hf,
            groups,
            full_heads,
            w_gate.dtype == torch.bfloat16,
            initial_state is not None,
            output_final_state,
            scale,
        )
        prep_key = (
            "prepare_paired",
            q.device.index,
            batch,
            hk,
            hf,
            length,
            g.shape[1],
            b.shape[1],
            g.dtype,
            b.dtype,
        )
        state_key = (
            "recurrent_v32",
            cores,
            full_heads,
            groups,
            q.device.index,
            batch,
            hk,
            hv,
            hf,
            length,
            w_gate.dtype,
            initial_state is not None,
            output_final_state,
            scale.hex(),
        )
        # Resolve both artifacts and all bindings before either launch. No Host
        # compilation/binding work is inserted between the two device stages.
        prep_artifact = _compiled_artifact(
            prepare_kernel, prep_key, prep_bindings, prep_args
        )
        state_artifact = _compiled_artifact(
            recurrent_kernel, state_key, state_bindings, state_args
        )
        prep_artifact(*prep_bindings, block_num=min(cores, factors))
        state_artifact(*state_bindings, block_num=min(cores, groups))
    return out, final_state


__all__ = ["gdn2_fwd"]
