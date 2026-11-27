# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Split a 2-D tensor into equal contiguous parts along the last dimension.

All parts are materialized into one packed ``[parts, M, W]`` output buffer. The
work is flattened across ``(part, row_group)`` and each AIV block strides over
that domain. Two independent UB slots, protected by separate mutexes, overlap
the next MTE2 load with the previous MTE3 write-back.
"""

import catlass.tla as tla

# Total staging budget for both ping-pong slots.
STAGE_ELEMS = 49152


@tla.kernel
def split_kernel(
    gm_src: tla.Tensor,  # [M, N] fp32
    gm_dst: tla.Tensor,  # [P, M, W] fp32
    M: tla.Constexpr[int],
    N: tla.Constexpr[int],
    W: tla.Constexpr[int],
    PAD_W: tla.Constexpr[int],
    P: tla.Constexpr[int],
    BR: tla.Constexpr[int],
) -> None:
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    # Each physical staging slot owns one mutex. Sharing a mutex across slots
    # would merge unrelated lifetimes and serialize the copy pipeline.
    mutex_ub0 = tla.mutex(resource="split_ub0", id=0)
    mutex_ub1 = tla.mutex(resource="split_ub1", id=1)

    ub0_ptr = tla.allocate(BR * PAD_W, tla.Float32, tla.AddressSpace.ub, 256)
    ub1_ptr = tla.allocate(BR * PAD_W, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ptrs = (ub0_ptr, ub1_ptr)
    mutexes = (mutex_ub0, mutex_ub1)
    ub_lay = tla.make_layout(
        tla.make_shape(BR, W),
        tla.make_stride(PAD_W, 1),
        origin_shape=tla.make_shape(BR, W),
    )
    src_lay = tla.make_layout(
        tla.make_shape(BR, W),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(BR, W),
    )
    dst_lay = tla.make_layout(
        tla.make_shape(BR, W),
        tla.make_stride(W, 1),
        origin_shape=tla.make_shape(BR, W),
    )

    row_lay = tla.make_layout(
        tla.make_shape(1, W),
        tla.make_stride(PAD_W, 1),
        origin_shape=tla.make_shape(1, W),
    )
    row_dst_lay = tla.make_layout(
        tla.make_shape(1, W),
        tla.make_stride(W, 1),
        origin_shape=tla.make_shape(1, W),
    )
    row_src_lay = tla.make_layout(
        tla.make_shape(1, W),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(1, W),
    )

    n_groups = M // BR
    tail = M % BR
    part_elems = M * W
    main_tasks = P * n_groups
    tail_tasks = P * tail

    with tla.vector():
        # The runtime loop keeps code size independent of `parts`. Flattening
        # all parts into one task domain avoids serializing the outer part loop.
        # Each iteration explicitly expands both ping-pong slots, so buffer and
        # mutex selection is resolved at compile time.
        pair_stride = 2 * aiv_num
        if tla.const_expr(n_groups > 0):
            for base in tla.range(gid, main_tasks, pair_stride):
                for bi in tla.range_constexpr(2):
                    work = base + bi * aiv_num
                    if work < main_tasks:
                        p = work // n_groups
                        b = work % n_groups
                        ub = tla.make_tensor(ub_ptrs[bi], ub_lay)

                        # MTE2 publishes the source tile. MTE3 then locks the
                        # same slot for RAW consumption; the next MTE2 lifetime
                        # forms WAR after the MTE3 copy releases the slot.
                        mutexes[bi].lock(pipe=tla.pipes.MTE2)
                        tla.copy(
                            ub,
                            tla.make_tensor(
                                gm_src.ptr
                                + tla.Int64(b) * tla.Int64(BR * N)
                                + p.to(tla.Int64) * tla.Int64(W),
                                src_lay,
                            ),
                        )
                        mutexes[bi].unlock(pipe=tla.pipes.MTE2)

                        # Destination tiles are disjoint, so no cross-core sync
                        # is needed. The other slot can load while this MTE3
                        # write-back is in flight.
                        mutexes[bi].lock(pipe=tla.pipes.MTE3)
                        tla.copy(
                            tla.make_tensor(
                                gm_dst.ptr
                                + p.to(tla.Int64) * tla.Int64(part_elems)
                                + tla.Int64(b) * tla.Int64(BR * W),
                                dst_lay,
                            ),
                            ub,
                        )
                        mutexes[bi].unlock(pipe=tla.pipes.MTE3)

        # Tail rows use the same slots and ownership protocol. Flattening
        # (part, tail_row) distributes short remainders across AIV blocks.
        if tla.const_expr(tail != 0):
            for base in tla.range(gid, tail_tasks, pair_stride):
                for bi in tla.range_constexpr(2):
                    work = base + bi * aiv_num
                    if work < tail_tasks:
                        p = work // tail
                        t = work % tail
                        ub_row = tla.make_tensor(ub_ptrs[bi], row_lay)

                        mutexes[bi].lock(pipe=tla.pipes.MTE2)
                        tla.copy(
                            ub_row,
                            tla.make_tensor(
                                gm_src.ptr
                                + (t + tla.Int32(n_groups * BR)).to(tla.Int64)
                                * tla.Int64(N)
                                + p.to(tla.Int64) * tla.Int64(W),
                                row_src_lay,
                            ),
                        )
                        mutexes[bi].unlock(pipe=tla.pipes.MTE2)

                        mutexes[bi].lock(pipe=tla.pipes.MTE3)
                        tla.copy(
                            tla.make_tensor(
                                gm_dst.ptr
                                + p.to(tla.Int64) * tla.Int64(part_elems)
                                + (t + tla.Int32(n_groups * BR)).to(tla.Int64)
                                * tla.Int64(W),
                                row_dst_lay,
                            ),
                            ub_row,
                        )
                        mutexes[bi].unlock(pipe=tla.pipes.MTE3)

        # A single drain after both task loops makes the final write visible.
        tla.pipe_barrier(tla.pipes.ALL)


_COMPILED = {}
_ROWMAJOR = tla.arch.RowMajor
_CORES = {}
_DEVICES = {}


def _device_index(device):
    name = str(device)
    index = _DEVICES.get(name)
    if index is None:
        index = int(name.split(":")[-1]) if ":" in name else 0
        _DEVICES[name] = index
    return index


def _cores(device_index):
    cores = _CORES.get(device_index)
    if cores is None:
        import torch

        cores = int(torch.npu.get_device_properties(device_index).vector_core_num)
        _CORES[device_index] = cores
    return cores


def _launch(kernel, src, dst, extra_args, key, device_index):
    """Compile once and reuse descriptors and the device core count."""
    from catlass.tla.runtime import from_dlpack

    plan = _COMPILED.get(key)
    if plan is None:
        src_desc = from_dlpack(src, layout_tag=_ROWMAJOR)
        dst_desc = from_dlpack(dst, layout_tag=_ROWMAJOR)
        artifact = tla.compile(
            kernel, src_desc, dst_desc, *extra_args, options="--npu-arch 3510"
        )
        plan = {
            "artifact": artifact,
            "cores": _cores(device_index),
            "src": src_desc,
            "dst": dst_desc,
        }
        _COMPILED[key] = plan

    src_desc = plan["src"]
    dst_desc = plan["dst"]
    src_ptr = int(src.data_ptr())
    if src_desc.data_ptr != src_ptr:
        src_desc.data_ptr = src_ptr
    dst_ptr = int(dst.data_ptr())
    if dst_desc.data_ptr != dst_ptr:
        dst_desc.data_ptr = dst_ptr
    plan["artifact"](src_desc, dst_desc, block_num=plan["cores"])


def _split_packed(x, parts, dim=-1):
    """Return the contiguous ``[parts, M, N/parts]`` backing buffer."""
    import torch

    if x.dtype != torch.float32 or x.dim() != 2:
        raise ValueError("x must be 2-D float32")
    if dim not in (-1, 1):
        raise ValueError("only the last dim is supported")
    if isinstance(parts, bool) or not isinstance(parts, int):
        raise TypeError("parts must be an int")
    if parts < 1:
        raise ValueError("parts must be >= 1")

    m, n = x.shape
    if n % parts != 0:
        raise ValueError("N must be divisible by parts")
    x = x if x.is_contiguous() else x.contiguous()
    w = n // parts
    if m == 0 or w == 0:
        return torch.empty((parts, m, w), dtype=torch.float32, device=x.device)

    device_index = _device_index(x.device)
    # UB rows are padded to 32 B so the final row of a strided DMA is not
    # truncated for non-aligned W. The GM strides remain exactly N and W.
    pad_w = ((w + 7) // 8) * 8
    if 2 * pad_w > STAGE_ELEMS:
        raise ValueError(
            "W is too large for double-buffered UB: "
            f"W={w}, padded={pad_w}, limit={STAGE_ELEMS // 2}"
        )
    br = max(1, min(32, STAGE_ELEMS // (2 * pad_w)))
    packed = torch.empty((parts, m, w), dtype=torch.float32, device=x.device)
    flat = packed.view(parts * m, w)
    _launch(
        split_kernel,
        x,
        flat,
        [m, n, w, pad_w, parts, br],
        ("split", device_index, m, n, w, pad_w, parts, br),
        device_index,
    )
    return packed


def split(x, parts, dim=-1):
    """Split the last dimension into dense ``[M, N/parts]`` tensors."""
    packed = _split_packed(x, parts, dim)
    return [packed[p] for p in range(parts)]


def run(x, parts, dim=-1):
    """Return the packed ``[parts, M, N/parts]`` result."""
    return _split_packed(x, parts, dim)


if __name__ == "__main__":
    import argparse

    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(0)

    for m, n, parts in (
        (37, 1024, 4),
        (37, 1001, 7),
        (9, 999, 3),
        (13, 4096, 8),
        (5, 63, 63),
        (1, 1, 1),
        (17, 3, 1),
        (17, 99, 9),
        (73, 1025, 1),
        (129, 4095, 3),
    ):
        x = torch.randn(m, n, device=f"npu:{args.device}")
        outputs = split(x, parts)
        packed_ref = torch.stack(torch.split(x.cpu(), n // parts, -1), dim=0)
        outputs_ref = torch.stack([out.cpu() for out in outputs], dim=0)
        dense = all(out.is_contiguous() for out in outputs)
        torch.testing.assert_close(outputs_ref, packed_ref, rtol=0, atol=0)
        if not dense:
            raise AssertionError(f"non-contiguous split result for {(m, n, parts)}")
        print(
            "split",
            (m, n),
            "parts",
            parts,
            "err=0.000e+00",
            "dense={}".format(dense),
        )
