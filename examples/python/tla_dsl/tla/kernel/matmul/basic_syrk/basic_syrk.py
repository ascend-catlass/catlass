# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full license.
"""Ascend950 SYRK: Y = alpha * X @ X.T + beta * R + diagonal * I, lower triangle only.

Each off-diagonal tile is written to GM twice: once where it was computed, and
once transposed into the upper triangle. The transpose runs in UB through the
``ub,RowMajor -> ub,ColumnMajor`` tla.copy route, so the fixpipe moves each tile
once instead of twice. This example uses FP16 or BF16 for X and R, and FP32 for Y.

A long K splits across cores: each slice writes an FP32 partial tile to a
workspace, then the vector units sum the slices and run the epilogue once.

See README.md for the data flow, the UB budget and measurements.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(
    str(next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir()))
)

import catlass.tla as tla
from catlass.params import (
    CopyL0C2DstParams,
    L0C2UBMode,
    LoadDist,
    NormalLoadParams,
    ScalarRoundMode,
)

UB_SIZE = 248 * 1024  # Ascend 950 (arch.hpp Ascend950::UB_SIZE)
L0C_SIZE = 256 * 1024  # Ascend 950 (arch.hpp Ascend950::L0C_SIZE)
CHUNK_SLOTS = 4

# Residual conversion to FP32 uses round-to-nearest-even without saturation.
_CAST_ROUND_EVEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)


@dataclass(frozen=True)
class SyrkTiling:
    """L1/L0 tiling for the SYRK kernel that mirrors via UB transpose.

    Each fixpipe transfers one chunk of the product tile from L0C, alternating
    between the two paired vector units. Products and outputs remain FP32;
    polynomial updates run in place in the vector epilogue. Each
    off-diagonal result is transposed in UB and stored in the upper triangle.
    """

    l1_tm: tla.Constexpr[int] = 256
    l1_tn: tla.Constexpr[int] = 256
    l1_tk: tla.Constexpr[int] = 128
    l0_tk: tla.Constexpr[int] = 64
    # Rows of the product tile one fixpipe delivers to one AIV.
    chunk_rows: tla.Constexpr[int] = 64
    simd_lanes: tla.Constexpr[int] = 64

    def chunks(self) -> int:
        """Chunks per tile; the AIC and both AIVs run exactly this many."""
        return (self.l1_tm + self.chunk_rows - 1) // self.chunk_rows

    def ub_bytes(self) -> int:
        """UB budget per AIV, including the optional polynomial residual buffer."""
        rows, cols = self.chunk_rows, self.l1_tn
        return 3 * rows * cols * 4 + rows * cols * 2

    def validate(self) -> None:
        if self.l1_tm != self.l1_tn:
            raise ValueError("the triangular tiling needs a square tile")
        if self.l1_tm * self.l1_tn * 4 > L0C_SIZE:
            raise ValueError(f"L0C tile {self.l1_tm}x{self.l1_tn} f32 exceeds L0C_SIZE")
        if self.ub_bytes() > UB_SIZE:
            raise ValueError(
                f"UB request {self.ub_bytes()} B exceeds UB_SIZE {UB_SIZE} B"
            )
        if self.l1_tm % self.chunk_rows:
            raise ValueError("chunk_rows must divide the tile edge")
        if self.chunks() != CHUNK_SLOTS:
            raise ValueError(
                f"this kernel names {CHUNK_SLOTS} flag pairs, so the tile must "
                f"split into exactly {CHUNK_SLOTS} chunks"
            )
        # TransDataTo5HD moves whole 16x16 blocks, so both row pitches have to
        # be a multiple of 16 elements.
        if self.chunk_rows % 16 or self.l1_tn % 16:
            raise ValueError("chunk_rows and l1_tn must be multiples of 16")


@dataclass(frozen=True)
class SyrkLaunchConfig:
    """Resolved tile size, K split and FP32 workspace size."""

    tile: int
    splitk: int
    tasks: int
    workspace_elems: int

    def tiling(self) -> SyrkTiling:
        return SyrkTiling(
            l1_tm=self.tile,
            l1_tn=self.tile,
            l1_tk=128 * 256 // self.tile,
            chunk_rows=self.tile // CHUNK_SLOTS,
        )


def choose_launch_config(
    batch: int,
    m: int,
    k: int,
    cores: int,
    tile: int = 256,
    splitk: int = 1,
    workspace_limit_bytes: int = 512 * 1024 * 1024,
) -> SyrkLaunchConfig:
    """Pick tile size and split count. Pass 0 for either to select automatically."""
    tiles = (256, 128, 64)
    splits = (1, 2, 4, 8, 16, 32)
    if min(batch, m, k, cores) <= 0:
        raise ValueError("batch, m, k and cores must be positive")
    if tile not in (0, *tiles):
        raise ValueError("tile must be 0, 64, 128 or 256")
    if splitk not in (0, *splits):
        raise ValueError("splitk must be 0 or a power of two up to 32")
    if workspace_limit_bytes < 0:
        raise ValueError("workspace_limit_bytes must be non-negative")

    best = None
    best_cost = math.inf
    for edge in tiles:
        if tile and tile != edge:
            continue
        rows = (m + edge - 1) // edge
        tasks = batch * rows * (rows + 1) // 2
        l1_tk = 128 * 256 // edge
        k_tiles = (k + l1_tk - 1) // l1_tk
        for requested in splits:
            if splitk and splitk != requested:
                continue
            if not splitk and requested > k_tiles:
                continue
            # Split whole L1 K tiles, so every slice is aligned and nonempty.
            slice_tiles = (k_tiles + requested - 1) // requested
            factor = (k_tiles + slice_tiles - 1) // slice_tiles
            workspace = tasks * factor * edge * edge if factor > 1 else 0
            if workspace * 4 > workspace_limit_bytes:
                continue
            penalty = {256: 1.0, 128: 1.4, 64: 2.5}[edge]
            rounds = (tasks * factor + cores - 1) // cores
            cost = rounds * edge * edge * k / factor * penalty
            if factor > 1:
                bands = tasks * CHUNK_SLOTS
                cost += 16.0 * workspace * 64.0 / min(64, bands)
            if cost < best_cost:
                best_cost = cost
                best = SyrkLaunchConfig(edge, factor, tasks, workspace)
    if best is None:
        raise ValueError("the requested launch exceeds the workspace limit")
    return best


def _chunk_strip(
    ub_p,
    ub_res,
    ub_out,
    row_i,
    col_j,
    remaining,
    row_base,
    alpha,
    beta,
    diagonal,
    add_diagonal,
    polynomial,
    use_residual,
    lanes,
):
    """Process one SIMD strip (1 × lanes elements) of a chunk.

    Returns the remaining element count so that the last strip in a row can
    mask off the tail.
    """
    c1 = 1
    p_chunk = tla.tile_view(
        ub_p, tla.make_shape(c1, lanes), tla.make_coord(row_i, col_j)
    )
    o_chunk = tla.tile_view(
        ub_out, tla.make_shape(c1, lanes), tla.make_coord(row_i, col_j)
    )
    tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
    acc = p_chunk.load()
    if polynomial:
        acc = tla.mul(acc, alpha, mask=tail)
        if use_residual:
            r_chunk = tla.tile_view(
                ub_res, tla.make_shape(c1, lanes), tla.make_coord(row_i, col_j)
            )
            res_val = r_chunk.load(
                NormalLoadParams(load_dist=LoadDist.DIST_UNPACK_B16)
            ).to(tla.Float32, _CAST_ROUND_EVEN)
            acc = tla.add(acc, tla.mul(res_val, beta, mask=tail), mask=tail)
    if add_diagonal:
        offset = tla.arange(col_j * lanes, dtype=tla.Int32) - (row_base + row_i)
        acc = tla.where(tla.cmp(offset, 0, "eq"), tla.add(acc, diagonal), acc)
    o_chunk.store(acc)
    return remaining


@tla.jit
def _chunk_epilogue(
    ub_p_ptr,
    ub_res_ptr,
    ub_out_ptr,
    ub_tr_ptr,
    res_loaded,
    out_ready,
    tr_ready,
    ub_free,
    gm_r_plane,
    gm_y_plane,
    row_origin,
    col_origin,
    cols,
    row_base,
    alpha,
    beta,
    diagonal,
    add_diagonal: tla.Constexpr[bool],
    mirror: tla.Constexpr[bool],
    _tiling: tla.Constexpr[SyrkTiling],
    chunk_rows: tla.Constexpr[int],
    polynomial: tla.Constexpr[bool],
    use_residual: tla.Constexpr[bool],
    wait_for_store: tla.Constexpr[bool] = True,
):
    """Apply the epilogue to one product chunk in this vector unit's UB.

    Stores the FP32 product in Y at
    ``(row_origin, col_origin)``. The polynomial path first applies
    ``alpha * P + beta * R`` and the optional diagonal term. When ``mirror``
    is set the FP32 result is transposed in UB and stored a second time at
    the mirrored position ``(col_origin, row_origin)``. Diagonal tiles are
    never mirrored.
    """
    c0 = 0
    c1 = 1

    ub_stride = _tiling.l1_tn
    tr_stride = chunk_rows
    # Reuse this chunk's FP32 product buffer after its final read.
    ub_out_ptr = ub_p_ptr
    epilogue_rows = chunk_rows if add_diagonal else 1
    epilogue_cols = _tiling.l1_tn if add_diagonal else chunk_rows * _tiling.l1_tn
    n_strips = (epilogue_cols + _tiling.simd_lanes - 1) // _tiling.simd_lanes

    compute_layout = tla.make_layout(
        tla.make_shape(chunk_rows, _tiling.l1_tn),
        tla.make_stride(ub_stride, c1),
        layoutTag=tla.arch.RowMajor,
    )
    # Only diagonal updates need row coordinates; other chunks are contiguous.
    epilogue_layout = tla.make_layout(
        tla.make_shape(epilogue_rows, epilogue_cols),
        tla.make_stride(epilogue_cols, c1),
        layoutTag=tla.arch.RowMajor,
    )
    ub_p = tla.make_tensor(ub_p_ptr, epilogue_layout)
    ub_res = ub_p
    if tla.const_expr(use_residual):
        ub_res = tla.make_tensor(ub_res_ptr, epilogue_layout)
    ub_out = tla.make_tensor(ub_out_ptr, epilogue_layout)

    gm_y_chunk = tla.get_tile(
        gm_y_plane,
        tla.make_coord(row_origin, col_origin),
        tla.make_shape(chunk_rows, cols),
    )
    # On the last tile of a row or column, GM has fewer elements than the
    # full UB layout. Use the clipped shape for the GM copy but keep UB
    # pitch at ub_stride.
    actual_layout = tla.make_layout(
        tla.make_shape(gm_y_chunk.origin_shape[0], gm_y_chunk.origin_shape[1]),
        tla.make_stride(ub_stride, c1),
        layoutTag=tla.arch.RowMajor,
    )
    ub_out_actual = tla.make_tensor(ub_out_ptr, actual_layout)

    # Wait for the previous chunk's GM stores (MTE3) to finish before
    # reloading the residual, output and transpose UB buffers (MTE2).
    if tla.const_expr(wait_for_store):
        tla.wait_flag(ub_free)
    if tla.const_expr(use_residual):
        gm_r_chunk = tla.get_tile(
            gm_r_plane,
            tla.make_coord(row_origin, col_origin),
            tla.make_shape(chunk_rows, cols),
        )
        ub_res_actual = tla.make_tensor(ub_res_ptr, actual_layout)
        tla.copy(ub_res_actual, gm_r_chunk)
    # Barrier: vector writes must wait for previous stores to complete,
    # even when there is no residual copy to create a pipeline dependency.
    tla.set_flag(res_loaded)
    tla.wait_flag(res_loaded)

    if tla.const_expr(polynomial):
        with tla.vec.func(mode="simd"):
            for row_i in tla.range(c0, epilogue_rows, c1):
                remaining = epilogue_cols
                for col_j in tla.range(c0, n_strips, c1):
                    remaining = _chunk_strip(
                        ub_p,
                        ub_res,
                        ub_out,
                        row_i,
                        col_j,
                        remaining,
                        row_base,
                        alpha,
                        beta,
                        diagonal,
                        add_diagonal,
                        polynomial,
                        use_residual,
                        _tiling.simd_lanes,
                    )

    tla.set_flag(out_ready)  # VECTOR -> MTE3
    tla.wait_flag(out_ready)
    tla.copy(gm_y_chunk, ub_out_actual)

    if tla.const_expr(mirror):
        # Transpose the finished block in UB to produce the mirrored copy
        tr_src = tla.make_tensor(ub_out_ptr, compute_layout)
        tr_dst = tla.make_tensor(
            ub_tr_ptr,
            tla.make_layout(
                tla.make_shape(chunk_rows, _tiling.l1_tn),
                tla.make_stride(c1, tr_stride),
                layoutTag=tla.arch.ColumnMajor,
            ),
        )
        tla.copy(tr_dst, tr_src)
        tla.set_flag(tr_ready)  # VECTOR -> MTE3
        tla.wait_flag(tr_ready)

        gm_y_mirror = tla.get_tile(
            gm_y_plane,
            tla.make_coord(col_origin, row_origin),
            tla.make_shape(_tiling.l1_tn, chunk_rows),
        )
        ub_tr_actual = tla.make_tensor(
            ub_tr_ptr,
            tla.make_layout(
                tla.make_shape(
                    gm_y_mirror.origin_shape[0], gm_y_mirror.origin_shape[1]
                ),
                tla.make_stride(tr_stride, c1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        tla.copy(gm_y_mirror, ub_tr_actual)

    tla.set_flag(ub_free)  # MTE3 -> MTE2


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@tla.jit
def _finish_chunk(
    prod_ptr,
    ub_res_ptr,
    ub_out_ptr,
    ub_tr_ptr,
    res_loaded,
    out_ready,
    tr_ready,
    ub_free,
    gm_r_plane,
    gm_y_plane,
    tile_row,
    tile_col,
    row_base,
    n_actual,
    alpha,
    beta,
    diagonal,
    _tiling: tla.Constexpr[SyrkTiling],
    _polynomial: tla.Constexpr[bool],
    _use_residual: tla.Constexpr[bool],
    wait_for_store: tla.Constexpr[bool],
):
    """Run the epilogue for one chunk, with diagonal and mirror as needed."""

    def apply(add_diagonal, mirror):
        """Forward to the epilogue; only the two trace-time flags differ."""
        _chunk_epilogue(
            prod_ptr,
            ub_res_ptr,
            ub_out_ptr,
            ub_tr_ptr,
            res_loaded,
            out_ready,
            tr_ready,
            ub_free,
            gm_r_plane,
            gm_y_plane,
            tile_row * _tiling.l1_tm + row_base,
            tile_col * _tiling.l1_tn,
            n_actual,
            row_base,
            alpha,
            beta,
            diagonal,
            add_diagonal,
            mirror,
            _tiling,
            _tiling.chunk_rows,
            _polynomial,
            _use_residual,
            wait_for_store,
        )

    if tile_row != tile_col:
        apply(False, True)
    elif tla.const_expr(_polynomial):
        # The diagonal term lands on diagonal tiles, which are never mirrored.
        apply(True, False)
    else:
        apply(False, False)


@tla.jit
def _syrk_impl(
    gm_x: tla.Tensor,  # (B*M, K)  f16 / bf16, RowMajor
    gm_r: tla.Tensor,  # (B*M, M)  residual, same 16-bit type as Y
    gm_y: tla.Tensor,  # (B*M, M)  output
    gm_workspace: tla.Tensor,
    alpha: tla.Float32,
    beta: tla.Float32,
    diagonal: tla.Float32,
    _tiling: tla.Constexpr[SyrkTiling],
    _polynomial: tla.Constexpr[bool],
    _use_residual: tla.Constexpr[bool],
    _splitk: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1

    dtype_x = gm_x.ptr.dtype

    m = gm_y.origin_shape[1]
    k = gm_x.origin_shape[1]
    batch_count = gm_y.origin_shape[0] // m

    # ---- intra-core pipe flags (cube side) -------------------------------
    l1x0_ready = tla.flag("l1x0_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1x1_ready = tla.flag("l1x1_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1t0_ready = tla.flag("l1t0_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1t1_ready = tla.flag("l1t1_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1x0_free = tla.flag("l1x0_free", tla.arch.MTE1, tla.arch.MTE2)
    l1x1_free = tla.flag("l1x1_free", tla.arch.MTE1, tla.arch.MTE2)
    l1t0_free = tla.flag("l1t0_free", tla.arch.MTE1, tla.arch.MTE2)
    l1t1_free = tla.flag("l1t1_free", tla.arch.MTE1, tla.arch.MTE2)
    l0a0_free = tla.flag("l0a0_free", tla.arch.CUBE, tla.arch.MTE1)
    l0a1_free = tla.flag("l0a1_free", tla.arch.CUBE, tla.arch.MTE1)
    l0b0_free = tla.flag("l0b0_free", tla.arch.CUBE, tla.arch.MTE1)
    l0b1_free = tla.flag("l0b1_free", tla.arch.CUBE, tla.arch.MTE1)
    l0ab_ready = tla.flag("l0ab_ready", tla.arch.MTE1, tla.arch.CUBE)
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l0c_free = tla.flag("l0c_free", tla.arch.FIX, tla.arch.CUBE)
    # AIV side
    res_loaded = tla.flag("res_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    out_ready = tla.flag("out_ready", tla.arch.VECTOR, tla.arch.MTE3)
    tr_ready = tla.flag("tr_ready", tla.arch.VECTOR, tla.arch.MTE3)
    ub_free = tla.flag("ub_free", tla.arch.MTE3, tla.arch.MTE2)

    # ---- flags between the AIC and the two AIVs, one pair per chunk -------
    aic_ready0 = tla.cross_flag("syrk_aic_ready0", mode=2)
    aic_ready1 = tla.cross_flag("syrk_aic_ready1", mode=2)
    aic_ready2 = tla.cross_flag("syrk_aic_ready2", mode=2)
    aic_ready3 = tla.cross_flag("syrk_aic_ready3", mode=2)
    aiv_free0 = tla.cross_flag("syrk_aiv_free0", mode=2)
    aiv_free1 = tla.cross_flag("syrk_aiv_free1", mode=2)
    aiv_free2 = tla.cross_flag("syrk_aiv_free2", mode=2)
    aiv_free3 = tla.cross_flag("syrk_aiv_free3", mode=2)

    split_done = tla.cross_flag("syrk_split_done", mode=2)
    split_barrier = tla.cross_flag("syrk_split_barrier", mode=0)
    reduce_ready = tla.flag("syrk_reduce_ready", tla.arch.MTE2, tla.arch.VECTOR)
    reduce_free = tla.flag("syrk_reduce_free", tla.arch.VECTOR, tla.arch.MTE2)

    # ---- on-chip buffers --------------------------------------------------
    l1x0_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tk, dtype_x, tla.AddressSpace.l1, 512
    )
    l1x1_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tk, dtype_x, tla.AddressSpace.l1, 512
    )
    l1t0_ptr = tla.allocate(
        _tiling.l1_tk * _tiling.l1_tn, dtype_x, tla.AddressSpace.l1, 512
    )
    l1t1_ptr = tla.allocate(
        _tiling.l1_tk * _tiling.l1_tn, dtype_x, tla.AddressSpace.l1, 512
    )
    l0a0_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l0_tk, dtype_x, tla.AddressSpace.l0a, 512
    )
    l0a1_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l0_tk, dtype_x, tla.AddressSpace.l0a, 512
    )
    l0b0_ptr = tla.allocate(
        _tiling.l0_tk * _tiling.l1_tn, dtype_x, tla.AddressSpace.l0b, 512
    )
    l0b1_ptr = tla.allocate(
        _tiling.l0_tk * _tiling.l1_tn, dtype_x, tla.AddressSpace.l0b, 512
    )
    l0c_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tn, tla.Float32, tla.AddressSpace.l0c, 1024
    )

    chunk_rows = _tiling.chunk_rows
    ub_stride = _tiling.l1_tn
    slot_elems = chunk_rows * ub_stride
    product_dtype = tla.Float32
    ub_p0_ptr = tla.allocate(slot_elems, product_dtype, tla.AddressSpace.ub, 256)
    ub_p1_ptr = tla.allocate(slot_elems, product_dtype, tla.AddressSpace.ub, 256)
    ub_out_ptr = ub_p0_ptr
    ub_res_ptr = ub_out_ptr
    if tla.const_expr(_use_residual):
        ub_res_ptr = tla.allocate(slot_elems, gm_r.ptr.dtype, tla.AddressSpace.ub, 256)
    ub_tr_ptr = tla.allocate(slot_elems, gm_y.ptr.dtype, tla.AddressSpace.ub, 256)

    # ---- triangular task space -------------------------------------------
    tile_count = (m + _tiling.l1_tm - 1) // _tiling.l1_tm
    tasks_per_batch = tile_count * (tile_count + 1) // 2
    total_tasks = batch_count * tasks_per_batch

    n_chunks = _tiling.chunks()

    with tla.cube():
        tla.set_flag(l1x0_free)
        tla.set_flag(l1x1_free)
        tla.set_flag(l1t0_free)
        tla.set_flag(l1t1_free)
        tla.set_flag(l0a0_free)
        tla.set_flag(l0a1_free)
        tla.set_flag(l0b0_free)
        tla.set_flag(l0b1_free)
        tla.set_flag(l0c_free)

        l1_buf = tla.as_numeric(0)
        l0_buf = tla.as_numeric(0)

        for task in tla.range(
            tla.arch.block_idx(), total_tasks * _splitk, tla.arch.block_num()
        ):
            output_task = task // _splitk
            slice_idx = task % _splitk
            batch_idx = output_task // tasks_per_batch
            tri_idx = output_task - batch_idx * tasks_per_batch
            # Map the linear triangular index back to (tile_row, tile_col).
            # The scalar unit's sqrt gives a good initial guess; a short
            # correction loop handles the rounding.
            radicand = (tri_idx * 8 + 1).to(tla.Float32)
            tile_row = ((radicand.sqrt() - 1.0) * 0.5).to(
                tla.Int32, round_mode=ScalarRoundMode.FLOOR
            )
            while (tile_row + 1) * (tile_row + 2) // 2 <= tri_idx:
                tile_row = tile_row + 1
            tile_col = tri_idx - tile_row * (tile_row + 1) // 2

            gm_x_plane = tla.tile_view(
                gm_x, tla.make_shape(m, k), tla.make_coord(batch_idx, c0)
            )
            gm_xt_plane = tla.make_tensor(
                gm_x.ptr + batch_idx * m * k,
                tla.make_layout(
                    tla.make_shape(k, m),
                    tla.make_stride(c1, k),
                    layoutTag=tla.arch.ColumnMajor,
                ),
            )
            gm_y_plane = tla.tile_view(
                gm_y, tla.make_shape(m, m), tla.make_coord(batch_idx, c0)
            )

            gm_x_rows = tla.tile_view(
                gm_x_plane,
                tla.make_shape(_tiling.l1_tm, k),
                tla.make_coord(tile_row, c0),
            )
            gm_xt_cols = tla.tile_view(
                gm_xt_plane,
                tla.make_shape(k, _tiling.l1_tn),
                tla.make_coord(c0, tile_col),
            )
            gm_y_tile = tla.tile_view(
                gm_y_plane,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(tile_row, tile_col),
            )
            m_actual = gm_y_tile.origin_shape[0]
            n_actual = gm_y_tile.origin_shape[1]

            l0_c = tla.make_tensor_like(l0c_ptr, gm_y_tile, tla.arch.L0Clayout)

            k_l1_count = (k + _tiling.l1_tk - 1) // _tiling.l1_tk
            slice_tiles = (k_l1_count + _splitk - 1) // _splitk
            k_l1_start = slice_idx * slice_tiles
            k_l1_end = k_l1_start + slice_tiles
            if k_l1_end > k_l1_count:
                k_l1_end = k_l1_count

            @tla.jit
            def load_l1(k_l1, l1_buf):
                gm_x_l1 = tla.tile_view(
                    gm_x_rows,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tk),
                    tla.make_coord(c0, k_l1),
                )
                gm_xt_l1 = tla.tile_view(
                    gm_xt_cols,
                    tla.make_shape(_tiling.l1_tk, _tiling.l1_tn),
                    tla.make_coord(k_l1, c0),
                )
                l1_x = tla.make_tensor_like(
                    l1x0_ptr if (l1_buf == c0) else l1x1_ptr, gm_x_l1, tla.arch.zN
                )
                l1_xt = tla.make_tensor_like(
                    l1t0_ptr if (l1_buf == c0) else l1t1_ptr, gm_xt_l1, tla.arch.nZ
                )

                if l1_buf == c0:
                    tla.wait_flag(l1x0_free)
                else:
                    tla.wait_flag(l1x1_free)
                tla.copy(l1_x, gm_x_l1)
                if l1_buf == c0:
                    tla.set_flag(l1x0_ready)
                else:
                    tla.set_flag(l1x1_ready)

                if l1_buf == c0:
                    tla.wait_flag(l1t0_free)
                else:
                    tla.wait_flag(l1t1_free)
                tla.copy(l1_xt, gm_xt_l1)
                if l1_buf == c0:
                    tla.set_flag(l1t0_ready)
                else:
                    tla.set_flag(l1t1_ready)
                return l1_x, l1_xt

            # Small split-K tiles benefit from overlapping the next L1 load.
            prefetch = _tiling.l1_tm == 64 and _splitk > 1
            if tla.const_expr(prefetch):
                _ = load_l1(k_l1_start, l1_buf)
            for k_l1 in tla.range(k_l1_start, k_l1_end, c1):
                if tla.const_expr(prefetch):
                    if k_l1 + c1 < k_l1_end:
                        _ = load_l1(k_l1 + c1, c1 - l1_buf)
                    gm_x_l1 = tla.tile_view(
                        gm_x_rows,
                        tla.make_shape(_tiling.l1_tm, _tiling.l1_tk),
                        tla.make_coord(c0, k_l1),
                    )
                    gm_xt_l1 = tla.tile_view(
                        gm_xt_cols,
                        tla.make_shape(_tiling.l1_tk, _tiling.l1_tn),
                        tla.make_coord(k_l1, c0),
                    )
                    l1_x = tla.make_tensor_like(
                        l1x0_ptr if (l1_buf == c0) else l1x1_ptr, gm_x_l1, tla.arch.zN
                    )
                    l1_xt = tla.make_tensor_like(
                        l1t0_ptr if (l1_buf == c0) else l1t1_ptr, gm_xt_l1, tla.arch.nZ
                    )
                else:
                    l1_x, l1_xt = load_l1(k_l1, l1_buf)

                k_l1_actual = l1_x.origin_shape[1]
                k_l0_count = (k_l1_actual + _tiling.l0_tk - 1) // _tiling.l0_tk
                for k_l0 in tla.range(c0, k_l0_count, c1):
                    l1_x_l0 = tla.tile_view(
                        l1_x,
                        tla.make_shape(_tiling.l1_tm, _tiling.l0_tk),
                        tla.make_coord(c0, k_l0),
                    )
                    l1_xt_l0 = tla.tile_view(
                        l1_xt,
                        tla.make_shape(_tiling.l0_tk, _tiling.l1_tn),
                        tla.make_coord(k_l0, c0),
                    )
                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf == c0) else l0a1_ptr, l1_x_l0, tla.arch.zN
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf == c0) else l0b1_ptr, l1_xt_l0, tla.arch.nZ
                    )

                    if k_l0 == c0:
                        if l1_buf == c0:
                            tla.wait_flag(l1x0_ready)
                        else:
                            tla.wait_flag(l1x1_ready)
                    if l0_buf == c0:
                        tla.wait_flag(l0a0_free)
                    else:
                        tla.wait_flag(l0a1_free)
                    tla.copy(l0_a, l1_x_l0)
                    if k_l0 == k_l0_count - c1:
                        if l1_buf == c0:
                            tla.set_flag(l1x0_free)
                        else:
                            tla.set_flag(l1x1_free)

                    if k_l0 == c0:
                        if l1_buf == c0:
                            tla.wait_flag(l1t0_ready)
                        else:
                            tla.wait_flag(l1t1_ready)
                    if l0_buf == c0:
                        tla.wait_flag(l0b0_free)
                    else:
                        tla.wait_flag(l0b1_free)
                    tla.copy(l0_b, l1_xt_l0)
                    if k_l0 == k_l0_count - c1:
                        if l1_buf == c0:
                            tla.set_flag(l1t0_free)
                        else:
                            tla.set_flag(l1t1_free)

                    tla.set_flag(l0ab_ready)
                    tla.wait_flag(l0ab_ready)

                    if k_l1 == k_l1_start and k_l0 == c0:
                        tla.wait_flag(l0c_free)
                    init_c = True if (k_l1 == k_l1_start and k_l0 == c0) else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c)

                    if l0_buf == c0:
                        tla.set_flag(l0a0_free)
                        tla.set_flag(l0b0_free)
                    else:
                        tla.set_flag(l0a1_free)
                        tla.set_flag(l0b1_free)
                    l0_buf = c1 - l0_buf
                l1_buf = c1 - l1_buf

            # ---- one fixpipe per chunk, to the two AIVs in turn -----------
            tla.set_flag(mmad_done)
            tla.wait_flag(mmad_done)

            if tla.const_expr(_splitk > 1):
                # Each task has its own padded FP32 tile in workspace; only the valid region is stored.
                gm_partial = tla.make_tensor(
                    gm_workspace.ptr + task * _tiling.l1_tm * _tiling.l1_tn,
                    tla.make_layout(
                        tla.make_shape(m_actual, n_actual),
                        tla.make_stride(_tiling.l1_tn, c1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                # MMAD and this store use the pipe flags above, not unit flags.
                tla.copy(gm_partial, l0_c, CopyL0C2DstParams())
            else:
                for chunk in tla.range_constexpr(0, n_chunks):
                    row_base = chunk * chunk_rows
                    free_flag = (aiv_free0, aiv_free1, aiv_free2, aiv_free3)[chunk]
                    ready_flag = (aic_ready0, aic_ready1, aic_ready2, aic_ready3)[chunk]
                    prod_ptr = ub_p0_ptr if (chunk // 2) % 2 == 0 else ub_p1_ptr

                    tla.cross_core_wait_flag(free_flag, tla.arch.FIX)
                    if row_base < m_actual:
                        rows_view = tla.tile_view(
                            l0_c,
                            tla.make_shape(chunk_rows, _tiling.l1_tn),
                            tla.make_coord(chunk, c0),
                        )
                        ub_rows = tla.make_tensor(
                            prod_ptr,
                            tla.make_layout(
                                tla.make_shape(rows_view.origin_shape[0], n_actual),
                                tla.make_stride(ub_stride, c1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        if chunk % 2 == 0:
                            tla.copy(
                                ub_rows,
                                rows_view,
                                CopyL0C2DstParams(
                                    l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_0
                                ),
                            )
                        else:
                            tla.copy(
                                ub_rows,
                                rows_view,
                                CopyL0C2DstParams(
                                    l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_1
                                ),
                            )
                    tla.cross_core_set_flag(ready_flag, tla.arch.FIX)

            tla.set_flag(l0c_free)

        tla.wait_flag(l1x0_free)
        tla.wait_flag(l1x1_free)
        tla.wait_flag(l1t0_free)
        tla.wait_flag(l1t1_free)
        tla.wait_flag(l0a0_free)
        tla.wait_flag(l0a1_free)
        tla.wait_flag(l0b0_free)
        tla.wait_flag(l0b1_free)
        tla.wait_flag(l0c_free)
        if tla.const_expr(_splitk > 1):
            tla.cross_core_set_flag(split_done, tla.arch.FIX)
        else:
            # The AIVs pre-set their free flags before their loop starts,
            # so the AIC can issue a full tile of fixpipes without stalling.
            tla.cross_core_wait_flag(aiv_free0, tla.arch.FIX)
            tla.cross_core_wait_flag(aiv_free1, tla.arch.FIX)
            tla.cross_core_wait_flag(aiv_free2, tla.arch.FIX)
            tla.cross_core_wait_flag(aiv_free3, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        sub_idx = tla.arch.sub_block_idx()

        tla.set_flag(ub_free)
        if tla.const_expr(_splitk > 1):
            # The paired AIC has finished all its partials. The all-core
            # barrier ensures every core's workspace writes are visible.
            tla.cross_core_wait_flag(split_done, tla.arch.MTE2)
            tla.cross_core_set_flag(split_barrier, tla.arch.MTE2)
            tla.cross_core_wait_flag(split_barrier, tla.arch.MTE2)
            tla.set_flag(reduce_free)
            aiv_id = tla.arch.block_idx() * 2 + sub_idx
            aiv_count = tla.arch.block_num() * 2
            for band in tla.range(aiv_id, total_tasks * n_chunks, aiv_count):
                task = band // n_chunks
                chunk = band % n_chunks
                row_base = chunk * chunk_rows
                batch_idx = task // tasks_per_batch
                tri_idx = task - batch_idx * tasks_per_batch
                # Same triangular index -> (row, col) mapping as the cube block.
                radicand = (tri_idx * 8 + 1).to(tla.Float32)
                tile_row = ((radicand.sqrt() - 1.0) * 0.5).to(
                    tla.Int32, round_mode=ScalarRoundMode.FLOOR
                )
                while (tile_row + 1) * (tile_row + 2) // 2 <= tri_idx:
                    tile_row = tile_row + 1
                tile_col = tri_idx - tile_row * (tile_row + 1) // 2
                gm_r_plane = tla.tile_view(
                    gm_r, tla.make_shape(m, m), tla.make_coord(batch_idx, c0)
                )
                gm_y_plane = tla.tile_view(
                    gm_y, tla.make_shape(m, m), tla.make_coord(batch_idx, c0)
                )
                gm_y_tile = tla.tile_view(
                    gm_y_plane,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                    tla.make_coord(tile_row, tile_col),
                )
                m_actual = gm_y_tile.origin_shape[0]
                n_actual = gm_y_tile.origin_shape[1]
                if row_base < m_actual:
                    rows_actual = m_actual - row_base
                    if rows_actual > chunk_rows:
                        rows_actual = chunk_rows
                    layout = tla.make_layout(
                        tla.make_shape(rows_actual, n_actual),
                        tla.make_stride(ub_stride, c1),
                        layoutTag=tla.arch.RowMajor,
                    )
                    ub_sum = tla.make_tensor(ub_p0_ptr, layout)
                    ub_stage = tla.make_tensor(ub_p1_ptr, layout)
                    # Wait for previous output stores before reusing the sum buffer.
                    tla.wait_flag(ub_free)
                    for slice_idx in tla.range(c0, _splitk, c1):
                        tla.wait_flag(reduce_free)
                        slot = (
                            (task * _splitk + slice_idx) * _tiling.l1_tm * _tiling.l1_tn
                        )
                        gm_partial = tla.make_tensor(
                            gm_workspace.ptr + slot + row_base * ub_stride, layout
                        )
                        if slice_idx == c0:
                            tla.copy(ub_sum, gm_partial)
                        else:
                            tla.copy(ub_stage, gm_partial)
                        tla.set_flag(reduce_ready)
                        tla.wait_flag(reduce_ready)
                        if slice_idx != c0:
                            # Supported tile pitches contain only full SIMD strips.
                            with tla.vec.func(mode="simd"):
                                for strip in tla.range(
                                    c0,
                                    rows_actual * ub_stride // _tiling.simd_lanes,
                                    c1,
                                ):
                                    flat_layout = tla.make_layout(
                                        tla.make_shape(c1, slot_elems),
                                        tla.make_stride(slot_elems, c1),
                                        layoutTag=tla.arch.RowMajor,
                                    )
                                    acc = tla.tile_view(
                                        tla.make_tensor(ub_p0_ptr, flat_layout),
                                        tla.make_shape(c1, _tiling.simd_lanes),
                                        tla.make_coord(c0, strip),
                                    )
                                    part = tla.tile_view(
                                        tla.make_tensor(ub_p1_ptr, flat_layout),
                                        tla.make_shape(c1, _tiling.simd_lanes),
                                        tla.make_coord(c0, strip),
                                    )
                                    acc.store(tla.add(acc.load(), part.load()))
                        tla.set_flag(reduce_free)
                    # Apply coefficients, residual and diagonal after summing all partials.
                    _finish_chunk(
                        ub_p0_ptr,
                        ub_res_ptr,
                        ub_out_ptr,
                        ub_tr_ptr,
                        res_loaded,
                        out_ready,
                        tr_ready,
                        ub_free,
                        gm_r_plane,
                        gm_y_plane,
                        tile_row,
                        tile_col,
                        row_base,
                        n_actual,
                        alpha,
                        beta,
                        diagonal,
                        _tiling,
                        _polynomial,
                        _use_residual,
                        False,
                    )
            tla.wait_flag(reduce_free)
        else:
            tla.cross_core_set_flag(aiv_free0, tla.arch.MTE3)
            tla.cross_core_set_flag(aiv_free1, tla.arch.MTE3)
            tla.cross_core_set_flag(aiv_free2, tla.arch.MTE3)
            tla.cross_core_set_flag(aiv_free3, tla.arch.MTE3)

            for task in tla.range(
                tla.arch.block_idx(), total_tasks, tla.arch.block_num()
            ):
                batch_idx = task // tasks_per_batch
                tri_idx = task - batch_idx * tasks_per_batch
                # Same triangular index -> (row, col) mapping as the cube block.
                radicand = (tri_idx * 8 + 1).to(tla.Float32)
                tile_row = ((radicand.sqrt() - 1.0) * 0.5).to(
                    tla.Int32, round_mode=ScalarRoundMode.FLOOR
                )
                while (tile_row + 1) * (tile_row + 2) // 2 <= tri_idx:
                    tile_row = tile_row + 1
                tile_col = tri_idx - tile_row * (tile_row + 1) // 2

                gm_r_plane = tla.tile_view(
                    gm_r, tla.make_shape(m, m), tla.make_coord(batch_idx, c0)
                )
                gm_y_plane = tla.tile_view(
                    gm_y, tla.make_shape(m, m), tla.make_coord(batch_idx, c0)
                )
                gm_y_tile = tla.tile_view(
                    gm_y_plane,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                    tla.make_coord(tile_row, tile_col),
                )
                m_actual = gm_y_tile.origin_shape[0]
                n_actual = gm_y_tile.origin_shape[1]

                for chunk in tla.range_constexpr(0, n_chunks):
                    row_base = chunk * chunk_rows
                    free_flag = (aiv_free0, aiv_free1, aiv_free2, aiv_free3)[chunk]
                    ready_flag = (aic_ready0, aic_ready1, aic_ready2, aic_ready3)[chunk]
                    prod_ptr = ub_p0_ptr if (chunk // 2) % 2 == 0 else ub_p1_ptr
                    owner = chunk % 2

                    tla.cross_core_wait_flag(ready_flag, tla.arch.MTE2)

                    if sub_idx == owner:
                        if row_base < m_actual:
                            _finish_chunk(
                                prod_ptr,
                                ub_res_ptr,
                                ub_out_ptr,
                                ub_tr_ptr,
                                res_loaded,
                                out_ready,
                                tr_ready,
                                ub_free,
                                gm_r_plane,
                                gm_y_plane,
                                tile_row,
                                tile_col,
                                row_base,
                                n_actual,
                                alpha,
                                beta,
                                diagonal,
                                _tiling,
                                _polynomial,
                                _use_residual,
                                True,
                            )
                    # Ensure GM stores (MTE3) complete before releasing the
                    # free flag, so this AIV doesn't outpace the AIC.
                    tla.pipe_barrier(tla.pipes.ALL)
                    tla.cross_core_set_flag(free_flag, tla.arch.MTE3)

        tla.wait_flag(ub_free)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def syrk_kernel(
    gm_x: tla.Tensor,
    gm_y: tla.Tensor,
    _tiling: tla.Constexpr[SyrkTiling],
):
    """Basic SYRK: Y = X @ X.T (no residual or polynomial arithmetic)."""
    _syrk_impl(gm_x, gm_y, gm_y, gm_y, 1.0, 0.0, 0.0, _tiling, False, False, 1)


@tla.kernel
def syrk_splitk_kernel(
    gm_x: tla.Tensor,
    gm_y: tla.Tensor,
    gm_workspace: tla.Tensor,
    _tiling: tla.Constexpr[SyrkTiling],
    _splitk: tla.Constexpr[int],
):
    """Basic SYRK with split-K: each core writes FP32 partials, then the vector units sum and store."""
    _syrk_impl(
        gm_x,
        gm_y,
        gm_y,
        gm_workspace,
        1.0,
        0.0,
        0.0,
        _tiling,
        False,
        False,
        _splitk,
    )


@tla.kernel
def syrk_polynomial_kernel(
    gm_x: tla.Tensor,
    gm_r: tla.Tensor,
    gm_y: tla.Tensor,
    alpha: tla.Float32,
    beta: tla.Float32,
    diagonal: tla.Float32,
    _tiling: tla.Constexpr[SyrkTiling],
    _use_residual: tla.Constexpr[bool],
):
    """Polynomial SYRK: Y = alpha * X @ X.T + beta * R + diagonal * I."""
    _syrk_impl(
        gm_x,
        gm_r,
        gm_y,
        gm_y,
        alpha,
        beta,
        diagonal,
        _tiling,
        True,
        _use_residual,
        1,
    )


@tla.kernel
def syrk_polynomial_splitk_kernel(
    gm_x: tla.Tensor,
    gm_r: tla.Tensor,
    gm_y: tla.Tensor,
    gm_workspace: tla.Tensor,
    alpha: tla.Float32,
    beta: tla.Float32,
    diagonal: tla.Float32,
    _tiling: tla.Constexpr[SyrkTiling],
    _use_residual: tla.Constexpr[bool],
    _splitk: tla.Constexpr[int],
):
    """Polynomial SYRK with split-K: sum FP32 partials, then apply coefficients and mirror."""
    _syrk_impl(
        gm_x,
        gm_r,
        gm_y,
        gm_workspace,
        alpha,
        beta,
        diagonal,
        _tiling,
        True,
        _use_residual,
        _splitk,
    )


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------
def validate_shape(m: int, k: int, batch: int = 1) -> None:
    """Validate supported SYRK input dimensions and batched storage bounds."""
    if (
        min(m, k, batch) <= 0
        or m > 8192
        or k > 4096
        or batch * m * k > 8192**2
        or batch * m * m > 8192**2
    ):
        raise ValueError(
            "SYRK sample requires 1<=M<=8192, 1<=K<=4096 and B*M*K, B*M*M <= 67108864"
        )


def _validate_inputs(x, residual, beta: float) -> None:
    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch tensor")
    if x.ndim not in (2, 3) or min(x.shape) <= 0:
        raise ValueError(
            "x must have positive dimensions and shape (M, K) or (B, M, K)"
        )
    validate_shape(x.shape[-2], x.shape[-1], x.shape[0] if x.ndim == 3 else 1)
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must be FP16 or BF16")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.requires_grad:
        raise ValueError("this example does not implement autograd; detach x first")
    if residual is not None:
        if not isinstance(residual, torch.Tensor):
            raise TypeError("residual must be a torch tensor or None")
        expected = (*x.shape[:-2], x.shape[-2], x.shape[-2])
        if residual.shape != expected:
            raise ValueError(f"residual must have shape {expected}")
        if residual.device != x.device or residual.dtype != x.dtype:
            raise ValueError("residual must have the same device and dtype as x")
        if not residual.is_contiguous():
            raise ValueError("residual must be contiguous")
        if residual.requires_grad:
            raise ValueError(
                "this example does not implement autograd; detach residual first"
            )
    elif beta != 0.0:
        raise ValueError("residual is required when beta is nonzero")
    if x.device.type != "npu":
        raise ValueError("x must be on an Ascend 950 NPU")


def _prepare_launch(
    x,
    residual,
    alpha: float,
    beta: float,
    diagonal: float,
    polynomial: bool,
    *,
    tile: int,
    splitk: int,
    workspace_limit_mb: int,
    block_num: int,
):
    import torch

    from helpers import create_tla_tensor, get_block_num

    batch = x.shape[0] if x.ndim == 3 else 1
    m, k = x.shape[-2:]
    block_num = get_block_num(block_num, x.device.index, kind="mix")
    config = choose_launch_config(
        batch, m, k, block_num, tile, splitk, workspace_limit_mb * 1024 * 1024
    )
    tiling = config.tiling()
    tiling.validate()
    y = torch.empty((*x.shape[:-2], m, m), device=x.device, dtype=torch.float32)
    x_tla = create_tla_tensor(x.reshape(batch * m, k), layout="row")
    y_tla = create_tla_tensor(y.reshape(batch * m, m), layout="row")
    kernel_fn = syrk_kernel
    launch_args = (x_tla, y_tla)
    constexpr_args = (tiling,)
    if polynomial:
        use_residual = beta != 0.0
        r_tla = y_tla
        if use_residual:
            r_tla = create_tla_tensor(residual.reshape(batch * m, m), layout="row")
        kernel_fn = syrk_polynomial_kernel
        launch_args = (
            x_tla,
            r_tla,
            y_tla,
            tla.Float32(alpha),
            tla.Float32(beta),
            tla.Float32(diagonal),
        )
        constexpr_args = (tiling, use_residual)
    if config.splitk > 1:
        # Local references keep the workspace tensor alive until the launch completes.
        workspace = torch.empty(
            (config.tasks * config.splitk * config.tile, config.tile),
            device=x.device,
            dtype=torch.float32,
        )
        workspace_tla = create_tla_tensor(workspace, layout="row")
        if polynomial:
            kernel_fn = syrk_polynomial_splitk_kernel
            launch_args = (*launch_args[:3], workspace_tla, *launch_args[3:])
        else:
            kernel_fn = syrk_splitk_kernel
            launch_args = (*launch_args, workspace_tla)
        constexpr_args = (*constexpr_args, config.splitk)
    kernel = tla.compile(
        kernel_fn, *launch_args, *constexpr_args, options="--npu-arch 3510"
    )
    return y, kernel, launch_args, block_num, config


def _execute(x, residual, alpha, beta, diagonal, **options):
    import torch
    import torch_npu  # noqa: F401

    # Any non-default coefficient selects the polynomial kernel; the basic one
    # has no epilogue arithmetic to apply.
    polynomial = residual is not None or (alpha, beta, diagonal) != (1.0, 0.0, 0.0)
    _validate_inputs(x, residual, beta)
    with torch.npu.device(x.device):
        y, kernel, launch_args, block_num, _ = _prepare_launch(
            x, residual, alpha, beta, diagonal, polynomial, **options
        )
        kernel(*launch_args, block_num=block_num)
        # Keep temporary tensors alive until this stream's kernel completes.
        torch.npu.current_stream(x.device).synchronize()
    return y


def syrk(
    x,
    residual=None,
    alpha: float = 1.0,
    beta: float = 0.0,
    diagonal: float = 0.0,
    *,
    tile: int = 256,
    splitk: int = 1,
    workspace_limit_mb: int = 512,
    block_num: int = -1,
):
    """Return alpha * X @ X.T + beta * residual + diagonal * I for a contiguous
    FP16/BF16 NPU tensor of shape (M, K) or (B, M, K).

    The defaults give FP32 Y = X @ X.T. The residual must be symmetric, contiguous,
    and match the output shape and input device and dtype; pass ``None`` when ``beta`` is
    zero and the kernel skips the residual load entirely. Pass ``tile=0`` or
    ``splitk=0`` to let the shape choose. The launch runs on Torch's current
    stream and waits for that stream before returning. ``block_num`` must be
    -1 (automatic) or positive. Autograd is not implemented.
    """
    return _execute(
        x,
        residual,
        float(alpha),
        float(beta),
        float(diagonal),
        tile=tile,
        splitk=splitk,
        workspace_limit_mb=workspace_limit_mb,
        block_num=block_num,
    )


def golden(
    x,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    residual=None,
    diagonal: float = 0.0,
):
    """Accumulate quantized input and coefficients in FP64, then round to FP32."""
    import torch

    x_fp64 = x.double()
    out = alpha * (x_fp64 @ x_fp64.transpose(-2, -1))
    if beta:
        if residual is None:
            raise ValueError("residual is required when beta is nonzero")
        out = out + beta * residual.double()
    if diagonal:
        out = out + diagonal * torch.eye(x.shape[-2], dtype=torch.float64)
    return out.float()


def check(actual, expected, k: int) -> bool:
    """Check output accuracy and symmetry using the shared tolerances."""
    import torch

    from helpers import compare

    passed = bool(torch.isfinite(actual).all()) and compare(actual, expected, k)
    symmetric = compare(actual, actual.transpose(-2, -1).contiguous(), k)
    print(f"passed={passed} symmetric={symmetric}")
    return passed and symmetric


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=-1)
    parser.add_argument(
        "--tile", type=int, default=256, help="0 picks the tile from the shape"
    )
    parser.add_argument(
        "--splitk", type=int, default=1, help="0 picks the K split from the shape"
    )
    parser.add_argument("--workspace-limit-mb", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--diagonal", type=float, default=0.0)
    args = parser.parse_args()

    try:
        validate_shape(args.m, args.k, args.batch)
    except ValueError as error:
        parser.error(str(error))
    if args.block_num != -1 and args.block_num <= 0:
        parser.error("--block-num must be -1 or positive")
    import torch
    import torch_npu  # noqa: F401

    dtype = torch.float16 if args.dtype == "f16" else torch.bfloat16
    torch.manual_seed(0)
    shape = (args.m, args.k) if args.batch == 1 else (args.batch, args.m, args.k)
    x = (torch.rand(*shape) * 10 - 5).to(dtype)
    residual = None
    if args.beta != 0.0:
        # The symmetry check below only holds for a symmetric residual.
        noise = torch.rand(*x.shape[:-2], args.m, args.m) * 2 - 1
        residual = ((noise + noise.transpose(-2, -1)) * 0.5).to(dtype)

    expected = golden(
        x,
        alpha=args.alpha,
        beta=args.beta,
        residual=residual,
        diagonal=args.diagonal,
    )
    device = f"npu:{args.device}"
    options = {
        "tile": args.tile,
        "splitk": args.splitk,
        "workspace_limit_mb": args.workspace_limit_mb,
        "block_num": args.block_num,
    }
    actual = syrk(
        x.to(device),
        None if residual is None else residual.to(device),
        args.alpha,
        args.beta,
        args.diagonal,
        **options,
    )
    torch.npu.synchronize(args.device)
    print(
        f"batch={args.batch} m={args.m} k={args.k} dtype={dtype} "
        f"tile={args.tile} splitk={args.splitk} blocks={args.block_num}"
    )
    return 0 if check(actual.cpu(), expected, args.k) else 1


if __name__ == "__main__":
    raise SystemExit(main())
