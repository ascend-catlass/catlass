# -----------------------------------------------------------------------------------------------------------
# GM→L1 Nd2Nz copy sources: use get_tile (ptr/make_tensor) so runtime GM memref
# is always rank-2. tile_view keeps kernel-arg rank-4 ABI; mixing both in one
# fused kernel makes getOrCreateRuntimeCall first-wins lock the stub and fail.
# Semantics match tile_view(parent, shape, tile_coord): element = tile * shape.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import catlass.tla as tla

# Scalar SSA-safe min (arith.minsi / select). Plain Python min() and tla.min
# (vector) both break when called from this undecorated helper with SSA extents.
from catlass.tla_ast_decorators import _internal_frontend_min as _smin


def gm_l1_tile(parent, shape0, shape1, tile_i=0, tile_j=0):
    """Sub-tile of ``parent`` equivalent to ``tile_view`` but via ``get_tile``."""
    return tla.get_tile(
        parent,
        tla.make_coord(tile_i * shape0, tile_j * shape1),
        tla.make_shape(shape0, shape1),
    )


def gm_l1_abs(parent, shape0, shape1, elem0=0, elem1=0):
    """Sub-tile at absolute element coords (not ``tile * shape`` indices).

    Needed for SwiGLU N-tail where the half-column start
    ``block_col * (l1_tn/2)`` is not a multiple of the rem ``half_raw``.

    ``get_tile`` already crops each dim to ``min(shape, origin - coord)``
    (static fold or ``arith.minsi``); no extra clamp here.
    """
    return tla.get_tile(
        parent,
        tla.make_coord(elem0, elem1),
        tla.make_shape(shape0, shape1),
    )


def gm_l1_abs_1d(parent, shape0, elem0=0):
    """1-D absolute-element view (bias half-columns).

    ``get_tile`` is flat 2-D only; VectorLayout bias uses ``ptr + elem`` then
    ``make_tensor`` (element stride 1).

    Callers should pass the clipped tile width (e.g. ``work_tn`` / ``half_raw``).
    Remaining parent length is also clamped so an N-tail cannot DMA past the
    allocated bias when a full ``l1_tn`` is passed by mistake.
    """
    rem = parent.origin_shape[0] - elem0
    use_n = _smin(shape0, rem)
    return tla.make_tensor(
        parent.ptr + elem0,
        tla.make_layout(
            tla.make_shape(use_n),
            tla.make_stride(1),
            layoutTag=tla.arch.VectorLayout,
        ),
    )
