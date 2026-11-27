"""UB->UB copy across dtypes and tile shapes, checked on device.

Covers both forms of the route: the straight copy, and the transposing copy a
ColumnMajor destination asks for (``--mode``, default both).

The frontend behaviour -- which dtypes the route accepts, and the two geometries
it refuses (an origin off a 32-byte boundary, and a row pitch too narrow to
carry the rounded row) -- lives in
tests/test_copy_lowering.py, and the lowering itself in
tests/lit/copy/copy-ub-to-ub.mlir. What needs a device, and so lives here, is
that the bytes actually land: every case checks the copied region against the
source and the region outside the view against a sentinel.

Every case copies a ROWS x VIEW_COLS sub-block between two UB tiles whose parent
widths may differ, so each side uses its own stride0. The destination parent is
pre-seeded with a sentinel and the whole parent is staged back out, so the check
covers both the copied region and every element outside it.

The cases deliberately cover the paths the simple tests miss:
  * partial registers  -- n < lanes, n = lanes +- 1, odd n
  * the chunking loop  -- more than 255 registers in one row (>64 KB)
  * padded views       -- stride0 > shape1, on one side and on both
  * differing pitches  -- src and dst parents of different widths
  * 2-byte lanes       -- f16/bf16 have 128 lanes, so the boundaries move
  * offset views       -- a view whose origin is not its parent's, on one side
                          and on both, with rows offset as well as columns
                          (OFFSET_CASES, driven by ub_to_ub_copy_offset)

The transposing form is expressed by tagging the destination ColumnMajor:
``tla.copy(dst_cm, src_rm)`` with both tiles of logical shape ``(M, N)`` writes
``dst[j][i] = src[i][j]``, an ``(N, M)`` array in memory. The layout tag is the
whole API -- no shape bookkeeping on the caller's side -- exactly as it is for
the ND->zN staging routes. Under it is the meta-op's ``VNCHWCONV`` (AscendC
``TransDataTo5HD``), the vector unit's native transpose: it moves a whole block
of 16 source rows by C0 columns, C0 being the elements in a 32-byte DataBlock,
and a 1-byte type covers its 32x32 block in four calls. Its cases run on that
block grid, and shapes a dtype's block size cannot express are skipped rather
than counted as failures. It has its own offset-view cases
(TRANSPOSE_OFFSET_CASES, driven by ub_to_ub_transpose_offset), which are what
pin the tile coordinate down on both descriptors -- and, through the non-square
ones, pin down that the ColumnMajor destination folds its coordinate with the
two terms swapped.
"""

from __future__ import annotations

import argparse
import builtins as _builtins

import catlass.tla as tla
import torch
from catlass.tla.runtime import from_dlpack

_DTYPES = {
    "f32": (tla.Float32, torch.float32, 64),
    "f16": (tla.Float16, torch.float16, 128),
    "bf16": (tla.BFloat16, torch.bfloat16, 128),
    "i32": (tla.Int32, torch.int32, 64),
    "i16": (tla.Int16, torch.int16, 128),
    "i8": (tla.Int8, torch.int8, 256),
    "f8e4m3": (tla.Float8E4M3FN, torch.float8_e4m3fn, 256),
    "f8e5m2": (tla.Float8E5M2, torch.float8_e5m2, 256),
}

# Parent widths (src_cols, dst_cols) are kept multiples of 32 ELEMENTS so that,
# for every dtype here, the GM<->UB staging of the whole parent is 32-byte
# block-aligned and always correct. That leaves the UB->UB copy under test as the
# only thing a case can exercise: the view width (view_cols) is free to be
# sub-block, odd, or straddle a register boundary. (A separate, dtype-independent
# GM<->UB staging bug corrupts multi-row tiles whose parent width is NOT a
# multiple of 32 bytes; it is out of scope here and is tracked on its own.)

# Each case gives, in order: name, rows, src_cols, dst_cols, view_cols.
CASES = [
    ("single element", 1, 32, 32, 1),
    ("one lane short", 1, 64, 64, 63),
    ("exactly one register", 1, 64, 64, 64),
    ("one past a register", 1, 96, 96, 65),
    ("two registers", 1, 128, 128, 128),
    ("odd, multi register", 1, 1024, 1024, 999),
    ("odd tail on 1000", 1, 1024, 1024, 1000),
    (">255 repeats (68KB)", 1, 17024, 17024, 17000),
    ("padded view", 64, 64, 64, 32),
    ("padded, odd width", 16, 64, 64, 17),
    ("differing pitches", 64, 64, 96, 32),
    ("view is full row", 32, 64, 64, 48),
    # 1-byte boundaries: the b8 vector is 256 lanes, so the masked-tail edges
    # sit at 256, not 64. These exercise the plt.b8 / vsldb.v256i8 fallback.
    ("small 15", 1, 32, 32, 15),
    ("small 16", 1, 32, 32, 16),
    ("small 33", 1, 64, 64, 33),
    ("b8 boundary -1", 1, 288, 288, 255),
    ("b8 boundary", 1, 288, 288, 256),
    ("b8 boundary +1", 1, 288, 288, 257),
    # Multi-row fallback: per-row loop over a sub-block view, few rows and many,
    # contiguous and padded, matching and differing pitches.
    ("sub-block view, few rows", 4, 64, 64, 48),
    ("sub-block view, many rows", 32, 64, 64, 48),
    ("odd view, multi row", 3, 128, 96, 33),
    ("differing pitches, multi row", 32, 64, 96, 40),
    ("sub-block view 40, many rows", 32, 64, 64, 40),
]

# Views whose origin is NOT the parent's. Everything above sits at tile (0, 0),
# which is exactly why the copy ignoring the descriptor's tile coordinate went
# unnoticed: the pitch was honoured, the base was not, and at offset 0 the two
# are indistinguishable. A view at tile (r, c) starts at element (r*vrows,
# c*vcols), and src and dst may sit at different origins.
#
# Rows that are not a whole number of 32-byte blocks take the register copy,
# which loads and stores whole aligned registers, so those origins must be
# 32-byte aligned too. Every offset below is a multiple of 32 bytes for every
# dtype in _DTYPES -- the 1-byte types bind hardest, so `tile * view_cols` is
# kept a multiple of 32 there.
#
# Each case gives, in order: name, parent rows, src parent cols, dst parent
# cols, view rows, view cols, src tile, dst tile.
OFFSET_CASES = [
    ("col offset, both", 64, 64, 64, 64, 32, (0, 1), (0, 1)),
    ("col offset, src only", 64, 64, 64, 64, 32, (0, 1), (0, 0)),
    ("col offset, dst only", 64, 64, 64, 64, 32, (0, 0), (0, 1)),
    ("row and col offset", 64, 64, 64, 16, 32, (1, 1), (1, 1)),
    ("differing row offsets", 64, 64, 96, 16, 32, (2, 1), (1, 1)),
    ("sub-block view, offset", 64, 128, 128, 64, 12, (0, 8), (0, 8)),
    ("sub-block, differing pitches", 64, 128, 192, 64, 12, (0, 8), (0, 8)),
    ("sub-block, few rows", 8, 128, 128, 8, 12, (0, 8), (0, 8)),
]

# Transposing cases: (name, M, N) -- src is M x N, dst is N x M.
TRANSPOSE_CASES = [
    ("square 16", 16, 16),
    ("square 32", 32, 32),
    ("square 64", 64, 64),
    ("wide 32x128", 32, 128),
    ("tall 128x32", 128, 32),
    ("square 128", 128, 128),
    ("wide 32x256", 32, 256),
    ("wide 64x256", 64, 256),
    # Column counts that are a whole number of DataBlocks without being a round
    # multiple of the block grid, so the meta-op's repeat loop ends on an
    # awkward boundary rather than a power of two. C0 is 8 / 16 / 32 elements
    # for b32 / b16 / b8, so a tail that is on the grid for one width is off it
    # for another and is skipped there.
    ("tail 64x72", 64, 72),
    ("tail 32x144", 32, 144),
    ("tail 32x288", 32, 288),
    ("tail 96x200", 96, 200),
]

# Transposing views whose origin is NOT the parent's. TRANSPOSE_CASES all sit at
# tile (0, 0), where a route that ignores the descriptor's tile coordinate is
# indistinguishable from one that honours it. These sit elsewhere, and the
# non-square ones also separate the two ways the destination coordinate can be
# folded: dst is ColumnMajor, so its logical column is its physical row, and
# folding it like the RowMajor source lands a non-square offset tile at the
# wrong address rather than merely at a shifted one.
#
# Every view extent stays on the meta-op's block grid (a multiple of 16x C0, or
# 32x32 for a 1-byte type), and the origins are whole multiples of the extents,
# so an offset case is skipped for a dtype on exactly the same grounds as a
# plain one.
#
# Each case gives: name, src parent M, src parent N, view M, view N, src tile,
# dst tile. The destination parent is the transpose of the source parent, and a
# dst tile (r, c) names the same logical (M, N) coordinate as a src tile does.
TRANSPOSE_OFFSET_CASES = [
    ("col offset", 64, 64, 32, 32, (0, 1), (0, 1)),
    ("row offset", 64, 64, 32, 32, (1, 0), (1, 0)),
    ("row and col offset", 64, 64, 32, 32, (1, 1), (1, 1)),
    ("differing origins", 64, 64, 32, 32, (0, 1), (1, 0)),
    # Non-square views: the two destination foldings disagree here.
    ("non-square, col offset", 64, 128, 32, 64, (0, 1), (0, 1)),
    ("non-square, both", 64, 128, 32, 64, (1, 1), (1, 1)),
    ("non-square, differing", 64, 128, 32, 64, (1, 0), (0, 1)),
]

_DT = tla.Float32
_TORCH_DT = torch.float32
_ROWS, _SRC_COLS, _DST_COLS, _VIEW_COLS = 1, 64, 64, 64
_P_ROWS, _P_SRC, _P_DST, _V_ROWS, _V_COLS = 64, 64, 64, 64, 32
_SRC_TILE, _DST_TILE = (0, 0), (0, 0)
_M, _N, _REPS = 32, 32, 1
_PM, _PN, _TM, _TN = 64, 64, 32, 32
_T_SRC_TILE, _T_DST_TILE = (0, 0), (0, 0)
SENTINEL = -7


def _is_fp8(torch_dt) -> bool:
    return torch_dt in (torch.float8_e4m3fn, torch.float8_e5m2)


def _bits(t):
    """fp8 has no comparison kernels; compare the bytes instead."""
    return t.view(torch.int8) if _is_fp8(t.dtype) else t


def _as_tla(tensors):
    """Hand device buffers to the DSL.

    fp8 does not cross dlpack, so those are handed over as their bytes with the
    element type declared, exactly as fp8_full_lanes.py does for the GM<->UB
    routes.
    """
    if _is_fp8(tensors[0].dtype):
        return [
            from_dlpack(
                t.view(torch.int8),
                layout_tag=tla.arch.RowMajor,
                origin_shape=tuple(t.shape),
                element_type=_DT,
            )
            for t in tensors
        ]
    return [from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in tensors]


@tla.kernel
def ub_to_ub_copy(mem_x: tla.Tensor, mem_y: tla.Tensor) -> None:
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)

    rows, sc, dc, vc = _ROWS, _SRC_COLS, _DST_COLS, _VIEW_COLS

    x_gm = tla.tile_view(mem_x, tla.make_shape(rows, sc), tla.make_coord(0, 0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(rows, dc), tla.make_coord(0, 0))

    x_ub = tla.make_tensor_like(
        tla.allocate((rows, sc), _DT, tla.AddressSpace.ub, 256), x_gm, tla.arch.RowMajor
    )
    y_ub = tla.make_tensor_like(
        tla.allocate((rows, dc), _DT, tla.AddressSpace.ub, 256), y_gm, tla.arch.RowMajor
    )

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        src_v = tla.tile_view(x_ub, tla.make_shape(rows, vc), tla.make_coord(0, 0))
        dst_v = tla.tile_view(y_ub, tla.make_shape(rows, vc), tla.make_coord(0, 0))
        tla.copy(dst_v, src_v)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(y_gm, y_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def ub_to_ub_copy_offset(mem_x: tla.Tensor, mem_y: tla.Tensor) -> None:
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)

    pr, sc, dc = _P_ROWS, _P_SRC, _P_DST
    vr, vc = _V_ROWS, _V_COLS
    src_r, src_c = _SRC_TILE
    dst_r, dst_c = _DST_TILE

    x_gm = tla.tile_view(mem_x, tla.make_shape(pr, sc), tla.make_coord(0, 0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(pr, dc), tla.make_coord(0, 0))

    x_ub = tla.make_tensor_like(
        tla.allocate((pr, sc), _DT, tla.AddressSpace.ub, 256), x_gm, tla.arch.RowMajor
    )
    y_ub = tla.make_tensor_like(
        tla.allocate((pr, dc), _DT, tla.AddressSpace.ub, 256), y_gm, tla.arch.RowMajor
    )

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        src_v = tla.tile_view(
            x_ub, tla.make_shape(vr, vc), tla.make_coord(src_r, src_c)
        )
        dst_v = tla.tile_view(
            y_ub, tla.make_shape(vr, vc), tla.make_coord(dst_r, dst_c)
        )
        tla.copy(dst_v, src_v)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(y_gm, y_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _make_inputs(rows, sc, dc, dtype_key):
    """Source ramp and sentinel-filled destination, plus the tla args for both."""
    _dt, torch_dt, _lanes = _DTYPES[dtype_key]
    if torch_dt in (torch.int32, torch.int16):
        x = (
            torch.arange(rows * sc, dtype=torch.int32)
            .to(torch_dt)
            .reshape(rows, sc)
            .npu()
        )
    else:
        x = (
            (torch.arange(rows * sc, dtype=torch.float32) % 977)
            .to(torch_dt)
            .reshape(rows, sc)
            .npu()
        )
    if torch_dt in (torch.float8_e4m3fn, torch.float8_e5m2):
        # torch.full has no fp8 kernel on NPU: fill as int8, then view.
        y = torch.full((rows, dc), SENTINEL, dtype=torch.int8).npu().view(torch_dt)
        # fp8 does not cross dlpack: hand over the bytes, declare the element type.
        args = [
            from_dlpack(
                t.view(torch.int8),
                layout_tag=tla.arch.RowMajor,
                origin_shape=tuple(t.shape),
                element_type=_dt,
            )
            for t in (x, y)
        ]
    else:
        y = torch.full((rows, dc), SENTINEL, dtype=torch_dt, device="npu")
        args = [from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in (x, y)]
    return x, y, args


def run_offset_case(name, pr, sc, dc, vr, vc, src_tile, dst_tile, dtype_key) -> bool:
    global _DT, _TORCH_DT, _P_ROWS, _P_SRC, _P_DST, _V_ROWS, _V_COLS
    global _SRC_TILE, _DST_TILE
    _DT, _TORCH_DT, _ = _DTYPES[dtype_key]
    _P_ROWS, _P_SRC, _P_DST, _V_ROWS, _V_COLS = pr, sc, dc, vr, vc
    _SRC_TILE, _DST_TILE = src_tile, dst_tile

    x, y, tla_args = _make_inputs(pr, sc, dc, dtype_key)
    artifact = tla.compile(ub_to_ub_copy_offset, *tla_args, options="--npu-arch 3510")
    artifact(*tla_args, block_num=1)
    torch.npu.synchronize()

    yb, xb = _bits(y), _bits(x)
    sr, sc0 = src_tile[0] * vr, src_tile[1] * vc
    dr, dc0 = dst_tile[0] * vr, dst_tile[1] * vc
    copied_ok = bool(
        (yb[dr : dr + vr, dc0 : dc0 + vc] == xb[sr : sr + vr, sc0 : sc0 + vc]).all()
    )
    # Everything outside the view -- and outside the 32-byte block its last
    # column sits in -- must still hold the sentinel.
    rest = yb.clone()
    rest[dr : dr + vr, dc0 : dc0 + _rounded_cols(vc, dtype_key)] = SENTINEL
    guard_ok = bool((rest == SENTINEL).all())

    print(
        f"  {name:<28} {vr}x{vc} src@{(sr, sc0)} dst@{(dr, dc0)} "
        f"of {sc}->{dc} [{dtype_key}]  "
        f"copy={'ok' if copied_ok else 'BAD'} "
        f"guard={'ok' if guard_ok else 'CLOBBERED'}"
    )
    if not copied_ok:
        got = yb[dr, dc0 : dc0 + 6].tolist()
        want = xb[sr, sc0 : sc0 + 6].tolist()
        print(f"      row {dr} got {got} want {want}")
    return copied_ok and guard_ok


def _rounded_cols(view_cols: int, dtype_key: str) -> int:
    """The number of destination columns the block move actually writes.

    MOV.UB.TO.UB counts in whole 32-byte blocks, so the meta-op rounds the row
    length up and carries the row's alignment fill with it. Columns between the
    view's width and that boundary are written with source bytes rather than
    left alone, so they are don't-care for the sentinel check -- they are inside
    the destination row's own 32-byte block. 32 is a whole number of elements
    for every dtype here.
    """
    esz = 256 // _DTYPES[dtype_key][2]
    return ((view_cols * esz + 31) // 32) * 32 // esz


def run_case(name, rows, sc, dc, vc, dtype_key) -> bool:
    global _DT, _TORCH_DT, _ROWS, _SRC_COLS, _DST_COLS, _VIEW_COLS
    _DT, _TORCH_DT, lanes = _DTYPES[dtype_key]
    _ROWS, _SRC_COLS, _DST_COLS, _VIEW_COLS = rows, sc, dc, vc

    if _TORCH_DT in (torch.int32, torch.int16):
        x = (
            torch.arange(rows * sc, dtype=torch.int32)
            .to(_TORCH_DT)
            .reshape(rows, sc)
            .npu()
        )
    else:
        x = (
            (torch.arange(rows * sc, dtype=torch.float32) % 977)
            .to(_TORCH_DT)
            .reshape(rows, sc)
            .npu()
        )
    if _is_fp8(_TORCH_DT):
        # torch.full has no fp8 kernel on NPU: fill as int8, then view.
        y = torch.full((rows, dc), SENTINEL, dtype=torch.int8).npu().view(_TORCH_DT)
        x = x.view(torch.int8).npu().view(_TORCH_DT) if not x.is_npu else x
    else:
        y = torch.full((rows, dc), SENTINEL, dtype=_TORCH_DT, device="npu")

    tla_args = _as_tla((x, y))
    artifact = tla.compile(ub_to_ub_copy, *tla_args, options="--npu-arch 3510")
    artifact(*tla_args, block_num=1)
    torch.npu.synchronize()

    yb, xb = _bits(y), _bits(x)
    copied_ok = bool((yb[:, :vc] == xb[:, :vc]).all())
    guard_from = _rounded_cols(vc, dtype_key)
    guard_ok = bool((yb[:, guard_from:] == SENTINEL).all()) if dc > guard_from else True
    ok = copied_ok and guard_ok
    reg = f"{vc / lanes:.2f} reg/row"
    print(
        f"  {name:<24} {rows}x{vc} of {sc}->{dc} [{dtype_key}, {reg}]  "
        f"copy={'ok' if copied_ok else 'BAD'} guard={'ok' if guard_ok else 'CLOBBERED'}"
    )
    if not ok:
        if not copied_ok:
            bad = int((y[:, :vc] != x[:, :vc]).sum())
            print(
                f"      {bad} wrong elements; row0 got {y[0, :6].tolist()} want {x[0, :6].tolist()}"
            )
        if not guard_ok:
            bad = int((y[:, guard_from:] != SENTINEL).sum())
            print(
                f"      {bad} guard elements clobbered; "
                f"row0 guard {y[0, guard_from : guard_from + 6].tolist()}"
            )
    return ok


def _repeat_transpose(dst, src, reps: int) -> None:
    """Emit `reps` back-to-back transposes into the same destination.

    Separates the transpose from the GM<->UB staging: staging is constant across
    rep counts, so the slope of device time over `reps` is the transpose alone.
    The loop lives in a helper because a Python `for` in traced kernel source is
    rewritten into a dynamic scf loop and fails to lower.
    """
    for _ in _builtins.range(reps):
        tla.copy(dst, src)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def ub_to_ub_transpose(mem_x: tla.Tensor, mem_y: tla.Tensor) -> None:
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)

    m, n = _M, _N

    x_gm = tla.tile_view(mem_x, tla.make_shape(m, n), tla.make_coord(0, 0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(n, m), tla.make_coord(0, 0))

    x_ub = tla.make_tensor_like(
        tla.allocate((m, n), _DT, tla.AddressSpace.ub, 256), x_gm, tla.arch.RowMajor
    )
    y_ptr = tla.allocate((n, m), _DT, tla.AddressSpace.ub, 256)
    # Two views of the same buffer: ColumnMajor (M, N) is the transpose
    # destination, RowMajor (N, M) is the very same bytes staged back out.
    y_ub_t = tla.make_tensor_like(y_ptr, x_gm, tla.arch.ColumnMajor)
    y_ub = tla.make_tensor_like(y_ptr, y_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        _repeat_transpose(y_ub_t, x_ub, _REPS)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(y_gm, y_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def ub_to_ub_transpose_offset(mem_x: tla.Tensor, mem_y: tla.Tensor) -> None:
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)

    pm, pn = _PM, _PN
    vm, vn = _TM, _TN
    src_r, src_c = _T_SRC_TILE
    dst_r, dst_c = _T_DST_TILE

    x_gm = tla.tile_view(mem_x, tla.make_shape(pm, pn), tla.make_coord(0, 0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(pn, pm), tla.make_coord(0, 0))

    x_ub = tla.make_tensor_like(
        tla.allocate((pm, pn), _DT, tla.AddressSpace.ub, 256), x_gm, tla.arch.RowMajor
    )
    y_ptr = tla.allocate((pn, pm), _DT, tla.AddressSpace.ub, 256)
    # The same two views of one buffer as the aligned transpose uses, over the
    # whole parent: ColumnMajor (PM, PN) to transpose into, RowMajor (PN, PM) to
    # stage the identical bytes back out. The views below then carve a tile out
    # of each, which is what puts a coordinate on both descriptors.
    y_ub_t = tla.make_tensor_like(y_ptr, x_gm, tla.arch.ColumnMajor)
    y_ub = tla.make_tensor_like(y_ptr, y_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        src_v = tla.tile_view(
            x_ub, tla.make_shape(vm, vn), tla.make_coord(src_r, src_c)
        )
        dst_v = tla.tile_view(
            y_ub_t, tla.make_shape(vm, vn), tla.make_coord(dst_r, dst_c)
        )
        tla.copy(dst_v, src_v)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(y_gm, y_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def run_transpose_offset_case(
    name, pm, pn, vm, vn, src_tile, dst_tile, dtype_key
) -> bool:
    global _DT, _TORCH_DT, _PM, _PN, _TM, _TN, _T_SRC_TILE, _T_DST_TILE
    _DT, _TORCH_DT, _ = _DTYPES[dtype_key]
    _PM, _PN, _TM, _TN = pm, pn, vm, vn
    _T_SRC_TILE, _T_DST_TILE = src_tile, dst_tile

    seed = (torch.arange(pm * pn, dtype=torch.int32) % 97).reshape(pm, pn)
    if _is_fp8(_TORCH_DT):
        x = seed.to(torch.int8).npu().view(_TORCH_DT)
        y = torch.full((pn, pm), SENTINEL, dtype=torch.int8).npu().view(_TORCH_DT)
    else:
        x = seed.to(_TORCH_DT).npu()
        y = torch.full((pn, pm), SENTINEL, dtype=_TORCH_DT, device="npu")

    tla_args = _as_tla((x, y))
    artifact = tla.compile(
        ub_to_ub_transpose_offset, *tla_args, options="--npu-arch 3510"
    )
    artifact(*tla_args, block_num=1)
    torch.npu.synchronize()

    got, src = _bits(y).cpu(), _bits(x).cpu()
    sr, sc = src_tile[0] * vm, src_tile[1] * vn
    # The destination region is the transpose of the source region's position:
    # the logical (M, N) coordinate (dr, dc) addresses row dc, column dr of the
    # (PN, PM) array.
    dr, dc = dst_tile[0] * vm, dst_tile[1] * vn
    want = src[sr : sr + vm, sc : sc + vn].t().contiguous()
    region = got[dc : dc + vn, dr : dr + vm]
    copied_ok = bool((region == want).all())
    # Nothing outside the destination tile may move: the transpose writes whole
    # DataBlocks and the extents are on the block grid, so there is no alignment
    # fill to excuse here as the straight copy has.
    rest = got.clone()
    rest[dc : dc + vn, dr : dr + vm] = SENTINEL
    guard_ok = bool((rest == SENTINEL).all())

    print(
        f"  {name:<28} {vm}x{vn} src@{(sr, sc)} dst@{(dc, dr)} "
        f"of {pm}x{pn} [{dtype_key}]  "
        f"copy={'ok' if copied_ok else 'BAD'} "
        f"guard={'ok' if guard_ok else 'CLOBBERED'}"
    )
    if not copied_ok:
        bad = int((region != want).sum())
        print(f"      {bad}/{vm * vn} wrong; row0 got {region[0, :8].tolist()}")
        print(f"                        want {want[0, :8].tolist()}")
    if not guard_ok:
        bad = int((rest != SENTINEL).sum())
        print(f"      {bad} elements outside the tile clobbered")
    return copied_ok and guard_ok


def run_transpose_case(name, m, n, dtype_key) -> bool:
    global _DT, _TORCH_DT, _M, _N
    _DT, _TORCH_DT, _ = _DTYPES[dtype_key]
    _M, _N = m, n

    seed = (torch.arange(m * n, dtype=torch.int32) % 97).reshape(m, n)
    if _is_fp8(_TORCH_DT):
        x = seed.to(torch.int8).npu().view(_TORCH_DT)
        y = torch.zeros((n, m), dtype=torch.int8).npu().view(_TORCH_DT)
    else:
        x = seed.to(_TORCH_DT).npu()
        y = torch.zeros((n, m), dtype=_TORCH_DT, device="npu")

    tla_args = _as_tla((x, y))
    artifact = tla.compile(ub_to_ub_transpose, *tla_args, options="--npu-arch 3510")
    artifact(*tla_args, block_num=1)
    torch.npu.synchronize()

    got, want = _bits(y).cpu(), _bits(x).cpu().t().contiguous()
    ok = bool((got == want).all())
    print(f"  {name:<24} {m}x{n} -> {n}x{m} [{dtype_key}]  {'ok' if ok else 'BAD'}")
    if not ok:
        bad = int((got != want).sum())
        print(f"      {bad}/{m * n} wrong; row0 got {got[0, :8].tolist()}")
        print(f"                        want {want[0, :8].tolist()}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtypes", default="f32,f16", help="comma-separated")
    parser.add_argument(
        "--mode",
        choices=("copy", "transpose", "both"),
        default="both",
        help="which form of the UB->UB route to check",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="transposes emitted per kernel; the slope over this isolates the "
        "transpose from the GM<->UB staging (correctness still holds, the "
        "transpose is idempotent into the same destination)",
    )
    parser.add_argument(
        "--shapes", default="", help="transpose only: comma-separated MxN"
    )
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    global _REPS
    _REPS = args.reps

    t_cases = TRANSPOSE_CASES
    if args.shapes:
        t_cases = [
            (spec, *(int(v) for v in spec.lower().split("x")))
            for spec in args.shapes.split(",")
        ]

    results = []
    for dtype_key in args.dtypes.split(","):
        print(f"\n=== dtype {dtype_key} ===")
        if args.mode in ("copy", "both"):
            for name, rows, sc, dc, vc in CASES:
                try:
                    ok = run_case(name, rows, sc, dc, vc, dtype_key)
                    results.append((f"{dtype_key}/copy {name}", ok))
                except Exception as exc:
                    kind = type(exc).__name__
                    print(f"  {name:<24} RAISED {kind}: {str(exc)[:70]}")
                    results.append((f"{dtype_key}/copy {name}", False))
        if args.mode in ("transpose", "both"):
            # VNCHWCONV works in whole blocks of 16 rows by C0 columns, C0
            # being the elements in a 32-byte DataBlock; a 1-byte type spans 32
            # rows because it covers its block in four calls. A shape off that
            # grid is not a failure. lanes is 256/esz and C0 is 32/esz, so C0 is
            # lanes/8 for every width.
            c0 = _DTYPES[dtype_key][2] // 8
            tile_rows = 32 if c0 == 32 else 16
            for name, m, n in t_cases:
                if m % tile_rows or n % c0:
                    print(
                        f"  {name:<24} {m}x{n} [{dtype_key}]  "
                        f"n/a (needs a multiple of {tile_rows}x{c0})"
                    )
                    continue
                try:
                    ok = run_transpose_case(name, m, n, dtype_key)
                    results.append((f"{dtype_key}/transpose {name}", ok))
                except Exception as exc:
                    kind = type(exc).__name__
                    print(f"  {name:<24} RAISED {kind}: {str(exc)[:70]}")
                    results.append((f"{dtype_key}/transpose {name}", False))

            if not args.shapes:
                print("  -- offset transposing views --")
                for (
                    name,
                    pm,
                    pn,
                    vm,
                    vn,
                    src_tile,
                    dst_tile,
                ) in TRANSPOSE_OFFSET_CASES:
                    if vm % tile_rows or vn % c0:
                        print(
                            f"  {name:<28} {vm}x{vn} [{dtype_key}]  "
                            f"n/a (needs a multiple of {tile_rows}x{c0})"
                        )
                        continue
                    try:
                        ok = run_transpose_offset_case(
                            name, pm, pn, vm, vn, src_tile, dst_tile, dtype_key
                        )
                        results.append((f"{dtype_key}/transpose offset {name}", ok))
                    except Exception as exc:
                        kind = type(exc).__name__
                        print(f"  {name:<28} RAISED {kind}: {str(exc)[:70]}")
                        results.append((f"{dtype_key}/transpose offset {name}", False))

        # Offset views are a form of the straight copy, so they follow --mode
        # copy, not the transposing cases above.
        if args.mode in ("copy", "both"):
            print("  -- offset views --")
            for name, pr, sc, dc, vr, vc, src_tile, dst_tile in OFFSET_CASES:
                try:
                    ok = run_offset_case(
                        name, pr, sc, dc, vr, vc, src_tile, dst_tile, dtype_key
                    )
                    results.append((f"{dtype_key}/offset {name}", ok))
                except Exception as exc:
                    kind = type(exc).__name__
                    print(f"  {name:<28} RAISED {kind}: {str(exc)[:70]}")
                    results.append((f"{dtype_key}/offset {name}", False))

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} cases correct")
    if failed:
        print("FAILED: " + ", ".join(failed))
    print("\nPASS" if not failed else "\nFAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
