"""MX MMAD: microscaling matmul on the cube, fp8 and fp4 through one kernel.

Tiled over M, N and K and spread across the cube cores, in the shape
basic_matmul uses: the tile extents are fixed Constexpr constants, the grid is
derived from the operands at runtime, and each core walks a strided slice of the
M x N tiles. The point is to exercise the scale path -- e8m0 blocks reaching L1,
riding the L1->L0 load, and steering the cube onto mad_mx -- on a kernel shaped
like the other matmul examples rather than a one-tile special case.

fp8 and fp4 share `mx_mmad_kernel`. They can, because the fp4-ness is not spelled
anywhere on the device side: the operand element type carries it, and the tiles
take their dtype straight off the operands (`gm_a.ptr.dtype`). Everything else,
the scale tiles, the flags, the pipeline, the `tla.mmad_mx`, is identical.

What genuinely differs is host-side: fp8 quantises with torch's float8 dtypes,
while fp4 needs a 16-entry LUT and nibble packing. Hence two `quantize_mx_*`
functions over one shared scale packing.

Scale blocks go over as the plain (M, ceil(K/32)) and (ceil(K/32), N) matrices,
padded to an even group count. The GM->L1
DMA does the zZMxScale / nNMxScale fractal reorder, steered by the GM-side layout tag, so the
host never pre-swizzles them. The host-side quantisation itself lives in the
sibling ``gen_data.py``: this file is the kernel plus the argument handling.
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
import torch
import torch_npu  # noqa: F401
from catlass.tla.runtime import from_dlpack
from gen_data import MX_SCALE_GROUP_NUM

# L1 tile extents. The problem is chunked at these widths and the kernel derives
# the grid and the K-tile count from the operands themselves.
MX_L1_TM = 128
MX_L1_TN = 128
MX_L1_TK = 128

# Host-side only: the tla element type each fp8 case's GM tensors carry. The
# torch dtype that selects the quantization grid lives in gen_data, keyed by the
# same case name. The kernel reads the dtype off the operands, so the selection
# never reaches it.
FP8_SEL = {"f8e4m3fn": tla.Float8E4M3FN, "f8e5m2": tla.Float8E5M2}
FP4_SEL = {"f4e2m1": tla.Float4E2M1, "f4e1m2": tla.Float4E1M2}


@dataclass(frozen=True)
class MxTilingParams:
    """Tile extents for the MX matmul, in the shape the other examples use.

    Fixed constants, independent of the problem shape: they size the on-chip
    buffers, which `tla.allocate` reserves at compile time. The kernel derives
    the grid and the K-tile count from the operands, and partial boundary tiles
    are handled by tile_view cropping, so one compile serves any M, N and K.
    """

    tm: tla.Constexpr[int] = MX_L1_TM
    tn: tla.Constexpr[int] = MX_L1_TN
    tk: tla.Constexpr[int] = MX_L1_TK


@tla.kernel
def mx_mmad_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_sa: tla.Tensor,
    gm_sb: tla.Tensor,
    gm_c: tla.Tensor,
    _tiling: tla.Constexpr[MxTilingParams],
) -> None:
    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype

    c0 = 0
    c1 = 1
    # Constexpr, because tla.allocate reserves on-chip bytes at compile time and
    # so needs a literal count. They cannot be read off the tensors: the operands
    # are layout-dynamic, so origin_shape is a runtime extent -- which is exactly
    # what lets the grid and k_tiles below be runtime loop bounds, and equally
    # what stops any of them sizing a buffer.
    tm = _tiling.tm
    tn = _tiling.tn
    tk = _tiling.tk
    sk = (tk // MX_SCALE_GROUP_NUM + 1) // 2 * 2

    # Derived, not passed: the extents come off the operands and the tile widths
    # off the tiling, so the two cannot disagree. All runtime values -- same
    # shape as basic_matmul's grid_m / grid_n / k_l1_count.
    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    grid_m = (m + tm - 1) // tm
    grid_n = (n + tn - 1) // tn
    total_blocks = grid_m * grid_n
    k_tiles = (gm_a.origin_shape[1] + tk - 1) // tk

    l1a = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    # The scale tiles are e8m0 whatever the operands are: one shared exponent
    # per 32 elements along K.
    l1sa = tla.allocate(tm * sk, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    # mxfp8 needs its L0 operands on a 1024-byte boundary. fp4 only needs
    # 512, and 1024 satisfies both, so one kernel can serve either family.
    l0a = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l0a, 1024)
    l0b = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l0b, 1024)
    l0c = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 1024)

    ab_ready = tla.flag("ab_ready", tla.arch.MTE2, tla.arch.MTE1)
    l0_ready = tla.flag("l0_ready", tla.arch.MTE1, tla.arch.CUBE)
    # Single-buffered L1/L0, so each stage must be told when its buffer is free
    # again: MTE1 releases L1 back to MTE2, and the cube releases L0 back to MTE1.
    l1_avail = tla.flag("l1_avail", tla.arch.MTE1, tla.arch.MTE2)
    l0_avail = tla.flag("l0_avail", tla.arch.CUBE, tla.arch.MTE1)

    with tla.cube():
        # One linear pass over the M x N tiles, strided by the core count, so a
        # core picks up every block_num-th tile. Same distribution basic_matmul
        # uses; the MX path is indifferent to it.
        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n

            gm_c_t = tla.tile_view(
                gm_c, tla.make_shape(tm, tn), tla.make_coord(block_row, block_col)
            )
            t_l0c = tla.make_tensor_like(l0c, gm_c_t)

            # Set here rather than once outside: each tile's K loop consumes both
            # flags on the way out, so the pair has to be re-armed per tile for
            # the counts to balance.
            tla.set_flag(l1_avail)
            tla.set_flag(l0_avail)
            for kk in tla.range(c0, k_tiles, c1):
                # A walks K along its columns; B (column-major) and both scale
                # blocks walk K along their rows. The M/N coordinate is the tile
                # this core is on.
                gm_a_t = tla.tile_view(
                    gm_a, tla.make_shape(tm, tk), tla.make_coord(block_row, kk)
                )
                gm_b_t = tla.tile_view(
                    gm_b, tla.make_shape(tk, tn), tla.make_coord(kk, block_col)
                )
                # A's scale block is the real (M, K/32) matrix, so chunk kk is a
                # column range of it; B's is (K/32, N), so chunk kk is a row
                # range. The M/N coordinate is shared with the operand: a scale
                # block describes the tile it scales, row for row.
                gm_sa_t = tla.tile_view(
                    gm_sa, tla.make_shape(tm, sk), tla.make_coord(block_row, kk)
                )
                gm_sb_t = tla.tile_view(
                    gm_sb, tla.make_shape(sk, tn), tla.make_coord(kk, block_col)
                )

                # The MX cube consumes whole 64-K tiles, but the GM copy brings
                # only the valid K of this chunk. So the L1 tiles are built with
                # the K extent rounded up to 64 (via make_layout_with_tag, which
                # takes the extent straight from the caller instead of from a
                # reference tile), the valid part is copied from GM, and the pad
                # tail is zero-filled explicitly below. Direction still follows
                # the GM tile -- exactly what make_tensor_like inferred before --
                # so the GM -> L1 routes are unchanged.
                m_valid = gm_a_t.origin_shape[0]
                n_valid = gm_b_t.origin_shape[1]
                k_valid = gm_a_t.origin_shape[1]
                k_l0 = (k_valid + 63) // 64 * 64
                a_tag = (
                    tla.arch.zN
                    if gm_a_t.layout_tag in ("RowMajor", "zN")
                    else tla.arch.nZ
                )
                b_tag = (
                    tla.arch.nZ
                    if gm_b_t.layout_tag in ("ColumnMajor", "nZ")
                    else tla.arch.zN
                )
                t_l1a = tla.make_tensor(
                    l1a, tla.make_layout_with_tag((m_valid, k_l0), dtype_a, a_tag)
                )
                t_l1b = tla.make_tensor(
                    l1b, tla.make_layout_with_tag((k_l0, n_valid), dtype_b, b_tag)
                )
                t_l1sa = tla.make_tensor_like(l1sa, gm_sa_t, tla.arch.zZMxScale)
                t_l1sb = tla.make_tensor_like(l1sb, gm_sb_t, tla.arch.nNMxScale)

                tla.wait_flag(l1_avail)
                # Copy only the valid extents: the L1 tiles are 64-K aligned but
                # the GM tiles stop at k_valid, so a full-tile copy would read
                # past the GM tensor.
                tla.copy(
                    tla.tile_view(
                        t_l1a, tla.make_shape(m_valid, k_valid), tla.make_coord(0, 0)
                    ),
                    gm_a_t,
                )
                tla.copy(
                    tla.tile_view(
                        t_l1b, tla.make_shape(k_valid, n_valid), tla.make_coord(0, 0)
                    ),
                    gm_b_t,
                )
                tla.copy(t_l1sa, gm_sa_t)
                tla.copy(t_l1sb, gm_sb_t)
                # Zero the pad tail [k_valid, k_l0) before the L1 -> L0 load.
                # Two device mechanisms share the work, complementary and
                # non-overlapping:
                #   - the GM -> L1 copy above zero-fills the destination's final
                #     partial C0 column. The DMA pads the last block of the copied
                #     extent; that covers the residue
                #     [k_valid, roundUp_C0(k_valid)) inside the last partially
                #     filled C0 column.
                #   - tla.fill below is whole-C0-column granular, so it covers
                #     only the [roundUp_C0(k_valid), k_l0) columns the copy did
                #     not touch.
                pad_a = tla.get_tile(
                    t_l1a,
                    tla.make_coord(0, k_valid),
                    tla.make_shape(m_valid, k_l0 - k_valid),
                )
                pad_a.fill(0)
                pad_b = tla.get_tile(
                    t_l1b,
                    tla.make_coord(k_valid, 0),
                    tla.make_shape(k_l0 - k_valid, n_valid),
                )
                pad_b.fill(0)
                tla.set_flag(ab_ready)
                tla.wait_flag(ab_ready)

                t_l0a = tla.make_tensor_like(l0a, t_l1a, tla.arch.zN)
                t_l0b = tla.make_tensor_like(l0b, t_l1b, tla.arch.nZ)

                tla.wait_flag(l0_avail)
                # The scale rides the L1->L0 load, not the matmul: this is what
                # attaches the e8m0 block and selects mad_mx.
                tla.copy(t_l0a, t_l1a, scale=t_l1sa)
                tla.copy(t_l0b, t_l1b, scale=t_l1sb)
                tla.set_flag(l1_avail)
                tla.set_flag(l0_ready)
                tla.wait_flag(l0_ready)

                init_c = True if kk == 0 else False
                unit_flag = 0b11 if kk == k_tiles - 1 else 0b10
                # tla.mmad_mx, not tla.mmad: both operands were loaded with a scale,
                # and the two ops are not interchangeable -- passing scaled operands
                # to tla.mmad is an error, and vice versa.
                tla.mmad_mx(t_l0c, t_l0a, t_l0b, init_c=init_c, unit_flag=unit_flag)
                tla.set_flag(l0_avail)

            tla.wait_flag(l1_avail)
            tla.wait_flag(l0_avail)
            tla.copy(gm_c_t, t_l0c, tla.params.CopyL0C2DstParams(unit_flag=0b11))


def _canImplement(args: argparse.Namespace) -> None:
    """Reject inputs that violate the README constraints before any real work.

    The README states what a legal mx_mmad call is; mirror those rules here so a
    bad shape, layout or case fails fast up front instead of surfacing later as
    an obscure device-side error.
    """
    is_fp4 = args.fp4_case is not None

    # Element-width consistency: fp8 never multiplies fp4. Within one family any
    # encoding may pair (fp8 x fp8, fp4 x fp4), but the family is fixed by case.
    if is_fp4:
        if args.fp4_case not in gen_data.FP4_FORMATS:
            raise SystemExit(f"unknown fp4 encoding: {args.fp4_case!r}")
        if args.dtype_a is not None or args.dtype_b is not None:
            raise SystemExit(
                "fp4 case cannot be combined with fp8 operand dtypes: "
                f"fp4_case={args.fp4_case!r} dtype_a={args.dtype_a!r} "
                f"dtype_b={args.dtype_b!r}"
            )
    else:
        if (
            args.dtype_a not in gen_data.FP8_DTYPES
            or args.dtype_b not in gen_data.FP8_DTYPES
        ):
            raise SystemExit(
                "fp8 dtype pair must each be one of "
                f"{', '.join(sorted(gen_data.FP8_DTYPES))}, got "
                f"dtype_a={args.dtype_a!r} dtype_b={args.dtype_b!r}"
            )

    # fp4 keeps K contiguous: the nibble pairing runs along it, and every copy
    # route into L1 -- plain, transposing or prepacked -- has to keep that pairing
    # on the contiguous axis. So A may be row-major (M, K/2) or the prepacked zN
    # tile, B column-major (N, K/2) or the prepacked nZ tile; `col` A and `row` B
    # put a stride on K and are refused. fp8 has no such constraint -- a whole
    # byte per element -- so every layout is available to it.
    if is_fp4 and (
        getattr(args, "layout_a", "row") not in ("row", "zN")
        or getattr(args, "layout_b", "col") not in ("col", "nZ")
    ):
        raise SystemExit(
            "packed fp4 requires --layout-a row|zN --layout-b col|nZ: the "
            "two-per-byte packing runs along K and must be the contiguous axis, "
            "so the transposing handovers (col A, row B) are unavailable"
        )


def run(args: argparse.Namespace) -> int:
    from helpers import compare, create_tla_tensor, get_block_num, host_nd2nz

    # One layout per side, driving both the operand and its scale block: a
    # scale block describes the operand it scales, so they are never independent.
    # Row-major A and column-major B reach L1 through a transposing DN2NZ, the
    # other orientations -- and the prepacked zN / nZ tiles -- through a plain
    # ND2NZ / block move. zN / nZ move no data through the DMA's re-layout path:
    # the host hands the tile over already in fractal order.
    layout_a = getattr(args, "layout_a", "row")
    layout_b = getattr(args, "layout_b", "col")

    _canImplement(args)

    torch.npu.set_device(args.device)
    m, n, k = args.m, args.n, args.k
    tm, tn, tk = MX_L1_TM, MX_L1_TN, MX_L1_TK
    is_fp4 = args.fp4_case is not None

    if is_fp4:
        fmt = args.fp4_case
        print(
            f"--- mx fp4 mnk=({m},{n},{k}) fmt={fmt} "
            f"grid={(m + tm - 1) // tm}x{(n + tn - 1) // tn} k_tiles={(k + tk - 1) // tk} ---"
        )
    else:
        print(
            f"--- mx fp8 mnk=({m},{n},{k}) "
            f"dtype={args.dtype_a}/{args.dtype_b} "
            f"grid={(m + tm - 1) // tm}x{(n + tn - 1) // tn} k_tiles={(k + tk - 1) // tk} ---"
        )
    # Host-side quantisation and the GM scale-block packing both live in
    # gen_data: one call produces the operands, the e8m0 blocks and the fp32
    # golden, already oriented for the layouts this run asked for. The returned
    # buffers are byte views (the operand element type is carried by the TLA
    # descriptor), so all that is left here is moving them to the device.
    inputs = gen_data.gen_mx_matmul_inputs(
        m,
        n,
        k,
        fp4_format=fmt if is_fp4 else None,
        dtype_a=args.dtype_a,
        dtype_b=args.dtype_b,
        layout_a=layout_a,
        layout_b=layout_b,
    )
    ref = inputs.ref

    sa_npu = inputs.scale_a.npu()
    sb_npu = inputs.scale_b.npu()
    c_npu = torch.zeros(m, n, dtype=torch.float32).npu()

    if is_fp4:
        # The GM tile carries the fp4 element type, exactly as the fp8 branch
        # below carries its own -- the host has no fp4 dtype, so the buffer is
        # exported as bytes and element_type says how to read them. It cannot use
        # create_tla_tensor only because origin_shape has to state the count in
        # fp4 ELEMENTS while the buffer holds half that many, and the helper
        # takes no origin_shape.
        #
        # `zN` / `nZ` prepack the same K-contiguous bytes, so the nibble pairing
        # survives: host_nd2nz moves whole bytes, and a 32-byte C0 run still
        # holds 64 consecutive K elements. Only the transposing handovers (col A,
        # row B) would put a stride on K, and _canImplement refuses those.
        #
        # The packing primitive always shards the buffer's *last* axis, so its
        # tag names the orientation the buffer already has, not the layout we
        # are building. A arrives as (M, K/2) and takes "zN". B arrives as
        # (N, K/2) -- the quantiser pairs nibbles along K and leaves K last --
        # so B takes the same "zN" call even though the packed order it
        # produces is B's `nZ` fractal layout. Asking host_nd2nz for "nZ" would
        # transpose first and shard N instead, scattering K across blocks.
        fp4_t = FP4_SEL[fmt]
        a_buf = host_nd2nz(inputs.a, "zN") if layout_a == "zN" else inputs.a
        b_buf = host_nd2nz(inputs.b, "zN") if layout_b == "nZ" else inputs.b
        a_t = from_dlpack(
            a_buf.npu(),
            layout_tag=tla.arch.zN if layout_a == "zN" else tla.arch.RowMajor,
            origin_shape=(m, k),
            element_type=fp4_t,
        ).mark_layout_dynamic()
        b_t = from_dlpack(
            b_buf.npu(),
            layout_tag=tla.arch.nZ if layout_b == "nZ" else tla.arch.ColumnMajor,
            origin_shape=(k, n),
            element_type=fp4_t,
        ).mark_layout_dynamic()
    else:
        # `row` / `col` upload the host buffer as the logical tile (col already
        # transposed by gen_data, because its buffer is (k, m) / (n, k)); `zN` /
        # `nZ` pack it into fractal order first and hand create_tla_tensor the
        # packed run plus the logical extent it came from. The packing is spelled
        # out here because it moves data -- folding it into create_tla_tensor
        # would hide a buffer rewrite behind a binder that otherwise only derives
        # metadata.
        elem_a = FP8_SEL[args.dtype_a]
        elem_b = FP8_SEL[args.dtype_b]
        if layout_a == "zN":
            a_t = create_tla_tensor(
                host_nd2nz(inputs.a, "zN").npu(), "zN", elem_a, origin_shape=(m, k)
            )
        else:
            a_t = create_tla_tensor(inputs.a.npu(), layout_a, elem_a)
        if layout_b == "nZ":
            b_t = create_tla_tensor(
                host_nd2nz(inputs.b, "nZ").npu(), "nZ", elem_b, origin_shape=(k, n)
            )
        else:
            b_t = create_tla_tensor(inputs.b.npu(), layout_b, elem_b)
    c_t = create_tla_tensor(c_npu, "row")
    # A scale block follows its operand's orientation: the transposing routes
    # (col A, row B) interleave their groups, the plain ones do not. zN is A's
    # plain form and nZ is B's, so both take their untransposed tag.
    tag_sa = (
        tla.arch.RowMajorMxScaleA
        if layout_a in ("row", "zN")
        else tla.arch.ColMajorMxScaleA
    )
    tag_sb = (
        tla.arch.RowMajorMxScaleB if layout_b == "row" else tla.arch.ColMajorMxScaleB
    )
    sk_pad = inputs.scale_groups
    sa_t = from_dlpack(
        sa_npu,
        layout_tag=tag_sa,
        origin_shape=(m, sk_pad),
        element_type=tla.Float8E8M0,
    ).mark_layout_dynamic()
    sb_t = from_dlpack(
        sb_npu,
        layout_tag=tag_sb,
        origin_shape=(sk_pad, n),
        element_type=tla.Float8E8M0,
    ).mark_layout_dynamic()

    artifact = tla.compile(
        mx_mmad_kernel,
        a_t,
        b_t,
        sa_t,
        sb_t,
        c_t,
        MxTilingParams(tm=tm, tn=tn, tk=tk),
        options="--npu-arch 3510",
    )
    block_num = get_block_num(args.block_num, args.device, kind="cube")
    artifact(a_t, b_t, sa_t, sb_t, c_t, block_num=block_num)
    torch.npu.synchronize()

    result = c_npu.detach().cpu().float()
    # The shared accumulative-precision check, as every other matmul example
    # uses: the tolerance scales with K, which is what a dot product's error
    # does. A misplaced scale block is off by a factor, not by rounding, so it
    # lands far outside this and is caught.
    passed = compare(result, ref, k, rtol=1.0 / 128.0 if is_fp4 else None)
    print(f"  passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    if not passed:
        d = (result - ref).abs()
        print(
            f"  max_abs_diff={d.max().item():.4f} ref_absmax={ref.abs().max().item():.4f}"
        )
        print(f"  result[0,:4]={result[0, :4].tolist()}")
        print(f"  ref[0,:4]={ref[0, :4].tolist()}")
    return 0 if passed else 1


def _apply_case(args: argparse.Namespace, case: str, parser) -> None:
    """Point `args` at one case: an fp8 operand pair, or a single fp4 encoding."""
    case = case.strip()
    if case in gen_data.FP4_FORMATS:
        args.fp4_case, args.dtype_a, args.dtype_b = case, None, None
        return
    parts = [part.strip() for part in case.split(",")]
    if len(parts) == 2 and all(part in gen_data.FP8_DTYPES for part in parts):
        args.fp4_case = None
        args.dtype_a, args.dtype_b = parts
        return
    parser.error(
        f"--case expects an fp4 encoding {{{', '.join(sorted(gen_data.FP4_FORMATS))}}} "
        f"or an fp8 pair DTYPE_A,DTYPE_B with each in "
        f"{{{', '.join(sorted(gen_data.FP8_DTYPES))}}}, got {case!r}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--block-num", type=int, default=-1)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=1024)
    # One per side: the operand and its scale block share an orientation. zN / nZ
    # hand the operand over already packed in fractal order, so the GM -> L1 hop
    # is a block move instead of an ND2NZ / DN2NZ re-layout; for fp4 they are the
    # only non-transposing alternative to the default row / col pair.
    p.add_argument("--layout-a", choices=("row", "col", "zN"), default="row")
    p.add_argument("--layout-b", choices=("row", "col", "nZ"), default="col")
    p.add_argument(
        "--case",
        action="append",
        metavar="FP4_FORMAT|DTYPE_A,DTYPE_B",
        help="case to run; repeat to run several in one process",
    )
    args = p.parse_args()

    cases = args.case or ["f8e4m3fn,f8e4m3fn"]
    failures = 0
    for case in cases:
        _apply_case(args, case, p)
        rc = run(args)
        failures += rc
    print(f"SUMMARY passed={len(cases) - failures}/{len(cases)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
