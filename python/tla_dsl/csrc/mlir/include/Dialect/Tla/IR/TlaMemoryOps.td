//===----------------------------------------------------------------------===//
// Memory-to-memory data movement: copy, copy_mx, img2col and fill, plus the parameter
// bundle tla.copy takes on the L0C->dst path.
//
// tla.load / tla.store are not here: they move UB <-> vector register and are
// only valid inside a tla.vec.func, so they live in TlaSimdOps.td.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_CopyL0C2DstParamsOp : Tla_Op<"CopyL0C2DstParams", [Pure, AttrSizedOperandSegments]> {
  let summary = "CopyL0C2DstParams op";
  let arguments = (ins
  // attributes
    I64Attr:$unit_flag,
    BoolAttr:$relu_enable,
    Tla_QuantModeAttr:$quant_mode,
    Tla_L0C2UBModeAttr:$l0c2ub_mode,
  // operands
    Optional<F32>:$quant_scale,
    Optional<I32>:$quant_offset,
    Optional<Tla_TensorType>:$quant_tensor
  );
  let results = (outs Tla_CopyL0C2DstParamsType:$params);
}

def Tla_CopyOp : Tla_Op<"copy", []> {
  let summary = "Copy";
  let description = [{
    Copy between tensors/views in different address spaces. Cube data-path
    copies (GM->L1, L1->L0A, L1->L0B, L0C->GM, L0C->UB, L0C->L1, L1->UB) must be nested
    inside a ``tla.cube`` region; vector staging copies (GM->UB, UB->GM, UB->L1)
    must be nested inside a ``tla.vector`` region.

    The atomic operation is enabled if ``atomic_mode`` provided, and currently only ``AtomicMode.ADD`` is supported, which is lowered to HIVM ``SetAtomicOp``

    A microscaling (MX) L1->L0A/L0B load is ``tla.copy_mx``, which carries the
    operand's e8m0 scale block. The frontend spells that as ``tla.copy`` with a
    ``scale=`` argument, mirroring AscendC's five-argument ``LoadData``, but it
    lowers to the separate op.
  }];
  let arguments = (ins
    Tla_TensorType:$dst, Tla_TensorType:$src,
    Optional<AnyTypeOf<[Tla_CopyL0C2DstParamsType]>>:$params,
    OptionalAttr<Tla_AtomicModeAttr>:$atomic_mode);
  let assemblyFormat = [{
    $dst `,` $src (`,` $params^)? attr-dict `:` qualified(type($dst)) `,` qualified(type($src)) (`,` qualified(type($params))^)?
  }];
  let hasVerifier = 1;
}

def Tla_CopyMxOp : Tla_Op<"copy_mx", []> {
  let summary = "Microscaling (MX) L1->L0 copy";
  let description = [{
    Copy an MX operand tile from L1 to L0A/L0B, attaching its shared-exponent
    scale block. Must be nested inside a ``tla.cube`` region.

    A separate op from ``tla.copy`` so that the plain copy keeps a fixed operand
    shape. The frontend still spells this as ``tla.copy(dst, src, scale=...)``,
    mirroring AscendC, where the MX load is the five-argument overload of
    ``LoadData`` rather than a distinct entry point.

    On this hardware the scale is consumed by the *load*, not by the matmul:
    ``tla.mmad_mx`` takes no scale operand, and the MX association is
    established when the operand lands in L0.

    ``scale`` is an L1 tile of OCP ``e8m0`` shared exponents, one per
    ``MX_SCALE_GROUP_NUM`` (32) elements along K. Its element type is the
    dialect's own ``!tla.f8e8m0`` (``tla.Float8E8M0`` on the Python side); like
    the other cube element formats it carries no LLVM lowering, so at the
    memref boundary it is buffered as i8, one byte per exponent. It must be
    tagged ``zZMxScale`` for an L1A and ``nNMxScale`` for an L1B.

    The destination L0 tile is what makes the matmul an MX matmul, so it must be
    consumed by ``tla.mmad_mx`` rather than ``tla.mmad``.
  }];
  let arguments = (ins Tla_TensorType:$dst, Tla_TensorType:$src, Tla_TensorType:$scale);
  let assemblyFormat = [{
    $dst `,` $src `,` $scale attr-dict `:`
    qualified(type($dst)) `,` qualified(type($src)) `,` qualified(type($scale))
  }];
  let hasVerifier = 1;
}

def Tla_Img2colOp : Tla_Op<"img2col", []> {
  let summary = "img2col L1->L0 load";
  let description = [{
    Load a tile from an L1 feature map into L0A (or L0B), expanding it on the way
    with the hardware's img2col unit (``load3dv2`` / ``img2colv2_cbuf_to_ca``).
    Must be nested inside a ``tla.cube`` region.

    This is the only conv-specific instruction in the dialect. It turns an
    ``(hi, wi)`` feature-map window into the implicit-GEMM matrix
    ``(M, K) = (ho * wo, cin1 * kh * kw * C0)``, applying padding, stride and
    dilation as it goes; what comes out of it is an ordinary L0 operand that
    ``tla.mmad`` consumes with no knowledge that a convolution is happening.
    Convolution is therefore a property of the *load*, not of the matmul.

    ``dst`` is the L0 tile and gives ``mExtension``/``kExtension`` through its
    shape and ``mStartPt``/``kStartPt`` through its coordinate -- so M and K of
    the implicit GEMM can be tiled by re-issuing this op into a different window
    of the same L1 tile, with no extra L1 traffic.

    ``src`` is the L1 feature-map tile. The feature-map extent
    (``fmap_h``/``fmap_w``, the ``set_fmatrix`` operands) is passed explicitly
    rather than read off the tile, because the L1 tile is a flat NC1HWC0 run of
    bytes whose logical rows are ``hi * wi``, not a 4-D spatial view.

    ``pad_*`` is the padding of *this* block, not of the whole problem: a tiled
    convolution pads only the blocks that touch the feature-map border, so these
    are runtime values. Padding is filled with ``pad_value``.

    Geometry operands are ``index``. Filter extents must be constants in
    1..511; source channels and destination K extent/coordinate must be static.
    The source descriptor supplies channelSize independently of the K window.
    K extents and coordinates span whole C1 slices (filter_h * filter_w * C0)
    and must stay within the source channel/filter extent. Channel count and
    K extent fit in 1..65535, and the K coordinate in 0..65535. Feature-map
    extent, padding, stride and dilation may be runtime values.
  }];
  let arguments = (ins Tla_TensorType:$dst, Tla_TensorType:$src,
                       Index:$fmap_h, Index:$fmap_w,
                       Index:$filter_h, Index:$filter_w,
                       Index:$stride_h, Index:$stride_w,
                       Index:$dilation_h, Index:$dilation_w,
                       Index:$pad_left, Index:$pad_right,
                       Index:$pad_top, Index:$pad_bottom,
                       I32Attr:$pad_value);
  let assemblyFormat = [{
    $dst `,` $src `,` `fmap` `(` $fmap_h `,` $fmap_w `)`
    `filter` `(` $filter_h `,` $filter_w `)`
    `stride` `(` $stride_h `,` $stride_w `)`
    `dilation` `(` $dilation_h `,` $dilation_w `)`
    `pad` `(` $pad_left `,` $pad_right `,` $pad_top `,` $pad_bottom `)`
    attr-dict `:` qualified(type($dst)) `,` qualified(type($src))
  }];
  let hasVerifier = 1;
}

def Tla_FillOp : Tla_Op<"fill", []> {
  let summary = "Fill a region of an L1 zN/nZ tile with a scalar bit pattern";
  let description = [{
    Fill ``dst`` with ``value`` over the region ``[coord, coord + shape)``.
    Cube only: the destination must be an L1 tile tagged ``zN`` or ``nZ``.
    Supported elements are i16/ui16/f16/bf16/i32/ui32/f32 and packed fp4/fp8.
    Packed fp4/fp8 require a constant zero bit pattern.

    ``value`` carries the target element's bits in i32; 16-bit destinations
    use its low 16 bits.
    The device aligns both ends of the layout's C0 axis up to 32-byte
    boundaries and writes ``[alignUp(start), alignUp(end))``. A partial
    starting block is untouched; the final block may include physical
    Padding beyond the logical extent, which must fit the allocation.
  }];
  let arguments = (ins Tla_TensorType:$dst, I32:$value,
                         Tla_CoordType:$coord, Tla_ShapeType:$shape);
  let assemblyFormat = [{
    $dst `,` $value `,` $coord `,` $shape attr-dict `:`
    qualified(type($dst)) `,` type($value) `,` type($coord) `,` type($shape)
  }];
  let hasVerifier = 1;
}
