//===----------------------------------------------------------------------===//
// Pointer, layout and tensor construction / view ops.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_PtrToIntOp : Tla_Op<"ptrtoint", [Pure]> {
  let summary = "tla.ptr to i64 address (``tla.ptrtoint``)";
  let description = [{
    Reads the address of a ``!tla.ptr`` as ``i64``. This is the direct inverse of
    ``tla.inttoptr``. SCF carries that integer; ``tla.inttoptr`` rebuilds the
    pointer where a tensor base is required. VectorRegion replaces it with the
    hardware ``.post`` pointer.
  }];
  let arguments = (ins Tla_PtrType:$ptr);
  let results = (outs I64:$addr);
  let assemblyFormat = [{
    $ptr attr-dict `:` qualified(type($ptr)) `->` type($addr)
  }];
}

def Tla_IntToPtrOp : Tla_Op<"inttoptr", [Pure]> {
  let summary = "Integer SSA to tla.ptr (``tla.inttoptr`` parity)";
  let description = [{
    Reinterprets an integer-like SSA address as the given ``!tla.ptr`` type.
    ``tla-lower-ptr`` normally converts it back to an ``i64`` byte address; an
    instance may remain temporarily at a ``tla.make_tensor{,_like}`` boundary so
    tensor lowering can construct a consumer-local memref descriptor.
  }];
  let arguments = (ins AnyType:$addr);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $addr attr-dict `:` type($addr) `->` qualified(type($ptr))
  }];
}

def Tla_RecastPtrOp : Tla_Op<"recast_ptr", [Pure]> {
  let summary = "Recast pointer element type (``tla.recast_ptr`` parity, dtype-only)";
  let description = [{
    Reinterprets a ``!tla.ptr`` SSA as a different logical pointee type while preserving
    address space and alignment.
    **No swizzle** metadata (Tla does not model swizzle on ``!tla.ptr``).
  }];
  let arguments = (ins Tla_PtrType:$src);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` qualified(type($ptr))
  }];
}

def Tla_TensorPtrOp : Tla_Op<"tensor_ptr", [Pure]> {
  let summary = "Extract the backing pointer from a tla.tensor";
  let description = [{
    Yields the ``!tla.ptr`` SSA value backing a ``!tla.tensor``. Kernel arguments
    are represented inside the function by a root ``tla.tensor_desc``.
  }];
  let arguments = (ins Tla_TensorType:$src);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` qualified(type($ptr))
  }];
  let hasVerifier = 1;
}

def Tla_PtrAddOp : Tla_Op<"ptr_add", [Pure]> {
  let summary = "Offset a tla.ptr by a scalar element count";
  let description = [{
    Returns a ``!tla.ptr`` advanced by ``offset`` **elements** of the pointer's pointee
    type (not bytes). The result preserves the source pointee type and address space;
    alignment is refined conservatively from a known constant offset (gcd), otherwise
    dropped to 1. The offset operand is an integer-or-index scalar SSA value.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_IntegerOrIndexType:$offset);
  let results = (outs Tla_PtrType:$result);
  let assemblyFormat = [{
    $ptr `,` $offset attr-dict `:` qualified(type($ptr)) `,` type($offset) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

// Scratch alloc uses a static ``size_bytes`` attribute (``i64``) on ``tla.alloc_ptr``.
def Tla_AllocPtrOp : Tla_Op<"alloc_ptr", [Pure]> {
  let summary = "Static on-chip scratch bytes";
  let description = [{
    Reserves one statically assigned ``size_bytes`` on-chip scratch slot. The slot
    lifetime is the whole kernel: executing an alloc nested in a region or loop
    yields the same address on every dynamic execution. Address space and element
    alignment appear only on ``!tla.ptr<pointee, addrspace, align>``.
    ``tla-lower-ptr`` replaces the result with its static ``i64`` byte address.
  }];
  let arguments = (ins I64Attr:$size_bytes);
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    `{` `size_bytes` `=` $size_bytes `}` attr-dict `->` qualified(type($ptr))
  }];
  let hasVerifier = 1;
}

def Tla_DynamicUbBaseOp : Tla_Op<"dynamic_ub_base", [Pure]> {
  let summary = "Base address of the launch-sized UB region";
  let description = [{
    Yields the address just above every static ``tla.alloc_ptr`` in UB -- the
    start of a region whose size is chosen at launch rather than compiled in.

    Reserves nothing. Static allocations occupy the bottom of UB, so the region
    always begins at their high-water mark: a compile-time constant even though
    its extent is not. ``tla-lower-ptr`` replaces the result with that address,
    rounded up to the pointer's alignment.

    How much of the region is backed by real UB is set per launch. Touching
    past it is not diagnosed and does not reliably fault; it corrupts whatever
    the unclaimed UB is being used for.
  }];
  let results = (outs Tla_PtrType:$ptr);
  let assemblyFormat = [{
    attr-dict `->` qualified(type($ptr))
  }];
  let hasVerifier = 1;
}

def Tla_MakeTensorLikeOp : Tla_Op<"make_tensor_like", []> {
  let summary = "Make tensor like";
  let description = [{
    Reconstruct a typed local tensor from a raw allocation pointer and a reference tensor.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_TensorType:$like, Tla_LayoutTagAttr:$layoutTag);
  let results = (outs Tla_TensorType:$tile);
  let assemblyFormat = [{
    $ptr `like` $like `layoutTag` `(` $layoutTag `)` attr-dict `:` qualified(type($ptr)) `,` qualified(type($like)) `->` qualified(type($tile))
  }];
}

def Tla_MakeTensorOp : Tla_Op<"make_tensor", [Pure]> {
  let summary = "Construct a tensor from an explicit pointer, layout, and coord";
  let description = [{
    Build a `!tla.tensor<layout, coord, ptr>` from explicit operands. Unlike
    `tla.make_tensor_like` (which clones layout/coord from a reference tensor), this op
    takes the layout and coord directly as operands, mirroring the tensor type's
    `(layout, coord, ptr)` structure. Address space and element type come from the `ptr`
    operand's `!tla.ptr` type; the layout tag, shape, stride, and origin come from the
    `!tla.layout` operand.
  }];
  let arguments = (ins Tla_PtrType:$ptr, Tla_LayoutType:$layout, Tla_CoordType:$coord);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $ptr `,` $layout `,` $coord attr-dict `:`
    qualified(type($ptr)) `,` qualified(type($layout)) `,` qualified(type($coord)) `->` qualified(type($result))
  }];
}

def Tla_TensorDescOp : Tla_Op<"tensor_desc", [Pure]> {
  let summary = "A tile's fully-resolved tensor descriptor as a single op";
  let description = [{
    Materialized by ``tla-lower-tensor-desc`` from a tensor-view producer chain.
    ``base`` is the resolved root storage: either a builtin ``memref`` kernel
    argument or a ``!tla.ptr`` whose pointer transforms have already been applied.
    The remaining operands are the descriptor's SSA-valued four-slot layout shape
    and stride, two-dimensional origin shape, and two-dimensional coordinate.
    Linear layouts use ``[M, N, 1, 1]`` shape and ``[sM, sN, 1, 1]`` stride;
    NZFamily layouts preserve their four shape/stride leaves.

    Downstream passes consume this op without interpreting the producer chain.
  }];
  let arguments = (ins
    AnyTypeOf<[Tla_PtrType, AnyMemRef]>:$base,
    Index:$shape0, Index:$shape1, Index:$shape2, Index:$shape3,
    Index:$stride0, Index:$stride1, Index:$stride2, Index:$stride3,
    Index:$origin_shape0, Index:$origin_shape1,
    Index:$coord0, Index:$coord1,
    Variadic<Index>:$batch_info);
  let results = (outs Tla_TensorType:$result);
  let assemblyFormat = [{
    $base `shape` `[` $shape0 `,` $shape1 `,` $shape2 `,` $shape3 `]`
    `stride` `[` $stride0 `,` $stride1 `,` $stride2 `,` $stride3 `]`
    `origin_shape` `[` $origin_shape0 `,` $origin_shape1 `]`
    `coord` `[` $coord0 `,` $coord1 `]`
    (`batch` `[` $batch_info^ `]`)?
    attr-dict `:` qualified(type($base)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tla_MakeShapeOp : Tla_Op<"make_shape", [Pure]> {
  let summary = "Shape value";
  let description = [{
    Build a ``!tla.shape`` from operands for each dynamic dimension; rank and static sizes are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_ShapeType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeCoordOp : Tla_Op<"make_coord", [Pure]> {
  let summary = "Coord value";
  let description = [{
    Build a ``!tla.coord`` from operands for each dynamic leaf; structure and static leaves are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_CoordType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeStrideOp : Tla_Op<"make_stride", [Pure]> {
  let summary = "Stride value";
  let description = [{
    Build a ``!tla.stride`` from operands for each dynamic leaf; structure and static leaves are fixed by the result type.
  }];
  let arguments = (ins Variadic<Index>:$dyn_elems);
  let results = (outs Tla_StrideType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Tla_MakeLayoutOp : Tla_Op<"make_layout", [Pure]> {
  let summary = "Layout value";
  let description = [{
    Combine a packed ``!tla.shape`` and ``!tla.stride`` into a ``!tla.layout``, analogous to ``tla.make_layout(shape, stride)``.
    The result type encodes tuple-tree structure as ``!tla.layout<!tla.shape<shape_repr>, !tla.stride<stride_repr>, layout_tag>``, or
    ``!tla.layout<!tla.shape<shape_repr>, !tla.stride<stride_repr>, !tla.shape<origin_shape_repr>, layout_tag>`` when an explicit origin shape is used.

    Optional ``origin_shape`` records the logical extent (e.g. pre-padding matrix size). When omitted,
    consumers should treat the layout's logical origin shape as identical to ``shape``.
  }];
  let arguments = (ins Tla_ShapeType:$shape,
                       Tla_StrideType:$stride,
                       Optional<Tla_ShapeType>:$origin_shape,
                       OptionalAttr<Tla_LayoutTagAttr>:$layoutTag);
  let results = (outs Tla_LayoutType:$result);
  let assemblyFormat = [{
    $shape `,` $stride (`origin` $origin_shape^)? attr-dict `:`
    qualified(type($shape)) `,` qualified(type($stride))
    (`origin` qualified(type($origin_shape))^)? `->` qualified(type($result))
  }];
}

def Tla_TileViewOp : Tla_Op<"tile_view", []> {
  let summary = "Tile view";
  let description = [{
    Create a tensor view over an existing ``!tla.tensor`` using packed
    ``!tla.shape`` and ``!tla.coord`` values.
  }];
  let arguments = (ins Tla_TensorType:$source, Tla_ShapeType:$shape, Tla_CoordType:$coord, OptionalAttr<Tla_LayoutTagAttr>:$layoutTag);
  let results = (outs Tla_TensorType:$view);
  let assemblyFormat = [{
    $source `,` $shape `,` $coord attr-dict `:` qualified(type($source)) `,` qualified(type($shape)) `,` qualified(type($coord)) `->` qualified(type($view))
  }];
}
