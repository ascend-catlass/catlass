# tests/test_framework_overview_interfaces.py
# Mirrors the interface table in docs/catlass_dsl_framework_overview.md:
# one test per Python API in core_api.py; entries without a binding are skipped in comments.

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import pytest

from catlass._mlir import ir as mlir_ir
from catlass import _tla_type_bridge
import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError


# -----------------------------------------------------------------------------
# 1. make_coord(...) → tla.make_coord (lowered Tla IR / capture only)
# -----------------------------------------------------------------------------


@tla.kernel
def _kernel_make_coord() -> None:
    _ = tla.make_coord(0, 1)


def test_interface_make_coord() -> None:
    """Framework table: make_coord emits tla.make_coord."""
    mlir = _kernel_make_coord.dump_mlir()
    assert "tla.make_coord" in mlir
    assert "!tla.coord<" in mlir
    assert "0" in mlir and "1" in mlir


# -----------------------------------------------------------------------------
# 2. make_shape / make_coord / make_stride — tuple-tree nesting (lowered Tla IR)
# -----------------------------------------------------------------------------


@tla.kernel
def _kernel_make_shape_coord_stride_nested(dim: tla.types.TlaIndex) -> None:
    # Flat shapes: 1D, 3D static, one dynamic extent
    _ = tla.make_shape(8)
    _ = tla.make_shape(4, 8, 16)
    _ = tla.make_shape(dim, 16)
    # Nested tuple tree (Tla-style): one SSA leaf inside a sub-shape
    _ = tla.make_shape((4, 8), 16)
    _ = tla.make_shape((dim, 8), 16)
    # Flat coord / stride
    _ = tla.make_coord(0, 1)
    _ = tla.make_stride(1, 100)
    # Nested coord / stride trees
    _ = tla.make_coord((0, 1), 2)
    _ = tla.make_stride((1, 16), 128)


def test_interface_make_shape_coord_stride_nested() -> None:
    """make_shape / make_coord / make_stride emit tla.make_*; flat + nested tuple trees."""
    mlir = _kernel_make_shape_coord_stride_nested.dump_mlir(type_args=(4,))
    assert "tla.make_shape" in mlir
    assert "tla.make_coord" in mlir
    assert "tla.make_stride" in mlir
    assert "!tla.shape<8>" in mlir
    assert "!tla.shape<4, 8, 16>" in mlir or "!tla.shape<4,8,16>" in mlir
    assert "!tla.shape<?,16>" in mlir
    assert "?" in mlir
    # Nested shape/coord/stride: parentheses encode sub-trees in the type string.
    m = mlir.replace(" ", "")
    assert "(4,8)" in m
    assert "(?,8),16" in m
    assert "(0,1)" in m
    assert "(1,16)" in m


# -----------------------------------------------------------------------------
# 3. make_layout → tla.make_layout (+ tla.make_shape / tla.make_stride), Tla-like IR
# -----------------------------------------------------------------------------


@tla.kernel
def _kernel_make_layout_comprehensive(dim: tla.types.TlaIndex) -> None:
    # Nested tuple tree (Tla-like): shape/stride SSA → make_layout
    sh = tla.make_shape((1, 1, 4))
    st = tla.make_stride((1, 1, 1))
    _ = tla.make_layout(sh, st)
    # Flat rank-3 (omitted origin_shape → two SSA operands)
    sh = tla.make_shape(1, 1, 4)
    st = tla.make_stride(1, 1, 1)
    _ = tla.make_layout(sh, st)
    # Dynamic dim in nested shape + layout (same tuple style as make_shape / make_stride tests)
    sh = tla.make_shape((dim, 8), 16)
    st = tla.make_stride((1, 16), 128)
    _ = tla.make_layout(sh, st)
    # Explicit origin_shape → third shape SSA
    sh = tla.make_shape(16, 16)
    st = tla.make_stride(16, 1)
    org = tla.make_shape(13, 7)
    _ = tla.make_layout(sh, st, origin_shape=org)
    # origin_shape same as shape SSA → folds to two operands
    sh = tla.make_shape(4, 8)
    st = tla.make_stride(8, 1)
    _ = tla.make_layout(sh, st, origin_shape=sh)
    # IR output:
    # module {
    #   "tla.func"() ({
    #   ^bb0(%arg0: index):
    #     %0 = "tla.make_shape"() : () -> !tla.shape<(1,1,4)>
    #     %1 = "tla.make_stride"() : () -> !tla.stride<(1,1,1)>
    #     %2 = "tla.make_layout"(%0, %1) : (!tla.shape<(1,1,4)>, !tla.stride<(1,1,1)>) -> !tla.layout<!tla.shape<(1,1,4)>, !tla.stride<(1,1,1)>, !tla.shape<(1,1,4)>, RowMajor>
    #     %3 = "tla.make_shape"() : () -> !tla.shape<1,1,4>
    #     %4 = "tla.make_stride"() : () -> !tla.stride<1,1,1>
    #     %5 = "tla.make_layout"(%3, %4) : (!tla.shape<1,1,4>, !tla.stride<1,1,1>) -> !tla.layout<!tla.shape<1,1,4>, !tla.stride<1,1,1>, !tla.shape<1,1,4>, RowMajor>
    #     %6 = "tla.make_shape"(%arg0) : (index) -> !tla.shape<(?,8),16>
    #     %7 = "tla.make_stride"() : () -> !tla.stride<(1,16),128>
    #     %8 = "tla.make_layout"(%6, %7) : (!tla.shape<(?,8),16>, !tla.stride<(1,16),128>) -> !tla.layout<!tla.shape<(?,8),16>, !tla.stride<(1,16),128>, !tla.shape<(?,8),16>, RowMajor>
    #     %9 = "tla.make_shape"() : () -> !tla.shape<16,16>
    #     %10 = "tla.make_stride"() : () -> !tla.stride<16,1>
    #     %11 = "tla.make_shape"() : () -> !tla.shape<13,7>
    #     %12 = "tla.make_layout"(%9, %10, %11) : (!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<13,7>) -> !tla.layout<!tla.shape<16,16>, !tla.stride<16,1>, !tla.shape<13,7>, RowMajor>
    #     %13 = "tla.make_shape"() : () -> !tla.shape<4,8>
    #     %14 = "tla.make_stride"() : () -> !tla.stride<8,1>
    #     %15 = "tla.make_layout"(%13, %14) : (!tla.shape<4,8>, !tla.stride<8,1>) -> !tla.layout<!tla.shape<4,8>, !tla.stride<8,1>, !tla.shape<4,8>, RowMajor>
    #     "tla.return"() : () -> ()
    #   }) {function_type = (index) -> (), sym_name = "_kernel_make_layout_comprehensive"} : () -> ()
    # }


def test_interface_make_layout_comprehensive() -> None:
    """make_layout: nested + flat + dynamic dim + optional origin; SSA operand counts per site."""
    mlir = _kernel_make_layout_comprehensive.dump_mlir(type_args=(4,))
    assert "tla.make_shape" in mlir
    assert "tla.make_stride" in mlir
    assert "tla.make_layout" in mlir
    assert "!tla.layout<" in mlir
    m = mlir.replace(" ", "")
    assert (
        "!tla.layout<!tla.shape<(1,1,4)>,!tla.stride<(1,1,1)>,!tla.shape<(1,1,4)>,RowMajor>"
        in m
    )
    assert (
        "!tla.layout<!tla.shape<1,1,4>,!tla.stride<1,1,1>,!tla.shape<1,1,4>,RowMajor>"
        in m
    )
    assert "(?,8),16" in m
    assert "(1,16),128" in m
    assert (
        "!tla.layout<!tla.shape<16,16>,!tla.stride<16,1>,!tla.shape<13,7>,RowMajor>"
        in m
    )
    assert (
        "!tla.layout<!tla.shape<4,8>,!tla.stride<8,1>,!tla.shape<4,8>,RowMajor>" in m
    )
    assert "13,7" in m or "origin" in mlir

    layout_operand_counts: list[int] = []
    for line in mlir.splitlines():
        if "tla.make_layout" not in line:
            continue
        pre = line.split(":", 1)[0] if ":" in line else line
        idx = pre.find("tla.make_layout")
        tail = pre[idx + len("tla.make_layout") :].strip()
        layout_operand_counts.append(tail.count("%"))
    # Order: nested, flat, dynamic dim, explicit origin, origin same as shape
    assert layout_operand_counts == [2, 2, 2, 3, 2]


# -----------------------------------------------------------------------------
# 4. make_ptr → tla.inttoptr + !tla.ptr<
# -----------------------------------------------------------------------------


@tla.kernel
def _kernel_make_ptr() -> None:
    _ = tla.make_ptr(tla.Float32, 4096, mem_space=tla.AddressSpace.l0c)
    _ = tla.make_ptr(None, 8192, mem_space=tla.AddressSpace.l0c, assumed_align=16)
    # IR output:
    # module {
    #   "tla.func"() ({
    #     %c4096_i32 = arith.constant 4096 : i32
    #     %0 = "tla.inttoptr"(%c4096_i32) : (i32) -> !tla.ptr<f32, l0c, 4>
    #     %c8192_i32 = arith.constant 8192 : i32
    #     %1 = "tla.inttoptr"(%c8192_i32) : (i32) -> !tla.ptr<i8, l0c, 16>
    #     "tla.return"() : () -> ()
    #   }) {function_type = () -> (), sym_name = "_kernel_make_ptr"} : () -> ()
    # }


def test_interface_make_ptr() -> None:
    """make_ptr lowers to ``tla.inttoptr`` with ``!tla.ptr<..., l0c, align>`` (see core_api)."""
    mlir = _kernel_make_ptr.dump_mlir()
    # Module shell may print as custom ``tla.func @name()`` or generic ``"tla.func"() ({...})``
    # depending on MLIR / dialect printers; assert the lowered body we care about.
    assert "inttoptr" in mlir
    assert "4096" in mlir
    assert "8192" in mlir
    assert "<f32, l0c, 4>" in mlir
    assert "<i8, l0c, 16>" in mlir


# -----------------------------------------------------------------------------
# 5. recast_ptr → ``make_ptr`` + ``tla.recast_ptr``
# -----------------------------------------------------------------------------


@tla.kernel
def _kernel_recast_ptr() -> None:
    p_l1 = tla.make_ptr(tla.Int8, 0, mem_space=tla.AddressSpace.l1, assumed_align=512)
    _ = tla.recast_ptr(p_l1, dtype=tla.Float16)
    p = tla.make_ptr(tla.Float32, 4096, mem_space=tla.AddressSpace.l0c)
    _ = tla.recast_ptr(p, dtype=tla.Float16)
    # IR output:
    # module {
    #   "tla.func"() ({
    #     %c0_i32 = arith.constant 0 : i32
    #     %0 = "tla.inttoptr"(%c0_i32) : (i32) -> !tla.ptr<i8, l1, 512>
    #     %1 = "tla.recast_ptr"(%0) : (!tla.ptr<i8, l1, 512>) -> !tla.ptr<f16, l1, 512>
    #     %c4096_i32 = arith.constant 4096 : i32
    #     %2 = "tla.inttoptr"(%c4096_i32) : (i32) -> !tla.ptr<f32, l0c, 4>
    #     %3 = "tla.recast_ptr"(%2) : (!tla.ptr<f32, l0c, 4>) -> !tla.ptr<f16, l0c, 4>
    #     "tla.return"() : () -> ()
    #   }) {function_type = () -> (), sym_name = "_kernel_recast_ptr"} : () -> ()
    # }


def test_interface_recast_ptr() -> None:
    """recast_ptr accepts only ``!tla.ptr``; use ``make_ptr`` then ``tla.recast_ptr``."""
    mlir = _kernel_recast_ptr.dump_mlir()
    assert "tla.inttoptr" in mlir
    assert "tla.recast_ptr" in mlir
    assert "!tla.ptr<f16, l1, 512>" in mlir
    assert "!tla.ptr<f16, l0c, 4>" in mlir


@tla.kernel
def _kernel_allocate_typed_ptr() -> None:
    _ = tla.allocate((16, 16), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 256)
    _ = tla.allocate(((2, 4), 8), tla.Int16, tla.AddressSpace.l0c, 1024)


def test_interface_allocate_typed_ptr() -> None:
    """tla.allocate emits typed alloc_ptr directly, without tla.recast_ptr."""
    mlir = _kernel_allocate_typed_ptr.dump_mlir()
    assert mlir.count("tla.alloc_ptr") == 3
    assert "tla.recast_ptr" not in mlir
    # Every request lands on a block: 512 bytes of f16 and of f32, and 128 bytes
    # of i16 rounded up to the 1024-byte L0C block.
    assert mlir.count("size_bytes = 512") == 2
    assert "size_bytes = 1024" in mlir
    assert "!tla.ptr<f16, l1, 512>" in mlir
    assert "!tla.ptr<f32, ub, 256>" in mlir
    assert "!tla.ptr<i16, l0c, 1024>" in mlir


# -----------------------------------------------------------------------------
# 7. tile_view → tla.tile_view (+ tla.make_shape / tla.make_coord)
#    make_tensor_like → tla.make_tensor_like (layoutTag StrAttr + result tensor type)
# -----------------------------------------------------------------------------


@tla.kernel
def _kernel_make_tensor_like_supported_layout_tags(mem: tla.Tensor) -> None:
    root = tla.tile_view(mem, tla.make_shape(32, 32), tla.make_coord(1, 2))
    l1_ptr = tla.allocate((32, 32), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.make_tensor_like(l1_ptr, root, tla.arch.RowMajor)
    _ = tla.make_tensor_like(l1_ptr, root, tla.arch.ColumnMajor)
    _ = tla.make_tensor_like(l1_ptr, root, tla.arch.zN)
    _ = tla.make_tensor_like(l1_ptr, root, tla.arch.nZ)
    _ = tla.make_tensor_like(l1_ptr, root, tla.arch.L0Clayout)
    # IR output:
    # module {
    #   "tla.func"() ({
    #   ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>):
    #     %0 = "tla.make_shape"() : () -> !tla.shape<32,32>
    #     %1 = "tla.make_coord"() : () -> !tla.coord<1,2>
    #     %2 = "tla.make_coord"() : () -> !tla.coord<32,64>
    #     %3 = "tla.tile_view"(%arg0, %0, %2) : (!tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>, !tla.shape<32,32>, !tla.coord<32,64>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<32,64>, !tla.ptr<f16, gm, 2>>
    #     %4 = "tla.alloc_ptr"() {size_bytes = 2048 : i64} : () -> !tla.ptr<f16, l1, 512>
    #     %5 = "tla.make_tensor_like"(%4, %3) {layoutTag = #tla.layout_tag<RowMajor>} : (!tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<32,64>, !tla.ptr<f16, gm, 2>>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    #     %6 = "tla.make_tensor_like"(%4, %3) {layoutTag = #tla.layout_tag<ColumnMajor>} : (!tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<32,64>, !tla.ptr<f16, gm, 2>>) -> !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    #     %7 = "tla.make_tensor_like"(%4, %3) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<32,64>, !tla.ptr<f16, gm, 2>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    #     %8 = "tla.make_tensor_like"(%4, %3) {layoutTag = #tla.layout_tag<nZ>} : (!tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<32,64>, !tla.ptr<f16, gm, 2>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(1,512),(16,256)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    #     %10 = "tla.make_tensor_like"(%4, %3) {layoutTag = #tla.layout_tag<L0Clayout>} : (!tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<32,64>, !tla.ptr<f16, gm, 2>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    #     "tla.return"() : () -> ()
    #   }) {function_type = (!tla.tensor<!tla.layout<!tla.shape<128,128>, !tla.stride<128,1>, !tla.shape<128,128>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>) -> (), sym_name = "_kernel_make_tensor_like_supported_layout_tags"} : () -> ()
    # }


def test_interface_make_tensor_like_supported_layout_tags(compiler_tlair) -> None:
    """``make_tensor_like`` layoutTag spellings in MLIR and recomputed type fragments."""
    root = make_fake_tensor(
               tla.Float16,
               (128, 128),
               (128, 1),
               origin_shape=(128, 128),
               layout_tag=tla.arch.RowMajor,
           )
    mlir = compiler_tlair(
        _kernel_make_tensor_like_supported_layout_tags, type_args=(root,)
    )
    assert mlir.count('"tla.make_tensor_like"') == 5
    assert 'layoutTag = #tla.layout_tag<RowMajor>' in mlir
    assert 'layoutTag = #tla.layout_tag<ColumnMajor>' in mlir
    assert 'layoutTag = #tla.layout_tag<zN>' in mlir
    assert 'layoutTag = #tla.layout_tag<nZ>' in mlir
    assert 'layoutTag = #tla.layout_tag<L0Clayout>' in mlir
    assert (
        "<!tla.layout<!tla.shape<32,32>, !tla.stride<128,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<32,64>, !tla.ptr<f16, gm, 2>>"
        in mlir
    )
    assert (
        "<!tla.layout<!tla.shape<32,32>, !tla.stride<32,1>, !tla.shape<32,32>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>"
        in mlir
    )
    assert (
        "<!tla.layout<!tla.shape<32,32>, !tla.stride<1,32>, !tla.shape<32,32>, ColumnMajor>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>"
        in mlir
    )
    assert (
        "<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>"
        in mlir
    )
    assert (
        "<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(1,512),(16,256)>, !tla.shape<32,32>, nZ>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>"
        in mlir
    )
    assert (
        "<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>"
        in mlir
    )


@tla.kernel
def _kernel_make_tensor_like_ptr_dtype(mem: tla.Tensor) -> None:
    root = tla.tile_view(mem, tla.make_shape(16, 16), tla.make_coord(0, 0))
    ptr = tla.allocate((16, 16), tla.Float32, tla.AddressSpace.l1, 512)
    _ = tla.make_tensor_like(ptr, root, tla.arch.RowMajor)


def _make_f16_tensor() -> tla.Tensor:
    return make_fake_tensor(
               tla.Float16,
               (16, 16),
               (16, 1),
               origin_shape=(16, 16),
               layout_tag=tla.arch.RowMajor,
           )


def test_make_tensor_like_uses_ptr_dtype_instead_of_like_dtype() -> None:
    mlir = _kernel_make_tensor_like_ptr_dtype.dump_mlir(
        type_args=(_make_f16_tensor(),)
    )
    assert "!tla.ptr<f16, gm, 2>" in mlir
    assert "!tla.ptr<f32, l1, 512>" in mlir
    assert "!tla.ptr<f16, l1, 512>" not in mlir


# -----------------------------------------------------------------------------
# 7b. tile_view on zN tensor: remapped fractal shape + static origin_shape bounds
# -----------------------------------------------------------------------------


def _zn_l0a_tensor(*, origin_m: int = 32, origin_n: int = 32) -> tla.Tensor:
    from catlass.core_api import _remap_tensor_like_prefix_fields_for_layout_trees

    remapped = _remap_tensor_like_prefix_fields_for_layout_trees(
        (origin_m, origin_n), "f16", "zN"
    )
    assert remapped is not None
    return make_fake_tensor(
        tla.Float16,
        remapped.shape,
        remapped.stride,
        addrspace=tla.AddressSpace.l0a,
        origin_shape=remapped.origin_shape,
        coord=remapped.coord,
        layout_tag=tla.arch.zN,
    )


@tla.kernel
def _kernel_tile_view_zn_remapped_shape(mem: tla.Tensor) -> None:
    """zN parent: ``tile_view`` result shape uses ``_remap_tensor_like_prefix_fields_for_layout``."""
    _ = tla.tile_view(mem, tla.make_shape(16, 16), tla.make_coord(0, 0))
    # module {
    #   "tla.func"() ({
    #   ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(1,1),(1,1)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 2>>):
    #     %0 = "tla.make_shape"() : () -> !tla.shape<16,16>
    #     %1 = "tla.make_coord"() : () -> !tla.coord<0,0>
    #     %2 = "tla.make_coord"() : () -> !tla.coord<0,0>
    #     %3 = "tla.tile_view"(%arg0, %0, %2) : (!tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(1,1),(1,1)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 2>>, !tla.shape<16,16>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,1),(1,1)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 2>>
    #     "tla.return"() : () -> ()
    #   }) {function_type = (!tla.tensor<!tla.layout<!tla.shape<(16,2),(16,2)>, !tla.stride<(1,1),(1,1)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 2>>) -> (), sym_name = "_kernel_tile_view_zn_remapped_shape"} : () -> ()
    # }


@tla.kernel
def _kernel_tile_view_zn_oob_origin(mem: tla.Tensor) -> None:
    """Tile (8,8) at tile index (2,0) → element offset 16 on M; fails when origin M is 16."""
    _ = tla.tile_view(mem, tla.make_shape(8, 8), tla.make_coord(2, 0))


def test_interface_tile_view_zn_remapped_result_shape() -> None:
    mem = _zn_l0a_tensor()
    mlir = _kernel_tile_view_zn_remapped_shape.dump_mlir(type_args=(mem,))
    assert "tla.tile_view" in mlir
    # Custom assembly prints the result as layout, coord, ptr (no outer ``!tla.tensor<...>``).
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,512)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 2>>"
        in mlir
    )


def test_interface_tile_view_zn_rejects_oob_relative_to_origin_shape() -> None:
    mem = _zn_l0a_tensor(origin_m=16, origin_n=16)
    with pytest.raises(TlaLoweringError, match="out of range"):
        _kernel_tile_view_zn_oob_origin.dump_mlir(type_args=(mem,))


# -----------------------------------------------------------------------------
# 8. tile_view comprehensive: remapped fractal shape + static origin_shape bounds
# -----------------------------------------------------------------------------


@tla.kernel
def _kernel_tile_view_comprehensive(mem: tla.Tensor, tile_row: tla.types.TlaIndex) -> None:
    root = tla.tile_view(mem, tla.make_shape(15, 15), tla.make_coord(3, 3))
    # Nested offset (1,1) × (8,8) = (8,8); parent tile is 15×15 with origin 15×15 after first view.
    # Tile offsets must stay within the parent's cropped logical origin (5,5 after the root view);
    # (1,1)×(8,8) would exceed that on the static path and is rejected by the frontend.
    leaf = tla.tile_view(root, tla.make_shape(8, 8), tla.make_coord(0, 0))
    _ = tla.tile_view(leaf, tla.make_shape(4, 4), tla.make_coord(tile_row, 0))
    # 8 x 8 x f16 bytes — on-chip tile shaped like ``leaf``
    l1_ptr = tla.allocate((8, 8), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.make_tensor_like(l1_ptr, leaf, tla.arch.zN)
    # IR output:
    # module {
    #   "tla.func"() ({
    #   ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<80,80>, !tla.stride<80,1>, !tla.shape<50,50>, RowMajor>, !tla.coord<10,10>, !tla.ptr<f16, gm, 2>>, %arg1: index):
    #     %0 = "tla.make_shape"() : () -> !tla.shape<15,15>
    #     %1 = "tla.make_coord"() : () -> !tla.coord<3,3>
    #     %2 = "tla.make_coord"() : () -> !tla.coord<45,45>
    #     %3 = "tla.tile_view"(%arg0, %0, %2) : (!tla.tensor<!tla.layout<!tla.shape<80,80>, !tla.stride<80,1>, !tla.shape<50,50>, RowMajor>, !tla.coord<10,10>, !tla.ptr<f16, gm, 2>>, !tla.shape<15,15>, !tla.coord<45,45>) -> !tla.tensor<!tla.layout<!tla.shape<15,15>, !tla.stride<80,1>, !tla.shape<5,5>, RowMajor>, !tla.coord<55,55>, !tla.ptr<f16, gm, 2>>
    #     %4 = "tla.make_shape"() : () -> !tla.shape<8,8>
    #     %5 = "tla.make_coord"() : () -> !tla.coord<0,0>
    #     %6 = "tla.make_coord"() : () -> !tla.coord<0,0>
    #     %7 = "tla.tile_view"(%3, %4, %6) : (!tla.tensor<!tla.layout<!tla.shape<15,15>, !tla.stride<80,1>, !tla.shape<5,5>, RowMajor>, !tla.coord<55,55>, !tla.ptr<f16, gm, 2>>, !tla.shape<8,8>, !tla.coord<0,0>) -> !tla.tensor<!tla.layout<!tla.shape<8,8>, !tla.stride<80,1>, !tla.shape<5,5>, RowMajor>, !tla.coord<55,55>, !tla.ptr<f16, gm, 2>>
    #     %8 = "tla.make_shape"() : () -> !tla.shape<4,4>
    #     %9 = "tla.make_coord"(%arg1) : (index) -> !tla.coord<?,0>
    #     %c4 = arith.constant 4 : index
    #     %10 = arith.muli %arg1, %c4 : index
    #     %11 = "tla.make_coord"(%10) : (index) -> !tla.coord<?,0>
    #     %12 = "tla.tile_view"(%7, %8, %11) : (!tla.tensor<!tla.layout<!tla.shape<8,8>, !tla.stride<80,1>, !tla.shape<5,5>, RowMajor>, !tla.coord<55,55>, !tla.ptr<f16, gm, 2>>, !tla.shape<4,4>, !tla.coord<?,0>) -> !tla.tensor<!tla.layout<!tla.shape<4,4>, !tla.stride<80,1>, !tla.shape<4,4>, RowMajor>, !tla.coord<?,55>, !tla.ptr<f16, gm, 2>>
    #     %13 = "tla.alloc_ptr"() {size_bytes = 128 : i64} : () -> !tla.ptr<f16, l1, 512>
    #     %14 = "tla.make_tensor_like"(%13, %7) {layoutTag = #tla.layout_tag<zN>} : (!tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<8,8>, !tla.stride<80,1>, !tla.shape<5,5>, RowMajor>, !tla.coord<55,55>, !tla.ptr<f16, gm, 2>>) -> !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<5,5>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    #     "tla.return"() : () -> ()
    #   }) {function_type = (!tla.tensor<!tla.layout<!tla.shape<80,80>, !tla.stride<80,1>, !tla.shape<50,50>, RowMajor>, !tla.coord<10,10>, !tla.ptr<f16, gm, 2>>, index) -> (), sym_name = "_kernel_tile_view_comprehensive"} : () -> ()
    # }


def test_interface_tile_view_comprehensive_explicit_root_metadata(
    compiler_tlair,
) -> None:
    root = make_fake_tensor(
               tla.Float16,
               (80, 80),
               (80, 1),
               origin_shape=(50, 50),
               coord=(10, 10),
               layout_tag=tla.arch.RowMajor,
           )
    assert root.__tla_type__() == (
        "!tla.tensor<!tla.layout<!tla.shape<80,80>, !tla.stride<80,1>, !tla.shape<50,50>, RowMajor>, !tla.coord<10,10>, !tla.ptr<f16, gm, 2>>"
    )
    mlir = compiler_tlair(_kernel_tile_view_comprehensive, type_args=(root, 0))
    assert "tla.tile_view" in mlir
    assert mlir.count("tla.tile_view") == 3
    assert "tla.make_shape" in mlir
    assert 'layoutTag = #tla.layout_tag<zN>' in mlir
    assert ", l1" in mlir
    assert "#tla.addrspace<" not in mlir
    assert "!tla.ptr<f16, gm, 2>" in mlir
    assert "zN" in mlir
    assert (
        "<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<5,5>, zN>"
        in mlir
    )
    assert "arith.muli" in mlir
    assert "<!tla.layout<!tla.shape<4,4>" in mlir


# -----------------------------------------------------------------------------
# 8. mmad → tla.mmad (acc / lhs / rhs tensor operands; init_c bool attr)
# -----------------------------------------------------------------------------


@tla.kernel
def _kernel_mmad_interface_example(
    lhs: tla.Tensor, rhs: tla.Tensor, acc: tla.Tensor
) -> None:
    """Nested ``make_shape`` tuple trees for the frontend ``mmad`` contract."""
    with tla.cube():
        _ = tla.mmad(acc, lhs, rhs, init_c=True)
    # IR output:
    # module attributes {tla.module_exec_units = "cube"} {
    #   "tla.func"() ({
    #   ^bb0(%arg0: !tla.tensor<!tla.layout<...>, !tla.coord<...>, !tla.ptr<...>>, ...):
    #     "tla.mmad"(%arg2, %arg0, %arg1) {init_c = true} : (...) -> ()  // generic asm (compiler_tlair)
    #     "tla.return"() : () -> ()
    #   }) {tla.exec_units = "cube", function_type = (!tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 2>>, !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0b, 2>>, !tla.tensor<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, L0Clayout>, !tla.coord<0,0>, !tla.ptr<f32, l0c, 4>>) -> (), sym_name = "_kernel_mmad_interface_example"} : () -> ()
    # }


def test_interface_mmad_nested_shape_contract_lowers_at_frontend(
    compiler_tlair,
) -> None:
    """``mmad`` expects acc ``L0Clayout``, lhs ``zN``, and rhs ``nZ``."""
    with mlir_ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        try:
            mlir_ir.Attribute.parse("#tla.compute_order<M_FIRST>", context=ctx)
        except mlir_ir.MLIRError as exc:
            if "compute_order" in str(exc) and "expected valid keyword" in str(exc):
                pytest.skip("linked BiShengIR lacks tla.compute_order enum used by mmad")
            raise
    lhs = make_fake_tensor(
              tla.Float16,
              ((16, 1), (16, 1)),
              ((16, 256), (1, 256)),
              addrspace=tla.AddressSpace.l0a,
              layout_tag=tla.arch.zN,
              origin_shape=(16, 16),
              coord=(0, 0),
          )
    rhs = make_fake_tensor(
              tla.Float16,
              ((16, 1), (16, 1)),
              ((1, 256), (16, 256)),
              addrspace=tla.AddressSpace.l0b,
              layout_tag=tla.arch.nZ,
              origin_shape=(16, 16),
              coord=(0, 0),
          )
    acc = make_fake_tensor(
              tla.Float32,
              ((16, 1), (16, 1)),
              ((16, 256), (1, 256)),
              addrspace=tla.AddressSpace.l0c,
              layout_tag=tla.arch.L0Clayout,
              origin_shape=(16, 16),
              coord=(0, 0),
          )
    mlir = compiler_tlair(_kernel_mmad_interface_example, type_args=(lhs, rhs, acc))
    assert "tla.mmad" in mlir
    assert '"arith.constant"() <{value = true}> : () -> i1' in mlir
    assert '"arith.constant"() <{value = 0 : i64}> : () -> i64' in mlir
    assert "!tla.ptr<f16, l0a, 2>" in mlir
    assert "!tla.ptr<f16, l0b, 2>" in mlir
    assert "!tla.ptr<f32, l0c, 4>" in mlir


# -----------------------------------------------------------------------------
# 7c. tla.copy (GM RowMajor → l1 zN): two f32 copies share one 4096B L1 buffer
#     (pointer_cast reuse). TLA MLIR / TlaCompile 期望见
#     tests/lit/copy/framework-overview-copy-gm-l1-zn-two-tiles.mlir；
#     同路由参考 tests/lit/copy/copy-gm-row-major-to-l1-zn.mlir。
# -----------------------------------------------------------------------------


@tla.kernel
def _kernel_copy_gm_row_major_to_l1_zn(mem: tla.Tensor, mem_i8: tla.Tensor) -> None:
    """两路 f32 ``tile_view``、一次 4096B L1 + ``make_tensor_like`` zN、两次 ``tla.copy`` 写同一 l1 缓冲；
    ``mem_i8`` 仅作第二路 GM 形参（与 ``mem`` 的 shape/origin/stride/dtype 均不同），便于 IR 里出现两条不相同的 ``!tla.tensor<…>`` 入口类型。

    ``make_coord(1,1)``/``(0,0)`` 与 ``32×32`` tile 须在 ``mem`` 的 ``origin_shape`` 裁剪范围内合法（与 ``mem`` 的逻辑 ``shape`` 可不一致）。"""
    tile_a = tla.tile_view(mem, tla.make_shape(32, 32), tla.make_coord(1, 1))
    tile_b = tla.tile_view(mem, tla.make_shape(32, 32), tla.make_coord(0, 0))
    ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512)
    local = tla.make_tensor_like(ptr, tile_a, tla.arch.zN)
    with tla.cube():
        tla.copy(local, tile_a)
        tla.copy(local, tile_b)


def test_interface_copy_gm_to_l1_zn_two_copies_dsl_mlir() -> None:
    """Eager ``Tensor`` → ``dump_mlir``：``mem`` / ``mem_i8`` 的 **逻辑 shape** 与 **origin_shape** 均不同，
    且 **origin 不含 128**；``stride`` 按各自 layout ``shape`` 构造合法的行主布局，
    以确保相邻行不重叠，同时保留 ``shape`` 与 ``origin_shape`` 不同的覆盖。
    两路 f32 ``tile_view``、两次 ``tla.copy``、单次 4096B typed alloc + like；第二参数仅用于入口类型差异。"""
    # Layout shape 大于 origin；row-major stride 按 layout shape 的列数构造。
    mem = make_fake_tensor(
              tla.Float32,
              (200, 260),
              (260, 1),
              origin_shape=(72, 88),
              layout_tag=tla.arch.RowMajor,
          )
    mem_i8 = make_fake_tensor(
                 tla.Int8,
                 (100, 140),
                 (140, 1),
                 origin_shape=(56, 92),
                 layout_tag=tla.arch.RowMajor,
             )
    mlir = _kernel_copy_gm_row_major_to_l1_zn.dump_mlir(type_args=(mem, mem_i8))
    assert "tla.tile_view" in mlir
    assert "tla.copy" in mlir
    assert "tla.make_tensor_like" in mlir
    assert "tla.recast_ptr" not in mlir
    assert (
        "<!tla.layout<!tla.shape<(16,2),(8,4)>, !tla.stride<(8,128),(1,256)>, !tla.shape<32,32>, zN>, !tla.coord<0,0>, !tla.ptr<f32, l1, 512>>"
        in mlir
    )
    assert "tla.copy" in mlir
    assert "tla.copy" in mlir
