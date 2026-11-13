from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401
from catlass.execution_lowering import TlaLoweringError


# --- Frontend IR emission -----------------------------------------------------


@tla.kernel
def make_tensor_makeptr_kernel() -> None:
    ptr = tla.make_ptr(tla.Float32, 4096, mem_space=tla.AddressSpace.l1)
    local = tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(16, 16), tla.make_stride(16, 1))
    )
    _ = local


@tla.kernel
def make_tensor_default_coord_kernel(mem_in: tla.Tensor) -> None:
    tile = tla.tile_view(mem_in, tla.make_shape(16, 16), tla.make_coord(0, 0))
    ptr = tla.allocate((16, 16), tla.Float32, tla.AddressSpace.ub, 256)
    # coord omitted: must default to a zero coord matching the rank-2 layout.
    local = tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(16, 16), tla.make_stride(16, 1))
    )
    with tla.vector():
        tla.copy(tile, local)


@tla.kernel
def make_tensor_all_layouts_kernel() -> None:
    ptr = tla.make_ptr(tla.Float16, 16384, mem_space=tla.AddressSpace.l1)

    row = tla.make_layout(
        tla.make_shape(32, 32),
        tla.make_stride(32, 1),
        layoutTag=tla.arch.RowMajor,
    )
    column = tla.make_layout(
        tla.make_shape(32, 32),
        tla.make_stride(1, 32),
        layoutTag=tla.arch.ColumnMajor,
    )
    zn = tla.make_layout(
        tla.make_shape((16, 2), (16, 2)),
        tla.make_stride((16, 256), (1, 512)),
        layoutTag=tla.arch.zN,
    )
    nz = tla.make_layout(
        tla.make_shape((16, 2), (16, 2)),
        tla.make_stride((1, 512), (16, 256)),
        layoutTag=tla.arch.nZ,
    )
    l0c = tla.make_layout(
        tla.make_shape((16, 2), (16, 2)),
        tla.make_stride((16, 256), (1, 512)),
        layoutTag=tla.arch.L0Clayout,
    )
    zn_unalign = tla.make_layout(
        tla.make_shape((32, 1), (16, 2)),
        tla.make_stride((16, 512), (1, 512)),
        layoutTag=tla.arch.zNUnAlign,
    )

    _ = tla.make_tensor(ptr, row)
    _ = tla.make_tensor(ptr, column)
    _ = tla.make_tensor(ptr, zn)
    _ = tla.make_tensor(ptr, nz)
    _ = tla.make_tensor(ptr, l0c)
    _ = tla.make_tensor(ptr, zn_unalign)


def test_make_tensor_emits_op_with_explicit_coord() -> None:
    mlir = make_tensor_makeptr_kernel.dump_mlir()
    assert "tla.make_tensor" in mlir
    assert "tla.make_tensor_like" not in mlir
    # Result tensor type carries the explicit layout/coord/ptr.
    assert "!tla.layout<!tla.shape<16,16>, !tla.stride<16,1>" in mlir
    assert "!tla.coord<0,0>" in mlir
    assert "!tla.ptr<f32, l1" in mlir


def test_make_tensor_coord_defaults_to_zero_matching_rank() -> None:
    mem = make_fake_tensor(
        tla.Float32,
        (16, 16),
        (16, 1),
        origin_shape=(16, 16),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = make_tensor_default_coord_kernel.dump_mlir(type_args=(mem,))
    assert "tla.make_tensor" in mlir
    # rank-2 layout -> default coord is (0, 0).
    assert "!tla.coord<0,0>" in mlir


def test_make_tensor_rank1_default_coord_is_single_zero() -> None:
    @tla.kernel
    def _rank1_kernel() -> None:
        ptr = tla.make_ptr(tla.Float32, 256, mem_space=tla.AddressSpace.ub)
        local = tla.make_tensor(
            ptr, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
        )
        _ = local

    mlir = _rank1_kernel.dump_mlir()
    assert "tla.make_tensor" in mlir
    # rank-1 layout keeps a rank-1 coord in the frontend IR (rank-2 promotion happens
    # in the C++ lowering); the default coord leaf is 0.
    assert "!tla.coord<0>" in mlir
    assert "!tla.coord<0,0>" not in mlir


def test_make_tensor_accepts_every_registered_layout() -> None:
    mlir = make_tensor_all_layouts_kernel.dump_mlir()
    assert mlir.count("tla.make_tensor ") == 6
    assert mlir.count("!tla.coord<0,0>") >= 6
    assert (
        "!tla.layout<!tla.shape<(16,2),(16,2)>, "
        "!tla.stride<(16,256),(1,512)>, !tla.shape<32,32>, zN>"
    ) in mlir
    assert (
        "!tla.layout<!tla.shape<(16,2),(16,2)>, "
        "!tla.stride<(1,512),(16,256)>, !tla.shape<32,32>, nZ>"
    ) in mlir
    assert "L0Clayout" in mlir
    assert "zNUnAlign" in mlir


def test_make_tensor_accepts_dynamic_shape_leaf() -> None:
    """A dynamic shape/stride leaf (``Int32`` from a tla.range induction variable)
    must be accepted, not rejected as index-type metadata.

    Regression: ``make_tensor`` previously built its ``TlaIndexTreeType`` from the raw
    index tree (which carries Numeric for dynamic leaves) and failed with
    ``TypeError: ... expects static int leaves or None for dynamic leaves; got ...``.
    The dynamic SSA value travels in the make_shape/make_stride operand; the tensor type
    only spells the leaf as ``?`` (``None``), exactly like ``tile_view``.
    """

    @tla.kernel
    def _dynamic_kernel() -> None:
        ptr = tla.make_ptr(tla.Float32, 4096, mem_space=tla.AddressSpace.ub)
        for i in tla.range(0, 8, 1):
            local = tla.make_tensor(
                ptr,
                tla.make_layout(tla.make_shape(16, i), tla.make_stride(i, 1)),
            )
            _ = local

    mlir = _dynamic_kernel.dump_mlir()
    assert "tla.make_tensor" in mlir
    # Dynamic leaf is spelled `?` in the shape/stride type, not a static int.
    assert "!tla.shape<16,?>" in mlir
    assert "!tla.stride<?,1>" in mlir
    # The make_shape op carries the dynamic SSA value as an operand (a static leaf would
    # be a nullary `tla.make_shape -> ...`); the dynamic value travels here, not in the type.
    assert "tla.make_shape %" in mlir
    assert "tla.make_stride %" in mlir


def test_make_tensor_accepts_dynamic_packed_shape_leaf() -> None:
    @tla.kernel
    def _dynamic_packed_kernel() -> None:
        ptr = tla.make_ptr(tla.Float32, 4096, mem_space=tla.AddressSpace.l1)
        for row_blocks in tla.range(1, 3, 1):
            layout = tla.make_layout(
                tla.make_shape((16, row_blocks), (8, 4)),
                tla.make_stride((8, 128), (1, row_blocks * 128)),
                layoutTag=tla.arch.zN,
            )
            local = tla.make_tensor(ptr, layout)
            _ = local

    mlir = _dynamic_packed_kernel.dump_mlir()
    assert "!tla.shape<(16,?),(8,4)>" in mlir
    assert "!tla.stride<(8,128),(1,?)>" in mlir
    # Omitted packed origin_shape is inferred as (16*row_blocks, 8*4).
    assert "!tla.shape<?,32>, zN>" in mlir
    assert "!tla.coord<0,0>" in mlir


def test_make_tensor_validates_static_trait_leaves_in_dynamic_layout() -> None:
    @tla.kernel
    def _dynamic_packed_kernel() -> None:
        ptr = tla.make_ptr(tla.Float32, 4096, mem_space=tla.AddressSpace.l1)
        for inner_rows in tla.range(1, 3, 1):
            layout = tla.make_layout(
                # shape[0][0] is a dynamic zN trait leaf, so only that check
                # must be deferred.
                tla.make_shape((inner_rows, 2), (8, 4)),
                # The independent static trait leaf stride[1][0] must still
                # be rejected because zN requires it to be 1.
                tla.make_stride((8, 128), (2, 256)),
                layoutTag=tla.arch.zN,
            )
            _ = tla.make_tensor(ptr, layout)

    with pytest.raises(TlaLoweringError, match=r"do not match layout 'zN'"):
        _dynamic_packed_kernel.dump_mlir()


def test_make_tensor_rank1_dynamic_extent_emits_tlair() -> None:
    """Rank-1 make_tensor(shape=m, stride=1) must emit dynamic extent TLAIR.

    The TensorToMemref fix for derived leading-stride SSA is covered by the lit
    test ``make-tensor-rank1-dynamic-extent.mlir`` (full TlaCompile pipeline).
    """

    @tla.kernel
    def _rank1_dyn_kernel() -> None:
        ptr = tla.make_ptr(tla.Float32, 4096, mem_space=tla.AddressSpace.ub)
        for m in tla.range(1, 9, 1):
            local = tla.make_tensor(
                ptr,
                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
            )
            _ = local

    mlir = _rank1_dyn_kernel.dump_mlir()
    assert "tla.make_tensor" in mlir
    assert "!tla.shape<?>" in mlir
    assert "!tla.stride<1>" in mlir
    assert "tla.make_shape %" in mlir


# --- Preconditions ------------------------------------------------------------
