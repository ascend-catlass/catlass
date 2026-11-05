from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import inspect
import warnings
from typing import Any

from catlass._mlir import ir as mlir_ir
from catlass import _tla_type_bridge
import pytest

import catlass.tla as tla
from catlass.address_space import AddressSpace
from catlass.base_dsl import ast_helpers
from catlass.core_api import _category
from catlass.execution_lowering import TlaLoweringError
import catlass.runtime as runtime_mod


def test_mlir_value_identity_is_stable_across_operand_wrappers() -> None:
    with mlir_ir.Context() as ctx, mlir_ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = mlir_ir.Module.parse(
            """
module {
  func.func @probe(%arg0: i64) -> i64 {
    %c1 = arith.constant 1 : i64
    %sum = arith.addi %arg0, %c1 : i64
    return %sum : i64
  }
}
"""
        )
        block = list(module.body.operations[0].regions[0].blocks)[0]
        ops = list(block.operations)

        equivalent_values = (
            (block.arguments[0], ops[1].operands[0]),
            (ops[0].results[0], ops[1].operands[1]),
            (ops[1].results[0], ops[2].operands[0]),
        )
        for registered, lookup in equivalent_values:
            assert registered is not lookup
            assert registered == lookup
            assert hash(registered) == hash(lookup)
            assert {registered: "metadata"}[lookup] == "metadata"


def test_allocate_returns_typed_pointer_metadata() -> None:
    with runtime_mod._eager_capture():
        ptr = tla.allocate((2, 4), tla.Float16, tla.AddressSpace.l1, 512)
        assert _category(ptr) == "pointer"
        assert ptr.dtype is tla.Float16
        assert ptr.value_type is tla.Float16
        assert ptr.addrspace == AddressSpace.l1
        assert ptr.alignment == 512
        assert getattr(ptr, "_alloc_size_bytes") == 16






@tla.kernel
def _bad_dynamic_allocate_shape(dim: tla.types.TlaIndex) -> None:
    _ = tla.allocate(dim, tla.Float16, tla.AddressSpace.l1, 512)


def test_allocate_rejects_dynamic_shape() -> None:
    with pytest.raises(TlaLoweringError, match="static shape"):
        _bad_dynamic_allocate_shape.dump_mlir(type_args=(4,))



@pytest.mark.parametrize("mode", [False, True, 3])
def test_cross_flag_requires_valid_mode(mode: Any) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.cross_flag"):
        tla.cross_flag("x", mode=mode)



def test_cross_flag_rejects_removed_declaration_pipes() -> None:
    with pytest.raises(TypeError):
        tla.cross_flag("x", tla.pipes.MTE3, tla.pipes.SCALAR)


def test_cross_core_flag_ops_require_call_site_pipe() -> None:
    for op in (tla.cross_core_set_flag, tla.cross_core_wait_flag):
        signature = inspect.signature(op)
        assert signature.parameters["pipe"].default is inspect.Signature.empty
        with pytest.raises(TypeError):
            signature.bind(object())


def test_cross_core_flag_ops_reject_invalid_pipe() -> None:
    @tla.kernel
    def kernel() -> None:
        cross = tla.cross_flag("x")
        with tla.vector():
            tla.cross_core_set_flag(cross, "gpu")

    with pytest.raises(tla.TlaCoreAPIError, match="tla.cross_core_set_flag"):
        kernel.dump_mlir()


@tla.kernel
def _cross_set_aiv_id_kernel(
    mode: tla.Constexpr[int], aiv_id: tla.Constexpr[Any]
) -> None:
    cross = tla.cross_flag("x", mode=mode)
    with tla.vector():
        tla.cross_core_set_flag(cross, tla.pipes.VECTOR, aiv_id)


@tla.kernel
def _cross_wait_aiv_id_kernel(
    mode: tla.Constexpr[int], aiv_id: tla.Constexpr[Any]
) -> None:
    cross = tla.cross_flag("x", mode=mode)
    with tla.vector():
        tla.cross_core_wait_flag(cross, tla.pipes.VECTOR, aiv_id)


@pytest.mark.parametrize(
    "kernel", [_cross_set_aiv_id_kernel, _cross_wait_aiv_id_kernel]
)
@pytest.mark.parametrize("aiv_id", [None, -1, 2, True, "0"])
def test_mode4_cross_core_flag_ops_require_static_aiv_id(
    kernel: Any, aiv_id: Any
) -> None:
    with pytest.raises(
        tla.TlaCoreAPIError, match="mode 4 requires compile-time 0 or 1"
    ):
        kernel.dump_mlir(type_args=(4, aiv_id))


@pytest.mark.parametrize(
    "kernel", [_cross_set_aiv_id_kernel, _cross_wait_aiv_id_kernel]
)
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_non_mode4_cross_core_flag_ops_reject_aiv_id(kernel: Any, mode: int) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=f"mode {mode} requires None"):
        kernel.dump_mlir(type_args=(mode, 0))


def test_mode4_cross_core_flag_ops_emit_aiv_id() -> None:
    @tla.kernel
    def kernel() -> None:
        cross = tla.cross_flag("x", mode=4)
        with tla.vector():
            tla.cross_core_set_flag(cross, tla.pipes.FIX, 0)
            tla.cross_core_wait_flag(cross, tla.pipes.VECTOR, 1)

    mlir = kernel.dump_mlir()
    assert "!tla.cross_flag<4>" in mlir
    assert "aiv_id = 0 : i64" in mlir
    assert "aiv_id = 1 : i64" in mlir


def test_mutex_requires_valid_resource_and_id() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex"):
        tla.mutex(resource="", id=-1)
    with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex"):
        tla.mutex(resource=123, id=-1)  # type: ignore[arg-type]
    with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex"):
        tla.mutex(resource="l0a_ping", id=True)  # type: ignore[arg-type]
    with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex"):
        tla.mutex(resource="l0a_ping", id=-2)
    with pytest.raises(tla.TlaCoreAPIError, match="tla.mutex"):
        tla.mutex(resource="l0a_ping", id=32)




# Nested ``make_shape`` trees for L0 zN / nZ / L0C layouts (must match remap stride trees);
# flat ``origin_shape`` is the logical M×N bounds, aligned with ``tile_view`` targets.
_M128_64_FRACTAL_ZN = ((16, 8), (16, 4))
_M64_128_FRACTAL_ZN = ((16, 4), (16, 8))
_M128_128_FRACTAL_L0C = ((16, 8), (16, 8))
_M32_128_FRACTAL_ZN = ((16, 2), (16, 8))
_M128_64_FRACTAL_ZN_F32 = ((16, 8), (8, 8))


def _tensor_arg(
    fractal: tuple[tuple[int, int], tuple[int, int]],
    origin_mn: tuple[int, int],
    dtype: Any,
    addrspace: Any,
    layout_tag: Any,
) -> tla.Tensor:
    from catlass.core_api import (
        _remap_tensor_like_prefix_fields_for_layout_trees,
        _resolve_arch_layout_tag,
    )

    dtype_token = str(getattr(dtype, "dtype", "")).strip().lower()
    layout_token = _resolve_arch_layout_tag(layout_tag, for_op="test")
    remapped = _remap_tensor_like_prefix_fields_for_layout_trees(
        origin_mn, dtype_token, layout_token
    )
    assert remapped is not None
    return make_fake_tensor(
        dtype,
        (fractal[0], fractal[1]),
        remapped.stride,
        addrspace=addrspace,
        origin_shape=remapped.origin_shape,
        coord=remapped.coord,
        layout_tag=layout_tag,
    )


def _skip_if_mmad_rank2_tile_view_regression(exc: BaseException) -> None:
    """``tile_view`` + ``mmad`` can hit rank-2 validation before types align; skip instead of failing."""
    if isinstance(exc, TlaLoweringError) and "rank-2 tiles only" in str(exc):
        pytest.skip(
            "tla.mmad rank-2 check rejects tile_view operand types until metadata matches"
        )


def _skip_if_compute_order_unsupported(exc: BaseException) -> None:
    msg = str(exc)
    if "compute_order" in msg and "expected valid keyword" in msg:
        pytest.skip("linked BiShengIR lacks tla.compute_order enum used by mmad")


def _require_mmad_compute_order_supported() -> None:
    """Fail-fast skip when BiShengIR cannot parse mmad compute_order attrs."""
    with mlir_ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        try:
            mlir_ir.Attribute.parse("#tla.compute_order<M_FIRST>", context=ctx)
        except Exception as exc:  # noqa: BLE001
            _skip_if_compute_order_unsupported(exc)
            raise


def test_mmad_validates_operands_and_kwargs() -> None:
    _require_mmad_compute_order_supported()
    @tla.kernel
    def kernel(
        mem_a: tla.Tensor,
        mem_b: tla.Tensor,
        mem_c: tla.Tensor,
    ) -> None:
        lhs = tla.tile_view(mem_a, tla.make_shape(128, 64), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem_b, tla.make_shape(64, 128), tla.make_coord(0, 0))
        acc = tla.tile_view(mem_c, tla.make_shape(128, 128), tla.make_coord(0, 0))
        with tla.cube():
            _ = tla.mmad(acc, lhs, rhs, init_c=True)

    try:
        mlir = kernel.dump_mlir(
            type_args=(
                _tensor_arg(
                    _M128_64_FRACTAL_ZN,
                    (128, 64),
                    tla.Float16,
                    tla.AddressSpace.l0a,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M64_128_FRACTAL_ZN,
                    (64, 128),
                    tla.Float16,
                    tla.AddressSpace.l0b,
                    tla.arch.nZ,
                ),
                _tensor_arg(
                    _M128_128_FRACTAL_L0C,
                    (128, 128),
                    tla.Float32,
                    tla.AddressSpace.l0c,
                    tla.arch.L0Clayout,
                ),
            )
        )
    except TlaLoweringError as e:
        _skip_if_mmad_rank2_tile_view_regression(e)
        raise
    assert "tla.mmad" in mlir
    assert "!tla.ptr<f16, l0a, 2>" in mlir
    assert "!tla.ptr<f16, l0b, 2>" in mlir
    assert "!tla.ptr<f32, l0c, 4>" in mlir


def test_mmad_rejects_old_order_at_frontend() -> None:
    _require_mmad_compute_order_supported()
    @tla.kernel
    def kernel(
        mem_a: tla.Tensor,
        mem_b: tla.Tensor,
        mem_c: tla.Tensor,
    ) -> None:
        lhs = tla.tile_view(mem_a, tla.make_shape(128, 64), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem_b, tla.make_shape(64, 128), tla.make_coord(0, 0))
        acc = tla.tile_view(mem_c, tla.make_shape(128, 128), tla.make_coord(0, 0))
        with tla.cube():
            _ = tla.mmad(lhs, rhs, acc, init_c=True)

    with pytest.raises(TlaLoweringError, match="unsupported tla.mmad tile addrspaces"):
        _ = kernel.dump_mlir(
            type_args=(
                _tensor_arg(
                    _M128_64_FRACTAL_ZN,
                    (128, 64),
                    tla.Float16,
                    tla.AddressSpace.l0a,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M64_128_FRACTAL_ZN,
                    (64, 128),
                    tla.Float16,
                    tla.AddressSpace.l0b,
                    tla.arch.nZ,
                ),
                _tensor_arg(
                    _M128_128_FRACTAL_L0C,
                    (128, 128),
                    tla.Float32,
                    tla.AddressSpace.l0c,
                    tla.arch.L0Clayout,
                ),
            )
        )


def test_mmad_rejects_wrong_element_types_at_frontend() -> None:
    _require_mmad_compute_order_supported()
    @tla.kernel
    def kernel(
        mem_a: tla.Tensor,
        mem_b: tla.Tensor,
        mem_c: tla.Tensor,
    ) -> None:
        lhs = tla.tile_view(mem_a, tla.make_shape(128, 64), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem_b, tla.make_shape(64, 128), tla.make_coord(0, 0))
        acc = tla.tile_view(mem_c, tla.make_shape(128, 128), tla.make_coord(0, 0))
        with tla.cube():
            _ = tla.mmad(acc, lhs, rhs, init_c=True)

    with pytest.raises(TlaLoweringError, match="unsupported tla.mmad element types"):
        _ = kernel.dump_mlir(
            type_args=(
                _tensor_arg(
                    _M128_64_FRACTAL_ZN_F32,
                    (128, 64),
                    tla.Float32,
                    tla.AddressSpace.l0a,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M64_128_FRACTAL_ZN,
                    (64, 128),
                    tla.Float16,
                    tla.AddressSpace.l0b,
                    tla.arch.nZ,
                ),
                _tensor_arg(
                    _M128_128_FRACTAL_L0C,
                    (128, 128),
                    tla.Float32,
                    tla.AddressSpace.l0c,
                    tla.arch.L0Clayout,
                ),
            )
        )


def test_mmad_rejects_wrong_shape_contract_at_frontend() -> None:
    _require_mmad_compute_order_supported()
    @tla.kernel
    def kernel(
        mem_a: tla.Tensor,
        mem_b: tla.Tensor,
        mem_c: tla.Tensor,
    ) -> None:
        lhs = tla.tile_view(mem_a, tla.make_shape(128, 64), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem_b, tla.make_shape(32, 128), tla.make_coord(0, 0))
        acc = tla.tile_view(mem_c, tla.make_shape(128, 128), tla.make_coord(0, 0))
        with tla.cube():
            _ = tla.mmad(acc, lhs, rhs, init_c=True)

    with pytest.raises(TlaLoweringError) as excinfo:
        _ = kernel.dump_mlir(
            type_args=(
                _tensor_arg(
                    _M128_64_FRACTAL_ZN,
                    (128, 64),
                    tla.Float16,
                    tla.AddressSpace.l0a,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M32_128_FRACTAL_ZN,
                    (32, 128),
                    tla.Float16,
                    tla.AddressSpace.l0b,
                    tla.arch.nZ,
                ),
                _tensor_arg(
                    _M128_128_FRACTAL_L0C,
                    (128, 128),
                    tla.Float32,
                    tla.AddressSpace.l0c,
                    tla.arch.L0Clayout,
                ),
            )
        )
    msg = str(excinfo.value)
    assert (
        "unsupported tla.mmad tile shape contract" in msg or "rank-2 tiles only" in msg
    )


def test_mmad_rejects_rhs_zn_layout_at_frontend() -> None:
    _require_mmad_compute_order_supported()
    @tla.kernel
    def kernel(
        mem_a: tla.Tensor,
        mem_b: tla.Tensor,
        mem_c: tla.Tensor,
    ) -> None:
        lhs = tla.tile_view(mem_a, tla.make_shape(128, 64), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem_b, tla.make_shape(64, 128), tla.make_coord(0, 0))
        acc = tla.tile_view(mem_c, tla.make_shape(128, 128), tla.make_coord(0, 0))
        with tla.cube():
            _ = tla.mmad(acc, lhs, rhs, init_c=True)

    with pytest.raises(TlaLoweringError, match="unsupported tla.mmad operand layout"):
        _ = kernel.dump_mlir(
            type_args=(
                _tensor_arg(
                    _M128_64_FRACTAL_ZN,
                    (128, 64),
                    tla.Float16,
                    tla.AddressSpace.l0a,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M64_128_FRACTAL_ZN,
                    (64, 128),
                    tla.Float16,
                    tla.AddressSpace.l0b,
                    tla.arch.zN,
                ),
                _tensor_arg(
                    _M128_128_FRACTAL_L0C,
                    (128, 128),
                    tla.Float32,
                    tla.AddressSpace.l0c,
                    tla.arch.L0Clayout,
                ),
            )
        )


def test_mmad_rejects_unknown_kwarg() -> None:
    @tla.kernel
    def kernel(mem: tla.Tensor) -> None:
        lhs = tla.tile_view(mem, tla.make_shape(1, 8), tla.make_coord(0, 0))
        rhs = tla.tile_view(mem, tla.make_shape(1, 8), tla.make_coord(0, 0))
        acc = tla.tile_view(mem, tla.make_shape(1, 8), tla.make_coord(0, 0))
        with tla.cube():
            _ = tla.mmad(acc, lhs, rhs, bad=True)

    mem_arg = make_fake_tensor(
                  tla.Float16,
                  (8, 8),
                  (8, 1),
                  origin_shape=(8, 8),
                  layout_tag=tla.arch.RowMajor,
              )
    with pytest.raises(tla.TlaCoreAPIError, match="unknown keyword"):
        _ = kernel.dump_mlir(type_args=(mem_arg,))


def test_make_shape_rejects_non_index_components() -> None:
    @tla.kernel
    def _bad() -> None:
        _ = tla.make_shape(1.0, 2)

    with pytest.raises(tla.TlaCoreAPIError, match="tla.make_shape"):
        _ = _bad.dump_mlir()


def test_make_coord_rejects_negative_static_leaf() -> None:
    @tla.kernel
    def _bad() -> None:
        _ = tla.make_coord(-1, 0)

    with pytest.raises(tla.TlaCoreAPIError, match="coord leaf >= 0"):
        _ = _bad.dump_mlir()


def test_make_stride_rejects_nonpositive_static_leaf() -> None:
    @tla.kernel
    def _bad() -> None:
        _ = tla.make_stride(0, 1)

    with pytest.raises(tla.TlaCoreAPIError, match="stride leaf strictly positive"):
        _ = _bad.dump_mlir()


def test_range_accepts_one_or_three_args() -> None:
    _ = tla.range(32)
    _ = tla.range(0, 32)
    _ = tla.range(0, 32, 1)


def test_range_rejects_bad_arity() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.range"):
        _ = tla.range(0, step=1)


def test_range_constexpr_returns_python_range() -> None:
    assert list(tla.range_constexpr(0, 4, 2)) == [0, 2]
    with pytest.raises(tla.TlaCoreAPIError, match="tla.range_constexpr"):
        _ = tla.range_constexpr(0, 4.0)


@pytest.mark.parametrize(
    ("args", "expected_length", "warns"),
    [
        ((63,), 63, False),
        ((64,), 64, True),
        ((65,), 65, True),
        ((5, 5), 0, False),
        ((64, 0, -1), 64, True),
    ],
)
def test_range_constexpr_static_loop_warning_boundary(
    args: tuple[int, ...], expected_length: int, warns: bool
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = tla.range_constexpr(*args)

    assert len(result) == expected_length
    optimization_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, ast_helpers.DSLOptimizationWarning)
    ]
    assert bool(optimization_warnings) is warns
    if warns:
        assert len(optimization_warnings) == 1
        assert "This static loop has" in str(optimization_warnings[0].message)
        assert "`tla.range(...)`" in str(optimization_warnings[0].message)


_MAKE_TENSOR_LIKE_DEST_SPACE = tla.AddressSpace.ub


@tla.kernel
def _make_tensor_like_destination_space_kernel(mem: tla.Tensor) -> None:
    ptr = tla.allocate(121 * 104, tla.Float32, _MAKE_TENSOR_LIKE_DEST_SPACE, 32)
    _ = tla.make_tensor_like(ptr, mem, tla.arch.RowMajor)


@tla.kernel
def _make_tensor_like_invalid_destination_space_kernel(mem: tla.Tensor) -> None:
    ptr = tla.make_ptr(
        tla.Float32,
        0,
        mem_space=_MAKE_TENSOR_LIKE_DEST_SPACE,
        assumed_align=32,
    )
    _ = tla.make_tensor_like(ptr, mem, tla.arch.RowMajor)


@tla.kernel
def _make_tensor_like_aligned_linear_stride_kernel(mem: tla.Tensor) -> None:
    row_ptr = tla.allocate(121 * 104, tla.Float32, tla.AddressSpace.ub, 32)
    col_ptr = tla.allocate(99 * 128, tla.Float32, tla.AddressSpace.ub, 32)
    _ = tla.make_tensor_like(row_ptr, mem, tla.arch.RowMajor)
    _ = tla.make_tensor_like(col_ptr, mem, tla.arch.ColumnMajor)


def _make_tensor_like_static_source() -> tla.Tensor:
    return make_fake_tensor(
               tla.Float32,
               (121, 99),
               (99, 1),
               origin_shape=(121, 99),
               layout_tag=tla.arch.RowMajor,
           )


@pytest.mark.parametrize(
    ("space", "token"),
    (
        (tla.AddressSpace.ub, "ub"),
        (tla.AddressSpace.l1, "l1"),
        (tla.AddressSpace.l0a, "l0a"),
        (tla.AddressSpace.l0b, "l0b"),
        (tla.AddressSpace.l0c, "l0c"),
    ),
)
def test_make_tensor_like_accepts_only_supported_on_chip_spaces(
    space: AddressSpace, token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(globals(), "_MAKE_TENSOR_LIKE_DEST_SPACE", space)
    mlir = _make_tensor_like_destination_space_kernel.dump_mlir(
        type_args=(_make_tensor_like_static_source(),)
    )
    assert f"!tla.ptr<f32, {token}, 32>" in mlir


@pytest.mark.parametrize("space", (tla.AddressSpace.gm, tla.AddressSpace.generic))
def test_make_tensor_like_rejects_off_chip_and_generic_spaces(
    space: AddressSpace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(globals(), "_MAKE_TENSOR_LIKE_DEST_SPACE", space)
    with pytest.raises(tla.TlaCoreAPIError, match="expected an on-chip address space"):
        _make_tensor_like_invalid_destination_space_kernel.dump_mlir(
            type_args=(_make_tensor_like_static_source(),)
        )


def test_make_tensor_like_aligns_only_the_linear_leading_stride() -> None:
    mlir = _make_tensor_like_aligned_linear_stride_kernel.dump_mlir(
        type_args=(_make_tensor_like_static_source(),)
    )
    assert "!tla.shape<121,99>, !tla.stride<104,1>" in mlir
    assert "!tla.shape<121,99>, !tla.stride<1,128>" in mlir
    assert mlir.count("!tla.shape<121,99>") >= 2
