from __future__ import annotations

import pytest

import catlass.runtime as runtime_mod
from catlass import tla
from catlass._mlir import ir as mlir_ir
from catlass.core_api import MaskSSA, VectorSSA
from catlass.tla.runtime import make_fake_tensor


def _vector_tensor() -> tla.Tensor:
    return make_fake_tensor(
        tla.Float32,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )


def _vector_tensor_for(dtype: type[tla.Numeric], shape: int) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (shape,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(shape,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def _mask_where_width(dtype: tla.Constexpr[type]) -> None:
    with tla.vector():  # noqa: SIM117 - DSL regions must remain syntactically nested.
        with tla.vec.func(mode="simd"):
            selector = tla.create_mask(pattern=tla.mask.H, dtype=dtype)
            x = tla.create_mask(pattern=tla.mask.Q, dtype=dtype)
            y = tla.create_mask(pattern=tla.mask.M4, dtype=dtype)
            selected = tla.where(selector, x, y)
            assert isinstance(selected, MaskSSA)
            _ = tla.bitwise_xor(selected, x)


@pytest.mark.parametrize(
    ("dtype", "lanes"),
    (
        (tla.Int64, 32),
        (tla.Float32, 64),
        (tla.Float16, 128),
        (tla.Int8, 256),
    ),
    ids=("mask32", "mask64", "mask128", "mask256"),
)
def test_mask_where_preserves_mask_type(dtype: type[tla.Numeric], lanes: int) -> None:
    mlir = _mask_where_width.dump_mlir(type_args=(dtype,))

    (where_line,) = [line for line in mlir.splitlines() if "tla.where" in line]
    assert where_line.count(f"!tla.mask<{lanes}>") == 4
    assert "tla.bitwise_xor" in mlir


@tla.kernel
def _mismatched_mask_where(
    selector_dtype: tla.Constexpr[type],
    x_dtype: tla.Constexpr[type],
    y_dtype: tla.Constexpr[type],
) -> None:
    with tla.vector():  # noqa: SIM117 - DSL regions must remain syntactically nested.
        with tla.vec.func(mode="simd"):
            selector = tla.create_mask(pattern=tla.mask.H, dtype=selector_dtype)
            x = tla.create_mask(pattern=tla.mask.Q, dtype=x_dtype)
            y = tla.create_mask(pattern=tla.mask.M4, dtype=y_dtype)
            _ = tla.where(selector, x, y)


@pytest.mark.parametrize(
    ("selector_dtype", "x_dtype", "y_dtype", "message"),
    (
        (
            tla.Float32,
            tla.Float32,
            tla.Float16,
            r"y has type !tla\.mask<128>.*!tla\.mask<64>",
        ),
        (
            tla.Float16,
            tla.Float32,
            tla.Float32,
            r"mask has type !tla\.mask<128>.*!tla\.mask<64>",
        ),
    ),
    ids=("arm-type", "selector-type"),
)
def test_mask_where_rejects_mismatched_types(
    selector_dtype: type[tla.Numeric],
    x_dtype: type[tla.Numeric],
    y_dtype: type[tla.Numeric],
    message: str,
) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=message):
        _mismatched_mask_where.dump_mlir(type_args=(selector_dtype, x_dtype, y_dtype))


@tla.kernel
def _mixed_where_arms(vector_first: tla.Constexpr[bool]) -> None:
    with tla.vector():  # noqa: SIM117 - DSL regions must remain syntactically nested.
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            vector = tla.full(1.0, tla.Float32)
            if tla.const_expr(vector_first):
                _ = tla.where(mask, vector, mask)
            else:
                _ = tla.where(mask, mask, vector)


@pytest.mark.parametrize(
    "vector_first", (False, True), ids=("mask-vector", "vector-mask")
)
def test_mask_where_rejects_mixed_arm_categories(vector_first: bool) -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="x and y must both be VectorSSA values or both be MaskSSA values",
    ):
        _mixed_where_arms.dump_mlir(type_args=(vector_first,))


@pytest.mark.parametrize(
    ("mask_type", "x_type", "y_type", "result_type"),
    (
        ("!tla.mask<128>", "!tla.mask<64>", "!tla.mask<64>", "!tla.mask<64>"),
        ("!tla.mask<64>", "!tla.mask<128>", "!tla.mask<64>", "!tla.mask<64>"),
        ("!tla.mask<64>", "!tla.mask<64>", "!tla.mask<128>", "!tla.mask<64>"),
        ("!tla.mask<64>", "!tla.mask<64>", "!tla.mask<64>", "!tla.mask<128>"),
    ),
    ids=("selector", "x", "y", "result"),
)
def test_mask_where_verifier_rejects_mismatched_mask_types(
    mask_type: str,
    x_type: str,
    y_type: str,
    result_type: str,
) -> None:
    with (
        runtime_mod._eager_capture(),
        pytest.raises(
            mlir_ir.MLIRError,
            match="requires mask, x, y, and result to have the same MaskSSA type",
        ),
    ):
        mlir_ir.Module.parse(
            f"""
                module {{
                  func.func @invalid(
                      %mask: {mask_type}, %x: {x_type}, %y: {y_type}) {{
                    "tla.vec.func"() ({{
                      %result = "tla.where"(%mask, %x, %y)
                          : ({mask_type}, {x_type}, {y_type}) -> {result_type}
                    }}) {{mode = "simd"}} : () -> ()
                    return
                  }}
                }}
                """
        )


def test_mask_where_verifier_rejects_mixed_categories() -> None:
    with (
        runtime_mod._eager_capture(),
        pytest.raises(
            mlir_ir.MLIRError,
            match="requires x, y, and result to all be VectorSSA or all be MaskSSA",
        ),
    ):
        mlir_ir.Module.parse(
            """
                module {
                  func.func @invalid(
                      %mask: !tla.mask<64>, %x: !tla.mask<64>,
                      %y: !tla.vector<64xf32>) {
                    "tla.vec.func"() ({
                      %result = "tla.where"(%mask, %x, %y)
                          : (!tla.mask<64>, !tla.mask<64>,
                             !tla.vector<64xf32>) -> !tla.mask<64>
                    }) {mode = "simd"} : () -> ()
                    return
                  }
                }
                """
        )


def test_mask_where_requires_vec_func() -> None:
    with runtime_mod._eager_capture():
        module = mlir_ir.Module.parse(
            "module { func.func @mask_arg(%arg: !tla.mask<64>) { return } }"
        )
        argument = module.body.operations[0].regions[0].blocks[0].arguments[0]
        mask = MaskSSA(argument)
        with pytest.raises(tla.TlaCoreAPIError, match="vec.func"):
            tla.where(mask, mask, mask)


@tla.kernel
def _mask_where_consumers(src: tla.Tensor, dst: tla.Tensor, limit: int) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():  # noqa: SIM117 - DSL regions must remain syntactically nested.
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            zero = tla.sub(value, value)
            selector = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            x = tla.create_mask(pattern=tla.mask.Q, dtype=tla.Float32)
            y = tla.create_mask(pattern=tla.mask.M4, dtype=tla.Float32)
            selected = tla.where(selector, x, y)
            selected = tla.bitwise_xor(selected, x)
            selected = tla.where(selected, y, x)
            for _ in tla.range(limit):
                selected = tla.where(selector, selected, x)
            if limit > 0:
                selected = tla.where(selector, selected, y)
            dst_tile.store(tla.where(selected, value, zero), mask=selected)


def test_mask_where_result_supports_mask_consumers_and_scf_carriers() -> None:
    tensor = _vector_tensor()
    mlir = _mask_where_consumers.dump_mlir(type_args=(tensor, tensor, 2))

    assert mlir.count("tla.where") == 5
    assert "tla.bitwise_xor" in mlir
    assert "scf.for" in mlir
    assert "scf.if" in mlir
    assert "!tla.mask<64>" in mlir


@tla.kernel
def _vector_where_regression(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():  # noqa: SIM117 - DSL regions must remain syntactically nested.
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            zero = tla.sub(value, value)
            mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            selected = tla.where(mask, value, zero)
            assert isinstance(selected, VectorSSA)
            dst_tile.store(selected, mask=mask)


def test_vector_where_still_returns_vector_ssa() -> None:
    tensor = _vector_tensor()
    mlir = _vector_where_regression.dump_mlir(type_args=(tensor, tensor))

    (where_line,) = [line for line in mlir.splitlines() if "tla.where" in line]
    assert "-> !tla.vector<64xf32>" in where_line


@tla.kernel
def _mask_where_through_vector_select(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():  # noqa: SIM117 - DSL regions must remain syntactically nested.
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            zero = tla.sub(value, value)
            selector = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            x = tla.create_mask(pattern=tla.mask.Q, dtype=tla.Float32)
            y = tla.create_mask(pattern=tla.mask.M4, dtype=tla.Float32)
            selected = tla.where(selector, x, y)
            dst_tile.store(tla.where(selected, value, zero))


def test_mask_where_feeds_vector_select_in_mlir() -> None:
    """MaskSSA where result must lower and feed a vector where (no card/compile)."""
    tensor = _vector_tensor()
    mlir = _mask_where_through_vector_select.dump_mlir(type_args=(tensor, tensor))
    assert mlir.count("tla.where") == 2
    assert "!tla.mask<64>" in mlir
    assert "!tla.vector<64xf32>" in mlir


@tla.kernel
def _vector_where_dtype(
    x: tla.Tensor,
    y: tla.Tensor,
    dst: tla.Tensor,
    dtype: tla.Constexpr[type],
    lanes: tla.Constexpr[int],
) -> None:
    x_tile = tla.tile_view(x, tla.make_shape(lanes), tla.make_coord(0))
    y_tile = tla.tile_view(y, tla.make_shape(lanes), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():  # noqa: SIM117 - DSL regions must remain syntactically nested.
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.H, dtype=dtype)
            selected = tla.where(mask, x_tile.load(), y_tile.load())
            dst_tile.store(selected)


@pytest.mark.parametrize(
    ("dtype", "lanes"),
    (
        (tla.Bool, 256),
        (tla.Int8, 256),
        (tla.UInt8, 256),
        (tla.Int16, 128),
        (tla.UInt16, 128),
        (tla.BFloat16, 128),
        (tla.Float16, 128),
        (tla.Int32, 64),
        (tla.UInt32, 64),
        (tla.Float32, 64),
        (tla.Int64, 32),
        (tla.UInt64, 32),
        (tla.Float8E4M3FN, 256),
        (tla.Float8E5M2, 256),
        (tla.HiFloat8, 256),
        (tla.Float4E2M1x2, 256),
        (tla.Float4E1M2x2, 256),
    ),
    ids=lambda value: getattr(value, "dtype", str(value)),
)
def test_vector_where_accepts_supported_element_types(
    dtype: type[tla.Numeric], lanes: int
) -> None:
    tensor = _vector_tensor_for(dtype, lanes)
    mlir = _vector_where_dtype.dump_mlir(
        type_args=(tensor, tensor, tensor, dtype, lanes)
    )

    (where_line,) = [line for line in mlir.splitlines() if "tla.where" in line]
    assert f"!tla.vector<{lanes}x" in where_line


@tla.kernel
def _vector_where_f64(x: tla.Tensor, y: tla.Tensor) -> None:
    x_tile = tla.tile_view(x, tla.make_shape(32), tla.make_coord(0))
    y_tile = tla.tile_view(y, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():  # noqa: SIM117 - DSL regions must remain syntactically nested.
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float64)
            _ = tla.where(mask, x_tile.load(), y_tile.load())


def test_vector_where_rejects_unsupported_element_type() -> None:
    tensor = _vector_tensor_for(tla.Float64, 32)
    with pytest.raises(tla.TlaCoreAPIError, match="unsupported select element type f64"):
        _vector_where_f64.dump_mlir(type_args=(tensor, tensor))
