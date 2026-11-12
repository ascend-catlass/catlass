from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import inspect

import pytest

import catlass.tla as tla


def _host_tensor(
    shape: tuple[int, ...] = (4, 4),
    *,
    dtype: type[tla.Numeric] = tla.Float32,
    layout: object = tla.arch.RowMajor,
    stride: tuple[int, ...] | None = None,
) -> tla.Tensor:
    if stride is None:
        if layout == tla.arch.ColumnMajor:
            stride = (1, shape[0]) if len(shape) >= 1 else (1,)
        elif len(shape) == 1:
            stride = (1,)
        else:
            # Compact row-major: last dim unit stride.
            leading = 1
            compact: list[int] = []
            for dim in reversed(shape):
                compact.append(leading)
                leading *= int(dim)
            stride = tuple(reversed(compact))
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        origin_shape=shape,
        layout_tag=layout,
    )


def _host_packed_tensor(layout: object) -> tla.Tensor:
    # Explicit f32 32x32 fractal packed trees (shape tree must match stride tree).
    packed = {
        "zN": (((16, 2), (8, 4)), ((8, 128), (1, 256))),
        "nZ": (((8, 4), (16, 2)), ((1, 256), (8, 128))),
        "L0Clayout": (((16, 2), (16, 2)), ((16, 256), (1, 512))),
        "zNUnAlign": (((32, 1), (8, 4)), ((8, 256), (1, 256))),
    }[str(layout)]
    shape, stride = packed
    return make_fake_tensor(
        tla.Float32,
        shape,
        stride,
        origin_shape=(32, 32),
        coord=(0, 0),
        layout_tag=layout,
    )


@tla.kernel
def _aiv_print_tensor(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value)


@tla.kernel
def _aiv_print_tensor_prefix(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 4)


@tla.kernel
def _aiv_print_tensor_runtime_length(value: tla.Tensor, length: tla.Int32) -> None:
    with tla.vector():
        tla.print(value, length)


@tla.kernel
def _aiv_print_tensor_in_dynamic_if(value: tla.Tensor, limit: int) -> None:
    with tla.vector():
        if limit > 0:
            tla.print(value, 4)


@tla.kernel
def _aiv_print_tensor_in_dynamic_for(value: tla.Tensor, limit: int) -> None:
    with tla.vector():
        for _ in tla.range(0, limit, 1):
            tla.print(value, 4)


@tla.kernel
def _aiv_print_tensor_in_dynamic_while(value: tla.Tensor, limit: int) -> None:
    with tla.vector():
        index = 0
        while index < limit:
            tla.print(value, 4)
            index = index + 1


@tla.kernel
def _aic_print_tensor(value: tla.Tensor) -> None:
    with tla.cube():
        tla.print(value)


def _l0c_print_kernel(
    dtype: type[tla.Numeric],
    length: int = 256,
    *,
    alignment: int = 1024,
    make_ptr: bool = False,
):
    @tla.kernel
    def kernel(value: tla.Tensor) -> None:
        # tla.allocate requires a whole number of blocks, so an under-aligned
        # buffer can only be spelled with make_ptr.
        ptr = (
            tla.make_ptr(dtype, 0x1000, tla.AddressSpace.l0c, assumed_align=alignment)
            if make_ptr
            else tla.allocate(256, dtype, tla.AddressSpace.l0c, alignment)
        )
        local = tla.make_tensor_like(ptr, value, tla.arch.L0Clayout)
        with tla.cube():
            tla.print(local, length)

    return kernel


def _l1_print_kernel(
    dtype: type[tla.Numeric] = tla.Float32,
    *,
    layout: object = tla.arch.zN,
    length: int = 8,
    alignment: int = 256,
    vector: bool = False,
    make_ptr: bool = False,
):
    @tla.kernel
    def kernel(value: tla.Tensor) -> None:
        # tla.allocate requires a whole number of blocks, so an under-aligned
        # buffer can only be spelled with make_ptr.
        ptr = (
            tla.make_ptr(dtype, 0x1000, tla.AddressSpace.l1, assumed_align=alignment)
            if make_ptr
            else tla.allocate(40, dtype, tla.AddressSpace.l1, alignment)
        )
        local = tla.make_tensor_like(ptr, value, layout)
        if vector:
            with tla.vector():
                tla.print(local, length)
        else:
            with tla.cube():
                tla.print(local, length)

    return kernel


@tla.kernel
def _aiv_print_dynamic_internal_ub_tensor(value: tla.Tensor, dim: "index") -> None:
    ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    shape = tla.make_shape(dim, 4)
    stride = tla.make_stride(4, 1)
    layout = tla.make_layout(shape, stride)
    local = tla.make_tensor(ptr, layout)
    with tla.vector():
        tla.print(local, 4)


@tla.kernel
def _regionless_print_tensor(value: tla.Tensor) -> None:
    tla.print(value)


@tla.kernel
def _mixed_print_tensor(value: tla.Tensor) -> None:
    with tla.cube():
        with tla.vector():
            tla.print(value)


@tla.kernel
def _print_tensor_length_false(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, False)


@tla.kernel
def _print_tensor_length_zero(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 0)


@tla.kernel
def _print_tensor_length_negative(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, -1)


@tla.kernel
def _print_tensor_length_too_large(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 262_113)


def test_print_has_positional_only_variadic_surface() -> None:
    signature = inspect.signature(tla.print)
    assert str(signature) == "(value, *args, /)"
    assert signature.parameters["value"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert signature.parameters["args"].kind is inspect.Parameter.VAR_POSITIONAL


@pytest.mark.parametrize(
    ("dtype", "token"),
    (
        (tla.Float16, "f16"),
        (tla.Float32, "f32"),
        (tla.Int8, "i8"),
        (tla.Int16, "i16"),
        (tla.Int32, "i32"),
        (tla.UInt8, "ui8"),
        (tla.UInt16, "ui16"),
        (tla.UInt32, "ui32"),
    ),
    ids=("f16", "f32", "i8", "i16", "i32", "u8", "u16", "u32"),
)
@pytest.mark.parametrize(
    ("kernel", "core_region"),
    ((_aiv_print_tensor, "tla.vector"), (_aic_print_tensor, "tla.cube")),
    ids=("aiv", "aic"),
)
def test_print_tensor_emits_typed_dedicated_tensor_marker(
    dtype: type[tla.Numeric], token: str, kernel: object, core_region: str
) -> None:
    mlir = kernel.dump_mlir(type_args=(_host_tensor(dtype=dtype),))

    assert mlir.count("tla.print_tensor") == 1
    assert "tla.debug_print" not in mlir
    assert f"!tla.ptr<{token}, gm" in mlir
    assert core_region in mlir


def test_print_tensor_accepts_single_element_rank_one_tensor() -> None:
    mlir = _aiv_print_tensor.dump_mlir(type_args=(_host_tensor((1,)),))

    assert mlir.count("tla.print_tensor") == 1


def test_print_tensor_accepts_large_source_with_prefix_length() -> None:
    mlir = _aiv_print_tensor_prefix.dump_mlir(type_args=(_host_tensor((1024,)),))

    assert "tla.print_tensor" in mlir
    assert "shape = [1024]" in mlir or "shape = array<i64: 1024>" in mlir
    assert "length = %" in mlir
    assert "arith.constant 4 : i64" in mlir


def test_print_tensor_accepts_dynamic_rank_two_shape_with_explicit_length() -> None:
    tensor = _host_tensor()
    tensor.mark_compact_shape_dynamic(0)

    mlir = _aiv_print_tensor_prefix.dump_mlir(type_args=(tensor,))

    assert "tla.print_tensor" in mlir
    assert "shape = [-1, 4]" in mlir or "shape = array<i64: -1, 4>" in mlir
    assert "length = %" in mlir


def test_print_tensor_accepts_runtime_integer_length() -> None:
    mlir = _aiv_print_tensor_runtime_length.dump_mlir(
        type_args=(_host_tensor((32,)), tla.Int32(4))
    )

    assert "tla.print_tensor" in mlir
    assert "arith.extsi" in mlir
    assert "length = %" in mlir


@pytest.mark.parametrize(
    ("kernel", "control_flow_op"),
    (
        (_aiv_print_tensor_in_dynamic_if, "scf.if"),
        (_aiv_print_tensor_in_dynamic_for, "scf.for"),
        (_aiv_print_tensor_in_dynamic_while, "scf.while"),
    ),
    ids=("if", "for", "while"),
)
def test_print_tensor_accepts_dynamic_control_flow(
    kernel: object, control_flow_op: str
) -> None:
    mlir = kernel.dump_mlir(type_args=(_host_tensor(), 2))

    assert control_flow_op in mlir
    assert "tla.print_tensor" in mlir
    assert "dynamic_execution" in mlir
    assert mlir.index(control_flow_op) < mlir.index("tla.print_tensor")


def test_print_tensor_accepts_dynamic_rank_one_shape_with_explicit_length() -> None:
    tensor = _host_tensor((32,))
    tensor.mark_compact_shape_dynamic(0)

    mlir = _aiv_print_tensor_prefix.dump_mlir(type_args=(tensor,))

    assert "tla.print_tensor" in mlir
    assert "shape = [-1]" in mlir or "shape = array<i64: -1>" in mlir


def test_print_tensor_requires_explicit_length_for_dynamic_shape() -> None:
    tensor = _host_tensor()
    tensor.mark_compact_shape_dynamic(0)

    with pytest.raises(tla.TlaCoreAPIError, match="explicit length"):
        _aiv_print_tensor.dump_mlir(type_args=(tensor,))


@pytest.mark.parametrize(
    "layout",
    (
        tla.arch.zN,
        tla.arch.nZ,
        tla.arch.L0Clayout,
        tla.arch.zNUnAlign,
    ),
)
def test_print_tensor_accepts_generic_layouts(layout: object) -> None:
    mlir = _aiv_print_tensor_prefix.dump_mlir(type_args=(_host_packed_tensor(layout),))

    assert "tla.print_tensor" in mlir
    assert "shape = [32, 32]" in mlir or "shape = array<i64: 32, 32>" in mlir


def test_print_tensor_accepts_noncontiguous_linear_stride() -> None:
    mlir = _aiv_print_tensor_prefix.dump_mlir(
        type_args=(_host_tensor((2, 2), stride=(8, 1)),)
    )

    assert "tla.print_tensor" in mlir


def test_print_tensor_accepts_column_major_layout() -> None:
    mlir = _aiv_print_tensor_prefix.dump_mlir(
        type_args=(_host_tensor((32, 32), layout=tla.arch.ColumnMajor),)
    )

    assert "tla.print_tensor" in mlir


@pytest.mark.parametrize(
    ("tensor", "match"),
    ((_host_tensor((262_113,)), "explicit length"),),
)
def test_print_tensor_rejects_unsupported_tensor_contract(
    tensor: tla.Tensor, match: str
) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=match):
        _aiv_print_tensor.dump_mlir(type_args=(tensor,))


@pytest.mark.parametrize(
    ("dtype", "token"),
    (
        (tla.BFloat16, "bf16"),
        (tla.Int64, "i64"),
        (tla.UInt64, "u64"),
        (tla.Bool, "i1"),
    ),
)
def test_print_tensor_rejects_deferred_dtype_exact(
    dtype: type[tla.Numeric], token: str
) -> None:
    expected = (
        f"unsupported tensor dtype {token}; supported dtypes: "
        "f16, f32, i8, i16, i32, u8, u16, u32"
    )
    with pytest.raises(tla.TlaCoreAPIError) as exc_info:
        _aiv_print_tensor.dump_mlir(type_args=(_host_tensor(dtype=dtype),))
    assert expected in str(exc_info.value)


def test_print_tensor_rejects_host_call() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="lowered Tla IR"):
        tla.print(_host_tensor())


def test_print_tensor_accepts_dynamic_internal_ub_shape() -> None:
    tensor = _host_tensor()
    tensor.mark_compact_shape_dynamic(0)

    mlir = _aiv_print_dynamic_internal_ub_tensor.dump_mlir(type_args=(tensor, 4))

    assert "tla.make_shape %" in mlir
    assert "!tla.shape<?,4>" in mlir
    assert "tla.print_tensor" in mlir


def test_print_tensor_rejects_regionless_and_mixed_placement() -> None:
    tensor = _host_tensor()
    with pytest.raises(tla.TlaCoreAPIError, match="tla.cube.*tla.vector"):
        _regionless_print_tensor.dump_mlir(type_args=(tensor,))
    with pytest.raises(tla.TlaCoreAPIError, match="mixed"):
        _mixed_print_tensor.dump_mlir(type_args=(tensor,))


@pytest.mark.parametrize(
    "kernel",
    (
        _print_tensor_length_false,
        _print_tensor_length_zero,
        _print_tensor_length_negative,
        _print_tensor_length_too_large,
    ),
)
def test_print_tensor_rejects_invalid_length(kernel: object) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="length"):
        kernel.dump_mlir(type_args=(_host_tensor((32,)),))


def test_print_tensor_rejects_length_above_tensor_size() -> None:
    @tla.kernel
    def kernel(value: tla.Tensor) -> None:
        with tla.vector():
            tla.print(value, 5)

    with pytest.raises(tla.TlaCoreAPIError, match="element count"):
        kernel.dump_mlir(type_args=(_host_tensor((4,)),))


@pytest.mark.parametrize(
    ("kernel", "tensor", "match"),
    (
        (
            _l0c_print_kernel(tla.Float32, alignment=16, make_ptr=True),
            _host_tensor((16, 16)),
            "32-byte aligned L0C address",
        ),
        (_l0c_print_kernel(tla.Float32), _host_tensor((8, 16)), "complete 16x16"),
        (_l0c_print_kernel(tla.Float32, 128), _host_tensor((16, 16)), "exactly 256"),
        (_l1_print_kernel(vector=True), _host_tensor((4, 8)), "requires AIC placement"),
        (
            _l1_print_kernel(layout=tla.arch.RowMajor),
            _host_tensor((4, 8)),
            "requires a zN or nZ layout",
        ),
        (_l1_print_kernel(), _host_tensor((4, 4)), r"logical shape \[M, 8\]"),
        (
            _l1_print_kernel(alignment=4, make_ptr=True),
            _host_tensor((4, 8)),
            "32-byte aligned L1 address",
        ),
        (_l1_print_kernel(length=9), _host_tensor((4, 8)), "between 1 and 8 elements"),
    ),
)
def test_print_tensor_rejects_invalid_local_contract(kernel, tensor, match) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=match):
        kernel.dump_mlir(type_args=(tensor,))


def test_print_tensor_rejects_keyword_argument() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="keyword"):
        tla.print(value=_host_tensor())
