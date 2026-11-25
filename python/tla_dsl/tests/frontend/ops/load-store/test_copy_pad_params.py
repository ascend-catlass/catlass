from __future__ import annotations

import pathlib
import re
import subprocess
from typing import Any

import pytest

import catlass.tla as tla
from catlass.params import CopyPadParams
from catlass.runtime import TlaCoreAPIError
from catlass.tla.runtime import make_fake_tensor


def _require_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    tla_compile = (
        repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    )
    if not tla_compile.exists():
        pytest.skip("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


def _gm_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    shape: tuple[int, ...] = (8, 7),
) -> tla.Tensor:
    stride = (shape[1], 1) if len(shape) == 2 else (1,)
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    shape: tuple[int, ...] = (8, 8),
) -> tla.Tensor:
    stride = (shape[1], 1) if len(shape) == 2 else (1,)
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        addrspace=tla.AddressSpace.ub,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


def test_copy_pad_params_validate_rejects_oversized_side() -> None:
    with pytest.raises(ValueError, match="32B"):
        CopyPadParams(right_padding=9)._validate(elem_nbytes=4)


def test_copy_pad_params_validate_rejects_b64_nonzero_value() -> None:
    with pytest.raises(ValueError, match="must be 0"):
        CopyPadParams(padding_value=1)._validate(elem_nbytes=8)


@tla.kernel
def gm2ub_with_right_pad(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1),
        )


@tla.kernel
def gm2ub_with_pad_value(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(
                padding_value=0.1,
                right_padding=1,
            ),
        )


@tla.kernel
def gm2ub_no_pad(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(ub_tensor, gm_tensor)


@tla.kernel
def ub2gm_with_pad_rejected(ub_tensor: tla.Tensor, gm_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            gm_tensor,
            ub_tensor,
            params=tla.params.CopyPadParams(right_padding=1),
        )


def test_gm2ub_pad_emits_ir_pad_operands(compiler_tlair: Any) -> None:
    mlir = gm2ub_with_right_pad.dump_mlir(
        type_args=(_gm_tensor(), _ub_tensor()),
    )
    # L0C-isomorphic: typed params SSA from tla.CopyPadParams, then copy(params=).
    assert "tla.CopyPadParams" in mlir
    assert "!tla.copy_pad_params" in mlir
    assert re.search(r"tla\.copy\b[^\n]*,\s*%[^\n]+", mlir)


def test_gm2ub_pad_value_encodes_f32_bits(compiler_tlair: Any) -> None:
    """padding_value=0.1 must emit Float32._host_bits bits (decimal 1036831949)."""
    mlir = gm2ub_with_pad_value.dump_mlir(
        type_args=(_gm_tensor(), _ub_tensor()),
    )
    assert "tla.CopyPadParams" in mlir
    assert "1036831949" in mlir


def test_gm2ub_default_has_no_pad_attrs(compiler_tlair: Any) -> None:
    mlir = gm2ub_no_pad.dump_mlir(type_args=(_gm_tensor(), _ub_tensor()))
    assert "tla.CopyPadParams" not in mlir
    assert "copy_pad_params" not in mlir


def test_ub2gm_rejects_copy_pad_params(compiler_tlair: Any) -> None:
    with pytest.raises(TlaCoreAPIError, match="CopyPadParams"):
        ub2gm_with_pad_rejected.dump_mlir(
            type_args=(_ub_tensor(), _gm_tensor()),
        )


def test_gm2ub_pad_rejects_dst_too_narrow(compiler_tlair: Any) -> None:
    with pytest.raises(TlaCoreAPIError, match="innermost origin extent"):
        gm2ub_pad_dst_too_narrow.dump_mlir(
            type_args=(_gm_tensor(shape=(8, 7)), _ub_tensor(shape=(8, 7))),
        )


def test_gm2ub_pad_rejects_unsafe_dtype(compiler_tlair: Any) -> None:
    with pytest.raises(TlaCoreAPIError, match="not supported for dtype"):
        gm2ub_pad_fp8_rejected.dump_mlir(
            type_args=(
                _gm_tensor(dtype=tla.Float8E4M3FN, shape=(8, 7)),
                _ub_tensor(dtype=tla.Float8E4M3FN, shape=(8, 8)),
            ),
        )


def test_gm2ub_pad_float_padding_value_encodes_as_float_bits(
    compiler_tlair: Any,
) -> None:
    """f32 padding_value=1.0 encodes IEEE 1.0f bits (bare int on f32 rejected)."""
    mlir = gm2ub_pad_value_float_literal.dump_mlir(
        type_args=(_gm_tensor(), _ub_tensor()),
    )
    # IEEE-754 f32 1.0 bit pattern as decimal i64 constant in IR.
    assert "1065353216" in mlir


def test_padding_value_to_i64_bits_strict_dtype() -> None:
    from catlass.core_api import _padding_value_to_i64_bits

    assert _padding_value_to_i64_bits(1.0, "f32", 4) == 0x3F800000
    assert _padding_value_to_i64_bits(1, "i32", 4) == 1
    assert _padding_value_to_i64_bits(False, "i1", 1) == 0
    assert _padding_value_to_i64_bits(True, "i1", 1) == 1
    assert _padding_value_to_i64_bits(tla.Bool(True), "i1", 1) == 1
    # Numeric must match tile dtype exactly.
    assert _padding_value_to_i64_bits(tla.Int32(1.0), "i32", 4) == 1
    assert _padding_value_to_i64_bits(tla.Float32(1), "f32", 4) == 0x3F800000
    # bf16 uses round-to-nearest-even (_host_bits), not truncating >> 16.
    assert _padding_value_to_i64_bits(0.1, "bf16", 2) == tla.BFloat16._host_bits(0.1)
    # Bare int 0 is the dataclass default: universal zero-fill.
    assert _padding_value_to_i64_bits(0, "f32", 4) == 0
    assert _padding_value_to_i64_bits(0, "i1", 1) == 0
    with pytest.raises(TypeError, match="bare int|must be float"):
        _padding_value_to_i64_bits(1, "f32", 4)
    with pytest.raises(TypeError, match="integer dtype"):
        _padding_value_to_i64_bits(1.0, "i32", 4)
    with pytest.raises(TypeError, match="does not match"):
        _padding_value_to_i64_bits(tla.Float32(1), "i32", 4)
    with pytest.raises(TypeError, match="does not match"):
        _padding_value_to_i64_bits(tla.Float16(1), "f32", 4)
    with pytest.raises(TypeError, match="does not match"):
        _padding_value_to_i64_bits(tla.Int16(1), "i32", 4)
    with pytest.raises(TypeError, match="bool/tla.Bool"):
        _padding_value_to_i64_bits(1, "i1", 1)
    with pytest.raises(TypeError, match="must not be bool"):
        _padding_value_to_i64_bits(True, "f32", 4)
    with pytest.raises(ValueError, match="overflows"):
        _padding_value_to_i64_bits(1e5, "f16", 2)


def test_gm2ub_pad_rejects_float_padding_on_int_dtype(compiler_tlair: Any) -> None:
    with pytest.raises(TlaCoreAPIError, match="integer dtype"):
        gm2ub_pad_int_float_value.dump_mlir(
            type_args=(
                _gm_tensor(dtype=tla.Int32, shape=(8, 7)),
                _ub_tensor(dtype=tla.Int32, shape=(8, 8)),
            ),
        )


def test_gm2ub_pad_rejects_float32_numeric_on_int_dtype(compiler_tlair: Any) -> None:
    """Float32(1) must not pass via host unwrap on integer tiles."""
    with pytest.raises(TlaCoreAPIError, match="does not match"):
        gm2ub_pad_int_float32_numeric.dump_mlir(
            type_args=(
                _gm_tensor(dtype=tla.Int32, shape=(8, 7)),
                _ub_tensor(dtype=tla.Int32, shape=(8, 8)),
            ),
        )


def test_gm2ub_pad_rejects_bare_int_on_float_dtype(compiler_tlair: Any) -> None:
    with pytest.raises(TlaCoreAPIError, match="bare int|must be float"):
        gm2ub_pad_value_bare_int_on_float.dump_mlir(
            type_args=(_gm_tensor(), _ub_tensor()),
        )


def test_gm2ub_pad_rejects_cross_width_float_numeric(compiler_tlair: Any) -> None:
    with pytest.raises(TlaCoreAPIError, match="does not match"):
        gm2ub_pad_float16_on_f32.dump_mlir(
            type_args=(_gm_tensor(), _ub_tensor()),
        )


def test_gm2ub_pad_accepts_int32_numeric_whole_float_host(
    compiler_tlair: Any,
) -> None:
    """Int32(1.0) padding_value on i32 tiles encodes as integer 1."""
    mlir = gm2ub_pad_int_int32_numeric_float_host.dump_mlir(
        type_args=(
            _gm_tensor(dtype=tla.Int32, shape=(8, 7)),
            _ub_tensor(dtype=tla.Int32, shape=(8, 8)),
        ),
    )
    assert "tla.CopyPadParams" in mlir
    assert "arith.constant 1 : i64" in mlir


def test_gm2ub_pad_rejects_bool_right_padding(compiler_tlair: Any) -> None:
    with pytest.raises(TlaCoreAPIError, match="bool"):
        gm2ub_pad_bool_right.dump_mlir(
            type_args=(_gm_tensor(), _ub_tensor()),
        )


def test_gm2ub_pad_rejects_static_uint32_oversized_right(
    compiler_tlair: Any,
) -> None:
    """UInt32(9)*4B=36B must hit the 32B side limit via pad-only const extract."""
    with pytest.raises(TlaCoreAPIError, match="32B|pad limit"):
        gm2ub_pad_uint32_right.dump_mlir(
            type_args=(
                _gm_tensor(dtype=tla.Int32, shape=(2, 7)),
                _ub_tensor(dtype=tla.Int32, shape=(2, 8)),
            ),
        )


def test_gm2ub_pad_rejects_int32_float_host_oversized_right(
    compiler_tlair: Any,
) -> None:
    """Int32(9.0) must participate in static 32B checks like Int32(9)."""
    with pytest.raises(TlaCoreAPIError, match="32B|pad limit"):
        gm2ub_pad_int32_float_host_right.dump_mlir(
            type_args=(
                _gm_tensor(dtype=tla.Int32, shape=(2, 7)),
                _ub_tensor(dtype=tla.Int32, shape=(2, 8)),
            ),
        )


def test_gm2ub_pad_uint32_float_host_right_emits_i64(
    compiler_tlair: Any,
) -> None:
    """UInt32(1.0) right_padding lowers as signless i64, not ui32 constant."""
    mlir = gm2ub_pad_uint32_float_host_right.dump_mlir(
        type_args=(
            _gm_tensor(dtype=tla.Int32, shape=(2, 7)),
            _ub_tensor(dtype=tla.Int32, shape=(2, 8)),
        ),
    )
    assert "arith.constant 1 : i64" in mlir
    assert "ui32" not in mlir
    assert "tla.CopyPadParams" in mlir


def test_arange_still_rejects_uint8_numeric_base(compiler_tlair: Any) -> None:
    """Pad unsigned extract must not widen shared _const_int_value for arange."""

    @tla.kernel
    def arange_uint8_base(dst: tla.Tensor) -> None:
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst_tile.store(tla.arange(tla.UInt8(3), dtype=tla.Int32))

    ub = make_fake_tensor(
        tla.Int32,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(
        (TlaCoreAPIError, TypeError, ValueError),
        match="integer literal|index SSA|base",
    ):
        arange_uint8_base.dump_mlir(type_args=(ub,))


def test_gm2ub_runtime_uint8_padding_value_compiles(
    compiler_tlair: Any, tmp_path: pathlib.Path
) -> None:
    """Runtime UInt8 padding_value: ui8→i8 (ucc) then arith.extui→i64."""
    tla_compile = _require_tla_compile()
    mlir = gm2ub_runtime_uint8_pad_value.dump_mlir(
        type_args=(
            _gm_tensor(dtype=tla.Int32, shape=(8, 7)),
            tla.UInt8(1),
        ),
    )
    assert "builtin.unrealized_conversion_cast" in mlir
    assert "arith.extui" in mlir
    path = tmp_path / "gm2ub_u8_pad_value.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@tla.kernel
def gm2ub_pad_uint32_right(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=tla.UInt32(9)),
        )


@tla.kernel
def gm2ub_pad_int32_float_host_right(
    gm_tensor: tla.Tensor, ub_tensor: tla.Tensor
) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=tla.Int32(9.0)),
        )


@tla.kernel
def gm2ub_pad_uint32_float_host_right(
    gm_tensor: tla.Tensor, ub_tensor: tla.Tensor
) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=tla.UInt32(1.0)),
        )


@tla.kernel
def gm2ub_runtime_uint8_pad_value(
    gm_tensor: tla.Tensor,
    pad_value: tla.UInt8,
) -> None:
    ub_ptr = tla.allocate(64, tla.Int32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(8, 8),
            tla.make_stride(8, 1),
            origin_shape=tla.make_shape(8, 8),
            layoutTag=tla.arch.RowMajor,
        ),
        tla.make_coord(0, 0),
    )
    with tla.vector():
        tla.copy(
            ub,
            gm_tensor,
            params=tla.params.CopyPadParams(
                right_padding=1,
                padding_value=pad_value,
            ),
        )


def test_gm2ub_pad_rejects_unaligned_multi_row_stride(compiler_tlair: Any) -> None:
    """dst stride must cover AlignUp(src+pad, ELE_NUM_PER_BLK) for multi-row."""
    with pytest.raises(TlaCoreAPIError, match="row stride"):
        gm2ub_pad_multi_row_narrow_stride.dump_mlir(
            type_args=(
                _gm_tensor(shape=(8, 7)),
                _ub_tensor(shape=(8, 9)),  # need=9, AlignUp(f32)=16
            ),
        )


def test_gm2ub_pad_rejects_non_block_multiple_multi_row_stride(
    compiler_tlair: Any,
) -> None:
    """dst stride 18 is >= AlignUp(9,8)=16 but not a multiple of 8 (f32 C0)."""
    with pytest.raises(TlaCoreAPIError, match="row stride|multiple"):
        gm2ub_pad_multi_row_narrow_stride.dump_mlir(
            type_args=(
                _gm_tensor(shape=(2, 7)),
                _ub_tensor(shape=(2, 18)),
            ),
        )


@tla.kernel
def gm2ub_with_right_pad_alloc(gm_tensor: tla.Tensor) -> None:
    """Kernel-arg GM only (UB allocated inside) so TlaCompile accepts it.

    UB is wider than GM by one element so right_padding=1 fits capacity check.
    """
    ub_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(8, 8),
            tla.make_stride(8, 1),
            origin_shape=tla.make_shape(8, 8),
            layoutTag=tla.arch.RowMajor,
        ),
        tla.make_coord(0, 0),
    )
    with tla.vector():
        tla.copy(
            ub,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1),
        )


@tla.kernel
def gm2ub_no_pad_alloc(gm_tensor: tla.Tensor) -> None:
    ub_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor_like(ub_ptr, gm_tensor, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(ub, gm_tensor)


@tla.kernel
def gm2ub_pad_dst_too_narrow(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1),
        )


@tla.kernel
def gm2ub_pad_fp8_rejected(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1),
        )


@tla.kernel
def gm2ub_pad_value_float_literal(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(
                padding_value=1.0,
                right_padding=1,
            ),
        )


@tla.kernel
def gm2ub_pad_value_bare_int_on_float(
    gm_tensor: tla.Tensor, ub_tensor: tla.Tensor
) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(
                padding_value=1,
                right_padding=1,
            ),
        )


@tla.kernel
def gm2ub_pad_float16_on_f32(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(
                padding_value=tla.Float16(1),
                right_padding=1,
            ),
        )


@tla.kernel
def gm2ub_pad_int_float_value(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(
                padding_value=1.0,
                right_padding=1,
            ),
        )


@tla.kernel
def gm2ub_pad_int_float32_numeric(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(
                padding_value=tla.Float32(1),
                right_padding=1,
            ),
        )


@tla.kernel
def gm2ub_pad_int_int32_numeric_float_host(
    gm_tensor: tla.Tensor, ub_tensor: tla.Tensor
) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(
                padding_value=tla.Int32(1.0),
                right_padding=1,
            ),
        )


@tla.kernel
def gm2ub_pad_bool_right(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(
                right_padding=tla.Bool(True),
            ),
        )


@tla.kernel
def gm2ub_pad_multi_row_narrow_stride(
    gm_tensor: tla.Tensor, ub_tensor: tla.Tensor
) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=2),
        )


@tla.kernel
def gm2ub_runtime_pad(
    gm_tensor: tla.Tensor,
    ub_tensor: tla.Tensor,
    left: tla.Int32,
    right: tla.Int32,
    value_bits: tla.Int64,
) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(
                padding_value=value_bits,
                left_padding=left,
                right_padding=right,
            ),
        )


@tla.kernel
def gm2ub_runtime_right_only_alloc(
    gm_tensor: tla.Tensor,
    right: tla.Int32,
) -> None:
    ub_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(8, 8),
            tla.make_stride(8, 1),
            origin_shape=tla.make_shape(8, 8),
            layoutTag=tla.arch.RowMajor,
        ),
        tla.make_coord(0, 0),
    )
    with tla.vector():
        tla.copy(
            ub,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=right),
        )


def test_gm2ub_runtime_pad_emits_ssa_operands(compiler_tlair: Any) -> None:
    mlir = gm2ub_runtime_pad.dump_mlir(
        type_args=(
            _gm_tensor(),
            _ub_tensor(),
            tla.Int32(0),
            tla.Int32(1),
            tla.Int64(0),
        ),
    )
    assert "tla.CopyPadParams" in mlir
    assert "!tla.copy_pad_params" in mlir
    # Runtime args must flow as SSA into CopyPadParams, not disappear as attrs.
    assert "pad_enable" not in mlir
    # Match three SSA operands (kernel args / extsi of args), not folded constants.
    copy_pad = re.search(
        r"tla\.CopyPadParams\s+(%[^\s,]+)\s*,\s*(%[^\s,]+)\s*,\s*(%[^\s,]+)",
        mlir,
    )
    assert copy_pad is not None, mlir
    for ssa in copy_pad.groups():
        # Operand itself must not be an arith.constant result name only used as
        # a folded pad attr; accept extsi/index_cast of a block argument.
        const_def = re.search(
            rf"{re.escape(ssa)}\s*=\s*arith\.constant\b",
            mlir,
        )
        assert const_def is None, (ssa, mlir)


def test_gm2ub_runtime_right_skips_static_byte_check(compiler_tlair: Any) -> None:
    """Runtime right_padding cannot prove pad-bytes; frontend must accept."""
    mlir = gm2ub_runtime_right_only_alloc.dump_mlir(
        type_args=(_gm_tensor(), tla.Int32(1)),
    )
    assert "tla.CopyPadParams" in mlir
    assert "!tla.copy_pad_params" in mlir


def _call_trailing_i64s(
    lowered: str, *, callee: str, expected_nargs: int | None = 31
) -> list[int]:
    match = re.search(rf"call @{re.escape(callee)}\(([^)]*)\)", lowered)
    assert match is not None, lowered
    operands = [op.strip() for op in match.group(1).split(",")]
    if expected_nargs is not None:
        assert len(operands) == expected_nargs, (len(operands), lowered)
    assert len(operands) >= 4, (len(operands), lowered)
    values: list[int] = []
    for ssa in operands[-4:]:
        const = re.search(
            rf"{re.escape(ssa)}\s*=\s*llvm\.mlir\.constant\((-?\d+)\s*:\s*i64\)",
            lowered,
        )
        assert const is not None, (ssa, lowered)
        values.append(int(const.group(1)))
    return values


def _call_trailing_ssa(
    lowered: str, *, callee: str, expected_nargs: int | None = 31
) -> list[str]:
    match = re.search(rf"call @{re.escape(callee)}\(([^)]*)\)", lowered)
    assert match is not None, lowered
    operands = [op.strip() for op in match.group(1).split(",")]
    if expected_nargs is not None:
        assert len(operands) == expected_nargs, (len(operands), lowered)
    assert len(operands) >= 4, (len(operands), lowered)
    return operands[-4:]


def test_lowering_pad_enable_and_right(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm2ub_with_right_pad_alloc.dump_mlir(type_args=(_gm_tensor(),))
    path = tmp_path / "gm2ub_pad.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    pad_enable, left, right, value_bits = _call_trailing_i64s(
        result.stdout, callee="copy_gm_RowMajor_to_ub_RowMajor_float"
    )
    assert (pad_enable, left, right, value_bits) == (1, 0, 1, 0)


def test_lowering_no_pad_enable_zero(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm2ub_no_pad_alloc.dump_mlir(type_args=(_gm_tensor(),))
    path = tmp_path / "gm2ub_nopad.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    pad_enable, left, right, value_bits = _call_trailing_i64s(
        result.stdout, callee="copy_gm_RowMajor_to_ub_RowMajor_float"
    )
    assert (pad_enable, left, right, value_bits) == (0, 0, 0, 0)


def test_lowering_runtime_right_not_folded(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm2ub_runtime_right_only_alloc.dump_mlir(
        type_args=(_gm_tensor(), tla.Int32(1)),
    )
    path = tmp_path / "gm2ub_runtime_pad.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    pad_enable_ssa, left_ssa, right_ssa, value_ssa = _call_trailing_ssa(
        result.stdout, callee="copy_gm_RowMajor_to_ub_RowMajor_float"
    )
    # pad_enable / left / value stay constants; right comes from a kernel arg.
    enable_const = re.search(
        rf"{re.escape(pad_enable_ssa)}\s*=\s*llvm\.mlir\.constant\(1\s*:\s*i64\)",
        result.stdout,
    )
    left_const = re.search(
        rf"{re.escape(left_ssa)}\s*=\s*llvm\.mlir\.constant\(0\s*:\s*i64\)",
        result.stdout,
    )
    value_const = re.search(
        rf"{re.escape(value_ssa)}\s*=\s*llvm\.mlir\.constant\(0\s*:\s*i64\)",
        result.stdout,
    )
    right_const = re.search(
        rf"{re.escape(right_ssa)}\s*=\s*llvm\.mlir\.constant\((-?\d+)\s*:\s*i64\)",
        result.stdout,
    )
    assert enable_const and left_const and value_const
    assert right_const is None, result.stdout


def _gm_vector(
    dtype: type[tla.Numeric] = tla.Float32,
    length: int = 7,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (length,),
        (1,),
        origin_shape=(length,),
        layout_tag=tla.arch.VectorLayout,
    )


def _ub_vector(
    dtype: type[tla.Numeric] = tla.Float32,
    length: int = 8,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (length,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(length,),
        layout_tag=tla.arch.VectorLayout,
    )


@tla.kernel
def gm2ub_vector_no_pad(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(ub_tensor, gm_tensor)


@tla.kernel
def gm2ub_vector_with_right_pad(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1),
        )


@tla.kernel
def gm2ub_vector_pad_alloc(gm_tensor: tla.Tensor) -> None:
    ub_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(8),
            tla.make_stride(1),
            origin_shape=tla.make_shape(8),
            layoutTag=tla.arch.VectorLayout,
        ),
        tla.make_coord(0),
    )
    with tla.vector():
        tla.copy(
            ub,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1),
        )


@tla.kernel
def gm2ub_vector_runtime_right_alloc(
    gm_tensor: tla.Tensor,
    right: tla.Int32,
) -> None:
    ub_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(8),
            tla.make_stride(1),
            origin_shape=tla.make_shape(8),
            layoutTag=tla.arch.VectorLayout,
        ),
        tla.make_coord(0),
    )
    with tla.vector():
        tla.copy(
            ub,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=right),
        )


@tla.kernel
def gm2ub_vector_bool_pad_alloc(gm_tensor: tla.Tensor) -> None:
    ub_ptr = tla.allocate(32, tla.Bool, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(32),
            tla.make_stride(1),
            origin_shape=tla.make_shape(32),
            layoutTag=tla.arch.VectorLayout,
        ),
        tla.make_coord(0),
    )
    with tla.vector():
        tla.copy(
            ub,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1, padding_value=False),
        )


def test_gm2ub_vector_two_arg_emits_route(compiler_tlair: Any) -> None:
    mlir = gm2ub_vector_no_pad.dump_mlir(
        type_args=(_gm_vector(), _ub_vector()),
    )
    assert "VectorLayout" in mlir
    assert "tla.copy" in mlir
    assert "tla.CopyPadParams" not in mlir
    assert "copy_pad_params" not in mlir


def test_gm2ub_vector_pad_emits_copy_pad_params(compiler_tlair: Any) -> None:
    mlir = gm2ub_vector_with_right_pad.dump_mlir(
        type_args=(_gm_vector(), _ub_vector()),
    )
    assert "tla.CopyPadParams" in mlir
    assert "!tla.copy_pad_params" in mlir
    assert re.search(r"tla\.copy\b[^\n]*,\s*%[^\n]+", mlir)
    assert "pad_enable" not in mlir


def test_gm2ub_vector_bool_pad_emits_copy_pad_params(compiler_tlair: Any) -> None:
    mlir = gm2ub_vector_with_right_pad.dump_mlir(
        type_args=(
            _gm_vector(dtype=tla.Bool, length=31),
            _ub_vector(dtype=tla.Bool, length=32),
        ),
    )
    assert "tla.CopyPadParams" in mlir
    assert "!tla.copy_pad_params" in mlir


def test_lowering_vector_pad_enable_and_right(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm2ub_vector_pad_alloc.dump_mlir(type_args=(_gm_vector(),))
    path = tmp_path / "gm2ub_vector_pad.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "copy_gm_VectorLayout_to_ub_VectorLayout_float" in result.stdout
    pad_enable, left, right, value_bits = _call_trailing_i64s(
        result.stdout, callee="copy_gm_VectorLayout_to_ub_VectorLayout_float"
    )
    assert (pad_enable, left, right, value_bits) == (1, 0, 1, 0)


def test_lowering_vector_bool_pad_enable_and_right(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    gm = _gm_vector(dtype=tla.Bool, length=31)
    mlir = gm2ub_vector_bool_pad_alloc.dump_mlir(type_args=(gm,))
    path = tmp_path / "gm2ub_vector_bool_pad.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "copy_gm_VectorLayout_to_ub_VectorLayout_bool" in result.stdout
    pad_enable, left, right, value_bits = _call_trailing_i64s(
        result.stdout, callee="copy_gm_VectorLayout_to_ub_VectorLayout_bool"
    )
    assert (pad_enable, left, right, value_bits) == (1, 0, 1, 0)


def test_lowering_vector_runtime_right_not_folded(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm2ub_vector_runtime_right_alloc.dump_mlir(
        type_args=(_gm_vector(), tla.Int32(1)),
    )
    path = tmp_path / "gm2ub_vector_runtime_pad.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    pad_enable_ssa, left_ssa, right_ssa, value_ssa = _call_trailing_ssa(
        result.stdout, callee="copy_gm_VectorLayout_to_ub_VectorLayout_float"
    )
    enable_const = re.search(
        rf"{re.escape(pad_enable_ssa)}\s*=\s*llvm\.mlir\.constant\(1\s*:\s*i64\)",
        result.stdout,
    )
    right_const = re.search(
        rf"{re.escape(right_ssa)}\s*=\s*llvm\.mlir\.constant\((-?\d+)\s*:\s*i64\)",
        result.stdout,
    )
    assert enable_const
    assert right_const is None, result.stdout


def test_gm2ub_b64_pad_accepts_zero_value(compiler_tlair: Any) -> None:
    mlir = gm2ub_with_right_pad.dump_mlir(
        type_args=(
            _gm_tensor(dtype=tla.Float64, shape=(8, 3)),
            _ub_tensor(dtype=tla.Float64, shape=(8, 4)),
        ),
    )
    assert "tla.CopyPadParams" in mlir


def test_gm2ub_b64_pad_accepts_numeric_zero_value(compiler_tlair: Any) -> None:
    @tla.kernel
    def gm2ub_b64_numeric_zero(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
        with tla.vector():
            tla.copy(
                ub_tensor,
                gm_tensor,
                params=CopyPadParams(right_padding=1, padding_value=tla.Int64(0)),
            )

    mlir = gm2ub_b64_numeric_zero.dump_mlir(
        type_args=(
            _gm_tensor(dtype=tla.Int64, shape=(8, 3)),
            _ub_tensor(dtype=tla.Int64, shape=(8, 4)),
        ),
    )
    assert "tla.CopyPadParams" in mlir


def test_gm2ub_b64_pad_requires_zero_value(compiler_tlair: Any) -> None:
    with pytest.raises(TlaCoreAPIError, match="padding_value|8B|must be 0"):
        gm2ub_with_pad_value.dump_mlir(
            type_args=(
                _gm_tensor(dtype=tla.Int64, shape=(8, 3)),
                _ub_tensor(dtype=tla.Int64, shape=(8, 4)),
            )
        )


def test_gm2ub_b64_pad_rejects_nonzero_numeric_value(compiler_tlair: Any) -> None:
    @tla.kernel
    def gm2ub_b64_numeric_nonzero(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
        with tla.vector():
            tla.copy(
                ub_tensor,
                gm_tensor,
                params=CopyPadParams(right_padding=1, padding_value=tla.Int64(1)),
            )

    with pytest.raises(TlaCoreAPIError, match="padding_value|8B|must be 0"):
        gm2ub_b64_numeric_nonzero.dump_mlir(
            type_args=(
                _gm_tensor(dtype=tla.Int64, shape=(8, 3)),
                _ub_tensor(dtype=tla.Int64, shape=(8, 4)),
            )
        )


def test_gm2ub_b64_pad_rejects_nonzero_float_numeric_value(
    compiler_tlair: Any,
) -> None:
    @tla.kernel
    def gm2ub_b64_f64_nonzero(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
        with tla.vector():
            tla.copy(
                ub_tensor,
                gm_tensor,
                params=CopyPadParams(right_padding=1, padding_value=tla.Float64(0.5)),
            )

    with pytest.raises(TlaCoreAPIError, match="padding_value|8B|must be 0"):
        gm2ub_b64_f64_nonzero.dump_mlir(
            type_args=(
                _gm_tensor(dtype=tla.Float64, shape=(8, 3)),
                _ub_tensor(dtype=tla.Float64, shape=(8, 4)),
            )
        )


def _gm_vec(dtype: type[tla.Numeric] = tla.Float32, n: int = 48) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (n,),
        (1,),
        origin_shape=(n,),
        layout_tag=tla.arch.VectorLayout,
    )


def _l1_vec(dtype: type[tla.Numeric] = tla.Float32, n: int = 56) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (n,),
        (1,),
        addrspace=tla.AddressSpace.l1,
        origin_shape=(n,),
        layout_tag=tla.arch.VectorLayout,
    )


@tla.kernel
def gm2l1_vl_with_right_pad(gm_tensor: tla.Tensor, l1_tensor: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(
            l1_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1, padding_value=0.0),
        )


@tla.kernel
def gm2l1_vl_bool_pad(gm_tensor: tla.Tensor, l1_tensor: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(
            l1_tensor,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1, padding_value=False),
        )


def test_gm2l1_vl_pad_emits_ir_pad_operands(compiler_tlair: Any) -> None:
    mlir = gm2l1_vl_with_right_pad.dump_mlir(
        type_args=(_gm_vec(n=48), _l1_vec(n=56)),
    )
    assert "tla.CopyPadParams" in mlir
    assert "!tla.copy_pad_params" in mlir
    assert "tla.copy" in mlir


@tla.kernel
def gm2l1_vl_with_right_pad_alloc(gm_tensor: tla.Tensor) -> None:
    """Kernel-arg GM only (L1 allocated inside) so TlaCompile accepts it."""
    l1_ptr = tla.allocate(56, tla.Float32, tla.AddressSpace.l1, 256)
    l1 = tla.make_tensor(
        l1_ptr,
        tla.make_layout(
            tla.make_shape(56),
            tla.make_stride(1),
            origin_shape=tla.make_shape(56),
            layoutTag=tla.arch.VectorLayout,
        ),
        tla.make_coord(0),
    )
    with tla.cube():
        tla.copy(
            l1,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1, padding_value=0.0),
        )


def test_gm2l1_vl_pad_lowering_trailing_i64s(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm2l1_vl_with_right_pad_alloc.dump_mlir(type_args=(_gm_vec(n=48),))
    path = tmp_path / "gm2l1_vl_pad.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    pad_enable, left, right, value_bits = _call_trailing_i64s(
        result.stdout,
        callee="copy_gm_VectorLayout_to_l1_VectorLayout_float",
        expected_nargs=None,
    )
    assert (pad_enable, left, right, value_bits) == (1, 0, 1, 0)


def test_gm2l1_vl_bool_pad_accepts(compiler_tlair: Any) -> None:
    mlir = gm2l1_vl_bool_pad.dump_mlir(
        type_args=(_gm_vec(dtype=tla.Bool, n=48), _l1_vec(dtype=tla.Bool, n=56)),
    )
    assert "tla.CopyPadParams" in mlir


def test_gm2l1_rm_zn_pad_rejected(compiler_tlair: Any) -> None:
    from catlass.core_api import _remap_tensor_like_prefix_fields_for_layout_trees

    @tla.kernel
    def gm2l1_rm_pad(gm_tensor: tla.Tensor, l1_tensor: tla.Tensor) -> None:
        with tla.cube():
            tla.copy(
                l1_tensor,
                gm_tensor,
                params=CopyPadParams(right_padding=1),
            )

    origin = (32, 32)
    geometry = _remap_tensor_like_prefix_fields_for_layout_trees(
        origin, tla.Float32.dtype, str(tla.arch.zN)
    )
    assert geometry is not None
    with pytest.raises(TlaCoreAPIError, match="VectorLayout→VectorLayout"):
        gm2l1_rm_pad.dump_mlir(
            type_args=(
                _gm_tensor(shape=origin),
                make_fake_tensor(
                    tla.Float32,
                    geometry.shape,
                    geometry.stride,
                    addrspace=tla.AddressSpace.l1,
                    origin_shape=origin,
                    layout_tag=tla.arch.zN,
                ),
            )
        )


def test_gm2l1_vl_pad_rejects_fp8(compiler_tlair: Any) -> None:
    @tla.kernel
    def gm2l1_fp8_pad(gm_tensor: tla.Tensor, l1_tensor: tla.Tensor) -> None:
        with tla.cube():
            tla.copy(
                l1_tensor,
                gm_tensor,
                params=CopyPadParams(right_padding=1),
            )

    with pytest.raises(TlaCoreAPIError, match="not supported for dtype"):
        gm2l1_fp8_pad.dump_mlir(
            type_args=(
                _gm_vec(dtype=tla.Float8E4M3FN, n=48),
                _l1_vec(dtype=tla.Float8E4M3FN, n=56),
            )
        )


def test_gm2l1_vl_hif8_pad_accepts(compiler_tlair: Any) -> None:
    @tla.kernel
    def gm2l1_hif8_pad(gm_tensor: tla.Tensor, l1_tensor: tla.Tensor) -> None:
        with tla.cube():
            tla.copy(
                l1_tensor,
                gm_tensor,
                params=CopyPadParams(right_padding=1, padding_value=0),
            )

    mlir = gm2l1_hif8_pad.dump_mlir(
        type_args=(
            _gm_vec(dtype=tla.HiFloat8, n=48),
            _l1_vec(dtype=tla.HiFloat8, n=56),
        )
    )
    assert "tla.CopyPadParams" in mlir


@tla.kernel
def gm2l1_vl_hif8_pad_alloc(gm_tensor: tla.Tensor) -> None:
    """Kernel-arg GM only so TlaCompile covers the hif8 uint8 ExtParams BC path."""
    l1_ptr = tla.allocate(56, tla.HiFloat8, tla.AddressSpace.l1, 256)
    l1 = tla.make_tensor(
        l1_ptr,
        tla.make_layout(
            tla.make_shape(56),
            tla.make_stride(1),
            origin_shape=tla.make_shape(56),
            layoutTag=tla.arch.VectorLayout,
        ),
        tla.make_coord(0),
    )
    with tla.cube():
        tla.copy(
            l1,
            gm_tensor,
            params=tla.params.CopyPadParams(right_padding=1, padding_value=0),
        )


def test_gm2l1_vl_hif8_pad_lowering_trailing_i64s(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm2l1_vl_hif8_pad_alloc.dump_mlir(
        type_args=(_gm_vec(dtype=tla.HiFloat8, n=48),)
    )
    path = tmp_path / "gm2l1_vl_hif8_pad.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "copy_gm_VectorLayout_to_l1_VectorLayout_hifloat8_t" in result.stdout
    pad_enable, left, right, value_bits = _call_trailing_i64s(
        result.stdout,
        callee="copy_gm_VectorLayout_to_l1_VectorLayout_hifloat8_t",
        expected_nargs=None,
    )
    assert (pad_enable, left, right, value_bits) == (1, 0, 1, 0)
