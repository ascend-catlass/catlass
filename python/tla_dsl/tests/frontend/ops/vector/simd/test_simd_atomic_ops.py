# -------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the
# License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------

"""Frontend surface of tla.arch.atomic_add / atomic_exch in SIMD regions.

v1 scope: pure pointer form, gm only, and the old value is only promised to be
writable straight back into a same-type tensor. No uint->int conversion and no
unsigned arithmetic / comparison is offered around the returned value.
"""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

_LENGTH = 8
_ADD_DTYPES = (tla.Int32, tla.UInt32, tla.Float32, tla.Int64, tla.UInt64)
_EXCH_DTYPES = (tla.UInt32, tla.UInt64)


def _tensor(dtype: type[tla.Numeric], addrspace: tla.AddressSpace = tla.AddressSpace.gm) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (_LENGTH,),
        (1,),
        addrspace=addrspace,
        origin_shape=(_LENGTH,),
        layout_tag=tla.arch.RowMajor,
    )


def _mlir_elem(dtype: type[tla.Numeric]) -> str:
    """``u32`` -> ``ui32``: the MLIR spelling of an unsigned element type."""
    token = dtype.dtype
    return "ui" + token[1:] if token.startswith("u") else token


def _dump_add(dtype: type[tla.Numeric]) -> str:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            old = tla.arch.atomic_add(out.ptr, 1)
            out[0] = old

    return k.dump_mlir(type_args=(_tensor(dtype),))


# --- emitted IR ------------------------------------------------------------


@pytest.mark.parametrize("dtype", _ADD_DTYPES)
def test_atomic_add_emits_simd_atomic_op(dtype: type[tla.Numeric]) -> None:
    mlir = _dump_add(dtype)
    assert "tla.simd_atomic_add" in mlir
    assert f"!tla.ptr<{_mlir_elem(dtype)}, gm" in mlir
    assert "tla.vector" in mlir


@pytest.mark.parametrize("dtype", _EXCH_DTYPES)
def test_atomic_exch_emits_simd_atomic_op(dtype: type[tla.Numeric]) -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            old = tla.arch.atomic_exch(out.ptr, 7)
            out[0] = old

    mlir = k.dump_mlir(type_args=(_tensor(dtype),))
    assert "tla.simd_atomic_exch" in mlir
    assert f"!tla.ptr<{_mlir_elem(dtype)}, gm" in mlir


def test_atomic_add_on_unsigned_pointer_keeps_unsigned_operand() -> None:
    """A u32 pointer takes a ui32 addend; the literal is bridged, not converted."""
    mlir = _dump_add(tla.UInt32)
    assert "builtin.unrealized_conversion_cast" in mlir
    assert "ui32" in mlir


@pytest.mark.parametrize(
    "dtype",
    [tla.Int32, tla.UInt32, tla.Float32, tla.Int64, tla.UInt64],
)
def test_atomic_add_old_value_is_stored_straight_back(
    dtype: type[tla.Numeric],
) -> None:
    """The old value is written into the same-type tensor: no conversion op.

    ``out[0] = old`` is accepted only when ``old`` already carries the tensor's
    dtype, and the emitted store must use that element type, which is what the
    v1 contract promises (u32 -> u32, i64 -> i64, ...).
    """
    elem = _mlir_elem(dtype)
    mlir = _dump_add(dtype)
    atomic_lines = [line for line in mlir.splitlines() if "tla.simd_atomic_add" in line]
    assert len(atomic_lines) == 1
    assert f"{elem} -> {elem}" in atomic_lines[0]
    assert "tla.scalar_store" in mlir
    assert ".to(" not in mlir


def test_atomic_exch_old_value_is_stored_straight_back() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            old = tla.arch.atomic_exch(out.ptr, 1)
            out[0] = old

    mlir = k.dump_mlir(type_args=(_tensor(tla.UInt32),))
    atomic_lines = [line for line in mlir.splitlines() if "tla.simd_atomic_exch" in line]
    assert len(atomic_lines) == 1
    assert "ui32 -> ui32" in atomic_lines[0]
    assert "tla.scalar_store" in mlir
    assert ".to(" not in mlir


def test_atomic_add_accepts_matching_numeric_value() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            old = tla.arch.atomic_add(out.ptr, tla.UInt32(9))
            out[0] = old

    mlir = k.dump_mlir(type_args=(_tensor(tla.UInt32),))
    assert "tla.simd_atomic_add" in mlir


# --- placement -------------------------------------------------------------


def test_atomic_add_accepts_cube_region() -> None:
    """The same scalar instruction is issued from AIC: tla.cube() is accepted."""
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.cube():
            tla.arch.atomic_add(out.ptr, 1)

    mlir = k.dump_mlir(type_args=(_tensor(tla.Int32),))
    assert "tla.simd_atomic_add" in mlir
    assert "tla.cube" in mlir


def test_atomic_exch_accepts_cube_region() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.cube():
            tla.arch.atomic_exch(out.ptr, 1)

    mlir = k.dump_mlir(type_args=(_tensor(tla.UInt32),))
    assert "tla.simd_atomic_exch" in mlir
    assert "tla.cube" in mlir


def test_atomic_add_rejects_non_vector_region() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        tla.arch.atomic_add(out.ptr, 1)

    with pytest.raises(tla.TlaCoreAPIError, match="must be nested inside tla.cube\\(\\) or tla.vector\\(\\)"):
        k.dump_mlir(type_args=(_tensor(tla.Int32),))


def test_atomic_add_rejects_nested_cube_inside_vector() -> None:
    """Mixed placement is rejected at the call site, not at kernel split."""

    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            with tla.cube():
                tla.arch.atomic_add(out.ptr, 1)

    with pytest.raises(tla.TlaCoreAPIError, match="mixed cube/vector placement"):
        k.dump_mlir(type_args=(_tensor(tla.Int32),))


def test_atomic_add_rejects_simt_vec_func() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.atomic_add(out.ptr, 1)

    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        k.dump_mlir(type_args=(_tensor(tla.Int32),))


def test_atomic_add_rejects_simd_vec_func() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simd"):
                tla.arch.atomic_add(out.ptr, 1)

    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        k.dump_mlir(type_args=(_tensor(tla.Int32),))


# --- type matrix -----------------------------------------------------------


@pytest.mark.parametrize("dtype", [tla.Int8, tla.Int16, tla.UInt16, tla.Float16, tla.BFloat16])
def test_atomic_add_rejects_unsupported_element_type(dtype: type[tla.Numeric]) -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.atomic_add(out.ptr, 1)

    with pytest.raises(tla.TlaCoreAPIError, match="supported types are i32, u32, f32, i64, u64"):
        k.dump_mlir(type_args=(_tensor(dtype),))


@pytest.mark.parametrize("dtype", [tla.Int32, tla.Int64, tla.Float32])
def test_atomic_exch_rejects_non_unsigned_element_type(dtype: type[tla.Numeric]) -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.atomic_exch(out.ptr, 1)

    with pytest.raises(tla.TlaCoreAPIError, match="supported types are u32, u64"):
        k.dump_mlir(type_args=(_tensor(dtype),))


def test_atomic_add_rejects_ub_pointer() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.atomic_add(out.ptr, 1)

    with pytest.raises(tla.TlaCoreAPIError, match="expected a gm pointer, got ub"):
        k.dump_mlir(type_args=(_tensor(tla.Int32, tla.AddressSpace.ub),))


def test_atomic_add_rejects_mismatched_value_type() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.atomic_add(out.ptr, tla.Int32(1))

    with pytest.raises(tla.TlaCoreAPIError, match="cast explicitly with .to\\(UInt32\\)"):
        k.dump_mlir(type_args=(_tensor(tla.UInt32),))


def test_atomic_add_rejects_out_of_range_unsigned_literal() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.atomic_add(out.ptr, 1 << 32)

    with pytest.raises(tla.TlaCoreAPIError, match="out of range for ui32"):
        k.dump_mlir(type_args=(_tensor(tla.UInt32),))


@pytest.mark.parametrize("op", ["add", "exch"])
def test_atomic_rejects_non_pointer_argument(op: str) -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            fn = tla.arch.atomic_add if op == "add" else tla.arch.atomic_exch
            fn(1, 1)

    with pytest.raises(tla.TlaCoreAPIError, match="expected pointer"):
        k.dump_mlir(type_args=(_tensor(tla.UInt32),))
