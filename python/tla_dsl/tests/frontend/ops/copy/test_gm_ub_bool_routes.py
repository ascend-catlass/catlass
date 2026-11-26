# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend acceptance for GM↔UB bool (i1) two-arg DataCopy routes (T25)."""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any

import pytest

from catlass import tla
from catlass.runtime import TlaCoreAPIError
from catlass.tla.runtime import make_fake_tensor

_ROWS = 8
_COLS = 32


def _require_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    tla_compile = (
        repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    )
    if not tla_compile.exists():
        pytest.skip("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


def _gm_rm(dtype=tla.Bool, shape=(_ROWS, _COLS)):
    return make_fake_tensor(
        dtype,
        shape,
        (shape[1], 1),
        addrspace=tla.AddressSpace.gm,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


def _ub_rm(dtype=tla.Bool, shape=(_ROWS, _COLS)):
    return make_fake_tensor(
        dtype,
        shape,
        (shape[1], 1),
        addrspace=tla.AddressSpace.ub,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


def _vec(dtype=tla.Bool, n=_COLS, addrspace=tla.AddressSpace.gm):
    return make_fake_tensor(
        dtype,
        (n,),
        (1,),
        addrspace=addrspace,
        origin_shape=(n,),
        layout_tag=tla.arch.VectorLayout,
    )


@tla.kernel
def gm_to_ub_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(dst, src)


@tla.kernel
def ub_to_gm_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(dst, src)


@tla.kernel
def gm_to_ub_alloc(gm_src: tla.Tensor) -> None:
    """Kernel-arg GM only so TlaCompile accepts the module."""
    ub_ptr = tla.allocate(256, tla.Bool, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(_ROWS, _COLS),
            tla.make_stride(_COLS, 1),
            origin_shape=tla.make_shape(_ROWS, _COLS),
            layoutTag=tla.arch.RowMajor,
        ),
        tla.make_coord(0, 0),
    )
    with tla.vector():
        tla.copy(ub, gm_src)


def test_gm_rowmajor_to_ub_accepts_bool(compiler_tlair: Any) -> None:
    mlir = gm_to_ub_kernel.dump_mlir(type_args=(_ub_rm(), _gm_rm()))
    assert "tla.copy" in mlir


def test_ub_rowmajor_to_gm_accepts_bool(compiler_tlair: Any) -> None:
    mlir = ub_to_gm_kernel.dump_mlir(type_args=(_gm_rm(), _ub_rm()))
    assert "tla.copy" in mlir


def test_gm_vector_to_ub_accepts_bool(compiler_tlair: Any) -> None:
    mlir = gm_to_ub_kernel.dump_mlir(
        type_args=(
            _vec(addrspace=tla.AddressSpace.ub),
            _vec(addrspace=tla.AddressSpace.gm),
        )
    )
    assert "tla.copy" in mlir


def test_gm_to_ub_bool_accepts_copy_pad_params(compiler_tlair: Any) -> None:
    """Bool element + CopyPadParams is in scope (pad counts still reject bool)."""
    # Multi-row pad needs AlignUp(src+left+right, 32) dst stride (1B elem → 64).
    mlir = gm_to_ub_with_pad.dump_mlir(
        type_args=(
            _ub_rm(shape=(_ROWS, 64)),
            _gm_rm(shape=(_ROWS, _COLS)),
        )
    )
    assert "tla.CopyPadParams" in mlir
    assert "!tla.copy_pad_params" in mlir
    # False → paddingValueBits 0
    assert "arith.constant 0 : i64" in mlir


def test_gm_to_ub_bool_pad_true_encodes_one(compiler_tlair: Any) -> None:
    mlir = gm_to_ub_with_pad_true.dump_mlir(
        type_args=(
            _ub_rm(shape=(_ROWS, 64)),
            _gm_rm(shape=(_ROWS, _COLS)),
        )
    )
    assert "tla.CopyPadParams" in mlir
    assert "arith.constant 1 : i64" in mlir


def test_gm_to_ub_bool_pad_rejects_int_fill(compiler_tlair: Any) -> None:
    with pytest.raises(TlaCoreAPIError, match="bool/tla.Bool|i1"):
        gm_to_ub_with_pad_int_fill_rejected.dump_mlir(
            type_args=(
                _ub_rm(shape=(_ROWS, 64)),
                _gm_rm(shape=(_ROWS, _COLS)),
            ),
        )


@tla.kernel
def gm_to_ub_with_pad(dst: tla.Tensor, src: tla.Tensor) -> None:
    from catlass.params import CopyPadParams

    with tla.vector():
        # i1 tiles require bool/tla.Bool fill (bare int is rejected).
        tla.copy(dst, src, params=CopyPadParams(right_padding=1, padding_value=False))


@tla.kernel
def gm_to_ub_with_pad_true(dst: tla.Tensor, src: tla.Tensor) -> None:
    from catlass.params import CopyPadParams

    with tla.vector():
        tla.copy(dst, src, params=CopyPadParams(right_padding=1, padding_value=True))


@tla.kernel
def gm_to_ub_with_pad_int_fill_rejected(dst: tla.Tensor, src: tla.Tensor) -> None:
    from catlass.params import CopyPadParams

    with tla.vector():
        # Non-zero bare int is rejected on i1 (0 remains the dataclass default).
        tla.copy(dst, src, params=CopyPadParams(right_padding=1, padding_value=1))


def test_lowering_gm_to_ub_bool_callee(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm_to_ub_alloc.dump_mlir(type_args=(_gm_rm(),))
    path = tmp_path / "gm2ub_bool.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "copy_gm_RowMajor_to_ub_RowMajor_bool" in result.stdout
    # 256B UB bool allocation is 256 elements (1B/elem), not 2048xi1 bit packing.
    assert "memref<256xi1" in result.stdout
    assert "memref<2048xi1" not in result.stdout


@tla.kernel
def gm_to_ub_bool_ptr_offset(gm_src: tla.Tensor) -> None:
    """Bool UB view starts mid-buffer via ptr_add (byte-addressed i1)."""
    ub_base = tla.allocate(512, tla.Bool, tla.AddressSpace.ub, 256)
    ub_ptr = ub_base + 256
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(1, 32),
            tla.make_stride(32, 1),
            origin_shape=tla.make_shape(1, 32),
            layoutTag=tla.arch.RowMajor,
        ),
        tla.make_coord(0, 0),
    )
    with tla.vector():
        tla.copy(ub, gm_src)


def test_lowering_gm_to_ub_bool_ptr_offset(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    gm = make_fake_tensor(
        tla.Bool,
        (1, 32),
        (32, 1),
        origin_shape=(1, 32),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = gm_to_ub_bool_ptr_offset.dump_mlir(type_args=(gm,))
    path = tmp_path / "gm2ub_bool_off.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "copy_gm_RowMajor_to_ub_RowMajor_bool" in result.stdout
