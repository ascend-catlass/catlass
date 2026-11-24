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

"""Frontend surface of tla.arch.get_ar / tla.arch.clear_ar."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

_LENGTH = 64


def _gm() -> tla.Tensor:
    return make_fake_tensor(
        tla.Int64,
        (_LENGTH,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(_LENGTH,),
        layout_tag=tla.arch.RowMajor,
    )


# --- emitted IR ------------------------------------------------------------


def test_get_ar_emits_its_op_and_an_int64() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            out[0] = tla.arch.get_ar()

    mlir = k.dump_mlir(type_args=(_gm(),))
    assert "tla.arch.get_ar" in mlir
    assert "-> i64" in mlir


def test_clear_ar_emits_its_op_in_a_simd_vec_func() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simd"):
                tla.arch.clear_ar()

    assert "tla.arch.clear_ar" in k.dump_mlir(type_args=(_gm(),))


def test_clear_ar_emits_its_op_in_the_scalar_region() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.clear_ar()

    assert "tla.arch.clear_ar" in k.dump_mlir(type_args=(_gm(),))


def test_repeated_ar_access_is_not_folded_away() -> None:
    """Two reads of AR are two observations, and two clears are two writes.

    Folding either pair would report a count from before the second operation.
    """

    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.clear_ar()
            out[0] = tla.arch.get_ar()
            tla.arch.clear_ar()
            out[1] = tla.arch.get_ar()

    mlir = k.dump_mlir(type_args=(_gm(),))
    assert mlir.count("tla.arch.clear_ar") == 2
    assert mlir.count("tla.arch.get_ar") == 2


# --- placement -------------------------------------------------------------


def test_get_ar_is_rejected_inside_a_simd_vec_func() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simd"):
                tla.arch.get_ar()

    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        k.dump_mlir(type_args=(_gm(),))


def test_get_ar_is_rejected_inside_a_simt_vec_func() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.get_ar()

    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        k.dump_mlir(type_args=(_gm(),))


def test_clear_ar_is_rejected_inside_a_simt_vec_func() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.clear_ar()

    with pytest.raises(tla.TlaCoreAPIError, match="mode='simt'"):
        k.dump_mlir(type_args=(_gm(),))


def test_get_ar_is_rejected_on_the_cube_core() -> None:
    """AR belongs to the vector unit, so the AIC has none to name."""

    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.cube():
            out[0] = tla.arch.get_ar()

    with pytest.raises(tla.TlaCoreAPIError, match="unavailable on the AIC"):
        k.dump_mlir(type_args=(_gm(),))


def test_clear_ar_is_rejected_on_the_cube_core() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.cube():
            tla.arch.clear_ar()

    with pytest.raises(tla.TlaCoreAPIError, match="unavailable on the AIC"):
        k.dump_mlir(type_args=(_gm(),))


def test_get_ar_is_rejected_outside_both_core_regions() -> None:
    """A kernel body outside both regions is duplicated to AIC and AIV."""

    @tla.kernel
    def k(out: tla.Tensor) -> None:
        out[0] = tla.arch.get_ar()

    with pytest.raises(tla.TlaCoreAPIError, match="must be nested inside tla.vector\\(\\)"):
        k.dump_mlir(type_args=(_gm(),))


def test_clear_ar_is_rejected_outside_both_core_regions() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        tla.arch.clear_ar()

    with pytest.raises(tla.TlaCoreAPIError, match="must be nested inside tla.vector\\(\\)"):
        k.dump_mlir(type_args=(_gm(),))


@pytest.mark.parametrize("name", ["get_ar", "clear_ar"])
def test_ar_access_requires_a_kernel(name: str) -> None:
    with pytest.raises(tla.TlaIRNotExecutableError):
        getattr(tla.arch, name)()
