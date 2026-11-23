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

"""Frontend surface of tla.arch.dcci_cacheline / dcci_cachelines / dci."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

_LENGTH = 8


def _gm() -> tla.Tensor:
    return make_fake_tensor(
        tla.Int32,
        (_LENGTH,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(_LENGTH,),
        layout_tag=tla.arch.RowMajor,
    )


def _ub() -> tla.Tensor:
    return make_fake_tensor(
        tla.Int32,
        (_LENGTH,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(_LENGTH,),
        layout_tag=tla.arch.RowMajor,
    )


# --- emitted IR ------------------------------------------------------------


def test_addressed_dcci_extracts_gm_pointer_address() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            out[0] = tla.Int32(1)
            tla.arch.dcci_cacheline(out.ptr)

    mlir = k.dump_mlir(type_args=(_gm(),))
    assert "tla.arch.dcci_cacheline" in mlir
    assert "tla.ptrtoint" in mlir
    assert "!tla.ptr<i32, gm, 4>" in mlir


def test_addressed_dcci_takes_an_offset_pointer() -> None:
    """``t.ptr + i`` is the element's address; ``t.ptr`` alone is the base."""

    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            out[3] = tla.Int32(1)
            tla.arch.dcci_cacheline(out.ptr + 3)

    mlir = k.dump_mlir(type_args=(_gm(),))
    assert "tla.ptr_add" in mlir
    assert "tla.arch.dcci_cacheline" in mlir


@pytest.mark.parametrize(
    "target, token",
    [
        (None, "all"),
        (tla.params.DcciTarget.ALL, "all"),
        (tla.params.DcciTarget.GM_L2, "gm_l2"),
        (tla.params.DcciTarget.ST_ATOMIC, "st_atomic"),
    ],
)
def test_whole_domain_dcci_carries_the_target(target, token: str) -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            out[0] = tla.Int32(1)
            if target is None:
                tla.arch.dcci_cachelines()
            else:
                tla.arch.dcci_cachelines(target)

    mlir = k.dump_mlir(type_args=(_gm(),))
    assert "tla.arch.dcci_cachelines" in mlir
    assert f"<{token}>" in mlir


def test_whole_domain_dcci_accepts_the_target_by_keyword() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            out[0] = tla.Int32(1)
            tla.arch.dcci_cachelines(target=tla.params.DcciTarget.GM_L2)

    assert "<gm_l2>" in k.dump_mlir(type_args=(_gm(),))


def test_dci_emits_its_op_in_a_cube_region() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.cube():
            out[0] = tla.Int32(1)
            tla.arch.dci()

    assert "tla.arch.dci" in k.dump_mlir(type_args=(_gm(),))


def test_repeated_maintenance_is_not_folded_away() -> None:
    """Two identical maintenance ops are two cache operations, not one."""

    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            out[0] = tla.Int32(1)
            tla.arch.dcci_cachelines(tla.params.DcciTarget.GM_L2)
            tla.arch.dcci_cachelines(tla.params.DcciTarget.GM_L2)

    assert k.dump_mlir(type_args=(_gm(),)).count("tla.arch.dcci_cachelines") == 2


# --- argument validation ---------------------------------------------------


def test_addressed_dcci_rejects_a_ub_pointer() -> None:
    @tla.kernel
    def k(out: tla.Tensor, scratch: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.dcci_cacheline(scratch.ptr)

    with pytest.raises(tla.TlaCoreAPIError, match="expected a gm pointer, got ub"):
        k.dump_mlir(type_args=(_gm(), _ub()))


def test_addressed_dcci_rejects_a_tensor() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.dcci_cacheline(out)

    with pytest.raises(tla.TlaCoreAPIError, match="expected pointer"):
        k.dump_mlir(type_args=(_gm(),))


def test_addressed_dcci_rejects_a_plain_integer_address() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.dcci_cacheline(4096)

    with pytest.raises(tla.TlaCoreAPIError, match="expected pointer"):
        k.dump_mlir(type_args=(_gm(),))


def test_whole_domain_dcci_rejects_the_ub_domain() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.dcci_cachelines(tla.params.DcciTarget.UB)

    with pytest.raises(tla.TlaCoreAPIError, match="DcciTarget.UB is not available"):
        k.dump_mlir(type_args=(_gm(),))


def test_whole_domain_dcci_rejects_a_non_enum_target() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.dcci_cachelines(2)

    with pytest.raises(tla.TlaCoreAPIError, match="expected tla.params.DcciTarget"):
        k.dump_mlir(type_args=(_gm(),))


# --- placement -------------------------------------------------------------


def test_maintenance_is_rejected_inside_a_simd_vec_func() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simd"):
                tla.arch.dci()

    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        k.dump_mlir(type_args=(_gm(),))


def test_maintenance_is_rejected_inside_a_simt_vec_func() -> None:
    @tla.kernel
    def k(out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.dci()

    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        k.dump_mlir(type_args=(_gm(),))


def test_maintenance_is_rejected_outside_both_core_regions() -> None:
    """A kernel body outside both regions is duplicated to AIC and AIV."""

    @tla.kernel
    def k(out: tla.Tensor) -> None:
        tla.arch.dcci_cachelines()

    with pytest.raises(
        tla.TlaCoreAPIError, match="must be nested inside tla.cube\\(\\) or tla.vector\\(\\)"
    ):
        k.dump_mlir(type_args=(_gm(),))


def test_maintenance_requires_a_kernel() -> None:
    with pytest.raises(tla.TlaIRNotExecutableError):
        tla.arch.dci()


# --- the public enum -------------------------------------------------------


def test_dcci_target_values_are_the_native_selector_codes() -> None:
    assert int(tla.params.DcciTarget.ALL) == 0
    assert int(tla.params.DcciTarget.UB) == 1
    assert int(tla.params.DcciTarget.GM_L2) == 2
    assert int(tla.params.DcciTarget.ST_ATOMIC) == 3
    assert str(tla.params.DcciTarget.GM_L2) == "GM_L2"
    assert tla.params.DcciTarget.ST_ATOMIC.mlir_token == "st_atomic"
