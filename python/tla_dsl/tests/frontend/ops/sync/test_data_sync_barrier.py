from __future__ import annotations

import pytest

import catlass.tla as tla


@tla.kernel
def data_sync_barrier_vector_kernel() -> None:
    with tla.vector():
        tla.data_sync_barrier(tla.params.DsbTarget.GM_UB)
        tla.data_sync_barrier(tla.params.DsbTarget.GM)
        tla.data_sync_barrier(tla.params.DsbTarget.UB)
        tla.data_sync_barrier(tla.params.DsbTarget.SEQ)


@tla.kernel
def data_sync_barrier_cube_kernel() -> None:
    with tla.cube():
        tla.data_sync_barrier(tla.params.DsbTarget.GM_UB)
        tla.data_sync_barrier(tla.params.DsbTarget.GM)
        tla.data_sync_barrier(tla.params.DsbTarget.SEQ)


@tla.kernel
def data_sync_barrier_kernel_body_kernel() -> None:
    # Plain scalar kernel body: no cube or vector region.
    tla.data_sync_barrier(tla.params.DsbTarget.GM)


@tla.kernel
def data_sync_barrier_in_vec_func_kernel() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.data_sync_barrier(tla.params.DsbTarget.GM)


@tla.kernel
def data_sync_barrier_ub_in_cube_kernel() -> None:
    with tla.cube():
        tla.data_sync_barrier(tla.params.DsbTarget.UB)


def test_data_sync_barrier_emits_tla_op_with_each_domain() -> None:
    mlir = data_sync_barrier_vector_kernel.dump_mlir()
    assert "tla.data_sync_barrier" in mlir
    assert "tla.data_sync_barrier <gm_ub>" in mlir
    assert "tla.data_sync_barrier <gm>" in mlir
    assert "tla.data_sync_barrier <ub>" in mlir
    assert "tla.data_sync_barrier <seq>" in mlir


def test_data_sync_barrier_allows_cube_and_kernel_body() -> None:
    cube_mlir = data_sync_barrier_cube_kernel.dump_mlir()
    assert "tla.data_sync_barrier <gm_ub>" in cube_mlir
    assert "tla.data_sync_barrier <gm>" in cube_mlir
    assert "tla.data_sync_barrier <seq>" in cube_mlir

    body_mlir = data_sync_barrier_kernel_body_kernel.dump_mlir()
    assert "tla.data_sync_barrier <gm>" in body_mlir


def test_data_sync_barrier_rejects_vec_func() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="must not be called inside tla.vec.func",
    ):
        data_sync_barrier_in_vec_func_kernel.dump_mlir()


def test_data_sync_barrier_rejects_ub_in_cube() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="DsbTarget.UB is not issued on AIC",
    ):
        data_sync_barrier_ub_in_cube_kernel.dump_mlir()


def test_data_sync_barrier_rejects_invalid_domain() -> None:
    @tla.kernel
    def invalid_domain_kernel() -> None:
        with tla.vector():
            tla.data_sync_barrier(1)

    with pytest.raises(
        tla.TlaCoreAPIError,
        match="invalid argument 'domain'",
    ):
        invalid_domain_kernel.dump_mlir()
