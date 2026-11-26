from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.core_api import TlaCoreAPIError
from catlass.execution_lowering import TlaLoweringError


@tla.kernel
def sync_all_aiv_kernel() -> None:
    with tla.vector():
        tla.sync_all_aiv()
        tla.sync_all_aiv("mte3", "all")


@tla.kernel
def sync_all_aiv_mte2_kernel() -> None:
    with tla.vector():
        tla.sync_all_aiv("mte2", "mte3")


@tla.kernel
def sync_all_aiv_token_pipe_kernel() -> None:
    with tla.vector():
        tla.sync_all_aiv(tla.pipes.MTE3, tla.pipes.ALL)


@tla.kernel(core_ratio=tla.arch.C1V2)
def sync_all_mix_c1v2_kernel() -> None:
    with tla.cube():
        tla.sync_all()
    with tla.vector():
        tla.sync_all()


@tla.kernel(core_ratio=tla.arch.C1V1)
def sync_all_mix_c1v1_kernel() -> None:
    with tla.cube():
        tla.sync_all()
    with tla.vector():
        tla.sync_all()


@tla.kernel
def sync_all_aiv_outside_vector_kernel() -> None:
    tla.sync_all_aiv()


@tla.kernel
def sync_all_without_core_ratio_kernel() -> None:
    with tla.vector():
        tla.sync_all()


@tla.kernel(core_ratio=tla.arch.C1V2)
def sync_all_outside_regions_kernel() -> None:
    with tla.cube():
        pass
    with tla.vector():
        pass
    tla.sync_all()


@tla.kernel
def reserved_flag_kernel() -> None:
    with tla.vector():
        tla.cross_flag("__tla_sync_all_inter_core_aiv_ready__")


@tla.kernel
def sync_all_aiv_bad_pipe_kernel() -> None:
    with tla.vector():
        tla.sync_all_aiv("fix")


def test_sync_all_aiv_emits_single_reused_flag() -> None:
    mlir = sync_all_aiv_kernel.dump_mlir()

    assert mlir.count('tla.cross_flag "__tla_sync_all_inter_core_aiv_ready__" -> !tla.cross_flag<0>') == 1
    assert mlir.count("tla.cross_core_set_flag") == 2
    assert mlir.count("tla.cross_core_wait_flag") == 2
    assert mlir.count("tla.pipe_barrier[<all>]") == 1
    assert mlir.count("tla.pipe_barrier[<mte3>]") == 1
    assert mlir.count("#tla.pipe<mte3>} : !tla.cross_flag<0>") == 2
    assert mlir.count("#tla.pipe<scalar>} : !tla.cross_flag<0>") == 2


def test_sync_all_aiv_honours_explicit_pipes() -> None:
    mlir = sync_all_aiv_mte2_kernel.dump_mlir()

    assert "tla.pipe_barrier[<mte2>]" in mlir
    assert "#tla.pipe<mte2>} : !tla.cross_flag<0>" in mlir
    assert "#tla.pipe<mte3>} : !tla.cross_flag<0>" in mlir


def test_sync_all_aiv_accepts_pipe_tokens() -> None:
    mlir = sync_all_aiv_token_pipe_kernel.dump_mlir()

    assert "tla.pipe_barrier[<mte3>]" in mlir
    assert "#tla.pipe<mte3>} : !tla.cross_flag<0>" in mlir
    # `tla.pipes.ALL` on the wait side folds to the scalar pipe.
    assert "#tla.pipe<scalar>} : !tla.cross_flag<0>" in mlir


def test_sync_all_mix_c1v2_shares_flags_and_releases_on_scalar() -> None:
    mlir = sync_all_mix_c1v2_kernel.dump_mlir()

    assert mlir.count('tla.cross_flag "__tla_sync_all_intra_block_aiv_ready__" -> !tla.cross_flag<2>') == 1
    assert mlir.count('tla.cross_flag "__tla_sync_all_intra_block_aic_ready__" -> !tla.cross_flag<2>') == 1
    assert mlir.count('tla.cross_flag "__tla_sync_all_inter_core_aic_ready__" -> !tla.cross_flag<0>') == 1
    # AIC: wait AIV arrival, fire FFTS, wait for it, release the AIVs.
    assert "#tla.pipe<fix>} : !tla.cross_flag<0>" in mlir
    assert "#tla.pipe<scalar>} : !tla.cross_flag<2>" in mlir
    assert "#tla.pipe<mte3>} : !tla.cross_flag<2>" in mlir
    # C1V2: the cube releases the vector cores from the scalar pipe.
    assert mlir.count("tla.cube {") == 1
    assert mlir.count('"tla.vector"() ({') == 1


def test_sync_all_mix_c1v1_releases_from_mte3() -> None:
    mlir = sync_all_mix_c1v1_kernel.dump_mlir()

    assert "#tla.pipe<mte3>} : !tla.cross_flag<2>" in mlir
    assert "#tla.pipe<scalar>} : !tla.cross_flag<2>" in mlir


def test_sync_all_aiv_requires_vector_region() -> None:
    with pytest.raises(TlaLoweringError) as excinfo:
        sync_all_aiv_outside_vector_kernel.dump_mlir()
    assert "must be called inside tla.vector()" in str(excinfo.value)


def test_sync_all_requires_core_ratio() -> None:
    with pytest.raises(TlaLoweringError) as excinfo:
        sync_all_without_core_ratio_kernel.dump_mlir()
    assert "declare @tla.kernel(core_ratio=...)" in str(excinfo.value)


def test_sync_all_requires_cube_or_vector_region() -> None:
    with pytest.raises(TlaLoweringError) as excinfo:
        sync_all_outside_regions_kernel.dump_mlir()
    assert "must be called inside tla.cube() or tla.vector()" in str(excinfo.value)


def test_reserved_sync_all_flag_name_is_rejected() -> None:
    with pytest.raises(TlaCoreAPIError) as excinfo:
        reserved_flag_kernel.dump_mlir()
    assert "reserved" in str(excinfo.value)


def test_sync_all_aiv_rejects_unsupported_pipe() -> None:
    with pytest.raises(TlaCoreAPIError) as excinfo:
        sync_all_aiv_bad_pipe_kernel.dump_mlir()
    assert "invalid argument 'trigger_pipe'" in str(excinfo.value)
