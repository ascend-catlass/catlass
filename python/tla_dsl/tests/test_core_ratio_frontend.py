from __future__ import annotations

import pytest

import catlass.tla as tla


@tla.kernel
def default_mix_kernel() -> None:
    with tla.cube():
        pass
    with tla.vector():
        pass


@tla.kernel(core_ratio=tla.arch.C1V2)
def explicit_12_kernel() -> None:
    with tla.cube():
        pass
    with tla.vector():
        pass


@tla.kernel(core_ratio=tla.arch.C1V1)
def explicit_11_kernel() -> None:
    with tla.cube():
        pass
    with tla.vector():
        pass


def test_core_ratio_defaults_absent_in_options() -> None:
    assert default_mix_kernel.options == {}


def test_core_ratio_c1v2_propagates_to_tla_func() -> None:
    assert explicit_12_kernel.options == {"core_ratio": (1, 2)}
    mlir = explicit_12_kernel.dump_mlir()
    assert "tla.core_ratio" in mlir
    assert "array<i64: 1, 2>" in mlir


def test_core_ratio_c1v1_propagates_to_tla_func() -> None:
    assert explicit_11_kernel.options == {"core_ratio": (1, 1)}
    mlir = explicit_11_kernel.dump_mlir()
    assert "array<i64: 1, 1>" in mlir


@pytest.mark.parametrize(
    "value",
    [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        "1:1",
        "1:2",
        "2:1",
        12,
        (1,),
        (1, 1, 1),
    ],
)
def test_core_ratio_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="core_ratio must be"):
        tla.kernel(core_ratio=value)  # type: ignore[arg-type]


def test_core_ratio_rejects_non_mix_kernel() -> None:
    @tla.kernel(core_ratio=tla.arch.C1V2)
    def aiv_only() -> None:
        with tla.vector():
            pass

    with pytest.raises(Exception, match="core_ratio is only valid on MIX"):
        aiv_only.dump_mlir()


def test_core_ratio_c1v1_rejects_aiv_id_1() -> None:
    @tla.kernel(core_ratio=tla.arch.C1V1)
    def bad_aiv() -> None:
        with tla.cube():
            flag = tla.cross_flag("done", mode=4)
            tla.cross_core_set_flag(flag, tla.arch.FIX, aiv_id=1)
        with tla.vector():
            pass

    with pytest.raises(Exception, match=r"core_ratio=tla\.arch\.C1V1|aiv_id"):
        bad_aiv.dump_mlir()


def test_core_ratio_c1v1_allows_aiv_id_0() -> None:
    @tla.kernel(core_ratio=tla.arch.C1V1)
    def ok_aiv() -> None:
        with tla.cube():
            flag = tla.cross_flag("done", mode=4)
            tla.cross_core_set_flag(flag, tla.arch.FIX, aiv_id=0)
        with tla.vector():
            pass

    mlir = ok_aiv.dump_mlir()
    assert "array<i64: 1, 1>" in mlir
    assert "aiv_id = 0" in mlir
