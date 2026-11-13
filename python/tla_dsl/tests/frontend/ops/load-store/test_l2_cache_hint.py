from __future__ import annotations

import pathlib
import re
import subprocess
from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError
from catlass.tla.runtime import make_fake_tensor


def _require_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    tla_compile = (
        repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    )
    if not tla_compile.exists():
        pytest.skip("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    extent: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


def _gm_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    shape: tuple[int, ...] = (64,),
) -> tla.Tensor:
    if len(shape) == 1:
        stride: tuple[int, ...] = (1,)
    elif len(shape) == 2:
        stride = (shape[1], 1)
    else:
        raise ValueError(f"unsupported rank for test helper: {shape!r}")
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def gm2ub_with_disable_hint(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            ub_tensor,
            gm_tensor,
            cache_policy=tla.params.CachePolicy.DISABLE,
        )


@tla.kernel
def gm2ub_default_normal(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(ub_tensor, gm_tensor)


@tla.kernel
def ub2gm_with_disable_hint(ub_tensor: tla.Tensor, gm_tensor: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            gm_tensor,
            ub_tensor,
            cache_policy=tla.params.CachePolicy.DISABLE,
        )


@tla.kernel
def copy_cache_policy_on_ubub_rejected(ub_src: tla.Tensor, ub_dst: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(ub_dst, ub_src, cache_policy=tla.params.CachePolicy.DISABLE)


def test_gm2ub_emits_src_cache_policy_disable(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        gm2ub_with_disable_hint, type_args=(_gm_tensor(), _ub_tensor())
    )
    assert "tla.copy" in mlir
    assert "src_cache_policy = #tla.cache_policy<disable>" in mlir
    assert "dst_cache_policy" not in mlir


def test_gm2ub_default_omits_cache_policy_attr(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(gm2ub_default_normal, type_args=(_gm_tensor(), _ub_tensor()))
    assert "tla.copy" in mlir
    assert "cache_policy" not in mlir


def test_ub2gm_emits_dst_cache_policy_disable(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        ub2gm_with_disable_hint, type_args=(_ub_tensor(), _gm_tensor())
    )
    assert "tla.copy" in mlir
    assert "dst_cache_policy = #tla.cache_policy<disable>" in mlir
    assert "src_cache_policy" not in mlir


def test_copy_cache_policy_rejects_non_gm_route() -> None:
    with pytest.raises(TlaLoweringError, match="requires a GM"):
        copy_cache_policy_on_ubub_rejected.dump_mlir(
            type_args=(_ub_tensor(), _ub_tensor())
        )


@tla.kernel
def gm2ub_disable_alloc(gm_tensor: tla.Tensor) -> None:
    """Kernel-arg GM only (UB allocated inside) so TlaCompile accepts it."""
    ub_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor_like(ub_ptr, gm_tensor, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(ub, gm_tensor, cache_policy=tla.params.CachePolicy.DISABLE)


@tla.kernel
def gm2ub_normal_alloc(gm_tensor: tla.Tensor) -> None:
    ub_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor_like(ub_ptr, gm_tensor, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(ub, gm_tensor)


@tla.kernel
def gm2ub_fp4_disable_alloc(gm_tensor: tla.Tensor) -> None:
    """FP4 GM→UB must forward gmCachePolicy through the FP4 BC wrapper."""
    ub_ptr = tla.allocate(64, tla.Float4E2M1, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor_like(ub_ptr, gm_tensor, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(ub, gm_tensor, cache_policy=tla.params.CachePolicy.DISABLE)


def _last_call_cache_policy_const(lowered: str, *, callee: str) -> int:
    match = re.search(rf"call @{re.escape(callee)}\(([^)]*)\)", lowered)
    assert match is not None, lowered
    operands = [op.strip() for op in match.group(1).split(",")]
    assert len(operands) == 27  # 2 memref + 24 desc + gmCachePolicy
    last = operands[-1]
    const = re.search(
        rf"{re.escape(last)}\s*=\s*llvm\.mlir\.constant\((-?\d+)\s*:\s*i64\)",
        lowered,
    )
    assert const is not None, (last, lowered)
    return int(const.group(1))


def test_lowering_appends_disable_cache_policy(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm2ub_disable_alloc.dump_mlir(type_args=(_gm_tensor(),))
    assert "src_cache_policy = #tla.cache_policy<disable>" in mlir
    path = tmp_path / "gm2ub_disable.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert (
        _last_call_cache_policy_const(
            result.stdout, callee="copy_gm_RowMajor_to_ub_RowMajor_float"
        )
        == 0
    )


def test_lowering_appends_normal_cache_policy_by_default(tmp_path: pathlib.Path) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm2ub_normal_alloc.dump_mlir(type_args=(_gm_tensor(),))
    assert "cache_policy" not in mlir
    path = tmp_path / "gm2ub_normal.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert (
        _last_call_cache_policy_const(
            result.stdout, callee="copy_gm_RowMajor_to_ub_RowMajor_float"
        )
        == 1
    )


def test_fp4_gm2ub_lowering_forwards_disable_cache_policy(
    tmp_path: pathlib.Path,
) -> None:
    tla_compile = _require_tla_compile()
    mlir = gm2ub_fp4_disable_alloc.dump_mlir(
        type_args=(_gm_tensor(dtype=tla.Float4E2M1),)
    )
    assert "src_cache_policy = #tla.cache_policy<disable>" in mlir
    path = tmp_path / "gm2ub_fp4_disable.mlir"
    path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(path), "-o", "-"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert (
        _last_call_cache_policy_const(
            result.stdout,
            callee="copy_gm_RowMajor_to_ub_RowMajor_float4_e2m1x2_t",
        )
        == 0
    )
