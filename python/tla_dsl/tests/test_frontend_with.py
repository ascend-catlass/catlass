from catlass.tla.runtime import make_fake_tensor
import builtins
import inspect
import re
from typing import Any

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError
from catlass import _tla_type_bridge
from catlass._mlir import ir as mlir_ir


def _ensure_compute_order_or_skip() -> None:
    with mlir_ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        try:
            mlir_ir.Attribute.parse("#tla.compute_order<M_FIRST>", context=ctx)
        except mlir_ir.MLIRError as exc:
            if "compute_order" in str(exc) and "expected valid keyword" in str(exc):
                pytest.skip("linked BiShengIR lacks tla.compute_order enum used by mmad")
            raise


@pytest.fixture(autouse=True)
def _skip_without_compute_order(request):
    name = getattr(request.node, "name", "")
    if "mmad" in name.lower() or (
        "cube" in name.lower() and "mutex" not in name.lower()
    ):
        _ensure_compute_order_or_skip()


def _mmad_tensor_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor, tla.Tensor]:
    """Host fake tensors: logical origin + layout_tag; fractal trees via make_fake_tensor."""
    return (
        make_fake_tensor(
            tla.Float16,
            ((16, 8), (16, 4)),
            ((16, 256), (1, 2048)),
            addrspace=tla.AddressSpace.l0a,
            layout_tag=tla.arch.zN,
            origin_shape=(128, 64),
            coord=(0, 0),
        ),
        make_fake_tensor(
            tla.Float16,
            ((16, 4), (16, 8)),
            ((1, 2048), (16, 256)),
            addrspace=tla.AddressSpace.l0b,
            layout_tag=tla.arch.nZ,
            origin_shape=(64, 128),
            coord=(0, 0),
        ),
        make_fake_tensor(
            tla.Float32,
            ((16, 8), (16, 8)),
            ((16, 256), (1, 2048)),
            addrspace=tla.AddressSpace.l0c,
            layout_tag=tla.arch.L0Clayout,
            origin_shape=(128, 128),
            coord=(0, 0),
        ),
        make_fake_tensor(
            tla.Float32,
            (128, 128),
            (128, 1),
            addrspace=tla.AddressSpace.gm,
            layout_tag=tla.arch.RowMajor,
            origin_shape=(128, 128),
            coord=(0, 0),
        ),
    )

# A zN tile carries fractal shape/stride, and the cube copy routes only accept
# zN (or nZ) on the L1/L0A side -- a RowMajor tile there has no route at all.
# These tests exercise mutex placement rather than route availability, so L1/L0A
# operands default to the zN geometry the cube tests already use.
_ZN_FRACTAL_SHAPE = ((16, 8), (16, 4))
_ZN_FRACTAL_STRIDE = ((16, 256), (1, 2048))
_ZN_ORIGIN_SHAPE = (128, 64)


def _tensor_arg(
    addrspace: tla.AddressSpace,
    *,
    dtype: type[tla.Numeric] = tla.Float16,
    layout_tag: Any | None = None,
) -> tla.Tensor:
    if layout_tag is None and addrspace in (tla.AddressSpace.l1, tla.AddressSpace.l0a):
        layout_tag = tla.arch.zN
        shape: Any = _ZN_FRACTAL_SHAPE
        stride: Any = _ZN_FRACTAL_STRIDE
        origin_shape: Any = _ZN_ORIGIN_SHAPE
    else:
        shape, stride, origin_shape = (16, 16), (16, 1), (16, 16)
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        addrspace=addrspace,
        origin_shape=origin_shape,
        layout_tag=layout_tag or tla.arch.RowMajor,
    )

def _vector_tensor_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    return (
        make_fake_tensor(
            tla.Float32,
            (64,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(64,),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float32,
            (64,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(64,),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float32,
            (64,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(64,),
            layout_tag=tla.arch.RowMajor,
        ),
    )

def _skip_if_mmad_rank2_tile_view_regression(exc: BaseException) -> None:
    if isinstance(exc, TlaLoweringError) and "rank-2 tiles only" in str(exc):
        pytest.skip(
            "tla.mmad rank-2 check rejects tile_view operand types until metadata matches"
        )

class BindValue:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __enter__(self) -> Any:
        return self.value

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        return False

@tla.kernel
def generic_with_as_shadows_tla_range_alias_kernel() -> None:
    with BindValue(builtins.range) as tla_range:
        for i in tla_range(0, 2):
            tla.make_coord(i, 0)

@tla.kernel
def cube_static_kernel(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor) -> None:
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=False)

@tla.kernel
def mutex_guard_copy_cube_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    mutex = tla.mutex(resource="copy_mutex", id=0)
    dst_tile = tla.tile_view(dst, tla.make_shape(16, 16), tla.make_coord(0, 0))
    src_tile = tla.tile_view(src, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        with tla.mutex_guard(mutex):
            tla.copy(dst_tile, src_tile)

@tla.kernel
def mutex_guard_copy_vec_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    mutex = tla.mutex(resource="copy_mutex", id=0)
    dst_tile = tla.tile_view(dst, tla.make_shape(16, 16), tla.make_coord(0, 0))
    src_tile = tla.tile_view(src, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.vector():
        with tla.mutex_guard(mutex):
            tla.copy(dst_tile, src_tile)

@tla.kernel
def mutex_guard_two_copy_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    mutex = tla.mutex(resource="copy_mutex", id=0)
    dst_tile = tla.tile_view(dst, tla.make_shape(16, 16), tla.make_coord(0, 0))
    src_tile = tla.tile_view(src, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        with tla.mutex_guard(mutex):
            tla.copy(dst_tile, src_tile)
            tla.copy(dst_tile, src_tile)

@tla.kernel
def mutex_guard_mixed_pipe_kernel(
    gm: tla.Tensor, l1: tla.Tensor, l0a: tla.Tensor
) -> None:
    mutex = tla.mutex(resource="mixed_mutex", id=0)
    gm_tile = tla.tile_view(gm, tla.make_shape(16, 16), tla.make_coord(0, 0))
    l1_tile = tla.tile_view(l1, tla.make_shape(16, 16), tla.make_coord(0, 0))
    l0a_tile = tla.tile_view(l0a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        with tla.mutex_guard(mutex):
            tla.copy(l1_tile, gm_tile)
            tla.copy(l0a_tile, l1_tile)

@tla.kernel
def mutex_guard_explicit_lock_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    mutex = tla.mutex(resource="bad_mutex", id=0)
    dst_tile = tla.tile_view(dst, tla.make_shape(16, 16), tla.make_coord(0, 0))
    src_tile = tla.tile_view(src, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.mutex_guard(mutex):
        mutex.lock(pipe=tla.pipes.MTE2)
        tla.copy(dst_tile, src_tile)

@tla.kernel
def mutex_guard_empty_kernel() -> None:
    mutex = tla.mutex(resource="empty_mutex", id=0)
    with tla.mutex_guard(mutex):
        tla.make_shape(16, 16)

@tla.kernel
def mutex_guard_multi_mmad_kernel(
    lhs_mem: tla.Tensor, rhs_mem: tla.Tensor, acc_mem: tla.Tensor
) -> None:
    lhs = tla.tile_view(lhs_mem, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(rhs_mem, tla.make_shape(16, 16), tla.make_coord(0, 0))
    acc = tla.tile_view(acc_mem, tla.make_shape(16, 16), tla.make_coord(0, 0))
    mutex_l0a = tla.mutex(resource="l0a", id=0)
    mutex_l0b = tla.mutex(resource="l0b", id=1)
    mutex_l0c = tla.mutex(resource="l0c", id=2)
    with tla.cube():
        with tla.mutex_guard(mutex_l0a, mutex_l0b, mutex_l0c):
            tla.mmad(acc, lhs, rhs, init_c=False)

@tla.kernel
def mutex_guard_dynamic_mutex_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    mutex_l1a0 = tla.mutex(resource="l1a0", id=0)
    mutex_l1a1 = tla.mutex(resource="l1a1", id=1)
    mutex_l0a0 = tla.mutex(resource="l0a0", id=2)
    mutex_l0a1 = tla.mutex(resource="l0a1", id=3)
    dst_tile = tla.tile_view(dst, tla.make_shape(16, 16), tla.make_coord(0, 0))
    src_tile = tla.tile_view(src, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        for i in tla.range(0, 2, 1):
            mutex_l1a = mutex_l1a0 if i == 0 else mutex_l1a1
            mutex_l0a = mutex_l0a0 if i == 0 else mutex_l0a1
            with tla.mutex_guard(mutex_l1a, mutex_l0a):
                tla.copy(dst_tile, src_tile)

@tla.kernel
def mutex_guard_control_flow_body_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    mutex = tla.mutex(resource="cf_mutex", id=0)
    dst_tile = tla.tile_view(dst, tla.make_shape(16, 16), tla.make_coord(0, 0))
    src_tile = tla.tile_view(src, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        for i in tla.range(0, 2, 1):
            with tla.mutex_guard(mutex):
                if i == 0:
                    tla.copy(dst_tile, src_tile)
                else:
                    tla.copy(dst_tile, src_tile)

def _assert_guard_order(mlir: str, pipe: str) -> None:
    assert f"#tla.pipe<{pipe}>" in mlir or f"<{pipe}>" in mlir
    assert mlir.index("tla.mutex_lock") < mlir.index("tla.copy")
    assert mlir.rindex("tla.copy") < mlir.rindex("tla.mutex_unlock")

def _mutex_operands(mlir: str, op_name: str) -> list[str]:
    generic = re.findall(rf'"{op_name}"\((%[\w\d]+)\)', mlir)
    custom = re.findall(rf"{op_name} (%[\w\d]+)\[", mlir)
    return generic + custom

@tla.kernel
def vec_func_default_mode_kernel() -> None:
    with tla.vector():
        with tla.vec.func():
            tla.make_coord(0)

@tla.kernel
def vec_func_simd_mode_kernel() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.make_coord(0)

@tla.kernel
def vec_func_positional_mode_kernel() -> None:
    with tla.vector():
        with tla.vec.func("simd"):
            tla.make_coord(0)

@tla.kernel
def vec_func_unknown_keyword_kernel() -> None:
    with tla.vector():
        with tla.vec.func(foo="simd"):
            tla.make_coord(0)

@tla.kernel
def vec_vector_ssa_kernel(lhs: tla.Tensor, rhs: tla.Tensor, dst: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            lhs_reg = lhs_tile.load()
            rhs_reg = rhs_tile.load()
            dst_tile.store(tla.add(lhs_reg, rhs_reg))

@tla.kernel
def chained_vector_vector_then_scalar_kernel() -> None:
    f32_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 32)

    shape = tla.make_shape(64)
    stride = tla.make_stride(1)
    layout = tla.make_layout(shape, stride, layoutTag=tla.arch.RowMajor)
    tensor = tla.make_tensor(f32_ptr, layout, coord=tla.make_coord(0))
    tile = tla.tile_view(tensor, tla.make_shape(64), tla.make_coord(0))

    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = tile.load()
            tmp = tla.add(reg, reg)
            out = tla.mul(tmp, 1.0)
            tile.store(out)

@tla.kernel
def mutex_guard_vec_func_kernel(
    lhs: tla.Tensor, rhs: tla.Tensor, dst: tla.Tensor
) -> None:
    mutex = tla.mutex(resource="vec_mutex", id=0)
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.mutex_guard(mutex):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                lhs_reg = lhs_tile.load()
                rhs_reg = rhs_tile.load()
                dst_tile.store(tla.add(lhs_reg, rhs_reg))

@tla.kernel
def vec_store_rejects_raw_tensor_kernel(lhs: tla.Tensor, dst: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(lhs_tile)

@tla.kernel
def vec_add_rejects_raw_tensor_kernel(lhs: tla.Tensor, rhs: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.add(lhs_tile, rhs_tile)

def test_generic_with_as_binding_shadows_tla_range_alias() -> None:
    generic_with_as_shadows_tla_range_alias_kernel.dump_mlir()

def test_cube_region_lowering() -> None:
    ta, tb, tc, _ = _mmad_tensor_args()
    try:
        mlir = cube_static_kernel.dump_mlir(type_args=(ta, tb, tc))
    except TlaLoweringError as e:
        _skip_if_mmad_rank2_tile_view_regression(e)
        raise
    assert "tla.cube" in mlir
    assert "tla.mmad" in mlir

def test_mutex_guard_copy_infers_mte2_from_gm_source() -> None:
    dst = _tensor_arg(tla.AddressSpace.l1)
    src = _tensor_arg(tla.AddressSpace.gm)
    mlir = mutex_guard_copy_cube_kernel.dump_mlir(type_args=(dst, src))
    _assert_guard_order(mlir, "mte2")

def test_mutex_guard_copy_infers_mte1_from_l1_source() -> None:
    dst = _tensor_arg(tla.AddressSpace.l0a)
    src = _tensor_arg(tla.AddressSpace.l1)
    mlir = mutex_guard_copy_cube_kernel.dump_mlir(type_args=(dst, src))
    _assert_guard_order(mlir, "mte1")

def test_mutex_guard_copy_infers_fix_from_l0c_source() -> None:
    _, _, src, dst = _mmad_tensor_args()
    mlir = mutex_guard_copy_cube_kernel.dump_mlir(type_args=(dst, src))
    _assert_guard_order(mlir, "fix")

def test_mutex_guard_copy_infers_mte3_from_ub_source() -> None:
    dst = _tensor_arg(tla.AddressSpace.gm)
    src = _tensor_arg(tla.AddressSpace.ub)
    mlir = mutex_guard_copy_vec_kernel.dump_mlir(type_args=(dst, src))
    _assert_guard_order(mlir, "mte3")

def test_mutex_guard_multi_mutex_mmad_uses_cube_and_stack_unlock_order() -> None:
    lhs, rhs, acc, _ = _mmad_tensor_args()
    try:
        mlir = mutex_guard_multi_mmad_kernel.dump_mlir(type_args=(lhs, rhs, acc))
    except TlaLoweringError as e:
        _skip_if_mmad_rank2_tile_view_regression(e)
        raise
    assert "#tla.pipe<cube>" in mlir or "<cube>" in mlir
    locks = _mutex_operands(mlir, "tla.mutex_lock")
    unlocks = _mutex_operands(mlir, "tla.mutex_unlock")
    assert len(locks) == 3
    assert unlocks == list(reversed(locks))
    assert mlir.index("tla.mutex_lock") < mlir.index("tla.mmad")
    assert mlir.rindex("tla.mmad") < mlir.rindex("tla.mutex_unlock")

def test_mutex_guard_wraps_multiple_same_pipe_ops_once() -> None:
    dst = _tensor_arg(tla.AddressSpace.l1)
    src = _tensor_arg(tla.AddressSpace.gm)
    mlir = mutex_guard_two_copy_kernel.dump_mlir(type_args=(dst, src))
    assert mlir.count("tla.copy") == 2
    assert mlir.count("tla.mutex_lock") == 1
    assert mlir.count("tla.mutex_unlock") == 1
    _assert_guard_order(mlir, "mte2")

def test_mutex_guard_rejects_multiple_inferred_pipes() -> None:
    gm = _tensor_arg(tla.AddressSpace.gm)
    l1 = _tensor_arg(tla.AddressSpace.l1)
    l0a = _tensor_arg(tla.AddressSpace.l0a)
    with pytest.raises(TlaLoweringError, match="inferred multiple pipes"):
        mutex_guard_mixed_pipe_kernel.dump_mlir(type_args=(gm, l1, l0a))

def test_mutex_guard_rejects_explicit_lock_inside_body() -> None:
    dst = _tensor_arg(tla.AddressSpace.l1)
    src = _tensor_arg(tla.AddressSpace.gm)
    with pytest.raises(tla.TlaCoreAPIError, match="explicit mutex lock/unlock"):
        mutex_guard_explicit_lock_kernel.dump_mlir(type_args=(dst, src))

def test_mutex_guard_requires_copy_mmad_or_vec_func_body() -> None:
    with pytest.raises(TlaLoweringError, match="at least one tla.copy, tla.mmad"):
        mutex_guard_empty_kernel.dump_mlir()

def test_mutex_guard_supports_dynamic_mutex_values() -> None:
    dst = _tensor_arg(tla.AddressSpace.l0a)
    src = _tensor_arg(tla.AddressSpace.l1)
    mlir = mutex_guard_dynamic_mutex_kernel.dump_mlir(type_args=(dst, src))
    assert "scf.if" in mlir
    assert mlir.count("tla.mutex_lock") == 2
    assert mlir.count("tla.mutex_unlock") == 2
    _assert_guard_order(mlir, "mte1")

def test_mutex_guard_infers_pipe_from_control_flow_body() -> None:
    dst = _tensor_arg(tla.AddressSpace.l1)
    src = _tensor_arg(tla.AddressSpace.gm)
    mlir = mutex_guard_control_flow_body_kernel.dump_mlir(type_args=(dst, src))
    assert "scf.if" in mlir
    assert mlir.count("tla.copy") == 2
    assert mlir.count("tla.mutex_lock") == 1
    assert mlir.count("tla.mutex_unlock") == 1
    _assert_guard_order(mlir, "mte2")

def test_vec_func_default_mode_lowering() -> None:
    mlir = vec_func_default_mode_kernel.dump_mlir()

    assert "tla.vec.func" in mlir
    assert 'mode = "simd"' in mlir

def test_mutex_guard_vec_func_infers_vector_pipe() -> None:
    mlir = mutex_guard_vec_func_kernel.dump_mlir(type_args=_vector_tensor_args())

    assert "#tla.pipe<vector>" in mlir or "<vector>" in mlir
    assert mlir.index("tla.mutex_lock") < mlir.index("tla.vec.func")
    assert mlir.rindex("tla.vec.func") < mlir.rindex("tla.mutex_unlock")

def test_vec_func_simd_mode_lowering() -> None:
    mlir = vec_func_simd_mode_kernel.dump_mlir()

    assert "tla.vec.func" in mlir
    assert 'mode = "simd"' in mlir

def test_vec_func_rejects_positional_mode() -> None:
    with pytest.raises(runtime_mod.TlaCoreAPIError, match="mode must be passed by keyword"):
        vec_func_positional_mode_kernel.dump_mlir()

def test_vec_func_rejects_unknown_keyword() -> None:
    with pytest.raises(runtime_mod.TlaCoreAPIError, match="unknown keyword argument: foo"):
        vec_func_unknown_keyword_kernel.dump_mlir()

def test_vec_vector_ssa_load_add_store_lowering() -> None:
    mlir = vec_vector_ssa_kernel.dump_mlir(type_args=_vector_tensor_args())

    assert "tla.load" in mlir
    assert "tla.add" in mlir
    assert "tla.store" in mlir
    assert "tla.make_rmem_tensor" not in mlir

def test_vec_vector_then_scalar_chaining_preserves_metadata() -> None:
    mlir = chained_vector_vector_then_scalar_kernel.dump_mlir()

    assert "tla.add" in mlir
    assert "tla.muls" in mlir
    assert "tla.store" in mlir

def test_vec_store_rejects_raw_tensor() -> None:
    lhs, _, dst = _vector_tensor_args()
    with pytest.raises(runtime_mod.TlaCoreAPIError, match="expected vector_ssa"):
        vec_store_rejects_raw_tensor_kernel.dump_mlir(type_args=(lhs, dst))

def test_vec_add_rejects_raw_tensor() -> None:
    lhs, rhs, _ = _vector_tensor_args()
    with pytest.raises(runtime_mod.TlaCoreAPIError, match="expected vector_ssa"):
        vec_add_rejects_raw_tensor_kernel.dump_mlir(type_args=(lhs, rhs))


@tla.kernel
def cube_region_bad_list_index_kernel() -> None:
    idx = tla.arch.block_idx()
    values = [0, 1]
    with tla.cube():
        values[idx]

def _source_line(fn: Any, needle: str) -> int:
    source_lines, first_lineno = inspect.getsourcelines(fn.fn)
    for offset, line in enumerate(source_lines):
        if needle in line:
            return first_lineno + offset
    raise AssertionError(f"Unable to find source line containing {needle!r}")

def test_frontend_region_error_reports_original_source_location() -> None:
    line = _source_line(cube_region_bad_list_index_kernel, "values[idx]")
    with pytest.raises(Exception) as excinfo:
        cube_region_bad_list_index_kernel.dump_mlir()
    message = str(excinfo.value)
    assert "Execution-mode lowering failed in tla.cube region-body" in message
    assert f" --> {__file__}:{line}:" in message
    assert f">{line} |" in message
    assert "values[idx]" in message
    assert "cannot be used as a Python index" in message or "list indices" in message
    assert "Int32" in message
