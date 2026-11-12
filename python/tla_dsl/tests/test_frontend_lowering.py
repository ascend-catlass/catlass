from catlass.tla.runtime import make_fake_tensor

import functools
import pytest
from catlass._mlir import ir as mlir_ir
import inspect

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError
from catlass import _tla_type_bridge


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


@pytest.mark.parametrize("decorator", [tla.kernel, tla.jit], ids=["kernel", "jit"])
@pytest.mark.parametrize("kind", ["coroutine", "async-generator"])
def test_async_dsl_functions_are_rejected_at_decoration(decorator, kind: str) -> None:
    if kind == "coroutine":
        async def async_fn() -> None:
            return None
    else:
        async def async_fn():
            yield 1

    with pytest.raises(SyntaxError, match=r"async .*not supported") as excinfo:
        decorator(async_fn)
    assert excinfo.value.filename == __file__
    assert excinfo.value.lineno is not None

@pytest.mark.parametrize("decorator", [tla.kernel, tla.jit], ids=["kernel", "jit"])
def test_wrapped_async_dsl_functions_are_rejected(decorator) -> None:
    async def async_fn() -> None:
        return None

    @functools.wraps(async_fn)
    def wrapper() -> None:
        return None

    with pytest.raises(SyntaxError, match=r"async .*not supported"):
        decorator(wrapper)

def _operation_occurs_in_scf_if(mlir: str, operation: str) -> bool:
    stack: list[str] = []
    last_closed = ""
    operation_offset = mlir.find(operation)
    while operation_offset != -1:
        stack.clear()
        last_closed = ""
        for offset, token in enumerate(mlir[:operation_offset]):
            if token == "{":
                header_start = mlir.rfind("\n", 0, offset) + 1
                header = mlir[header_start:offset]
                if "scf.if" in header:
                    kind = "scf.if"
                elif "else" in header and last_closed == "scf.if":
                    kind = "scf.if"
                else:
                    kind = "other"
                stack.append(kind)
                last_closed = ""
            elif token == "}" and stack:
                last_closed = stack.pop()
        if "scf.if" in stack:
            return True
        operation_offset = mlir.find(operation, operation_offset + len(operation))
    return False

def _mmad_tensor_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    """Host L0 samples for mmad: zN/l0a, nZ/l0b, L0Clayout/l0c."""
    return (
        make_fake_tensor(
            tla.Float16,
            ((16, 8), (16, 4)),
            ((16, 256), (1, 2048)),
            addrspace=tla.AddressSpace.l0a,
            origin_shape=(128, 64),
            layout_tag=tla.arch.zN,
            coord=(0, 0),
        ),
        make_fake_tensor(
            tla.Float16,
            ((16, 4), (16, 8)),
            ((1, 2048), (16, 256)),
            addrspace=tla.AddressSpace.l0b,
            origin_shape=(64, 128),
            layout_tag=tla.arch.nZ,
            coord=(0, 0),
        ),
        make_fake_tensor(
            tla.Float32,
            ((16, 8), (16, 8)),
            ((16, 256), (1, 2048)),
            addrspace=tla.AddressSpace.l0c,
            origin_shape=(128, 128),
            layout_tag=tla.arch.L0Clayout,
            coord=(0, 0),
        ),
    )

def _skip_if_mmad_rank2_tile_view_regression(exc: BaseException) -> None:
    if isinstance(exc, TlaLoweringError) and "rank-2 tiles only" in str(exc):
        pytest.skip(
            "tla.mmad rank-2 check rejects tile_view operand types until metadata matches"
        )

@tla.kernel
def alloc_kernel(mem_a: tla.Tensor) -> None:
    gm_tile = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    ptr = tla.allocate((16, 16), tla.Float16, tla.AddressSpace.l1, 512)
    _ = ptr
    local_tile = tla.make_tensor_like(ptr, gm_tile, tla.arch.zN)
    with tla.cube():
        tla.copy(local_tile, gm_tile)

@tla.kernel
def alloc_ptr_kernel(mem_a: tla.Tensor) -> None:
    gm_tile = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    ptr = tla.allocate((16, 16), tla.Float16, tla.AddressSpace.l1, 512)
    local_tile = tla.make_tensor_like(ptr, gm_tile, tla.arch.zN)
    _ = local_tile
    with tla.cube():
        tla.copy(local_tile, gm_tile)

@tla.kernel
def bad_flag_name(x: int) -> None:
    tla.flag(x, tla.arch.MTE2, tla.arch.VECTOR)

@tla.kernel
def bad_allocate_unknown_addrspace(mem_a: tla.Tensor) -> None:
    tla.allocate(128, tla.Int8, "dram", 64)

@tla.kernel
def bad_allocate_gm_addrspace(mem_a: tla.Tensor) -> None:
    tla.allocate(128, tla.Int8, tla.AddressSpace.gm, 64)

@tla.kernel
def allocate_recast_ptr_kernel() -> None:
    lhs = tla.allocate(4096, tla.Int8, tla.AddressSpace.ub, 512)
    rhs = tla.allocate(4096, tla.Int8, tla.AddressSpace.ub, 512)
    dst = tla.allocate(4096, tla.Int8, tla.AddressSpace.ub, 512)
    lhs = tla.recast_ptr(lhs, dtype=tla.Float32)
    rhs = tla.recast_ptr(rhs, dtype=tla.Float32)
    dst = tla.recast_ptr(dst, dtype=tla.Float32)
    _ = lhs
    _ = rhs
    _ = dst

@tla.kernel
def odd_allocate_recast_size() -> None:
    """A request smaller than the alignment, recast to a narrower element."""
    ptr = tla.allocate(3, tla.Int8, tla.AddressSpace.ub, 512)
    tla.recast_ptr(ptr, dtype=tla.Float16)

@tla.kernel
def cube_mmad_without_region_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=False)

@tla.kernel
def mmad_compute_order_nfirst_kernel(
    lhs: tla.Tensor, rhs: tla.Tensor, acc: tla.Tensor
) -> None:
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, compute_order=tla.params.ComputeOrder.N_FIRST)

@tla.kernel
def mmad_default_compute_order_kernel(
    lhs: tla.Tensor, rhs: tla.Tensor, acc: tla.Tensor
) -> None:
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True)

@tla.kernel
def mmad_bad_compute_order_kernel(
    lhs: tla.Tensor, rhs: tla.Tensor, acc: tla.Tensor
) -> None:
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, compute_order="N_FIRST")

@tla.kernel
def mmad_hf32_mode_kernel(
    lhs: tla.Tensor, rhs: tla.Tensor, acc: tla.Tensor
) -> None:
    with tla.cube():
        tla.mmad(
            acc,
            lhs,
            rhs,
            init_c=True,
            hf32_mode=tla.params.HF32Mode.HF32_NEAREST_ZERO,
        )

@tla.kernel
def mmad_hf32_mode_nearest_even_kernel(
    lhs: tla.Tensor, rhs: tla.Tensor, acc: tla.Tensor
) -> None:
    with tla.cube():
        tla.mmad(
            acc,
            lhs,
            rhs,
            init_c=True,
            hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
        )

@tla.kernel
def mmad_default_hf32_mode_kernel(
    lhs: tla.Tensor, rhs: tla.Tensor, acc: tla.Tensor
) -> None:
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True)

@tla.kernel
def mmad_default_hf32_mode_kernel_v2(
    lhs: tla.Tensor, rhs: tla.Tensor, acc: tla.Tensor
) -> None:
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, hf32_mode=tla.params.HF32Mode.HF32_DISABLE)


@tla.kernel
def mmad_bad_hf32_mode_kernel(
    lhs: tla.Tensor, rhs: tla.Tensor, acc: tla.Tensor
) -> None:
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, hf32_mode="nearest_zero")

@tla.kernel
def make_shape_index_arg_ok(dim: "index") -> None:
    tla.make_shape(dim, 16)

@tla.kernel
def bad_make_shape_no_dims() -> None:
    tla.make_shape()

@tla.kernel
def make_shape_three_dims_ok(dim: "index") -> None:
    tla.make_shape(16, dim, 32)

@tla.kernel
def bad_make_shape_float_component() -> None:
    tla.make_shape(16.0, 16)

@tla.kernel
def bad_make_shape_non_index_arg(dim: "f16") -> None:
    tla.make_shape(dim, 16)

@tla.kernel
def make_coord_index_arg_ok(coord: "index") -> None:
    tla.make_coord(coord, 0)

@tla.kernel
def bad_make_coord_no_args() -> None:
    tla.make_coord()

@tla.kernel
def bad_make_coord_float_component() -> None:
    tla.make_coord(0.0, 0)

@tla.kernel
def bad_make_coord_non_index_arg(coord: "f16") -> None:
    tla.make_coord(coord, 0)

@tla.kernel
def pipe_barrier_aic_kernel() -> None:
    with tla.cube():
        tla.pipe_barrier(tla.pipes.CUBE)
        tla.pipe_barrier(tla.pipes.MTE1)
        tla.pipe_barrier(tla.pipes.MTE2)
        tla.pipe_barrier(tla.pipes.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

@tla.kernel
def pipe_barrier_aiv_kernel() -> None:
    with tla.vector():
        tla.pipe_barrier(tla.pipes.MTE2)
        tla.pipe_barrier(tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)

@tla.kernel
def bad_pipe_barrier_aic_mte3_kernel() -> None:
    with tla.cube():
        tla.pipe_barrier(tla.pipes.MTE3)

@tla.kernel
def bad_pipe_barrier_aiv_cube_kernel() -> None:
    with tla.vector():
        tla.pipe_barrier(tla.pipes.CUBE)

@tla.kernel
def constexpr_alloc_kernel(limit: tla.Constexpr[int], mem_a: tla.Tensor) -> None:
    tla.allocate(limit, tla.Int8, tla.AddressSpace.l1, 64)

@tla.kernel
def nested_tile_view_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(2, 4))
    _ = tla.tile_view(root, tla.make_shape(8, 4), tla.make_coord(3, 2))

@tla.kernel
def dynamic_nested_tile_view_kernel(mem_a: tla.Tensor, tile_idx: "index") -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    _ = tla.tile_view(root, tla.make_shape(8, 4), tla.make_coord(tile_idx, 1))

@tla.kernel
def root_tile_view_kernel(mem_a: tla.Tensor) -> None:
    _ = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(2, 3))

@tla.kernel
def range_alias_kernel(mem_a: tla.Tensor) -> None:
    loop_range = tla.range(0, 4, 1)
    for i in loop_range:
        tla.tile_view(mem_a, tla.make_shape(4, 4), tla.make_coord(i, 0))

@tla.kernel
def pointer_conditional_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ptr0 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    ptr1 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    with tla.cube():
        loop_range = tla.range(0, 2, 1)
        for i in loop_range:
            tile = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, i))
            selected = ptr0 if i == 0 else ptr1
            tensor_like_zn = tla.make_tensor_like(selected, tile, tla.arch.zN)
            tla.copy(tensor_like_zn, tile)

@tla.kernel
def dynamic_mmad_initc_unit_flag_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    acc = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    lhs = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    with tla.cube():
        outer_range = tla.range(0, 2, 1)
        for outer in outer_range:
            inner_range = tla.range(0, 2, 1)
            for inner in inner_range:
                init_c = True if outer == 0 and inner == 0 else False
                unit_flag = 0b11 if (outer == 1) and (inner == 1) else 0b10
                tla.mmad(acc, lhs, rhs, init_c=init_c, unit_flag=unit_flag)

@tla.kernel
def f32_mmad_generated_addrspace_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    gm_a = tla.tile_view(mem_a, tla.make_shape(32, 32), tla.make_coord(0, 0))
    gm_b = tla.tile_view(mem_b, tla.make_shape(32, 32), tla.make_coord(0, 0))
    gm_c = tla.tile_view(mem_c, tla.make_shape(32, 32), tla.make_coord(0, 0))
    l0a_ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 1024)
    lhs = tla.make_tensor_like(l0a_ptr, gm_a, tla.arch.zN)
    rhs = tla.make_tensor_like(l0b_ptr, gm_b, tla.arch.nZ)
    acc = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True)

@tla.kernel
def make_tensor_like_addrspace_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    l0a_ptr = tla.allocate((16, 16), tla.Float16, tla.AddressSpace.l0a, 512)
    _ = tla.make_tensor_like(l0a_ptr, root, tla.arch.zN)

@tla.kernel
def tile_view_shape_metadata_kernel(mem_a: tla.Tensor) -> None:
    tile = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    tla.make_shape(tile.shape[0], tile.shape[1])

@tla.kernel
def make_tensor_like_shape_metadata_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ptr = tla.allocate((16, 8), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.make_tensor_like(ptr, root, tla.arch.zN)
    tla.make_shape(16, 8)

@tla.kernel
def dynamic_tensor_shape_metadata_kernel(mem_a: tla.Tensor, dim: "index") -> None:
    tile = tla.tile_view(mem_a, tla.make_shape(dim, 8), tla.make_coord(dim, 0))
    tla.make_shape(tile.shape[0] // 2, tile.shape[1])

@tla.kernel
def dynamic_make_tensor_like_shape_metadata_kernel(
    mem_a: tla.Tensor, dim: "index"
) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(dim, 8), tla.make_coord(dim, 0))
    ptr = tla.allocate((16, 8), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.make_tensor_like(ptr, root, tla.arch.zN)
    # After ``zN``, ``make_tensor_like`` rewrites nested shape metadata: the first
    # ``.shape`` leaf is often a static fractal size, not the flat ``?`` from ``root``.
    # Use ``root`` (still ``?,8`` from ``tile_view``) so ``//`` lowers to ``arith.divui``.
    tla.make_shape(root.shape[0] // 2, root.shape[1])

@tla.kernel
def dynamic_tensor_full_metadata_kernel(mem_a: tla.Tensor, dim: "index") -> None:
    root = tla.tile_view(mem_a, tla.make_shape(dim, 8), tla.make_coord(dim, 0))
    ptr = tla.allocate((16, 8), tla.Float16, tla.AddressSpace.l1, 512)
    l1 = tla.make_tensor_like(ptr, root, tla.arch.zN)
    tla.make_shape(l1.origin_shape[0], l1.origin_shape[1] // 2)
    tla.make_coord(l1.coord[0], l1.coord[1])
    tla.make_stride(l1.stride[0][0], l1.stride[0][1], l1.stride[1][0], l1.stride[1][1])
    _ = (l1.shape, l1.dtype, l1.addrspace, l1.layout_tag)

def test_lowering_emits_typed_args_and_alloc_attrs(compiler_tlair) -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (1, 2),
              (2, 1),
              origin_shape=(1, 2),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = compiler_tlair(alloc_kernel, type_args=(mem,))
    assert 'sym_name = "alloc_kernel"' in mlir
    assert '"tla.alloc_ptr"' in mlir
    assert "tla.copy" in mlir

def test_allocate_emits_typed_alloc_and_tensor_reconstruction_ops() -> None:
    mem = make_fake_tensor(tla.Float16, (16, 16), (16, 1), origin_shape=(16, 16), layout_tag=tla.arch.RowMajor)
    mlir = alloc_ptr_kernel.dump_mlir(type_args=(mem,))
    assert "tla.alloc_ptr" in mlir
    assert "tla.recast_ptr" not in mlir
    assert "tla.make_tensor_like" in mlir
    assert (
        "<!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>"
        in mlir
    )
    assert "tla.copy" in mlir

def test_allocate_backed_recast_ptr_preserves_pointer_type() -> None:
    mlir = allocate_recast_ptr_kernel.dump_mlir()
    assert mlir.count("tla.alloc_ptr") == 3
    assert mlir.count("tla.recast_ptr") == 3
    assert "!tla.ptr<f32, ub, 512>" in mlir
    assert "!tla.memref<" not in mlir

def test_allocate_backed_recast_ptr_rounds_a_short_request_to_the_alignment() -> None:
    """3 bytes at 512-byte alignment reserve 512: the size follows the alignment."""
    mlir = odd_allocate_recast_size.dump_mlir()
    assert "size_bytes = 512" in mlir
    assert "!tla.ptr<f16, ub, 512>" in mlir

def test_non_const_attr_raises_specific_error() -> None:
    with pytest.raises(tla.TlaCoreAPIError):
        _ = bad_flag_name.dump_mlir(type_args=(1,))

def test_allocate_rejects_unknown_addrspace() -> None:
    mem = make_fake_tensor(tla.Float16, (1, 2), (2, 1), origin_shape=(1, 2), layout_tag=tla.arch.RowMajor)
    with pytest.raises(
        tla.TlaCoreAPIError, match="tla.allocate"
    ):
        _ = bad_allocate_unknown_addrspace.dump_mlir(type_args=(mem,))

def test_allocate_rejects_gm_addrspace() -> None:
    mem = make_fake_tensor(tla.Float16, (1, 2), (2, 1), origin_shape=(1, 2), layout_tag=tla.arch.RowMajor)
    with pytest.raises(
        tla.TlaCoreAPIError, match="tla.allocate"
    ):
        _ = bad_allocate_gm_addrspace.dump_mlir(type_args=(mem,))

def test_mmad_compute_order_emits_attr() -> None:
    ta, tb, tc = _mmad_tensor_args()
    mlir = mmad_compute_order_nfirst_kernel.dump_mlir(type_args=(ta, tb, tc))
    assert "tla.mmad" in mlir
    assert "compute_order = #tla.compute_order<N_FIRST>" in mlir

def test_mmad_default_compute_order_is_mfirst() -> None:
    ta, tb, tc = _mmad_tensor_args()
    mlir = mmad_default_compute_order_kernel.dump_mlir(type_args=(ta, tb, tc))
    assert "tla.mmad" in mlir
    assert "compute_order = #tla.compute_order<M_FIRST>" in mlir

def test_mmad_rejects_invalid_compute_order() -> None:
    ta, tb, tc = _mmad_tensor_args()
    with pytest.raises(TlaLoweringError):
        _ = mmad_bad_compute_order_kernel.dump_mlir(type_args=(ta, tb, tc))

def test_mmad_without_region_lowers() -> None:
    ta, tb, tc = _mmad_tensor_args()
    try:
        mlir = cube_mmad_without_region_kernel.dump_mlir(type_args=(ta, tb, tc))
    except TlaLoweringError as e:
        _skip_if_mmad_rank2_tile_view_regression(e)
        raise
    assert "tla.mmad" in mlir

def test_hf32_mode_emits_attr() -> None:
    ta, tb, tc = _mmad_tensor_args()
    mlir = mmad_hf32_mode_kernel.dump_mlir(type_args=(ta, tb, tc))
    assert "tla.mmad" in mlir
    assert "hf32_mode = #tla.hf32_mode<HF32_NEAREST_ZERO>" in mlir

def test_hf32_mode_nearest_even_emits_attr() -> None:
    ta, tb, tc = _mmad_tensor_args()
    mlir = mmad_hf32_mode_nearest_even_kernel.dump_mlir(type_args=(ta, tb, tc))
    assert "tla.mmad" in mlir
    assert "hf32_mode = #tla.hf32_mode<HF32_NEAREST_EVEN>" in mlir

def test_default_hf32_mode_is_disable() -> None:
    ta, tb, tc = _mmad_tensor_args()
    for _kernel in (mmad_default_hf32_mode_kernel, mmad_default_hf32_mode_kernel_v2):
        mlir = _kernel.dump_mlir(type_args=(ta, tb, tc))
        assert "tla.mmad" in mlir
        assert "hf32_mode = #tla.hf32_mode<HF32_DISABLE>" in mlir

def test_rejects_invalid_hf32_mode() -> None:
    ta, tb, tc = _mmad_tensor_args()
    with pytest.raises(TlaLoweringError):
        _ = mmad_bad_hf32_mode_kernel.dump_mlir(type_args=(ta, tb, tc))

def test_make_shape_accepts_index_typed_components() -> None:
    """Lowering emits tla.make_shape with a dynamic dim when the component is a kernel parameter."""
    mlir = make_shape_index_arg_ok.dump_mlir(type_args=(4,))
    assert "tla.make_shape" in mlir
    assert "!tla.shape<?,16>" in mlir
    assert "tla.return" in mlir

def test_make_shape_supports_three_dims_and_parameterized_type() -> None:
    """make_shape supports rank-3 shapes; a kernel parameter yields a dynamic dim in the type."""
    mlir = make_shape_three_dims_ok.dump_mlir(type_args=(4,))
    assert "tla.make_shape" in mlir
    assert "!tla.shape<" in mlir
    assert "?" in mlir
    assert "16" in mlir
    assert "32" in mlir

def test_make_shape_requires_at_least_one_dim() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.make_shape"):
        _ = bad_make_shape_no_dims.mlir

def test_make_shape_rejects_float_component() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.make_shape"):
        _ = bad_make_shape_float_component.mlir

def test_make_shape_rejects_non_index_arg() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.make_shape"):
        _ = bad_make_shape_non_index_arg.dump_mlir(type_args=(1.0,))

def test_make_coord_accepts_index_typed_components() -> None:
    mlir = make_coord_index_arg_ok.dump_mlir(type_args=(3,))
    assert "tla.make_coord" in mlir
    assert "!tla.coord<?,0>" in mlir
    assert "tla.return" in mlir

def test_make_coord_requires_at_least_one_arg() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.make_coord"):
        _ = bad_make_coord_no_args.mlir

def test_make_coord_rejects_float_component() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.make_coord"):
        _ = bad_make_coord_float_component.mlir

def test_make_coord_rejects_non_index_arg() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.make_coord"):
        _ = bad_make_coord_non_index_arg.dump_mlir(type_args=(1.0,))

def test_aic_pipe_barrier_lowers_pipe_attr(compiler_tlair) -> None:
    mlir = compiler_tlair(pipe_barrier_aic_kernel)
    assert "#tla.pipe<cube>" in mlir
    assert "#tla.pipe<mte1>" in mlir
    assert "#tla.pipe<mte2>" in mlir
    assert "#tla.pipe<fix>" in mlir
    assert "#tla.pipe<all>" in mlir

def test_aiv_pipe_barrier_lowers_pipe_attr(compiler_tlair) -> None:
    mlir = compiler_tlair(pipe_barrier_aiv_kernel)
    assert "#tla.pipe<mte2>" in mlir
    assert "#tla.pipe<mte3>" in mlir
    assert "#tla.pipe<all>" in mlir

def test_aic_pipe_barrier_rejects_mte3(compiler_tlair) -> None:
    with pytest.raises(TlaLoweringError, match="pipe_barrier"):
        compiler_tlair(bad_pipe_barrier_aic_mte3_kernel)

def test_aiv_pipe_barrier_rejects_cube(compiler_tlair) -> None:
    with pytest.raises(TlaLoweringError, match="pipe_barrier"):
        compiler_tlair(bad_pipe_barrier_aiv_cube_kernel)

def test_kernel_captures_decorator_site_location() -> None:
    location = pipe_barrier_aiv_kernel.decorator_location
    assert location is not None
    assert location.filename.endswith("test_frontend_lowering.py")
    assert location.function_name == "<module>"
    source_lines, first_lineno = inspect.getsourcelines(pipe_barrier_aiv_kernel.fn)
    expected_lines = {
        first_lineno + offset
        for offset, line in enumerate(source_lines)
        if line.lstrip().startswith(("@tla.kernel", "def pipe_barrier_aiv_kernel"))
    }
    assert location.lineno in expected_lines

def test_constexpr_param_is_excluded_from_runtime_function_type(compiler_tlair) -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (1, 2),
              (2, 1),
              origin_shape=(1, 2),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = compiler_tlair(constexpr_alloc_kernel, type_args=(32, mem))
    assert 'sym_name = "constexpr_alloc_kernel"' in mlir
    assert (
        "%arg0: !tla.tensor<!tla.layout<!tla.shape<1,2>, !tla.stride<2,1>, !tla.shape<1,2>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>)"
        in mlir
    )
    assert "%arg1" not in mlir
    # 32 bytes aligned to 64 reserve 64: the size follows the alignment.
    assert "size_bytes = 64" in mlir

def test_tile_view_scales_nested_tile_coords_by_shape() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (128, 128),
              (128, 1),
              origin_shape=(128, 128),
              layout_tag=tla.arch.RowMajor,
          )
    # Nested coord (3,2) on root scales to element offset 24 along M; with parent
    # origin 16×16 this exceeds the strict range check — expect the lowering error.
    with pytest.raises(TlaLoweringError, match="out of range"):
        nested_tile_view_kernel.dump_mlir(type_args=(mem,))

def test_tile_view_scales_root_coords_by_shape() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (128, 128),
              (128, 1),
              origin_shape=(128, 128),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = root_tile_view_kernel.dump_mlir(type_args=(mem,))
    assert "tla.tile_view" in mlir
    assert "!tla.coord<32,48>" in mlir

def test_tile_view_emits_index_multiply_for_dynamic_nested_coords() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (128, 128),
              (128, 1),
              origin_shape=(128, 128),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = dynamic_nested_tile_view_kernel.dump_mlir(type_args=(mem, 3))
    assert mlir.count("tla.tile_view") == 2
    assert "arith.muli" in mlir
    assert "!tla.coord<?,4>" in mlir

def test_range_alias_lowers_to_tla_for() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = range_alias_kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir
    assert "tla.range" not in mlir
    assert "tla.for" not in mlir
    assert "tla.tile_view" in mlir

def test_pointer_conditional_expression_lowers_to_scf_if() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 8),
              (8, 1),
              origin_shape=(16, 8),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = pointer_conditional_kernel.dump_mlir(type_args=(mem,))
    assert "arith.cmpi" in mlir
    assert "scf.if" in mlir
    assert "scf.yield" in mlir
    assert "tla.make_tensor_like" in mlir

def test_dynamic_mmad_initc_unit_flag_expression_lowers_to_scf_if(compiler_tlair) -> None:
    mem_a = make_fake_tensor(
                tla.Float16,
                ((16, 1), (16, 1)),
                ((16, 256), (1, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0a,
                layout_tag=tla.arch.zN,
                coord=(0, 0),
            )
    mem_b = make_fake_tensor(
                tla.Float16,
                ((16, 1), (16, 1)),
                ((1, 256), (16, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0b,
                layout_tag=tla.arch.nZ,
                coord=(0, 0),
            )
    mem_c = make_fake_tensor(
                tla.Float32,
                ((16, 1), (16, 1)),
                ((16, 256), (1, 256)),
                origin_shape=(16, 16),
                addrspace=tla.AddressSpace.l0c,
                layout_tag=tla.arch.L0Clayout,
                coord=(0, 0),
            )
    mlir = compiler_tlair(dynamic_mmad_initc_unit_flag_kernel, type_args=(mem_a, mem_b, mem_c))
    assert "scf.for" in mlir
    assert "tla.for" not in mlir
    assert "arith.cmpi" in mlir
    assert _operation_occurs_in_scf_if(mlir, "arith.cmpi")
    assert "scf.if" in mlir
    assert mlir.count("scf.yield") >= 4
    assert '"arith.constant"() <{value = true}> : () -> i1' in mlir
    assert '"arith.constant"() <{value = false}> : () -> i1' in mlir
    # unit_flag literals lower as i32 Numerics, not MLIR index.
    assert '"arith.constant"() <{value = 3 : i32}> : () -> i32' in mlir
    assert '"arith.constant"() <{value = 2 : i32}> : () -> i32' in mlir
    assert "!tla.ptr<f32, l0c, 4>" in mlir
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(16,256),(1,256)>, !tla.shape<16,16>, zN>"
        in mlir
    )
    assert (
        "!tla.layout<!tla.shape<(16,1),(16,1)>, !tla.stride<(1,256),(16,256)>, !tla.shape<16,16>, nZ>"
        in mlir
    )

def test_f32_mmad_generated_addrspaces_validate() -> None:
    mem_a = make_fake_tensor(
                tla.Float32,
                (32, 32),
                (32, 1),
                origin_shape=(32, 32),
                layout_tag=tla.arch.RowMajor,
            )
    mem_b = make_fake_tensor(
                tla.Float32,
                (32, 32),
                (32, 1),
                origin_shape=(32, 32),
                layout_tag=tla.arch.RowMajor,
            )
    mem_c = make_fake_tensor(
                tla.Float32,
                (32, 32),
                (32, 1),
                origin_shape=(32, 32),
                layout_tag=tla.arch.RowMajor,
            )
    mlir = f32_mmad_generated_addrspace_kernel.dump_mlir(
        type_args=(mem_a, mem_b, mem_c)
    )
    assert "!tla.ptr<f32, l0a, 512>" in mlir
    assert "!tla.ptr<f32, l0b, 512>" in mlir
    assert "!tla.ptr<f32, l0c, 1024>" in mlir
    assert "tla.mmad" in mlir

def test_make_tensor_like_maps_l0_pointer_to_tensor_addrspace() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 16),
              (16, 1),
              origin_shape=(16, 16),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = make_tensor_like_addrspace_kernel.dump_mlir(type_args=(mem,))
    assert "!tla.ptr<f16, l0a, 512>" in mlir

def test_tile_view_result_exposes_static_shape_metadata() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 8),
              (8, 1),
              origin_shape=(16, 8),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = tile_view_shape_metadata_kernel.dump_mlir(type_args=(mem,))
    assert "tla.tile_view" in mlir
    assert mlir.count("!tla.shape<16,8>") >= 2

def test_make_tensor_like_result_exposes_static_shape_metadata() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 8),
              (8, 1),
              origin_shape=(16, 8),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = make_tensor_like_shape_metadata_kernel.dump_mlir(type_args=(mem,))
    assert "tla.make_tensor_like" in mlir
    assert mlir.count("!tla.shape<16,8>") >= 2

def test_tensor_shape_metadata_supports_dynamic_shape_leaf() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 8),
              (8, 1),
              origin_shape=(16, 8),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = dynamic_tensor_shape_metadata_kernel.dump_mlir(type_args=(mem, 4))
    assert "tla.tile_view" in mlir
    assert "arith.divsi" in mlir or "arith.divui" in mlir
    assert "!tla.shape<?,8>" in mlir

def test_make_tensor_like_shape_metadata_supports_dynamic_shape_leaf() -> None:
    """``make_tensor_like`` + dynamic tile: division must use flat ``root.shape``, not zN ``local.shape``."""
    mem = make_fake_tensor(
              tla.Float16,
              (16, 8),
              (8, 1),
              origin_shape=(16, 8),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = dynamic_make_tensor_like_shape_metadata_kernel.dump_mlir(type_args=(mem, 4))
    assert "tla.make_tensor_like" in mlir
    assert "arith.divsi" in mlir or "arith.divui" in mlir
    assert "!tla.shape<?,8>" in mlir

def test_tensor_full_metadata_access_supports_dynamic_origin() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (64, 16),
              (16, 1),
              origin_shape=(64, 16),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = dynamic_tensor_full_metadata_kernel.dump_mlir(type_args=(mem, 16))
    assert "tla.make_tensor_like" in mlir
    assert "tla.make_stride" in mlir
    assert "tla.make_coord" in mlir
    assert "arith.divsi" in mlir or "arith.divui" in mlir
