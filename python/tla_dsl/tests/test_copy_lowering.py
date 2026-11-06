import pathlib
import re
import subprocess

import pytest

import catlass.runtime as runtime_mod
import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError
from catlass.tla.runtime import make_fake_tensor


def _require_hivm_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = (
        repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    )
    if not tla_compile.exists():
        raise AssertionError("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile


@tla.kernel
def copy_gm_to_l1_kernel(mem_in: tla.Tensor) -> None:
    tile = tla.tile_view(mem_in, tla.make_shape(32, 32), tla.make_coord(1, 1))
    ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512)
    local = tla.make_tensor_like(ptr, tile, tla.arch.zN)
    with tla.cube():
        tla.copy(local, tile)


@tla.kernel
def copy_kernel_arg_directly_to_ub_kernel(mem_in: tla.Tensor) -> None:
    ub_ptr = tla.allocate((16, 16), tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(16, 16),
            tla.make_stride(16, 1),
        ),
    )
    with tla.vector():
        tla.copy(ub, mem_in)


@tla.kernel
def copy_l1_to_gm_kernel(mem_in: tla.Tensor) -> None:
    """L1 -> GM: the copy unit has no such direction at all."""
    l1 = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512),
        mem_in,
        tla.arch.zN,
    )
    with tla.cube():
        tla.copy(mem_in, l1)


@tla.kernel
def copy_ub_to_ub_kernel(mem_in: tla.Tensor) -> None:
    src_ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.ub, 256)
    dst_ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.ub, 256)
    src = tla.make_tensor_like(src_ptr, mem_in, tla.arch.RowMajor)
    dst = tla.make_tensor_like(dst_ptr, mem_in, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(dst, src)


def test_frontend_rejects_copy_l1_to_gm() -> None:
    """A direction the hardware does not have is named as such, not as an
    unimplemented route."""
    mem = make_fake_tensor(
        tla.Float32,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"Hardware unsupported copy route",
    ):
        copy_l1_to_gm_kernel.dump_mlir(type_args=(mem,))


@tla.kernel
def copy_fp8_gm_to_ub_kernel(mem_in: tla.Tensor) -> None:
    """fp8 staged through UB on the vector path."""
    ub = tla.make_tensor(
        tla.allocate((32, 32), tla.Float8E4M3FN, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(32, 1)),
    )
    with tla.vector():
        tla.copy(ub, mem_in)


def test_frontend_accepts_fp8_gm_to_ub_copy() -> None:
    """fp8 has a vector staging route now.

    It used to be refused on the grounds that fp8 was a cube *operand* format.
    That was a statement about the registrations rather than about the
    hardware: Vector/dma.cpp registers GM <-> UB for both fp8 encodings, so the
    front-end route table has to admit them or the copy dies before it reaches
    the lowering.
    """
    mem = make_fake_tensor(
        tla.Float8E4M3FN,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=tla.arch.RowMajor,
    )
    copy_fp8_gm_to_ub_kernel.dump_mlir(type_args=(mem,))


@tla.kernel
def copy_fp4_to_l0a_without_scale_kernel() -> None:
    """A packed fp4 tile loaded into L0A with no scale block attached."""
    l1 = tla.make_tensor(
        tla.allocate((128, 64), tla.Float4E2M1, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
    )
    l0a = tla.make_tensor_like(
        tla.allocate((128, 64), tla.Float4E2M1, tla.AddressSpace.l0a, 512),
        l1,
        tla.arch.zN,
    )
    with tla.cube():
        tla.copy(l0a, l1)


def test_frontend_rejects_unscaled_fp4_l1_to_l0a_copy() -> None:
    """fp4 is microscaling-only: the L1->L0A load has to carry a scale."""
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"packed fp4 to l0a/l0b must carry scale",
    ):
        copy_fp4_to_l0a_without_scale_kernel.dump_mlir()


def test_frontend_copy_gm_to_l1_lowers_to_runtime_call(tmp_path) -> None:
    tla_compile = _require_hivm_tla_compile()
    mem = make_fake_tensor(
        tla.Float32,
        (128, 128),
        (128, 1),
        origin_shape=(128, 128),
        layout_tag=tla.arch.RowMajor,
    )

    mlir = copy_gm_to_l1_kernel.dump_mlir(type_args=(mem,))
    assert "!tla.layout<!tla.shape<(16,2),(8,4)>" in mlir
    assert "!tla.ptr<f32, l1, 512>" in mlir
    assert "tla.copy" in mlir

    input_path = tmp_path / "copy_gm_to_l1.mlir"
    input_path.write_text(mlir)
    try:
        result = subprocess.run(
            [str(tla_compile), str(input_path), "-o", "-"],
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if "missing descriptor for tla.tile_view result" in exc.stderr:
            pytest.skip(
                "tla.tile_view descriptor lowering is not available in this build"
            )
        raise

    lowered = result.stdout
    assert "copy_gm_RowMajor_to_l1_zN_float" in lowered
    assert "_mlir_ciface_copy_gm_RowMajor_to_l1_zN_float" not in lowered
    assert "hacc.always_inline" in lowered
    assert "hivm.func_core_type = #hivm.func_core_type<AIC>" in lowered
    assert "llvm.emit_c_interface" in lowered
    assert "memref.subview" not in lowered
    assert "memref.cast %arg0 : memref<128x128xf32" in lowered
    assert lowered.count("hivm.hir.pointer_cast") == 1
    assert '"tla.copy"' not in lowered
    assert '"tla.alloc_ptr"' not in lowered
    assert '"tla.recast_ptr"' not in lowered


def test_kernel_gm_arg_copies_directly_to_ub(tmp_path) -> None:
    tla_compile = _require_hivm_tla_compile()
    mem_in = make_fake_tensor(
        tla.Float32,
        (16, 16),
        (16, 1),
        origin_shape=(16, 16),
        layout_tag=tla.arch.RowMajor,
    )

    mlir = copy_kernel_arg_directly_to_ub_kernel.dump_mlir(type_args=(mem_in,))
    assert "tla.tile_view" not in mlir
    assert "tla.make_tensor_like" not in mlir
    assert mlir.count("tla.make_tensor ") == 1
    source_copy = next(line for line in mlir.splitlines() if "tla.copy" in line)
    assert "%arg0" in source_copy

    input_path = tmp_path / "copy_kernel_arg_directly_to_ub.mlir"
    input_path.write_text(mlir)
    result = subprocess.run(
        [
            str(tla_compile),
            str(input_path),
            "-o",
            "-",
            "--mlir-print-ir-after=tla-lower-func",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    descriptor_match = re.search(
        r"(?P<descriptor>%[A-Za-z0-9_]+) = tla\.tensor_desc %arg0 shape\[",
        result.stderr,
    )
    assert descriptor_match is not None, result.stderr
    lowered_copy = next(
        line for line in result.stderr.splitlines() if "tla.copy" in line
    )
    assert descriptor_match.group("descriptor") in lowered_copy
    assert "%arg0" not in lowered_copy
    assert "copy_gm_RowMajor_to_ub_RowMajor_float" in result.stdout
    assert "tla.copy" not in result.stdout


@tla.kernel
def copy_dynamic_l0c_to_ub_split_m_kernel(gm_c: tla.Tensor) -> None:
    """Reproduce a dynamically clipped N extent flowing from L0C into UB."""
    l0c_ptr = tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l0c, 512)
    ub_ptr = tla.allocate((64, 128), tla.Float32, tla.AddressSpace.ub, 512)

    for col in tla.range(tla.arch.block_idx(), 1, tla.arch.block_num()):
        gm_tile = tla.tile_view(gm_c, tla.make_shape(128, 128), tla.make_coord(0, col))
        l0c = tla.make_tensor_like(l0c_ptr, gm_tile, tla.arch.L0Clayout)
        ub = tla.make_tensor_like(ub_ptr, l0c, tla.arch.RowMajor)
        with tla.cube():
            tla.copy(
                ub,
                l0c,
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
            )


def test_split_m_dynamic_RowMajor_stride_uses_child_n_extent(tmp_path) -> None:
    """Split-M destination stride0 must be aligned N, not the L0C packing stride."""
    tla_compile = _require_hivm_tla_compile()
    gm_c = make_fake_tensor(
        tla.Float32,
        (128, 128),
        (1, 128),
        origin_shape=(128, 128),
        layout_tag=tla.arch.ColumnMajor,
    )

    mlir = copy_dynamic_l0c_to_ub_split_m_kernel.dump_mlir(type_args=(gm_c,))
    assert "!tla.shape<128,?>" in mlir
    assert "!tla.stride<?,1>" in mlir

    input_path = tmp_path / "copy_dynamic_l0c_to_ub_split_m.mlir"
    input_path.write_text(mlir)
    result = subprocess.run(
        [str(tla_compile), str(input_path), "-o", "-"],
        text=True,
        capture_output=True,
        check=True,
    )

    call_match = re.search(
        r"call @copy_l0c_to_ub_RowMajor_splitm_float_float\(([^)]*)\)",
        result.stdout,
    )
    assert call_match is not None, result.stdout
    operands = [operand.strip() for operand in call_match.group(1).split(",")]
    # 2 memref + 12 src (L0C) + 12 dst (row-major, unified 4D) + unitFlag + subBlockId = 28.
    assert len(operands) == 28
    dst_shape1 = operands[15]
    dst_stride0 = operands[18]
    _assert_lowered_f32_dynamic_stride_is_32b_aligned(
        result.stdout, extent=dst_shape1, stride=dst_stride0
    )


_MAKE_TENSOR_LIKE_LAYOUT_CASES = (
    ("RowMajor", tla.arch.RowMajor),
    ("ColumnMajor", tla.arch.ColumnMajor),
    ("zN", tla.arch.zN),
    ("nZ", tla.arch.nZ),
    ("L0Clayout", tla.arch.L0Clayout),
    ("zNUnAlign", tla.arch.zNUnAlign),
)
_MAKE_TENSOR_LIKE_PARENT_LAYOUT = tla.arch.RowMajor
_MAKE_TENSOR_LIKE_CHILD_LAYOUT = tla.arch.RowMajor


@tla.kernel
def make_tensor_like_layout_pair_kernel(mem_in: tla.Tensor, m: int, n: int) -> None:
    """Build one dynamic parent->child make_tensor_like layout pair."""
    root = tla.make_tensor(
        mem_in.ptr,
        tla.make_layout(
            tla.make_shape(m, n),
            tla.make_stride(n, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    parent_ptr = tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512)
    child_ptr = tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512)
    with tla.cube():
        parent = tla.make_tensor_like(parent_ptr, root, _MAKE_TENSOR_LIKE_PARENT_LAYOUT)
        child = tla.make_tensor_like(child_ptr, parent, _MAKE_TENSOR_LIKE_CHILD_LAYOUT)
        tla.make_shape(child.origin_shape[0], child.origin_shape[1])


def _tensor_desc_metadata(
    line: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    match = re.search(
        r"tla\.tensor_desc\s+%\S+\s+shape\[([^\]]+)\]\s+"
        r"stride\[([^\]]+)\]\s+origin_shape\[([^\]]+)\]\s+"
        r"coord\[([^\]]+)\]\s+:",
        line,
    )
    assert match is not None, line
    return tuple(
        [value.strip() for value in match.group(group).split(",")]
        for group in range(1, 5)
    )


def _index_binary_operands(mlir: str, result: str, op_name: str) -> tuple[str, str]:
    ssa = r"%[A-Za-z0-9_]+"
    match = re.search(
        rf"^\s*{re.escape(result)} = arith\.{op_name} "
        rf"(?P<lhs>{ssa}), (?P<rhs>{ssa}) : index$",
        mlir,
        re.MULTILINE,
    )
    assert match is not None, f"missing defining arith.{op_name} for {result}"
    return match.group("lhs"), match.group("rhs")


def _assert_f32_dynamic_stride_is_32b_aligned(
    mlir: str, extent: str, stride: str
) -> None:
    quotient, alignment = _index_binary_operands(mlir, stride, "muli")
    adjusted, divisor = _index_binary_operands(mlir, quotient, "divsi")
    input_extent, delta = _index_binary_operands(mlir, adjusted, "addi")
    assert input_extent == extent
    assert divisor == alignment
    assert re.search(
        rf"^\s*{re.escape(alignment)} = arith\.constant 8 : index$",
        mlir,
        re.MULTILINE,
    )
    assert re.search(
        rf"^\s*{re.escape(delta)} = arith\.constant 7 : index$",
        mlir,
        re.MULTILINE,
    )


def _llvm_i64_binary_operands(mlir: str, result: str, op_name: str) -> tuple[str, str]:
    ssa = r"%[A-Za-z0-9_]+"
    match = re.search(
        rf"^\s*{re.escape(result)} = llvm\.{op_name} "
        rf"(?P<lhs>{ssa}), (?P<rhs>{ssa})\s*: i64$",
        mlir,
        re.MULTILINE,
    )
    assert match is not None, f"missing defining llvm.{op_name} for {result}"
    return match.group("lhs"), match.group("rhs")


def _assert_lowered_f32_dynamic_stride_is_32b_aligned(
    mlir: str, extent: str, stride: str
) -> None:
    quotient, alignment = _llvm_i64_binary_operands(mlir, stride, "mul")
    adjusted, divisor = _llvm_i64_binary_operands(mlir, quotient, "sdiv")
    input_extent, delta = _llvm_i64_binary_operands(mlir, adjusted, "add")
    assert input_extent == extent
    assert divisor == alignment
    assert re.search(
        rf"^\s*{re.escape(alignment)} = llvm\.mlir\.constant\(8 : index\) : i64$",
        mlir,
        re.MULTILINE,
    )
    assert re.search(
        rf"^\s*{re.escape(delta)} = llvm\.mlir\.constant\(7 : index\) : i64$",
        mlir,
        re.MULTILINE,
    )


@pytest.mark.parametrize(
    ("parent_name", "parent_layout"),
    _MAKE_TENSOR_LIKE_LAYOUT_CASES,
)
@pytest.mark.parametrize(
    ("child_name", "child_layout"),
    _MAKE_TENSOR_LIKE_LAYOUT_CASES,
)
def test_make_tensor_like_supports_every_layout_pair(
    parent_name: str,
    parent_layout: object,
    child_name: str,
    child_layout: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """All seven layouts can be the parent and child of make_tensor_like."""
    monkeypatch.setitem(globals(), "_MAKE_TENSOR_LIKE_PARENT_LAYOUT", parent_layout)
    monkeypatch.setitem(globals(), "_MAKE_TENSOR_LIKE_CHILD_LAYOUT", child_layout)
    mem_in = make_fake_tensor(
        tla.Float32,
        (128, 128),
        (128, 1),
        origin_shape=(128, 128),
        layout_tag=tla.arch.RowMajor,
    )

    mlir = make_tensor_like_layout_pair_kernel.dump_mlir(type_args=(mem_in, 64, 96))
    assert f"layoutTag(<{parent_name}>)" in mlir
    assert f"layoutTag(<{child_name}>)" in mlir
    assert "!tla.shape<?,?>" in mlir

    input_path = tmp_path / f"layout_pair_{parent_name}_to_{child_name}.mlir"
    input_path.write_text(mlir)
    result = subprocess.run(
        [
            str(_require_hivm_tla_compile()),
            str(input_path),
            "-o",
            "-",
            "--mlir-print-ir-after=tla-lower-tensor-desc",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    desc_lines = [
        line.strip()
        for line in result.stderr.splitlines()
        if " = tla.tensor_desc " in line
    ]
    assert len(desc_lines) >= 4, result.stderr
    child_desc = desc_lines[-1]
    assert child_name in child_desc
    shape, stride, origin, coord = _tensor_desc_metadata(child_desc)
    assert len(shape) == 4
    assert len(stride) == 4
    assert len(origin) == 2
    assert len(coord) == 2

    if child_name == "RowMajor":
        assert shape[:2] == origin
        assert shape[2:] == stride[2:]
        _assert_f32_dynamic_stride_is_32b_aligned(
            result.stderr, extent=shape[1], stride=stride[0]
        )
    elif child_name == "ColumnMajor":
        assert shape[:2] == origin
        assert shape[2:] == stride[2:]
        _assert_f32_dynamic_stride_is_32b_aligned(
            result.stderr, extent=shape[0], stride=stride[1]
        )
    else:
        assert child_name in {"zN", "nZ", "L0Clayout", "zNUnAlign"}


@tla.kernel
def nested_ub_subtile_copy_kernel(mem_in: tla.Tensor, mem_out: tla.Tensor) -> None:
    gm_in = tla.tile_view(mem_in, tla.make_shape(64, 64), tla.make_coord(0, 0))
    gm_out = tla.tile_view(mem_out, tla.make_shape(32, 32), tla.make_coord(0, 0))
    ub_ptr = tla.allocate((64, 64), tla.Float32, tla.AddressSpace.ub, 256)
    ub_root = tla.make_tensor_like(ub_ptr, gm_in, tla.arch.RowMajor)
    ub_tile = tla.tile_view(ub_root, tla.make_shape(32, 32), tla.make_coord(1, 1))
    with tla.vector():
        tla.copy(ub_root, gm_in)
        tla.copy(gm_out, ub_tile)


def test_nested_ub_subtile_copy_lowers(tmp_path) -> None:
    """GM<->UB staging copies lower to vector-core (AIV) cifax runtime calls."""
    tla_compile = _require_hivm_tla_compile()
    mem_in = make_fake_tensor(
        tla.Float32,
        (64, 64),
        (64, 1),
        origin_shape=(64, 64),
        layout_tag=tla.arch.RowMajor,
    )
    mem_out = make_fake_tensor(
        tla.Float32,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=tla.arch.RowMajor,
    )

    mlir = nested_ub_subtile_copy_kernel.dump_mlir(type_args=(mem_in, mem_out))
    input_path = tmp_path / "nested_ub_subtile_copy.mlir"
    input_path.write_text(mlir)

    result = subprocess.run(
        [str(tla_compile), str(input_path), "-o", "-"],
        text=True,
        capture_output=True,
        check=True,
    )
    lowered = result.stdout
    # The kernel issues both a GM->UB and a UB->GM staging copy; each lowers to its
    # own inlinable AIV cifax runtime template (bc/Vector/dma.cpp).
    assert "copy_gm_RowMajor_to_ub_RowMajor_float" in lowered
    assert "copy_ub_RowMajor_to_gm_RowMajor_float" in lowered
    assert "hivm.func_core_type = #hivm.func_core_type<AIV>" in lowered
    assert '"tla.copy"' not in lowered


@tla.kernel
def ptradd_ub_subtile_copy_kernel(mem_in: tla.Tensor, mem_src: tla.Tensor) -> None:
    gm_root = tla.tile_view(mem_in, tla.make_shape(64, 64), tla.make_coord(0, 0))
    gm_src = tla.tile_view(mem_src, tla.make_shape(32, 32), tla.make_coord(0, 0))
    ub_ptr = tla.allocate(64 * 64 + 16, tla.Float32, tla.AddressSpace.ub, 256) + 16
    ub_root = tla.make_tensor_like(ub_ptr, gm_root, tla.arch.RowMajor)
    ub_tile = tla.tile_view(ub_root, tla.make_shape(32, 32), tla.make_coord(1, 1))
    with tla.vector():
        tla.copy(ub_tile, gm_src)


def test_ptradd_ub_subtile_copy_applies_ptr_offset(tmp_path) -> None:
    """The ptr_add offset is preserved in the cifax base pointer_cast and tile payload."""
    tla_compile = _require_hivm_tla_compile()
    mem_in = make_fake_tensor(
        tla.Float32,
        (64, 64),
        (64, 1),
        origin_shape=(64, 64),
        layout_tag=tla.arch.RowMajor,
    )
    mem_src = make_fake_tensor(
        tla.Float32,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=tla.arch.RowMajor,
    )

    mlir = ptradd_ub_subtile_copy_kernel.dump_mlir(type_args=(mem_in, mem_src))
    input_path = tmp_path / "ptradd_ub_subtile.mlir"
    (input_path.write_text(mlir),)
    output_path = tmp_path / "out.mlir"
    result = subprocess.run(
        [
            str(tla_compile),
            str(input_path),
            "-o",
            str(output_path),
            "--mlir-print-ir-after=tla-finalize-memref",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout + result.stderr  # print-ir-after goes to stderr
    # GM->UB staging copy lowers to the AIV cifax runtime template.
    assert "copy_gm_RowMajor_to_ub_RowMajor_float" in out
    assert "hivm.func_core_type = #hivm.func_core_type<AIV>" in out
    # ptr_add contributes 64 bytes (16 f32 elements) to the UB base pointer_cast;
    # the UB sub-tile (coord (1,1) of a 64-wide buffer) carries stride0=64 and
    # coord (32,32) in the i64 payload, so the stub computes the flat offset
    # (32*64+32 = 2080) from coord/stride at runtime.
    assert "arith.constant 64 : i64" in out
    assert '"tla.copy"' not in out


# UB -> UB is not an MTE transfer -- no engine has UB on both sides. It goes
# through the DataCopy meta-op (MOV.UB.TO.UB) for every row length: the move
# counts in whole 32-byte blocks and rounds a ragged row up to one. The frontend
# must accept the route inside a tla.vector region rather than rejecting it as
# unsupported.
def _ub_to_ub_kernel(dtype, cols):
    """A UB->UB copy of a 1 x cols tile, built for one dtype."""

    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src = tla.make_tensor_like(
            tla.allocate((1, cols), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst = tla.make_tensor_like(
            tla.allocate((1, cols), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        with tla.vector():
            tla.copy(dst, src)

    return kernel


# The route is accepted for every registered width. A 64-element row is a whole
# number of 32-byte blocks for each of these dtypes, so it takes the DataCopy
# meta-op path (a tla.copy op).
@pytest.mark.parametrize(
    "dtype",
    [
        tla.Float32,
        tla.Float16,
        tla.BFloat16,
        tla.Int32,
        tla.Int16,
        tla.Int8,
        tla.Float8E4M3FN,
        tla.Float8E5M2,
    ],
)
def test_frontend_copy_ub_to_ub_accepts_every_registered_dtype(dtype) -> None:
    mem = make_fake_tensor(
        dtype, (1, 64), (64, 1), origin_shape=(1, 64), layout_tag=tla.arch.RowMajor
    )
    mlir = _ub_to_ub_kernel(dtype, 64).dump_mlir(type_args=(mem,))
    assert "tla.copy" in mlir


# A byte row that is not a whole number of 32-byte blocks takes the same
# DataCopy meta-op as a block-aligned one: the move rounds the row up to whole
# blocks. Every width is accepted -- b8 included.
@pytest.mark.parametrize("dtype", [tla.Int8, tla.Float8E4M3FN, tla.Float8E5M2])
@pytest.mark.parametrize("cols", [1, 17, 48, 63, 64])
def test_frontend_copy_ub_to_ub_accepts_byte_rows(dtype, cols) -> None:
    mem = make_fake_tensor(
        dtype,
        (1, cols),
        (cols, 1),
        origin_shape=(1, cols),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = _ub_to_ub_kernel(dtype, cols).dump_mlir(type_args=(mem,))
    assert "tla.copy" in mlir


def test_frontend_copy_ub_to_ub_lowers_to_runtime_call() -> None:
    mem = make_fake_tensor(
        tla.Float32,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = copy_ub_to_ub_kernel.dump_mlir(type_args=(mem,))
    assert "tla.copy" in mlir


# An enclosing tla.vec.func region does not change which route a UB->UB copy
# takes: the meta-op is emitted inside one exactly as it is outside, and a ragged
# row is rounded up to whole blocks either way.
def _ub_to_ub_in_vec_func_kernel(dtype, cols):
    """A UB->UB copy of a 1 x cols tile, written inside a tla.vec.func region."""

    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src = tla.make_tensor_like(
            tla.allocate((1, cols), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst = tla.make_tensor_like(
            tla.allocate((1, cols), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        with tla.vector():
            with tla.vec.func(mode="simd"):
                tla.copy(dst, src)

    return kernel


# 48 bytes of f32 (12 elements) and 48 bytes of i8 are both one and a half
# blocks; 17 bytes of fp8 is just over half of one. All of them are copied.
@pytest.mark.parametrize(
    ("dtype", "cols"),
    [
        (tla.Float32, 12),
        (tla.Float16, 24),
        (tla.Int8, 48),
        (tla.Float8E4M3FN, 17),
        (tla.Float32, 8),
        (tla.Int8, 32),
    ],
)
def test_frontend_copy_ub_to_ub_inside_vec_func_takes_the_metaop(dtype, cols) -> None:
    mem = make_fake_tensor(
        dtype,
        (1, cols),
        (cols, 1),
        origin_shape=(1, cols),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = _ub_to_ub_in_vec_func_kernel(dtype, cols).dump_mlir(type_args=(mem,))
    assert "tla.copy" in mlir


# A padded sub-tile with a ragged row is copied as long as the pitch leaves a
# whole number of blocks of slack for the rounded row. A 64x64 f32 parent sliced
# to 64x12 has a 256-byte pitch and a 48-byte row, rounded to 64.
def test_frontend_copy_ub_to_ub_accepts_ragged_subtile_with_padded_pitch() -> None:
    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src_root = tla.make_tensor_like(
            tla.allocate((64, 64), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst_root = tla.make_tensor_like(
            tla.allocate((64, 64), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        src = tla.tile_view(src_root, tla.make_shape(64, 12), tla.make_coord(0, 0))
        dst = tla.tile_view(dst_root, tla.make_shape(64, 12), tla.make_coord(0, 0))
        with tla.vector():
            tla.copy(dst, src)

    mem = make_fake_tensor(
        tla.Float32,
        (64, 64),
        (64, 1),
        origin_shape=(64, 64),
        layout_tag=tla.arch.RowMajor,
    )
    assert "tla.copy" in kernel.dump_mlir(type_args=(mem,))


# The block move addresses whole blocks, so a tile that does not *start* on one
# has no route. A 12-element f32 tile at tile index 1 begins at byte 48.
def test_frontend_copy_ub_to_ub_rejects_unaligned_origin() -> None:
    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src_root = tla.make_tensor_like(
            tla.allocate((64, 64), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst_root = tla.make_tensor_like(
            tla.allocate((64, 64), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        src = tla.tile_view(src_root, tla.make_shape(64, 12), tla.make_coord(0, 1))
        dst = tla.tile_view(dst_root, tla.make_shape(64, 12), tla.make_coord(0, 1))
        with tla.vector():
            tla.copy(dst, src)

    mem = make_fake_tensor(
        tla.Float32,
        (64, 64),
        (64, 1),
        origin_shape=(64, 64),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"source tile starts at byte 48 of its parent",
    ):
        kernel.dump_mlir(type_args=(mem,))


# A multi-row ragged tile is copied, because a UB allocation's row pitch is
# padded to 32 bytes: 4 x 12 f32 is a 48-byte row with a 64-byte pitch, so the
# tail the block move rounds up to lands in that padding rather than on the next
# row. (The frontend still refuses a pitch too narrow to carry it, which is what
# makes the meta-op's gap arithmetic exact, but an allocation cannot produce
# one.)
def test_frontend_copy_ub_to_ub_accepts_ragged_rows_with_padded_pitch() -> None:
    mem = make_fake_tensor(
        tla.Float32,
        (4, 12),
        (12, 1),
        origin_shape=(4, 12),
        layout_tag=tla.arch.RowMajor,
    )

    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src = tla.make_tensor_like(
            tla.allocate((4, 12), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst = tla.make_tensor_like(
            tla.allocate((4, 12), tla.Float32, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        with tla.vector():
            tla.copy(dst, src)

    assert "tla.copy" in kernel.dump_mlir(type_args=(mem,))


# UB -> UB with a ColumnMajor destination is a transposing copy: both tiles keep
# the logical (M, N) shape and the bytes land as (N, M). It is its own route,
# served by the meta-op's VNCHWCONV, so it reaches a runtime call distinct from
# the straight RowMajor->RowMajor one.
def _ub_to_ub_transpose_kernel(dtype, rows, cols):
    """A transposing UB->UB copy of a rows x cols tile, built for one dtype."""

    @tla.kernel
    def kernel(mem_in: tla.Tensor) -> None:
        src = tla.make_tensor_like(
            tla.allocate((rows, cols), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.RowMajor,
        )
        dst = tla.make_tensor_like(
            tla.allocate((cols, rows), dtype, tla.AddressSpace.ub, 256),
            mem_in,
            tla.arch.ColumnMajor,
        )
        with tla.vector():
            tla.copy(dst, src)

    return kernel


@pytest.mark.parametrize(
    "dtype",
    [
        tla.Float32,
        tla.Float16,
        tla.BFloat16,
        tla.Int32,
        tla.Int16,
        tla.Int8,
        tla.Float8E4M3FN,
        tla.Float8E5M2,
    ],
)
def test_frontend_copy_ub_to_ub_transpose_accepts_every_registered_dtype(dtype) -> None:
    mem = make_fake_tensor(
        dtype, (64, 64), (64, 1), origin_shape=(64, 64), layout_tag=tla.arch.RowMajor
    )
    mlir = _ub_to_ub_transpose_kernel(dtype, 64, 64).dump_mlir(type_args=(mem,))
    # Every registered width rides VNCHWCONV, so the op survives to a tla.copy.
    assert "tla.copy" in mlir


def test_frontend_copy_ub_to_ub_transpose_is_its_own_route() -> None:
    """The transposing copy must not be served by the straight route's callee."""
    mem = make_fake_tensor(
        tla.Float32,
        (64, 64),
        (64, 1),
        origin_shape=(64, 64),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = _ub_to_ub_transpose_kernel(tla.Float32, 64, 64).dump_mlir(type_args=(mem,))
    assert "tla.copy" in mlir
    # The layout tags are what separate the two routes, and the destination's
    # ColumnMajor tag is what picks the transposing callee further down.
    assert "ColumnMajor" in mlir


def test_frontend_copy_ub_to_ub_transpose_refuses_a_shape_the_metaop_cannot_do() -> (
    None
):
    """A tile that is not a whole number of VNCHWCONV blocks is refused.

    16 rows by C0 columns is the block, so a 24x40 f16 tile is neither a whole
    number of rows nor of columns and would have the meta-op read and write past
    the tile's edges.
    """
    mem = make_fake_tensor(
        tla.Float16,
        (24, 40),
        (40, 1),
        origin_shape=(24, 40),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(Exception, match="multiple of"):
        _ub_to_ub_transpose_kernel(tla.Float16, 24, 40).dump_mlir(type_args=(mem,))


# --- fixpipe PER_TENSOR (on-the-fly) quant frontend checks ---
def _fp_tensor(dtype: object, layout_tag: object) -> object:
    return make_fake_tensor(
        dtype,
        (32, 32),
        (32, 1),
        origin_shape=(32, 32),
        layout_tag=layout_tag,
    )


@tla.kernel
def copy_l0c_to_gm_pertensor_kernel(gm_c: tla.Tensor) -> None:
    """L0C -> GM with PER_TENSOR quant; dtype-c comes from ``gm_c``."""
    gm_tile = tla.tile_view(gm_c, tla.make_shape(32, 32), tla.make_coord(0, 0))
    l0c = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 512),
        gm_tile,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.copy(
            gm_tile,
            l0c,
            tla.params.CopyL0C2DstParams(
                quant_mode=tla.params.QuantMode.PER_TENSOR,
                quant_scale=1.0,
            ),
        )


@tla.kernel
def copy_l0c_to_gm_pertensor_no_scale_kernel(gm_c: tla.Tensor) -> None:
    """PER_TENSOR without a quant_scale operand."""
    gm_tile = tla.tile_view(gm_c, tla.make_shape(32, 32), tla.make_coord(0, 0))
    l0c = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 512),
        gm_tile,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.copy(
            gm_tile,
            l0c,
            tla.params.CopyL0C2DstParams(
                quant_mode=tla.params.QuantMode.PER_TENSOR,
            ),
        )


def test_frontend_accepts_l0c_to_gm_pertensor_f16() -> None:
    """f32 accumulator -> f16 GM with PER_TENSOR quant is a supported route."""
    gm = _fp_tensor(tla.Float16, tla.arch.RowMajor)
    mlir = copy_l0c_to_gm_pertensor_kernel.dump_mlir(type_args=(gm,))
    assert "quant_mode = #tla.quant_mode<PER_TENSOR>" in mlir


def test_frontend_accepts_l0c_to_gm_pertensor_int8() -> None:
    """f32 accumulator -> int8 GM is admitted by the PER_TENSOR route table."""
    gm = _fp_tensor(tla.Int8, tla.arch.RowMajor)
    mlir = copy_l0c_to_gm_pertensor_kernel.dump_mlir(type_args=(gm,))
    assert "quant_mode = #tla.quant_mode<PER_TENSOR>" in mlir


def test_frontend_rejects_pertensor_without_quant_scale() -> None:
    """PER_TENSOR requires a quant_scale operand."""
    gm = _fp_tensor(tla.Float16, tla.arch.RowMajor)
    with pytest.raises(
        TlaLoweringError,
        match=r"quant_scale is required when quant_mode is PER_TENSOR",
    ):
        copy_l0c_to_gm_pertensor_no_scale_kernel.dump_mlir(type_args=(gm,))


def test_frontend_rejects_pertensor_unsupported_dst_dtype() -> None:
    """fp8 is not a PER_TENSOR quant destination in the fixpipe route table.

    e5m2 is the encoding to test: it has no quant route in any quant mode, so
    this rejection holds even if e4m3fn is enabled later.
    """
    gm = _fp_tensor(tla.Float8E5M2, tla.arch.RowMajor)
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"unsupported fixpipe route",
    ):
        copy_l0c_to_gm_pertensor_kernel.dump_mlir(type_args=(gm,))


@tla.kernel
def copy_l0c_to_gm_pertensor_offset_kernel(gm_c: tla.Tensor) -> None:
    """PER_TENSOR quant with a runtime-style int quant_offset."""
    gm_tile = tla.tile_view(gm_c, tla.make_shape(32, 32), tla.make_coord(0, 0))
    l0c = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 512),
        gm_tile,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.copy(
            gm_tile,
            l0c,
            tla.params.CopyL0C2DstParams(
                quant_mode=tla.params.QuantMode.PER_TENSOR,
                quant_scale=1.0,
                quant_offset=3,
            ),
        )


def test_frontend_accepts_pertensor_int_offset() -> None:
    """quant_offset accepts a Python int (8-bit destinations only)."""
    gm = _fp_tensor(tla.Int8, tla.arch.RowMajor)
    mlir = copy_l0c_to_gm_pertensor_offset_kernel.dump_mlir(type_args=(gm,))
    assert "quant_mode = #tla.quant_mode<PER_TENSOR>" in mlir


@tla.kernel
def copy_l0c_to_gm_pertensor_i32_offset_kernel(gm_c: tla.Tensor) -> None:
    """PER_TENSOR quant with an explicit Int32 quant_offset."""
    gm_tile = tla.tile_view(gm_c, tla.make_shape(32, 32), tla.make_coord(0, 0))
    l0c = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 512),
        gm_tile,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.copy(
            gm_tile,
            l0c,
            tla.params.CopyL0C2DstParams(
                quant_mode=tla.params.QuantMode.PER_TENSOR,
                quant_scale=1.0,
                quant_offset=tla.Int32(3),
            ),
        )


def test_frontend_accepts_pertensor_i32_offset() -> None:
    """An explicit Int32 is passed through as the operand's own type."""
    gm = _fp_tensor(tla.Int8, tla.arch.RowMajor)
    mlir = copy_l0c_to_gm_pertensor_i32_offset_kernel.dump_mlir(type_args=(gm,))
    assert "i32" in mlir
    assert "quant_mode = #tla.quant_mode<PER_TENSOR>" in mlir


def _pertensor_offset_kernel(offset: object):
    """Build a kernel whose quant_offset is ``offset``.

    The operand is a fixed I32, so anything else has to be rejected by the
    frontend; letting it through only moves the failure to the verifier, which
    reports "operand #1 must be 32-bit signless integer" with no hint that the
    offset was the problem.
    """

    @tla.kernel
    def kernel(gm_c: tla.Tensor) -> None:
        gm_tile = tla.tile_view(gm_c, tla.make_shape(32, 32), tla.make_coord(0, 0))
        l0c = tla.make_tensor_like(
            tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 512),
            gm_tile,
            tla.arch.L0Clayout,
        )
        with tla.cube():
            tla.copy(
                gm_tile,
                l0c,
                tla.params.CopyL0C2DstParams(
                    quant_mode=tla.params.QuantMode.PER_TENSOR,
                    quant_scale=1.0,
                    quant_offset=offset,
                ),
            )

    return kernel


@pytest.mark.parametrize(
    ("label", "offset"),
    [
        ("int16", tla.Int16(3)),
        ("int64", tla.Int64(3)),
        ("uint32", tla.UInt32(3)),
        ("bool", True),
    ],
)
def test_frontend_rejects_pertensor_offset_that_is_not_i32(
    label: str, offset: object
) -> None:
    """A non-I32 quant_offset is refused before the IR reaches the verifier."""
    gm = _fp_tensor(tla.Int8, tla.arch.RowMajor)
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"quant_offset.*expected int or Int32",
    ):
        _pertensor_offset_kernel(offset).dump_mlir(type_args=(gm,))


def _perchannel_quant_kernel():
    """Build a PER_CHANNEL kernel that hands ``quant`` to the fixpipe as-is.

    The element type, address space and layout of ``quant`` all come from the
    host tensor bound at ``type_args``; the frontend checks that triple where
    the copy's parameters are built.
    """

    @tla.kernel
    def kernel(gm_c: tla.Tensor, quant: tla.Tensor) -> None:
        gm_tile = tla.tile_view(gm_c, tla.make_shape(32, 32), tla.make_coord(0, 0))
        l0c = tla.make_tensor_like(
            tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 512),
            gm_tile,
            tla.arch.L0Clayout,
        )
        with tla.cube():
            tla.copy(
                gm_tile,
                l0c,
                tla.params.CopyL0C2DstParams(
                    quant_mode=tla.params.QuantMode.PER_CHANNEL,
                    quant_tensor=quant,
                ),
            )

    return kernel


def _quant_tensor(
    dtype: object,
    *,
    addrspace: object = tla.AddressSpace.l1,
    layout_tag: object = tla.arch.VectorLayout,
    stride: object = 1,
    origin_shape: object = (32,),
) -> object:
    """A host-side sample of the flat L1 dequant run the fixpipe reads."""
    return make_fake_tensor(
        dtype,
        origin_shape,
        (stride,),
        origin_shape=origin_shape,
        layout_tag=layout_tag,
        addrspace=addrspace,
    )


def test_frontend_accepts_perchannel_u64_quant_tensor() -> None:
    """The packed dequant elements are u64; that is the accepted element type."""
    gm = _fp_tensor(tla.Int8, tla.arch.RowMajor)
    mlir = _perchannel_quant_kernel().dump_mlir(
        type_args=(gm, _quant_tensor(tla.UInt64))
    )
    assert "quant_mode = #tla.quant_mode<PER_CHANNEL>" in mlir


@pytest.mark.parametrize(
    ("label", "quant_len"),
    [
        ("short", 4),
        ("long", 40),
    ],
)
def test_frontend_rejects_perchannel_quant_run_not_matching_static_channels(
    label: str, quant_len: int
) -> None:
    """A run whose static length differs from the destination's static N is
    refused, whichever side is longer.

    The destination tile is the fixed 32x32 sample, so a run of 4 or 40
    elements cannot line up with its 32 channels. The check needs both
    lengths static; the dynamic pass-through is covered separately.
    """
    gm = _fp_tensor(tla.Int8, tla.arch.RowMajor)
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=(
            rf"quant_tensor.*holds {quant_len} elements but the destination "
            r"tile spans 32 output channels; the run must match the "
            r"destination's channel count exactly"
        ),
    ):
        _perchannel_quant_kernel().dump_mlir(
            type_args=(
                gm,
                _quant_tensor(tla.UInt64, origin_shape=(quant_len,)),
            )
        )


@pytest.mark.parametrize(
    ("label", "dtype"),
    [
        ("u32", tla.UInt32),
        ("i64", tla.Int64),
        ("f32", tla.Float32),
        ("u8", tla.UInt8),
    ],
)
def test_frontend_rejects_perchannel_quant_tensor_that_is_not_u64(
    label: str, dtype: object
) -> None:
    """A quant tile of any other width is refused, naming the argument."""
    gm = _fp_tensor(tla.Int8, tla.arch.RowMajor)
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"quant_tensor.*expected a u64 tile",
    ):
        _perchannel_quant_kernel().dump_mlir(type_args=(gm, _quant_tensor(dtype)))


@pytest.mark.parametrize(
    ("label", "addrspace", "layout_tag", "message"),
    [
        (
            "gm-vectorlayout",
            tla.AddressSpace.gm,
            tla.arch.VectorLayout,
            r"must be staged in l1",
        ),
        (
            "l1-rowmajor",
            tla.AddressSpace.l1,
            tla.arch.RowMajor,
            r"must be a flat VectorLayout run",
        ),
    ],
)
def test_frontend_rejects_perchannel_quant_tensor_outside_the_l1_run(
    label: str, addrspace: object, layout_tag: object, message: str
) -> None:
    """u64 alone is not the contract: the runtime reads an L1 VectorLayout run.

    Both shapes below compile through bc -- the wrapper builds the operand as an
    L1 VectorLayout tile regardless -- so a GM or two-dimensional scale tile has
    to be refused where it is passed, not left to read the wrong memory.
    """
    gm = _fp_tensor(tla.Int8, tla.arch.RowMajor)
    with pytest.raises(runtime_mod.TlaCoreAPIError, match=message):
        _perchannel_quant_kernel().dump_mlir(
            type_args=(
                gm,
                _quant_tensor(
                    tla.UInt64, addrspace=addrspace, layout_tag=layout_tag
                ),
            )
        )


def test_frontend_rejects_perchannel_quant_tensor_that_is_not_unit_stride() -> None:
    """The device side builds the run with a literal stride of 1.

    `makeVectorTlaLayout` hardcodes it, so a strided run would be read
    contiguously and dequantize against the wrong elements. `make_tensor` already
    refuses a strided VectorLayout tile; a tensor bound from the host does not go
    through it, so the copy checks again.
    """
    gm = _fp_tensor(tla.Int8, tla.arch.RowMajor)
    with pytest.raises(
        runtime_mod.TlaCoreAPIError,
        match=r"quant_tensor.*must be a unit-stride run",
    ):
        _perchannel_quant_kernel().dump_mlir(
            type_args=(gm, _quant_tensor(tla.UInt64, stride=2))
        )


def _u64_tile_kernel(addrspace: object, layout_tag: object):
    """Build a u64 tile in ``addrspace`` with ``layout_tag``."""

    @tla.kernel
    def kernel() -> None:
        ptr = tla.allocate(32, tla.UInt64, addrspace, 512)
        tla.make_tensor(
            ptr,
            tla.make_layout(
                tla.make_shape(32), tla.make_stride(1), layoutTag=layout_tag
            ),
        )

    return kernel


def test_frontend_accepts_u64_l1_vectorlayout_tile() -> None:
    """The per-channel dequant run is the one u64 tile the frontend admits."""
    _u64_tile_kernel(tla.AddressSpace.l1, tla.arch.VectorLayout).dump_mlir()


@pytest.mark.parametrize(
    ("label", "addrspace", "layout_tag"),
    [
        ("ub", tla.AddressSpace.ub, tla.arch.VectorLayout),
        ("l1-rowmajor", tla.AddressSpace.l1, tla.arch.RowMajor),
    ],
)
def test_frontend_rejects_u64_tile_outside_the_dequant_run(
    label: str, addrspace: object, layout_tag: object
) -> None:
    """u64 has exactly one copy route, so any other tile is refused at build time.

    Admitting u64 globally would let a UB or row-major L1 u64 tile through the
    frontend, to fail much later in bc where no wrapper exists.
    """
    with pytest.raises(
        TlaLoweringError,
        match=r"u64 is only accepted as the L1 VectorLayout tile",
    ):
        _u64_tile_kernel(addrspace, layout_tag).dump_mlir()
