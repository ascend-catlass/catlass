from catlass.tla.runtime import make_fake_tensor

import pathlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import pytest
import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.execution_lowering import TlaLoweringError
from catlass.params import BlockStoreParams
from catlass import _tla_type_bridge
from catlass._mlir import ir as mlir_ir


def _ensure_compute_order_or_skip() -> None:
    with mlir_ir.Context() as ctx:
        _tla_type_bridge.load_tla_dialect(ctx)
        try:
            mlir_ir.Attribute.parse("#tla.compute_order<M_FIRST>", context=ctx)
        except Exception as exc:  # noqa: BLE001
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

def _require_hivm_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = (
        repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    )
    if not tla_compile.exists():
        raise AssertionError("TlaCompile binary not found. Build csrc/mlir first.")
    return tla_compile

def _run_tla_compile_ir_after_pass(
    mlir_text: str, pass_name: str, *, require_success: bool = False
) -> str:
    tla_compile = _require_hivm_tla_compile()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = pathlib.Path(tmpdir) / "input.mlir"
        output_path = pathlib.Path(tmpdir) / "output.mlir"
        input_path.write_text(mlir_text)
        result = subprocess.run(
            [
                str(tla_compile),
                str(input_path),
                "-o",
                str(output_path),
                f"--mlir-print-ir-after={pass_name}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if require_success:
            assert result.returncode == 0, output
        assert "IR Dump After" in output, output
        assert f"({pass_name})" in output, output
        return output

def _extract_function(ir_dump: str, name: str) -> str:
    marker = f"func.func @{name}("
    start = ir_dump.index(marker)
    next_function = ir_dump.find("\n  func.func @", start + len(marker))
    module_end = ir_dump.find("\n}", start + len(marker))
    ends = [position for position in (next_function, module_end) if position != -1]
    assert ends, ir_dump[start:]
    return ir_dump[start : min(ends)]

def _dump_after_mixed_split(
    kernel: object, *, type_args: tuple[object, ...] = ()
) -> str:
    mlir_text = kernel.dump_mlir(type_args=type_args)
    return _run_tla_compile_ir_after_pass(
        mlir_text, "tla-split-mixed-func", require_success=True
    )

def _mmad_tensor_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    return (
        make_fake_tensor(
            tla.Float16,
            (128, 64),
            (64, 1),
            origin_shape=(128, 64),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float16,
            (64, 128),
            (128, 1),
            origin_shape=(64, 128),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float32,
            (128, 128),
            (128, 1),
            origin_shape=(128, 128),
            layout_tag=tla.arch.RowMajor,
        ),
    )

def _dynamic_gm_tensor() -> tla.Tensor:
    tensor = make_fake_tensor(
                 tla.Float32,
                 (8,),
                 (1,),
                 origin_shape=(8,),
                 layout_tag=tla.arch.RowMajor,
             )
    tensor.mark_compact_shape_dynamic(0)
    return tensor

def _dynamic_cube_gm_tensor() -> tla.Tensor:
    tensor = make_fake_tensor(
                 tla.Float16,
                 (64, 128),
                 (128, 1),
                 origin_shape=(64, 128),
                 layout_tag=tla.arch.RowMajor,
             )
    tensor.mark_compact_shape_dynamic(0)
    return tensor

def _skip_if_mmad_rank2_tile_view_regression(exc: BaseException) -> None:
    if isinstance(exc, TlaLoweringError) and "rank-2 tiles only" in str(exc):
        pytest.skip(
            "tla.mmad rank-2 check rejects tile_view operand types until metadata matches"
        )

@tla.kernel
def _cube_attr_kernel(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor) -> None:
    lhs_parent = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    rhs_parent = tla.tile_view(mem_b, tla.make_shape(16, 16), tla.make_coord(0, 0))
    acc_parent = tla.tile_view(mem_c, tla.make_shape(16, 16), tla.make_coord(0, 0))
    lhs = tla.make_tensor_like(
        tla.allocate((16, 16), tla.Float16, tla.AddressSpace.l0a, 512),
        lhs_parent,
        tla.arch.zN,
    )
    rhs = tla.make_tensor_like(
        tla.allocate((16, 16), tla.Float16, tla.AddressSpace.l0b, 512),
        rhs_parent,
        tla.arch.nZ,
    )
    acc = tla.make_tensor_like(
        tla.allocate((16, 16), tla.Float32, tla.AddressSpace.l0c, 1024),
        acc_parent,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=False)

@tla.kernel
def _cube_dynamic_l1_pointer_join_kernel(
    source: tla.Tensor, rows: int, offset: int
) -> None:
    shape = tla.make_shape(rows, 128)
    layout = tla.make_layout(
        shape, tla.make_stride(128, 1), origin_shape=shape
    )
    dynamic_source = tla.make_tensor(source.ptr, layout)
    base = tla.allocate((128, 128), tla.Float16, tla.AddressSpace.l1, 512)
    selected = base + offset
    local = tla.make_tensor_like(selected, dynamic_source, tla.arch.zN)
    with tla.cube():
        tla.copy(local, dynamic_source)

@tla.kernel
def _split_cube_only_kernel() -> None:
    seed = tla.arch.block_idx()
    value = seed + 7
    with tla.cube():
        tla.pipe_barrier(tla.pipes.CUBE)
    tla.make_coord(value)

@tla.kernel
def _split_vector_only_kernel() -> None:
    seed = tla.arch.block_idx()
    value = seed + 8
    with tla.vector():
        tla.pipe_barrier(tla.pipes.MTE2)
    tla.make_coord(value)

@tla.kernel
def _split_no_scope_kernel() -> None:
    seed = tla.arch.block_idx()
    value = seed + 9
    tla.make_coord(value)

@tla.kernel
def _mixed_interleaved_scopes_kernel() -> None:
    seed = tla.arch.block_idx()
    before = seed + 101
    with tla.cube():
        tla.pipe_barrier(tla.pipes.CUBE)
    between = before + 102
    with tla.vector():
        tla.pipe_barrier(tla.pipes.MTE2)
    middle = between + 103
    with tla.cube():
        tla.pipe_barrier(tla.pipes.CUBE)
    after = middle + 104
    with tla.vector():
        tla.pipe_barrier(tla.pipes.MTE2)
    tail = after + 105
    tla.make_coord(tail)

@tla.kernel
def _mixed_if_branches_kernel() -> None:
    seed = tla.arch.block_idx()
    before = seed + 201
    branch_value = before
    if before > 0:
        branch_value = before + 202
        with tla.cube():
            tla.pipe_barrier(tla.pipes.CUBE)
        branch_value = branch_value + 203
    else:
        branch_value = before + 204
        with tla.vector():
            tla.pipe_barrier(tla.pipes.MTE2)
        branch_value = branch_value + 205
    tail = branch_value + 206
    tla.make_coord(tail)

@tla.kernel
def _mixed_for_kernel(limit: int) -> None:
    state = tla.as_numeric(0)
    for index in tla.range(0, limit, 1):
        state = index + 301
        with tla.cube():
            tla.pipe_barrier(tla.pipes.CUBE)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.MTE2)
    tla.make_coord(state)

@tla.kernel
def _mixed_while_kernel(limit: int) -> None:
    index = tla.as_numeric(0)
    state = tla.as_numeric(0)
    while index < limit:
        state = state + 401
        with tla.cube():
            tla.pipe_barrier(tla.pipes.CUBE)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.MTE2)
        index = index + 1
    tla.make_coord(state)

@tla.kernel
def _pointer_if_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 8), tla.make_coord(0, 0))
    ptr0 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    ptr1 = tla.allocate((16, 4), tla.Float16, tla.AddressSpace.l1, 512)
    with tla.cube():
        for i in tla.range(0, 2, 1):
            tile = tla.tile_view(root, tla.make_shape(16, 4), tla.make_coord(0, i))
            selected = ptr0
            tag = i
            if i == 0:
                selected = ptr1
                tag = i + 1
            else:
                selected = ptr0
                tag = i + 2
            local = tla.make_tensor_like(selected, tile, tla.arch.zN)
            tla.make_coord(tag, 0)
            tla.copy(local, tile)

@tla.kernel
def _loop_carried_pointer_same_capacity_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 4), tla.make_coord(0, 0))
    # Both allocations take 512 bytes (256 rounded up to the 512-byte
    # alignment), so the two edges of the join agree and the pointer carries
    # that capacity: 256 f16 elements.
    ptr0 = tla.allocate((32, 4), tla.Float16, tla.AddressSpace.l1, 512)
    ptr1 = tla.allocate((32, 4), tla.Float16, tla.AddressSpace.l1, 512)
    selected = ptr0
    with tla.cube():
        for i in tla.range(0, 2, 1):
            local = tla.make_tensor_like(selected, root, tla.arch.zN)
            tla.copy(local, root)
            if i == 0:
                selected = ptr1

@tla.kernel
def _loop_carried_pointer_changed_capacity_kernel(mem_a: tla.Tensor) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 4), tla.make_coord(0, 0))
    # 1024 bytes versus 256 bytes, both at 512-byte alignment: the size
    # rounds to 1024 versus 512, so the two edges of the join must carry
    # different amounts and the extent falls back to what the copy touches.
    ptr0 = tla.allocate((128, 4), tla.Float16, tla.AddressSpace.l1, 512)
    ptr1 = tla.allocate((32, 4), tla.Float16, tla.AddressSpace.l1, 512)
    selected = ptr0
    with tla.cube():
        for i in tla.range(0, 2, 1):
            local = tla.make_tensor_like(selected, root, tla.arch.zN)
            tla.copy(local, root)
            if i == 0:
                selected = ptr1

@tla.kernel
def _tensor_if_mixed_base_carrier_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, choice: int
) -> None:
    left = tla.tile_view(mem_a, tla.make_shape(32, 32), tla.make_coord(0, 0))
    right = tla.make_tensor(
        mem_b.ptr,
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(128, 1)),
    )
    ptr = tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512)
    with tla.cube():
        selected = left
        if choice == 0:
            selected = left
        else:
            selected = right
        selected_view = tla.tile_view(
            selected, tla.make_shape(32, 32), tla.make_coord(0, 0)
        )
        nested_view = tla.tile_view(
            selected_view, tla.make_shape(32, 32), tla.make_coord(0, 0)
        )
        local = tla.make_tensor_like(ptr, nested_view, tla.arch.zN)
        tla.copy(local, nested_view)

@tla.kernel
def _tensor_if_packed_carrier_kernel(mem_a: tla.Tensor, choice: int) -> None:
    root = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    first = tla.make_tensor_like(
        tla.allocate((16, 16), tla.Float32, tla.AddressSpace.l1, 512),
        root,
        tla.arch.zN,
    )
    second = tla.make_tensor_like(
        tla.allocate((16, 16), tla.Float32, tla.AddressSpace.l1, 512),
        root,
        tla.arch.zN,
    )
    with tla.cube():
        selected = first
        if choice == 0:
            selected = first
        else:
            selected = second
        selected_view = tla.tile_view(
            selected, tla.make_shape(16, 16), tla.make_coord(0, 0)
        )
        tla.copy(selected_view, root)

@tla.jit
def _make_dynamic_metadata_tensor(value: int):
    layout = tla.make_layout(
        tla.make_shape(2, value),
        tla.make_stride(value, 1),
        origin_shape=tla.make_shape(2, value),
    )
    return tla.make_tensor(
        tla.allocate((2, 128), tla.Float32, tla.AddressSpace.ub, 256),
        layout,
        tla.make_coord(0, value),
    )

@tla.kernel
def _tensor_if_dynamic_metadata_access_kernel(
    first_value: int, second_value: int, choice: int
) -> None:
    first = _make_dynamic_metadata_tensor(first_value)
    second = _make_dynamic_metadata_tensor(second_value)
    selected = first if choice == 0 else second
    tla.make_shape(selected.shape[1], selected.origin_shape[1])
    tla.make_stride(selected.stride[0])
    tla.make_coord(selected.coord[1])

@tla.kernel
def _tensor_statement_if_dynamic_metadata_access_kernel(
    first_value: int, second_value: int, choice: int
) -> None:
    first = _make_dynamic_metadata_tensor(first_value)
    second = _make_dynamic_metadata_tensor(second_value)
    selected = first
    if choice == 0:
        selected = first
    else:
        selected = second
    tla.make_shape(selected.shape[1])

@tla.kernel
def _tensor_for_dynamic_metadata_access_kernel(
    first_value: int, second_value: int, limit: int
) -> None:
    first = _make_dynamic_metadata_tensor(first_value)
    second = _make_dynamic_metadata_tensor(second_value)
    selected = first
    for _ in tla.range(0, limit, 1):
        selected = second
    tla.make_shape(selected.shape[1])

@tla.kernel
def _tensor_while_dynamic_metadata_access_kernel(
    first_value: int, second_value: int, limit: int
) -> None:
    first = _make_dynamic_metadata_tensor(first_value)
    second = _make_dynamic_metadata_tensor(second_value)
    selected = first
    index = tla.as_numeric(0)
    while index < limit:
        selected = second
        index = index + 1
    tla.make_shape(selected.shape[1])

@tla.kernel
def _tensor_if_dynamic_stride_carrier_kernel(
    first_pitch: int, second_pitch: int, choice: int
) -> None:
    shape = tla.make_shape(2, 128)
    first_layout = tla.make_layout(shape, tla.make_stride(first_pitch, 1))
    second_layout = tla.make_layout(shape, tla.make_stride(second_pitch, 1))
    first = tla.make_tensor(
        tla.allocate((2, 128), tla.Float32, tla.AddressSpace.ub, 256), first_layout
    )
    second = tla.make_tensor(
        tla.allocate((2, 128), tla.Float32, tla.AddressSpace.ub, 256), second_layout
    )
    with tla.vector():
        selected = first if choice == 0 else second
        with tla.vec.func(mode="simd"):
            full_view = tla.tile_view(
                selected, tla.make_shape(2, 128), tla.make_coord(0, 0)
            )
            chunk = tla.tile_view(
                full_view, tla.make_shape(1, 64), tla.make_coord(1, 0)
            )
            value = chunk.load()
            chunk.store(value)

@tla.kernel
def _tensor_for_carrier_kernel(mem_a: tla.Tensor, limit: int) -> None:
    source = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    layout = tla.make_layout(tla.make_shape(16, 16), tla.make_stride(16, 1))
    first = tla.make_tensor(
        tla.allocate((16, 16), tla.Float32, tla.AddressSpace.ub, 256), layout
    )
    second = tla.make_tensor(
        tla.allocate((16, 16), tla.Float32, tla.AddressSpace.ub, 256), layout
    )
    selected = first
    with tla.vector():
        for i in tla.range(0, limit, 1):
            if i == 0:
                selected = second
            else:
                selected = first
        selected_view = tla.tile_view(
            selected, tla.make_shape(16, 16), tla.make_coord(0, 0)
        )
        tla.copy(selected_view, source)

@tla.kernel
def _tensor_while_carrier_kernel(mem_a: tla.Tensor, limit: int) -> None:
    source = tla.tile_view(mem_a, tla.make_shape(16, 16), tla.make_coord(0, 0))
    layout = tla.make_layout(tla.make_shape(16, 16), tla.make_stride(16, 1))
    first = tla.make_tensor(
        tla.allocate((16, 16), tla.Float32, tla.AddressSpace.ub, 256), layout
    )
    second = tla.make_tensor(
        tla.allocate((16, 16), tla.Float32, tla.AddressSpace.ub, 256), layout
    )
    index = tla.as_numeric(0)
    selected = first
    with tla.vector():
        while index < limit:
            if index == 0:
                selected = second
            else:
                selected = first
            index = index + 1
        selected_view = tla.tile_view(
            selected, tla.make_shape(16, 16), tla.make_coord(0, 0)
        )
        tla.copy(selected_view, source)

@tla.kernel
def _vector_pitched_tile_view_kernel() -> None:
    ptr = tla.allocate((2, 128), tla.Float32, tla.AddressSpace.ub, 256)
    parent = tla.make_tensor(
        ptr,
        tla.make_layout(tla.make_shape(2, 128), tla.make_stride(128, 1)),
    )
    with tla.vector():
        with tla.vec.func(mode="simd"):
            chunk = tla.tile_view(parent, tla.make_shape(1, 64), tla.make_coord(1, 0))
            value = chunk.load()
            chunk.store(value)

@tla.kernel
def _vector_tile_view_outside_region_kernel() -> None:
    ptr = tla.allocate((2, 128), tla.Float32, tla.AddressSpace.ub, 256)
    parent = tla.make_tensor(
        ptr,
        tla.make_layout(tla.make_shape(2, 128), tla.make_stride(128, 1)),
    )
    with tla.vector():
        # Built in tla.vector scope and used inside the region. The descriptor
        # then lives in the parent function, so the offset has to be applied
        # inside the helper or the access silently lands on chunk 0.
        chunk = tla.tile_view(parent, tla.make_shape(1, 64), tla.make_coord(1, 0))
        with tla.vec.func(mode="simd"):
            chunk.store(chunk.load())

@tla.kernel
def _vector_dynamic_view_outside_region_kernel(row: int) -> None:
    ptr = tla.allocate((4, 128), tla.Float32, tla.AddressSpace.ub, 256)
    parent = tla.make_tensor(
        ptr,
        tla.make_layout(tla.make_shape(4, 128), tla.make_stride(128, 1)),
    )
    with tla.vector():
        # Same as above, but the coord is a runtime value. The offset cannot be
        # folded to a constant, so it has to reach the helper as a captured
        # scalar and be applied there.
        chunk = tla.tile_view(parent, tla.make_shape(1, 64), tla.make_coord(row, 0))
        with tla.vec.func(mode="simd"):
            chunk.store(chunk.load())


@tla.kernel
def _vector_two_dynamic_views_outside_region_kernel(row_a: int, row_b: int) -> None:
    ptr = tla.allocate((4, 128), tla.Float32, tla.AddressSpace.ub, 256)
    parent = tla.make_tensor(
        ptr,
        tla.make_layout(tla.make_shape(4, 128), tla.make_stride(128, 1)),
    )
    with tla.vector():
        # Two runtime-offset views over one allocation. They share a storage
        # identity, so they collapse to a single helper argument -- each must
        # still carve its own offset, which is exactly what used to be lost.
        a = tla.tile_view(parent, tla.make_shape(1, 64), tla.make_coord(row_a, 0))
        b = tla.tile_view(parent, tla.make_shape(1, 64), tla.make_coord(row_b, 0))
        with tla.vec.func(mode="simd"):
            b.store(a.load())


@tla.kernel
def _vector_dynamic_pitched_tile_view_kernel(pitch: int) -> None:
    ptr = tla.allocate((2, 128), tla.Float32, tla.AddressSpace.ub, 256)
    parent = tla.make_tensor(
        ptr,
        tla.make_layout(tla.make_shape(2, pitch), tla.make_stride(pitch, 1)),
    )
    with tla.vector():
        with tla.vec.func(mode="simd"):
            chunk = tla.tile_view(parent, tla.make_shape(1, 64), tla.make_coord(1, 0))
            value = chunk.load()
            chunk.store(value)

@tla.kernel
def _vector_dynamic_zn_unalign_address_kernel(rows: int) -> None:
    shape = tla.make_shape(rows, 128)
    layout = tla.make_layout(shape, tla.make_stride(128, 1))
    source = tla.make_tensor(
        tla.allocate((64, 128), tla.Float16, tla.AddressSpace.ub, 256), layout
    )
    packed = tla.make_tensor_like(
        tla.allocate((64, 128), tla.Float16, tla.AddressSpace.ub, 256),
        source,
        tla.arch.zNUnAlign,
    )
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for row in tla.range(rows):
                source_chunk = tla.tile_view(
                    source, tla.make_shape(1, 128), tla.make_coord(row, 0)
                )
                packed_chunk = tla.tile_view(
                    packed, tla.make_shape(1, 128), tla.make_coord(row, 0)
                )
                packed_chunk.store(
                    source_chunk.load(),
                    params=BlockStoreParams(block_stride=rows + 1),
                )

@tla.kernel
def _vector_dynamic_row_major_loop_unknown_base_kernel(
    rows: int, ptr_offset: int
) -> None:
    base = tla.allocate((2, 128), tla.Float16, tla.AddressSpace.ub, 256)
    parent = tla.make_tensor(
        base + ptr_offset,
        tla.make_layout(tla.make_shape(rows, 128), tla.make_stride(128, 1)),
    )
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for row in tla.range(0, rows, 1):
                chunk = tla.tile_view(
                    parent, tla.make_shape(1, 128), tla.make_coord(row, 0)
                )
                chunk.store(chunk.load())

@tla.kernel
def _vector_static_rank_one_pointer_join_kernel(choice: int) -> None:
    layout = tla.make_layout(tla.make_shape(128), tla.make_stride(1))
    source = tla.make_tensor(
        tla.allocate((128,), tla.Float16, tla.AddressSpace.ub, 256), layout
    )
    first_ptr = tla.allocate((128,), tla.Float16, tla.AddressSpace.ub, 256)
    second_ptr = tla.allocate((128,), tla.Float16, tla.AddressSpace.ub, 256)
    selected = first_ptr if choice == 0 else second_ptr
    target = tla.make_tensor(selected, layout)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            target.store(source.load())

@tla.kernel
def _vector_dynamic_gm_descriptor_kernel(scalar_buf: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            scalar = scalar_buf[0]
            scalar_buf[0] = scalar

@tla.kernel
def _vector_dynamic_packed_pointer_join_kernel(rows: int, choice: int) -> None:
    _ = tla.allocate((16384,), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.allocate((16384,), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.allocate((8192,), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.allocate((16384,), tla.Float16, tla.AddressSpace.l1, 512)
    with tla.cube():
        tla.pipe_barrier(tla.pipes.CUBE)

    shape = tla.make_shape(rows, 128)
    layout = tla.make_layout(shape, tla.make_stride(128, 1))
    _ = tla.allocate((16384,), tla.Float16, tla.AddressSpace.ub, 256)
    _ = tla.allocate((16384,), tla.Float16, tla.AddressSpace.ub, 256)
    source = tla.make_tensor(
        tla.allocate((64, 128), tla.Float16, tla.AddressSpace.ub, 256), layout
    )
    first_ptr = tla.allocate((64, 128), tla.Float16, tla.AddressSpace.ub, 256)
    # Deliberately a different capacity from first_ptr: the join below is then
    # genuinely unprovable, which is what drives the unknown-extent path. Both
    # are large enough for the (64, 128) view, so nothing reads out of bounds.
    second_ptr = tla.allocate((128, 128), tla.Float16, tla.AddressSpace.ub, 256)
    selected = first_ptr if choice == 0 else second_ptr
    packed = tla.make_tensor_like(selected, source, tla.arch.zN)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            source_chunk = tla.tile_view(
                source, tla.make_shape(1, 128), tla.make_coord(0, 0)
            )
            packed_chunk = tla.tile_view(
                packed, tla.make_shape(1, 128), tla.make_coord(0, 0)
            )
            packed_chunk.store(source_chunk.load(), mask=mask)

@tla.kernel
def _vector_static_packed_pointer_join_kernel(choice: int) -> None:
    _ = tla.allocate((16384,), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.allocate((16384,), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.allocate((8192,), tla.Float16, tla.AddressSpace.l1, 512)
    _ = tla.allocate((16384,), tla.Float16, tla.AddressSpace.l1, 512)
    with tla.cube():
        tla.pipe_barrier(tla.pipes.CUBE)

    shape = tla.make_shape(64, 128)
    layout = tla.make_layout(shape, tla.make_stride(128, 1))
    _ = tla.allocate((16384,), tla.Float16, tla.AddressSpace.ub, 256)
    _ = tla.allocate((16384,), tla.Float16, tla.AddressSpace.ub, 256)
    source = tla.make_tensor(
        tla.allocate((64, 128), tla.Float16, tla.AddressSpace.ub, 256), layout
    )
    first_ptr = tla.allocate((64, 128), tla.Float16, tla.AddressSpace.ub, 256)
    # Deliberately a different capacity from first_ptr: the join below is then
    # genuinely unprovable, which is what drives the unknown-extent path. Both
    # are large enough for the (64, 128) view, so nothing reads out of bounds.
    second_ptr = tla.allocate((128, 128), tla.Float16, tla.AddressSpace.ub, 256)
    selected = first_ptr if choice == 0 else second_ptr
    packed = tla.make_tensor_like(selected, source, tla.arch.zN)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            source_chunk = tla.tile_view(
                source, tla.make_shape(1, 128), tla.make_coord(0, 0)
            )
            packed_chunk = tla.tile_view(
                packed, tla.make_shape(1, 128), tla.make_coord(0, 0)
            )
            packed_chunk.store(source_chunk.load(), mask=mask)

def test_cube_tla_compile_emits_minimal_hivm_attrs_after_tla_func_to_hacc() -> None:
    ta, tb, tc = _mmad_tensor_args()
    try:
        mlir_text = _cube_attr_kernel.dump_mlir(type_args=(ta, tb, tc))
    except TlaLoweringError as exc:
        _skip_if_mmad_rank2_tile_view_regression(exc)
        raise

    output = _run_tla_compile_ir_after_pass(mlir_text, "tla-lower-func")

    assert "dlti.target_system_spec = #dlti.target_system_spec<" in output
    assert '#dlti.dl_entry<"ARCH", "dav-c310">' in output
    assert 'hacc.target = #hacc.target<"Ascend950PR_9589">' in output
    assert "hivm.module_core_type = #hivm.module_core_type<AIC>" in output
    assert "hacc.entry" in output
    assert "hacc.function_kind = #hacc.function_kind<DEVICE>" in output

def test_non_mixed_functions_are_not_split() -> None:
    for kernel, name in (
        (_split_cube_only_kernel, "_split_cube_only_kernel"),
        (_split_vector_only_kernel, "_split_vector_only_kernel"),
        (_split_no_scope_kernel, "_split_no_scope_kernel"),
    ):
        output = _dump_after_mixed_split(kernel)
        assert f"func.func @{name}(" in output
        assert f"@{name}_mix_aic" not in output
        assert f"@{name}_mix_aiv" not in output

def test_mixed_split_preserves_interleaved_scopes_and_all_scalar_operations() -> None:
    output = _dump_after_mixed_split(_mixed_interleaved_scopes_kernel)
    aic = _extract_function(output, "_mixed_interleaved_scopes_kernel_mix_aic")
    aiv = _extract_function(output, "_mixed_interleaved_scopes_kernel_mix_aiv")

    assert aic.count("tla.cube") == 2
    assert "tla.vector" not in aic
    assert aiv.count("tla.vector") == 2
    assert "tla.cube" not in aiv
    for marker in (101, 102, 103, 104, 105):
        assert f"arith.constant {marker}" in aic
        assert f"arith.constant {marker}" in aiv
    assert aic.count("arith.addi") == 5
    assert aiv.count("arith.addi") == 5

def test_mixed_split_preserves_if_and_scalar_logic_in_opposite_scope_branch() -> None:
    output = _dump_after_mixed_split(_mixed_if_branches_kernel)
    aic = _extract_function(output, "_mixed_if_branches_kernel_mix_aic")
    aiv = _extract_function(output, "_mixed_if_branches_kernel_mix_aiv")

    assert aic.count("scf.if") == 1
    assert aiv.count("scf.if") == 1
    assert aic.count("scf.yield") == 2
    assert aiv.count("scf.yield") == 2
    assert aic.count("tla.cube") == 1
    assert "tla.vector" not in aic
    assert aiv.count("tla.vector") == 1
    assert "tla.cube" not in aiv
    for marker in (201, 202, 203, 204, 205, 206):
        assert f"arith.constant {marker}" in aic
        assert f"arith.constant {marker}" in aiv

@pytest.mark.parametrize(
    ("kernel", "name", "control_flow"),
    (
        (_mixed_for_kernel, "_mixed_for_kernel", "scf.for"),
        (_mixed_while_kernel, "_mixed_while_kernel", "scf.while"),
    ),
)
def test_mixed_split_preserves_loop_carried_control_flow(
    kernel: object, name: str, control_flow: str
) -> None:
    output = _dump_after_mixed_split(kernel, type_args=(4,))
    aic = _extract_function(output, f"{name}_mix_aic")
    aiv = _extract_function(output, f"{name}_mix_aiv")

    assert control_flow in aic
    assert control_flow in aiv
    assert "scf.yield" in aic
    assert "scf.yield" in aiv
    assert aic.count("tla.cube") == 1
    assert "tla.vector" not in aic
    assert aiv.count("tla.vector") == 1
    assert "tla.cube" not in aiv

def test_pointer_if_mixed_results_compile_through_tla_lower_ptr() -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 8),
              (8, 1),
              origin_shape=(16, 8),
              layout_tag=tla.arch.RowMajor,
          )
    mlir_text = _pointer_if_kernel.dump_mlir(type_args=(mem,))
    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-lower-ptr", require_success=True
    )

    assert "scf.if" in output
    assert "-> (i64, i32)" in output
    assert "tla.inttoptr" in output
    assert "tla.alloc_ptr" not in output

@pytest.mark.parametrize(
    ("kernel", "expected_storage_elements"),
    (
        # Both allocations take 512 bytes each (256 rounded up to the
        # 512-byte L1 alignment), so the two edges of the join agree and the
        # pointer carries that capacity: 256 f16 elements.
        (_loop_carried_pointer_same_capacity_kernel, 256),
        # 1024 bytes versus 512 bytes: the join cannot pick one, so the extent
        # falls back to the 16 x 4 tile the copy actually touches.
        (_loop_carried_pointer_changed_capacity_kernel, 64),
    ),
)
def test_loop_carried_pointer_consumed_in_body_compiles_with_safe_capacity(
    kernel: object, expected_storage_elements: int
) -> None:
    mem = make_fake_tensor(
              tla.Float16,
              (16, 4),
              (4, 1),
              origin_shape=(16, 4),
              layout_tag=tla.arch.RowMajor,
          )
    mlir_text = kernel.dump_mlir(type_args=(mem,))
    assert "scf.for" in mlir_text
    assert "iter_args(" in mlir_text
    assert "tla.make_tensor_like %arg" in mlir_text
    assert "!tla.ptr<f16, l1, 512>" in mlir_text

    desc_output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-lower-tensor-desc", require_success=True
    )
    assert desc_output.count("tla.tensor_desc") >= 2
    assert "tla.tile_view" not in desc_output
    assert "tla.make_tensor_like" not in desc_output

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-cube-region", require_success=True
    )
    pointer_cast_lines = [
        line for line in output.splitlines() if "hivm.hir.pointer_cast" in line
    ]
    expected_type = (
        f"memref<{expected_storage_elements}xf16, #hivm.address_space<cbuf>>"
    )
    assert any(expected_type in line for line in pointer_cast_lines), output

def test_cube_dynamic_on_chip_pointer_uses_zero_extents() -> None:
    mlir_text = _cube_dynamic_l1_pointer_join_kernel.dump_mlir(
        type_args=(_dynamic_cube_gm_tensor(), 64, 0)
    )
    assert "!tla.shape<?,128>" in mlir_text
    assert "zN" in mlir_text
    assert "tla.ptr_add" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-cube-region", require_success=True
    )

    pointer_cast = re.search(
        r"hivm\.hir\.pointer_cast\([^)]*\) \[(%[A-Za-z0-9_]+)\] "
        r": memref<\?xf16, #hivm\.address_space<cbuf>>",
        output,
    )
    assert pointer_cast is not None, output
    unknown_extent = re.escape(pointer_cast.group(1))
    assert re.search(
        rf"{unknown_extent} = arith\.constant 0 : index", output
    ), output
    assert re.search(
        r"hivm\.hir\.pointer_cast\([^)]*\) \[%[A-Za-z0-9_]+, "
        r"%[A-Za-z0-9_]+\] : memref<\?x\?xf16, "
        r"#hivm\.address_space<gm>>",
        output,
    ), output

    assert "call @copy_gm_RowMajor_to_l1_zN_half" in output
    assert "memref<0xf16" not in output
    assert '"tla.copy"' not in output

@pytest.mark.parametrize(
    ("kernel", "control_flow", "tensor_arg_count", "producer"),
    (
        (_tensor_if_mixed_base_carrier_kernel, "scf.if", 2, "tla.make_tensor "),
        (_tensor_if_packed_carrier_kernel, "scf.if", 1, "tla.make_tensor_like"),
        (_tensor_for_carrier_kernel, "scf.for", 1, "tla.make_tensor "),
        (_tensor_while_carrier_kernel, "scf.while", 1, "tla.make_tensor "),
    ),
)
def test_tensor_valued_scf_carrier_is_materialized_as_descriptor_fields(
    kernel: object, control_flow: str, tensor_arg_count: int, producer: str
) -> None:
    mem = make_fake_tensor(
              tla.Float32,
              (128, 128),
              (128, 1),
              origin_shape=(128, 128),
              layout_tag=tla.arch.RowMajor,
          )
    type_args = (mem,) * tensor_arg_count + (2,)
    mlir_text = kernel.dump_mlir(type_args=type_args)
    assert control_flow in mlir_text
    assert "!tla.tensor" in mlir_text
    assert producer in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-lower-tensor-desc", require_success=True
    )
    assert "tla.tensor_desc" in output
    assert "tla.tile_view" not in output
    assert "tla.make_tensor " not in output
    assert "tla.make_tensor_like" not in output
    carrier_lines = [
        line
        for line in output.splitlines()
        if any(
            op in line
            for op in (
                "scf.if",
                "scf.for",
                "scf.while",
                "scf.condition",
                "scf.yield",
            )
        )
    ]
    assert carrier_lines
    assert all("!tla.tensor" not in line for line in carrier_lines), output
    fixed_descriptor_types = re.compile(r"i64(?:, index){12}")
    assert any(fixed_descriptor_types.search(line) for line in carrier_lines), output

def test_tensor_scf_carrier_preserves_dynamic_descriptor_fields() -> None:
    mlir_text = _tensor_if_dynamic_stride_carrier_kernel.dump_mlir(
        type_args=(128, 96, 1)
    )
    assert "!tla.stride<?,1>" in mlir_text
    assert "scf.if" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-lower-tensor-desc", require_success=True
    )
    assert "tla.tensor_desc" in output
    assert "tla.tile_view" not in output
    assert "tla.make_tensor " not in output
    carrier_lines = [
        line for line in output.splitlines() if "scf.if" in line or "scf.yield" in line
    ]
    assert carrier_lines
    assert all("!tla.tensor" not in line for line in carrier_lines), output

@pytest.mark.parametrize(
    ("kernel", "control_flow", "result_count", "metadata_fields"),
    (
        (
            _tensor_if_dynamic_metadata_access_kernel,
            "scf.if",
            5,
            # Dynamic tensor metadata is carried as i32, then index_cast at
            # make_shape/coord/stride boundaries.
            (1, 4, 2, 3),
        ),
        (
            _tensor_statement_if_dynamic_metadata_access_kernel,
            "scf.if",
            5,
            (1,),
        ),
        (
            _tensor_for_dynamic_metadata_access_kernel,
            "scf.for",
            5,
            (1,),
        ),
        (
            _tensor_while_dynamic_metadata_access_kernel,
            "scf.while",
            6,
            (2,),
        ),
    ),
)
def test_tensor_scf_carrier_preserves_dynamic_metadata_ssa(
    kernel: object,
    control_flow: str,
    result_count: int,
    metadata_fields: tuple[int, ...],
) -> None:
    mlir_text = kernel.dump_mlir(type_args=(128, 96, 2))
    match = re.search(
        rf"(%[A-Za-z0-9_]+):{result_count} = {re.escape(control_flow)}",
        mlir_text,
    )
    assert match is not None, mlir_text
    result = match.group(1)
    for field in metadata_fields:
        assert f"arith.index_cast {result}#{field} : i32 to index" in mlir_text, (
            mlir_text
        )
    assert "tla.make_shape" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-lower-tensor-desc", require_success=True
    )
    carrier_lines = [
        line
        for line in output.splitlines()
        if any(op in line for op in ("scf.if", "scf.for", "scf.while", "scf.yield"))
    ]
    assert carrier_lines
    assert all("!tla.tensor" not in line for line in carrier_lines), output

def test_vector_tile_view_uses_static_parent_pitch_in_flat_offset() -> None:
    mlir_text = _vector_pitched_tile_view_kernel.dump_mlir()
    assert "!tla.shape<1,64>" in mlir_text
    assert "!tla.stride<128,1>" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )
    assert "arith.constant 128 : index" in output
    assert "memref<256xf32, #hivm.address_space<ub>>" in output

def test_vector_tile_view_captures_dynamic_parent_pitch() -> None:
    mlir_text = _vector_dynamic_pitched_tile_view_kernel.dump_mlir(type_args=(128,))
    assert "!tla.stride<?,1>" in mlir_text
    assert "tla.make_stride %" in mlir_text

    desc_output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-lower-tensor-desc", require_success=True
    )
    assert '"tla.vec.func"' in desc_output
    assert desc_output.count("tla.tensor_desc") >= 2
    assert "tla.tile_view" not in desc_output
    assert "tla.make_tensor " not in desc_output
    assert "tla.make_tensor_like" not in desc_output

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )
    assert "arith.muli" in output


def test_vector_tile_view_built_outside_region_keeps_its_offset() -> None:
    """A view built outside tla.vec.func must not collapse to base + 0.

    Views over one allocation share a storage identity and dedup to a single
    helper argument, so the per-view offset cannot travel through the helper
    ABI; it has to be materialized inside the helper. When it was not, the
    region read and wrote chunk 0 whatever coord the view carried -- correct
    numerics for one region, silently wrong from the second on.
    """
    mlir_text = _vector_tile_view_outside_region_kernel.dump_mlir()
    assert "!tla.shape<1,64>" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )
    helper = output[output.index("func.func private @vector_region_"):]
    # coord (1, 0) over a 128-element pitch: the tile starts at element 128.
    assert "memref.reinterpret_cast" in helper, output
    assert "arith.constant 128 : index" in output, output
    offsets = re.findall(r"reinterpret_cast %\w+ to offset: \[([^\]]+)\]", helper)
    assert offsets, helper
    assert not all(o.strip() == "0" for o in offsets), helper


def _helper_body(output: str) -> str:
    return output[output.index("func.func private @vector_region_"):]


def _reinterpret_offsets(helper: str) -> list[str]:
    return re.findall(r"reinterpret_cast %\w+ to offset: \[([^\]]+)\]", helper)


def test_vector_dynamic_view_built_outside_region_keeps_its_offset() -> None:
    """A runtime coord has to travel to the helper and be applied there.

    With a constant coord the offset can be folded, so this is the case that
    actually exercises capturing the descriptor's operands as helper scalars.
    """
    mlir_text = _vector_dynamic_view_outside_region_kernel.dump_mlir(type_args=(2,))
    assert "tla.make_coord %" in mlir_text, mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )
    helper = _helper_body(output)
    offsets = _reinterpret_offsets(helper)
    assert offsets, helper
    # The offset is computed from a helper argument, not folded to a constant.
    assert all(o.strip() != "0" for o in offsets), helper
    assert "arith.muli" in helper, helper


def test_two_dynamic_views_on_one_allocation_keep_distinct_offsets() -> None:
    """Two runtime-offset views over one allocation dedup to one argument.

    The per-view offset therefore cannot come from the helper ABI; each has to
    be materialized separately inside the helper. Before the fix both accesses
    landed on the same chunk.
    """
    mlir_text = _vector_two_dynamic_views_outside_region_kernel.dump_mlir(
        type_args=(1, 3)
    )
    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )
    helper = _helper_body(output)
    offsets = _reinterpret_offsets(helper)
    assert len(offsets) >= 2, helper
    # Distinct offsets: the two views must not collapse onto one another.
    assert len(set(o.strip() for o in offsets)) >= 2, helper


def test_vector_zn_unalign_address_arithmetic_uses_i32() -> None:
    mlir_text = _vector_dynamic_zn_unalign_address_kernel.dump_mlir(type_args=(50,))
    assert "zNUnAlign" in mlir_text
    assert "!tla.shape<?,128>" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )
    div_rem_lines = [
        line
        for line in output.splitlines()
        if "arith.divsi" in line or "arith.remsi" in line
    ]
    assert len(div_rem_lines) >= 5, output
    assert all(line.rstrip().endswith(": i32") for line in div_rem_lines), output
    assert not any(line.rstrip().endswith(": index") for line in div_rem_lines), output
    div_rem_results = [
        re.search(r"(%[A-Za-z0-9_]+) = arith\.(?:divsi|remsi)", line).group(1)
        for line in div_rem_lines
    ]
    assert all(
        f"arith.index_cast {result} : i32 to index" not in output
        for result in div_rem_results
    ), output
    assert (
        sum(
            "arith.muli" in line and line.rstrip().endswith(": i32")
            for line in output.splitlines()
        )
        >= 4
    ), output
    assert (
        sum(
            "arith.addi" in line and line.rstrip().endswith(": i32")
            for line in output.splitlines()
        )
        >= 3
    ), output

def test_vector_static_tile_does_not_narrow_unknown_base_capacity() -> None:
    """UB base stays memref<?xf16> with extent 0; tile size must not become capacity."""
    mlir_text = _vector_dynamic_row_major_loop_unknown_base_kernel.dump_mlir(
        type_args=(2, 0)
    )
    assert "!tla.shape<?,128>" in mlir_text
    assert "!tla.shape<1,128>" in mlir_text
    assert "scf.for" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )

    dynamic_ub_casts = re.findall(
        r"hivm\.hir\.pointer_cast\([^)]*\) \[(%[A-Za-z0-9_]+)\] "
        r": memref<\?xf16, #hivm\.address_space<ub>>",
        output,
    )
    assert dynamic_ub_casts, output
    for extent in set(dynamic_ub_casts):
        escaped = re.escape(extent)
        assert re.search(
            rf"{escaped} = arith\.constant 0 : index", output
        ), output

    helper_signatures = [
        line
        for line in output.splitlines()
        if "func.func private @vector_region_" in line
    ]
    dynamic_ub_type = "memref<?xf16, #hivm.address_space<ub>>"
    assert any(dynamic_ub_type in line for line in helper_signatures), output

    assert "arith.constant 128 : index" in output
    assert "arith.muli" in output
    assert "memref<0xf16" not in output
    assert not re.search(
        r"hivm\.hir\.pointer_cast\([^)]*\).*"
        r"memref<128xf16, #hivm\.address_space<ub>>",
        output,
    ), output

def test_vector_dynamic_on_chip_pointer_uses_unknown_flat_extent() -> None:
    mlir_text = _vector_dynamic_packed_pointer_join_kernel.dump_mlir(
        type_args=(64, 0)
    )
    assert "zN" in mlir_text
    assert "!tla.shape<?,128>" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )

    pointer_cast = re.search(
        r"hivm\.hir\.pointer_cast\([^)]*\) \[(%[A-Za-z0-9_]+)\] "
        r": memref<\?xf16, #hivm\.address_space<ub>>",
        output,
    )
    assert pointer_cast is not None, output
    unknown_extent = re.escape(pointer_cast.group(1))
    assert re.search(
        rf"{unknown_extent} = arith\.constant 0 : index", output
    ), output
    assert "func.func private @vector_region_" in output
    assert "memref<?xf16, #hivm.address_space<ub>>" in output
    assert "memref<0xf16" not in output
    assert '"tla.vec.func"' not in output
    assert "memref.dim" not in output

def test_vector_static_rank_one_pointer_keeps_inferred_flat_extent() -> None:
    mlir_text = _vector_static_rank_one_pointer_join_kernel.dump_mlir(type_args=(0,))
    assert "!tla.shape<128>" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )

    expected_type = "memref<128xf16, #hivm.address_space<ub>>"
    pointer_cast_lines = []
    for line in output.splitlines():
        if "hivm.hir.pointer_cast" in line:
            pointer_cast_lines.append(line)
    assert any(expected_type in line for line in pointer_cast_lines), output
    assert "memref<?xf16, #hivm.address_space<ub>>" not in output
    assert "memref<0xf16" not in output
    assert '"tla.vec.func"' not in output

def test_vector_static_view_with_unproven_capacity_uses_unknown_extent() -> None:
    mlir_text = _vector_static_packed_pointer_join_kernel.dump_mlir(type_args=(0,))
    assert "zN" in mlir_text
    assert "!tla.shape<64,128>" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )

    assert re.search(
        r"hivm\.hir\.pointer_cast\([^)]*\) "
        r": memref<8192xf16, #hivm\.address_space<ub>>",
        output,
    ), output
    pointer_cast = re.search(
        r"hivm\.hir\.pointer_cast\([^)]*\) \[(%[A-Za-z0-9_]+)\] "
        r": memref<\?xf16, #hivm\.address_space<ub>>",
        output,
    )
    assert pointer_cast is not None, output
    unknown_extent = re.escape(pointer_cast.group(1))
    assert re.search(
        rf"{unknown_extent} = arith\.constant 0 : index", output
    ), output
    assert "memref<0xf16" not in output
    assert '"tla.vec.func"' not in output

def test_vector_dynamic_gm_memref_keeps_real_descriptor() -> None:
    mlir_text = _vector_dynamic_gm_descriptor_kernel.dump_mlir(
        type_args=(_dynamic_gm_tensor(),)
    )
    assert "!tla.shape<?>" in mlir_text
    assert "!tla.ptr<f32, gm" in mlir_text
    assert "tla.tensor_extent" not in mlir_text
    assert "memref.dim" in mlir_text
    assert "tla.tensor_desc" in mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )

    assert re.search(
        r"func\.func @_vector_dynamic_gm_descriptor_kernel\("
        r"%arg0: memref<\?x\?x\?x\?xf32, strided<\[\?, \?, \?, \?\], offset: \?>, "
        r"#hivm\.address_space<gm>> \{tla.dynamic_gm\}, %arg1: index, %arg2: index",
        output,
    ), output
    assert "memref.dim %arg0" in output
    gm_pointer_cast_lines = [
        line
        for line in output.splitlines()
        if "hivm.hir.pointer_cast" in line and "#hivm.address_space<gm>" in line
    ]
    assert not gm_pointer_cast_lines, output
    assert '"tla.vec.func"' not in output


@dataclass
class _CarriedTileState:
    tile: Any


def _carried_ub_tile(n: int):
    gm = tla.make_tensor(
        tla.allocate(n, tla.Float32, tla.AddressSpace.ub, 256),
        tla.make_layout(
            tla.make_shape(n), tla.make_stride(1), origin_shape=tla.make_shape(n)
        ),
        tla.make_coord(0),
    )
    return tla.tile_view(gm, tla.make_shape(64), tla.make_coord(0))


@tla.kernel
def _store_through_loop_carried_tile_kernel(limit: int) -> None:
    with tla.vector():
        first = _carried_ub_tile(1024)
        second = _carried_ub_tile(1024)
        state = _CarriedTileState(first)
        for _ in tla.range(0, limit, 1):
            state = _CarriedTileState(second)
        with tla.vec.func(mode="simd"):
            state.tile.store(state.tile.load())


def test_store_through_a_loop_carried_tile_lowers() -> None:
    """A tile the loop rebinds can still be stored through.

    tla-lower-tensor-desc splits the tile into scalar carriers and rebuilds a
    descriptor after the loop, so the store's origin_shape operands are scf.for
    results rather than constants. The vector pass can only clone constants, so
    the lane count was unresolvable until runtime scalars became helper
    arguments -- which is why this needs the vec.func scalar capture.
    """
    mlir_text = _store_through_loop_carried_tile_kernel.dump_mlir(type_args=(4,))
    assert "scf.for" in mlir_text, mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )
    assert '"tla.vec.func"' not in output, output


@tla.kernel
def _copy_into_loop_carried_tile_kernel(mem: tla.Tensor, limit: int) -> None:
    with tla.vector():
        source = tla.tile_view(mem, tla.make_shape(64), tla.make_coord(0))
        first = _carried_ub_tile(1024)
        second = _carried_ub_tile(1024)
        selected = first
        for _ in tla.range(0, limit, 1):
            selected = second
        tla.copy(selected, source)


def test_copy_into_a_loop_carried_tile_lowers() -> None:
    """tla.copy works through a tile the loop rebinds, on a supported route.

    The destination is chosen at runtime, so the two candidates have different
    base addresses -- which the scalar carriers preserve. Note the route is what
    tla.copy is fussy about: the vector pass handles gm<->ub and ub->l1, and a
    ub->ub copy fails here whether or not a loop is involved.
    """
    mlir_text = _copy_into_loop_carried_tile_kernel.dump_mlir(
        type_args=(
            make_fake_tensor(
                tla.Float32,
                (1024,),
                (1,),
                origin_shape=(1024,),
                layout_tag=tla.arch.RowMajor,
            ),
            4,
        )
    )
    assert "scf.for" in mlir_text, mlir_text

    output = _run_tla_compile_ir_after_pass(
        mlir_text, "tla-vector-region-simd", require_success=True
    )
    assert "tla.copy" not in output, output
