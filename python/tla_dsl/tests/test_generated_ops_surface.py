from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor

import inspect

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass._mlir.dialects import tla as tla_ops_gen


@tla.kernel
def _mask_bitwise_surface_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            zero = tla.sub(reg, reg)
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            h_mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            q_mask = tla.create_mask(pattern=tla.mask.Q, dtype=tla.Float32)
            not_mask = tla.bitwise_not(q_mask, mask=all_mask)
            and_mask = tla.bitwise_and(h_mask, q_mask)
            or_mask = tla.bitwise_or(h_mask, q_mask, mask=all_mask)
            xor_mask = tla.bitwise_xor(h_mask, q_mask, mask=all_mask)
            tmp0 = tla.where(not_mask, reg, zero)
            tmp1 = tla.where(and_mask, tmp0, zero)
            tmp2 = tla.where(or_mask, tmp1, zero)
            dst_tile.store(tla.where(xor_mask, tmp2, zero), mask=all_mask)


@tla.kernel
def _ops_surface_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    shape = tla.make_shape(1, 16)
    coord = tla.make_coord(0, 0)
    src_tile = tla.tile_view(src, shape, coord)
    dst_tile = tla.tile_view(dst, shape, coord)
    local_ptr = tla.allocate((1, 16), tla.Float16, tla.AddressSpace.l1, 32)
    _ = local_ptr

    with tla.vector():
        ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
        tla.set_flag(ready)
        tla.copy(dst_tile, src_tile)
        tla.wait_flag(ready)
        cross = tla.cross_flag("x")
        tla.cross_core_set_flag(cross, tla.pipes.MTE3)
        tla.cross_core_wait_flag(cross, tla.pipes.SCALAR)
        tla.pipe_barrier(tla.pipes.MTE3)
        mutex_ping = tla.mutex(resource="l0a_ping", id=0)
        mutex_pong = tla.mutex(resource="l0a_pong", id=1)
        block_range = tla.range(0, 10, 1)
        for idx in block_range:
            mutex = mutex_ping if idx % 2 == 0 else mutex_pong
            mutex.lock(pipe=tla.arch.MTE2)
            mutex.unlock(pipe=tla.arch.MTE2)


def test_generated_binding_symbols_exist_for_wrapped_ops() -> None:
    required = (
        "tile_view",
        "copy",
        "debug_print",
        "flag",
        "cross_flag",
        "cross_core_set_flag",
        "cross_core_wait_flag",
        "set_flag",
        "wait_flag",
        "pipe_barrier",
        "mutex",
        "mutex_lock",
        "mutex_unlock",
        "make_shape",
        "make_coord",
        "make_stride",
        "make_layout",
        "mmad",
        "arch_block_idx",
        "arch_thread_block_dim",
        "arch_block_num",
        "arch_thread_idx",
        "arch_lane_idx",
        "arch_lanemask",
        "simt_add",
        "simt_sub",
        "simt_mul",
        "simt_div",
        "simt_pow",
        "simt_sqrt",
        "simt_exp",
        "simt_abs",
        "simt_load",
        "simt_store",
        "arch_sync_threads",
        "recast_ptr",
        "adds",
        "subs",
        "muls",
        "maxs",
        "mins",
        "divs",
        "shift_left",
        "shift_right",
        "shift_lefts",
        "shift_rights",
        "bitwise_xor",
        "bitwise_or",
        "bitwise_and",
        "bitwise_not",
        "neg",
        "cmp",
        "bitcast",
        "interleave",
        "deinterleave",
    )
    for symbol in required:
        assert hasattr(tla_ops_gen, symbol)
    assert not hasattr(tla_ops_gen, "hivm_memref_as_ptr")
    assert not hasattr(tla_ops_gen, "HivmMemrefAsPtrOp")


def test_mask_bitwise_public_dispatch_emits_mask_ops() -> None:
    src = make_fake_tensor(
              tla.Float32,
              (64,),
              (1,),
              addrspace=tla.AddressSpace.ub,
              origin_shape=(64,),
              layout_tag=tla.arch.RowMajor,
          )
    dst = make_fake_tensor(
              tla.Float32,
              (64,),
              (1,),
              addrspace=tla.AddressSpace.ub,
              origin_shape=(64,),
              layout_tag=tla.arch.RowMajor,
          )
    mlir = _mask_bitwise_surface_kernel.dump_mlir(type_args=(src, dst))
    for op_name in (
        "tla.bitwise_not",
        "tla.bitwise_and",
        "tla.bitwise_or",
        "tla.bitwise_xor",
    ):
        assert op_name in mlir


def test_cross_flag_public_api_emits_call_site_pipes() -> None:
    src = make_fake_tensor(
              tla.Float16,
              (1, 16),
              (16, 1),
              origin_shape=(1, 16),
              layout_tag=tla.arch.RowMajor,
          )
    dst = make_fake_tensor(
              tla.Float16,
              (1, 16),
              (16, 1),
              origin_shape=(1, 16),
              addrspace=tla.AddressSpace.ub,
              layout_tag=tla.arch.RowMajor,
          )
    mlir = _ops_surface_kernel.dump_mlir(type_args=(src, dst))
    cross_flag_line = next(
        line
        for line in mlir.splitlines()
        if 'tla.cross_flag "x"' in line or '"tla.cross_flag"' in line
    )
    assert '"x"' in cross_flag_line
    assert "src_pipe" not in cross_flag_line
    assert "dst_pipe" not in cross_flag_line
    assert "tla.cross_core_set_flag" in mlir
    assert "tla.cross_core_wait_flag" in mlir
    set_line = next(
        line for line in mlir.splitlines() if "tla.cross_core_set_flag" in line
    )
    wait_line = next(
        line for line in mlir.splitlines() if "tla.cross_core_wait_flag" in line
    )
    assert "pipe = #tla.pipe<mte3>" in set_line
    assert "pipe = #tla.pipe<scalar>" in wait_line


def test_public_api_exports_representative_helpers() -> None:
    assert callable(tla.tile_view)
    assert callable(tla.copy)
    assert callable(tla.print)
    assert not hasattr(tla, "debug_print")
    assert not hasattr(tla, "print_tensor")
    assert not hasattr(tla, "debug_printf")
    assert callable(tla.flag)
    assert callable(tla.cross_flag)
    assert callable(tla.mutex)
    assert callable(tla.mutex_guard)
    assert callable(tla.make_tensor_like)
    assert "dst_dtype" not in inspect.signature(tla.make_tensor_like).parameters
    assert callable(tla.const_expr)
    assert hasattr(tla, "Constexpr")
    assert callable(tla.range_constexpr)
    assert not hasattr(tla, "constexpr")
    assert not hasattr(runtime_mod, "constexpr")
    assert callable(tla.arch.block_idx)
    assert callable(tla.allocate)
    assert callable(tla.recast_ptr)
    assert callable(tla.bitwise_not)
    assert callable(tla.bitwise_and)
    assert callable(tla.bitwise_or)
    assert callable(tla.bitwise_xor)
    assert callable(tla.shift_left)
    assert callable(tla.shift_right)
    assert tla.arch.FIX is tla.pipes.FIX
    assert tla.pipes.ALL is not None


def test_ops_surface_kernel_lowers_key_op_families() -> None:
    src = make_fake_tensor(
              tla.Float16,
              (1, 16),
              (16, 1),
              origin_shape=(1, 16),
              layout_tag=tla.arch.RowMajor,
          )
    dst = make_fake_tensor(
              tla.Float16,
              (1, 16),
              (16, 1),
              origin_shape=(1, 16),
              addrspace=tla.AddressSpace.ub,
              layout_tag=tla.arch.RowMajor,
          )
    mlir = _ops_surface_kernel.dump_mlir(type_args=(src, dst))
    for op_name in (
        "tla.alloc_ptr",
        "tla.tile_view",
        "tla.copy",
        "tla.flag",
        "tla.set_flag",
        "tla.wait_flag",
        "tla.cross_flag",
        "tla.cross_core_set_flag",
        "tla.cross_core_wait_flag",
        "tla.pipe_barrier",
        "tla.mutex",
        "tla.mutex_lock",
        "tla.mutex_unlock",
        "tla.make_shape",
        "tla.make_coord",
    ):
        assert op_name in mlir
