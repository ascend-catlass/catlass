"""SIMT atomic add regression; run while holding the project's NPU lock."""

from __future__ import annotations

import argparse
import json

from catlass import tla

GUARD_INT = -13
GUARD_FLOAT = -13.0
OUTPUT_ELEMS = 64

# (label, tensor dtype, unsigned view, addend, threads)
CASES = (
    ("f16", "float16", None, 1.0, 32),
    ("f16", "float16", None, 1.0, 49),
    ("f32", "float32", None, 0.5, 32),
    ("f32", "float32", None, 0.5, 49),
    ("bf16", "bfloat16", None, 1.0, 32),
    ("i32", "int32", None, 3, 32),
    ("i32", "int32", None, 3, 49),
    ("ui32", "int32", "uint32", 3, 32),
    ("i64", "int64", None, 3, 32),
    ("i64", "int64", None, 3, 49),
    ("ui64", "int64", "uint64", 3, 32),
    ("i32", "int32", None, -2, 32),
)

# Same shape, but every thread of the single block contends on one UB element.
# UB is private to a block, so a second block would see its own buffer rather
# than the same one; the cases therefore always run with block_num=1.
CASES_UB = (
    ("f16", "float16", None, 1.0, 32),
    ("f16", "float16", None, 1.0, 49),
    ("f32", "float32", None, 0.5, 32),
    ("f32", "float32", None, 0.5, 49),
    ("bf16", "bfloat16", None, 1.0, 32),
    ("i32", "int32", None, 3, 32),
    ("i32", "int32", None, 3, 49),
    ("ui32", "int32", "uint32", 3, 32),
    ("i32", "int32", None, -2, 32),
)

# Torch dtype name -> tla element type for the UB staging buffer. The viewed
# name wins, so the uint32 case allocates a ui32 buffer rather than an i32 one.
_TLA_TYPES = {
    "float16": tla.Float16,
    "bfloat16": tla.BFloat16,
    "float32": tla.Float32,
    "int32": tla.Int32,
    "uint32": tla.UInt32,
}

def make_kernel(threads, record_old, old_kind):
    @tla.kernel
    def kernel(source: tla.Tensor, output: tla.Tensor, oldvals: tla.Tensor) -> None:
        block = tla.arch.block_idx()
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=threads):
                tx, _, _ = tla.arch.thread_idx()
                index = block * threads + tx
                old = tla.arch.atomic_add(output.ptr, source[index])
                if record_old:
                    oldvals[index] = old.to(old_kind)
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel

def _recorded_old_value(guard, addend, view_name, rank):
    """The element thread `rank` reads, in the bits the harness records.

    Unsigned elements wrap at their own width and are recorded zero-extended;
    ui64 is the exception, because its bit pattern rides the signed i64 the
    harness records, so a value past the sign bit folds back.
    """
    value = guard + addend * rank
    if not view_name:
        return value
    width = 32 if view_name == "uint32" else 64
    value %= 1 << width
    if value >= 1 << 63:
        value -= 1 << 64
    return value

def check_case(torch, case, blocks, npu_arch):
    from catlass.tla.runtime import from_dlpack

    label, dtype_name, view_name, addend, threads = case
    dtype = getattr(torch, dtype_name)
    guard = GUARD_FLOAT if dtype.is_floating_point else GUARD_INT
    count = threads * blocks

    source = torch.full((count,), addend, dtype=dtype, device="npu")
    output = torch.full((OUTPUT_ELEMS,), guard, dtype=dtype, device="npu")
    view = getattr(torch, view_name) if view_name else None

    source_arg = from_dlpack(
        source.view(view) if view else source, layout_tag=tla.arch.RowMajor
    )
    output_arg = from_dlpack(
        output.view(view) if view else output, layout_tag=tla.arch.RowMajor
    )

    # f16/bf16 return nothing: their hardware return value is unusable. Every
    # other dtype records the old value each thread saw; the atom unit
    # serialises fully, so the recorded values are guard + addend * rank of the
    # thread in the serial order, i.e. a permutation of guard + addend * 0..N-1.
    record_old = label not in ("f16", "bf16")
    if record_old:
        old_kind = tla.Int64 if not dtype.is_floating_point else tla.Float32
        old_dtype = torch.int64 if not dtype.is_floating_point else torch.float32
        oldvals = torch.zeros((count,), dtype=old_dtype, device="npu")
        oldvals_arg = from_dlpack(oldvals, layout_tag=tla.arch.RowMajor)
    else:
        old_kind = None
        oldvals = None
        oldvals_arg = output_arg  # placeholder argument, never written

    artifact = tla.compile(
        make_kernel(threads, record_old, old_kind), source_arg, output_arg, oldvals_arg,
        options=f"--npu-arch {npu_arch}",
    )
    artifact(source_arg, output_arg, oldvals_arg, block_num=blocks)
    torch.npu.synchronize()

    actual = output.cpu()
    expected = guard + count * addend
    if actual[0].item() != expected:
        raise AssertionError(f"output[0]={actual[0].item()} expected {expected}")
    if not torch.all(actual[1:] == guard):
        raise AssertionError("guard elements changed")
    if record_old:
        # Each thread ranks somewhere in the serial order, so thread t sees
        # guard + addend * rank(t) and the multiset is guard + addend * 0..N-1.
        # Compare the sorted sequences item by item: a sum-only check cannot
        # tell a duplicate from a missing rank -- [0, 0, 3, 3] sums exactly like
        # [0, 1, 2, 3] -- so a lost or a doubled update would slip through.
        observed = torch.sort(oldvals.cpu()).values
        expected = torch.tensor(
            sorted(_recorded_old_value(guard, addend, view_name, rank) for rank in range(count)),
            dtype=old_dtype,
        )
        if dtype.is_floating_point:
            equal = torch.all((observed - expected).abs() <= 1e-3)
        else:
            equal = torch.equal(observed, expected)
        if not equal:
            raise AssertionError(
                f"old values mismatch: observed {observed.tolist()} "
                f"expected {expected.tolist()}"
            )

    return {
        "dtype": label,
        "space": "gm",
        "threads": threads,
        "blocks": blocks,
        "added": count * addend,
        "passed": True,
    }

def make_ub_kernel(threads, dtype):
    @tla.kernel
    def kernel(source: tla.Tensor, output: tla.Tensor) -> None:
        ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
        vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
        out_ub_ptr = tla.allocate(OUTPUT_ELEMS, dtype, tla.AddressSpace.ub, 256)
        out_ub = tla.make_tensor_like(out_ub_ptr, output, tla.arch.RowMajor)

        with tla.vector():
            tla.copy(out_ub, output)
            tla.set_flag(ub_loaded)
            tla.wait_flag(ub_loaded)
            with tla.vec.func(mode="simt", thread_block_dim=threads):
                tx, _, _ = tla.arch.thread_idx()
                tla.arch.atomic_add(out_ub.ptr, source[tx])
            tla.set_flag(vec_done)
            tla.wait_flag(vec_done)
            tla.copy(output, out_ub)
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel

def check_ub_case(torch, case, npu_arch):
    from catlass.tla.runtime import from_dlpack

    label, dtype_name, view_name, addend, threads = case
    dtype = getattr(torch, dtype_name)
    guard = GUARD_FLOAT if dtype.is_floating_point else GUARD_INT
    tla_dtype = _TLA_TYPES[view_name or dtype_name]

    source = torch.full((threads,), addend, dtype=dtype, device="npu")
    output = torch.full((OUTPUT_ELEMS,), guard, dtype=dtype, device="npu")
    view = getattr(torch, view_name) if view_name else None

    source_arg = from_dlpack(
        source.view(view) if view else source, layout_tag=tla.arch.RowMajor
    )
    output_arg = from_dlpack(
        output.view(view) if view else output, layout_tag=tla.arch.RowMajor
    )

    artifact = tla.compile(
        make_ub_kernel(threads, tla_dtype),
        source_arg,
        output_arg,
        options=f"--npu-arch {npu_arch}",
    )
    artifact(source_arg, output_arg, block_num=1)
    torch.npu.synchronize()

    actual = output.cpu()
    expected = guard + threads * addend
    if actual[0].item() != expected:
        raise AssertionError(f"output[0]={actual[0].item()} expected {expected}")
    if not torch.all(actual[1:] == guard):
        raise AssertionError("guard elements changed")

    return {
        "dtype": label,
        "space": "ub",
        "threads": threads,
        "blocks": 1,
        "added": threads * addend,
        "passed": True,
    }

def make_tile_alias_kernel(threads, rows, cols):
    """Every way of naming a view's storage must land on the revealed element.

    ``tile.ptr`` is the storage base; ``tile.ptr_with_coord`` is the view's
    start. ``make_tensor_like(tile.ptr, ...)`` re-shapes the base, so its
    ``ptr_with_coord`` describes the same view start. A runtime-stride view
    takes that start from the descriptor's SSA stride, not from a constant.
    ``make_tensor`` rebuilt at the origin must stay at element 0, and a
    ``get_tile`` child must add its parent's start exactly once. A capture taken
    at the wrong base, or a parent counted twice, moves the writes to elements
    the guard check below sees unchanged.
    """

    @tla.kernel
    def kernel(output: tla.Tensor) -> None:
        ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
        vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
        buf_ptr = tla.allocate(rows * cols, tla.Int32, tla.AddressSpace.ub, 256)
        buf = tla.make_tensor_like(buf_ptr, output, tla.arch.RowMajor)
        # All of these live outside the region, so their descriptors are ordinary
        # captures; only the pointers they name cross, through the tensors.
        layout = tla.make_layout(tla.make_shape(rows, cols), tla.make_stride(cols, 1))
        whole = tla.make_tensor(buf_ptr, layout, tla.make_coord(0, 0))
        tile = tla.tile_view(buf, tla.make_shape(2, 4), tla.make_coord(1, 1))
        alias = tla.make_tensor_like(tile.ptr, tile, tla.arch.RowMajor)
        inner = tla.get_tile(tile, tla.make_coord(0, 0), tla.make_shape(1, 2))
        # block_idx() keeps the pitch runtime-valued, so the view offset has to
        # come from the descriptor's SSA stride rather than a static constant.
        pitch = tla.arch.block_idx() + 24
        dyn_layout = tla.make_layout(tla.make_shape(rows, cols), tla.make_stride(pitch, 1))
        dyn_whole = tla.make_tensor(buf_ptr, dyn_layout, tla.make_coord(0, 0))
        dyn_tile = tla.tile_view(dyn_whole, tla.make_shape(2, 4), tla.make_coord(1, 1))
        dyn_alias = tla.make_tensor_like(dyn_tile.ptr, dyn_tile, tla.arch.RowMajor)

        with tla.vector():
            tla.copy(buf, output)
            tla.set_flag(ub_loaded)
            tla.wait_flag(ub_loaded)
            with tla.vec.func(mode="simt", thread_block_dim=threads):
                tx, _, _ = tla.arch.thread_idx()
                tla.arch.atomic_add(whole.ptr + tx, tla.Int32(1))
                tla.arch.atomic_add(tile.ptr + tx, tla.Int32(1))
                tla.arch.atomic_add(inner.ptr_with_coord + tx, tla.Int32(1))
                tla.arch.atomic_add(tile.ptr_with_coord + tx, tla.Int32(1))
                tla.arch.atomic_add(alias.ptr_with_coord + tx, tla.Int32(1))
                tla.arch.atomic_add(dyn_tile.ptr_with_coord + tx, tla.Int32(1))
                tla.arch.atomic_add(dyn_alias.ptr_with_coord + tx, tla.Int32(1))
            tla.set_flag(vec_done)
            tla.wait_flag(vec_done)
            tla.copy(output, buf)
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel

def make_gm_rebuild_kernel(threads, rows, cols):
    """A zero-coordinate 2D GM view rebuilt from ``.ptr`` must stay at the base.

    ``make_tensor(output.ptr, layout, (0, 0))`` names the same storage as
    ``output`` itself, so the atomic must land on elements 0..threads-1. A
    coordinate read from the wrong descriptor operand would shift the launch
    pointer by a row stride and move the writes to the buffer tail.
    """

    @tla.kernel
    def kernel(output: tla.Tensor) -> None:
        layout = tla.make_layout(tla.make_shape(rows, cols), tla.make_stride(cols, 1))
        view = tla.make_tensor(output.ptr, layout, tla.make_coord(0, 0))
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=threads):
                tx, _, _ = tla.arch.thread_idx()
                tla.arch.atomic_add(view.ptr + tx, tla.Int32(1))
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel

def check_tile_alias_case(torch, npu_arch):
    from catlass.tla.runtime import from_dlpack

    threads = 4
    rows, cols = 8, 16
    output = torch.full((rows, cols), GUARD_INT, dtype=torch.int32, device="npu")
    output_arg = from_dlpack(output, layout_tag=tla.arch.RowMajor)

    artifact = tla.compile(
        make_tile_alias_kernel(threads, rows, cols),
        output_arg,
        options=f"--npu-arch {npu_arch}",
    )
    artifact(output_arg, block_num=1)
    torch.npu.synchronize()

    # The tile coordinate is counted in tiles, so (1, 1) of a 2x4 tile is
    # element (2, 4): flat 2 * 16 + 4 = 36. `whole.ptr` and `tile.ptr` both hit
    # the storage base (elements 0..threads-1); `inner`, `tile.ptr_with_coord`
    # and `alias.ptr_with_coord` hit the view start once each. The runtime
    # pitch is 24 with block 0, so the dynamic views start at 2 * 24 + 4 = 52.
    expected = torch.full((rows, cols), GUARD_INT, dtype=torch.int32)
    expected.view(-1)[0:threads] = GUARD_INT + 2
    expected.view(-1)[36 : 36 + threads] = GUARD_INT + 3
    expected.view(-1)[52 : 52 + threads] = GUARD_INT + 2
    actual = output.cpu()
    if not torch.equal(actual, expected):
        raise AssertionError(
            f"tile alias writes {actual.tolist()} expected {expected.tolist()}"
        )

    return {
        "dtype": "i32",
        "space": "ub",
        "threads": threads,
        "blocks": 1,
        "added": 7 * threads,
        "passed": True,
        "case": "tile-alias",
    }

def check_gm_rebuild_case(torch, npu_arch):
    from catlass.tla.runtime import from_dlpack

    threads = 4
    rows, cols = 8, 8
    output = torch.full((rows, cols), GUARD_INT, dtype=torch.int32, device="npu")
    output_arg = from_dlpack(output, layout_tag=tla.arch.RowMajor)

    artifact = tla.compile(
        make_gm_rebuild_kernel(threads, rows, cols),
        output_arg,
        options=f"--npu-arch {npu_arch}",
    )
    artifact(output_arg, block_num=1)
    torch.npu.synchronize()

    expected = torch.full((rows, cols), GUARD_INT, dtype=torch.int32)
    expected.view(-1)[0:threads] = GUARD_INT + 1
    actual = output.cpu()
    if not torch.equal(actual, expected):
        raise AssertionError(
            f"gm rebuild writes {actual.tolist()} expected {expected.tolist()}"
        )

    return {
        "dtype": "i32",
        "space": "gm",
        "threads": threads,
        "blocks": 1,
        "added": threads,
        "passed": True,
        "case": "gm-rebuild",
    }

def make_shared_ptr_kernel(threads, elems):
    """One UB pointer used both directly and advanced: ``p`` and ``p + tx``.

    ``p = buf.ptr`` names a UB address that the lowering folds to i64. The
    direct atomic and the offset atomic must read the same base address, with
    the offset only from ``ptr_add``. Element 0 receives one add per thread
    from the direct call plus thread 0's offset add; every other element gets
    one offset add.
    """

    @tla.kernel
    def kernel(output: tla.Tensor) -> None:
        ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
        vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
        buf_ptr = tla.allocate(elems, tla.Int32, tla.AddressSpace.ub, 256)
        buf = tla.make_tensor_like(buf_ptr, output, tla.arch.RowMajor)
        with tla.vector():
            tla.copy(buf, output)
            tla.set_flag(ub_loaded)
            tla.wait_flag(ub_loaded)
            with tla.vec.func(mode="simt", thread_block_dim=threads):
                tx, _, _ = tla.arch.thread_idx()
                p = buf.ptr
                tla.arch.atomic_add(p, tla.Int32(1))
                tla.arch.atomic_add(p + tx, tla.Int32(2))
            tla.set_flag(vec_done)
            tla.wait_flag(vec_done)
            tla.copy(output, buf)
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel

def check_shared_ptr_case(torch, npu_arch):
    from catlass.tla.runtime import from_dlpack

    threads = 32
    elems = 32
    output = torch.zeros((elems,), dtype=torch.int32, device="npu")
    output_arg = from_dlpack(output, layout_tag=tla.arch.RowMajor)

    artifact = tla.compile(
        make_shared_ptr_kernel(threads, elems),
        output_arg,
        options=f"--npu-arch {npu_arch}",
    )
    artifact(output_arg, block_num=1)
    torch.npu.synchronize()

    expected = torch.full((elems,), 2, dtype=torch.int32)
    expected[0] = threads + 2
    actual = output.cpu()
    if not torch.equal(actual, expected):
        raise AssertionError(
            f"shared ptr writes {actual.tolist()} expected {expected.tolist()}"
        )

    return {
        "dtype": "i32",
        "space": "ub",
        "threads": threads,
        "blocks": 1,
        "added": threads + 2 * threads,
        "passed": True,
        "case": "shared-ptr",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--npu-arch", default="3510")
    parser.add_argument("--only", default="", help="run one label, e.g. f16")
    args = parser.parse_args()

    import torch

    torch.npu.set_device(args.device)

    failures = 0
    if not args.only or args.only == "tile-alias":
        try:
            print(json.dumps(check_tile_alias_case(torch, args.npu_arch)))
        except Exception as error:
            failures += 1
            print(json.dumps({"dtype": "i32", "space": "gm", "case": "tile-alias", "passed": False, "error": str(error)}))
    if not args.only or args.only == "gm-rebuild":
        try:
            print(json.dumps(check_gm_rebuild_case(torch, args.npu_arch)))
        except Exception as error:
            failures += 1
            print(json.dumps({"dtype": "i32", "space": "gm", "case": "gm-rebuild", "passed": False, "error": str(error)}))
    if not args.only or args.only == "shared-ptr":
        try:
            print(json.dumps(check_shared_ptr_case(torch, args.npu_arch)))
        except Exception as error:
            failures += 1
            print(json.dumps({"dtype": "i32", "space": "ub", "case": "shared-ptr", "passed": False, "error": str(error)}))
    cases = [(case, "gm") for case in CASES] + [(case, "ub") for case in CASES_UB]
    for case, space in cases:
        if args.only and args.only != case[0]:
            continue
        try:
            if space == "gm":
                result = check_case(torch, case, args.blocks, args.npu_arch)
            else:
                result = check_ub_case(torch, case, args.npu_arch)
            print(json.dumps(result))
        except Exception as error:
            failures += 1
            print(
                json.dumps(
                    {
                        "dtype": case[0],
                        "space": space,
                        "threads": case[4],
                        "passed": False,
                        "error": str(error),
                    }
                )
            )
    raise SystemExit(1 if failures else 0)

if __name__ == "__main__":
    main()
