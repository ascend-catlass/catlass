"""One kernel chaining cube, SIMD and SIMT work, handing off entirely through UB.

    cube (AIC)   c    = lhs @ rhs                   tla.mmad, then L0C -> UB
    simd (AIV)   mid  = c + addend_simd             whole-vector add, out of UB
    simt (AIV)   out  = mid + addend_simt           per-thread add, same UB

The three stages form a pipeline: each consumes its predecessor's result from
UB, so neither the matmul output nor the SIMD result round-trips through GM.
The only GM traffic left on the vector side is loading the two addends and
storing the two results.

The cube's L0C -> UB copy uses ``L0C2UBMode.SPLIT_M``, which gives each AIV
sub-block its own half of the rows. That split is why both stages work on
``SIMD_ROWS`` rows selected with ``sub_block_idx``: each sub-block owns a
distinct row range of the output, in UB and in GM alike.

The SIMT stage indexes UB linearly, so the same allocations are viewed a second
time as flat 1-D tensors (``tla.make_tensor`` over the same pointer). The
outlining pass hands those views to the vector function as buffer parameters,
exactly as it does for the shared scratch in ``reduction_simt``. Note that the
SIMT stage reads ``ub_result`` -- the buffer the SIMD stage just wrote -- so
this also exercises a SIMD -> SIMT handoff inside one vector region, and it
writes its own result back into that same buffer element by element.

Ordering: the cube signals fix_done after the fixpipe and the vector region
waits on it before reading c; the SIMD and SIMT stages are ordered by a
pipe barrier, since they run on the same core and share ``ub_result``.

The intermediate ``mid`` is stored to GM as well, so a failure can be pinned on
the SIMD stage or the SIMT stage rather than just on the pipeline.

It is also the worked example for launch-sized UB: three of its four UB buffers
are compiled in with ``tla.allocate``, while the result buffer the SIMD and SIMT
stages share comes from ``tla.arch.get_dyn_ub`` and is sized at launch with
``ub=``. The two kinds sit in one kernel deliberately -- the dynamic region
begins above the static high-water mark, so a kernel can mix them freely.
"""

from __future__ import annotations

import argparse
import catlass.tla as tla

# M drives the UB footprint: each sub-block stages its own half of the rows, so
# the four UB tensors below come to 4 * (M_DIM / 2) * N_DIM * 4 bytes -- 128 KB
# here, far past what a SIMT kernel can reach unless its UB is declared.
M_DIM = 256
N_DIM = 64
K_DIM = 32
N_ELE = M_DIM * N_DIM

SIMD_ROWS = M_DIM // 2  # each AIV sub-block's half of the rows
SIMD_ELE = SIMD_ROWS * N_DIM
_ELEM_BYTES = 4  # _KERNEL_DTYPE is f32

# The SIMD stage stores REG_TILE_M whole rows per vector register, so the tile
# has to fill one exactly. N_DIM is therefore capped at one register's worth of
# columns; a wider N would have to be tiled along the columns as well.
_VECTOR_REG_BYTES = 256
REG_TILE_M = _VECTOR_REG_BYTES // (N_DIM * _ELEM_BYTES)
assert REG_TILE_M >= 1, "N_DIM * element size must not exceed one vector register"
assert SIMD_ROWS % REG_TILE_M == 0, "REG_TILE_M must divide the sub-block's rows"

THREADS = 128

L1_STAGE_BYTES = 256 * 1024
L0A_BYTES = M_DIM * K_DIM * 4
L0B_BYTES = K_DIM * N_DIM * 4
L0C_BYTES = M_DIM * N_DIM * 4

_KERNEL_DTYPE = tla.Float32


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def simt_mixed(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    gm_addend_simd: tla.Tensor,
    gm_addend_simt: tla.Tensor,
    gm_mid: tla.Tensor,
    gm_out: tla.Tensor,
) -> None:
    l1_loaded = tla.flag("l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)

    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    simt_done = tla.flag("simt_done", tla.arch.VECTOR, tla.arch.MTE3)

    fix_done = tla.cross_flag("fix_done")

    l1a_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_BYTES // 4, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_BYTES // 4, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_BYTES // 4, tla.Float32, tla.AddressSpace.l0c, 1024)

    c_ub_ptr = tla.allocate(SIMD_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    addend_simd_ub_ptr = tla.allocate(SIMD_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    addend_simt_ub_ptr = tla.allocate(SIMD_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    # The result buffer is the one piece of UB this kernel does not compile in:
    # it comes from the launch-sized region, which starts above the three static
    # allocations above. The SIMD and SIMT stages share it -- SIMT writes its
    # result back in place -- so one region serves both.
    #
    # Sized per launch with `ub=`, the way CuTe DSL sizes dynamic shared memory
    # with `launch(smem=...)`. Asking for only what is used matters here: this
    # kernel has a SIMT stage, and whatever UB goes unclaimed becomes Data Cache.
    result_ub_ptr = tla.arch.get_dyn_ub(_KERNEL_DTYPE, byte_alignment=256)

    # ---- cube: c = lhs @ rhs, delivered straight into UB ----
    with tla.cube():
        a_tile = tla.tile_view(lhs, tla.make_shape(M_DIM, K_DIM), tla.make_coord(0, 0))
        b_tile = tla.tile_view(rhs, tla.make_shape(K_DIM, N_DIM), tla.make_coord(0, 0))
        c_tile = tla.tile_view(
            gm_mid, tla.make_shape(M_DIM, N_DIM), tla.make_coord(0, 0)
        )

        l1_a = tla.make_tensor_like(l1a_ptr, a_tile, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, b_tile, tla.arch.zN)
        tla.copy(l1_a, a_tile)
        tla.copy(l1_b, b_tile)

        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        l1_a_l0 = tla.tile_view(
            l1_a, tla.make_shape(M_DIM, K_DIM), tla.make_coord(0, 0)
        )
        l1_b_l0 = tla.tile_view(
            l1_b, tla.make_shape(K_DIM, N_DIM), tla.make_coord(0, 0)
        )
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, c_tile, tla.arch.L0Clayout)
        tla.copy(l0_a, l1_a_l0)
        tla.copy(l0_b, l1_b_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)

        # L0C -> UB. SPLIT_M hands each AIV sub-block its own half of the rows,
        # so no vector half has to read the matmul result back from GM.
        ub_c_cube = tla.make_tensor_like(c_ub_ptr, l0_c, tla.arch.RowMajor)
        tla.copy(
            ub_c_cube,
            l0_c,
            tla.params.CopyL0C2DstParams(
                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
            ),
        )

        tla.cross_core_set_flag(fix_done, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        # This sub-block's row range, matching the SPLIT_M half the cube wrote.
        vec_idx = tla.arch.sub_block_idx()
        addend_simd_tile = tla.tile_view(
            gm_addend_simd, tla.make_shape(SIMD_ROWS, N_DIM), tla.make_coord(vec_idx, 0)
        )
        addend_simt_tile = tla.tile_view(
            gm_addend_simt, tla.make_shape(SIMD_ROWS, N_DIM), tla.make_coord(vec_idx, 0)
        )
        mid_tile = tla.tile_view(
            gm_mid, tla.make_shape(SIMD_ROWS, N_DIM), tla.make_coord(vec_idx, 0)
        )
        out_tile = tla.tile_view(
            gm_out, tla.make_shape(SIMD_ROWS, N_DIM), tla.make_coord(vec_idx, 0)
        )

        ub_c = tla.make_tensor_like(c_ub_ptr, mid_tile, tla.arch.RowMajor)
        ub_addend_simd = tla.make_tensor_like(
            addend_simd_ub_ptr, addend_simd_tile, tla.arch.RowMajor
        )
        ub_addend_simt = tla.make_tensor_like(
            addend_simt_ub_ptr, addend_simt_tile, tla.arch.RowMajor
        )
        ub_result = tla.make_tensor_like(result_ub_ptr, mid_tile, tla.arch.RowMajor)
        ub_simt_result = tla.make_tensor_like(
            result_ub_ptr, out_tile, tla.arch.RowMajor
        )

        tla.copy(ub_addend_simd, addend_simd_tile)
        tla.copy(ub_addend_simt, addend_simt_tile)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        # The SIMD stage reads c from UB, so wait for the fixpipe first.
        tla.cross_core_wait_flag(fix_done, tla.arch.VECTOR)

        # ---- SIMD: mid = c + addend_simd, whole-vector, entirely in UB ----
        for row_tile in tla.range(0, SIMD_ROWS // REG_TILE_M, 1):
            with tla.vec.func(mode="simd"):
                c_chunk = tla.tile_view(
                    ub_c, tla.make_shape(REG_TILE_M, N_DIM), tla.make_coord(row_tile, 0)
                )
                addend_chunk = tla.tile_view(
                    ub_addend_simd,
                    tla.make_shape(REG_TILE_M, N_DIM),
                    tla.make_coord(row_tile, 0),
                )
                result_chunk = tla.tile_view(
                    ub_result,
                    tla.make_shape(REG_TILE_M, N_DIM),
                    tla.make_coord(row_tile, 0),
                )
                result_chunk.store(tla.add(c_chunk.load(), addend_chunk.load()))

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(mid_tile, ub_result)

        # The SIMT stage consumes what the SIMD stage just wrote to ub_result.
        tla.pipe_barrier(tla.pipes.ALL)

        # ---- SIMT: out = mid + addend_simt, per thread, over the same UB ----
        # Flat views of the very same allocations, so the per-thread loop can
        # index them linearly.
        flat_layout = tla.make_layout(tla.make_shape(SIMD_ELE), tla.make_stride(1))
        ub_result_flat = tla.make_tensor(result_ub_ptr, flat_layout)
        ub_simt_flat = tla.make_tensor(result_ub_ptr, flat_layout)
        addend_flat_layout = flat_layout
        ub_addend_simt_flat = tla.make_tensor(addend_simt_ub_ptr, addend_flat_layout)

        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            nthreads, _, _ = tla.arch.thread_block_dim()

            for i in tla.range(tid, SIMD_ELE, nthreads):
                ub_simt_flat[i] = ub_result_flat[i] + ub_addend_simt_flat[i]

        tla.set_flag(simt_done)
        tla.wait_flag(simt_done)
        tla.copy(out_tile, ub_simt_result)

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

_SENTINEL = -999.0


# Arch selection is the only Host compile knob now; caching moved to env vars
# (dsl e745bf10 converged the Host surface). --force-recompile / --no-cache are
# kept as flags and translated here so the runner scripts keep working.
NPU_ARCH = "--npu-arch 3510"


def _apply_cache_env(args) -> None:
    import os

    if getattr(args, "force_recompile", False):
        os.environ["CATLASS_DSL_FORCE_RECOMPILE"] = "1"
    if getattr(args, "no_cache", False):
        os.environ["CATLASS_DSL_CACHE"] = "0"
    if getattr(args, "cache_dir", None):
        os.environ["CATLASS_DSL_CACHE_DIR"] = str(args.cache_dir)


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    def as_tla(dev_buf):
        return from_dlpack(dev_buf.contiguous(), layout_tag=tla.arch.RowMajor)

    _apply_cache_env(args)

    torch.npu.set_device(args.device)
    print(
        f"--- simt_mixed {M_DIM}x{K_DIM}x{N_DIM} cube -> simd -> simt "
        f"chained through UB, block={THREADS} ---"
    )

    lhs = (
        torch.arange(M_DIM * K_DIM, dtype=torch.float32, device="npu") % 7.0 - 3.0
    ).reshape(M_DIM, K_DIM)
    rhs = (
        torch.arange(K_DIM * N_DIM, dtype=torch.float32, device="npu") % 5.0 - 2.0
    ).reshape(K_DIM, N_DIM)
    addend_simd = (
        torch.arange(N_ELE, dtype=torch.float32, device="npu") % 11.0
    ).reshape(M_DIM, N_DIM)
    addend_simt = (
        torch.arange(N_ELE, dtype=torch.float32, device="npu") % 3.0 - 1.0
    ).reshape(M_DIM, N_DIM)
    mid = torch.full((M_DIM, N_DIM), _SENTINEL, dtype=torch.float32, device="npu")
    out = torch.full((M_DIM, N_DIM), _SENTINEL, dtype=torch.float32, device="npu")

    expected_mid = lhs @ rhs + addend_simd
    expected_out = expected_mid + addend_simt

    tensors = [
        as_tla(lhs),
        as_tla(rhs),
        as_tla(addend_simd),
        as_tla(addend_simt),
        as_tla(mid),
        as_tla(out),
    ]
    artifact = tla.compile(
        simt_mixed,
        *tensors,
        options=NPU_ARCH,
    )
    # Back the region with exactly the result buffer it holds, and no more.
    result_ub_bytes = SIMD_ELE * 4
    print(
        f"compiled UB {artifact.get_kernel_ub_size()} B static, "
        f"launch adds {result_ub_bytes} B dynamic "
        f"(up to {artifact.max_ub_bytes} B available)"
    )
    artifact(*tensors, block_num=1, ub=result_ub_bytes)
    torch.npu.synchronize()

    simd_ok = bool(torch.allclose(mid, expected_mid, rtol=1e-5, atol=1e-4))
    simt_ok = bool(torch.allclose(out, expected_out, rtol=1e-5, atol=1e-4))
    untouched = int((mid.view(-1) == _SENTINEL).sum()) + int(
        (out.view(-1) == _SENTINEL).sum()
    )

    for name, got, want_t in (("simd", mid, expected_mid), ("simt", out, expected_out)):
        if not bool(torch.allclose(got, want_t, rtol=1e-5, atol=1e-4)):
            flat, want = got.view(-1), want_t.view(-1)
            k = int((flat - want).abs().argmax())
            print(
                f"        {name} worst at {k} (row {k // N_DIM}): got={float(flat[k])} want={float(want[k])}"
            )

    passed = simd_ok and simt_ok
    print(f"untouched={untouched}/{2 * N_ELE}")
    print(f"simd_ok={simd_ok} simt_ok={simt_ok}")
    print(f"passed={passed} cache_key={artifact.cache_key}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile and run a kernel mixing cube, SIMD and SIMT work through UB."
    )
    parser.add_argument("--device", type=int, default=0)
    # Default None, not this example's own artifacts directory: setting it
    # exports CATLASS_DSL_CACHE_DIR, and bc_compile resolves the *BC* cache
    # through that same variable -- so a per-example default sends the BC
    # lookup somewhere nothing has compiled it. Pass --cache-dir to opt in.
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
