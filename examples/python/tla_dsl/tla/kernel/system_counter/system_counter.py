"""Fenced issuer-local Vector timing; no profiler or custom launch ABI.

Run on a selected device with ``--device 0`` after applying the normal mask.
The result is elapsed system ticks, not Vector-active time or pure wait time.
"""

from __future__ import annotations

import argparse
import json
import statistics

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


@tla.kernel
def timed_vector_add(
    a_gm: tla.Tensor,
    b_gm: tla.Tensor,
    c_gm: tla.Tensor,
    counters: tla.Tensor,
    passes: tla.Int32,
):
    a_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    b_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    a = tla.make_tensor_like(a_ptr, a_gm, tla.arch.RowMajor)
    b = tla.make_tensor_like(b_ptr, b_gm, tla.arch.RowMajor)
    ready = tla.flag("inputs_ready", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("output_ready", tla.arch.VECTOR, tla.arch.MTE3)
    retired = tla.flag("vector_retired", tla.arch.VECTOR, tla.arch.SCALAR)
    with tla.vector():
        tla.copy(a, a_gm)
        tla.copy(b, b_gm)
        tla.set_flag(ready)
        tla.wait_flag(ready)
        tla.pipe_barrier(tla.pipes.ALL)
        before = tla.arch.get_system_cycle()
        for i in tla.range(passes):
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                a.store(tla.add(a.load(), b.load(), mask=mask), mask=mask)
        tla.set_flag(retired)
        tla.wait_flag(retired)
        after = tla.arch.get_system_cycle()
        counters[0] = before
        counters[1] = after
        counters[2] = after - before
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(c_gm, a)
        tla.pipe_barrier(tla.pipes.ALL)


def validate_counter_sample(row):
    """Check counter records even when Python optimization disables asserts."""
    if len(row) != 8 or row[3:] != [-1] * 5:
        raise RuntimeError(f"Counter guard corruption: {row}")
    if row[2] != (row[1] - row[0]) % (1 << 64):
        raise RuntimeError(f"Counter delta mismatch: {row}")


def summarize_counter_samples(passes, samples):
    elapsed = [row[2] for row in samples]
    if passes >= 32 and not any(elapsed):
        raise RuntimeError(
            f"System counter did not advance over {passes} Vector passes"
        )
    return {
        "passes": passes,
        "samples": len(samples),
        "elapsed_ticks_median": statistics.median(elapsed),
    }


def main():
    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--passes", type=int, nargs="+", default=[0, 32, 256, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if min(args.passes) < 0 or args.warmup < 0 or args.samples <= 0:
        parser.error("passes/warmup must be nonnegative; samples must be positive")
    torch.npu.set_device(args.device)
    a = torch.ones(64, device=f"npu:{args.device}")
    b = torch.ones_like(a)
    c = torch.empty_like(a)
    counters = torch.full((8,), -1, dtype=torch.int64, device=a.device)
    tensors = []
    for tensor in (a, b, c, counters):
        tensors.append(from_dlpack(tensor, layout_tag=tla.arch.RowMajor))
    artifact = tla.compile(
        timed_vector_add, *tensors, tla.Int32(0), options="--npu-arch 3510"
    )
    for passes in args.passes:
        values = []
        for launch in range(args.warmup + args.samples):
            counters.fill_(-1)
            artifact(*tensors, tla.Int32(passes), block_num=1)
            torch.npu.synchronize()
            torch.testing.assert_close(
                c.cpu(), torch.full((64,), float(1 + passes)), rtol=0, atol=0
            )
            row = counters.cpu().tolist()
            validate_counter_sample(row)
            if launch >= args.warmup:
                values.append(row)
        print(json.dumps(summarize_counter_samples(passes, values)), flush=True)


if __name__ == "__main__":
    main()
