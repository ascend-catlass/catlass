"""Device checks for inplace_mul_add and preservation of its first input."""

import argparse

import catlass.tla as tla


def make_kernel(dtype, lanes, valid):
    @tla.kernel
    def compute(source: tla.Tensor, output: tla.Tensor, count: tla.Int32):
        src_ptr = tla.allocate(3 * lanes, dtype, tla.AddressSpace.ub, 256)
        out_ptr = tla.allocate(7 * lanes, dtype, tla.AddressSpace.ub, 256)
        src_ub = tla.make_tensor_like(src_ptr, source, tla.arch.RowMajor)
        out_ub = tla.make_tensor_like(out_ptr, output, tla.arch.RowMajor)
        layout = tla.make_layout(tla.make_shape(valid), tla.make_stride(1))
        ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(src_ub, source)
            tla.copy(out_ub, output)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                dst = tla.make_tensor(src_ptr, layout).load()
                factor = tla.make_tensor(src_ptr + lanes, layout).load()
                bias = tla.make_tensor(src_ptr + 2 * lanes, layout).load()
                result = tla.arch.inplace_mul_add(dst, factor, bias)
                prefix, _ = tla.update_mask(count, dtype=dtype)
                active = tla.bitwise_and(
                    prefix, tla.create_mask(pattern=tla.mask.M3, dtype=dtype)
                )
                masked = tla.arch.inplace_mul_add(dst, factor, bias, mask=active)
                # Reactivate exactly the lanes skipped by the first masked update.
                reactivated = tla.bitwise_not(active)
                resumed = tla.arch.inplace_mul_add(
                    masked, factor, bias, mask=reactivated
                )
                kept = tla.where(active, masked, dst)
                resumed_kept = tla.arch.inplace_mul_add(
                    kept, factor, bias, mask=reactivated
                )
                # Full logical stores expose inactive lanes; rows 0 and 6 are guards.
                tla.make_tensor(out_ptr + lanes, layout).store(result)
                tla.make_tensor(out_ptr + 2 * lanes, layout).store(masked)
                tla.make_tensor(out_ptr + 3 * lanes, layout).store(dst)
                tla.make_tensor(out_ptr + 4 * lanes, layout).store(resumed)
                tla.make_tensor(out_ptr + 5 * lanes, layout).store(resumed_kept)
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(output, out_ub)
            tla.pipe_barrier(tla.pipes.ALL)

    return compute


def samples(dtype):
    import torch

    info = torch.finfo(dtype)
    eps = info.eps
    step = 1024 * eps  # One output ULP near 1536.
    return torch.tensor(
        [
            # dst, multiplier, addend, independently rounded result
            (23, 2, 3, 49),
            (-2, 4, 7, -1),
            (1 + eps, 1 - eps, -1, -(eps * eps)),
            (1 + eps, 1536, -(2**-14), 1536 + step),
            (1 + eps, 1536, 0, 1536 + 2 * step),
            (1 + eps, 1536, 2**-14, 1536 + 2 * step),
            (-0.0, 1, -0.0, -0.0),
            (info.tiny, 1, 0, info.tiny),
            (info.max, 2, 0, float("inf")),
            (float("inf"), 0, 1, float("nan")),
            (float("nan"), 1, 0, float("nan")),
        ],
        dtype=dtype,
    )


def main():
    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("f16", "bf16", "f32"), required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    dtype, torch_dtype = {
        "f16": (tla.Float16, torch.float16),
        "bf16": (tla.BFloat16, torch.bfloat16),
        "f32": (tla.Float32, torch.float32),
    }[args.dtype]
    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    lanes = 2048 // dtype.width
    data = samples(torch_dtype)
    indices = torch.arange(lanes)
    for valid in (lanes, 13):
        kernel = make_kernel(dtype, lanes, valid)
        compiled = None
        for run, count in enumerate((0, 13, lanes)):
            # Reuse the object with different values and runtime predicate lengths.
            dst, factor, bias, wanted = data[(indices + run) % len(data)].T
            source = torch.stack((dst, factor, bias)).flatten()
            expected = torch.full((7, lanes), -17, dtype=torch_dtype)
            device_src = source.to(device)
            device_out = expected.flatten().to(device)
            source_view = tla.from_dlpack(device_src, layout_tag=tla.arch.RowMajor)
            output_view = tla.from_dlpack(device_out, layout_tag=tla.arch.RowMajor)
            if compiled is None:
                compiled = tla.compile(kernel, source_view, output_view, count)
            compiled(source_view, output_view, count, block_num=1)
            torch.npu.synchronize()
            expected[1, :valid] = wanted[:valid]
            active = (indices < count) & (indices % 3 == 0)
            expected[2, :valid] = torch.where(active, wanted, 0.0)[:valid]
            expected[3, :valid] = dst[:valid]
            from_zero = torch.zeros_like(dst) * factor + bias
            expected[4, :valid] = torch.where(active, 0.0, from_zero)[:valid]
            expected[5, :valid] = torch.where(active, 0.0, wanted)[:valid]
            actual = device_out.cpu().reshape_as(expected)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            zeros = expected == 0
            torch.testing.assert_close(
                actual[zeros].signbit(), expected[zeros].signbit()
            )
            torch.testing.assert_close(
                device_src.cpu().view(torch.uint8), source.view(torch.uint8)
            )
    print(f"{args.dtype}: full and short vectors, 6 calls passed")


if __name__ == "__main__":
    main()
