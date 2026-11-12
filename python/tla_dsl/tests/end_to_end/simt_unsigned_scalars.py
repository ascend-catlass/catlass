"""Unsigned SIMT scalar regression; run while holding the project's NPU lock."""

from __future__ import annotations

import argparse
import json
import os
import tempfile

from catlass import tla


def make_kernel(unsigned_type, width, threads):
    @tla.kernel
    def unsigned_scalars(source: tla.Tensor, output: tla.Tensor) -> None:
        block = tla.arch.block_idx()
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=threads):
                tx, _, _ = tla.arch.thread_idx()
                index = block * threads + tx
                value = unsigned_type(source[index])
                high = unsigned_type(1 << (width - 1))
                ones = unsigned_type((1 << width) - 1)
                offset = 16 + index * 11
                output[offset] = value.to(tla.Int64)
                output[offset + 1] = (value > unsigned_type((1 << (width - 1)) - 1)).to(
                    tla.Int64
                )
                output[offset + 2] = unsigned_type(ones.ir_value()).to(tla.Int64)
                output[offset + 3] = value.to(tla.UInt64).to(tla.Int64)
                output[offset + 4] = source[index].to(tla.Int64)
                output[offset + 5] = (value & high).to(tla.Int64)
                output[offset + 6] = (value | unsigned_type(1)).to(tla.Int64)
                output[offset + 7] = (value ^ ones).to(tla.Int64)
                output[offset + 8] = (value >> unsigned_type(1)).to(tla.Int64)
                output[offset + 9] = (value << unsigned_type(1)).to(tla.Int64)
                output[offset + 10] = abs(value).to(tla.Int64)
            tla.pipe_barrier(tla.pipes.ALL)

    return unsigned_scalars


def check_print(unsigned_type, source, width, signed_value):
    @tla.kernel
    def print_values(source: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=1):
                value = unsigned_type(source[0])
                tla.print(value)
                tla.print("unsigned={} signed={}", value, source[0])

    artifact = tla.compile(print_values, source, options="--npu-arch 3510")
    with tempfile.TemporaryFile() as captured:
        stdout = os.dup(1)
        try:
            os.dup2(captured.fileno(), 1)
            artifact(source, block_num=1)
        finally:
            os.dup2(stdout, 1)
            os.close(stdout)
        captured.seek(0)
        text = captured.read().decode()
    expected = signed_value & ((1 << width) - 1)
    if (
        f"x={expected}" not in text
        or f"unsigned={expected} signed={signed_value}" not in text
    ):
        raise AssertionError(f"Incorrect unsigned print output: {text}")


def check_unsigned_tensors(unsigned_type, width, dtype, threads):
    import torch
    from catlass.tla.runtime import from_dlpack

    @tla.kernel
    def kernel(arg: object, src: tla.Tensor, unsigned_src: tla.Tensor, dst: tla.Tensor):
        block = tla.arch.block_idx()
        with tla.vector():
            captured = unsigned_src[3]
            with tla.vec.func(mode="simt", thread_block_dim=threads):
                tid, _, _ = tla.arch.thread_idx()
                i = block * threads + tid
                dst[16 + i * 8] = unsigned_type(src[i])
                dst[17 + i * 8] = unsigned_src[i] + unsigned_type(1)
                dst[18 + i * 8] = unsigned_src[i] >> unsigned_type(1)
                if tid < 16:
                    dst[19 + i * 8] = captured + unsigned_type(1)
                else:
                    dst[19 + i * 8] = captured + unsigned_type(1)
                dst[20 + i * 8] = arg >> unsigned_type(1)
                dst[21 + i * 8] = captured // unsigned_type(3)
                dst[22 + i * 8] = arg % unsigned_type(3)
            with tla.vec.func(mode="simt", thread_block_dim=threads):
                tid, _, _ = tla.arch.thread_idx()
                i = block * threads + tid
                dst[23 + i * 8] = tla.where(arg > unsigned_type(1), captured, arg)

    count = threads * 3
    high = 1 << (width - 1)
    values = [0, 1, -high, -1, -2, high - 1]
    values = [values[i % len(values)] for i in range(count)]
    src = torch.tensor(values, dtype=dtype, device="npu")
    unsigned_dtype = getattr(torch, f"uint{width}")
    unsigned_src = src.view(unsigned_dtype)
    output = torch.full((count * 8 + 32,), -13, dtype=dtype, device="npu")
    buffers = (src, unsigned_src, output.view(unsigned_dtype))
    tensors = [from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in buffers]
    artifact = tla.compile(kernel, unsigned_type(high), *tensors, options="--npu-arch 3510")
    artifact(unsigned_type(high), *tensors, block_num=3)
    torch.npu.synchronize()
    mask = (1 << width) - 1
    expected = []
    for v in values:
        for x in (v & mask, (v + 1) & mask, (v & mask) >> 1, 0, high >> 1, mask // 3, high % 3, mask):
            expected.append(x if x < high else x - (1 << width))
    actual = output.cpu()
    if not torch.all(actual[:16] == -13) or not torch.all(actual[-16:] == -13):
        raise AssertionError("Unsigned tensor output guards changed")
    torch.testing.assert_close(actual[16:-16], torch.tensor(expected, dtype=dtype), rtol=0, atol=0)


def signed64(value):
    value &= (1 << 64) - 1
    return value if value < 1 << 63 else value - (1 << 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    for unsigned_type, width, dtype in (
        (tla.UInt8, 8, torch.int8),
        (tla.UInt16, 16, torch.int16),
        (tla.UInt32, 32, torch.int32),
        (tla.UInt64, 64, torch.int64),
    ):
        high = 1 << (width - 1)
        ones = (1 << width) - 1
        if width <= 32:
            for signed_value in (-1, -2, -high):
                print_source = torch.tensor([signed_value], dtype=dtype, device="npu")
                tensor = from_dlpack(print_source, layout_tag=tla.arch.RowMajor)
                check_print(unsigned_type, tensor, width, signed_value)
        for threads in (32, 49):
            check_unsigned_tensors(unsigned_type, width, dtype, threads)
            count = threads * 3
            samples = (0, 1, high - 1, -high, -1, -2, 2)
            values = [samples[i % len(samples)] for i in range(count)]
            source = torch.tensor(values, dtype=dtype, device="npu")
            output = torch.full(
                (count * 11 + 32,), -13, dtype=torch.int64, device="npu"
            )
            buffers = (source, output)
            tensors = [from_dlpack(b, layout_tag=tla.arch.RowMajor) for b in buffers]
            kernel = make_kernel(unsigned_type, width, threads)
            artifact = tla.compile(kernel, *tensors, options="--npu-arch 3510")
            artifact(*tensors, block_num=3)
            torch.npu.synchronize()
            rows = []
            for value in values:
                u = value & ones
                rows.append(
                    [
                        signed64(v)
                        for v in (
                            u,
                            int(u > high - 1),
                            ones,
                            u,
                            value,
                            u & high,
                            u | 1,
                            u ^ ones,
                            u >> 1,
                            (u << 1) & ones,
                            u,
                        )
                    ]
                )
            expected = torch.tensor(rows, dtype=torch.int64)
            actual = output.cpu()
            if not torch.all(actual[:16] == -13):
                raise AssertionError("Leading output guard was modified")
            if not torch.all(actual[-16:] == -13):
                raise AssertionError("Trailing output guard was modified")
            torch.testing.assert_close(
                actual[16:-16].reshape(count, 11), expected, rtol=0, atol=0
            )
            print(
                json.dumps(
                    {
                        "width": width,
                        "threads": threads,
                        "blocks": 3,
                        "threads_checked": count,
                        "passed": True,
                        "binary": str(artifact.kernel_binary_path),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
