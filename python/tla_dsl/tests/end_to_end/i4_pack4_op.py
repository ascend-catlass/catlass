"""Device regression for Int4x2 -> f16/bf16/i16 widening.

Source is a packed int4 buffer (low nibble first). UNPACK4_B8 expands one
64-byte view into the full int4x2 register; reg_slot ONE selects the populated
quarter, giving the 128 logical int4 values in order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catlass import tla
from catlass.params import CastParams, LoadDist, NormalLoadParams, RegSlot, RoundMode
from catlass.tla.runtime import from_dlpack, make_fake_tensor


def _fake(dtype, count):
    return make_fake_tensor(
        dtype,
        (count,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(count,),
        layout_tag=tla.arch.RowMajor,
    )


def _decode_i4(raw: np.ndarray) -> np.ndarray:
    low = raw & 0xF
    high = (raw >> 4) & 0xF
    values = np.empty(raw.size * 2, dtype=np.int8)
    values[0::2] = low
    values[1::2] = high
    return values - 16 * (values > 7)


def make_kernel(dst_dtype):
    @tla.kernel
    def convert(source: tla.Tensor, destination: tla.Tensor) -> None:
        src_ptr = tla.allocate(64, tla.Int4x2, tla.AddressSpace.ub, 256)
        dst_ptr = tla.allocate(128, dst_dtype, tla.AddressSpace.ub, 256)
        src_tile = tla.make_tensor(
            src_ptr, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
        )
        dst_tile = tla.make_tensor(
            dst_ptr, tla.make_layout(tla.make_shape(128), tla.make_stride(1))
        )
        ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(src_tile, source)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                expanded = src_tile.load(
                    NormalLoadParams(load_dist=LoadDist.DIST_UNPACK4_B8)
                )
                dst_tile.store(
                    expanded.to(
                        dst_dtype,
                        CastParams(
                            reg_slot=RegSlot.ZERO,
                            round_mode=RoundMode.CAST_NONE,
                        ),
                    )
                )
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(destination, dst_tile)

    return convert


def _torch_dtype(name: str):
    import torch

    return {
        "Float16": torch.float16,
        "BFloat16": torch.bfloat16,
        "Int16": torch.int16,
    }[name]


def run(args):
    records = []
    raw_cases = []
    for repeat in range(2):
        raw = np.empty(64, np.uint8)
        for index in range(raw.size):
            low = (2 * index + repeat) % 16
            high = (2 * index + 1 + repeat) % 16
            raw[index] = (high << 4) | low
        raw_cases.append(raw)
    if args.run:
        import torch
        import torch_npu  # noqa: F401 -- register the device backend

        torch.npu.set_device(args.device)

    for dst_dtype in (tla.Float16, tla.BFloat16, tla.Int16):
        kernel = make_kernel(dst_dtype)
        compiled = kernel.compile(
            type_args=(_fake(tla.Int4x2, 64), _fake(dst_dtype, 128)),
            options="--npu-arch 3510",
        )
        entry = {
            "case": f"i4-to-{dst_dtype.__name__}",
            "kernel": str(compiled.kernel_binary_path),
            "launches": 0,
        }
        if args.run:
            source_device = torch.from_numpy(raw_cases[0].copy()).to(
                f"npu:{args.device}"
            )
            destination_device = torch.zeros(
                128, dtype=_torch_dtype(dst_dtype.__name__)
            ).to(f"npu:{args.device}")
            source_view = from_dlpack(
                source_device,
                element_type=tla.Int4x2,
                origin_shape=(64,),
                layout_tag=tla.arch.RowMajor,
            )
            destination_view = from_dlpack(
                destination_device, layout_tag=tla.arch.RowMajor
            )
            for raw in raw_cases:
                source_device.copy_(torch.from_numpy(raw.copy()))
                destination_device.zero_()
                compiled(source_view, destination_view, block_num=1)
                torch.npu.synchronize(args.device)
                expected = torch.tensor(
                    _decode_i4(raw), dtype=_torch_dtype(dst_dtype.__name__)
                )
                np.testing.assert_array_equal(
                    destination_device.cpu().to(torch.float32).numpy(),
                    expected.to(torch.float32).numpy(),
                )
                entry["launches"] += 1
        records.append(entry)
        print(f"case={entry['case']}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    records = run(args)
    print(
        json.dumps(
            dict(passed=True, cases=len(records), launches=sum(x["launches"] for x in records)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
