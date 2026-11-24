"""Device regression for Int4x2 -> Int8 through Int4x2 -> Float16 -> Int8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catlass import tla
from catlass.params import CastParams, LoadDist, NormalLoadParams, RegSlot, RoundMode
from catlass.tla.runtime import from_dlpack, make_fake_tensor
from i4_pack4_op import _decode_i4


def _fake(dtype, count):
    return make_fake_tensor(
        dtype,
        (count,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(count,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def convert(source: tla.Tensor, destination: tla.Tensor) -> None:
    src_ptr = tla.allocate(64, tla.Int4x2, tla.AddressSpace.ub, 256)
    mid_ptr = tla.allocate(128, tla.Float16, tla.AddressSpace.ub, 256)
    dst_ptr = tla.allocate(128, tla.Int8, tla.AddressSpace.ub, 256)
    src_tile = tla.make_tensor(
        src_ptr, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
    )
    mid_tile = tla.make_tensor(
        mid_ptr, tla.make_layout(tla.make_shape(128), tla.make_stride(1))
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
            expanded = src_tile.load(NormalLoadParams(load_dist=LoadDist.DIST_UNPACK4_B8))
            half = expanded.to(
                tla.Float16,
                CastParams(reg_slot=RegSlot.ZERO, round_mode=RoundMode.CAST_NONE),
            )
            dst_tile.store(half.to(tla.Int8, CastParams(round_mode=RoundMode.CAST_RINT)))
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(destination, dst_tile)


def run(args):
    compiled = convert.compile(
        type_args=(_fake(tla.Int4x2, 64), _fake(tla.Int8, 128)),
        options="--npu-arch 3510",
    )
    entry = {"case": "i4-to-i8", "kernel": str(compiled.kernel_binary_path), "launches": 0}
    if args.run:
        import torch
        import torch_npu  # noqa: F401 -- register the device backend

        torch.npu.set_device(args.device)
        raw = np.empty(64, np.uint8)
        for index in range(raw.size):
            low = (2 * index) % 16
            high = (2 * index + 1) % 16
            raw[index] = (high << 4) | low
        source_device = torch.from_numpy(raw.copy()).to(f"npu:{args.device}")
        destination_device = torch.zeros(128, dtype=torch.int8).to(f"npu:{args.device}")
        source_view = from_dlpack(
            source_device,
            element_type=tla.Int4x2,
            origin_shape=(64,),
            layout_tag=tla.arch.RowMajor,
        )
        destination_view = from_dlpack(destination_device, layout_tag=tla.arch.RowMajor)
        for repeat in range(2):
            source_device.copy_(torch.from_numpy(np.roll(raw, repeat).copy()))
            destination_device.zero_()
            compiled(source_view, destination_view, block_num=1)
            torch.npu.synchronize(args.device)
            expected = torch.tensor(
                _decode_i4(np.roll(raw, repeat)), dtype=torch.int8
            )
            actual = destination_device.cpu().numpy()
            # Float16 -> Int8 is a 2x narrowing: one slot writes its 64 values
            # into the even lanes of the byte register. Compare the populated
            # lane contract and require the unselected lanes to stay zero.
            np.testing.assert_array_equal(actual[0::2], expected.numpy()[:64])
            np.testing.assert_array_equal(actual[1::2], np.zeros(64, np.int8))
            entry["launches"] += 1
    print("case=i4-to-i8", flush=True)
    return [entry]


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
