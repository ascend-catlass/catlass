"""Device regressions for packed Int4x2 register casts.

Source is a packed int4 buffer (low nibble first). UNPACK4_B8 expands one
64-byte view into the full int4x2 register. Widening selects the populated
quarter, giving the 128 logical int4 values in order.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from catlass import tla
from catlass.params import (
    CastParams,
    LoadDist,
    NormalLoadParams,
    RegSlot,
    RoundMode,
    SatMode,
)
from catlass.tla.runtime import from_dlpack, make_fake_tensor
from register_casts_op import check_masked_casts


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
        ready = tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR)
        done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
        with tla.vector():
            tla.copy(src_tile, source)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                expanded = src_tile.load(
                    NormalLoadParams(load_dist=LoadDist.DIST_UNPACK4_B8)
                )
                if tla.const_expr(dst_dtype is tla.Int8):
                    half = expanded.to(tla.Float16, CastParams())
                    dst_tile.store(
                        half.to(tla.Int8, CastParams(round_mode=RoundMode.CAST_RINT))
                    )
                else:
                    dst_tile.store(expanded.to(dst_dtype, CastParams()))
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(destination, dst_tile)

    return convert


def _packed_i4(phase):
    low = (2 * np.arange(64, dtype=np.uint8) + phase) & 0xF
    return (((low + 1) & 0xF) << 4) | low


def _run_unpack(args, torch):
    to_i8 = args.case == "to-i8"
    raw = _packed_i4(0)
    raw_cases = (raw, np.roll(raw, 1) if to_i8 else _packed_i4(1))
    destinations = (tla.Int8,) if to_i8 else (tla.Float16, tla.BFloat16, tla.Int16)
    launches = 0
    if torch is not None:
        device = f"npu:{args.device}"
        source = torch.from_numpy(raw.copy()).to(device)
        source_view = from_dlpack(
            source,
            element_type=tla.Int4x2,
            origin_shape=(64,),
            layout_tag=tla.arch.RowMajor,
        )
        torch_types = {
            tla.Float16: torch.float16,
            tla.BFloat16: torch.bfloat16,
            tla.Int16: torch.int16,
            tla.Int8: torch.int8,
        }

    for dst_dtype in destinations:
        compiled = make_kernel(dst_dtype).compile(
            type_args=(_fake(tla.Int4x2, 64), _fake(dst_dtype, 128)),
            options="--npu-arch 3510",
        )
        if torch is not None:
            destination = torch.zeros(128, dtype=torch_types[dst_dtype]).to(device)
            destination_view = from_dlpack(destination, layout_tag=tla.arch.RowMajor)
            for packed in raw_cases:
                source.copy_(torch.from_numpy(packed))
                destination.zero_()
                compiled(source_view, destination_view, block_num=1)
                torch.npu.synchronize(args.device)
                expected = _decode_i4(packed)
                if to_i8:
                    # F16 -> I8 writes slot ZERO at even byte positions.
                    actual = destination.cpu().numpy()
                    np.testing.assert_array_equal(actual[::2], expected[:64])
                    np.testing.assert_array_equal(actual[1::2], np.zeros(64, np.int8))
                else:
                    actual = destination.cpu().to(torch.float32).numpy()
                    np.testing.assert_array_equal(actual, expected)
                launches += 1
        label = "i8" if to_i8 else dst_dtype.__name__
        print(f"case=i4-to-{label}", flush=True)
    return len(destinations), launches


def _round_int4(source, mode):
    if mode is RoundMode.CAST_ROUND:
        return np.copysign(np.floor(np.abs(source) + 0.5), source)
    return {
        RoundMode.CAST_RINT: np.rint,
        RoundMode.CAST_FLOOR: np.floor,
        RoundMode.CAST_CEIL: np.ceil,
        RoundMode.CAST_TRUNC: np.trunc,
    }[mode](source)


def _run_narrow(args):
    values = np.array(
        [-17, -9, -8, -7, -3, -1, 0, 1, 2, 5, 6, 7, 8, 9, 18], dtype=np.int16
    )
    halves = np.concatenate(
        [values, [-8.5, -7.5, -2.5, -1.5, -0.5, -0.25, 0.25, 0.5, 1.5, 2.5, 6.5, 7.5]]
    ).astype(np.float16)
    for dtype, samples in ((tla.Int16, values), (tla.Float16, halves)):
        source = np.resize(samples, (3, 128))
        default_mode = (
            RoundMode.CAST_RINT if dtype is tla.Float16 else RoundMode.CAST_NONE
        )
        traits = [
            CastParams(reg_slot=slot, sat_mode=sat, round_mode=default_mode)
            for slot in RegSlot
            for sat in (SatMode.SAT, SatMode.NOSAT)
        ]
        if dtype is tla.Float16:
            traits.extend(
                CastParams(reg_slot=RegSlot.ONE, sat_mode=SatMode.SAT, round_mode=mode)
                for mode in (
                    RoundMode.CAST_ROUND,
                    RoundMode.CAST_FLOOR,
                    RoundMode.CAST_CEIL,
                    RoundMode.CAST_TRUNC,
                )
            )
        cases = [_narrow_case(dtype, source, params) for params in traits]
        compiled = check_masked_casts(
            args.device, not args.run, dtype, tla.Int4x2, source, cases
        )
        if not compiled.kernel_binary_path.is_file():
            raise FileNotFoundError(
                f"Missing compiled kernel: {compiled.kernel_binary_path}"
            )


def _narrow_case(dtype, source, params):
    rounded = _round_int4(source, params.round_mode) if dtype is tla.Float16 else source
    if params.sat_mode is SatMode.SAT:
        rounded = np.clip(rounded, -8, 7)
    nibbles = rounded.astype(np.int16) & 0xF
    packed = nibbles[:, ::2] | (nibbles[:, 1::2] << 4)
    expected = np.zeros((2, source.shape[0], 256), dtype=np.uint8)
    slot = tuple(RegSlot).index(params.reg_slot)
    pairs = np.arange(64)
    expected[0, :, slot::4] = packed
    expected[1, :, slot::4] = np.where(pairs % 3 == 0, packed, 0)
    # Exercise both legal mask widths, with M3 aligned to the even source lanes.
    return params, dtype if slot % 2 else tla.UInt32, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("widen", "to-i8", "narrow"), default="widen")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    torch = None
    if args.run:
        import torch
        import torch_npu  # noqa: F401 -- register the device backend

        torch.npu.set_device(args.device)
    if args.case == "narrow":
        _run_narrow(args)
        return
    cases, launches = _run_unpack(args, torch)
    print(json.dumps(dict(passed=True, cases=cases, launches=launches), indent=2))


if __name__ == "__main__":
    main()
