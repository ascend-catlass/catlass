# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device regressions for register casts, rounding, and E8M0 scale layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

import catlass.tla as tla
from catlass.params import CastParams, RegSlot, RoundMode, SatMode
from catlass.tla.runtime import from_dlpack, make_fake_tensor

VL_ELE = 64
NUM_CHUNKS = 4
VECTOR_ELE = NUM_CHUNKS * VL_ELE
ALL_DTYPES = ("f32",)
_HALF_ULP_AT_ONE = 2.0**-11

_TRAIT_NONE = tla.params.CastParams()
_TRAIT_RINT = tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_RINT)
_TRAIT_ROUND = tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_ROUND)
_TRAIT_ODD = tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_ODD)


@tla.kernel
def cast_round_modes_op(
    mem_a: tla.Tensor,
    mem_rint: tla.Tensor,
    mem_default: tla.Tensor,
    mem_round: tla.Tensor,
    mem_odd: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    a_ub = _make_ub_tensor(mem_a)
    rint_ub = _make_ub_tensor(mem_rint)
    default_ub = _make_ub_tensor(mem_default)
    round_ub = _make_ub_tensor(mem_round)
    odd_ub = _make_ub_tensor(mem_odd)

    with tla.vector():
        tla.copy(a_ub, mem_a)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range(NUM_CHUNKS):
                a_v = _chunk(a_ub, i).load()
                # The rounded integers are small enough for exact Float32 output.
                rint = a_v.to(tla.Int32, _TRAIT_RINT).to(tla.Float32, _TRAIT_ROUND)
                default = a_v.to(tla.Int32, _TRAIT_NONE).to(tla.Float32, _TRAIT_ROUND)
                rounded = a_v.to(tla.Int32, _TRAIT_ROUND).to(tla.Float32, _TRAIT_ROUND)
                odd = a_v.to(tla.Float16, _TRAIT_ODD).to(tla.Float32, _TRAIT_NONE)
                _chunk(rint_ub, i).store(rint)
                _chunk(default_ub, i).store(default)
                _chunk(round_ub, i).store(rounded)
                _chunk(odd_ub, i).store(odd)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(mem_rint, rint_ub)
        tla.copy(mem_default, default_ub)
        tla.copy(mem_round, round_ub)
        tla.copy(mem_odd, odd_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(
        tensor,
        tla.make_shape(VL_ELE),
        tla.make_coord(chunk_idx),
    )


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "cast_round_modes": {
            # Integer cast results packed back to f32 are exact here.
            "default_atol": 0.0,
        },
    }


def _make_inputs(_args: Any, _dtype_name: str, torch: Any) -> tuple[Any, ...]:
    # Include ties for both integer rounding and Float16 narrowing.
    pattern = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            -0.5,
            -1.0,
            -1.5,
            -2.5,
            4.25,
            -4.25,
            8.0,
            -8.0,
            1.0 + _HALF_ULP_AT_ONE,
            -(1.0 + _HALF_ULP_AT_ONE),
        ],
        dtype=torch.float32,
        device="npu",
    )
    idx = torch.arange(VECTOR_ELE, device="npu")
    return (pattern[idx % pattern.numel()],)


def _expected(_op_name: str, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    import torch

    (a,) = inputs
    host = a.cpu()
    magnitude = host.abs()
    rint = host.round()
    rounded = torch.copysign((magnitude + 0.5).floor(), host)
    # All other inputs are exact in Float16. At the midpoint near 1.0, the
    # odd neighbour is one Float16 ULP farther from zero than 1.0.
    odd = torch.where(
        magnitude == 1.0 + _HALF_ULP_AT_ONE,
        host.sign() * (1.0 + 2.0 * _HALF_ULP_AT_ONE),
        host,
    )
    return rint, rint, rounded, odd


def _run_round_modes(remaining):
    repo_root = Path(__file__).resolve().parents[4]
    vector_simd_ops = str(
        repo_root
        / "examples"
        / "python"
        / "tla_dsl"
        / "dsl_tutorials"
        / "vector_simd_ops"
    )

    if vector_simd_ops not in sys.path:
        # append (not insert(0)) to avoid shadowing installed packages (G.PSL.03).
        sys.path.append(vector_simd_ops)

    from vector_op_harness import (
        DirectVectorOpConfig,
        DirectVectorOpHarness,
        vector_kernel_config,
    )

    def _set_kernel_config(
        op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
    ) -> tuple[type[Any], Any, float | int]:
        if op_name not in _operator_specs():
            raise SystemExit("unknown cast_round_modes operator")
        del shape
        config = vector_kernel_config(dtype_name, (VECTOR_ELE,), ALL_DTYPES)
        return config.tla_dtype, config.torch_dtype, config.default_sentinel

    harness = DirectVectorOpHarness(
        DirectVectorOpConfig(
            description="Check Float32 register cast rounding on device.",
            kernel=cast_round_modes_op,
            all_dtypes=ALL_DTYPES,
            operator_specs=_operator_specs,
            set_kernel_config=_set_kernel_config,
            get_vector_elements=lambda: VECTOR_ELE,
            get_kernel_shape=lambda: (VECTOR_ELE,),
            make_inputs=_make_inputs,
            expected=_expected,
            unsupported_case=lambda _op_name, _dtype_name: False,
            print_skip=lambda _op_name, _dtype_name, _shape: None,
            script_path=Path(__file__).resolve(),
            float_dtypes=frozenset({"f32"}),
            output_count=4,
        )
    )
    saved_argv = sys.argv
    sys.argv = [sys.argv[0], *remaining]
    try:
        return harness.main()
    finally:
        sys.argv = saved_argv


# Each tile loads one 256-byte register and stores one 256-byte register.
TILES = 128
DEST_SIZE = TILES * 256


def _make_e8m0_kernel(source_type, slot=RegSlot.ZERO, partial=False):
    pscale = source_type is tla.Float32
    widening = source_type is tla.Float8E8M0
    target_type = tla.BFloat16 if widening else tla.Float8E8M0
    source_lanes = 2048 // source_type.width
    target_lanes = 2048 // target_type.width
    even = CastParams(RegSlot.ZERO, SatMode.SAT, RoundMode.CAST_RINT)
    odd = CastParams(RegSlot.ONE, SatMode.SAT, RoundMode.CAST_RINT)
    trait = CastParams(reg_slot=slot)

    @tla.kernel
    def convert(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_ub = tla.make_tensor_like(
            tla.allocate(TILES * source_lanes, source_type, tla.AddressSpace.ub, 256),
            src,
            tla.arch.RowMajor,
        )
        dst_ub = tla.make_tensor_like(
            tla.allocate(TILES * target_lanes, target_type, tla.AddressSpace.ub, 256),
            dst,
            tla.arch.RowMajor,
        )
        ready = tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR)
        done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
        with tla.vector():
            tla.copy(src_ub, src)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                mask = None
                if tla.const_expr(partial):
                    mask = tla.create_mask(dtype=source_type, pattern=tla.mask.VL3)
                for tile in tla.range(TILES):
                    value = tla.tile_view(
                        src_ub, tla.make_shape(source_lanes), tla.make_coord(tile)
                    ).load()
                    output = tla.tile_view(
                        dst_ub, tla.make_shape(target_lanes), tla.make_coord(tile)
                    )
                    if tla.const_expr(pscale):
                        a = value.to(tla.BFloat16, even).to(tla.Float8E8M0, trait)
                        b = value.to(tla.BFloat16, odd).to(tla.Float8E8M0, trait)
                        packed = tla.bitwise_or(
                            a.bitcast(tla.UInt8), b.bitcast(tla.UInt8)
                        )
                        padding = tla.full(0x7F, tla.Int8).bitcast(tla.UInt8)
                        first, _ = tla.deinterleave(packed, padding)
                        output.store(first.bitcast(tla.Float8E8M0))
                    else:
                        output.store(value.to(target_type, trait, mask))
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(dst, dst_ub)

    return convert


def _pscale_data():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2**32, TILES * 64, dtype=np.uint32)
    # Finite normalized FP32 values, with explicit BF16 rounding boundaries.
    bits = (bits & 0x807FFFFF) | (
        rng.integers(1, 253, bits.size, dtype=np.uint32) << 23
    )
    bits[:6] = [0x3FFF7FFF, 0x3FFF8000, 0x3FFF8001, 0xBFFF8000, 0, 0x80000000]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    exponents = ((rounded >> 7) & 0xFF).astype(np.uint8).reshape(TILES, 64)
    expected = np.full((TILES, 256), 0x7F, dtype=np.uint8)
    expected[:, :128] = np.repeat(exponents, 2, axis=1)
    return bits.view(np.float32), expected.ravel()


def _fake_vector(dtype, size):
    return make_fake_tensor(dtype, (size,), (1,), layout_tag=tla.arch.RowMajor)


def _run_e8m0(device, torch):
    def check(dtype, slot, partial, chunks):
        target = tla.BFloat16 if dtype is tla.Float8E8M0 else tla.Float8E8M0
        source_size = TILES * (2048 // dtype.width)
        target_size = TILES * (2048 // target.width)
        compiled = _make_e8m0_kernel(dtype, slot, partial).compile(
            type_args=(
                _fake_vector(dtype, source_size),
                _fake_vector(target, target_size),
            ),
            options="--npu-arch 3510",
        )
        if torch is not None:
            for values, initial, expected in chunks:
                x = torch.from_numpy(values).to(device=f"npu:{device}")
                if dtype is tla.BFloat16:
                    x = x.view(torch.bfloat16)
                y = torch.from_numpy(initial).to(device=f"npu:{device}")
                compiled(
                    from_dlpack(x, layout_tag=tla.arch.RowMajor, element_type=dtype),
                    from_dlpack(y, layout_tag=tla.arch.RowMajor, element_type=target),
                    block_num=1,
                )
                torch.npu.synchronize(device)
                np.testing.assert_array_equal(y.cpu().numpy(), expected)
        print(f"{dtype.__name__} slot={slot} partial={partial} passed")

    bits = np.arange(65536, dtype=np.uint16)
    initial = (np.arange(DEST_SIZE) * 37 + 0xA5).astype(np.uint8)
    for slot in (RegSlot.ZERO, RegSlot.ONE):
        for partial in (False, True):
            chunks = []
            for chunk in np.split(bits, 4):
                active = np.arange(chunk.size) % 128 < (3 if partial else 128)
                expected = np.zeros_like(initial)
                byte_indices = 2 * np.flatnonzero(active) + int(slot is RegSlot.ONE)
                selected = chunk[active]
                encoded = (selected >> 7) & 0xFF
                encoded[(selected & 0x7FFF) == 0x7F80] = 0xFE
                expected[byte_indices] = encoded
                chunks.append((chunk.view(np.int16), initial, expected))
            check(tla.BFloat16, slot, partial, chunks)

    # A different phase in each register exercises every encoding in both
    # source slots, including under the three-byte predicate.
    bf16_by_code = np.empty(256, dtype=np.uint16)
    finite_scales = np.ldexp(np.float32(1), np.arange(255) - 127, dtype=np.float32)
    bf16_by_code[:255] = finite_scales.view(np.uint32) >> 16
    bf16_by_code[255] = 0x7FC0
    byte_indices = np.arange(256)
    phases = np.arange(TILES)[:, None] * 2
    for slot in (RegSlot.ZERO, RegSlot.ONE):
        slot_offset = 1 if slot is RegSlot.ONE else 0
        addresses = byte_indices[slot_offset::2]
        for partial in (False, True):
            chunks = []
            for phase in (0, 1):
                encoded = (byte_indices + phases + phase).astype(np.uint8)
                expected = bf16_by_code[encoded[:, addresses]]
                if partial:
                    expected[:, addresses >= 3] = 0
                initial_bf16 = np.full(TILES * 128, 0x3F80, dtype=np.int16)
                chunks.append(
                    (encoded.ravel(), initial_bf16, expected.astype(np.int16).ravel())
                )
            check(tla.Float8E8M0, slot, partial, chunks)

    values, expected = _pscale_data()
    check(tla.Float32, RegSlot.ZERO, False, [(values, initial, expected)])


# Signed carriers are used for UInt16/UInt32 because device transfers need
# only the bits, while the TLA tensor descriptor supplies the real dtype.
_CARRIERS = {
    tla.Int16: np.int16,
    tla.Int32: np.int32,
    tla.UInt8: np.uint8,
    tla.UInt16: np.int16,
    tla.UInt32: np.int32,
    tla.Float16: np.float16,
    tla.Int4x2: np.uint8,
    tla.HiFloat8: np.uint8,
}


def _make_masked_cast_kernel(src_type, dst_type, traits, tiles):
    src_lanes = 2048 // src_type.width
    dst_lanes = 2048 // dst_type.width

    @tla.kernel
    def masked_cast(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_ptr = tla.allocate(tiles * src_lanes, src_type, tla.AddressSpace.ub, 256)
        dst_ptr = tla.allocate(
            len(traits) * 2 * tiles * dst_lanes, dst_type, tla.AddressSpace.ub, 256
        )
        src_ub = tla.make_tensor_like(src_ptr, src, tla.arch.RowMajor)
        dst_ub = tla.make_tensor_like(dst_ptr, dst, tla.arch.RowMajor)
        ready = tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR)
        done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
        with tla.vector():
            tla.arch.set_ctrl(
                tla.params.CTRLValue(
                    saturation_strategy=tla.params.OverrideStrategy.USE_API
                )
            )
            tla.copy(src_ub, src)
            # Seed the UB so an omitted store cannot pass as a masked zero.
            tla.copy(dst_ub, dst)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                store_mask = tla.create_mask(dtype=dst_type, pattern=tla.mask.ALL)
                for variant in tla.range_constexpr(len(traits)):
                    params, mask_type = traits[variant]
                    all_lanes = tla.create_mask(dtype=mask_type, pattern=tla.mask.ALL)
                    sparse = tla.create_mask(dtype=mask_type, pattern=tla.mask.M3)
                    for i in tla.range(tiles):
                        value = tla.tile_view(
                            src_ub, tla.make_shape(src_lanes), tla.make_coord(i)
                        ).load()
                        offset = variant * 2 * tiles + i
                        full_out = tla.tile_view(
                            dst_ub, tla.make_shape(dst_lanes), tla.make_coord(offset)
                        )
                        sparse_out = tla.tile_view(
                            dst_ub,
                            tla.make_shape(dst_lanes),
                            tla.make_coord(offset + tiles),
                        )
                        full_out.store(
                            value.to(dst_type, params, all_lanes), mask=store_mask
                        )
                        sparse_out.store(
                            value.to(dst_type, params, sparse), mask=store_mask
                        )
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(dst, dst_ub)

    return masked_cast


def check_masked_casts(device, compile_only, src_type, dst_type, source, cases):
    """Check all cases in one device launch and return the compiled kernel."""

    traits = tuple((params, mask_type) for params, mask_type, _ in cases)
    expected = np.stack([result for _, _, result in cases])
    compiled = _make_masked_cast_kernel(
        src_type, dst_type, traits, source.shape[0]
    ).compile(
        type_args=(
            _fake_vector(src_type, source.size),
            _fake_vector(dst_type, expected.size),
        ),
        options="--npu-arch 3510",
    )
    if compile_only:
        return compiled

    import torch

    source_device = torch.from_numpy(source.ravel().view(_CARRIERS[src_type])).to(
        f"npu:{device}"
    )
    initial = np.full(expected.size, 33, dtype=_CARRIERS[dst_type])
    destination_device = torch.from_numpy(initial).to(f"npu:{device}")
    compiled(
        from_dlpack(source_device, element_type=src_type, layout_tag=tla.arch.RowMajor),
        from_dlpack(
            destination_device, element_type=dst_type, layout_tag=tla.arch.RowMajor
        ),
        block_num=1,
    )
    torch.npu.synchronize(device)
    actual = (
        destination_device.cpu().numpy().view(expected.dtype).reshape(expected.shape)
    )
    for index, (params, _, _) in enumerate(cases):
        np.testing.assert_array_equal(
            actual[index],
            expected[index],
            err_msg=f"{src_type.__name__} -> {dst_type.__name__} {params}",
        )
    return compiled


def _run_mixed_int(device, compile_only):
    routes = (
        (tla.Int16, tla.UInt8, np.int16, np.uint8),
        (tla.Int16, tla.UInt32, np.int16, np.uint32),
        (tla.Int32, tla.UInt8, np.int32, np.uint8),
        (tla.Int32, tla.UInt16, np.int32, np.uint16),
        (tla.UInt32, tla.Int16, np.uint32, np.int16),
    )
    for src_type, dst_type, src_dtype, dst_dtype in routes:
        src_info, dst_info = np.iinfo(src_dtype), np.iinfo(dst_dtype)
        boundaries = [src_info.min, src_info.max, -257, -1, 0, 1, 127, 128, 256]
        for edge in (dst_info.min, dst_info.max):
            boundaries.extend((int(edge) - 1, int(edge), int(edge) + 1))
        values = [x for x in boundaries if src_info.min <= x <= src_info.max]
        src_lanes = 2048 // src_type.width
        dst_lanes = 2048 // dst_type.width
        source = np.resize(np.array(values, dtype=src_dtype), (3, src_lanes))
        widen = src_lanes > dst_lanes
        stride = max(src_lanes, dst_lanes) // min(src_lanes, dst_lanes)
        saturation_modes = (SatMode.UNKNOWN,) if widen else (SatMode.SAT, SatMode.NOSAT)
        cases = []
        for position, slot in enumerate(tuple(RegSlot)[:stride]):
            indices = position + stride * np.arange(min(src_lanes, dst_lanes))
            selected = source[:, indices] if widen else source
            for saturation in saturation_modes:
                converted = selected.astype(np.int64)
                if saturation is not SatMode.NOSAT:
                    converted = np.clip(converted, dst_info.min, dst_info.max)
                converted = converted.astype(dst_dtype)
                expected = np.zeros((2, source.shape[0], dst_lanes), dtype=dst_dtype)
                destination = slice(None) if widen else indices
                active = indices % 3 == 0 if widen else np.arange(src_lanes) % 3 == 0
                expected[0][:, destination] = converted
                expected[1][:, destination] = np.where(active, converted, 0)
                params = CastParams(reg_slot=slot, sat_mode=saturation)
                cases.append((params, src_type, expected))
        compiled = check_masked_casts(
            device, compile_only, src_type, dst_type, source, cases
        )
        if not compiled.kernel_binary_path.is_file():
            raise FileNotFoundError(
                f"Missing compiled kernel: {compiled.kernel_binary_path}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("round-modes", "e8m0", "mixed-int"))
    case, remaining = parser.parse_known_args()
    if case.case in (None, "round-modes"):
        return _run_round_modes(remaining)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(remaining)
    torch = None
    if not args.compile_only:
        import torch
        import torch_npu  # noqa: F401 -- register the device backend

        torch.npu.set_device(args.device)
    if case.case == "e8m0":
        _run_e8m0(args.device, torch)
    else:
        _run_mixed_int(args.device, args.compile_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
