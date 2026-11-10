"""Verify extern register ABIs by inverting all 256 bytes of each register."""

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


REGISTER_TYPES = {
    "f16": (tla.Float16, "vector_half"),
    "bf16": (tla.BFloat16, "vector_bfloat16_t"),
    "f32": (tla.Float32, "vector_float"),
    "i8": (tla.Int8, "vector_int8_t"),
    "u8": (tla.UInt8, "vector_uint8_t"),
    "i16": (tla.Int16, "vector_int16_t"),
    "u16": (tla.UInt16, "vector_uint16_t"),
    "i32": (tla.Int32, "vector_int32_t"),
    "u32": (tla.UInt32, "vector_uint32_t"),
    "i64": (tla.Int64, "vector_int64_t"),
    "u64": (tla.UInt64, "vector_uint64_t"),
}

SOURCE = '#include "c_api/asc_simd.h"\n' + "\n".join(
    f"""
extern "C" __aicore__ __attribute__((always_inline))
{native} pack_{name}(vector_uint8_t x) {{ return ({native})x; }}
extern "C" __aicore__ __attribute__((always_inline))
{native} invert_{name}({native} x) {{
    vector_uint32_t dst;
    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    asc_not(dst, (vector_uint32_t)x, mask);
    return ({native})dst;
}}
extern "C" __aicore__ __attribute__((always_inline))
vector_uint8_t unpack_{name}({native} x) {{ return (vector_uint8_t)x; }}
"""
    for name, (_, native) in REGISTER_TYPES.items()
)


def _declare(symbol, input_dtype, output_dtype):
    def entry(x: tla.VectorSSA[input_dtype]) -> tla.VectorSSA[output_dtype]: ...

    return tla.extern(source=SOURCE, name=symbol)(entry)


@tla.kernel
def invert_register(
    src: tla.Tensor,
    dst: tla.Tensor,
    pack: tla.Constexpr,
    invert: tla.Constexpr,
    unpack: tla.Constexpr,
) -> None:
    x = tla.make_tensor_like(
        tla.allocate(256, tla.UInt8, tla.AddressSpace.ub, 256), src, tla.arch.RowMajor
    )
    y = tla.make_tensor_like(
        tla.allocate(256, tla.UInt8, tla.AddressSpace.ub, 256), dst, tla.arch.RowMajor
    )
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(x, src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            y.store(unpack(invert(pack(x.load()))))
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(dst, y)
        tla.pipe_barrier(tla.pipes.ALL)


def run(device: int = 0, dtype: str = "all") -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    # Use every possible byte value; compare raw bits, including floating NaN payloads.
    src = torch.arange(256, dtype=torch.int32).to(dtype=torch.uint8, device="npu")
    dst = torch.empty_like(src)
    args = [from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in (src, dst)]
    names = REGISTER_TYPES if dtype == "all" else [dtype]
    for name in names:
        element, _ = REGISTER_TYPES[name]
        ops = (
            _declare(f"pack_{name}", tla.UInt8, element),
            _declare(f"invert_{name}", element, element),
            _declare(f"unpack_{name}", element, tla.UInt8),
        )
        kernel = tla.compile(invert_register, *args, *ops, options="--npu-arch 3510")
        kernel(*args, block_num=1)
        torch.npu.synchronize()
        torch.testing.assert_close(
            dst.cpu(), torch.bitwise_not(src.cpu()), rtol=0, atol=0
        )
        print(f"passed: {name}; kernel={kernel.kernel_binary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=["all", *REGISTER_TYPES], default="all")
    args = parser.parse_args()
    run(args.device, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
