"""Add a tile using an inline Ascend C register operation inside the DSL's SIMD VF."""

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


TILE = 256


@tla.extern(
    source=r"""
#include "c_api/asc_simd.h"

extern "C" __aicore__ __attribute__((always_inline))
void add_ub(uint64_t a, uint64_t b, uint64_t c, int32_t n) {
    auto x = reinterpret_cast<__ubuf__ float *>(a);
    auto y = reinterpret_cast<__ubuf__ float *>(b);
    auto z = reinterpret_cast<__ubuf__ float *>(c);
    constexpr int32_t lanes = 64;
    for (int32_t i = 0; i < n; i += lanes) {
        uint32_t remaining = n - i;
        vector_bool mask = asc_update_mask_b32(remaining);
        vector_float lhs, rhs, result;
        asc_loadalign(lhs, x + i);
        asc_loadalign(rhs, y + i);
        asc_add(result, lhs, rhs, mask);
        asc_storealign(z + i, result, mask);
    }
}
"""
)
def add_ub(
    a: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    b: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    c: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    n: tla.Int32,
) -> None: ...


@tla.kernel
def extern_simd_vf_add(a: tla.Tensor, b: tla.Tensor, c: tla.Tensor) -> None:
    x = tla.make_tensor_like(
        tla.allocate(TILE, tla.Float32, tla.AddressSpace.ub, 256), a, tla.arch.RowMajor
    )
    y = tla.make_tensor_like(
        tla.allocate(TILE, tla.Float32, tla.AddressSpace.ub, 256), b, tla.arch.RowMajor
    )
    z = tla.make_tensor_like(
        tla.allocate(TILE, tla.Float32, tla.AddressSpace.ub, 256), c, tla.arch.RowMajor
    )
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(x, a)
        tla.copy(y, b)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            add_ub(x.ptr, y.ptr, z.ptr, TILE)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(c, z)
        tla.pipe_barrier(tla.pipes.ALL)


def run(device: int = 0) -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    a, b = (torch.rand(TILE, device="npu", dtype=torch.float32) for _ in range(2))
    c = torch.empty_like(a)
    args = [from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in (a, b, c)]
    kernel = tla.compile(extern_simd_vf_add, *args, options="--npu-arch 3510")
    kernel(*args, block_num=1)
    torch.npu.synchronize()
    torch.testing.assert_close(c, a + b, rtol=0, atol=0)
    print(f"passed; kernel={kernel.kernel_binary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    run(parser.parse_args().device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
