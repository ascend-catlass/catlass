"""Pass SIMD registers to an inline Ascend C function and store its result."""

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


TILE = 256
LANES = 64
SOURCE = r"""
#include "c_api/asc_simd.h"

extern "C" __aicore__ __attribute__((always_inline))
vector_float add_reg(vector_float a, vector_float b) {
    vector_float dst;
    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    asc_add(dst, a, b, mask);
    return dst;
}
"""


@tla.extern(source=SOURCE)
def add_reg(
    a: tla.VectorSSA[tla.Float32],
    b: tla.VectorSSA[tla.Float32],
) -> tla.VectorSSA[tla.Float32]: ...


@tla.kernel
def extern_simd_reg_add(a: tla.Tensor, b: tla.Tensor, c: tla.Tensor) -> None:
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
            for i in tla.range(TILE // LANES):
                a_tile = tla.tile_view(x, tla.make_shape(LANES), tla.make_coord(i))
                b_tile = tla.tile_view(y, tla.make_shape(LANES), tla.make_coord(i))
                c_tile = tla.tile_view(z, tla.make_shape(LANES), tla.make_coord(i))
                c_tile.store(add_reg(a_tile.load(), b_tile.load()))
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(c, z)
        tla.pipe_barrier(tla.pipes.ALL)


def run(device: int = 0) -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    a, b = (torch.randn(TILE, device="npu", dtype=torch.float32) for _ in range(2))
    c = torch.empty_like(a)
    args = [from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in (a, b, c)]
    kernel = tla.compile(extern_simd_reg_add, *args, options="--npu-arch 3510")
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
