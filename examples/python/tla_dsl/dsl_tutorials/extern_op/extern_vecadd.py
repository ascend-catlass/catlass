"""Vecadd whose GM-to-UB copies are supplied by a user Ascend C function."""

from __future__ import annotations

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


TILE_ELE = 256
VL_ELE = 64

OP_SOURCE_CODES = r"""
#include <cstdint>
#include "kernel_operator.h"

extern "C" {

__aicore__ __attribute__((always_inline)) void tla_user_gm_to_ub_f32(
    uint64_t src_gm_addr, uint64_t dst_ub_addr, int32_t count) {
  AscendC::GlobalTensor<float> src;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src_gm_addr),
                      static_cast<uint32_t>(count));
  AscendC::LocalTensor<float> dst(AscendC::TPosition::VECCALC,
                                  static_cast<uint32_t>(dst_ub_addr),
                                  static_cast<uint32_t>(count));
  AscendC::DataCopy(dst, src, static_cast<uint32_t>(count));
}

} // extern "C"
"""


@tla.extern(
    name="tla_user_gm_to_ub_f32",
    source=OP_SOURCE_CODES,
)
def tla_user_gm_to_ub_f32(
    gm_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.gm],
    ub_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    ele_num: tla.Int32,
) -> None: ...


@tla.kernel
def extern_vecadd(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    ub_ptr_a = tla.allocate(TILE_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ptr_b = tla.allocate(TILE_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ptr_c = tla.allocate(TILE_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_a = tla.make_tensor_like(ub_ptr_a, gm_a, tla.arch.RowMajor)
    ub_b = tla.make_tensor_like(ub_ptr_b, gm_b, tla.arch.RowMajor)
    ub_c = tla.make_tensor_like(ub_ptr_c, gm_c, tla.arch.RowMajor)

    with tla.vector():
        tla_user_gm_to_ub_f32(gm_a.ptr, ub_ptr_a, TILE_ELE)
        tla_user_gm_to_ub_f32(gm_b.ptr, ub_ptr_b, TILE_ELE)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            for i in tla.range(TILE_ELE // VL_ELE):
                a = tla.tile_view(ub_a, tla.make_shape(VL_ELE), tla.make_coord(i))
                b = tla.tile_view(ub_b, tla.make_shape(VL_ELE), tla.make_coord(i))
                c = tla.tile_view(ub_c, tla.make_shape(VL_ELE), tla.make_coord(i))
                c.store(tla.add(a.load(), b.load()))

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_c, ub_c)
        tla.pipe_barrier(tla.pipes.ALL)


def run(device: int = 0) -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    a = torch.rand(TILE_ELE, dtype=torch.float32, device="npu")
    b = torch.rand(TILE_ELE, dtype=torch.float32, device="npu")
    c = torch.empty_like(a)
    tla_a = from_dlpack(a, layout_tag=tla.arch.RowMajor)
    tla_b = from_dlpack(b, layout_tag=tla.arch.RowMajor)
    tla_c = from_dlpack(c, layout_tag=tla.arch.RowMajor)

    executor = tla.compile(
        extern_vecadd,
        tla_a,
        tla_b,
        tla_c,
        options="--npu-arch 3510",
    )
    executor(tla_a, tla_b, tla_c, block_num=1)
    torch.npu.synchronize()
    torch.testing.assert_close(c, a + b, rtol=0.0, atol=1e-4)
    print(f"passed; kernel={executor.kernel_binary_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    run(parser.parse_args().device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
