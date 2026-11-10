"""One Ascend C function called from both AIC and AIV regions."""

from __future__ import annotations

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


AIC_VALUE = 101
AIV_VALUE = 202
CACHE_LINE_BYTES = 64
INT32_BYTES = 4
ELEMENTS_PER_CACHE_LINE = CACHE_LINE_BYTES // INT32_BYTES
RESULT_SIZE = 3 * ELEMENTS_PER_CACHE_LINE

OP_SOURCE_CODES = r"""
#include <cstdint>
#include "kernel_operator.h"

extern "C" {

__aicore__ __attribute__((always_inline)) void tla_user_store_i32(
    uint64_t dst_gm_addr, int32_t index, int32_t value) {
  auto dst = reinterpret_cast<__gm__ int32_t *>(dst_gm_addr);
  dst[index] = value;
}

} // extern "C"
"""


@tla.extern(
    name="tla_user_store_i32",
    source=OP_SOURCE_CODES,
)
def tla_user_store_i32(
    dst: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
    index: tla.Int32,
    value: tla.Int32,
) -> None: ...


@tla.kernel
def extern_dual_core(result: tla.Tensor) -> None:
    with tla.cube():
        tla_user_store_i32(result.ptr, 0, AIC_VALUE)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        # Ensure each sub-block writes to a different cache line in the result
        # tensor. This avoids write conflicts.
        index = (1 + tla.arch.sub_block_idx()) * ELEMENTS_PER_CACHE_LINE
        tla_user_store_i32(result.ptr, index, AIV_VALUE)
        tla.pipe_barrier(tla.pipes.ALL)


def run(device: int = 0) -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    result = torch.zeros(RESULT_SIZE, dtype=torch.int32, device="npu")
    tla_result = from_dlpack(result, layout_tag=tla.arch.RowMajor)

    executor = tla.compile(
        extern_dual_core,
        tla_result,
        options="--npu-arch 3510",
    )
    executor(tla_result, block_num=1)
    torch.npu.synchronize()

    expected = torch.zeros(RESULT_SIZE, dtype=torch.int32, device="npu")
    expected[0] = AIC_VALUE
    expected[ELEMENTS_PER_CACHE_LINE] = AIV_VALUE
    expected[2 * ELEMENTS_PER_CACHE_LINE] = AIV_VALUE
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)
    print(f"passed; kernel={executor.kernel_binary_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    run(parser.parse_args().device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
