# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce the official `DataSyncBarrier` CANN usage example in TLA Python.

AIV-only kernel over two blocks (`--block-num 2`), split by `tla.arch.block_idx()`:

- Block 0 (producer): write the data with `tla.arch.store_bypass_dcache`, then use
  `tla.data_sync_barrier(DsbTarget.GM)` -- the ISA `dsb` that blocks subsequent GM
  writes until the earlier one is visible to other cores -- then bypass-write
  the sync flag.
- Block 1 (consumer): poll the flag with bypass reads, then read the data
  and write `2 * data` through a bypass write.

The consumer polls through an independent AscendC extern. It leaves the
output at its initial value if the producer fails to publish the flag.

Scope note: the case verifies the officially prescribed pattern end to end, not
that `dsb` is load-bearing on this silicon -- measured on Ascend 950PR,
back-to-back bypass GM writes from one core reached the other block in order in
15/15 runs even with the barrier removed, so the ISA-level window the example
guards against did not manifest here.
"""

import argparse

import catlass.tla as tla

DATA_VALUE = 1234
SYNC_FLAG = 777

OP_SOURCE_CODES = r"""
#include <cstdint>
#include "kernel_operator.h"

constexpr int32_t SYNC_FLAG = 777;

extern "C" __aicore__ __attribute__((noinline)) void tla_user_wait_flag_read_copy(
    uint64_t src_addr, uint64_t dst_addr) {
  __gm__ int32_t *src = reinterpret_cast<__gm__ int32_t *>(src_addr);
  for (int i = 0; i < 1000000; ++i) {
    if (AscendC::ReadGmBypassDCache<int32_t>(src) == SYNC_FLAG) {
      int32_t v = AscendC::ReadGmBypassDCache<int32_t>(src + 1);
      AscendC::WriteGmBypassDCache<int32_t>(
          reinterpret_cast<__gm__ int32_t *>(dst_addr), 2 * v);
      return;
    }
  }
}
"""


@tla.extern(name="tla_user_wait_flag_read_copy", source=OP_SOURCE_CODES)
def wait_flag_read_copy(
    src: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
    dst: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
) -> None: ...


@tla.kernel
def data_sync_barrier_demo(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        bid = tla.arch.block_idx()
        if bid == 0:
            # Producer: bypass-write the data, wait the GM domain out so the
            # write is visible to other cores, then bypass-write the flag.
            tla.arch.store_bypass_dcache(src.ptr + 1, DATA_VALUE)
            tla.data_sync_barrier(tla.params.DsbTarget.GM)
            tla.arch.store_bypass_dcache(src.ptr, SYNC_FLAG)
        if bid == 1:
            # Consumer: spin on the bypass-read flag, then read the data.
            wait_flag_read_copy(src.ptr, dst.ptr)
        tla.pipe_barrier(tla.pipes.ALL)


def main():
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=2)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    src = torch.full((64,), -1, dtype=torch.int32).npu()
    dst = torch.full((1,), -1, dtype=torch.int32).npu()
    inputs = [
        from_dlpack(tensor, layout_tag=tla.arch.RowMajor)
        for tensor in (src, dst)
    ]
    artifact = tla.compile(
        data_sync_barrier_demo, *inputs, options="--npu-arch 3510"
    )
    artifact(*inputs, block_num=args.block_num)
    torch.testing.assert_close(
        dst.cpu(), torch.full((1,), 2 * DATA_VALUE, dtype=torch.int32).cpu(), rtol=0, atol=0
    )
    flag_value = src[0].item()
    data_value = src[1].item()
    if flag_value != SYNC_FLAG or data_value != DATA_VALUE:
        raise AssertionError(
            f"unexpected src contents: flag={flag_value}, data={data_value}; "
            f"expected flag={SYNC_FLAG}, data={DATA_VALUE}"
        )
    print(artifact.kernel_binary_path, flush=True)


if __name__ == "__main__":
    main()
