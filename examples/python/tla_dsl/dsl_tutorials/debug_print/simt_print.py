# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def simt_print(source: tla.Tensor, output: tla.Tensor, count: tla.Int32):
    local_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    local = tla.make_tensor_like(local_ptr, source, tla.arch.RowMajor)
    with tla.vector():
        tla.print("outer={}", count)
        with tla.vec.func(mode="simd"):
            tla.print("simd={}", count)
        with tla.vec.func(mode="simt", thread_block_dim=64):
            thread, _, _ = tla.arch.thread_idx()
            local[thread] = source[thread] * 2.0
            tla.arch.sync_threads()
            if thread == 0:
                tla.print(local, count)
                tla.print(source, count)
            if thread < 4:
                tla.print("tid={} x={}", thread, local[thread])
            # Keep the shared prefix stable until every print has read it.
            tla.arch.sync_threads()
            local[thread] = local[thread] + 1.0
            output[thread] = local[thread]


def _verify_simt_print_output(output: str, *, input_offset: int) -> None:
    if output.count("outer=8") != 1 or output.count("simd=8") != 1:
        raise RuntimeError(f"Unexpected outer or SIMD output: {output!r}")
    if output.count("tid=") != 4 or output.count("DumpTensor:") != 2:
        raise RuntimeError(f"Unexpected SIMT record count: {output!r}")
    for thread in range(4):
        if f"tid={thread} x={float((thread + input_offset) * 2):f}" not in output:
            raise RuntimeError(f"Unexpected SIMT thread value: {output!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print scalar and GM/UB prefixes from SIMT threads."
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    if args.run and args.device is None:
        parser.error("--run requires --device")
    tensor = make_fake_tensor(
        tla.Float32, (64,), (1,), origin_shape=(64,), layout_tag=tla.arch.RowMajor
    )
    compiled = simt_print.compile(
        type_args=(tensor, tensor, tla.Int32(8)), options="--npu-arch 3510"
    )
    print(f"kernel.o path={compiled.kernel_binary_path}")
    if not args.run:
        return
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    try:
        from .debug_print import _capture_c_stdout
    except ImportError:
        from debug_print import _capture_c_stdout
    torch.npu.set_device(args.device)
    for update in (0, 1):
        source = (
            torch.arange(64, dtype=torch.float32, device=f"npu:{args.device}") + update
        )
        output = torch.empty_like(source)
        captured = _capture_c_stdout(
            lambda: compiled(
                from_dlpack(source, layout_tag=tla.arch.RowMajor),
                from_dlpack(output, layout_tag=tla.arch.RowMajor),
                tla.Int32(8),
                block_num=1,
            )
        )
        torch.testing.assert_close(output.cpu(), source.cpu() * 2 + 1, rtol=0, atol=0)
        _verify_simt_print_output(captured, input_offset=update)
        print(captured, end="")
    print("output_ok=True")


if __name__ == "__main__":
    main()
