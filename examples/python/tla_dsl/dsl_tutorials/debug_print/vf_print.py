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
def vf_print(source: tla.Tensor, destination: tla.Tensor, case_id: tla.Int32):
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    source_ub_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    destination_ub_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    source_ub = tla.make_tensor_like(source_ub_ptr, source, tla.arch.RowMajor)
    destination_ub = tla.make_tensor_like(
        destination_ub_ptr, destination, tla.arch.RowMajor
    )
    with tla.vector():
        tla.copy(source_ub, source)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            source_register = source_ub.load()
            result_register = source_register * 2.0 + 1.0
            tla.print("vf_case={} scale={}", case_id, 2.0)
            tla.print(result_register, 4)
            tla.print(source_ub, 4)
            destination_ub.store(result_register)
            tla.print(destination_ub, 4)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(destination, destination_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile one scalar/register/UB SIMD print example."
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--calls", type=int, default=2)
    args = parser.parse_args()
    if args.run and args.device is None:
        parser.error("--run requires an explicit --device")
    compile_tensor = make_fake_tensor(
        tla.Float32,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )
    compiled = vf_print.compile(
        type_args=(compile_tensor, compile_tensor, tla.Int32(7)),
        options="--npu-arch 3510",
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
    source = torch.arange(64, dtype=torch.float32) * 0.25 - 4.0
    source_device = source.to(f"npu:{args.device}")
    destination_device = torch.empty_like(source_device)
    if args.calls < 1:
        parser.error("--calls must be positive")
    for call_index in range(args.calls):
        current_source = source + call_index
        source_device.copy_(current_source.to(f"npu:{args.device}"))
        case_value = 7 + call_index
        output = _capture_c_stdout(
            lambda: compiled(
                from_dlpack(source_device, layout_tag=tla.arch.RowMajor),
                from_dlpack(destination_device, layout_tag=tla.arch.RowMajor),
                tla.Int32(case_value),
                block_num=1,
            )
        )
        torch.npu.synchronize(args.device)
        reference = current_source * 2.0 + 1.0
        torch.testing.assert_close(destination_device.cpu(), reference, rtol=0, atol=0)
        input_prefix = ", ".join(f"{value:g}" for value in current_source[:4].tolist())
        result_prefix = ", ".join(f"{value:g}" for value in reference[:4].tolist())
        expected = (
            f"vf_case={case_value} scale=2.000000",
            f"position=REG, shape=[4] dump_size=4 [{result_prefix}]",
            f"position=UB, shape=[64] dump_size=4 [{input_prefix}]",
            f"position=UB, shape=[64] dump_size=4 [{result_prefix}]",
        )
        if any(output.count(text) != 1 for text in expected):
            raise RuntimeError(f"Unexpected or duplicated VF output: {output!r}")
        print(output, end="")
    print("output_ok=True")


if __name__ == "__main__":
    main()
