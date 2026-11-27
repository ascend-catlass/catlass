"""Check selected-bit updates and explicit restoration on Ascend 950."""

import argparse

from catlass import tla


def _observations(out):
    out[8] = tla.arch.get_ctrl()
    original_mode = tla.arch.get_ctrl(start_bit=59, end_bit=59)
    original_saturation = tla.arch.get_ctrl(start_bit=53, end_bit=53)
    tla.arch.set_ctrl(tla.params.CTRLValue(integer_saturation=True))
    out[9] = tla.arch.get_ctrl()
    tla.arch.set_ctrl(0, start_bit=53, end_bit=53)
    out[10] = tla.arch.get_ctrl(start_bit=53, end_bit=53)
    tla.arch.set_ctrl(1, start_bit=59, end_bit=59)
    tla.arch.set_ctrl(1, start_bit=53, end_bit=53)
    out[11] = tla.arch.get_ctrl(start_bit=53, end_bit=53)
    tla.arch.set_ctrl(original_saturation, start_bit=53, end_bit=53)
    out[12] = tla.arch.get_ctrl()
    tla.arch.set_ctrl(original_mode, start_bit=59, end_bit=59)
    out[13] = tla.arch.get_ctrl()


@tla.kernel
def _query(out: tla.Tensor, cube: tla.Constexpr):
    if cube:
        with tla.cube():
            _observations(out)
    else:
        with tla.vector():
            _observations(out)


@tla.kernel
def _double(src: tla.Tensor, out: tla.Tensor, modes: tla.Tensor, patch: tla.Constexpr):
    a = tla.make_tensor_like(
        tla.allocate(128, src.element_type, tla.AddressSpace.ub, 256),
        src,
        tla.arch.RowMajor,
    )
    b = tla.make_tensor_like(
        tla.allocate(128, src.element_type, tla.AddressSpace.ub, 256),
        out,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(a, src)
        tla.pipe_barrier(tla.pipes.ALL)
        tla.arch.set_ctrl(patch)
        modes[0] = tla.arch.get_ctrl(start_bit=56, end_bit=56)
        with tla.vec.func(mode="simd"):
            value = a.load()
            b.store(value + value)
        modes[1] = tla.arch.get_ctrl(start_bit=56, end_bit=56)
        tla.pipe_barrier(tla.pipes.ALL)
        tla.copy(out, b)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def _atomic_updates(
    src: tla.Tensor, first: tla.Tensor, inner: tla.Tensor, last: tla.Tensor
):
    data = tla.make_tensor_like(
        tla.allocate(128, src.element_type, tla.AddressSpace.ub, 256),
        src,
        tla.arch.RowMajor,
    )
    add = tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.ADD)
    with tla.vector():
        tla.copy(data, src)
        tla.pipe_barrier(tla.pipes.ALL)
        tla.arch.set_ctrl(
            tla.params.CTRLValue(
                float_saturation=True,
                saturation_strategy=tla.params.OverrideStrategy.USE_GLOBAL,
            )
        )
        tla.copy(first, data, add)
        tla.pipe_barrier(tla.pipes.ALL)
        tla.arch.set_ctrl(1, start_bit=48, end_bit=48)
        tla.copy(inner, data, add)
        tla.pipe_barrier(tla.pipes.ALL)
        tla.arch.set_ctrl(0, start_bit=48, end_bit=48)
        tla.copy(last, data, add)
        tla.pipe_barrier(tla.pipes.ALL)


def main():
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    for cube in (False, True):
        output = torch.full((32,), 13, dtype=torch.uint64).npu()
        tensor = from_dlpack(output, layout_tag=tla.arch.RowMajor)
        artifact = tla.compile(_query, tensor, cube, options="--npu-arch 3510")
        artifact(tensor, block_num=1)
        actual = output.cpu().tolist()
        before = actual[8]
        expected = [13] * 32
        expected[8:14] = [before, before | (1 << 53), 0, 1, before | (1 << 59), before]
        if actual != expected:
            raise AssertionError(f"CTRL snapshot mismatch on cube={cube}: {actual}")
        print(artifact.kernel_binary_path, flush=True)

    src = torch.full((128,), 60000.0, dtype=torch.float16, device="npu")
    outputs = [src.clone() for _ in range(3)]
    buffers = (src, *outputs)
    tensors = [from_dlpack(b, layout_tag=tla.arch.RowMajor) for b in buffers]
    artifact = tla.compile(_atomic_updates, *tensors, options="--npu-arch 3510")
    artifact(*tensors, block_num=1)
    for output, expected in zip(outputs, (65504.0, float("inf"), 65504.0)):
        torch.testing.assert_close(
            output.cpu(), torch.full_like(src.cpu(), expected), atol=0, rtol=0
        )

    for fp, dtype, value, saturated, normal in (
        (False, torch.int16, 30000, 32767, -5536),
        (True, torch.float16, 60000.0, 65504.0, float("inf")),
    ):
        src = torch.full((128,), value, dtype=dtype, device="npu")
        for saturate in (False, True):
            out = torch.full_like(src, 17)
            modes = torch.full((2,), 2, dtype=torch.uint64).npu()
            inputs = [
                from_dlpack(t, layout_tag=tla.arch.RowMajor)
                for t in (src, out, modes)
            ]
            patch = tla.params.CTRLValue(
                float_saturation=saturate if fp else None,
                integer_saturation=None if fp else saturate,
                atomic_format=None if fp else tla.params.AtomicFormat.S16,
                saturation_strategy=tla.params.OverrideStrategy.USE_GLOBAL,
            )
            artifact = tla.compile(_double, *inputs, patch, options="--npu-arch 3510")
            artifact(*inputs, block_num=1)
            if modes.cpu().tolist() != [0, 0]:
                raise AssertionError(f"Unexpected mask mode across vector entry: {modes.cpu().tolist()}")
            expected = torch.full(
                (128,), saturated if saturate else normal, dtype=dtype
            )
            torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


if __name__ == "__main__":
    main()
