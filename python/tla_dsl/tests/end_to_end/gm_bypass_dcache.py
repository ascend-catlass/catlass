"""Check that a GM bypass read can feed a scalar tensor assignment."""

import argparse

from catlass import tla


@tla.kernel
def read_first(src: tla.Tensor, dst: tla.Tensor):
    with tla.cube():
        dst[0] = tla.arch.load_bypass_dcache(src.ptr)


@tla.kernel
def read_after_dma_write(
    target: tla.Tensor, replacement: tla.Tensor, observed: tla.Tensor
):
    scratch_ptr = tla.allocate(32, tla.Int32, tla.AddressSpace.ub, 32)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.MTE3)
    cached = tla.flag("cached", tla.arch.SCALAR, tla.arch.MTE3)
    written = tla.flag("written", tla.arch.MTE3, tla.arch.SCALAR)
    with tla.vector():
        scratch = tla.make_tensor(
            scratch_ptr,
            tla.make_layout(tla.make_shape(1, 32), tla.make_stride(32, 1)),
        )
        tla.copy(scratch, replacement)
        tla.set_flag(loaded)
        # Fill the scalar cache before MTE3 overwrites the GM line.
        observed[0] = target[0, 0]
        tla.set_flag(cached)
        tla.wait_flag(loaded)
        tla.wait_flag(cached)
        tla.copy(target, scratch)
        tla.set_flag(written)
        tla.wait_flag(written)
        observed[1] = target[0, 0]
        observed[2] = tla.arch.load_bypass_dcache(target.ptr)


def main():
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    source = torch.tensor([-123, 45, 67, 89], dtype=torch.int32)
    expected = torch.full_like(source, -1)
    expected[0] = source[0]
    src = source.npu()
    dst = torch.full_like(src, -1)
    inputs = [from_dlpack(buf, layout_tag=tla.arch.RowMajor) for buf in (src, dst)]
    artifact = tla.compile(read_first, *inputs, options="--npu-arch 3510")
    artifact(*inputs, block_num=1)
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(src.cpu(), source, rtol=0, atol=0)
    print(artifact.kernel_binary_path, flush=True)

    target = torch.full((1, 32), 11, dtype=torch.int32).npu()
    replacement = torch.full_like(target, 29)
    observed = torch.full((3,), -1, dtype=torch.int32).npu()
    inputs = [
        from_dlpack(buf, layout_tag=tla.arch.RowMajor)
        for buf in (target, replacement, observed)
    ]
    artifact = tla.compile(read_after_dma_write, *inputs, options="--npu-arch 3510")
    artifact(*inputs, block_num=1)
    print(
        "ordinary before, ordinary after, bypass after:",
        observed.cpu().tolist(),
        flush=True,
    )
    torch.testing.assert_close(
        observed.cpu(), torch.tensor([11, 11, 29], dtype=torch.int32)
    )


if __name__ == "__main__":
    main()
