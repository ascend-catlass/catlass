"""Show full and partial SIMT warp active masks on Ascend 950."""

import argparse

from catlass import tla


@tla.kernel
def active_masks(out: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=(48, 1, 1)):
            tid, _, _ = tla.arch.thread_idx()
            out[tid] = tla.arch.activemask().to(tla.UInt64).to(tla.Int64)
        tla.pipe_barrier(tla.pipes.ALL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    out = torch.full((48,), -1, dtype=torch.int64, device="npu")
    tensor = from_dlpack(out, layout_tag=tla.arch.RowMajor)
    artifact = tla.compile(active_masks, tensor, options="--npu-arch 3510")
    artifact(tensor, block_num=1)
    torch.npu.synchronize()
    expected = torch.tensor([0xFFFFFFFF] * 32 + [0xFFFF] * 16, dtype=torch.int64)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    print("Full warp: 0xffffffff; partial warp: 0x0000ffff")


if __name__ == "__main__":
    main()
