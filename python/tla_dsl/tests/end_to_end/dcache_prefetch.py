"""Exercise GM prefetch on each scalar issuer using separate guarded lines."""

import argparse

from catlass import tla


@tla.kernel
def prefetch(
    src: tla.Tensor, out: tla.Tensor, cube: tla.Constexpr, vector: tla.Constexpr
):
    if cube:
        with tla.cube():
            tla.arch.dcache_prefetch(src.ptr, byte_offset=64)
            out[16] = src[16]
    if vector:
        with tla.vector():
            subblock = tla.arch.sub_block_idx() if cube else 0
            view = tla.tile_view(src, tla.make_shape(16), tla.make_coord(2 + subblock))
            tla.arch.dcache_prefetch(
                view.ptr, byte_offset=tla.Int64((2 + subblock) * 64 + 4)
            )
            out[48 + 16 * subblock] = view[1]


def main():
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    reference = torch.arange(128, dtype=torch.int32)
    src = reference.npu()
    for cube, vector in ((True, False), (False, True), (True, True)):
        output = torch.full_like(src, -1)
        inputs = [
            from_dlpack(tensor, layout_tag=tla.arch.RowMajor)
            for tensor in (src, output)
        ]
        artifact = tla.compile(
            prefetch, *inputs, cube, vector, options="--npu-arch 3510"
        )
        artifact(*inputs, block_num=1)
        expected = torch.full_like(reference, -1)
        if cube:
            expected[16] = reference[16]
        if vector:
            expected[48] = reference[33]
            if cube:
                expected[64] = reference[49]
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        print(artifact.kernel_binary_path, flush=True)


if __name__ == "__main__":
    main()
