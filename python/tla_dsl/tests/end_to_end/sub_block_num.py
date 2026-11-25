"""Read the sub-block count from every participating core on Ascend 950."""

import argparse

from catlass import tla


def cube(out: tla.Tensor) -> None:
    count = tla.arch.sub_block_num()
    with tla.cube():
        slot = tla.arch.block_idx() * 48
        out[slot] = count
        out[slot + 1] = count + tla.Int64(4294967296)
        out[slot + 2] = count.to(tla.Int32).to(tla.Int64)


def vector(out: tla.Tensor) -> None:
    count = tla.arch.sub_block_num()
    with tla.vector():
        slot = tla.arch.block_idx() * 48 + (1 + tla.arch.sub_block_idx()) * 16
        out[slot] = count
        out[slot + 1] = count + tla.Int64(4294967296)
        out[slot + 2] = count.to(tla.Int32).to(tla.Int64)


def mixed(out: tla.Tensor) -> None:
    count = tla.arch.sub_block_num()
    with tla.cube():
        slot = tla.arch.block_idx() * 48
        out[slot] = count
        out[slot + 1] = count + tla.Int64(4294967296)
        out[slot + 2] = count.to(tla.Int32).to(tla.Int64)
    with tla.vector():
        slot = tla.arch.block_idx() * 48 + (1 + tla.arch.sub_block_idx()) * 16
        out[slot] = count
        out[slot + 1] = count + tla.Int64(4294967296)
        out[slot + 2] = count.to(tla.Int32).to(tla.Int64)


def capture_simd(out: tla.Tensor) -> None:
    with tla.vector():
        n = tla.arch.sub_block_num()
        with tla.vec.func(mode="simd"):
            out[0] = n + n


def capture_simt(out: tla.Tensor) -> None:
    with tla.vector():
        n = tla.arch.sub_block_num()
        with tla.vec.func(mode="simt", thread_block_dim=(32, 1, 1)):
            tid, _, _ = tla.arch.thread_idx()
            out[tid] = n + n


def main():
    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    for fn, size in ((capture_simd, 1), (capture_simt, 32)):
        out = torch.full((size,), -1, dtype=torch.int64, device=f"npu:{args.device}")
        tensor = tla.from_dlpack(out, layout_tag=tla.arch.RowMajor)
        artifact = tla.compile(tla.kernel(fn), tensor, options="--npu-arch 3510")
        artifact(tensor, block_num=1)
        torch.npu.synchronize()
        actual = out.cpu().tolist()
        if actual != [2] * size:
            raise AssertionError(f"{fn.__name__}: expected 2 at every output, got {actual}")

    for fn, options, values in (
        (cube, {}, (1, -1, -1)),
        (vector, {}, (-1, 1, -1)),
        (mixed, {"core_ratio": tla.arch.C1V1}, (1, 1, -1)),
        (mixed, {"core_ratio": tla.arch.C1V2}, (1, 2, 2)),
        (mixed, {}, (1, 2, 2)),
    ):
        kernel = tla.kernel(fn, **options)
        for blocks in (1, 2):
            out = torch.full((blocks * 48,), -1, dtype=torch.int64, device=f"npu:{args.device}")
            tensor = tla.from_dlpack(out, layout_tag=tla.arch.RowMajor)
            artifact = tla.compile(kernel, tensor, options="--npu-arch 3510")
            if fn is mixed and not options:
                lowered = artifact.kernel_binary_path.with_name("lowered.mlir").read_text()
                if "#hivm.core_ratio<1, 2>" not in lowered:
                    raise AssertionError(
                        f"mixed kernel with {blocks} blocks: lowered.mlir is missing #hivm.core_ratio<1, 2>"
                    )
            artifact(tensor, block_num=blocks)
            torch.npu.synchronize()
            actual = out.cpu().tolist()
            expected = [-1] * len(actual)
            for block in range(blocks):
                for core, value in enumerate(values):
                    if value >= 0:
                        slot = block * 48 + core * 16
                        expected[slot:slot + 3] = [value, 2**32 + value, value]
            if actual != expected:
                raise AssertionError(
                    f"{fn.__name__} with {options} and {blocks} blocks: expected {expected}, got {actual}"
                )


if __name__ == "__main__":
    main()
