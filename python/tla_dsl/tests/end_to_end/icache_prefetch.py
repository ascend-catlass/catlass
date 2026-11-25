"""Check instruction-cache status on the issuing scalar cores."""

import argparse

from catlass import tla
from catlass.tla.params import ICachePrefetchStatus


@tla.jit
def poll_status(out: tla.Tensor, index: tla.Int32):
    for i in tla.range(8):
        status = tla.arch.get_icache_prefetch_status()
        if status == ICachePrefetchStatus.COMPLETE:
            out[index + i] = 0
        elif status == ICachePrefetchStatus.INCOMPLETE:
            out[index + i] = 1
        else:
            out[index + i] = -2


@tla.kernel
def prefetch(
    out: tla.Tensor,
    cube: tla.Constexpr,
    vector: tla.Constexpr,
    cube_units: tla.Constexpr,
    vector_units: tla.Constexpr,
):
    if cube:
        with tla.cube():
            if cube_units:
                tla.arch.icache_prefetch(cube_units)
                poll_status(out, 0)
            else:
                out[0] = 2
    if vector:
        with tla.vector():
            subblock = tla.arch.sub_block_idx() if cube else 0
            if vector_units:
                tla.arch.icache_prefetch(vector_units)
                poll_status(out, 16 + 16 * subblock)
            else:
                out[16 + 16 * subblock] = 2


def main():
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    cases = (
        (True, False, 1, 0),
        (True, False, 15, 0),
        (False, True, 0, 1),
        (False, True, 0, 7),
        (True, True, 15, 0),
        (True, True, 0, 7),
        (True, True, 1, 1),
    )
    for cube, vector, cube_units, vector_units in cases:
        output = torch.full((48,), -1, dtype=torch.int32, device="npu")
        arg = from_dlpack(output, layout_tag=tla.arch.RowMajor)
        inputs = (arg, cube, vector, cube_units, vector_units)
        for _ in range(2):
            artifact = tla.compile(prefetch, *inputs, options="--npu-arch 3510")
            output.fill_(-1)
            artifact(arg, block_num=1)
            values = output.cpu().tolist()
            for offset in (0, 16, 32):
                active = cube if offset == 0 else vector and (offset == 16 or cube)
                units = cube_units if offset == 0 else vector_units
                observed = values[offset : offset + 16]
                if active and units:
                    statuses = observed[:8]
                    if any(code not in (0, 1) for code in statuses):
                        raise RuntimeError(
                            f"Invalid icache status at offset {offset}: {statuses}"
                        )
                    expected = statuses + [-1] * 8
                else:
                    expected = [2] + [-1] * 15 if active else [-1] * 16
                if observed != expected:
                    raise RuntimeError(
                        f"Icache output at offset {offset}: expected {expected}, got {observed}"
                    )
            print(artifact.kernel_binary_path, flush=True)


if __name__ == "__main__":
    main()
