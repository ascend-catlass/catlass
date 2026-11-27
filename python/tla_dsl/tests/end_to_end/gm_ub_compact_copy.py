"""Check compact GM↔UB rows against ordinary single-row transfers on device."""

import argparse

from catlass import tla


@tla.jit
def _packed_ub():
    # Reserve space for four aligned rows as well, so a stride regression stays in bounds.
    ptr = tla.allocate(4 * 32, tla.UInt8, tla.AddressSpace.ub, 32)
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(4, 8), tla.make_stride(8, 1))
    )


@tla.kernel
def load_compact(rows: tla.Tensor, output: tla.Tensor) -> None:
    packed = _packed_ub()
    flat = tla.make_tensor_like(packed.ptr, output, tla.arch.RowMajor)
    tile = tla.tile_view(rows, tla.make_shape(4, 8), tla.make_coord(0, 0))
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.MTE3)
    with tla.vector():
        tla.copy(packed, tile, tla.params.CopyGmToUbParams(compact=True))
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.copy(output, flat)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def store_compact(source: tla.Tensor, rows: tla.Tensor) -> None:
    packed = _packed_ub()
    flat = tla.make_tensor_like(packed.ptr, source, tla.arch.RowMajor)
    tile = tla.tile_view(rows, tla.make_shape(4, 8), tla.make_coord(0, 0))
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.MTE3)
    with tla.vector():
        tla.copy(flat, source)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.copy(tile, packed, tla.params.CopyUbToGmParams(compact=True))
        tla.pipe_barrier(tla.pipes.ALL)


def main():
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    payload = torch.arange(32, dtype=torch.uint8).reshape(4, 8)
    padded = torch.full((4, 16), 0xA5, dtype=torch.uint8)
    padded[:, :8] = payload
    rows_in = padded.npu()
    flat_out = torch.full((1, 32), 0xFF, dtype=torch.uint8).npu()
    inputs = (
        from_dlpack(rows_in, layout_tag=tla.arch.RowMajor),
        from_dlpack(flat_out, layout_tag=tla.arch.RowMajor),
    )
    tla.compile(load_compact, *inputs, options="--npu-arch 3510")(*inputs, block_num=1)
    torch.testing.assert_close(flat_out.cpu(), payload.reshape(1, 32), rtol=0, atol=0)

    # Give all 128 UB bytes known values: reading rows at 32-byte intervals picks up 0xEE.
    source = torch.full((1, 128), 0xEE, dtype=torch.uint8)
    source[0, :32] = payload.flatten()
    flat_in = source.npu()
    rows_out = torch.full_like(rows_in, 0xA5)
    inputs = (
        from_dlpack(flat_in, layout_tag=tla.arch.RowMajor),
        from_dlpack(rows_out, layout_tag=tla.arch.RowMajor),
    )
    tla.compile(store_compact, *inputs, options="--npu-arch 3510")(*inputs, block_num=1)
    torch.testing.assert_close(rows_out.cpu(), padded, rtol=0, atol=0)


if __name__ == "__main__":
    main()
