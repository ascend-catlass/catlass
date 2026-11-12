"""Small on-device regression for mixed print capacity and VF progress."""

import pytest

pytestmark = pytest.mark.npu


def _capacity_kernel():
    global tla
    import catlass.tla as tla

    @tla.kernel
    def capacity_print(
        source: tla.Tensor, destination: tla.Tensor, repetitions: tla.Int32
    ):
        pointer = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        local = tla.make_tensor_like(pointer, source, tla.arch.RowMajor)
        loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
        stored = tla.flag("stored", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(local, source)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            tla.print("start")
            tla.print(source, 4)
            with tla.vec.func(mode="simd"):
                values = local.load()
                for iteration in tla.range(0, repetitions, 1):
                    tla.print(values, 4)
                local.store(values + 1.0)
            tla.print("end")
            tla.set_flag(stored)
            tla.wait_flag(stored)
            tla.copy(destination, local)
            tla.pipe_barrier(tla.pipes.ALL)

    return capacity_print


def test_shared_print_capacity(device, monkeypatch, capfd):
    import torch
    import torch_npu  # noqa: F401
    import catlass.tla as tla
    from catlass.tla.runtime import make_fake_tensor, from_dlpack
    from catlass.base_dsl.runtime import ascend_debug_fifo as fifo

    torch.npu.set_device(device)
    fake = make_fake_tensor(
        tla.Float32, (64,), (1,), origin_shape=(64,), layout_tag=tla.arch.RowMajor
    )
    compiled = _capacity_kernel().compile(
        type_args=(fake, fake, tla.Int32(2)), options="--npu-arch 3510"
    )
    # Private test capacities preserve the CANN control record's 8-byte alignment.
    cases = (
        (512, 2, False, 3),
        (376, 2, False, 3),
        (368, 2, True, 3),
        (512, 20, True, 4),
        (24, 2, True, 0),
    )
    for capacity, repetitions, truncated, records in cases:
        monkeypatch.setattr(fifo, "_RING_BUFFER_BYTES", capacity)
        source = torch.arange(64, dtype=torch.float32, device=f"npu:{device}")
        guarded = torch.full((96,), -99.0, device=f"npu:{device}")
        destination = guarded[16:80]
        capfd.readouterr()
        compiled(
            from_dlpack(source, layout_tag=tla.arch.RowMajor),
            from_dlpack(destination, layout_tag=tla.arch.RowMajor),
            repetitions,
            block_num=1,
        )
        output = capfd.readouterr().out
        torch.testing.assert_close(destination.cpu(), source.cpu() + 1, rtol=0, atol=0)
        assert (guarded[:16].cpu() == -99).all() and (guarded[80:].cpu() == -99).all()
        assert ("output truncated" in output) == truncated, output
        assert output.count("DumpTensor:") == records, output
        assert "malformed" not in output


@pytest.mark.parametrize("subblocks", [1, 2])
def test_mixed_print_core_ratio(device, subblocks, capfd):
    import re
    import torch
    import torch_npu  # noqa: F401
    import catlass.tla as tla
    from catlass.tla.runtime import from_dlpack, make_fake_tensor
    import example_runner

    torch.npu.set_device(device)
    ratio = tla.arch.C1V1 if subblocks == 1 else tla.arch.C1V2
    example = example_runner.load_example(
        example_runner.resolve_script("dsl_tutorials/debug_print/vf_print_mixed.py")
    )
    kernel = tla.kernel(core_ratio=ratio)(example.vf_print_mixed.fn)
    blocks = 3
    # The example uses a two-tile stride. C1V1 visits even tiles only, which
    # also gives a protected gap for detecting writes by a spurious subblock.
    elements = blocks * 2 * 64
    fake = make_fake_tensor(tla.Float32, (elements,), (1,), origin_shape=(elements,))
    compiled = kernel.compile(type_args=(fake, fake), options="--npu-arch 3510")
    assert compiled.jit_module.print_config.aiv_subblock_count == subblocks
    pattern = re.compile(
        r"DumpTensor: call=([01]), block=(\d+), subblock=([01]), "
        r"data_type=float32, position=(REG|UB), shape=\[(?:4|64)\] dump_size=4 \[(.*?)\]"
    )
    for update in (0, 17):
        source = (
            torch.arange(elements, dtype=torch.float32, device=f"npu:{device}") + update
        )
        destination = torch.full_like(source, -99)
        expected = torch.full((elements,), -99.0)
        for block in range(blocks):
            for subblock in range(subblocks):
                start = (block * 2 + subblock) * 64
                expected[start : start + 64] = (
                    torch.arange(start, start + 64) * 2 + update * 2 + 1
                )
        capfd.readouterr()
        compiled(
            from_dlpack(source, layout_tag=tla.arch.RowMajor),
            from_dlpack(destination, layout_tag=tla.arch.RowMajor),
            block_num=blocks,
        )
        output = capfd.readouterr().out
        torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
        matches = pattern.findall(output)
        identities = [
            (int(block), int(subblock), position)
            for _, block, subblock, position, _ in matches
        ]
        assert len(matches) == blocks * subblocks * 2, output
        assert set(identities) == {
            (block, subblock, position)
            for block in range(blocks)
            for subblock in range(subblocks)
            for position in ("REG", "UB")
        }, output
        for call, block, subblock, position, values in matches:
            assert int(call) == (0 if position == "REG" else 1)
            start = (int(block) * 2 + int(subblock)) * 64
            assert [float(x) for x in values.split(",")] == expected[
                start : start + 4
            ].tolist()
        for block in range(blocks):
            assert output.count(f"cube_block={block}\n") == 1, output
            for subblock in range(subblocks):
                tile = block * 2 + subblock
                assert (
                    output.count(
                        f"simt_tile={tile} value={float(tile * 64 + update):f}\n"
                    )
                    == 1
                ), output
        assert (
            len(re.findall(r"DumpTensor: call=2, block=\d+, subblock=", output))
            == blocks * subblocks
        ), output
        assert output.count("DumpTensor: call=3,") == blocks * subblocks, output
        assert output.count("DumpTensor: block=(") == blocks * subblocks, output
        assert "malformed" not in output and "output truncated" not in output
