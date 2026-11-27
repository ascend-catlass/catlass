"""Validate L1 numeric fill through an independently referenced MX matmul."""

import pytest

from catlass import tla
from catlass.tla.runtime import from_dlpack

pytestmark = pytest.mark.npu


def _kernel_for(dtype, a_value, b_value, a_scale, b_scale, pattern):
    @tla.kernel
    def kernel(output: tla.Tensor, rows: tla.Int32):
        m = 48
        n = 48
        k = 256
        groups = 8
        pa = tla.allocate(m * k, dtype, tla.AddressSpace.l1, 512)
        pb = tla.allocate(k * n, dtype, tla.AddressSpace.l1, 512)
        psa = tla.allocate(m * groups, tla.Float8E8M0, tla.AddressSpace.l1, 512)
        psb = tla.allocate(groups * n, tla.Float8E8M0, tla.AddressSpace.l1, 512)
        p0a = tla.allocate(m * k, dtype, tla.AddressSpace.l0a, 1024)
        p0b = tla.allocate(k * n, dtype, tla.AddressSpace.l0b, 1024)
        p0c = tla.allocate(m * n, tla.Float32, tla.AddressSpace.l0c, 1024)
        a = tla.make_tensor(pa, tla.make_layout_with_tag((m, k), dtype, tla.arch.zN))
        b = tla.make_tensor(pb, tla.make_layout_with_tag((k, n), dtype, tla.arch.nZ))
        sa = tla.make_tensor(
            psa,
            tla.make_layout_with_tag((m, groups), tla.Float8E8M0, tla.arch.zZMxScale),
        )
        sb = tla.make_tensor(
            psb,
            tla.make_layout_with_tag((groups, n), tla.Float8E8M0, tla.arch.nNMxScale),
        )
        a0 = tla.make_tensor(p0a, tla.make_layout_with_tag((m, k), dtype, tla.arch.zN))
        b0 = tla.make_tensor(p0b, tla.make_layout_with_tag((k, n), dtype, tla.arch.nZ))
        c0 = tla.make_tensor(
            p0c, tla.make_layout_with_tag((m, n), tla.Float32, tla.arch.L0Clayout)
        )
        filled = tla.flag("filled", tla.pipes.MTE2, tla.pipes.MTE1)
        loaded = tla.flag("loaded", tla.pipes.MTE1, tla.pipes.CUBE)
        with tla.cube():
            a.fill(a_value)
            b.fill(b_value)
            sa.fill(a_scale)
            sb.fill(b_scale)
            if tla.const_expr(pattern == "subtiles"):
                aa = tla.get_tile(a, tla.make_coord(16, 64), tla.make_shape(16, 128))
                bb = tla.get_tile(b, tla.make_coord(64, 16), tla.make_shape(128, 16))
                saa = tla.get_tile(sa, tla.make_coord(16, 2), tla.make_shape(16, 4))
                sbb = tla.get_tile(sb, tla.make_coord(2, 16), tla.make_shape(4, 16))
                aa.fill(-1.0)
                bb.fill(-0.5)
                saa.fill(2.0)
                sbb.fill(0.5)
            if tla.const_expr(pattern == "dynamic-prefix"):
                saa = tla.get_tile(sa, tla.make_coord(0, 2), tla.make_shape(rows, 2))
                sbb = tla.get_tile(sb, tla.make_coord(2, 0), tla.make_shape(2, rows))
                saa.fill(2.0)
                sbb.fill(0.5)
            if tla.const_expr(pattern == "unaligned-start"):
                saa = tla.get_tile(sa, tla.make_coord(1, 1), tla.make_shape(31, 3))
                sbb = tla.get_tile(sb, tla.make_coord(1, 1), tla.make_shape(3, 31))
                saa.fill(2.0)
                sbb.fill(0.5)
            tla.set_flag(filled)
            tla.wait_flag(filled)
            tla.copy(a0, a, scale=sa)
            tla.copy(b0, b, scale=sb)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            tla.mmad_mx(c0, a0, b0, init_c=True, unit_flag=3)
            tla.copy(output, c0, tla.params.CopyL0C2DstParams(unit_flag=3))
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel


def _check_numeric_fill(device, dtype, a, b, sa, sb, pattern):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    guarded = torch.full(
        (48 * 48 + 128,), -42.0, dtype=torch.float32, device=f"npu:{device}"
    )
    output = guarded[64:-64].view(48, 48)
    argument = from_dlpack(output, layout_tag=tla.arch.RowMajor)
    compiled = _kernel_for(dtype, a, b, sa, sb, pattern).compile(
        type_args=(argument, tla.Int32(0)), options="--npu-arch 3510"
    )
    # Correctness-only repetitions exercise dynamic empty/tail extents and
    # reuse the same output and on-chip allocation addresses.
    for rows in (
        [32, 0, 1, 15, 17, 16, 31, 32] if pattern == "dynamic-prefix" else [0, 0]
    ):
        guarded.fill_(-42.0)
        compiled(argument, rows, block_num=1)
        torch.npu.synchronize()
        aa = torch.full((48, 256), a)
        bb = torch.full((256, 48), b)
        saa = torch.full((48, 8), sa)
        sbb = torch.full((8, 48), sb)
        if pattern == "subtiles":
            aa[16:32, 64:192] = -1.0
            bb[64:192, 16:32] = -0.5
            saa[16:32, 2:6] = 2.0
            sbb[2:6, 16:32] = 0.5
        if pattern == "dynamic-prefix":
            physical_rows = (rows + 15) // 16 * 16
            saa[:physical_rows, 2:4] = 2.0
            sbb[2:4, :physical_rows] = 0.5
        if pattern == "unaligned-start":
            # The leading partial 16-by-2 fractals are untouched.
            saa[16:32, 2:4] = 2.0
            sbb[2:4, 16:32] = 0.5
        expected = (aa * saa.repeat_interleave(32, dim=1)) @ (
            bb * sbb.repeat_interleave(32, dim=0)
        )
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        assert (guarded[:64].cpu() == -42).all()
        assert (guarded[-64:].cpu() == -42).all()


@pytest.mark.parametrize(
    "dtype,a,b,sa,sb",
    [
        (tla.Float8E4M3FN, 1.0, -2.0, 0.5, 2.0),
        (tla.Float8E5M2, -1.5, 2.0, 2.0, 0.5),
        (tla.Float4E2M1, 1.5, -0.5, 1.0, 0.25),
        (tla.Float4E1M2, 1.25, -0.75, 4.0, 0.5),
    ],
)
@pytest.mark.parametrize(
    "pattern", ["full", "subtiles", "dynamic-prefix", "unaligned-start"]
)
def test_numeric_fill_mx_matmul(device, dtype, a, b, sa, sb, pattern):
    _check_numeric_fill(device, dtype, a, b, sa, sb, pattern)


@pytest.mark.parametrize("dtype", [tla.Float8E4M3FN, tla.Float4E2M1])
def test_zero_fill_mx_matmul(device, dtype):
    _check_numeric_fill(device, dtype, 0.0, 1.0, 1.0, 1.0, "full")
