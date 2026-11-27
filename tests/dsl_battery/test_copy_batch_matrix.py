"""Rank-3 copy with random values, independent pitches and interleaved NZ."""

import pytest

from catlass import tla
from catlass.tla.runtime import from_dlpack

pytestmark = pytest.mark.npu


def _kernel(dtype, scale_column):
    @tla.kernel
    def kernel(
        data: tla.Tensor,
        exponents: tla.Tensor,
        ones: tla.Tensor,
        unit_exponents: tla.Tensor,
        output: tla.Tensor,
        count: tla.Int32,
        rows: tla.Int32,
        src_start: tla.Int32,
        dst_start: tla.Int32,
        data_step: tla.Int64,
        scale_step: tla.Int64,
    ):
        cap = 5
        m = 32
        k = 256
        n = 16
        groups = 8
        a_ptr = tla.allocate(cap * m * k, dtype, tla.AddressSpace.l1, 512)
        sa_ptr = tla.allocate(
            cap * m * groups, tla.Float8E8M0, tla.AddressSpace.l1, 512
        )
        b_ptr = tla.allocate(k * n, dtype, tla.AddressSpace.l1, 512)
        sb_ptr = tla.allocate(groups * n, tla.Float8E8M0, tla.AddressSpace.l1, 512)
        a0_ptr = tla.allocate(m * k, dtype, tla.AddressSpace.l0a, 1024)
        b0_ptr = tla.allocate(k * n, dtype, tla.AddressSpace.l0b, 1024)
        c_ptr = tla.allocate(m * n, tla.Float32, tla.AddressSpace.l0c, 1024)
        all_a = tla.make_tensor(
            a_ptr, tla.make_layout_with_tag((cap * m, k), dtype, tla.arch.zN)
        )
        all_sa = tla.make_tensor(
            sa_ptr,
            tla.make_layout_with_tag(
                (cap * m, groups), tla.Float8E8M0, tla.arch.zZMxScale
            ),
        )
        b = tla.make_tensor(b_ptr, tla.make_layout_with_tag((k, n), dtype, tla.arch.nZ))
        sb = tla.make_tensor(
            sb_ptr,
            tla.make_layout_with_tag((groups, n), tla.Float8E8M0, tla.arch.nNMxScale),
        )
        # Initialize with existing 2-D copies; nonzero Tensor.fill is not
        # required by this feature or its tests.
        ones_gm = tla.make_tensor(
            tla.recast_ptr(ones.ptr, dtype=dtype),
            tla.make_layout(
                tla.make_shape(k, n),
                tla.make_stride(1, k),
                layoutTag=tla.arch.ColumnMajor,
            ),
        )
        unit_sa_gm = tla.make_tensor(
            tla.recast_ptr(unit_exponents.ptr, dtype=tla.Float8E8M0),
            tla.make_layout(
                tla.make_shape((1, cap * m), (2, groups // 2)),
                tla.make_stride((1, groups), (1, 2)),
                origin_shape=tla.make_shape(cap * m, groups),
                layoutTag=tla.arch.RowMajorMxScaleA,
            ),
        )
        unit_sb_gm = tla.make_tensor(
            tla.recast_ptr(unit_exponents.ptr, dtype=tla.Float8E8M0),
            tla.make_layout(
                tla.make_shape((2, groups // 2), (1, n)),
                tla.make_stride((1, 2), (1, groups)),
                origin_shape=tla.make_shape(groups, n),
                layoutTag=tla.arch.ColMajorMxScaleB,
            ),
        )
        source = tla.make_tensor(
            tla.recast_ptr(data.ptr, dtype=dtype),
            tla.make_layout(
                tla.make_shape(cap, m, k), tla.make_stride(data_step, k, 1)
            ),
        )
        # Each matrix owns m rows inside every C0 slab of the shared pool.
        destination = tla.make_tensor(
            a_ptr,
            tla.make_layout(
                tla.make_shape(cap, (16, 2), (32, 8)),
                tla.make_stride(m * 32, (32, 512), (1, cap * m * 32)),
                origin_shape=tla.make_shape(cap, m, k),
                layoutTag=tla.arch.zN,
            ),
        )
        if tla.const_expr(scale_column):
            source_scale = tla.make_tensor(
                tla.recast_ptr(exponents.ptr, dtype=tla.Float8E8M0),
                tla.make_layout(
                    tla.make_shape(cap, (1, m), (2, 4)),
                    tla.make_stride(scale_step, (1, 2), (1, m * 2)),
                    origin_shape=tla.make_shape(cap, m, groups),
                    layoutTag=tla.arch.ColMajorMxScaleA,
                ),
            )
        else:
            source_scale = tla.make_tensor(
                tla.recast_ptr(exponents.ptr, dtype=tla.Float8E8M0),
                tla.make_layout(
                    tla.make_shape(cap, (1, m), (2, 4)),
                    tla.make_stride(scale_step, (1, groups), (1, 2)),
                    origin_shape=tla.make_shape(cap, m, groups),
                    layoutTag=tla.arch.RowMajorMxScaleA,
                ),
            )
        destination_scale = tla.make_tensor(
            sa_ptr,
            tla.make_layout(
                tla.make_shape(cap, (16, 2), (2, 4)),
                tla.make_stride(m * groups, (2, groups * 16), (1, 32)),
                origin_shape=tla.make_shape(cap, m, groups),
                layoutTag=tla.arch.zZMxScale,
            ),
        )
        source = tla.get_tile(
            source, tla.make_coord(src_start, 0, 0), tla.make_shape(count, rows, k)
        )
        destination = tla.get_tile(
            destination, tla.make_coord(dst_start, 0, 0), tla.make_shape(count, rows, k)
        )
        source_scale = tla.get_tile(
            source_scale,
            tla.make_coord(src_start, 0, 0),
            tla.make_shape(count, rows, groups),
        )
        destination_scale = tla.get_tile(
            destination_scale,
            tla.make_coord(dst_start, 0, 0),
            tla.make_shape(count, rows, groups),
        )
        a0 = tla.make_tensor(
            a0_ptr, tla.make_layout_with_tag((m, k), dtype, tla.arch.zN)
        )
        b0 = tla.make_tensor(
            b0_ptr, tla.make_layout_with_tag((k, n), dtype, tla.arch.nZ)
        )
        c = tla.make_tensor(
            c_ptr, tla.make_layout_with_tag((m, n), tla.Float32, tla.arch.L0Clayout)
        )
        filled = tla.flag("batch_filled", tla.pipes.MTE2, tla.pipes.MTE1)
        loaded = tla.flag("batch_loaded", tla.pipes.MTE1, tla.pipes.CUBE)
        consumed = tla.flag("batch_consumed", tla.pipes.CUBE, tla.pipes.MTE1)
        with tla.cube():
            all_a.fill(0)
            tla.copy(all_sa, unit_sa_gm)
            tla.copy(b, ones_gm)
            tla.copy(sb, unit_sb_gm)
            tla.copy(destination, source)
            tla.copy(destination_scale, source_scale)
            tla.set_flag(filled)
            tla.wait_flag(filled)
            for batch in tla.range(count):
                a = tla.make_tensor(
                    a_ptr + (batch + dst_start) * m * 32,
                    tla.make_layout(
                        tla.make_shape((16, 2), (32, 8)),
                        tla.make_stride((32, 512), (1, cap * m * 32)),
                        origin_shape=tla.make_shape(m, k),
                        layoutTag=tla.arch.zN,
                    ),
                )
                sa = tla.make_tensor(
                    sa_ptr + (batch + dst_start) * m * groups,
                    tla.make_layout_with_tag(
                        (m, groups), tla.Float8E8M0, tla.arch.zZMxScale
                    ),
                )
                tla.copy(a0, a, scale=sa)
                tla.copy(b0, b, scale=sb)
                tla.set_flag(loaded)
                tla.wait_flag(loaded)
                tla.mmad_mx(c, a0, b0, init_c=True, unit_flag=3)
                tla.set_flag(consumed)
                tla.wait_flag(consumed)
                out = tla.make_tensor(
                    output.ptr + batch * m * n,
                    tla.make_layout(tla.make_shape(m, n), tla.make_stride(n, 1)),
                )
                tla.copy(out, c, tla.params.CopyL0C2DstParams(unit_flag=3))
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel


def _kernel_b(dtype, scale_column):
    @tla.kernel
    def kernel(
        data: tla.Tensor,
        exponents: tla.Tensor,
        ones: tla.Tensor,
        unit_exponents: tla.Tensor,
        output: tla.Tensor,
        count: tla.Int32,
        rows: tla.Int32,
        src_start: tla.Int32,
        dst_start: tla.Int32,
        data_step: tla.Int64,
        scale_step: tla.Int64,
    ):
        cap, extent, k, m, groups = 5, 32, 256, 16, 8
        bp = tla.allocate(cap * extent * k, dtype, tla.AddressSpace.l1, 512)
        sp = tla.allocate(
            cap * extent * groups, tla.Float8E8M0, tla.AddressSpace.l1, 512
        )
        ap = tla.allocate(m * k, dtype, tla.AddressSpace.l1, 512)
        sap = tla.allocate(m * groups, tla.Float8E8M0, tla.AddressSpace.l1, 512)
        a0p = tla.allocate(m * k, dtype, tla.AddressSpace.l0a, 1024)
        b0p = tla.allocate(k * extent, dtype, tla.AddressSpace.l0b, 1024)
        cp = tla.allocate(m * extent, tla.Float32, tla.AddressSpace.l0c, 1024)
        all_b = tla.make_tensor(
            bp, tla.make_layout_with_tag((k, cap * extent), dtype, tla.arch.nZ)
        )
        all_s = tla.make_tensor(
            sp,
            tla.make_layout_with_tag(
                (groups, cap * extent), tla.Float8E8M0, tla.arch.nNMxScale
            ),
        )
        a = tla.make_tensor(ap, tla.make_layout_with_tag((m, k), dtype, tla.arch.zN))
        sa = tla.make_tensor(
            sap,
            tla.make_layout_with_tag((m, groups), tla.Float8E8M0, tla.arch.zZMxScale),
        )
        ones_gm = tla.make_tensor(
            tla.recast_ptr(ones.ptr, dtype=dtype),
            tla.make_layout(tla.make_shape(m, k), tla.make_stride(k, 1)),
        )
        unit_sa_gm = tla.make_tensor(
            tla.recast_ptr(unit_exponents.ptr, dtype=tla.Float8E8M0),
            tla.make_layout(
                tla.make_shape((1, m), (2, groups // 2)),
                tla.make_stride((1, groups), (1, 2)),
                origin_shape=tla.make_shape(m, groups),
                layoutTag=tla.arch.RowMajorMxScaleA,
            ),
        )
        unit_sb_gm = tla.make_tensor(
            tla.recast_ptr(unit_exponents.ptr, dtype=tla.Float8E8M0),
            tla.make_layout(
                tla.make_shape((2, groups // 2), (1, cap * extent)),
                tla.make_stride((1, 2), (1, groups)),
                origin_shape=tla.make_shape(groups, cap * extent),
                layoutTag=tla.arch.ColMajorMxScaleB,
            ),
        )
        source = tla.make_tensor(
            tla.recast_ptr(data.ptr, dtype=dtype),
            tla.make_layout(
                tla.make_shape(cap, k, extent),
                tla.make_stride(data_step, 1, k),
                layoutTag=tla.arch.ColumnMajor,
            ),
        )
        destination = tla.make_tensor(
            bp,
            tla.make_layout(
                tla.make_shape(cap, (32, 8), (16, 2)),
                tla.make_stride(extent * 32, (1, cap * extent * 32), (32, 512)),
                origin_shape=tla.make_shape(cap, k, extent),
                layoutTag=tla.arch.nZ,
            ),
        )
        if tla.const_expr(scale_column):
            source_scale = tla.make_tensor(
                tla.recast_ptr(exponents.ptr, dtype=tla.Float8E8M0),
                tla.make_layout(
                    tla.make_shape(cap, (2, 4), (1, extent)),
                    tla.make_stride(scale_step, (1, 2), (1, groups)),
                    origin_shape=tla.make_shape(cap, groups, extent),
                    layoutTag=tla.arch.ColMajorMxScaleB,
                ),
            )
        else:
            source_scale = tla.make_tensor(
                tla.recast_ptr(exponents.ptr, dtype=tla.Float8E8M0),
                tla.make_layout(
                    tla.make_shape(cap, (2, 4), (1, extent)),
                    tla.make_stride(scale_step, (1, extent * 2), (1, 2)),
                    origin_shape=tla.make_shape(cap, groups, extent),
                    layoutTag=tla.arch.RowMajorMxScaleB,
                ),
            )
        destination_scale = tla.make_tensor(
            sp,
            tla.make_layout(
                tla.make_shape(cap, (2, 4), (16, 2)),
                tla.make_stride(extent * groups, (1, 32), (2, groups * 16)),
                origin_shape=tla.make_shape(cap, groups, extent),
                layoutTag=tla.arch.nNMxScale,
            ),
        )
        source = tla.get_tile(
            source, tla.make_coord(src_start, 0, 0), tla.make_shape(count, k, rows)
        )
        destination = tla.get_tile(
            destination, tla.make_coord(dst_start, 0, 0), tla.make_shape(count, k, rows)
        )
        source_scale = tla.get_tile(
            source_scale,
            tla.make_coord(src_start, 0, 0),
            tla.make_shape(count, groups, rows),
        )
        destination_scale = tla.get_tile(
            destination_scale,
            tla.make_coord(dst_start, 0, 0),
            tla.make_shape(count, groups, rows),
        )
        a0 = tla.make_tensor(a0p, tla.make_layout_with_tag((m, k), dtype, tla.arch.zN))
        b0 = tla.make_tensor(
            b0p, tla.make_layout_with_tag((k, extent), dtype, tla.arch.nZ)
        )
        c = tla.make_tensor(
            cp, tla.make_layout_with_tag((m, extent), tla.Float32, tla.arch.L0Clayout)
        )
        filled = tla.flag("batch_b_filled", tla.pipes.MTE2, tla.pipes.MTE1)
        loaded = tla.flag("batch_b_loaded", tla.pipes.MTE1, tla.pipes.CUBE)
        consumed = tla.flag("batch_b_consumed", tla.pipes.CUBE, tla.pipes.MTE1)
        with tla.cube():
            all_b.fill(0)
            tla.copy(all_s, unit_sb_gm)
            tla.copy(a, ones_gm)
            tla.copy(sa, unit_sa_gm)
            tla.copy(destination, source)
            tla.copy(destination_scale, source_scale)
            tla.set_flag(filled)
            tla.wait_flag(filled)
            for batch in tla.range(count):
                b = tla.make_tensor(
                    bp + (batch + dst_start) * extent * 32,
                    tla.make_layout(
                        tla.make_shape((32, 8), (16, 2)),
                        tla.make_stride((1, cap * extent * 32), (32, 512)),
                        origin_shape=tla.make_shape(k, extent),
                        layoutTag=tla.arch.nZ,
                    ),
                )
                sb = tla.make_tensor(
                    sp + (batch + dst_start) * extent * groups,
                    tla.make_layout_with_tag(
                        (groups, extent), tla.Float8E8M0, tla.arch.nNMxScale
                    ),
                )
                tla.copy(a0, a, scale=sa)
                tla.copy(b0, b, scale=sb)
                tla.set_flag(loaded)
                tla.wait_flag(loaded)
                tla.mmad_mx(c, a0, b0, init_c=True, unit_flag=3)
                tla.set_flag(consumed)
                tla.wait_flag(consumed)
                out = tla.make_tensor(
                    output.ptr + batch * m * extent,
                    tla.make_layout(
                        tla.make_shape(m, extent), tla.make_stride(extent, 1)
                    ),
                )
                tla.copy(out, c, tla.params.CopyL0C2DstParams(unit_flag=3))
            tla.pipe_barrier(tla.pipes.ALL)

    return kernel


@pytest.mark.parametrize("dtype", [tla.Float8E4M3FN, tla.Float8E5M2])
@pytest.mark.parametrize("scale_column", [False, True])
@pytest.mark.parametrize("operand_b", [False, True])
def test_batch_copy_random_dynamic_interleaved(device, dtype, scale_column, operand_b):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    device_name = f"npu:{device}"
    torch_dtype = (
        torch.float8_e4m3fn if dtype == tla.Float8E4M3FN else torch.float8_e5m2
    )
    data_step = 32 * 256 + 256
    scale_step = 32 * 8 + 32
    data = torch.empty(5 * data_step, dtype=torch.uint8, device=device_name)
    scales = torch.empty(5 * scale_step, dtype=torch.uint8, device=device_name)
    ones = torch.ones(256 * 16).to(torch_dtype).view(torch.uint8).to(device_name)
    unit_exponents = torch.full(
        (5 * 32 * 8,), 127, dtype=torch.uint8, device=device_name
    )
    guarded = torch.full(
        (5 * 32 * 16 + 128,), -42.0, dtype=torch.float32, device=device_name
    )
    output = guarded[64:-64]
    a_arg = from_dlpack(data, layout_tag=tla.arch.RowMajor)
    s_arg = from_dlpack(scales, layout_tag=tla.arch.RowMajor)
    one_arg = from_dlpack(ones, layout_tag=tla.arch.RowMajor)
    unit_arg = from_dlpack(unit_exponents, layout_tag=tla.arch.RowMajor)
    o_arg = from_dlpack(output, layout_tag=tla.arch.RowMajor)
    factory = _kernel_b if operand_b else _kernel
    compiled = factory(dtype, scale_column).compile(
        type_args=(
            a_arg,
            s_arg,
            one_arg,
            unit_arg,
            o_arg,
            tla.Int32(0),
            tla.Int32(0),
            tla.Int32(0),
            tla.Int32(0),
            tla.Int64(data_step),
            tla.Int64(scale_step),
        ),
        options="--npu-arch 3510",
    )
    generator = torch.Generator().manual_seed(48217)
    # Reuse identical input/output allocations while changing values, exponents,
    # batch count and effective rows. These are correctness invocations only.
    for phase in range(2):
        host_a = torch.randint(-3, 4, (5, 32, 256), generator=generator).float()
        host_s = torch.randint(125, 130, (5, 32, 8), generator=generator).to(
            torch.uint8
        )
        packed_a = torch.zeros(5 * data_step, dtype=torch.float32)
        packed_s = torch.full((5 * scale_step,), 127, dtype=torch.uint8)
        for batch in range(5):
            packed_a[batch * data_step : batch * data_step + 32 * 256] = host_a[
                batch
            ].flatten()
            scale = host_s[batch]
            if scale_column != operand_b:
                scale = scale.view(32, 4, 2).permute(1, 0, 2).contiguous()
            packed_s[batch * scale_step : batch * scale_step + 32 * 8] = scale.flatten()
        data.copy_(packed_a.to(torch_dtype).view(torch.uint8))
        scales.copy_(packed_s)
        cases = [
            (c, r, 0, 0)
            for c, r in [
                (0, 32),
                (1, 0),
                (1, 1),
                (2, 15),
                (3, 16),
                (5, 17),
                (5, 31),
                (5, 32),
                (2, 0),
                (3, 32),
            ]
        ]
        cases += [
            (3, 17, 2, 1),
            (4, 31, 1, 0),
            (2, 1, 0, 3),
            (3, 32, 1, 2),
            (5, 32, 0, 0),
        ]
        for count, rows, src_start, dst_start in cases:
            guarded.fill_(-42.0)
            compiled(
                a_arg,
                s_arg,
                one_arg,
                unit_arg,
                o_arg,
                count,
                rows,
                src_start,
                dst_start,
                tla.Int64(data_step),
                tla.Int64(scale_step),
                block_num=1,
            )
            torch.npu.synchronize()
            result = (
                output.cpu().view(5, 16, 32).transpose(-2, -1)
                if operand_b
                else output.cpu().view(5, 32, 16)
            )
            scale_values = torch.pow(2.0, host_s.float() - 127).repeat_interleave(
                32, dim=-1
            )
            expected = (
                (host_a * scale_values).sum(-1).unsqueeze(-1).expand(-1, -1, 16).clone()
            )
            expected[:, rows:] = 0
            torch.testing.assert_close(
                result[:count], expected[src_start : src_start + count], rtol=0, atol=0
            )
            assert (result[count:] == -42).all()
            assert (guarded[:64].cpu() == -42).all()
            assert (guarded[-64:].cpu() == -42).all()
