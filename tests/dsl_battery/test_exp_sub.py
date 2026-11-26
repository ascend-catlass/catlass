"""ExpSub device checks for both input types, both parts and native ZEROING."""

import pytest

pytestmark = pytest.mark.npu


def _kernel(mask_mode="explicit", lhs_lanes=None, rhs_lanes=None):
    global tla
    import catlass.tla as tla

    @tla.kernel
    def exp_sub_vectors(
        lhs: tla.Tensor,
        rhs: tla.Tensor,
        masks: tla.Tensor,
        part0: tla.Tensor,
        part1: tla.Tensor,
        valid: tla.Int32,
        lanes: tla.Constexpr[int],
        rows: tla.Constexpr[int],
        source_dtype: tla.Constexpr[type],
        irregular: tla.Constexpr[bool],
    ):
        lhs_ptr = tla.allocate(rows * lanes, source_dtype, tla.AddressSpace.ub, 32)
        rhs_ptr = tla.allocate(rows * lanes, source_dtype, tla.AddressSpace.ub, 32)
        mask_ptr = tla.allocate(rows * lanes, source_dtype, tla.AddressSpace.ub, 32)
        out0_ptr = tla.allocate(rows * 64, tla.Float32, tla.AddressSpace.ub, 32)
        out1_ptr = tla.allocate(rows * 64, tla.Float32, tla.AddressSpace.ub, 32)
        local_lhs = tla.make_tensor_like(lhs_ptr, lhs, tla.arch.RowMajor)
        local_rhs = tla.make_tensor_like(rhs_ptr, rhs, tla.arch.RowMajor)
        local_masks = tla.make_tensor_like(mask_ptr, masks, tla.arch.RowMajor)
        local_out0 = tla.make_tensor_like(out0_ptr, part0, tla.arch.RowMajor)
        local_out1 = tla.make_tensor_like(out1_ptr, part1, tla.arch.RowMajor)
        loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
        stored = tla.flag("stored", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(local_lhs, lhs)
            tla.copy(local_rhs, rhs)
            tla.copy(local_masks, masks)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            with tla.vec.func(mode="simd"):
                prefix_mask, _ = tla.update_mask(valid, dtype=source_dtype)
                for row in tla.range(0, rows, 1):
                    if tla.const_expr(lhs_lanes is None):
                        a = tla.tile_view(
                            local_lhs, tla.make_shape(lanes), tla.make_coord(row)
                        ).load()
                    else:
                        a = tla.make_tensor(
                            lhs_ptr,
                            tla.make_layout(
                                tla.make_shape(lanes),
                                tla.make_stride(1),
                                origin_shape=tla.make_shape(lhs_lanes),
                            ),
                            coord=tla.make_coord(row * lanes),
                        ).load()
                    if tla.const_expr(rhs_lanes is None):
                        b = tla.tile_view(
                            local_rhs, tla.make_shape(lanes), tla.make_coord(row)
                        ).load()
                    else:
                        b = tla.make_tensor(
                            rhs_ptr,
                            tla.make_layout(
                                tla.make_shape(lanes),
                                tla.make_stride(1),
                                origin_shape=tla.make_shape(rhs_lanes),
                            ),
                            coord=tla.make_coord(row * lanes),
                        ).load()
                    if tla.const_expr(irregular):
                        mask_values = tla.tile_view(
                            local_masks, tla.make_shape(lanes), tla.make_coord(row)
                        ).load()
                        active = tla.cmp(mask_values, 0.0, "gt")
                    else:
                        active = prefix_mask
                    if tla.const_expr(mask_mode == "omitted"):
                        result0 = tla.arch.sub_exp(a, b)
                        result1 = tla.arch.sub_exp(
                            a, b, reg_slot=tla.params.RegSlot.ONE
                        )
                    elif tla.const_expr(mask_mode == "none"):
                        result0 = tla.arch.sub_exp(a, b, mask=None)
                        result1 = tla.arch.sub_exp(
                            a, b, reg_slot=tla.params.RegSlot.ONE, mask=None
                        )
                    else:
                        result0 = tla.arch.sub_exp(
                            a, b, reg_slot=tla.params.RegSlot.ZERO, mask=active
                        )
                        result1 = tla.arch.sub_exp(
                            a, b, reg_slot=tla.params.RegSlot.ONE, mask=active
                        )
                    # Store every output lane so ZEROING itself is observed.
                    tla.tile_view(
                        local_out0, tla.make_shape(64), tla.make_coord(row)
                    ).store(result0)
                    tla.tile_view(
                        local_out1, tla.make_shape(64), tla.make_coord(row)
                    ).store(result1)
            tla.set_flag(stored)
            tla.wait_flag(stored)
            tla.copy(part0, local_out0)
            tla.copy(part1, local_out1)
            tla.pipe_barrier(tla.pipes.ALL)

    return exp_sub_vectors


def _check(
    device,
    dtype_name,
    valid,
    pattern,
    mask_mode="explicit",
    lhs_lanes=None,
    rhs_lanes=None,
):
    import torch
    import torch_npu  # noqa: F401
    import catlass.tla as tla
    from catlass.tla.runtime import from_dlpack, make_fake_tensor

    kernel = _kernel(mask_mode, lhs_lanes, rhs_lanes)
    torch.npu.set_device(device)
    half_input = dtype_name == "f16"
    source_dtype = tla.Float16 if half_input else tla.Float32
    torch_dtype = torch.float16 if half_input else torch.float32
    bits_dtype = torch.int16 if half_input else torch.int32
    lanes = 128 if half_input else 64
    rows = 128
    lhs = torch.empty(rows * lanes, dtype=torch_dtype, device=f"npu:{device}")
    rhs = torch.empty_like(lhs)
    masks = torch.empty_like(lhs)
    guarded = [
        torch.full(
            (rows * 64 + 32,),
            -42.0,
            dtype=torch.float32,
            device=lhs.device,
        )
        for _ in range(2)
    ]
    outputs = [buffer[16:-16] for buffer in guarded]
    source_fake = make_fake_tensor(
        source_dtype,
        (rows * lanes,),
        (1,),
        origin_shape=(rows * lanes,),
        layout_tag=tla.arch.RowMajor,
    )
    output_fake = make_fake_tensor(
        tla.Float32,
        (rows * 64,),
        (1,),
        origin_shape=(rows * 64,),
        layout_tag=tla.arch.RowMajor,
    )
    compiled = kernel.compile(
        type_args=(
            source_fake,
            source_fake,
            source_fake,
            output_fake,
            output_fake,
            tla.Int32(valid),
            lanes,
            rows,
            source_dtype,
            pattern != "prefix",
        ),
        options="--npu-arch 3510",
    )
    for seed in [123, 456]:
        generator = torch.Generator().manual_seed(seed)
        a = (torch.rand((rows, lanes), generator=generator) * 8 - 4).to(torch_dtype)
        b = (torch.rand((rows, lanes), generator=generator) * 8 - 4).to(torch_dtype)
        tiny = 2**-10 if half_input else 2**-20
        a[:, :8] = torch.tensor(
            [8.0, 8.0, -8.0, -8.0, 0.0, -0.0, 2.0, -2.0], dtype=torch_dtype
        )
        b[:, :8] = torch.tensor(
            [tiny, -tiny, tiny, -tiny, 0.0, 0.0, 2.0, -2.0], dtype=torch_dtype
        )
        if mask_mode != "explicit":
            active = torch.ones((rows, lanes), dtype=torch.bool)
        elif pattern == "prefix":
            active = (torch.arange(lanes) < valid).expand(rows, lanes).clone()
        elif pattern == "holes":
            active = (
                torch.arange(lanes)[None, :] + torch.arange(rows)[:, None] + seed
            ) % 3 != 0
        else:
            active = torch.rand((rows, lanes), generator=generator) > 0.5
        # Keep the supplied mask independent of static bounds: it may enable
        # poisoned tail lanes that ExpSub must suppress from the vector types.
        mask_values = active.to(torch_dtype)
        static_bound = min(n for n in (lanes, lhs_lanes, rhs_lanes) if n is not None)
        active &= torch.arange(lanes) < static_bound
        # Inactive NaN/Inf inputs must still produce positive zero. Changing
        # masks between rows and calls also exposes stale merging results.
        a[~active] = float("nan") if seed == 123 else float("inf")
        b[~active] = 0.0 if seed == 123 else float("inf")
        lhs.copy_(a.flatten())
        rhs.copy_(b.flatten())
        masks.copy_(mask_values.flatten())
        for buffer in guarded:
            buffer.fill_(-42)
        compiled(
            from_dlpack(lhs, layout_tag=tla.arch.RowMajor),
            from_dlpack(rhs, layout_tag=tla.arch.RowMajor),
            from_dlpack(masks, layout_tag=tla.arch.RowMajor),
            from_dlpack(outputs[0], layout_tag=tla.arch.RowMajor),
            from_dlpack(outputs[1], layout_tag=tla.arch.RowMajor),
            valid,
            block_num=1,
        )
        torch.npu.synchronize()
        reference = torch.exp(a.float() - b.float())
        for part, (output, buffer) in enumerate(zip(outputs, guarded)):
            kept = active[:, part::2] if half_input else active
            values = reference[:, part::2] if half_input else reference
            expected = torch.where(kept, values, torch.zeros_like(values))
            actual = output.cpu().reshape(rows, 64)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            assert (actual[~kept].view(torch.int32) == 0).all()
            assert (buffer[:16].cpu() == -42).all()
            assert (buffer[-16:].cpu() == -42).all()
        assert torch.equal(lhs.cpu().view(bits_dtype), a.flatten().view(bits_dtype))
        assert torch.equal(rhs.cpu().view(bits_dtype), b.flatten().view(bits_dtype))
        assert torch.equal(masks.cpu(), mask_values.flatten())


@pytest.mark.parametrize(
    "dtype_name,valid",
    [("f16", n) for n in [0, 1, 2, 63, 64, 65, 127, 128]]
    + [("f32", n) for n in [0, 1, 2, 31, 32, 33, 63, 64]],
)
def test_exp_sub_prefix_zeroing_and_reused_buffers(device, dtype_name, valid):
    _check(device, dtype_name, valid, "prefix")


@pytest.mark.parametrize("dtype_name", ["f16", "f32"])
@pytest.mark.parametrize("pattern", ["holes", "random"])
def test_exp_sub_nonprefix_zeroing(device, dtype_name, pattern):
    _check(device, dtype_name, 0, pattern)


@pytest.mark.parametrize("dtype_name", ["f16", "f32"])
@pytest.mark.parametrize("mask_mode", ["omitted", "none"])
def test_exp_sub_optional_mask_is_all_active(device, dtype_name, mask_mode):
    # Even with an existing all-inactive prefix mask, the default is all-active.
    _check(device, dtype_name, 0, "prefix", mask_mode)


@pytest.mark.parametrize(
    "dtype_name,static_lanes",
    [("f16", n) for n in [1, 3, 64, 65, 127]]
    + [("f32", n) for n in [1, 3, 32, 33, 63]],
)
@pytest.mark.parametrize("mask_mode", ["omitted", "none", "explicit"])
def test_exp_sub_static_prefix_zeroing(device, dtype_name, static_lanes, mask_mode):
    capacity = 128 if dtype_name == "f16" else 64
    _check(
        device,
        dtype_name,
        capacity,
        "prefix",
        mask_mode,
        static_lanes,
        static_lanes,
    )


@pytest.mark.parametrize("dtype_name", ["f16", "f32"])
@pytest.mark.parametrize("pattern", ["holes", "random"])
def test_exp_sub_static_prefix_with_nonprefix_mask(device, dtype_name, pattern):
    _check(device, dtype_name, 0, pattern, lhs_lanes=3, rhs_lanes=3)


@pytest.mark.parametrize("dtype_name", ["f16", "f32"])
@pytest.mark.parametrize("rhs_lanes", [None, 1])
def test_exp_sub_different_static_prefixes(device, dtype_name, rhs_lanes):
    capacity = 128 if dtype_name == "f16" else 64
    _check(
        device,
        dtype_name,
        capacity,
        "prefix",
        lhs_lanes=3,
        rhs_lanes=rhs_lanes,
    )


@pytest.mark.parametrize("dtype_name", ["f16", "f32"])
def test_exp_sub_static_prefix_all_inactive(device, dtype_name):
    _check(device, dtype_name, 0, "prefix", lhs_lanes=3, rhs_lanes=3)
