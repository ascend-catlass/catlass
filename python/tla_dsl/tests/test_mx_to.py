from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _ub(dtype: type[tla.Numeric], lanes: int) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


def test_mx_to_rejects_ambiguous_scale_axes() -> None:
    @tla.kernel
    def k():
        a = tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 256)
        b = tla.allocate(128, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        s = tla.allocate(128, tla.Int8, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(2, 64), tla.make_stride(64, 1))
        with tla.vector():
            tla.experimental.mx_to(tla.make_tensor(a, layout),
                                   (tla.make_tensor(b, layout), tla.make_tensor(s, layout)))

    with pytest.raises(tla.TlaCoreAPIError, match="ambiguous"):
        k.dump_mlir()


def test_mx_to_rejects_same_payload_allocation_through_recasts() -> None:
    @tla.kernel
    def k():
        data = tla.allocate(128, tla.Int8, tla.AddressSpace.ub, 256)
        scale = tla.allocate(4, tla.Int8, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(2, 64), tla.make_stride(64, 1))
        scales = tla.make_layout(tla.make_shape(2, 2), tla.make_stride(2, 1))
        a = tla.make_tensor(tla.recast_ptr(data, dtype=tla.Float4E2M1), layout)
        b = tla.make_tensor(tla.recast_ptr(data, dtype=tla.Float8E4M3FN), layout)
        s = tla.make_tensor(scale, scales)
        with tla.vector():
            tla.experimental.mx_to((a, s), (b, s))

    with pytest.raises(tla.TlaCoreAPIError, match="must not overlap"):
        k.dump_mlir()


def test_mx_to_rejects_payload_and_scale_with_same_backing() -> None:
    @tla.kernel
    def k():
        src_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        data_ptr = tla.allocate(64, tla.Int8, tla.AddressSpace.ub, 256)
        payload_layout = tla.make_layout(tla.make_shape(64), tla.make_stride(1))
        scale_layout = tla.make_layout(tla.make_shape(2), tla.make_stride(1))
        src = tla.make_tensor(src_ptr, payload_layout)
        dst = tla.make_tensor(tla.recast_ptr(data_ptr, dtype=tla.Float8E4M3FN), payload_layout)
        scale = tla.make_tensor(data_ptr, scale_layout)
        with tla.vector():
            tla.experimental.mx_to(src, (dst, scale))

    with pytest.raises(tla.TlaCoreAPIError, match="payload and scale storage must not overlap"):
        k.dump_mlir()


def test_mx_to_dynamic_blocked_layout_with_unambiguous_axis() -> None:
    @tla.kernel
    def k(cols: tla.Int32):
        a = tla.allocate(8192, tla.Float32, tla.AddressSpace.ub, 256)
        b = tla.allocate(8192, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        s = tla.allocate(8192, tla.Float8E8M0, tla.AddressSpace.ub, 256)
        src = tla.make_tensor(a, tla.make_layout_with_tag((2, cols), tla.Float32, tla.arch.zN))
        dst = tla.make_tensor(b, tla.make_layout_with_tag((2, cols), tla.Float8E4M3FN, tla.arch.nZ))
        scale = tla.make_tensor(s, tla.make_layout_with_tag((2, (cols + 31) // 32), tla.Float8E8M0, tla.arch.zZMxScale))
        with tla.vector():
            tla.experimental.mx_to(src, (dst, scale))

    assert "tla_mx_layout_quant_f32_e4m3" in str(k.dump_mlir(type_args=(tla.Int32(65),)))


def test_mx_to_rejects_runtime_shape_with_unknown_grouping_axis() -> None:
    @tla.kernel
    def k(cols: tla.Int32, groups: tla.Int32):
        a = tla.allocate(32768, tla.Float32, tla.AddressSpace.ub, 256)
        b = tla.allocate(32768, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        s = tla.allocate(8192, tla.Int8, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(64, cols), tla.make_stride(cols, 1))
        scale_layout = tla.make_layout(tla.make_shape(64, groups), tla.make_stride(groups, 1))
        with tla.vector():
            tla.experimental.mx_to(tla.make_tensor(a, layout),
                                   (tla.make_tensor(b, layout), tla.make_tensor(s, scale_layout)))

    with pytest.raises(tla.TlaCoreAPIError, match="cannot be determined unambiguously"):
        k.dump_mlir(type_args=(tla.Int32(64), tla.Int32(2)))


def test_experimental_mx_to_covers_dense_mx_and_mx_dense_routes() -> None:
    """The tensor API owns loads, conversion, scale pairing and stores."""

    @tla.kernel
    def k(
        src: tla.Tensor,
        mx_payload: tla.Tensor,
        mx_scale: tla.Tensor,
        dense_dst: tla.Tensor,
        mx_dst: tla.Tensor,
        out_scale: tla.Tensor,
    ) -> None:
        with tla.vector():
            mx_input = (mx_payload, mx_scale)
            mx_output = (mx_dst, out_scale)
            tla.experimental.mx_to(src, mx_input)
            tla.experimental.mx_to(mx_input, dense_dst)
            tla.experimental.mx_to(mx_input, mx_output)

    text = str(
        k.dump_mlir(
            type_args=(
                _ub(tla.Float32, 64),
                _ub(tla.Float8E4M3FN, 64),
                _ub(tla.Int8, 2),
                _ub(tla.Float32, 64),
                _ub(tla.Float8E4M3FN, 64),
                _ub(tla.Int8, 2),
            )
        )
    )
    assert text.count("tla.call_extern") == 3
    assert "tla_mx_tensor_quant_f32_e4m3" in text
    assert "tla_mx_tensor_dequant_e4m3_f32" in text
    assert "tla_mx_tensor_requant_e4m3_e4m3" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_rejects_dense_to_dense_casts() -> None:
    @tla.kernel
    def k(src: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to(src, dst)

    with pytest.raises(tla.TlaCoreAPIError, match="at least one of src or dst"):
        k.dump_mlir(type_args=(_ub(tla.Float32, 64), _ub(tla.Float16, 64)))


def test_dense_to_a_bare_fp8_stays_a_plain_cast() -> None:
    """``.to(Float8E4M3FN)`` must not become a microscaling conversion.

    """

    @tla.kernel
    def k(src: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst.store(src.load().to(tla.Float8E4M3FN, tla.params.CastParams()))

    text = str(k.dump_mlir(type_args=(_ub(tla.Float32, 64), _ub(tla.Float8E4M3FN, 64))))
    assert "tla.cast" in text


@pytest.mark.parametrize(
    ("payload_dtype", "scale_dtype", "payload_size", "scale_size", "error"),
    (
        (tla.Float32, tla.Int8, 64, 2, "unsupported dtype"),
        (tla.Float8E4M3FN, tla.Int16, 64, 2, "scale must be"),
        (tla.Float8E4M3FN, tla.Int8, 64, 1, "scale shape"),
        (tla.Float8E4M3FN, tla.Int8, 32, 1, "payload shapes must match"),
    ),
)
def test_experimental_mx_to_rejects_invalid_tensor_pairs(
    payload_dtype, scale_dtype, payload_size, scale_size, error
) -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to(src, (payload, scale))

    with pytest.raises(tla.TlaCoreAPIError, match=error):
        k.dump_mlir(type_args=(_ub(tla.Float32, 64), _ub(payload_dtype, payload_size), _ub(scale_dtype, scale_size)))


@pytest.mark.parametrize("dense_dtype", (tla.Float32, tla.Float16, tla.BFloat16))
@pytest.mark.parametrize("mx_dtype", (tla.Float4E2M1, tla.Float4E1M2, tla.Float8E4M3FN, tla.Float8E5M2))
@pytest.mark.parametrize("to_mx", (True, False))
def test_experimental_mx_to_dispatches_dense_pairs_to_bitcode(dense_dtype, mx_dtype, to_mx) -> None:
    @tla.kernel
    def k(dense: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            if to_mx:
                tla.experimental.mx_to(dense, (payload, scale))
            else:
                tla.experimental.mx_to((payload, scale), dense)

    text = str(k.dump_mlir(type_args=(_ub(dense_dtype, 64), _ub(mx_dtype, 64), _ub(tla.Int8, 2))))
    assert "tla.call_extern @tla_mx_tensor_" in text
    assert "tla.vec.func" not in text


@pytest.mark.parametrize("src_dtype", (tla.Float4E2M1, tla.Float4E1M2, tla.Float8E4M3FN, tla.Float8E5M2))
@pytest.mark.parametrize("dst_dtype", (tla.Float4E2M1, tla.Float4E1M2, tla.Float8E4M3FN, tla.Float8E5M2))
def test_experimental_mx_to_dispatches_mx_pairs_to_bitcode(src_dtype, dst_dtype) -> None:
    @tla.kernel
    def k(src: tla.Tensor, src_scale: tla.Tensor, dst: tla.Tensor, dst_scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to((src, src_scale), (dst, dst_scale))

    text = str(k.dump_mlir(type_args=(_ub(src_dtype, 64), _ub(tla.Int8, 2), _ub(dst_dtype, 64), _ub(tla.Int8, 2))))
    assert "tla.call_extern @tla_mx_tensor_" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_rejects_large_masked_tile_until_supported() -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to(src, (payload, scale), mask=tla.mask.M3)

    with pytest.raises(tla.TlaCoreAPIError, match="unsupported mask"):
        k.dump_mlir(type_args=(_ub(tla.Float32, 128), _ub(tla.Float8E4M3FN, 128), _ub(tla.Int8, 4)))


def test_experimental_mx_to_rejects_vec_func() -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simd"):
                tla.experimental.mx_to(src, (payload, scale))

    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        k.dump_mlir(type_args=(_ub(tla.Float32, 64), _ub(tla.Float8E4M3FN, 64), _ub(tla.Int8, 2)))


def test_experimental_mx_to_rejects_small_unsupported_mask() -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to(src, (payload, scale), mask=tla.mask.M3)

    with pytest.raises(tla.TlaCoreAPIError, match="unsupported mask"):
        k.dump_mlir(type_args=(_ub(tla.Float32, 64), _ub(tla.Float8E4M3FN, 64), _ub(tla.Int8, 2)))


def test_experimental_mx_to_full_mask_uses_whole_tile_bitcode() -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to(src, (payload, scale), mask=tla.mask.ALL)

    text = str(k.dump_mlir(type_args=(_ub(tla.Float32, 128), _ub(tla.Float8E4M3FN, 128), _ub(tla.Int8, 4))))
    assert "tla.call_extern @tla_mx_tensor_quant_f32_e4m3" in text
    assert "tla.vec.func" not in text


@pytest.mark.parametrize("dense_dtype", (tla.Float32, tla.Float16, tla.BFloat16))
@pytest.mark.parametrize("mx_dtype", (tla.Float4E2M1, tla.Float4E1M2, tla.Float8E4M3FN, tla.Float8E5M2))
def test_experimental_mx_to_vl16_quantizes_with_bitcode(dense_dtype, mx_dtype) -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to(src, (payload, scale), mask=tla.mask.VL16)

    text = str(k.dump_mlir(type_args=(_ub(dense_dtype, 128), _ub(mx_dtype, 128), _ub(tla.Int8, 4))))
    assert "tla.call_extern @tla_mx_tensor_quant_" in text
    assert "_masked" in text
    assert "tla.vec.func" not in text


@pytest.mark.parametrize("pattern", ("ALLF", "VL1", "VL2", "VL3", "VL4", "VL8", "VL16", "VL32", "VL64", "VL128"))
def test_experimental_mx_to_prefix_mask_uses_bitcode(pattern) -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to(src, (payload, scale), mask=pattern)

    text = str(k.dump_mlir(type_args=(_ub(tla.Float32, 128), _ub(tla.Float8E4M3FN, 128), _ub(tla.Int8, 4))))
    assert "tla.call_extern @tla_mx_tensor_quant_f32_e4m3_masked" in text


@pytest.mark.parametrize("dense_dtype", (tla.Float32, tla.Float16, tla.BFloat16))
@pytest.mark.parametrize("mx_dtype", (tla.Float4E2M1, tla.Float4E1M2, tla.Float8E4M3FN, tla.Float8E5M2))
def test_experimental_mx_to_prefix_dequantizes_with_bitcode(dense_dtype, mx_dtype) -> None:
    @tla.kernel
    def k(payload: tla.Tensor, scale: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to((payload, scale), dst, mask=tla.mask.VL16)

    text = str(k.dump_mlir(type_args=(_ub(mx_dtype, 128), _ub(tla.Int8, 4), _ub(dense_dtype, 128))))
    assert "tla.call_extern @tla_mx_tensor_dequant_" in text
    assert "_masked" in text
    assert "tla.vec.func" not in text


@pytest.mark.parametrize("src_dtype", (tla.Float4E2M1, tla.Float4E1M2, tla.Float8E4M3FN, tla.Float8E5M2))
@pytest.mark.parametrize("dst_dtype", (tla.Float4E2M1, tla.Float4E1M2, tla.Float8E4M3FN, tla.Float8E5M2))
def test_experimental_mx_to_prefix_requantizes_with_bitcode(src_dtype, dst_dtype) -> None:
    @tla.kernel
    def k(src: tla.Tensor, src_scale: tla.Tensor, dst: tla.Tensor, dst_scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to((src, src_scale), (dst, dst_scale), mask=tla.mask.VL16)

    text = str(k.dump_mlir(type_args=(_ub(src_dtype, 128), _ub(tla.Int8, 4), _ub(dst_dtype, 128), _ub(tla.Int8, 4))))
    assert "tla.call_extern @tla_mx_tensor_" in text
    assert "_masked" in text
    assert "tla.vec.func" not in text



def test_experimental_mx_to_exposes_rounding_without_register_slots() -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to(src, (payload, scale), round_mode=tla.params.RoundMode.CAST_FLOOR)

    text = str(k.dump_mlir(type_args=(_ub(tla.Float32, 64), _ub(tla.Float8E4M3FN, 64), _ub(tla.Int8, 2))))
    assert "tla.call_extern @tla_mx_tensor_quant_f32_e4m3" in text
    assert "arith.constant 1 : i32" in text


def test_experimental_mx_to_rejects_unsupported_fp4_rounding() -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to(src, (payload, scale), round_mode=tla.params.RoundMode.CAST_FLOOR)

    with pytest.raises(tla.TlaCoreAPIError, match="CAST_ROUND"):
        k.dump_mlir(type_args=(_ub(tla.Float32, 64), _ub(tla.Float4E2M1, 64), _ub(tla.Int8, 2)))


def test_experimental_mx_to_accepts_odd_length_packed_source() -> None:
    @tla.kernel
    def k(payload: tla.Tensor, scale: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to((payload, scale), dst)

    text = str(k.dump_mlir(type_args=(_ub(tla.Float4E2M1, 33), _ub(tla.Int8, 2), _ub(tla.Float32, 33))))
    assert "tla.call_extern @tla_mx_tensor_dequant_e2m1_f32" in text


def test_experimental_mx_to_accepts_odd_length_packed_destination() -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            tla.experimental.mx_to(src, (payload, scale))

    text = str(k.dump_mlir(type_args=(_ub(tla.Float32, 33), _ub(tla.Float4E2M1, 33), _ub(tla.Int8, 2))))
    assert "tla.call_extern @tla_mx_tensor_quant_f32_e2m1" in text


def test_experimental_mx_to_accepts_contiguous_first_axis() -> None:
    @tla.kernel
    def k() -> None:
        dense_ptr = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
        mx_ptr = tla.allocate(128, tla.Float4E2M1, tla.AddressSpace.ub, 256)
        scale_ptr = tla.allocate(4, tla.Int8, tla.AddressSpace.ub, 256)
        payload_layout = tla.make_layout(tla.make_shape(64, 2), tla.make_stride(1, 64), layoutTag=tla.arch.ColumnMajor)
        scale_layout = tla.make_layout(tla.make_shape(2, 2), tla.make_stride(1, 2), layoutTag=tla.arch.ColumnMajor)
        dense = tla.make_tensor(dense_ptr, payload_layout)
        payload = tla.make_tensor(mx_ptr, payload_layout)
        scale = tla.make_tensor(scale_ptr, scale_layout)
        with tla.vector():
            tla.experimental.mx_to(dense, (payload, scale))

    text = str(k.dump_mlir())
    assert "tla.call_extern @tla_mx_tensor_quant_f32_e2m1" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_gathers_different_payload_axes() -> None:
    @tla.kernel
    def k() -> None:
        dense_ptr = tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 256)
        mx_ptr = tla.allocate(128, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        scale_ptr = tla.allocate(64, tla.Int8, tla.AddressSpace.ub, 256)
        src_layout = tla.make_layout(tla.make_shape(64, 2), tla.make_stride(1, 64), layoutTag=tla.arch.ColumnMajor)
        dst_layout = tla.make_layout(tla.make_shape(64, 2), tla.make_stride(2, 1))
        scale_layout = tla.make_layout(tla.make_shape(64, 1), tla.make_stride(1, 1))
        dense = tla.make_tensor(dense_ptr, src_layout)
        payload = tla.make_tensor(mx_ptr, dst_layout)
        scale = tla.make_tensor(scale_ptr, scale_layout)
        with tla.vector():
            tla.experimental.mx_to(dense, (payload, scale))

    text = str(k.dump_mlir())
    assert "tla.call_extern @tla_mx_gather_quant_f32_e4m3" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_scatters_different_payload_axes() -> None:
    @tla.kernel
    def k() -> None:
        mx_ptr = tla.allocate(128, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        scale_ptr = tla.allocate(64, tla.Int8, tla.AddressSpace.ub, 256)
        dense_ptr = tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 256)
        mx_layout = tla.make_layout(tla.make_shape(64, 2), tla.make_stride(2, 1))
        scale_layout = tla.make_layout(tla.make_shape(64, 1), tla.make_stride(1, 1))
        dst_layout = tla.make_layout(tla.make_shape(64, 2), tla.make_stride(1, 64), layoutTag=tla.arch.ColumnMajor)
        payload = tla.make_tensor(mx_ptr, mx_layout)
        scale = tla.make_tensor(scale_ptr, scale_layout)
        dense = tla.make_tensor(dense_ptr, dst_layout)
        with tla.vector():
            tla.experimental.mx_to((payload, scale), dense)

    text = str(k.dump_mlir())
    assert "tla.call_extern @tla_mx_scatter_dequant_e4m3_f32" in text
    assert "tla.vec.func" not in text


@pytest.mark.parametrize(
    ("src_type", "symbol"),
    [(tla.Float8E4M3FN, "tla_mx_gather_requant_e4m3_e5m2"), (tla.Float4E2M1, "tla_mx_gather_fp4_e2m1_e5m2")],
)
def test_experimental_mx_to_gathers_strided_mx_source(src_type, symbol) -> None:
    @tla.kernel
    def k() -> None:
        src_ptr = tla.allocate(128, src_type, tla.AddressSpace.ub, 256)
        dst_ptr = tla.allocate(128, tla.Float8E5M2, tla.AddressSpace.ub, 256)
        src_scale_ptr = tla.allocate(64, tla.Int8, tla.AddressSpace.ub, 256)
        dst_scale_ptr = tla.allocate(64, tla.Int8, tla.AddressSpace.ub, 256)
        src_layout = tla.make_layout(tla.make_shape(64, 2), tla.make_stride(1, 64), layoutTag=tla.arch.ColumnMajor)
        dst_layout = tla.make_layout(tla.make_shape(64, 2), tla.make_stride(2, 1))
        scale_layout = tla.make_layout(tla.make_shape(64, 1), tla.make_stride(1, 1))
        src = tla.make_tensor(src_ptr, src_layout)
        dst = tla.make_tensor(dst_ptr, dst_layout)
        src_scale = tla.make_tensor(src_scale_ptr, scale_layout)
        dst_scale = tla.make_tensor(dst_scale_ptr, scale_layout)
        with tla.vector():
            tla.experimental.mx_to((src, src_scale), (dst, dst_scale))

    text = str(k.dump_mlir())
    assert f"tla.call_extern @{symbol}" in text
    assert "tla.vec.func" not in text


@pytest.mark.parametrize(
    ("dst_type", "symbol"),
    [(tla.Float8E5M2, "tla_mx_scatter_mx_e4m3_e5m2"), (tla.Float4E2M1, "tla_mx_scatter_fp4_e4m3_e2m1")],
)
def test_experimental_mx_to_scatters_strided_mx_destination(dst_type, symbol) -> None:
    @tla.kernel
    def k() -> None:
        src_ptr = tla.allocate(128, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        dst_ptr = tla.allocate(128, dst_type, tla.AddressSpace.ub, 256)
        src_scale_ptr = tla.allocate(64, tla.Int8, tla.AddressSpace.ub, 256)
        dst_scale_ptr = tla.allocate(64, tla.Int8, tla.AddressSpace.ub, 256)
        src_layout = tla.make_layout(tla.make_shape(2, 64), tla.make_stride(64, 1))
        dst_layout = tla.make_layout(tla.make_shape(2, 64), tla.make_stride(1, 2), layoutTag=tla.arch.ColumnMajor)
        scale_layout = tla.make_layout(tla.make_shape(2, 2), tla.make_stride(2, 1))
        src = tla.make_tensor(src_ptr, src_layout)
        dst = tla.make_tensor(dst_ptr, dst_layout)
        src_scale = tla.make_tensor(src_scale_ptr, scale_layout)
        dst_scale = tla.make_tensor(dst_scale_ptr, scale_layout)
        with tla.vector():
            tla.experimental.mx_to((src, src_scale), (dst, dst_scale))

    text = str(k.dump_mlir())
    assert f"tla.call_extern @{symbol}" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_gathers_and_scatters_mx() -> None:
    @tla.kernel
    def k() -> None:
        src_ptr = tla.allocate(128, tla.Float4E2M1, tla.AddressSpace.ub, 256)
        dst_ptr = tla.allocate(128, tla.Float4E1M2, tla.AddressSpace.ub, 256)
        src_scale_ptr = tla.allocate(4, tla.Int8, tla.AddressSpace.ub, 256)
        dst_scale_ptr = tla.allocate(4, tla.Int8, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(2, 64), tla.make_stride(1, 2), layoutTag=tla.arch.ColumnMajor)
        scale_layout = tla.make_layout(tla.make_shape(2, 2), tla.make_stride(2, 1))
        src = tla.make_tensor(src_ptr, layout)
        dst = tla.make_tensor(dst_ptr, layout)
        src_scale = tla.make_tensor(src_scale_ptr, scale_layout)
        dst_scale = tla.make_tensor(dst_scale_ptr, scale_layout)
        with tla.vector():
            tla.experimental.mx_to((src, src_scale), (dst, dst_scale))

    text = str(k.dump_mlir())
    assert "tla.call_extern @tla_mx_strided_e2m1_e1m2" in text
    assert "tla.vec.func" not in text


@pytest.mark.parametrize("to_mx", [True, False])
def test_experimental_mx_to_gathers_and_scatters_dense(to_mx) -> None:
    @tla.kernel
    def k() -> None:
        dense_ptr = tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 256)
        mx_ptr = tla.allocate(128, tla.Float4E2M1, tla.AddressSpace.ub, 256)
        scale_ptr = tla.allocate(4, tla.Int8, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(2, 64), tla.make_stride(1, 2), layoutTag=tla.arch.ColumnMajor)
        scale_layout = tla.make_layout(tla.make_shape(2, 2), tla.make_stride(2, 1))
        dense = tla.make_tensor(dense_ptr, layout)
        payload = tla.make_tensor(mx_ptr, layout)
        scale = tla.make_tensor(scale_ptr, scale_layout)
        with tla.vector():
            if tla.const_expr(to_mx):
                tla.experimental.mx_to(dense, (payload, scale))
            else:
                tla.experimental.mx_to((payload, scale), dense)

    text = str(k.dump_mlir())
    symbol = "quant_f32_e2m1" if to_mx else "dequant_e2m1_f32"
    assert f"tla.call_extern @tla_mx_strided_{symbol}" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_regroups_different_scale_axes() -> None:
    @tla.kernel
    def k() -> None:
        src_ptr = tla.allocate(128, tla.Float4E2M1, tla.AddressSpace.ub, 256)
        dst_ptr = tla.allocate(128, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        src_scale_ptr = tla.allocate(64, tla.Int8, tla.AddressSpace.ub, 256)
        dst_scale_ptr = tla.allocate(4, tla.Int8, tla.AddressSpace.ub, 256)
        src_layout = tla.make_layout(tla.make_shape(2, 64), tla.make_stride(1, 2), layoutTag=tla.arch.ColumnMajor)
        dst_layout = tla.make_layout(tla.make_shape(2, 64), tla.make_stride(64, 1))
        src_scale_layout = tla.make_layout(tla.make_shape(1, 64), tla.make_stride(64, 1))
        dst_scale_layout = tla.make_layout(tla.make_shape(2, 2), tla.make_stride(2, 1))
        src = tla.make_tensor(src_ptr, src_layout)
        dst = tla.make_tensor(dst_ptr, dst_layout)
        src_scale = tla.make_tensor(src_scale_ptr, src_scale_layout)
        dst_scale = tla.make_tensor(dst_scale_ptr, dst_scale_layout)
        with tla.vector():
            tla.experimental.mx_to((src, src_scale), (dst, dst_scale))

    text = str(k.dump_mlir())
    assert "tla.call_extern @tla_mx_regroup_e2m1_e4m3" in text
    assert "tla.vec.func" not in text


@pytest.mark.parametrize("kind", ["quant", "dequant", "requant"])
def test_experimental_mx_to_accepts_fractal_layouts(kind) -> None:
    @tla.kernel
    def k() -> None:
        dense_ptr = tla.allocate(4096, tla.Float32, tla.AddressSpace.ub, 256)
        src_ptr = tla.allocate(4096, tla.Float4E2M1, tla.AddressSpace.ub, 256)
        dst_ptr = tla.allocate(4096, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        src_scale_ptr = tla.allocate(4096, tla.Float8E8M0, tla.AddressSpace.ub, 256)
        dst_scale_ptr = tla.allocate(4096, tla.Float8E8M0, tla.AddressSpace.ub, 256)
        dense = tla.make_tensor(dense_ptr, tla.make_layout_with_tag((2, 65), tla.Float32, tla.arch.zN))
        src = tla.make_tensor(src_ptr, tla.make_layout_with_tag((2, 65), tla.Float4E2M1, tla.arch.zN))
        dst = tla.make_tensor(dst_ptr, tla.make_layout_with_tag((2, 65), tla.Float8E4M3FN, tla.arch.nZ))
        src_scale = tla.make_tensor(src_scale_ptr, tla.make_layout_with_tag((1, 65), tla.Float8E8M0, tla.arch.zZMxScale))
        dst_scale = tla.make_tensor(dst_scale_ptr, tla.make_layout_with_tag((2, 3), tla.Float8E8M0, tla.arch.zZMxScale))
        with tla.vector():
            if tla.const_expr(kind == "quant"):
                tla.experimental.mx_to(dense, (dst, dst_scale))
            elif tla.const_expr(kind == "dequant"):
                tla.experimental.mx_to((src, src_scale), dense)
            else:
                tla.experimental.mx_to((src, src_scale), (dst, dst_scale))

    text = str(k.dump_mlir())
    assert f"tla.call_extern @tla_mx_layout_{kind}_" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_accepts_strided_scale_groups() -> None:
    @tla.kernel
    def k() -> None:
        dense_ptr = tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 256)
        mx_ptr = tla.allocate(128, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        scale_ptr = tla.allocate(16, tla.Int8, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(2, 64), tla.make_stride(64, 1))
        scale_layout = tla.make_layout(tla.make_shape(2, 2), tla.make_stride(1, 2), layoutTag=tla.arch.ColumnMajor)
        dense = tla.make_tensor(dense_ptr, layout)
        payload = tla.make_tensor(mx_ptr, layout)
        scale = tla.make_tensor(scale_ptr, scale_layout)
        with tla.vector():
            tla.experimental.mx_to(dense, (payload, scale))

    text = str(k.dump_mlir())
    assert "tla.call_extern @tla_mx_scale_stride_quant_f32_e4m3" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_accepts_rank_one_dynamic_stride() -> None:
    @tla.kernel
    def k(stride: tla.Int32) -> None:
        dense_ptr = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
        mx_ptr = tla.allocate(256, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        scale_ptr = tla.allocate(8, tla.Float8E8M0, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(65), tla.make_stride(stride))
        scale_layout = tla.make_layout(tla.make_shape(3), tla.make_stride(1))
        dense = tla.make_tensor(dense_ptr, layout)
        payload = tla.make_tensor(mx_ptr, layout)
        scale = tla.make_tensor(scale_ptr, scale_layout)
        with tla.vector():
            tla.experimental.mx_to(dense, (payload, scale))

    text = str(k.dump_mlir(type_args=(tla.Int32(2),)))
    assert "tla.call_extern @tla_mx_layout_quant_f32_e4m3" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_accepts_dynamic_outer_coordinate() -> None:
    @tla.kernel
    def k(row: tla.Int32) -> None:
        dense_ptr = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
        mx_ptr = tla.allocate(256, tla.Float4E2M1, tla.AddressSpace.ub, 256)
        scale_ptr = tla.allocate(8, tla.Int8, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(4, 64), tla.make_stride(64, 1))
        scale_layout = tla.make_layout(tla.make_shape(4, 2), tla.make_stride(2, 1))
        dense = tla.make_tensor(dense_ptr, layout)
        payload = tla.make_tensor(mx_ptr, layout)
        scale = tla.make_tensor(scale_ptr, scale_layout)
        with tla.vector():
            dense_tile = tla.tile_view(dense, tla.make_shape(1, 64), tla.make_coord(row, 0))
            mx_tile = tla.tile_view(payload, tla.make_shape(1, 64), tla.make_coord(row, 0))
            scale_tile = tla.tile_view(scale, tla.make_shape(1, 2), tla.make_coord(row, 0))
            tla.experimental.mx_to(dense_tile, (mx_tile, scale_tile))

    text = str(k.dump_mlir(type_args=(tla.Int32(0),)))
    assert "tla.call_extern @tla_mx_tensor_quant_f32_e2m1" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_accepts_dynamic_group_coordinate() -> None:
    @tla.kernel
    def k(group: tla.Int32) -> None:
        dense_ptr = tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 256)
        mx_ptr = tla.allocate(128, tla.Float4E2M1, tla.AddressSpace.ub, 256)
        scale_ptr = tla.allocate(4, tla.Int8, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(2, 64), tla.make_stride(64, 1))
        scale_layout = tla.make_layout(tla.make_shape(2, 2), tla.make_stride(2, 1))
        dense = tla.make_tensor(dense_ptr, layout)
        payload = tla.make_tensor(mx_ptr, layout)
        scale = tla.make_tensor(scale_ptr, scale_layout)
        with tla.vector():
            dense_tile = tla.tile_view(dense, tla.make_shape(2, 32), tla.make_coord(0, group))
            mx_tile = tla.tile_view(payload, tla.make_shape(2, 32), tla.make_coord(0, group))
            scale_tile = tla.tile_view(scale, tla.make_shape(2, 1), tla.make_coord(0, group))
            tla.experimental.mx_to(dense_tile, (mx_tile, scale_tile))

    text = str(k.dump_mlir(type_args=(tla.Int32(0),)))
    assert "tla.call_extern @tla_mx_tensor_quant_f32_e2m1" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_accepts_dynamic_tile_shape() -> None:
    @tla.kernel
    def k(cols: tla.Int32) -> None:
        dense_ptr = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
        mx_ptr = tla.allocate(256, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        scale_ptr = tla.allocate(8, tla.Int8, tla.AddressSpace.ub, 256)
        layout = tla.make_layout(tla.make_shape(2, cols), tla.make_stride(cols, 1))
        scale_cols = (cols + 31) // 32
        scale_layout = tla.make_layout(tla.make_shape(2, scale_cols), tla.make_stride(scale_cols, 1))
        dense = tla.make_tensor(dense_ptr, layout)
        payload = tla.make_tensor(mx_ptr, layout)
        scale = tla.make_tensor(scale_ptr, scale_layout)
        with tla.vector():
            tla.experimental.mx_to(dense, (payload, scale))

    text = str(k.dump_mlir(type_args=(tla.Int32(64),)))
    assert "tla.call_extern @tla_mx_tensor_quant_f32_e4m3" in text
    assert "tla.vec.func" not in text


def test_experimental_mx_to_rejects_known_mismatch_with_dynamic_shape() -> None:
    @tla.kernel
    def k(cols: tla.Int32) -> None:
        dense_ptr = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
        mx_ptr = tla.allocate(256, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
        scale_ptr = tla.allocate(8, tla.Int8, tla.AddressSpace.ub, 256)
        dense = tla.make_tensor(dense_ptr, tla.make_layout(tla.make_shape(2, cols), tla.make_stride(cols, 1)))
        payload = tla.make_tensor(mx_ptr, tla.make_layout(tla.make_shape(3, cols), tla.make_stride(cols, 1)))
        scale_cols = (cols + 31) // 32
        scale = tla.make_tensor(scale_ptr, tla.make_layout(tla.make_shape(3, scale_cols), tla.make_stride(scale_cols, 1)))
        with tla.vector():
            tla.experimental.mx_to(dense, (payload, scale))

    with pytest.raises(tla.TlaCoreAPIError, match="payload shapes must match"):
        k.dump_mlir(type_args=(tla.Int32(64),))


def test_mx_to_returns_the_same_destination_pair() -> None:
    @tla.kernel
    def k(src: tla.Tensor, payload: tla.Tensor, scale: tla.Tensor) -> None:
        with tla.vector():
            dst = (payload, scale)
            assert tla.experimental.mx_to(src, dst) is dst

    text = str(k.dump_mlir(type_args=(
        _ub(tla.Float32, 64),
        _ub(tla.Float8E4M3FN, 64),
        _ub(tla.Int8, 2),
    )))
    assert "tla.call_extern @tla_mx_tensor_" in text


def test_mx_to_returns_the_same_dense_destination() -> None:
    @tla.kernel
    def k(payload: tla.Tensor, scale: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            assert tla.experimental.mx_to((payload, scale), dst) is dst

    text = str(k.dump_mlir(type_args=(
        _ub(tla.Float8E4M3FN, 64),
        _ub(tla.Int8, 2),
        _ub(tla.Float32, 64),
    )))
    assert "tla.call_extern @tla_mx_tensor_" in text
