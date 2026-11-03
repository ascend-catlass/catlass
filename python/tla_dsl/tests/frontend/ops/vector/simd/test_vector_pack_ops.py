from __future__ import annotations

import pytest

from catlass import tla
from catlass.core_api import MaskSSA, VectorSSA
from catlass.tla.runtime import make_fake_tensor


_PACK_VECTOR_PAIRS = [
    (tla.Int16, tla.UInt8),
    (tla.UInt16, tla.UInt8),
    (tla.Int32, tla.UInt16),
    (tla.UInt32, tla.UInt16),
]
_UNPACK_PAIRS = [
    (tla.Int8, tla.Int16),
    (tla.UInt8, tla.UInt16),
    (tla.Int16, tla.Int32),
    (tla.UInt16, tla.UInt32),
]
_SIGNED_DTYPE = {8: tla.Int8, 16: tla.Int16, 32: tla.Int32, 64: tla.Int64}
_PACK_CASES = [
    ("mask", tla.Int32, tla.Int16),
    ("mask", tla.Int16, tla.Int8),
    *[("vector", src, dst) for src, dst in _PACK_VECTOR_PAIRS],
]


@tla.kernel
def _pack_emit(
    src_kind: tla.Constexpr[str],
    src_dtype: tla.Constexpr[type],
    dst_dtype: tla.Constexpr[type],
    storage_dtype: tla.Constexpr[type],
    part: tla.Constexpr[object],
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            if tla.const_expr(src_kind == "mask"):
                source = tla.create_mask(pattern=tla.mask.H, dtype=src_dtype)
            else:
                source = tla.full(-1, storage_dtype).bitcast(src_dtype)
            packed = tla.pack(source, part=part)
            if tla.const_expr(src_kind == "mask"):
                assert isinstance(packed, MaskSSA)
                tla.where(packed, tla.full(1, dst_dtype), tla.full(0, dst_dtype))
            else:
                assert isinstance(packed, VectorSSA)


@pytest.mark.parametrize(("src_kind", "src_dtype", "dst_dtype"), _PACK_CASES)
@pytest.mark.parametrize("part", [tla.RegPart.LOW, tla.RegPart.HIGH])
def test_pack_emits_mlir(src_kind, src_dtype, dst_dtype, part) -> None:
    src_bits = int(src_dtype.dtype[1:])
    dst_bits = int(dst_dtype.dtype[1:])
    mlir = _pack_emit.dump_mlir(
        type_args=(src_kind, src_dtype, dst_dtype, _SIGNED_DTYPE[src_bits], part)
    )
    (line,) = [line for line in mlir.splitlines() if "tla.pack" in line]
    assert f"<{part.value}>" in line
    assert "dtype" not in line
    if src_kind == "mask":
        assert f"!tla.mask<{2048 // src_bits}>" in line
        assert f"!tla.mask<{2048 // dst_bits}>" in line
        assert "tla.where" in mlir
    else:
        src_type = f"ui{src_bits}" if src_dtype.dtype.startswith("u") else f"i{src_bits}"
        assert f"!tla.vector<{2048 // src_bits}x{src_type}>" in line
        assert f"!tla.vector<{2048 // dst_bits}xui{dst_bits}>" in line


@tla.kernel
def _unpack_emit(
    src_dtype: tla.Constexpr[type],
    dst_dtype: tla.Constexpr[type],
    storage_dtype: tla.Constexpr[type],
    part: tla.Constexpr[object],
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source = tla.full(-1, storage_dtype).bitcast(src_dtype)
            result = tla.unpack(source, part=part)
            assert isinstance(result, VectorSSA)


@pytest.mark.parametrize(("src_dtype", "dst_dtype"), _UNPACK_PAIRS)
@pytest.mark.parametrize("part", [tla.RegPart.LOW, tla.RegPart.HIGH])
def test_unpack_emits_mlir(src_dtype, dst_dtype, part) -> None:
    src_bits = int(src_dtype.dtype[1:])
    dst_bits = int(dst_dtype.dtype[1:])
    mlir = _unpack_emit.dump_mlir(
        type_args=(src_dtype, dst_dtype, _SIGNED_DTYPE[src_bits], part)
    )
    (line,) = [line for line in mlir.splitlines() if "tla.unpack" in line]
    assert f"<{part.value}>" in line
    assert "dtype" not in line
    src_type = f"ui{src_bits}" if src_dtype.dtype.startswith("u") else f"i{src_bits}"
    dst_type = f"ui{dst_bits}" if dst_dtype.dtype.startswith("u") else f"i{dst_bits}"
    assert f"!tla.vector<{2048 // src_bits}x{src_type}>" in line
    assert f"!tla.vector<{2048 // dst_bits}x{dst_type}>" in line


@tla.kernel
def _mask_pack_compile_kernel(
    src: tla.Tensor,
    dst: tla.Tensor,
    value_dtype: tla.Constexpr[type],
    source_mask_dtype: tla.Constexpr[type],
    lanes: tla.Constexpr[int],
) -> None:
    loaded = tla.flag("mask_pack_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    computed = tla.flag("mask_pack_computed", tla.arch.VECTOR, tla.arch.MTE3)
    src_ptr = tla.allocate(lanes, value_dtype, tla.AddressSpace.ub, 256)
    dst_ptr = tla.allocate(lanes, value_dtype, tla.AddressSpace.ub, 256)
    ub_src = tla.make_tensor_like(src_ptr, src, tla.arch.RowMajor)
    ub_dst = tla.make_tensor_like(dst_ptr, dst, tla.arch.RowMajor)

    with tla.vector():  # noqa: SIM117 - DSL regions must remain syntactically nested.
        tla.copy(ub_src, src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            value = ub_src.load()
            zero = tla.sub(value, value)
            source = tla.create_mask(pattern=tla.mask.H, dtype=source_mask_dtype)
            packed = tla.pack(source, part=tla.RegPart.LOW)
            ub_dst.store(tla.where(packed, value, zero))
        tla.set_flag(computed)
        tla.wait_flag(computed)
        tla.copy(dst, ub_dst)
        tla.pipe_barrier(tla.pipes.ALL)


@pytest.mark.parametrize(
    ("value_dtype", "source_mask_dtype", "lanes"),
    [
        (tla.Int16, tla.Int32, 128),
        (tla.Int8, tla.Int16, 256),
    ],
    ids=["mask64-mask128", "mask128-mask256"],
)
def test_pack_mask_compiles_to_kernel_object(
    isolated_compile_cache, value_dtype, source_mask_dtype, lanes
) -> None:
    del isolated_compile_cache
    tensors = tuple(
        make_fake_tensor(
            value_dtype,
            (lanes,),
            (1,),
            origin_shape=(lanes,),
            layout_tag=tla.arch.RowMajor,
        )
        for _ in range(2)
    )

    compiled = tla.compile(
        _mask_pack_compile_kernel,
        *tensors,
        value_dtype,
        source_mask_dtype,
        lanes,
        options="--npu-arch 3510",
    )

    assert compiled.kernel_binary_path.is_file()
    assert compiled.kernel_binary_path.stat().st_size > 0


@tla.kernel
def _pack_with_invalid_part(part: tla.Constexpr[object]) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source = tla.create_mask(pattern=tla.mask.H, dtype=tla.Int32)
            tla.pack(source, part=part)


def test_pack_rejects_plain_integer_part() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"expected tla\.RegPart\.LOW or tla\.RegPart\.HIGH",
    ):
        _pack_with_invalid_part.dump_mlir(type_args=(0,))


@tla.kernel
def _pack_unsupported_mask(dtype: tla.Constexpr[type]) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype)
            tla.pack(mask, part=tla.RegPart.LOW)


@pytest.mark.parametrize(
    ("dtype", "width"),
    [(tla.Int64, 32), (tla.Int8, 256)],
)
def test_pack_rejects_unsupported_mask_widths(dtype, width) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=f"unsupported mask width {width}"):
        _pack_unsupported_mask.dump_mlir(type_args=(dtype,))


@tla.kernel
def _pack_in_simt() -> None:
    with tla.vector():
        with tla.vec.func(mode="simt"):
            source = tla.create_mask(pattern=tla.mask.H, dtype=tla.Int32)
            tla.pack(source, part=tla.RegPart.LOW)


def test_pack_rejects_simt_mode() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="only supported.*simd"):
        _pack_in_simt.dump_mlir()


@pytest.mark.parametrize("src_lanes", [64, 128])
@pytest.mark.parametrize("part", ["low", "high"])
def test_pack_verifier_accepts_lowered_func(src_lanes, part) -> None:
    import catlass.runtime as runtime
    from catlass._mlir import ir

    with runtime._eager_capture():
        module = ir.Module.parse(
            f"""
            module {{
              func.func @lowered(%src: !tla.mask<{src_lanes}>)
                  -> !tla.mask<{2 * src_lanes}> {{
                %result = tla.pack %src, <{part}>
                    : !tla.mask<{src_lanes}> -> !tla.mask<{2 * src_lanes}>
                return %result : !tla.mask<{2 * src_lanes}>
              }}
            }}
            """
        )
        assert module.operation.verify()


@tla.kernel
def _pack_compile_kernel(
    src: tla.Tensor,
    dst: tla.Tensor,
    src_dtype: tla.Constexpr[type],
    dst_dtype: tla.Constexpr[type],
    src_storage_dtype: tla.Constexpr[type],
    dst_storage_dtype: tla.Constexpr[type],
    src_lanes: tla.Constexpr[int],
    dst_lanes: tla.Constexpr[int],
    part: tla.Constexpr[object],
) -> None:
    loaded = tla.flag("pack_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    computed = tla.flag("pack_computed", tla.arch.VECTOR, tla.arch.MTE3)
    src_ptr = tla.allocate(src_lanes, src_storage_dtype, tla.AddressSpace.ub, 256)
    dst_ptr = tla.allocate(dst_lanes, dst_storage_dtype, tla.AddressSpace.ub, 256)
    ub_src = tla.make_tensor_like(src_ptr, src, tla.arch.RowMajor)
    ub_dst = tla.make_tensor_like(dst_ptr, dst, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(ub_src, src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            source = ub_src.load().bitcast(src_dtype)
            result = tla.pack(source, part=part)
            ub_dst.store(result.bitcast(dst_storage_dtype))
        tla.set_flag(computed)
        tla.wait_flag(computed)
        tla.copy(dst, ub_dst)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def _unpack_compile_kernel(
    src: tla.Tensor,
    dst: tla.Tensor,
    src_dtype: tla.Constexpr[type],
    dst_dtype: tla.Constexpr[type],
    src_storage_dtype: tla.Constexpr[type],
    dst_storage_dtype: tla.Constexpr[type],
    src_lanes: tla.Constexpr[int],
    dst_lanes: tla.Constexpr[int],
    part: tla.Constexpr[object],
) -> None:
    loaded = tla.flag("unpack_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    computed = tla.flag("unpack_computed", tla.arch.VECTOR, tla.arch.MTE3)
    src_ptr = tla.allocate(src_lanes, src_storage_dtype, tla.AddressSpace.ub, 256)
    dst_ptr = tla.allocate(dst_lanes, dst_storage_dtype, tla.AddressSpace.ub, 256)
    ub_src = tla.make_tensor_like(src_ptr, src, tla.arch.RowMajor)
    ub_dst = tla.make_tensor_like(dst_ptr, dst, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(ub_src, src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            source = ub_src.load().bitcast(src_dtype)
            result = tla.unpack(source, part=part)
            ub_dst.store(result.bitcast(dst_storage_dtype))
        tla.set_flag(computed)
        tla.wait_flag(computed)
        tla.copy(dst, ub_dst)
        tla.pipe_barrier(tla.pipes.ALL)


@pytest.fixture(scope="module")
def vector_pack_compile_cache(tmp_path_factory):
    from catlass.bc_compile import init

    # Existing aggregates are reused by init(). Compile the new stub in a
    # fresh directory, even when running from a dirty source checkout.
    cache_root = tmp_path_factory.mktemp("vector_pack_compile")
    bc_dir = init(cache_dir=cache_root / "bc")
    assert (bc_dir / "pack_reg.aiv.c310.bc").is_file()
    return cache_root


def _compile_inputs(src_dtype, dst_dtype):
    src_bits = int(src_dtype.dtype[1:])
    dst_bits = int(dst_dtype.dtype[1:])
    src_storage = _SIGNED_DTYPE[src_bits]
    dst_storage = _SIGNED_DTYPE[dst_bits]
    src_lanes = 2048 // src_bits
    dst_lanes = 2048 // dst_bits
    tensors = tuple(
        make_fake_tensor(
            dtype,
            (lanes,),
            (1,),
            origin_shape=(lanes,),
            layout_tag=tla.arch.RowMajor,
        )
        for dtype, lanes in [(src_storage, src_lanes), (dst_storage, dst_lanes)]
    )
    return (*tensors, src_dtype, dst_dtype, src_storage, dst_storage, src_lanes, dst_lanes)


@pytest.mark.parametrize(("src_dtype", "dst_dtype"), _PACK_VECTOR_PAIRS)
@pytest.mark.parametrize("part", [tla.RegPart.LOW, tla.RegPart.HIGH])
def test_pack_vector_compiles_to_kernel_object(
    vector_pack_compile_cache, monkeypatch, src_dtype, dst_dtype, part
) -> None:
    monkeypatch.setenv("CATLASS_DSL_CACHE_DIR", str(vector_pack_compile_cache))
    compiled = tla.compile(
        _pack_compile_kernel,
        *_compile_inputs(src_dtype, dst_dtype),
        part,
        options="--npu-arch 3510",
    )
    assert compiled.kernel_binary_path.is_file()
    assert compiled.kernel_binary_path.stat().st_size > 0


@pytest.mark.parametrize(("src_dtype", "dst_dtype"), _UNPACK_PAIRS)
@pytest.mark.parametrize("part", [tla.RegPart.LOW, tla.RegPart.HIGH])
def test_unpack_compiles_to_kernel_object(
    vector_pack_compile_cache, monkeypatch, src_dtype, dst_dtype, part
) -> None:
    monkeypatch.setenv("CATLASS_DSL_CACHE_DIR", str(vector_pack_compile_cache))
    compiled = tla.compile(
        _unpack_compile_kernel,
        *_compile_inputs(src_dtype, dst_dtype),
        part,
        options="--npu-arch 3510",
    )
    assert compiled.kernel_binary_path.is_file()
    assert compiled.kernel_binary_path.stat().st_size > 0


@pytest.mark.parametrize("src", [tla.Int8, tla.UInt8, tla.Int64, tla.UInt64, tla.Float32])
def test_pack_rejects_invalid_vector_source(src) -> None:
    # pack only derives a destination for 16/32-bit integer sources; 64-bit
    # sources are not supported (no native vpack/vunpack for that width).
    storage = src if src is tla.Float32 else _SIGNED_DTYPE[int(src.dtype[1:])]
    with pytest.raises(tla.TlaCoreAPIError, match="unsupported vector element type"):
        _pack_emit.dump_mlir(type_args=("vector", src, tla.UInt8, storage, tla.RegPart.LOW))


@pytest.mark.parametrize("src", [tla.Int32, tla.UInt32, tla.Int64, tla.UInt64, tla.Float32])
def test_unpack_rejects_invalid_vector_source(src) -> None:
    # unpack only derives a destination for 8/16-bit integer sources; a
    # 32-bit source would need a 64-bit destination, which is not supported.
    storage = src if src is tla.Float32 else _SIGNED_DTYPE[int(src.dtype[1:])]
    with pytest.raises(tla.TlaCoreAPIError, match="unsupported vector element type"):
        _unpack_emit.dump_mlir(type_args=(src, tla.Int32, storage, tla.RegPart.LOW))


@pytest.mark.parametrize("src_kind", ["mask", "vector"])
def test_pack_rejects_integer_part(src_kind) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=r"expected tla\.RegPart"):
        _pack_emit.dump_mlir(type_args=(src_kind, tla.Int16, tla.UInt8, tla.Int16, 0))


def test_unpack_rejects_integer_part() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=r"expected tla\.RegPart"):
        _unpack_emit.dump_mlir(type_args=(tla.Int8, tla.Int16, tla.Int8, 0))


@tla.kernel
def _pack_fragment() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source = tla.full(1, tla.Int16).reduce(
                tla.ReductionOp.ADD,
                mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16),
            )
            tla.pack(source, part=tla.RegPart.LOW)


def test_pack_rejects_vector_fragments() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="requires full 256-byte"):
        _pack_fragment.dump_mlir()


@tla.kernel
def _unpack_fragment() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source = tla.full(1, tla.Int16).reduce(
                tla.ReductionOp.ADD,
                mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16),
            )
            tla.unpack(source, part=tla.RegPart.LOW)


def test_unpack_rejects_fragments() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="requires full 256-byte"):
        _unpack_fragment.dump_mlir()


@tla.kernel
def _unpack_mask() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
            tla.unpack(source, part=tla.RegPart.LOW)


def test_unpack_rejects_mask() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="expected vector_ssa"):
        _unpack_mask.dump_mlir()


@pytest.mark.parametrize(
    ("op", "src_type", "dst_type"),
    [
        ("pack", "64xi32", "128xui16"),
        ("pack", "128xui16", "256xui8"),
        ("unpack", "256xi8", "128xi16"),
        ("unpack", "128xui16", "64xui32"),
    ],
)
@pytest.mark.parametrize("part", ["low", "high"])
def test_vector_pack_unpack_verifier_accepts_lowered_func(op, src_type, dst_type, part) -> None:
    import catlass.runtime as runtime
    from catlass._mlir import ir

    with runtime._eager_capture():
        module = ir.Module.parse(
            f"""
            module {{
              func.func @lowered(%src: !tla.vector<{src_type}>)
                  -> !tla.vector<{dst_type}> {{
                %result = tla.{op} %src, <{part}>
                    : !tla.vector<{src_type}> -> !tla.vector<{dst_type}>
                return %result : !tla.vector<{dst_type}>
              }}
            }}
            """
        )
        assert module.operation.verify()
