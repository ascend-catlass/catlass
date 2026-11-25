import pytest

from catlass import tla
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def _u64_scalar_op(src: tla.Tensor, out: tla.Tensor) -> None:
    with tla.vector():
        value = src[0]
        out[0] = tla.arch.bcnt0(value)
        out[1] = tla.arch.bcnt1(value)
        out[2] = tla.arch.clz(value)
        out[3] = tla.arch.sff0(value)
        out[4] = tla.arch.sff1(value)


def test_each_query_emits_its_mode() -> None:
    src = make_fake_tensor(tla.UInt64, (8,), (1,))
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    mlir = _u64_scalar_op.dump_mlir(type_args=(src, out))

    assert "tla.scalar_bit_op" in mlir
    for mode in ("bcnt0", "bcnt1", "clz", "sff0", "sff1"):
        assert f'"{mode}"' in mlir


@tla.kernel
def _cube_query(src: tla.Tensor, out: tla.Tensor) -> None:
    with tla.cube():
        out[0] = tla.arch.bcnt0(src[0])


def test_query_is_allowed_in_a_cube_region() -> None:
    src = make_fake_tensor(tla.UInt64, (8,), (1,))
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    assert "tla.scalar_bit_op" in _cube_query.dump_mlir(type_args=(src, out))


@tla.kernel
def _literal(out: tla.Tensor) -> None:
    with tla.vector():
        out[0] = tla.arch.bcnt1(255)


def test_accepts_python_int_literal() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    mlir = _literal.dump_mlir(type_args=(out,))
    assert "tla.scalar_bit_op" in mlir
    assert '"bcnt1"' in mlir


@tla.kernel
def _high_bit_literal(out: tla.Tensor) -> None:
    with tla.vector():
        out[0] = tla.arch.bcnt1(0xFFFFFFFFFFFFFFFF)


def test_high_bit_literal_wraps_to_i64_minus_one() -> None:
    """A literal >= 2^63 must land as a negative signless i64 constant."""
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    mlir = _high_bit_literal.dump_mlir(type_args=(out,))
    assert "tla.scalar_bit_op" in mlir
    assert "arith.constant -1 : i64" in mlir
    assert "ui64" not in mlir


# --- runtime scalar argument (frontend proxy bound to a block-value SSA) ---


@tla.kernel
def _runtime_value(out: tla.Tensor, value: tla.UInt64) -> None:
    with tla.vector():
        out[0] = tla.arch.bcnt0(value)
        out[1] = tla.arch.bcnt1(value)
        out[2] = tla.arch.clz(value)
        out[3] = tla.arch.sff0(value)
        out[4] = tla.arch.sff1(value)


def test_accepts_runtime_uint64_argument() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    mlir = _runtime_value.dump_mlir(type_args=(out, tla.UInt64(0)))
    assert mlir.count("tla.scalar_bit_op") == 5
    assert "ui64" in mlir
    assert "builtin.unrealized_conversion_cast" in mlir


@tla.kernel
def _runtime_signed(out: tla.Tensor, value: tla.Int64) -> None:
    with tla.vector():
        out[0] = tla.arch.bcnt0(value)


def test_rejects_runtime_signed_argument() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="64-bit unsigned"):
        _runtime_signed.dump_mlir(type_args=(out, tla.Int64(0)))


@tla.kernel
def _runtime_narrow(out: tla.Tensor, value: tla.UInt32) -> None:
    with tla.vector():
        out[0] = tla.arch.bcnt0(value)


def test_rejects_runtime_narrow_argument() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="64-bit unsigned"):
        _runtime_narrow.dump_mlir(type_args=(out, tla.UInt32(0)))


@tla.kernel
def _runtime_bool(out: tla.Tensor, value: tla.Bool) -> None:
    with tla.vector():
        out[0] = tla.arch.bcnt0(value)


def test_rejects_runtime_bool_argument() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="64-bit unsigned"):
        _runtime_bool.dump_mlir(type_args=(out, tla.Bool(True)))


@tla.kernel
def _reject_signed(src: tla.Tensor, out: tla.Tensor) -> None:
    with tla.vector():
        out[0] = tla.arch.bcnt0(src[0])


def test_rejects_signed_input() -> None:
    src = make_fake_tensor(tla.Int64, (8,), (1,))
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(Exception, match="UInt64"):
        _reject_signed.dump_mlir(type_args=(src, out))


@tla.kernel
def _negative_literal(out: tla.Tensor) -> None:
    with tla.vector():
        out[0] = tla.arch.bcnt0(-1)


def test_rejects_negative_literal() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(Exception, match="non-negative"):
        _negative_literal.dump_mlir(type_args=(out,))


def _overflow_kernel(value: int):
    @tla.kernel
    def overflow(out: tla.Tensor) -> None:
        with tla.vector():
            out[0] = tla.arch.bcnt0(value)

    return overflow


@pytest.mark.parametrize("value", [1 << 64, (1 << 64) + 1])
def test_rejects_out_of_range_int_literal(value: int) -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(Exception, match="fits in 64 bits"):
        _overflow_kernel(value).dump_mlir(type_args=(out,))


@tla.kernel
def _bool_literal(out: tla.Tensor) -> None:
    with tla.vector():
        out[0] = tla.arch.bcnt0(True)


def test_rejects_bool_literal() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(Exception, match="bool"):
        _bool_literal.dump_mlir(type_args=(out,))


@tla.kernel
def _bool_numeric(out: tla.Tensor) -> None:
    with tla.vector():
        out[0] = tla.arch.bcnt0(tla.Bool(True))


def test_rejects_bool_numeric() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(Exception, match="UInt64"):
        _bool_numeric.dump_mlir(type_args=(out,))


@tla.kernel
def _i1_value(out: tla.Tensor) -> None:
    with tla.vector():
        flag = tla.Int32(1) < tla.Int32(2)
        out[0] = tla.arch.bcnt0(flag)


def test_rejects_i1_ssa() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(Exception, match="UInt64"):
        _i1_value.dump_mlir(type_args=(out,))


# --- placement: a scalar op needs a core region outside tla.vec.func ---


@tla.kernel
def _outside_core_regions(src: tla.Tensor, out: tla.Tensor) -> None:
    out[0] = tla.arch.bcnt0(src[0])


def test_query_is_rejected_outside_both_core_regions() -> None:
    src = make_fake_tensor(tla.UInt64, (8,), (1,))
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="must be nested inside tla.cube\\(\\) or tla.vector\\(\\)",
    ):
        _outside_core_regions.dump_mlir(type_args=(src, out))


@tla.kernel
def _inside_simd(out: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            out[0] = tla.arch.bcnt0(255)


def test_query_is_rejected_inside_a_simd_vec_func() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        _inside_simd.dump_mlir(type_args=(out,))


@tla.kernel
def _inside_simt(out: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=32):
            out[0] = tla.arch.bcnt0(255)


def test_query_is_rejected_inside_a_simt_vec_func() -> None:
    out = make_fake_tensor(tla.Int64, (8,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        _inside_simt.dump_mlir(type_args=(out,))


def test_query_requires_a_kernel() -> None:
    with pytest.raises(tla.TlaIRNotExecutableError):
        tla.arch.bcnt0(0)
