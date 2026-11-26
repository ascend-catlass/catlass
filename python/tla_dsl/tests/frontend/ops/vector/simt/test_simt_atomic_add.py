"""Frontend API for the SIMT atomic add.

``tla.arch.atomic_add(ptr, value)`` is SIMT-only and lowers to
``tla.simt_atomic_add``; the lowering itself is covered by the lit test
``lit/vector/simt/simt-atomic-add-lowering.mlir`` and the verifier rules by
``lit/vector/simt/simt-atomic-add-diagnostics.mlir``.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import pytest

from catlass._mlir.dialects import tla as tla_ops_gen
from catlass import tla
from catlass.tla.runtime import make_fake_tensor


def _run_tla_compile(mlir_text: str, pass_name: str) -> str:
    """Run the native pipeline and return the IR printed after `pass_name`.

    The frontend's printed IR is not the contract: a constant of a type the
    LLVM dialect cannot carry only fails once the lowering reaches it. Runs the
    full pipeline so a rejection shows up here rather than on the device.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[5]
    tla_compile = repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    if not tla_compile.exists():
        raise AssertionError("TlaCompile binary not found. Build csrc/mlir first.")
    with tempfile.TemporaryDirectory() as tmpdir:
        source = pathlib.Path(tmpdir) / "kernel.mlir"
        output = pathlib.Path(tmpdir) / "kernel.out"
        source.write_text(mlir_text)
        result = subprocess.run(
            [str(tla_compile), str(source), "-o", str(output), f"--mlir-print-ir-after={pass_name}"],
            check=False,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def _tensor(dtype, *, addrspace=tla.AddressSpace.gm, shape=(64,)):
    stride = tuple(shape[index + 1] for index in range(len(shape) - 1)) + (1,)
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        addrspace=addrspace,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


def test_generated_binding_exists_for_atomic_add() -> None:
    assert hasattr(tla_ops_gen, "simt_atomic_add")
    assert hasattr(tla.arch, "atomic_add")
    # The tensor+coordinate form was replaced, not kept as an overload.
    assert not hasattr(tla, "atomic_add")


@pytest.mark.parametrize(
    ("dtype", "addrspace"),
    [
        (tla.Int32, tla.AddressSpace.gm),
        (tla.UInt32, tla.AddressSpace.gm),
        (tla.Int64, tla.AddressSpace.gm),
        (tla.UInt64, tla.AddressSpace.gm),
        (tla.Float32, tla.AddressSpace.ub),
        (tla.Float16, tla.AddressSpace.gm),
        (tla.BFloat16, tla.AddressSpace.ub),
    ],
)
def test_atomic_add_emits_simt_atomic_add(dtype, addrspace) -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tid, _, _ = tla.arch.thread_idx()
                tla.arch.atomic_add(dst.ptr + tid, dtype(1))

    ir = kernel.dump_mlir(type_args=(_tensor(dtype, addrspace=addrspace),))
    assert "tla.simt_atomic_add" in ir
    assert "tla.simt_store" not in ir
    # The address is built inside the region from the tensor's backing pointer.
    assert "tla.tensor_ptr" in ir
    assert "tla.ptr_add" in ir


def test_atomic_add_accepts_constant_element_offset() -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.atomic_add(dst.ptr + 3, tla.Float32(1.0))

    ir = kernel.dump_mlir(type_args=(_tensor(tla.Float32, shape=(8, 8)),))
    assert "tla.simt_atomic_add" in ir
    assert "arith.constant 3 : index" in ir


def test_atomic_add_accepts_python_literal() -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.atomic_add(dst.ptr, 1.5)

    ir = kernel.dump_mlir(type_args=(_tensor(tla.Float32),))
    assert "tla.simt_atomic_add" in ir
    assert "arith.constant 1.500000e+00 : f32" in ir


def test_atomic_add_accepts_non_tensor_pointer() -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                p = tla.arch.get_dyn_ub(tla.Int32)
                tla.arch.atomic_add(p + 4, tla.Int32(1))

    ir = kernel.dump_mlir(type_args=(_tensor(tla.Int32),))
    assert "tla.dynamic_ub_base" in ir
    assert "tla.simt_atomic_add" in ir


def test_atomic_add_outside_simt_uses_scalar_form() -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            tla.arch.atomic_add(dst.ptr, 1.0)

    # Outside a SIMT region the shared entry takes the scalar form and emits the
    # SIMD atomic op instead of the per-thread simt op.
    ir = kernel.dump_mlir(type_args=(_tensor(tla.Float32),))
    assert "tla.simd_atomic_add" in ir
    assert "tla.simt_atomic_add" not in ir


def test_atomic_add_rejects_tensor_argument() -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.atomic_add(dst, tla.Float32(1.0))

    with pytest.raises(Exception, match="expected pointer"):
        kernel.dump_mlir(type_args=(_tensor(tla.Float32),))


@pytest.mark.parametrize("dtype", [tla.Int8, tla.Int16, tla.UInt8])
def test_atomic_add_rejects_unsupported_element_type(dtype) -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.atomic_add(dst.ptr, dtype(1))

    with pytest.raises(Exception, match="unsupported pointee type"):
        kernel.dump_mlir(type_args=(_tensor(dtype),))


@pytest.mark.parametrize("dtype", [tla.Int64, tla.UInt64])
def test_atomic_add_rejects_64bit_on_ub(dtype) -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.atomic_add(dst.ptr, dtype(1))

    with pytest.raises(Exception, match="only supported on gm"):
        kernel.dump_mlir(type_args=(_tensor(dtype, addrspace=tla.AddressSpace.ub),))


def test_atomic_add_rejects_implicit_type_conversion() -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.atomic_add(dst.ptr, src[0])

    with pytest.raises(Exception, match="cast explicitly"):
        kernel.dump_mlir(
            type_args=(_tensor(tla.Int32), _tensor(tla.Float32)),
        )


def test_atomic_add_returns_previous_value() -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor, out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                old = tla.arch.atomic_add(dst.ptr, tla.Int32(3))
                out[0] = old.to(tla.Int64)

    ir = kernel.dump_mlir(type_args=(_tensor(tla.Int32), _tensor(tla.Int64)))
    assert "tla.simt_atomic_add" in ir
    assert "arith.constant 3 : i32" in ir
    # The previous value rides the op result and the store consumes it.
    assert "tla.simt_store" in ir


def test_atomic_add_returns_unsigned_previous_value() -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor, out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                old = tla.arch.atomic_add(dst.ptr, tla.UInt32(3))
                out[0] = old.to(tla.Int64)

    ir = kernel.dump_mlir(type_args=(_tensor(tla.UInt32), _tensor(tla.Int64)))
    assert "tla.simt_atomic_add" in ir
    # The ui32 result is fed through the signless cast before Numeric use.
    assert "builtin.unrealized_conversion_cast" in ir


@pytest.mark.parametrize("dtype", [tla.Float16, tla.BFloat16])
def test_atomic_add_returns_none_for_f16_bf16(dtype) -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor, out: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                old = tla.arch.atomic_add(dst.ptr, dtype(1.0))
                # Assert on the value itself: a `0.0 if old is None else ...`
                # fallback would pass whether or not the API withheld it.
                assert old is None, f"atomic_add returned {old!r} for {dtype}"
                out[0] = tla.Float32(0.0)

    ir = kernel.dump_mlir(
        type_args=(_tensor(dtype, addrspace=tla.AddressSpace.gm), _tensor(tla.Float32))
    )
    # The op still fires; only the untrustworthy hardware return value is withheld.
    assert "tla.simt_atomic_add" in ir
    assert "f16x2" not in ir and "bf16x2" not in ir


def test_atomic_add_unsigned_literal_compiles_natively() -> None:
    """A host literal on an unsigned pointee must survive the whole pipeline.

    arith has no unsigned integer type, so emitting the literal as a ``ui32``
    constant looks fine in the frontend's IR and then fails in the native
    lowering with "integer return type must be signless". Only running the
    pipeline proves the constant is one the backend accepts.
    """

    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tid, _, _ = tla.arch.thread_idx()
                tla.arch.atomic_add(dst.ptr + tid, 1)

    ir = _run_tla_compile(
        kernel.dump_mlir(type_args=(_tensor(tla.UInt32),)), "tla-vector-region-simt"
    )
    assert "llvm.hivm.atom.ADD.G.s32" in ir


def test_atomic_add_uint64_literal_above_sign_bit() -> None:
    """A uint64 literal past 2**63 must arrive as its bit pattern.

    The addend is emitted as a signless constant and MLIR's integer attribute is
    signed, so 2**64 - 1 has to be spelled -1 rather than overflow the attribute.
    """

    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tid, _, _ = tla.arch.thread_idx()
                tla.arch.atomic_add(dst.ptr + tid, 2**64 - 1)

    frontend_ir = kernel.dump_mlir(type_args=(_tensor(tla.UInt64),))
    assert "arith.constant -1 : i64" in frontend_ir
    ir = _run_tla_compile(frontend_ir, "tla-vector-region-simt")
    assert "llvm.hivm.atom.ADD.G.s64" in ir


def test_atomic_add_rejects_unsigned_literal_out_of_range() -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                tla.arch.atomic_add(dst.ptr, 2**32)

    with pytest.raises(Exception, match="out of range"):
        kernel.dump_mlir(type_args=(_tensor(tla.UInt32),))
