"""On-chip capacity budget for ``tla.allocate``.

A kernel's allocations in one address space share one buffer, so the sum is
what has to fit, not each request on its own. Both layers enforce that:

- the frontend, while tracing, refuses a request that does not fit above the
  ones already made (alignment padding included);
- ``tla-lower-ptr`` refuses an over-subscribed space in IR that reaches it
  directly, which is the only check that sees allocations the frontend never
  ran.

UB is capped twice, at two different numbers. The frontend only knows the whole
buffer, because the core type and the ``cce-disable-*-reserved-ubuf`` switches
that move the real ceiling are not available while tracing. ``tla-lower-ptr``
applies the sharp one (248 KiB for SIMD, less for SIMT).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import catlass.tla as tla

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARCH_HEADER = _REPO_ROOT / "include" / "catlass" / "arch" / "arch.hpp"
_TLA_COMPILE = (
    Path(__file__).resolve().parents[1]
    / "csrc"
    / "mlir"
    / "build"
    / "tools"
    / "tla-compile"
    / "TlaCompile"
)

_BYTE_ALIGNMENT = 32
# Ascend950, from struct Ascend950 in include/catlass/arch/arch.hpp.
# L1 / L0A / L0B / L0C are claimable in full: no compiler reserve is carved out
# of them. UB is listed at the whole buffer because that is the frontend's
# bound; arch.hpp's 248 KiB is the lowering-time one, and the two are checked
# separately below.
_ASCEND950_CAPACITY_BYTES = {
    tla.AddressSpace.l1: 512 * 1024,
    tla.AddressSpace.l0a: 64 * 1024,
    tla.AddressSpace.l0b: 64 * 1024,
    tla.AddressSpace.l0c: 256 * 1024,
    tla.AddressSpace.ub: 256 * 1024,
    tla.AddressSpace.biasbuf: 4 * 1024,
}

# arch.hpp field backing each address space.
_ASCEND950_HEADER_FIELDS = {
    tla.AddressSpace.l1: "L1_SIZE",
    tla.AddressSpace.l0a: "L0A_SIZE",
    tla.AddressSpace.l0b: "L0B_SIZE",
    tla.AddressSpace.l0c: "L0C_SIZE",
    tla.AddressSpace.ub: "UB_SIZE",
    tla.AddressSpace.biasbuf: "BIAS_SIZE",
}

_SCOPE_NAMES = {scope: scope.name.upper() for scope in _ASCEND950_CAPACITY_BYTES}


def _ascend950_header_capacity(field: str) -> int:
    """Evaluate one ``static constexpr`` size field of ``struct Ascend950``."""
    text = _ARCH_HEADER.read_text(encoding="utf-8")
    struct_match = re.search(r"struct\s+Ascend950\s*\{(?P<body>.*?)\};", text, re.S)
    assert struct_match is not None, f"struct Ascend950 not found in {_ARCH_HEADER}"
    field_match = re.search(
        rf"\b{field}\s*=\s*(?P<expr>[^;]+);", struct_match.group("body")
    )
    assert field_match is not None, f"{field} not found in struct Ascend950"
    value = 1
    for factor in field_match.group("expr").split("*"):
        value *= int(factor.strip())
    return value


# Smallest admissible local-memory alignment per space, from the Ascend 950 buffer
# alignment table. A buffer may not start mid-block.

_ASCEND950_LOCALMEM_ALIGN_BYTES = {
    tla.AddressSpace.l1: 32,
    tla.AddressSpace.l0a: 512,
    tla.AddressSpace.l0b: 512,
    tla.AddressSpace.l0c: 1024,
    tla.AddressSpace.ub: 32,
    tla.AddressSpace.biasbuf: 64,
}


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _allocate_kernel(mem_scope: tla.AddressSpace, *sizes: int):
    """A kernel whose allocations in ``mem_scope`` are the given byte counts."""
    # Each space's own local-memory alignment: nothing else is admissible.
    alignment = _ASCEND950_LOCALMEM_ALIGN_BYTES[mem_scope]

    @tla.kernel
    def kernel() -> None:
        for size in sizes:
            _ = tla.allocate((size,), tla.Int8, mem_scope, alignment)

    return kernel


def _allocated_sizes(mlir: str) -> list[int]:
    """Byte counts of the ``tla.alloc_ptr`` results, generic and pretty alike."""
    return [int(size) for size in re.findall(r"size_bytes = (\d+)", mlir)]


@pytest.mark.parametrize("mem_scope", [s for s in _ASCEND950_CAPACITY_BYTES if s is not tla.AddressSpace.ub])
def test_local_mem_constant_matches_header(mem_scope: tla.AddressSpace) -> None:
    """L1 / L0 capacities must stay equal to what arch.hpp declares."""
    assert _ASCEND950_CAPACITY_BYTES[mem_scope] == _ascend950_header_capacity(
        _ASCEND950_HEADER_FIELDS[mem_scope]
    )


def test_ub_frontend_bound_is_the_whole_buffer() -> None:
    """The frontend holds UB to 256 KiB, not arch.hpp's 248 KiB.

    A kernel that sizes its UB from the full buffer and releases the compiler
    reserve is legal, and the frontend cannot tell whether it did; refusing it
    at 248 KiB would be a false positive. arch.hpp's figure is the SIMD
    lowering-time limit, applied by tla-lower-ptr.
    """
    assert _ASCEND950_CAPACITY_BYTES[tla.AddressSpace.ub] == 256 * 1024
    assert _ascend950_header_capacity("UB_SIZE") == 248 * 1024


@pytest.mark.parametrize("mem_scope", list(_ASCEND950_CAPACITY_BYTES))
def test_allocations_filling_the_space_are_accepted(
    mem_scope: tla.AddressSpace,
) -> None:
    """Two halves summing to exactly the capacity still lower."""
    capacity = _ASCEND950_CAPACITY_BYTES[mem_scope]
    half = capacity // 2

    mlir = _allocate_kernel(mem_scope, half, half).dump_mlir()

    assert _allocated_sizes(mlir) == [half, half]
    assert mlir.count("tla.alloc_ptr") == 2


@pytest.mark.parametrize(
    ("mem_scope", "request_bytes", "reserved_bytes"),
    (
        (tla.AddressSpace.ub, 16, 32),
        (tla.AddressSpace.ub, 48, 64),
        (tla.AddressSpace.ub, 4096, 4096),
        (tla.AddressSpace.l1, 16, 32),
        (tla.AddressSpace.l1, 33, 64),
        (tla.AddressSpace.l0c, 65, 1024),
        (tla.AddressSpace.l0c, 1024, 1024),
        (tla.AddressSpace.l0c, 1025, 2048),
        (tla.AddressSpace.l0a, 100, 512),
        (tla.AddressSpace.l0a, 512, 512),
        (tla.AddressSpace.l0a, 513, 1024),
        (tla.AddressSpace.l0b, 100, 512),
    ),
)
def test_short_allocation_takes_a_whole_alignment_period(
    mem_scope: tla.AddressSpace, request_bytes: int, reserved_bytes: int
) -> None:
    """A short buffer still takes a whole alignment period.

    16 bytes in UB at alignment 32 takes 32: the extra bytes are space the
    kernel has to find room for, so they show up as the ``size_bytes`` the IR
    carries rather than only as an alignment rule on the address.
    """
    mlir = _allocate_kernel(mem_scope, request_bytes).dump_mlir()

    assert _allocated_sizes(mlir) == [reserved_bytes]


@pytest.mark.parametrize(
    ("mem_scope", "byte_alignment", "request_bytes", "reserved_bytes"),
    (
        # Alignment must be a whole number of blocks; multiples are kept as
        # asked, power of two or not. The size follows the same alignment.
        (tla.AddressSpace.ub, 32, 16, 32),
        (tla.AddressSpace.ub, 64, 16, 64),
        (tla.AddressSpace.ub, 96, 16, 96),
        (tla.AddressSpace.l0a, 512, 16, 512),
        (tla.AddressSpace.l0a, 1024, 16, 1024),
        (tla.AddressSpace.l0a, 1536, 16, 1536),
        (tla.AddressSpace.l0a, 1024, 768, 1024),
        (tla.AddressSpace.l0c, 1024, 16, 1024),
        (tla.AddressSpace.l0c, 2048, 16, 2048),
        (tla.AddressSpace.l0c, 3072, 16, 3072),
    ),
)
def test_alignment_is_a_whole_number_of_blocks(
    mem_scope: tla.AddressSpace,
    byte_alignment: int,
    request_bytes: int,
    reserved_bytes: int,
) -> None:
    """Alignment is a block-multiple question, not a power-of-two one.

    1536 is legal in L0A because 1536 = 3 x 512; 96 is legal in UB because
    96 = 3 x 32. Both are non-powers of two and both keep every address a whole
    number of blocks in. A power-of-two-only rule would reject them.

    The allocation size is rounded to the requested ``byte_alignment``: the
    alignment both places the buffer and decides how much room it takes.
    """

    @tla.kernel
    def kernel() -> None:
        _ = tla.allocate((request_bytes,), tla.Int8, mem_scope, byte_alignment)

    mlir = kernel.dump_mlir()

    assert f"!tla.ptr<i8, {mem_scope.name}, {byte_alignment}>" in mlir
    assert _allocated_sizes(mlir) == [reserved_bytes]


@pytest.mark.parametrize(
    ("mem_scope", "byte_alignment", "reason"),
    (
        # Not a multiple of the space's local-memory alignment: the address would start mid-block.
        (tla.AddressSpace.l0a, 768, "expected a multiple of the 512-byte L0A local-memory alignment"),
        (tla.AddressSpace.l0b, 768, "expected a multiple of the 512-byte L0B local-memory alignment"),
        (tla.AddressSpace.ub, 48, "expected a multiple of the 32-byte UB local-memory alignment"),
        (tla.AddressSpace.l0c, 512, "expected a multiple of the 1024-byte L0C local-memory alignment"),
        # Below the local-memory alignment, and refused rather than raised: rounding up
        # would place the buffer at an alignment the caller did not ask for.
        (tla.AddressSpace.ub, 16, "expected a multiple of the 32-byte UB local-memory alignment"),
        (tla.AddressSpace.ub, 24, "expected a multiple of the 32-byte UB local-memory alignment"),
        (tla.AddressSpace.l0a, 256, "expected a multiple of the 512-byte L0A local-memory alignment"),
        (tla.AddressSpace.l0c, 1536, "expected a multiple of the 1024-byte L0C local-memory alignment"),
    ),
)
def test_alignment_off_the_localmem_grid_is_refused(
    mem_scope: tla.AddressSpace, byte_alignment: int, reason: str
) -> None:
    """An alignment the space's local-memory alignment cannot express is refused, not relaxed."""

    @tla.kernel
    def kernel() -> None:
        _ = tla.allocate((16,), tla.Int8, mem_scope, byte_alignment)

    with pytest.raises(tla.TlaCoreAPIError, match=reason):
        kernel.dump_mlir()


@tla.kernel
def _aligned_then_coarser_kernel() -> None:
    """A block-aligned buffer followed by a coarser-aligned one."""
    _ = tla.allocate((512,), tla.Int8, tla.AddressSpace.l0a, 512)
    _ = tla.allocate((512,), tla.Int8, tla.AddressSpace.l0a, 2048)


def test_coarser_alignment_places_and_grows_the_buffer() -> None:
    """Alignment decides where the buffer starts and how much room it takes.

    The second request is 512 bytes at 2048-byte alignment, so it reserves 2048
    and starts at 2048 -- the size follows the request's own alignment.
    """
    mlir = _aligned_then_coarser_kernel.dump_mlir()

    assert _allocated_sizes(mlir) == [512, 2048]
    assert "!tla.ptr<i8, l0a, 512>" in mlir
    assert "!tla.ptr<i8, l0a, 2048>" in mlir


@pytest.mark.parametrize("mem_scope", list(_ASCEND950_CAPACITY_BYTES))
def test_allocations_past_the_space_are_refused(mem_scope: tla.AddressSpace) -> None:
    """Each request fits; together they do not, and the second one is refused."""
    capacity = _ASCEND950_CAPACITY_BYTES[mem_scope]
    localmem_alignment = _ASCEND950_LOCALMEM_ALIGN_BYTES[mem_scope]
    # Just over half, rounded to a block so the allocation size is the request.
    request = _round_up(capacity // 2 + 1, localmem_alignment)
    name = _SCOPE_NAMES[mem_scope]

    assert request <= capacity
    assert 2 * request > capacity

    with pytest.raises(tla.TlaCoreAPIError) as exc_info:
        _allocate_kernel(mem_scope, request, request).dump_mlir()

    message = str(exc_info.value)
    assert f"exceeds {name} capacity" in message
    # The message has to say what the total would have been, not just "too big":
    # the request alone was fine, so the second allocation is the interesting one.
    assert f"{request} bytes already allocated" in message
    assert f"past the {capacity}-byte capacity" in message


def test_alignment_padding_counts_toward_the_capacity() -> None:
    """A total that only exceeds the space once padded is still refused.

    L0A is 64 KiB, or 128 blocks of 512. The first buffer takes 127 blocks and
    leaves exactly one; the second asks for 512 bytes aligned to 1024, which
    cannot start there, so it is pushed to 65536 and ends 512 bytes past the
    end. The plain sum (65024 + 512) is exactly the capacity, which is why a
    check that ignores the placement gap lets this through.
    """
    capacity = _ASCEND950_CAPACITY_BYTES[tla.AddressSpace.l0a]
    first, second, second_align = 65024, 512, 1024
    assert first + second == capacity

    @tla.kernel
    def kernel() -> None:
        _ = tla.allocate((first,), tla.Int8, tla.AddressSpace.l0a, 512)
        _ = tla.allocate((second,), tla.Int8, tla.AddressSpace.l0a, second_align)

    with pytest.raises(tla.TlaCoreAPIError) as exc_info:
        kernel.dump_mlir()

    message = str(exc_info.value)
    assert "exceeds L0A capacity" in message
    assert f"{first} bytes already allocated, + 512 bytes of alignment padding" in message
    assert f"past the {capacity}-byte capacity" in message


def test_request_past_the_remaining_space_is_refused() -> None:
    """A coarser-aligned request rounds up, and one block too many no longer fits.

    65024 bytes aligned to 1024 reserve 65536 -- the size follows the request's
    own alignment -- so the second, 1024-byte request starts at 65536 and ends
    1024 bytes past the end of the 64 KiB L0A.
    """
    first, second = 65024, 1024

    @tla.kernel
    def kernel() -> None:
        _ = tla.allocate((first,), tla.Int8, tla.AddressSpace.l0a, 1024)
        _ = tla.allocate((second,), tla.Int8, tla.AddressSpace.l0a, 512)

    with pytest.raises(tla.TlaCoreAPIError) as exc_info:
        kernel.dump_mlir()

    message = str(exc_info.value)
    assert "exceeds L0A capacity" in message
    assert "65536 bytes already allocated" in message
    assert "+ 1024 bytes requested = 66560 bytes" in message


@tla.kernel
def _dynamic_ub_kernel() -> None:
    """Static UB filling the whole space, plus a dynamic region above it."""
    _ = tla.allocate((65536,), tla.Int8, tla.AddressSpace.ub, _BYTE_ALIGNMENT)
    _ = tla.arch.get_dyn_ub(tla.Int8, byte_alignment=_BYTE_ALIGNMENT)


def test_dynamic_ub_does_not_spend_the_static_budget() -> None:
    """``get_dyn_ub`` reserves nothing here: its extent is set at launch.

    Counting it would charge the kernel twice, since the region starts above the
    static allocations rather than being one of them.
    """
    mlir = _dynamic_ub_kernel.dump_mlir()
    assert _allocated_sizes(mlir) == [65536]
    assert mlir.count("tla.dynamic_ub_base") == 1


def _run_tla_compile(module_text: str) -> subprocess.CompletedProcess[str]:
    if not _TLA_COMPILE.is_file():
        raise AssertionError(f"TlaCompile not found at {_TLA_COMPILE}; run ./build.sh")
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as handle:
        handle.write(module_text)
        path = handle.name
    try:
        return subprocess.run(
            [str(_TLA_COMPILE), path, "-o", "-"],
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("addrspace", "capacity", "alignment"),
    [
        ("l1", 512 * 1024, 32),
        ("l0a", 64 * 1024, 512),
        ("l0b", 64 * 1024, 512),
        ("l0c", 256 * 1024, 1024),
        ("biasbuf", 4 * 1024, 64),
    ],
)
def test_lowering_accepts_space_filled_exactly(
    addrspace: str, capacity: int, alignment: int
) -> None:
    """A space filled to the byte still compiles."""
    module_text = f"""module {{
  tla.func @kernel() {{
    %0 = tla.alloc_ptr{{size_bytes = {capacity // 2}}} -> !tla.ptr<i8, {addrspace}, {alignment}>
    %1 = tla.alloc_ptr{{size_bytes = {capacity // 2}}} -> !tla.ptr<i8, {addrspace}, {alignment}>
    tla.return
  }}
}}
"""
    result = _run_tla_compile(module_text)

    assert result.returncode == 0, result.stdout + result.stderr



# --- mxfp8 raises the L0 alignment, which only the matmul can see -----------


def _mx_kernel(elem: type[tla.Numeric], l0_align: int):
    """An mxfp8 matmul whose L0 operands are allocated at ``l0_align``."""

    @tla.kernel
    def kernel() -> None:
        l1a = tla.make_tensor(
            tla.allocate((128, 64), elem, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
        )
        l1b = tla.make_tensor(
            tla.allocate((64, 128), elem, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
        )
        accp = tla.make_tensor(
            tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
        )
        sa_p = tla.make_tensor(
            tla.allocate((128, 2), tla.Float8E8M0, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 2), tla.make_stride(2, 1)),
        )
        sb_p = tla.make_tensor(
            tla.allocate((2, 128), tla.Float8E8M0, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(2, 128), tla.make_stride(128, 1)),
        )
        lhs = tla.make_tensor_like(
            tla.allocate((128, 64), elem, tla.AddressSpace.l0a, l0_align),
            l1a,
            tla.arch.zN,
        )
        rhs = tla.make_tensor_like(
            tla.allocate((64, 128), elem, tla.AddressSpace.l0b, l0_align),
            l1b,
            tla.arch.nZ,
        )
        acc = tla.make_tensor_like(
            tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l0c, 1024),
            accp,
            tla.arch.L0Clayout,
        )
        sa = tla.make_tensor_like(
            tla.allocate((128, 2), tla.Float8E8M0, tla.AddressSpace.l1, 512),
            sa_p,
            tla.arch.zZMxScale,
        )
        sb = tla.make_tensor_like(
            tla.allocate((2, 128), tla.Float8E8M0, tla.AddressSpace.l1, 512),
            sb_p,
            tla.arch.nNMxScale,
        )
        src_a = tla.make_tensor_like(
            tla.allocate((128, 64), elem, tla.AddressSpace.l1, 512), l1a, tla.arch.zN
        )
        src_b = tla.make_tensor_like(
            tla.allocate((64, 128), elem, tla.AddressSpace.l1, 512), l1b, tla.arch.nZ
        )
        with tla.cube():
            tla.copy(lhs, src_a, scale=sa)
            tla.copy(rhs, src_b, scale=sb)
            tla.mmad_mx(acc, lhs, rhs, init_c=True, unit_flag=3)

    return kernel


def _mx_mlir(elem: type[tla.Numeric], l0_align: int) -> str:
    return _mx_kernel(elem, l0_align).dump_mlir()


def _ir_elem(elem: type[tla.Numeric]) -> str:
    """The element keyword as the printer spells it (``f8E4M3FN``, ``f4e2m1``)."""
    return {
        tla.Float8E4M3FN: "f8E4M3FN",
        tla.Float8E5M2: "f8E5M2",
        tla.Float4E2M1: "f4e2m1",
        tla.Float4E1M2: "f4e1m2",
    }[elem]


@pytest.mark.parametrize("elem", (tla.Float8E4M3FN, tla.Float8E5M2))
@pytest.mark.parametrize(
    ("l0_align", "reason"),
    (
        # Below the requirement: half a block off.
        (512, "aligned to 512 bytes"),
        # A legal L0A alignment (3 x 512) that is not a multiple of 1024, so an
        # address it promises can still land 512 bytes into an mxfp8 block.
        # This is the case a ">= 1024" test would wrongly accept.
        (1536, "aligned to 1536 bytes"),
    ),
)
def test_mxfp8_needs_a_1024_byte_l0_alignment(
    elem: type[tla.Numeric], l0_align: int, reason: str
) -> None:
    """512 is legal for every L0 user, so only the matmul can refuse mxfp8.

    ``tla.allocate`` sees the address space and nothing else -- 512 is the right
    the local-memory alignment for L0A / L0B in general -- so the frontend accepts both. The
    operand format only becomes known at ``tla.mmad_mx``, and that is where the
    check lives.

    The requirement is that 1024 *divides* the alignment, not that the alignment
    is at least 1024: an N-byte alignment promises multiples of N, which are
    multiples of 1024 only when 1024 | N.
    """
    # The frontend accepts the operand: nothing is known about it yet.
    mlir = _mx_mlir(elem, l0_align)
    assert f"!tla.ptr<{_ir_elem(elem)}, l0a, {l0_align}>" in mlir

    result = _run_tla_compile(mlir)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "must be 1024-byte aligned for an mxfp8 MX operand" in combined
    assert reason in combined


@pytest.mark.parametrize("elem", (tla.Float8E4M3FN, tla.Float8E5M2))
@pytest.mark.parametrize("l0_align", (1024, 2048, 3072))
def test_mxfp8_at_a_multiple_of_1024_is_accepted(
    elem: type[tla.Numeric], l0_align: int
) -> None:
    """Any multiple of 1024 works: 2048 and 3072 promise 1024-byte alignment too."""
    mlir = _mx_mlir(elem, l0_align)
    assert f"!tla.ptr<{_ir_elem(elem)}, l0a, {l0_align}>" in mlir

    result = _run_tla_compile(mlir)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("elem", (tla.Float4E2M1, tla.Float4E1M2))
def test_mxfp4_keeps_the_512_byte_l0_block(elem: type[tla.Numeric]) -> None:
    """Packed fp4 is not mxfp8: it is addressed in 512-byte blocks like any L0 tile."""
    mlir = _mx_mlir(elem, 512)

    result = _run_tla_compile(mlir)

    assert result.returncode == 0, result.stdout + result.stderr


