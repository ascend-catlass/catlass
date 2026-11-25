"""SIMD atomic add / exch on real global memory (Ascend 950).

Checks the v1 contract on hardware:

- every block's ``tla.arch.atomic_add`` / ``atomic_exch`` to one GM address is
  serialized by the atom unit, so the recorded old values form the expected
  multiset and the final element is the total;
- the old value is written straight back into a same-typed tensor
  (i32 / f32 / u32 / u64), never converted;
- a scalar read after the atom sees the new value once the line is maintained
  with ``tla.arch.dcci_cacheline`` (the documented probe; the plain read before
  it is reported, not asserted);
- a SIMD region that also does vector work keeps the atom in its scalar part,
  while the vector store is outlined into a helper that carries no atom;
- a ``tla.cube`` region (AIC) drives the same GM atom unit, so a cube-only
  kernel reaches the same final counter.
"""

from __future__ import annotations

import argparse
import json

from catlass import tla

BLOCKS = 8
# AIC blocks for the cube-only cases; well under the 32 cube cores.
AIC_BLOCKS = 8
# Each block records its old value 128 bytes apart: the scalar stores of two
# cores must not share a dcache line, or the line write-back of one would drop
# the other's record (this is a test-harness property, not the atomic).
OLD_STRIDE = 32
GUARD_INT = -99
TENSOR_LEN = 128

# (label, torch dtype, TLA element type, unsigned view for from_dlpack)
CASES = (
    ("i32", "int32", tla.Int32, None),
    ("f32", "float32", tla.Float32, None),
    ("u32", "int32", tla.UInt32, "uint32"),
    ("u64", "int64", tla.UInt64, "uint64"),
)

# AIC (tla.cube) cases: ADD includes signed i64, the width the AIV matrix does
# not cover; EXCH keeps the unsigned 32-bit form.
AIC_ADD_CASES = (
    ("i64", "int64", None),
    ("u32", "int32", "uint32"),
)
AIC_EXCH_CASES = (("u32", "int32", "uint32"),)


def make_add_kernel(addend):
    @tla.kernel
    def atomic_add_kernel(dst: tla.Tensor, old: tla.Tensor) -> None:
        block = tla.arch.block_idx()
        with tla.vector():
            old[block * OLD_STRIDE] = tla.arch.atomic_add(dst.ptr, addend)

    return atomic_add_kernel


def make_exch_kernel():
    @tla.kernel
    def atomic_exch_kernel(dst: tla.Tensor, old: tla.Tensor) -> None:
        block = tla.arch.block_idx()
        with tla.vector():
            old[block * OLD_STRIDE] = tla.arch.atomic_exch(dst.ptr, 1)

    return atomic_exch_kernel


def make_probe_kernel():
    @tla.kernel
    def probe_kernel(
        dst: tla.Tensor,
        old: tla.Tensor,
        plain: tla.Tensor,
        maintained: tla.Tensor,
    ) -> None:
        with tla.vector():
            old[0] = tla.arch.atomic_add(dst.ptr, 1)
            plain[0] = dst[0]
            tla.arch.dcci_cacheline(dst.ptr)
            maintained[0] = dst[0]

    return probe_kernel


def make_outline_kernel(dtype):
    """A SIMD region with vector work: the atom stays in the scalar part."""
    # One 256-byte register: 64 lanes for 32-bit elements, 32 for 64-bit.
    lanes = 64 if dtype.width == 32 else 32

    @tla.kernel
    def outline_kernel(dst: tla.Tensor, old: tla.Tensor) -> None:
        ub_ptr = tla.allocate(lanes, dtype, tla.AddressSpace.ub, 256)
        ub = tla.make_tensor_like(ub_ptr, dst, tla.arch.RowMajor)
        with tla.vector():
            old[0] = tla.arch.atomic_add(dst.ptr, 1)
            with tla.vec.func(mode="simd"):
                tile = tla.tile_view(ub, tla.make_shape(lanes), tla.make_coord(0))
                tile.store(tile.load())

    return outline_kernel


def make_aic_add_kernel(addend):
    """A cube-only kernel: AIC issues the same scalar atom."""

    @tla.kernel
    def aic_atomic_add_kernel(dst: tla.Tensor, old: tla.Tensor) -> None:
        block = tla.arch.block_idx()
        with tla.cube():
            old[block * OLD_STRIDE] = tla.arch.atomic_add(dst.ptr, addend)

    return aic_atomic_add_kernel


def make_aic_exch_kernel():
    @tla.kernel
    def aic_atomic_exch_kernel(dst: tla.Tensor, old: tla.Tensor) -> None:
        block = tla.arch.block_idx()
        with tla.cube():
            old[block * OLD_STRIDE] = tla.arch.atomic_exch(dst.ptr, 1)

    return aic_atomic_exch_kernel


def _guarded(torch, dtype, length):
    """All-guard tensor; the unsigned views keep the same guard pattern."""
    if dtype.is_floating_point:
        return torch.full((length,), float(GUARD_INT), dtype=dtype, device="npu")
    return torch.full((length,), GUARD_INT, dtype=dtype, device="npu")


def _recorded(old_values, blocks=BLOCKS):
    """Values each block stored, one 128-byte slot apart."""
    return [old_values[block * OLD_STRIDE] for block in range(blocks)]


def _check_guards(values, label, blocks=BLOCKS):
    recorded = set(block * OLD_STRIDE for block in range(blocks))
    guard = float(GUARD_INT) if isinstance(values[0], float) else GUARD_INT
    for index, value in enumerate(values):
        if index not in recorded and value != guard:
            raise AssertionError(f"{label}: guard at {index} changed to {value}")


def run_add(torch, from_dlpack, label, torch_dtype, tla_dtype, view, npu_arch):
    addend = 1
    dst = torch.zeros((1,), dtype=torch_dtype, device="npu")
    old = _guarded(torch, torch_dtype, BLOCKS * OLD_STRIDE + 8)
    dst_arg = from_dlpack(
        dst.view(getattr(torch, view)) if view else dst, layout_tag=tla.arch.RowMajor
    )
    old_arg = from_dlpack(
        old.view(getattr(torch, view)) if view else old, layout_tag=tla.arch.RowMajor
    )
    artifact = tla.compile(
        make_add_kernel(addend), dst_arg, old_arg, options=f"--npu-arch {npu_arch}"
    )
    artifact(dst_arg, old_arg, block_num=BLOCKS)
    torch.npu.synchronize()

    final = dst.cpu().tolist()[0]
    expected_final = BLOCKS * addend
    if final != expected_final:
        raise AssertionError(f"{label} add: dst={final} expected {expected_final}")
    seen = old.cpu().tolist()
    _check_guards(seen, f"{label} add")
    recorded = _recorded(seen)
    if sorted(recorded) != [i * addend for i in range(BLOCKS)]:
        raise AssertionError(f"{label} add: old values {sorted(recorded)} are not the multiset")
    return {"case": f"add-{label}", "old_multiset": sorted(recorded), "passed": True}


def run_exch(torch, from_dlpack, label, torch_dtype, view, npu_arch):
    dst = torch.zeros((1,), dtype=torch_dtype, device="npu")
    old = _guarded(torch, torch_dtype, BLOCKS * OLD_STRIDE + 8)
    dst_arg = from_dlpack(
        dst.view(getattr(torch, view)) if view else dst, layout_tag=tla.arch.RowMajor
    )
    old_arg = from_dlpack(
        old.view(getattr(torch, view)) if view else old, layout_tag=tla.arch.RowMajor
    )
    artifact = tla.compile(make_exch_kernel(), dst_arg, old_arg, options=f"--npu-arch {npu_arch}")
    artifact(dst_arg, old_arg, block_num=BLOCKS)
    torch.npu.synchronize()

    if dst.cpu().tolist()[0] != 1:
        raise AssertionError(f"{label} exch: dst={dst.cpu().tolist()[0]} expected 1")
    seen = old.cpu().tolist()
    _check_guards(seen, f"{label} exch")
    recorded = _recorded(seen)
    expected = sorted([0] + [1] * (BLOCKS - 1))
    if sorted(recorded) != expected:
        raise AssertionError(f"{label} exch: old values {sorted(recorded)} are not {expected}")
    return {"case": f"exch-{label}", "old_multiset": sorted(recorded), "passed": True}


def run_probe(torch, from_dlpack, npu_arch):
    dst = torch.zeros((1,), dtype=torch.int32, device="npu")
    old = _guarded(torch, torch.int32, 2)
    plain = _guarded(torch, torch.int32, 2)
    maintained = _guarded(torch, torch.int32, 2)
    args = [
        from_dlpack(t, layout_tag=tla.arch.RowMajor)
        for t in (dst, old, plain, maintained)
    ]
    artifact = tla.compile(make_probe_kernel(), *args, options=f"--npu-arch {npu_arch}")
    artifact(*args, block_num=1)
    torch.npu.synchronize()

    if maintained.cpu().tolist()[0] != 1:
        raise AssertionError(
            "probe: maintained read did not observe the atomic result "
            f"({maintained.cpu().tolist()[0]})"
        )
    return {
        "case": "dcci-probe",
        "plain_read": plain.cpu().tolist()[0],
        "maintained_read": maintained.cpu().tolist()[0],
        "passed": True,
    }


def run_outline(torch, from_dlpack, label, torch_dtype, tla_dtype, view, npu_arch):
    dst = torch.zeros((TENSOR_LEN,), dtype=torch_dtype, device="npu")
    old = _guarded(torch, torch_dtype, 2)
    dst_arg = from_dlpack(
        dst.view(getattr(torch, view)) if view else dst, layout_tag=tla.arch.RowMajor
    )
    old_arg = from_dlpack(
        old.view(getattr(torch, view)) if view else old, layout_tag=tla.arch.RowMajor
    )
    artifact = tla.compile(
        make_outline_kernel(tla_dtype), dst_arg, old_arg, options=f"--npu-arch {npu_arch}"
    )
    artifact(dst_arg, old_arg, block_num=1)
    torch.npu.synchronize()

    if dst.cpu().tolist()[0] != 1:
        raise AssertionError(f"{label} outline: dst[0]={dst.cpu().tolist()[0]} expected 1")
    if old.cpu().tolist()[0] != 0:
        raise AssertionError(f"{label} outline: old[0]={old.cpu().tolist()[0]} expected 0")
    if any(value != 0 for value in dst.cpu().tolist()[1:]):
        raise AssertionError(f"{label} outline: elements past 0 changed")
    return {"case": f"outline-{label}", "passed": True}


def run_aic_add(torch, from_dlpack, label, torch_dtype, view, npu_arch):
    addend = 1
    dst = torch.zeros((1,), dtype=torch_dtype, device="npu")
    old = _guarded(torch, torch_dtype, AIC_BLOCKS * OLD_STRIDE + 8)
    dst_arg = from_dlpack(
        dst.view(getattr(torch, view)) if view else dst, layout_tag=tla.arch.RowMajor
    )
    old_arg = from_dlpack(
        old.view(getattr(torch, view)) if view else old, layout_tag=tla.arch.RowMajor
    )
    artifact = tla.compile(
        make_aic_add_kernel(addend), dst_arg, old_arg, options=f"--npu-arch {npu_arch}"
    )
    artifact(dst_arg, old_arg, block_num=AIC_BLOCKS)
    torch.npu.synchronize()

    final = dst.cpu().tolist()[0]
    if final != AIC_BLOCKS * addend:
        raise AssertionError(f"AIC add-{label}: dst={final} expected {AIC_BLOCKS * addend}")
    seen = old.cpu().tolist()
    _check_guards(seen, f"AIC add-{label}", AIC_BLOCKS)
    recorded = _recorded(seen, AIC_BLOCKS)
    if sorted(recorded) != [i * addend for i in range(AIC_BLOCKS)]:
        raise AssertionError(f"AIC add-{label}: old values {sorted(recorded)} are not the multiset")
    return {"case": f"aic-add-{label}", "old_multiset": sorted(recorded), "passed": True}


def run_aic_exch(torch, from_dlpack, label, torch_dtype, view, npu_arch):
    dst = torch.zeros((1,), dtype=torch_dtype, device="npu")
    old = _guarded(torch, torch_dtype, AIC_BLOCKS * OLD_STRIDE + 8)
    dst_arg = from_dlpack(
        dst.view(getattr(torch, view)) if view else dst, layout_tag=tla.arch.RowMajor
    )
    old_arg = from_dlpack(
        old.view(getattr(torch, view)) if view else old, layout_tag=tla.arch.RowMajor
    )
    artifact = tla.compile(
        make_aic_exch_kernel(), dst_arg, old_arg, options=f"--npu-arch {npu_arch}"
    )
    artifact(dst_arg, old_arg, block_num=AIC_BLOCKS)
    torch.npu.synchronize()

    final = dst.cpu().tolist()[0]
    if final != 1:
        raise AssertionError(f"AIC exch-{label}: dst={final} expected 1")
    seen = old.cpu().tolist()
    _check_guards(seen, f"AIC exch-{label}", AIC_BLOCKS)
    recorded = _recorded(seen, AIC_BLOCKS)
    expected = sorted([0] + [1] * (AIC_BLOCKS - 1))
    if sorted(recorded) != expected:
        raise AssertionError(f"AIC exch-{label}: old values {sorted(recorded)} are not {expected}")
    return {"case": f"aic-exch-{label}", "old_multiset": sorted(recorded), "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--npu-arch", default="3510")
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    for label, dtype_name, tla_dtype, view in CASES:
        torch_dtype = getattr(torch, dtype_name)
        for runner in (run_add, run_outline):
            result = runner(
                torch, from_dlpack, label, torch_dtype, tla_dtype, view, args.npu_arch
            )
            print(json.dumps(result), flush=True)
    for label, dtype_name, _tla_dtype, view in CASES:
        if view is None:
            continue
        result = run_exch(torch, from_dlpack, label, getattr(torch, dtype_name), view, args.npu_arch)
        print(json.dumps(result), flush=True)
    print(json.dumps(run_probe(torch, from_dlpack, args.npu_arch)), flush=True)
    for label, dtype_name, view in AIC_ADD_CASES:
        result = run_aic_add(
            torch, from_dlpack, label, getattr(torch, dtype_name), view, args.npu_arch
        )
        print(json.dumps(result), flush=True)
    for label, dtype_name, view in AIC_EXCH_CASES:
        result = run_aic_exch(
            torch, from_dlpack, label, getattr(torch, dtype_name), view, args.npu_arch
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
