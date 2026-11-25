"""tla.full: broadcast a scalar or a one-lane fragment across every supported
AscendC Duplicate element type.

One kernel per dtype fills four UB tiles:

* ``out_scalar`` — ``tla.full(SCALAR)``  no mask, every lane = SCALAR
* ``out_scalar_masked`` — ``tla.full(SCALAR, mask=half)``  first
  ``MASK_LANES`` lanes of each block
* ``out_fragment`` — ``tla.full(reduced)``  fragment source (only for dtypes
  whose ``tla.reduce`` is supported)
* ``out_fragment_masked`` — ``tla.full(reduced, mask=half)``

Bool, hif8, e8m0 and the packed fp4 pairs ride an i8 carrier byte; unsigned
uses the same-width signless carrier; fp8 broadcasts natively; i64/u64 go
through linked bitcode helpers. Host storage for the sub-byte/encoded formats
is a uint8 buffer bound with ``element_type=``.

Each buffer carries a guard region before and after the payload; the guards are
checked afterwards to catch a broadcast or store that runs past the tile.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import catlass.tla as tla

REGISTER_BYTES = 256
GUARD_BYTES = REGISTER_BYTES

# Only these element types have a tla.reduce; the fragment-source path runs
# for them and their two vector outputs are checked.
_VEC_SOURCE_DTYPES = frozenset({"f16", "f32", "i16", "i32", "u16", "u32"})


@dataclass(frozen=True)
class DTypeCase:
    tla_dtype: Any
    torch_dtype: Any
    scalar: Any
    expected: Any
    element_bytes: int
    host_override: bool = False


DTYPE_CASES: dict[str, DTypeCase] = {
    "bool": DTypeCase(tla.Bool, None, True, True, 1),
    "i8": DTypeCase(tla.Int8, "int8", -3, -3, 1),
    "u8": DTypeCase(tla.UInt8, "uint8", 200, 200, 1),
    "i16": DTypeCase(tla.Int16, "int16", -300, -300, 2),
    "u16": DTypeCase(tla.UInt16, "uint16", 60000, 60000, 2),
    "i32": DTypeCase(tla.Int32, "int32", -70000, -70000, 4),
    "u32": DTypeCase(tla.UInt32, "uint32", 4000000000, 4000000000, 4),
    "i64": DTypeCase(tla.Int64, "int64", -(2**40), -(2**40), 8),
    # Torch cannot fill uint64 above 2**63; store the same bits in an int64
    # buffer and retype it through element_type=UInt64.
    "u64": DTypeCase(tla.UInt64, "int64", 2**63 + 5, -(2**63) + 5, 8, True),
    "f16": DTypeCase(tla.Float16, "float16", 1.5, 1.5, 2),
    "bf16": DTypeCase(tla.BFloat16, "bfloat16", 1.5, 1.5, 2),
    "f32": DTypeCase(tla.Float32, "float32", 1.5, 1.5, 4),
    # fp8 1.5 = 0x3C (e4m3fn) / 0x3E (e5m2); the host buffer is uint8.
    "f8e4m3fn": DTypeCase(tla.Float8E4M3FN, "uint8", 1.5, 0x3C, 1, True),
    "f8e5m2": DTypeCase(tla.Float8E5M2, "uint8", 1.5, 0x3E, 1, True),
    # e8m0 2.0 encodes as exponent 1 -> code 128.
    "f8e8m0": DTypeCase(tla.Float8E8M0, "uint8", 2.0, 128, 1, True),
    "hif8": DTypeCase(tla.HiFloat8, "uint8", 0x38, 0x38, 1, True),
    "f4e2m1x2": DTypeCase(tla.Float4E2M1x2, "uint8", 0x21, 0x21, 1, True),
    "f4e1m2x2": DTypeCase(tla.Float4E1M2x2, "uint8", 0x21, 0x21, 1, True),
}
ALL_DTYPES = tuple(DTYPE_CASES)

# --- kernel configuration (set per case before tla.compile) -----------------
_KERNEL_DTYPE = tla.Float32
_SCALAR: Any = 1.5
_VEC_SOURCE = True
ELEMENT_BYTES = 4
LANES = 64
LOOPS = 8
VECTOR_ELE = 512
MASK_LANES = 40


def _element_bytes(dtype_name: str) -> int:
    return DTYPE_CASES[dtype_name].element_bytes


def _configure(dtype_name: str, shape: int) -> DTypeCase:
    global _KERNEL_DTYPE, _SCALAR, _VEC_SOURCE, ELEMENT_BYTES, LANES, LOOPS
    global VECTOR_ELE, MASK_LANES
    if dtype_name not in DTYPE_CASES:
        raise SystemExit(
            f"unsupported --dtype {dtype_name!r}; expected one of: "
            f"{', '.join(ALL_DTYPES)}"
        )
    case = DTYPE_CASES[dtype_name]
    ELEMENT_BYTES = case.element_bytes
    LANES = REGISTER_BYTES // ELEMENT_BYTES
    if shape % LANES:
        raise SystemExit(
            f"full_op.py: {dtype_name} requires --shape to be a multiple of "
            f"{LANES} (one full vector register)"
        )
    VECTOR_ELE = shape
    LOOPS = shape // LANES
    MASK_LANES = LANES // 2
    _KERNEL_DTYPE = case.tla_dtype
    _SCALAR = case.scalar
    _VEC_SOURCE = dtype_name in _VEC_SOURCE_DTYPES
    return case


def _view(mem: Any) -> Any:
    return tla.tile_view(mem, tla.make_shape(VECTOR_ELE), tla.make_coord(0))


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(tensor, tla.make_shape(LANES), tla.make_coord(chunk_idx))


def _make_ub(like: Any) -> Any:
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like, tla.arch.RowMajor)


def _store_chunk(tensor: Any, chunk_idx: Any, value: Any, tail: Any) -> None:
    # A 64-bit store must cover the whole register, and the tutorial shape is a
    # lane multiple, so the last chunk is full and needs no tail predicate.
    tile = _chunk(tensor, chunk_idx)
    if _KERNEL_DTYPE in (tla.Int64, tla.UInt64):
        tile.store(value)
    else:
        tile.store(value, mask=tail)


@tla.kernel
def full_op(
    mem_x: tla.Tensor,
    mem_scalar: tla.Tensor,
    mem_scalar_masked: tla.Tensor,
    mem_fragment: tla.Tensor,
    mem_fragment_masked: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("full_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("full_vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    x_ub = _make_ub(mem_x)
    scalar_ub = _make_ub(mem_scalar)
    scalar_masked_ub = _make_ub(mem_scalar_masked)
    fragment_ub = _make_ub(mem_fragment)
    fragment_masked_ub = _make_ub(mem_fragment_masked)

    with tla.vector():
        tla.copy(x_ub, _view(mem_x))
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            half, _ = tla.update_mask(MASK_LANES, _KERNEL_DTYPE)
            if _VEC_SOURCE:
                all_lanes, _ = tla.update_mask(LANES, _KERNEL_DTYPE)
            remaining = VECTOR_ELE
            for i in tla.range(LOOPS):
                tail, remaining = tla.update_mask(remaining, _KERNEL_DTYPE)
                _store_chunk(scalar_ub, i, tla.full(_SCALAR, _KERNEL_DTYPE), tail)
                _store_chunk(
                    scalar_masked_ub,
                    i,
                    tla.full(_SCALAR, _KERNEL_DTYPE, mask=half),
                    tail,
                )
                if _VEC_SOURCE:
                    reduced = (
                        _chunk(x_ub, i)
                        .load()
                        .reduce(tla.ReductionOp.ADD, mask=all_lanes)
                    )
                    _store_chunk(fragment_ub, i, tla.full(reduced, _KERNEL_DTYPE), tail)
                    _store_chunk(
                        fragment_masked_ub,
                        i,
                        tla.full(reduced, _KERNEL_DTYPE, mask=half),
                        tail,
                    )

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(_view(mem_scalar), scalar_ub)
        tla.copy(_view(mem_scalar_masked), scalar_masked_ub)
        if _VEC_SOURCE:
            tla.copy(_view(mem_fragment), fragment_ub)
            tla.copy(_view(mem_fragment_masked), fragment_masked_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _host_dtype(case: DTypeCase, torch: Any) -> Any:
    return torch.bool if case.torch_dtype is None else getattr(torch, case.torch_dtype)


def _bind(buf: Any, case: DTypeCase, elems: int) -> Any:
    from catlass.tla.runtime import from_dlpack

    kwargs: dict[str, Any] = {
        "layout_tag": tla.arch.RowMajor,
        "origin_shape": (elems,),
    }
    if case.host_override:
        kwargs["element_type"] = case.tla_dtype
    return from_dlpack(buf, **kwargs)


def _make_buffer(torch: Any, case: DTypeCase, value: Any, elems: int) -> Any:
    torch_dtype = _host_dtype(case, torch)
    return torch.full((elems,), value, dtype=torch_dtype)


def _guard_value(case: DTypeCase, torch: Any) -> Any:
    if case.torch_dtype is None:
        return True
    if case.host_override or case.torch_dtype == "uint8":
        return 0xA5
    return -999 if case.torch_dtype in ("float16", "float32", "bfloat16") else 7


def _check_expected(
    case: DTypeCase, actual: Any, expected: Any, torch: Any, atol: float
) -> bool:
    act = actual.cpu()
    if case.torch_dtype is None:
        return bool(act.all()) if expected is True else bool((~act).all())
    if isinstance(expected, float):
        return bool(
            torch.isclose(
                act, torch.full_like(act, expected), rtol=0.0, atol=atol
            ).all()
        )
    return bool(act.eq(expected).all())


def _run_case(
    dtype_name: str, shape: int, device: int, block_num: int, torch: Any
) -> bool:
    case = _configure(dtype_name, shape)
    guard = _guard_value(case, torch)

    elems = shape
    guard_elems = GUARD_BYTES // case.element_bytes
    total = elems + 2 * guard_elems

    def build(payload: Any):
        buf = _make_buffer(torch, case, guard, total)
        buf[guard_elems : guard_elems + elems] = payload
        return buf.npu()

    # x holds ones so the reduction is exactly one register's element count.
    x = build(1)
    outs = {
        name: build(0)
        for name in (
            "out_scalar",
            "out_scalar_masked",
            "out_fragment",
            "out_fragment_masked",
        )
    }

    def bind(buf):
        payload = buf.view(-1)[guard_elems : guard_elems + elems]
        return _bind(payload, case, elems)

    tla_args = [bind(x)] + [
        bind(outs[name])
        for name in (
            "out_scalar",
            "out_scalar_masked",
            "out_fragment",
            "out_fragment_masked",
        )
    ]
    torch.npu.synchronize()

    artifact = tla.compile(full_op, *tla_args, options="--npu-arch 3510")
    artifact(*tla_args, block_num=block_num)
    torch.npu.synchronize()

    ok = True
    ok &= _check_expected(
        case, outs["out_scalar"][guard_elems:-guard_elems], case.expected, torch, 1e-3
    )
    masked = outs["out_scalar_masked"][guard_elems:-guard_elems].reshape(LOOPS, LANES)[
        :, :MASK_LANES
    ]
    ok &= _check_expected(case, masked, case.expected, torch, 1e-3)
    if _VEC_SOURCE:
        ok &= _check_expected(
            case, outs["out_fragment"][guard_elems:-guard_elems], LANES, torch, 1e-3
        )
        masked_vec = outs["out_fragment_masked"][guard_elems:-guard_elems].reshape(
            LOOPS, LANES
        )[:, :MASK_LANES]
        ok &= _check_expected(case, masked_vec, LANES, torch, 1e-3)
    for name, buf in outs.items():
        cpu_buf = buf.cpu()
        guards_ok = bool(cpu_buf[:guard_elems].eq(guard).all()) and bool(
            cpu_buf[-guard_elems:].eq(guard).all()
        )
        if not guards_ok:
            print(f"full_op: {dtype_name} guard bytes changed around output {name}")
            ok = False

    print(
        f"{'PASS' if ok else 'FAIL'} dtype={dtype_name} shape={shape} "
        f"scalar={case.scalar} head={outs['out_scalar'][guard_elems : guard_elems + 4].cpu().tolist()}"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    parser.add_argument("--dtype", default="f32", choices=ALL_DTYPES)
    parser.add_argument(
        "--all-dtypes",
        action="store_true",
        help="run every supported element type (overrides --dtype)",
    )
    parser.add_argument("--shape", type=int, default=512)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    dtypes = ALL_DTYPES if args.all_dtypes else (args.dtype,)
    passed = 0
    failed = 0
    for dtype_name in dtypes:
        try:
            if _run_case(dtype_name, args.shape, args.device, args.block_num, torch):
                passed += 1
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL dtype={dtype_name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"tla.full dtypes: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
