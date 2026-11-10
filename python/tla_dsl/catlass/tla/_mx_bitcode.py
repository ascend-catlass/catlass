"""Tensor-level bitcode dispatch for the experimental MX conversion API."""

from __future__ import annotations

from .. import runtime as _runtime
from ..address_space import AddressSpace
from ..base_dsl.typing import Int8, Int32, Numeric
from ..core_api import (
    _as_value,
    _MaskPattern,
    _op_error,
    _tla_ops_gen,
    allocate,
    recast_ptr,
)
from .typing import Tensor


def _is_mx_tensor_pair(value):
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(part, Tensor) for part in value)
    )


_FORMATS = {
    "f4e2m1": "e2m1",
    "f4e1m2": "e1m2",
    "f8e4m3fn": "e4m3",
    "f8e5m2": "e5m2",
}
_DENSE = {"f32": ("f32", 4), "f16": ("f16", 2), "bf16": ("bf16", 2)}
_PREFIX_MASKS = {"ALLF": 0, **{f"VL{n}": n for n in (1, 2, 3, 4, 8, 16, 32, 64, 128)}}


def _backing_pointer(tensor):
    """Find the original pointer for direct tensor construction and recasts."""
    value = tensor.value
    while getattr(getattr(value, "owner", None), "name", None) in (
        "tla.make_tensor",
        "tla.make_tensor_like",
        "tla.tensor_ptr",
        "tla.recast_ptr",
    ):
        value = value.owner.operands[0]
        value = getattr(value, "value", value)
    return value


def _allocation_bytes(tensor):
    value = _backing_pointer(tensor)
    owner = getattr(value, "owner", None)
    if getattr(owner, "name", None) == "tla.alloc_ptr":
        return int(owner.attributes["size_bytes"].value)
    return None


def _static_layout(tensor: Tensor, *, dynamic_coord=False, dynamic_shape=False):
    shape, stride, coord = tensor.shape, tensor.stride, tensor.coord
    if not all(isinstance(tree, tuple) for tree in (shape, stride, coord)):
        return None
    if (
        len(shape) not in (1, 2)
        or len(stride) != len(shape)
        or len(coord) != len(shape)
    ):
        return None
    values = shape + stride + coord
    if not all(isinstance(x, (int, Numeric)) for x in values):
        return None
    if not dynamic_shape and not all(isinstance(x, int) for x in shape + stride):
        return None
    if not dynamic_coord and not all(isinstance(x, int) for x in coord):
        return None
    if any(x < 0 for x in values if isinstance(x, int)):
        return None
    return shape, stride, coord


def _statically_different(left, right):
    return len(left) != len(right) or any(
        isinstance(a, int) and isinstance(b, int) and a != b
        for a, b in zip(left, right, strict=True)
    )


def _payload_layout(tensor: Tensor, group_axis: int):
    layout = _static_layout(tensor, dynamic_coord=True, dynamic_shape=True)
    if layout is None or tensor.addrspace != "ub":
        return None
    shape, stride, coord = _orient_layout(layout, group_axis)
    if not isinstance(stride[-1], int) or stride[-1] != 1:
        return None
    dtype = tensor.dtype
    packed = dtype in ("f4e2m1", "f4e1m2")
    width = _DENSE[dtype][1] if dtype in _DENSE else 1
    row_stride = stride[-2] if len(shape) > 1 else shape[0]
    if packed:
        if (isinstance(coord[-1], int) and coord[-1] % 2) or (
            len(shape) > 1 and isinstance(row_stride, int) and row_stride % 2
        ):
            return None
        # Dynamic coordinates must satisfy the documented byte alignment.
        leading = zip(coord[:-1], stride[:-1], strict=True)
        offset = coord[-1] // 2 + sum(c * (s // 2) for c, s in leading)
        row_stride = (row_stride + 1) // 2
    else:
        offset = sum(c * s for c, s in zip(coord, stride, strict=True))
        offset *= width
        row_stride *= width
    return shape, coord, row_stride, offset


def _scale_layout(scale: Tensor, shape: tuple[int, ...], group_axis: int):
    layout = _static_layout(scale, dynamic_coord=True, dynamic_shape=True)
    if layout is None or scale.addrspace != "ub" or scale.dtype not in ("i8", "f8e8m0"):
        return None
    extent, stride, coord = _orient_layout(layout, group_axis)
    if len(extent) != len(shape):
        return None
    if (
        isinstance(extent[-1], int)
        and isinstance(shape[-1], int)
        and extent[-1] < (shape[-1] + 31) // 32
    ) or _statically_different(extent[:-1], shape[:-1]):
        return None
    return (
        stride[-2] if len(shape) > 1 else extent[0],
        sum(c * s for c, s in zip(coord, stride, strict=True)),
        stride[-1],
    )


def validate_mx_tensor_arguments(src, dst):
    """Reject statically provable dtype, memory, and shape mismatches."""
    operands = (("src", src), ("dst", dst))
    payloads = []
    for name, value in operands:
        is_mx = _is_mx_tensor_pair(value)
        payload = value[0] if is_mx else value
        payloads.append(payload)
        if payload.addrspace != "ub":
            _op_error("mx_to", f"{name} payload must be in UB, got {payload.addrspace}")
        allowed = _FORMATS if is_mx else _DENSE
        if payload.dtype not in allowed:
            _op_error("mx_to", f"{name} payload has unsupported dtype {payload.dtype}")
        if not is_mx:
            continue
        scale = value[1]
        if scale.addrspace != "ub" or scale.dtype not in ("i8", "f8e8m0"):
            _op_error("mx_to", f"{name} scale must be an i8 or f8e8m0 UB tensor")
        # Check the logical extents before choosing any fast path. A padded
        # scale may cover both axes, in which case choosing either is unsafe.
        pd, sd = _layout_descriptor(payload), _layout_descriptor(scale)
        if pd is not None and sd is not None and len(payload.shape) == 2:
            ps, ss = pd[0], sd[0]
            if all(isinstance(x, int) for x in (*ps, *ss)):
                candidates = [
                    axis
                    for axis in (0, 1)
                    if ss[axis] >= (ps[axis] + 31) // 32
                    and ss[1 - axis] == ps[1 - axis]
                ]
                if len(candidates) > 1:
                    _op_error(
                        "mx_to",
                        f"{name} scale shape {ss} is ambiguous for payload shape {ps}; use unambiguous scale extents",
                    )
            elif _layout_scale_axis(ss, ps) is None:
                _op_error(
                    "mx_to",
                    f"{name} scale grouping axis cannot be determined unambiguously from runtime-shaped tensors",
                )
        pl, sc = (
            _static_layout(payload, dynamic_coord=True, dynamic_shape=True),
            _static_layout(scale, dynamic_coord=True, dynamic_shape=True),
        )
        if pl is None or sc is None:
            continue
        ps, ss = pl[0], sc[0]
        if len(ps) != len(ss):
            _op_error("mx_to", f"{name} scale rank must match payload rank")
        if all(isinstance(x, int) for x in (*ps, *ss)) and not any(
            ss[axis] >= (ps[axis] + 31) // 32
            and all(ss[i] == ps[i] for i in range(len(ps)) if i != axis)
            for axis in range(len(ps))
        ):
            _op_error(
                "mx_to",
                f"{name} scale shape {ss} does not cover payload shape {ps} in groups of 32",
            )
    a, b = (
        _static_layout(payload, dynamic_coord=True, dynamic_shape=True)
        for payload in payloads
    )
    if a is not None and b is not None and _statically_different(a[0], b[0]):
        _op_error(
            "mx_to",
            f"source and destination payload shapes must match, got {a[0]} and {b[0]}",
        )
    if _backing_pointer(payloads[0]) == _backing_pointer(payloads[1]):
        _op_error("mx_to", "source and destination payloads must not overlap")
    for value in (src, dst):
        if _is_mx_tensor_pair(value):
            scale_pointer = _backing_pointer(value[1])
            if any(scale_pointer == _backing_pointer(payload) for payload in payloads):
                _op_error("mx_to", "payload and scale storage must not overlap")


def _byte_ptr(tensor: Tensor, offset):
    ptr = recast_ptr(tensor.ptr, dtype=Int8)
    return ptr if isinstance(offset, int) and offset == 0 else ptr + offset


def _emit_bitcode(symbol, pointers, scalars, loc):
    """Emit the common pointer-and-i32 ABI used by the MX bitcode helpers."""
    _tla_ops_gen.call_extern(
        symbol,
        [_as_value(ptr) for ptr in pointers]
        + [Int32(value).ir_value(loc=loc) for value in scalars],
        loc=loc,
    )


def _orient_layout(layout, axis):
    """Make the grouping axis the last axis of a flat layout."""
    if axis == 0 and len(layout[0]) == 2:
        return tuple(tuple(reversed(part)) for part in layout)
    return layout


def _rows_cols(shape):
    return (shape[0], shape[1]) if len(shape) == 2 else (1, shape[0])


def _try_dense_gather_scatter(src, dst, params, mask_limit, loc, *, quant):
    """Convert between a strided dense tile and a contiguous MX payload."""
    dense, mx = (src, dst) if quant else (dst, src)
    for axis in (-1, 0):
        payload = _payload_layout(mx[0], axis)
        raw = _static_layout(dense, dynamic_coord=True, dynamic_shape=True)
        if payload is None or raw is None:
            continue
        if axis == 0 and len(raw[0]) != 2:
            continue
        shape, stride, coord = _orient_layout(raw, axis)
        if _statically_different(shape, payload[0]):
            continue
        step = stride[-1]
        width = _DENSE[dense.dtype][1]
        if not isinstance(step, int) or step <= 0 or (width == 2 and step * 31 > 65535):
            continue
        scale = _scale_layout(mx[1], shape, axis)
        if scale is None or (isinstance(payload[1][-1], int) and payload[1][-1] % 32):
            continue
        rows, cols = _rows_cols(shape)
        offset = sum(c * s for c, s in zip(coord, stride, strict=True)) * width
        row_stride = (stride[-2] if len(shape) == 2 else shape[0] * step) * width
        scratch = allocate(256, Int8, AddressSpace.ub, 256, loc=loc)
        if quant:
            args = [
                _byte_ptr(dense, offset),
                _byte_ptr(mx[0], payload[3]),
                _byte_ptr(mx[1], scale[1]),
                scratch,
            ]
            prefix = (rows, cols, row_stride, step, payload[2], scale[0], scale[2])
            symbol = (
                f"tla_mx_gather_quant_{_DENSE[dense.dtype][0]}_{_FORMATS[mx[0].dtype]}"
            )
        else:
            args = [
                _byte_ptr(mx[0], payload[3]),
                _byte_ptr(mx[1], scale[1]),
                _byte_ptr(dense, offset),
                scratch,
            ]
            prefix = (rows, cols, payload[2], scale[0], scale[2], row_stride, step)
            symbol = f"tla_mx_scatter_dequant_{_FORMATS[mx[0].dtype]}_{_DENSE[dense.dtype][0]}"
        values = prefix + (
            params.codes()[2],
            rows * cols if mask_limit is None else mask_limit,
        )
        _emit_bitcode(symbol, args, values, loc)
        return True
    return False


def _try_mx_gather_scatter(src, dst, params, mask_limit, loc, *, gather):
    """Convert MX tiles when exactly one payload has a strided group axis."""
    contiguous, strided = (dst[0], src[0]) if gather else (src[0], dst[0])
    packed = strided.dtype in ("f4e2m1", "f4e1m2")
    for axis in (-1, 0):
        payload = _payload_layout(contiguous, axis)
        raw = _static_layout(strided, dynamic_coord=True, dynamic_shape=True)
        if payload is None or raw is None:
            continue
        if axis == 0 and len(raw[0]) != 2:
            continue
        shape, stride, coord = _orient_layout(raw, axis)
        if _statically_different(shape, payload[0]):
            continue
        step = stride[-1]
        if (
            not isinstance(step, int)
            or step <= int(packed and not gather)
            or step * 31 + int(packed) > 65535
        ):
            continue
        source_scale = _scale_layout(src[1], shape, axis)
        dest_scale = _scale_layout(dst[1], shape, axis)
        if (
            source_scale is None
            or dest_scale is None
            or any(isinstance(c, int) and c % 32 for c in (coord[-1], payload[1][-1]))
        ):
            continue
        rows, cols = _rows_cols(shape)
        offset = sum(c * s for c, s in zip(coord, stride, strict=True))
        row_stride = stride[-2] if len(shape) == 2 else shape[0] * step
        scratch = allocate(256, Int8, AddressSpace.ub, 256, loc=loc)
        if gather:
            args = [
                _byte_ptr(src[0], 0 if packed else offset),
                _byte_ptr(src[1], source_scale[1]),
                _byte_ptr(dst[0], payload[3]),
                _byte_ptr(dst[1], dest_scale[1]),
                scratch,
            ]
            prefix = (
                (rows, cols, row_stride, step, offset)
                if packed
                else (rows, cols, row_stride, step)
            )
            values = prefix + (
                source_scale[0],
                source_scale[2],
                payload[2],
                dest_scale[0],
                dest_scale[2],
            )
            symbol = "tla_mx_gather_fp4" if packed else "tla_mx_gather_requant"
        else:
            args = [
                _byte_ptr(src[0], payload[3]),
                _byte_ptr(src[1], source_scale[1]),
                _byte_ptr(dst[0], 0 if packed else offset),
                _byte_ptr(dst[1], dest_scale[1]),
                scratch,
            ]
            prefix = (
                rows,
                cols,
                payload[2],
                source_scale[0],
                source_scale[2],
                row_stride,
                step,
            )
            if packed:
                prefix += (offset,)
            values = prefix + (dest_scale[0], dest_scale[2])
            symbol = "tla_mx_scatter_fp4" if packed else "tla_mx_scatter_mx"
        values += (params.codes()[2], rows * cols if mask_limit is None else mask_limit)
        _emit_bitcode(
            f"{symbol}_{_FORMATS[src[0].dtype]}_{_FORMATS[dst[0].dtype]}",
            args,
            values,
            loc,
        )
        return True
    return False


def _try_strided(src, dst, params, mask_limit, loc):
    """Convert when both payloads are strided along the grouping axis."""
    src_mx, dst_mx = _is_mx_tensor_pair(src), _is_mx_tensor_pair(dst)
    sp, dp = (src[0] if src_mx else src), (dst[0] if dst_mx else dst)
    source = _static_layout(sp, dynamic_coord=True, dynamic_shape=True)
    dest = _static_layout(dp, dynamic_coord=True, dynamic_shape=True)
    if source is None or dest is None:
        return False
    for axis in (-1, 0):
        if axis == 0 and len(source[0]) != 2:
            continue
        shape, src_stride, src_coord = _orient_layout(source, axis)
        dst_shape, dst_stride, dst_coord = _orient_layout(dest, axis)
        if _statically_different(shape, dst_shape):
            continue
        sx, dx = src_stride[-1], dst_stride[-1]
        dp4 = dst_mx and dp.dtype in ("f4e2m1", "f4e1m2")
        if (
            not all(isinstance(x, int) for x in (sx, dx))
            or sx <= 0
            or dx <= int(dp4)
            or sx * 31 + 1 > 65535
            or dx * 31 + 1 > 65535
        ):
            continue
        if src_mx and dst_mx:
            ss, ds = (
                _scale_layout(src[1], shape, axis),
                _scale_layout(dst[1], shape, axis),
            )
            if (
                ss is None
                or ds is None
                or any(
                    isinstance(c, int) and c % 32
                    for c in (src_coord[-1], dst_coord[-1])
                )
            ):
                continue
        else:
            mx = src if src_mx else dst
            scale = _scale_layout(mx[1], shape, axis)
            mx_coord = src_coord if src_mx else dst_coord
            if scale is None or (isinstance(mx_coord[-1], int) and mx_coord[-1] % 32):
                continue
        rows, cols = _rows_cols(shape)
        so = sum(c * st for c, st in zip(src_coord, src_stride, strict=True))
        do = sum(c * st for c, st in zip(dst_coord, dst_stride, strict=True))
        sr = src_stride[-2] if len(shape) == 2 else shape[0] * sx
        dr = dst_stride[-2] if len(shape) == 2 else shape[0] * dx
        scratch = allocate(512, Int8, AddressSpace.ub, 256, loc=loc)
        if src_mx and dst_mx:
            args = [
                _byte_ptr(sp, 0),
                _byte_ptr(src[1], ss[1]),
                _byte_ptr(dp, 0),
                _byte_ptr(dst[1], ds[1]),
                scratch,
            ]
            values = (
                rows,
                cols,
                sr,
                sx,
                so,
                ss[0],
                ss[2],
                dr,
                dx,
                do,
                ds[0],
                ds[2],
                params.codes()[2],
                rows * cols if mask_limit is None else mask_limit,
            )
            symbol = f"tla_mx_strided_{_FORMATS[sp.dtype]}_{_FORMATS[dp.dtype]}"
        else:
            args = [
                _byte_ptr(sp, 0),
                _byte_ptr(dp, 0),
                _byte_ptr(mx[1], scale[1]),
                scratch,
            ]
            values = (
                rows,
                cols,
                sr,
                sx,
                so,
                dr,
                dx,
                do,
                scale[0],
                scale[2],
                params.codes()[2],
                rows * cols if mask_limit is None else mask_limit,
            )
            kind = "dequant" if src_mx else "quant"
            sn = _FORMATS[sp.dtype] if src_mx else _DENSE[sp.dtype][0]
            dn = _FORMATS[dp.dtype] if dst_mx else _DENSE[dp.dtype][0]
            symbol = f"tla_mx_strided_{kind}_{sn}_{dn}"
        _emit_bitcode(symbol, args, values, loc)
        return True
    return False


def _try_regroup(src, dst, params, mask_limit, loc):
    source, dest, scale_raw = (
        _static_layout(x, dynamic_coord=True, dynamic_shape=True)
        for x in (src[0], dst[0], src[1])
    )
    if source is None or dest is None or scale_raw is None or len(source[0]) != 2:
        return False
    for axis in (-1, 0):
        shape, ss, sc = _orient_layout(source, axis)
        dshape, ds, dc = _orient_layout(dest, axis)
        qshape, qs, qc = _orient_layout(scale_raw, axis)
        if _statically_different(shape, dshape):
            continue
        dest_scale = _scale_layout(dst[1], shape, axis)
        if dest_scale is None or _statically_different(qshape[1:], shape[1:]):
            continue
        if (
            isinstance(qshape[0], int)
            and isinstance(shape[0], int)
            and qshape[0] < (shape[0] + 31) // 32
        ):
            continue
        sx, dx, qstep = ss[-1], ds[-1], qs[-1]
        if (
            not all(isinstance(x, int) for x in (sx, dx, qstep))
            or sx <= 0
            or dx <= 0
            or qstep < 0
            or max(sx, dx, qstep) * 31 + 1 > 65535
        ):
            continue
        if any(isinstance(c, int) and c % 32 for c in (sc[0], dc[1])):
            continue
        so = sum(c * st for c, st in zip(sc, ss, strict=True))
        do = sum(c * st for c, st in zip(dc, ds, strict=True))
        dp4 = dst[0].dtype in ("f4e2m1", "f4e1m2")
        if (
            dp4
            and dx == 1
            and (
                (isinstance(do, int) and do % 2)
                or (isinstance(ds[0], int) and ds[0] % 2)
            )
        ):
            continue
        scale_offset = sum(c * st for c, st in zip(qc, qs, strict=True))
        rows, cols = shape
        if isinstance(rows, int) and isinstance(cols, int):
            scale_capacity = rows * ((cols + 31) // 32)
        else:
            scale_capacity = _allocation_bytes(dst[1])
            if scale_capacity is None:
                continue
        scratch = allocate(768 + scale_capacity, Int8, AddressSpace.ub, 256, loc=loc)
        args = [
            _byte_ptr(src[0], 0),
            _byte_ptr(src[1], scale_offset),
            _byte_ptr(dst[0], 0),
            _byte_ptr(dst[1], dest_scale[1]),
            scratch,
        ]
        values = (
            rows,
            cols,
            ss[0],
            sx,
            so,
            qs[0],
            qstep,
            ds[0],
            dx,
            do,
            dest_scale[0],
            dest_scale[2],
            params.codes()[2],
            rows * cols if mask_limit is None else mask_limit,
        )
        _emit_bitcode(
            f"tla_mx_regroup_{_FORMATS[src[0].dtype]}_{_FORMATS[dst[0].dtype]}",
            args,
            values,
            loc,
        )
        return True
    return False


def _layout_descriptor(tensor):
    shape, stride, coord = tensor.shape, tensor.stride, tensor.coord
    parts = (shape, stride, coord)
    if not all(isinstance(x, tuple) and len(x) in (1, 2) for x in parts):
        return None
    if not (len(shape) == len(stride) == len(coord)):
        return None
    if len(shape) == 1:
        if not all(isinstance(x, (int, Numeric)) for x in shape + stride + coord):
            return None
        if any(isinstance(x, int) and x < 0 for x in shape + stride + coord):
            return None
        return (1, shape[0]), (1, 1, 0, 0, 0, stride[0], 0, coord[0])
    if not all(isinstance(c, (int, Numeric)) for c in coord):
        return None
    nested = all(isinstance(x, tuple) and len(x) == 2 for x in shape + stride)
    if nested:
        rb, cb = shape[0][0], shape[1][0]
        if not isinstance(rb, int) or not isinstance(cb, int) or rb <= 0 or cb <= 0:
            return None
        r0, r1 = stride[0]
        c0, c1 = stride[1]
        extent = tensor.origin_shape
    else:
        if not all(isinstance(x, (int, Numeric)) for x in shape + stride):
            return None
        rb, cb, r0, c0 = 1, 1, 0, 0
        r1, c1 = stride
        extent = shape
    if (
        not isinstance(extent, tuple)
        or len(extent) != 2
        or not all(isinstance(x, (int, Numeric)) for x in extent + (r0, r1, c0, c1))
    ):
        return None
    if any(isinstance(x, int) and x < 0 for x in extent + coord + (r0, r1, c0, c1)):
        return None
    return extent, (rb, cb, r0, r1, c0, c1, *coord)


def _layout_scale_axis(scale_shape, payload_shape):
    if not all(isinstance(x, int) for x in scale_shape + payload_shape):
        # Runtime dimensions still have an unambiguous axis if the other
        # possibility is statically impossible. Runtime group extents are
        # required to contain at least one complete 32-element group.
        candidates = []
        for axis in (1, 0):
            grouped = payload_shape[axis]
            other_payload, other_scale = payload_shape[1 - axis], scale_shape[1 - axis]
            if isinstance(grouped, int) and grouped < 32:
                continue
            if (
                isinstance(other_payload, int)
                and isinstance(other_scale, int)
                and other_payload != other_scale
            ):
                continue
            if (
                isinstance(grouped, int)
                and isinstance(scale_shape[axis], int)
                and scale_shape[axis] < (grouped + 31) // 32
            ):
                continue
            candidates.append(axis)
        return candidates[0] if len(candidates) == 1 else None
    candidates = [
        axis
        for axis in (1, 0)
        if (
            scale_shape[axis] >= (payload_shape[axis] + 31) // 32
            and scale_shape[1 - axis] == payload_shape[1 - axis]
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


def _try_general_layout(src, dst, params, mask_limit, loc):
    src_mx, dst_mx = _is_mx_tensor_pair(src), _is_mx_tensor_pair(dst)
    sp, dp = (src[0] if src_mx else src), (dst[0] if dst_mx else dst)
    a, b = _layout_descriptor(sp), _layout_descriptor(dp)
    if (
        a is None
        or b is None
        or len(sp.shape) != len(dp.shape)
        or _statically_different(a[0], b[0])
    ):
        return False
    shape = a[0]
    q = _layout_descriptor(src[1]) if src_mx else None
    p = _layout_descriptor(dst[1]) if dst_mx else None
    if (src_mx and q is None) or (dst_mx and p is None):
        return False

    def scale_axis(scale, descriptor):
        if len(scale.shape) != len(sp.shape):
            return None
        if len(scale.shape) == 1:
            count, width = descriptor[0][1], shape[1]
            if (
                isinstance(count, int)
                and isinstance(width, int)
                and count < (width + 31) // 32
            ):
                return None
            return 1
        return _layout_scale_axis(descriptor[0], shape)

    source_axis = scale_axis(src[1], q) if src_mx else None
    dest_axis = scale_axis(dst[1], p) if dst_mx else source_axis
    if (src_mx and source_axis is None) or dest_axis is None:
        return False
    if not src_mx:
        source_axis = dest_axis
    for tensor, axis in (
        (sp, source_axis if src_mx else None),
        (dp, dest_axis if dst_mx else None),
    ):
        coord = (
            tensor.coord[0]
            if len(tensor.coord) == 1
            else tensor.coord[axis]
            if axis is not None
            else 0
        )
        if axis is not None and isinstance(coord, int) and coord % 32:
            return False
    rows, cols = (shape[0], shape[1]) if dest_axis == 1 else (shape[1], shape[0])
    if dst_mx:
        if isinstance(rows, int) and isinstance(cols, int):
            scale_capacity = rows * ((cols + 31) // 32)
        else:
            scale_capacity = _allocation_bytes(dst[1])
            if scale_capacity is None:
                return False
    else:
        scale_capacity = 0
    scratch = allocate(1792 + scale_capacity, Int8, AddressSpace.ub, 256, loc=loc)
    values = (
        rows,
        cols,
        source_axis,
        dest_axis,
        params.codes()[2],
        rows * cols if mask_limit is None else mask_limit,
    )
    args = [
        _byte_ptr(sp, 0),
        _byte_ptr(src[1], 0) if src_mx else _byte_ptr(sp, 0),
        _byte_ptr(dp, 0),
        _byte_ptr(dst[1], 0) if dst_mx else _byte_ptr(dp, 0),
        scratch,
    ]
    descriptor = (
        a[1]
        + b[1]
        + (q[1] if q else (1, 1, 0, 0, 0, 0, 0, 0))
        + (p[1] if p else (1, 1, 0, 0, 0, 0, 0, 0))
    )
    kind = "requant" if src_mx and dst_mx else "dequant" if src_mx else "quant"
    sn = _FORMATS[sp.dtype] if src_mx else _DENSE[sp.dtype][0]
    dn = _FORMATS[dp.dtype] if dst_mx else _DENSE[dp.dtype][0]
    _emit_bitcode(f"tla_mx_layout_{kind}_{sn}_{dn}", args, values + descriptor, loc)
    return True


def try_mx_tensor_bitcode(src, dst, params, mask, *, loc=None) -> bool:
    """Emit a whole-tile bitcode call for a supported contiguous layout.

    Return False when the layout is not yet supported by the bitcode path.
    """
    mask_token = mask._token if isinstance(mask, _MaskPattern) else mask
    if _runtime._has_enclosing_region("vec.func"):
        return False
    src_mx, dst_mx = _is_mx_tensor_pair(src), _is_mx_tensor_pair(dst)
    prefix_mask = _PREFIX_MASKS.get(mask_token) if isinstance(mask_token, str) else None
    is_all = isinstance(mask_token, str) and mask_token == "ALL"
    if mask is not None and not is_all and prefix_mask is None:
        return False
    src_payload = src[0] if src_mx else src
    dst_payload = dst[0] if dst_mx else dst
    if (src_mx and src_payload.dtype not in _FORMATS) or (
        dst_mx and dst_payload.dtype not in _FORMATS
    ):
        return False
    if (not src_mx and src_payload.dtype not in _DENSE) or (
        not dst_mx and dst_payload.dtype not in _DENSE
    ):
        return False
    if src_mx and dst_mx:
        # A different scale mapping may alias the source. The general path
        # writes its output scales only after all input scales have been read.
        source_desc, dest_desc = _layout_descriptor(src[1]), _layout_descriptor(dst[1])
        if source_desc is not None and dest_desc is not None:
            s, d = source_desc[1], dest_desc[1]
            same_mapping = all(
                isinstance(x, int) and isinstance(y, int) and x == y
                for x, y in zip(s, d, strict=True)
            )
            payload_desc = _layout_descriptor(src_payload)
            src_axis = (
                _layout_scale_axis(source_desc[0], payload_desc[0])
                if payload_desc
                else None
            )
            dst_axis = (
                _layout_scale_axis(dest_desc[0], payload_desc[0])
                if payload_desc
                else None
            )
            source_base, dest_base = _backing_pointer(src[1]), _backing_pointer(dst[1])
            separate_allocations = (
                source_base != dest_base
                and getattr(getattr(source_base, "owner", None), "name", None)
                == "tla.alloc_ptr"
                and getattr(getattr(dest_base, "owner", None), "name", None)
                == "tla.alloc_ptr"
            )
            if (
                not separate_allocations
                and not same_mapping
                and (src_axis is None or dst_axis is None or src_axis == dst_axis)
            ):
                return _try_general_layout(src, dst, params, prefix_mask, loc)
    # Normalize either contiguous axis to the inner dimension of the bitcode
    # loop. Both payloads and their scales must use that same group axis.
    for group_axis in (-1, 0):
        sl, dl = (
            _payload_layout(src_payload, group_axis),
            _payload_layout(dst_payload, group_axis),
        )
        if sl is None or dl is None or _statically_different(sl[0], dl[0]):
            continue
        if group_axis == 0 and len(sl[0]) != 2:
            continue
        shape = sl[0]
        source_scale = _scale_layout(src[1], shape, group_axis) if src_mx else None
        dest_scale = _scale_layout(dst[1], shape, group_axis) if dst_mx else None
        if (
            src_mx
            and (
                source_scale is None or (isinstance(sl[1][-1], int) and sl[1][-1] % 32)
            )
        ) or (
            dst_mx
            and (dest_scale is None or (isinstance(dl[1][-1], int) and dl[1][-1] % 32))
        ):
            continue
        break
    else:
        if not src_mx and dst_mx:
            return (
                _try_dense_gather_scatter(
                    src, dst, params, prefix_mask, loc, quant=True
                )
                or _try_strided(src, dst, params, prefix_mask, loc)
                or _try_general_layout(src, dst, params, prefix_mask, loc)
            )
        if src_mx and not dst_mx:
            return (
                _try_dense_gather_scatter(
                    src, dst, params, prefix_mask, loc, quant=False
                )
                or _try_strided(src, dst, params, prefix_mask, loc)
                or _try_general_layout(src, dst, params, prefix_mask, loc)
            )
        if src_mx and dst_mx:
            return (
                _try_mx_gather_scatter(src, dst, params, prefix_mask, loc, gather=True)
                or _try_mx_gather_scatter(
                    src, dst, params, prefix_mask, loc, gather=False
                )
                or _try_strided(src, dst, params, prefix_mask, loc)
                or _try_regroup(src, dst, params, prefix_mask, loc)
                or _try_general_layout(src, dst, params, prefix_mask, loc)
            )
        return False
    rows, cols = _rows_cols(shape)
    round_mode = params.codes()[2]
    # The meta-ops implement CAST_ROUND (0), the three directed modes (1-3) and,
    # by leaving the hardware cast alone, CAST_NONE (4) and CAST_RINT (5).
    # CAST_ODD (6) has no implementation, so refuse it rather than silently
    # returning one of the others.
    if round_mode > 5:
        return False
    if dst_mx and dst_payload.dtype in ("f4e2m1", "f4e1m2") and round_mode != 0:
        return False
    a, b = _byte_ptr(src_payload, sl[3]), _byte_ptr(dst_payload, dl[3])
    source_step = source_scale[2] if src_mx else 1
    dest_step = dest_scale[2] if dst_mx else 1
    if not (
        isinstance(source_step, int)
        and source_step == 1
        and isinstance(dest_step, int)
        and dest_step == 1
    ):
        if src_mx and dst_mx:
            kind = (
                "widen"
                if src_payload.dtype.startswith("f4")
                and dst_payload.dtype.startswith("f8")
                else "requant"
            )
            src_name, dst_name = (
                _FORMATS[src_payload.dtype],
                _FORMATS[dst_payload.dtype],
            )
        elif dst_mx:
            kind = "quant"
            src_name, dst_name = (
                _DENSE[src_payload.dtype][0],
                _FORMATS[dst_payload.dtype],
            )
        else:
            kind = "dequant"
            src_name, dst_name = (
                _FORMATS[src_payload.dtype],
                _DENSE[dst_payload.dtype][0],
            )
        args = [
            a,
            b,
            _byte_ptr(src[1], source_scale[1]) if src_mx else a,
            _byte_ptr(dst[1], dest_scale[1]) if dst_mx else b,
        ]
        values = (
            rows,
            cols,
            sl[2],
            dl[2],
            source_scale[0] if src_mx else 0,
            dest_scale[0] if dst_mx else 0,
            source_step,
            dest_step,
            round_mode,
            rows * cols if prefix_mask is None else prefix_mask,
        )
        _emit_bitcode(
            f"tla_mx_scale_stride_{kind}_{src_name}_{dst_name}", args, values, loc
        )
        return True
    if (
        src_mx
        and dst_mx
        and src_payload.dtype.startswith("f4")
        and dst_payload.dtype.startswith("f8")
    ):
        symbol = (
            f"tla_mx_tensor_{_FORMATS[src_payload.dtype]}_{_FORMATS[dst_payload.dtype]}"
        )
        if prefix_mask is not None:
            symbol += "_masked"
        args = [
            a,
            b,
            _byte_ptr(src[1], source_scale[1]),
            _byte_ptr(dst[1], dest_scale[1]),
        ]
        values = (rows, cols, sl[2], dl[2], source_scale[0], dest_scale[0])
        if prefix_mask is not None:
            values += (prefix_mask,)
    elif src_mx and dst_mx:
        symbol = f"tla_mx_tensor_requant_{_FORMATS[src_payload.dtype]}_{_FORMATS[dst_payload.dtype]}"
        if prefix_mask is not None:
            symbol += "_masked"
        args = [
            a,
            _byte_ptr(src[1], source_scale[1]),
            b,
            _byte_ptr(dst[1], dest_scale[1]),
        ]
        values = (rows, cols, sl[2], source_scale[0], dl[2], dest_scale[0], round_mode)
        if prefix_mask is not None:
            values += (prefix_mask,)
    elif dst_mx:
        symbol = f"tla_mx_tensor_quant_{_DENSE[src_payload.dtype][0]}_{_FORMATS[dst_payload.dtype]}"
        if prefix_mask is not None:
            symbol += "_masked"
        args = [a, b, _byte_ptr(dst[1], dest_scale[1])]
        values = (
            rows,
            cols,
            sl[2] // _DENSE[src_payload.dtype][1],
            dl[2],
            dest_scale[0],
            round_mode,
        )
        if prefix_mask is not None:
            values += (prefix_mask,)
    else:
        symbol = f"tla_mx_tensor_dequant_{_FORMATS[src_payload.dtype]}_{_DENSE[dst_payload.dtype][0]}"
        if prefix_mask is not None:
            symbol += "_masked"
        args = [a, _byte_ptr(src[1], source_scale[1]), b]
        values = (
            rows,
            cols,
            sl[2],
            source_scale[0],
            dl[2] // _DENSE[dst_payload.dtype][1],
            round_mode,
        )
        if prefix_mask is not None:
            values += (prefix_mask,)
    _emit_bitcode(symbol, args, values, loc)
    return True
