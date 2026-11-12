"""Generate lit input from legal Python ``@tla.kernel`` programs."""

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import argparse

import catlass.tla as tla
import catlass.runtime as runtime_mod


@tla.kernel
def _static_f16_mmad_kernel() -> None:
    lhs_parent = tla.make_tensor(
        tla.allocate((128, 64), tla.Float16, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
    )
    rhs_parent = tla.make_tensor(
        tla.allocate((64, 128), tla.Float16, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
    )
    acc_parent = tla.make_tensor(
        tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
    )
    lhs = tla.make_tensor_like(
        tla.allocate((128, 64), tla.Float16, tla.AddressSpace.l0a, 512),
        lhs_parent,
        tla.arch.zN,
    )
    rhs = tla.make_tensor_like(
        tla.allocate((64, 128), tla.Float16, tla.AddressSpace.l0b, 512),
        rhs_parent,
        tla.arch.nZ,
    )
    acc = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l0c, 1024),
        acc_parent,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)


@tla.kernel
def _static_bf16_mmad_kernel() -> None:
    lhs_parent = tla.make_tensor(
        tla.allocate((128, 64), tla.BFloat16, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
    )
    rhs_parent = tla.make_tensor(
        tla.allocate((64, 128), tla.BFloat16, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
    )
    acc_parent = tla.make_tensor(
        tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
    )
    lhs = tla.make_tensor_like(
        tla.allocate((128, 64), tla.BFloat16, tla.AddressSpace.l0a, 512),
        lhs_parent,
        tla.arch.zN,
    )
    rhs = tla.make_tensor_like(
        tla.allocate((64, 128), tla.BFloat16, tla.AddressSpace.l0b, 512),
        rhs_parent,
        tla.arch.nZ,
    )
    acc = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l0c, 1024),
        acc_parent,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)


@tla.kernel
def _static_f32_mmad_kernel() -> None:
    lhs_parent = tla.make_tensor(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(32, 1)),
    )
    rhs_parent = tla.make_tensor(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(32, 1)),
    )
    acc_parent = tla.make_tensor(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(32, 1)),
    )
    lhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0a, 512),
        lhs_parent,
        tla.arch.zN,
    )
    rhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0b, 512),
        rhs_parent,
        tla.arch.nZ,
    )
    acc = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 1024),
        acc_parent,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)


@tla.kernel
def _static_i8_mmad_kernel() -> None:
    lhs_parent = tla.make_tensor(
        tla.allocate((128, 64), tla.Int8, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
    )
    rhs_parent = tla.make_tensor(
        tla.allocate((64, 128), tla.Int8, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
    )
    acc_parent = tla.make_tensor(
        tla.allocate((128, 128), tla.Int32, tla.AddressSpace.l1, 512),
        tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
    )
    lhs = tla.make_tensor_like(
        tla.allocate((128, 64), tla.Int8, tla.AddressSpace.l0a, 512),
        lhs_parent,
        tla.arch.zN,
    )
    rhs = tla.make_tensor_like(
        tla.allocate((64, 128), tla.Int8, tla.AddressSpace.l0b, 512),
        rhs_parent,
        tla.arch.nZ,
    )
    # The integer route accumulates in i32, not fp32.
    acc = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Int32, tla.AddressSpace.l0c, 1024),
        acc_parent,
        tla.arch.L0Clayout,
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)


def _mx_mmad_kernel(elem_a, elem_b, l0_align=1024):
    """MX kernel: scales ride the L1->L0 load, and the matmul is tla.mmad_mx.

    ``l0_align`` is what the L0A / L0B operands are allocated with. mxfp8 needs
    1024; passing 512 produces the misaligned case the verifier has to reject.
    """

    @tla.kernel
    def _kernel() -> None:
        l1a = tla.make_tensor(
            tla.allocate((128, 64), elem_a, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
        )
        l1b = tla.make_tensor(
            tla.allocate((64, 128), elem_b, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
        )
        accp = tla.make_tensor(
            tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
        )
        # One e8m0 exponent per 32 elements along K.
        sa_p = tla.make_tensor(
            tla.allocate((128, 2), tla.Float8E8M0, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 2), tla.make_stride(2, 1)),
        )
        sb_p = tla.make_tensor(
            tla.allocate((2, 128), tla.Float8E8M0, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(2, 128), tla.make_stride(128, 1)),
        )
        # mxfp8 needs the L0A / L0B operands on a 1024-byte boundary; the 512
        # every other L0 user gets by with is half a block here.
        lhs = tla.make_tensor_like(
            tla.allocate((128, 64), elem_a, tla.AddressSpace.l0a, l0_align),
            l1a,
            tla.arch.zN,
        )
        rhs = tla.make_tensor_like(
            tla.allocate((64, 128), elem_b, tla.AddressSpace.l0b, l0_align),
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
        # The L1 operands carry the fractal tags a real kernel gives them --
        # make_tensor_like from a GM tile does this -- because the L1 -> L0 load
        # is named after the source layout, zN or nZ, not just the destination.
        src_a = tla.make_tensor_like(
            tla.allocate((128, 64), elem_a, tla.AddressSpace.l1, 512), l1a, tla.arch.zN
        )
        src_b = tla.make_tensor_like(
            tla.allocate((64, 128), elem_b, tla.AddressSpace.l1, 512), l1b, tla.arch.nZ
        )
        with tla.cube():
            tla.copy(lhs, src_a, scale=sa)
            tla.copy(rhs, src_b, scale=sb)
            tla.mmad_mx(acc, lhs, rhs, init_c=True, unit_flag=3)

    return _kernel


def _mx_half_scaled_mmad_kernel(scale_lhs=True, use_mx_op=False):
    """Negative case: lhs MX-loaded, rhs loaded plain, then one tla.mmad.

    tla.mmad rejects the scaled lhs outright -- it lowers to mad, which would
    drop the scale.

    mad_mx scales both operands, so there is no instruction for this. Nothing
    about the operands says so -- the element types and layout tags are those of
    both a legal MX matmul and a legal plain one -- which is the point: only the
    provenance of each operand separates the three cases.
    """

    @tla.kernel
    def _kernel() -> None:
        l1a_p = tla.make_tensor(
            tla.allocate((128, 64), tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
        )
        l1b_p = tla.make_tensor(
            tla.allocate((64, 128), tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
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
        # zN / nZ on the L1 side so the plain (unscaled) L1->L0 route exists.
        l1a = tla.make_tensor_like(
            tla.allocate((128, 64), tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
            l1a_p,
            tla.arch.zN,
        )
        l1b = tla.make_tensor_like(
            tla.allocate((64, 128), tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
            l1b_p,
            tla.arch.nZ,
        )
        lhs = tla.make_tensor_like(
            tla.allocate((128, 64), tla.Float8E4M3FN, tla.AddressSpace.l0a, 512),
            l1a,
            tla.arch.zN,
        )
        rhs = tla.make_tensor_like(
            tla.allocate((64, 128), tla.Float8E4M3FN, tla.AddressSpace.l0b, 512),
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
        with tla.cube():
            if scale_lhs:
                tla.copy(lhs, l1a, scale=sa)
            else:
                tla.copy(lhs, l1a)
            tla.copy(rhs, l1b)
            if use_mx_op:
                tla.mmad_mx(acc, lhs, rhs, init_c=True, unit_flag=3)
            else:
                tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)

    return _kernel


_FP4_TYPES = {"f4e2m1": tla.Float4E2M1, "f4e1m2": tla.Float4E1M2}


def _mxfp4_mmad_kernel(fmt, rhs_fmt=None, scale=True):
    """MX fp4: i4 elements on the ordinary zN/nZ layouts, encoding from the load.

    `rhs_fmt` loads the rhs with a different encoding than the lhs; `scale=False`
    drops the scale from both loads and matmuls with the plain tla.mmad, leaving
    packed fp4 tiles on a route the cube does not have.
    """

    lhs_t = _FP4_TYPES[fmt]
    rhs_t = _FP4_TYPES[rhs_fmt or fmt]

    @tla.kernel
    def _kernel() -> None:
        # Shapes are in fp4 elements; storage is half that in bytes.
        l1a = tla.make_tensor(
            tla.allocate((128, 64), lhs_t, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
        )
        l1b = tla.make_tensor(
            tla.allocate((64, 128), rhs_t, tla.AddressSpace.l1, 512),
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
        src_a = tla.make_tensor_like(
            tla.allocate((128, 64), lhs_t, tla.AddressSpace.l1, 512), l1a, tla.arch.zN
        )
        src_b = tla.make_tensor_like(
            tla.allocate((64, 128), rhs_t, tla.AddressSpace.l1, 512), l1b, tla.arch.nZ
        )
        lhs = tla.make_tensor_like(
            tla.allocate((128, 64), lhs_t, tla.AddressSpace.l0a, 512), src_a, tla.arch.zN
        )
        rhs = tla.make_tensor_like(
            tla.allocate((64, 128), rhs_t, tla.AddressSpace.l0b, 512), src_b, tla.arch.nZ
        )
        acc = tla.make_tensor_like(
            tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l0c, 1024),
            accp,
            tla.arch.L0Clayout,
        )
        sa = tla.make_tensor_like(
            tla.allocate((128, 2), tla.Float8E8M0, tla.AddressSpace.l1, 512), sa_p, tla.arch.zZMxScale
        )
        sb = tla.make_tensor_like(
            tla.allocate((2, 128), tla.Float8E8M0, tla.AddressSpace.l1, 512), sb_p, tla.arch.nNMxScale
        )
        with tla.cube():
            if scale:
                tla.copy(lhs, src_a, scale=sa)
                tla.copy(rhs, src_b, scale=sb)
            else:
                tla.copy(lhs, src_a)
                tla.copy(rhs, src_b)
            if scale:
                tla.mmad_mx(acc, lhs, rhs, init_c=True, unit_flag=3)
            else:
                # What a user who did not know fp4 is microscaling-only would
                # write: the plain route has no fp4 variant at all.
                tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)

    return _kernel


@tla.kernel
def _dynamic_init_mmad_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    lhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0a, 512), mem_a, tla.arch.zN
    )
    rhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0b, 512), mem_b, tla.arch.nZ
    )
    acc = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 1024), mem_c, tla.arch.L0Clayout
    )
    with tla.cube():
        for outer in tla.range(0, 2, 1):
            for inner in tla.range(0, 2, 1):
                init_c = True if outer == 0 and inner == 0 else False
                tla.mmad(acc, lhs, rhs, init_c=init_c, unit_flag=3)


@tla.kernel
def _dynamic_unit_mmad_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    lhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0a, 512), mem_a, tla.arch.zN
    )
    rhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0b, 512), mem_b, tla.arch.nZ
    )
    acc = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 1024), mem_c, tla.arch.L0Clayout
    )
    with tla.cube():
        for outer in tla.range(0, 2, 1):
            for inner in tla.range(0, 2, 1):
                unit_flag = 3 if outer == 1 and inner == 1 else 2
                tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=unit_flag)


@tla.kernel
def _dynamic_init_unit_mmad_kernel(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor
) -> None:
    lhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0a, 512), mem_a, tla.arch.zN
    )
    rhs = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0b, 512), mem_b, tla.arch.nZ
    )
    acc = tla.make_tensor_like(
        tla.allocate((32, 32), tla.Float32, tla.AddressSpace.l0c, 1024), mem_c, tla.arch.L0Clayout
    )
    with tla.cube():
        for outer in tla.range(0, 2, 1):
            for inner in tla.range(0, 2, 1):
                init_c = True if outer == 0 and inner == 0 else False
                unit_flag = 3 if outer == 1 and inner == 1 else 2
                tla.mmad(acc, lhs, rhs, init_c=init_c, unit_flag=unit_flag)


def _f32_mmad_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    return (
        make_fake_tensor(
            tla.Float32,
            (32, 32),
            (32, 1),
            origin_shape=(32, 32),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float32,
            (32, 32),
            (32, 1),
            origin_shape=(32, 32),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float32,
            (32, 32),
            (32, 1),
            origin_shape=(32, 32),
            layout_tag=tla.arch.RowMajor,
        ),
    )


def _fp8_mmad_kernel(elem_a, elem_b):
    """Build a cube kernel with the given fp8 operand formats."""

    @tla.kernel
    def _kernel() -> None:
        lhs_parent = tla.make_tensor(
            tla.allocate((128, 64), elem_a, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
        )
        rhs_parent = tla.make_tensor(
            tla.allocate((64, 128), elem_b, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(64, 128), tla.make_stride(128, 1)),
        )
        acc_parent = tla.make_tensor(
            tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l1, 512),
            tla.make_layout(tla.make_shape(128, 128), tla.make_stride(128, 1)),
        )
        lhs = tla.make_tensor_like(
            tla.allocate((128, 64), elem_a, tla.AddressSpace.l0a, 512),
            lhs_parent,
            tla.arch.zN,
        )
        rhs = tla.make_tensor_like(
            tla.allocate((64, 128), elem_b, tla.AddressSpace.l0b, 512),
            rhs_parent,
            tla.arch.nZ,
        )
        acc = tla.make_tensor_like(
            tla.allocate((128, 128), tla.Float32, tla.AddressSpace.l0c, 1024),
            acc_parent,
            tla.arch.L0Clayout,
        )
        with tla.cube():
            tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)

    return _kernel


@tla.kernel
def _fp8_vector_copy_kernel(mem_a: tla.Tensor) -> None:
    """fp8 staged through UB on the vector path.

    Once rejected as a cube-operand-only format; Vector/dma.cpp now registers
    GM<->UB and UB->L1 for both fp8 encodings, so the route resolves and the
    copy lowers like any other element type.
    """
    ub = tla.make_tensor(
        tla.allocate((32, 32), tla.Float8E4M3FN, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(32, 32), tla.make_stride(32, 1)),
    )
    with tla.vector():
        tla.copy(ub, mem_a)


def _fp8_vector_copy_args() -> tuple[tla.Tensor]:
    return (
        make_fake_tensor(
            tla.Float8E4M3FN,
            (32, 32),
            (32, 1),
            origin_shape=(32, 32),
            layout_tag=tla.arch.RowMajor,
        ),
    )


def _i8_fixpipe_args() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    """GM operands / result for the integer fixpipe kernel: i8 in, i32 out."""
    return (
        make_fake_tensor(
            tla.Int8, (128, 64), (64, 1), origin_shape=(128, 64), layout_tag=tla.arch.RowMajor
        ),
        make_fake_tensor(
            tla.Int8, (64, 128), (128, 1), origin_shape=(64, 128), layout_tag=tla.arch.RowMajor
        ),
        make_fake_tensor(
            tla.Int32, (128, 128), (128, 1), origin_shape=(128, 128), layout_tag=tla.arch.RowMajor
        ),
    )


@tla.kernel
def _i8_fixpipe_kernel(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor) -> None:
    """Every way out of an i32 accumulator: L0C -> GM, L0C -> UB, L0C -> L1.

    The i32 accumulator has no narrowing path, so each copy must resolve to an
    int32_t -> int32_t runtime callee rather than one of the fp32 ones.
    """
    lhs = tla.make_tensor_like(
        tla.allocate((128, 64), tla.Int8, tla.AddressSpace.l0a, 512), mem_a, tla.arch.zN
    )
    rhs = tla.make_tensor_like(
        tla.allocate((64, 128), tla.Int8, tla.AddressSpace.l0b, 512), mem_b, tla.arch.nZ
    )
    acc = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Int32, tla.AddressSpace.l0c, 1024), mem_c, tla.arch.L0Clayout
    )
    ub_c = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Int32, tla.AddressSpace.ub, 256), mem_c, tla.arch.RowMajor
    )
    l1_c = tla.make_tensor_like(
        tla.allocate((128, 128), tla.Int32, tla.AddressSpace.l1, 512), mem_c, tla.arch.zN
    )
    with tla.cube():
        tla.mmad(acc, lhs, rhs, init_c=True, unit_flag=3)
        tla.copy(mem_c, acc)
        tla.copy(ub_c, acc, tla.params.CopyL0C2DstParams(
            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
        ))
        tla.copy(l1_c, acc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        choices=(
            "f16",
            "bf16",
            "f32",
            "i8",
            "i8-fixpipe",
            "fp8",
            "fp8-mixed",
            "fp8-vector-copy",
            "mx",
            "mx-mixed",
            "mx-misaligned",
            "mxfp4",
            "mxfp4-e1m2",
            "mx-half-scaled",
            "mx-op-without-scale",
            "mxfp4-mixed-encodings",
            "mxfp4-unscaled",
            "dynamic-init",
            "dynamic-unit",
            "dynamic-both",
        ),
    )
    case = parser.parse_args().case
    if case == "f16":
        kernel, type_args = _static_f16_mmad_kernel, ()
    elif case == "bf16":
        kernel, type_args = _static_bf16_mmad_kernel, ()
    elif case == "f32":
        kernel, type_args = _static_f32_mmad_kernel, ()
    elif case == "i8":
        kernel, type_args = _static_i8_mmad_kernel, ()
    elif case == "i8-fixpipe":
        kernel, type_args = _i8_fixpipe_kernel, _i8_fixpipe_args()
    elif case == "fp8":
        kernel, type_args = _fp8_mmad_kernel(tla.Float8E4M3FN, tla.Float8E4M3FN), ()
    elif case == "fp8-mixed":
        kernel, type_args = _fp8_mmad_kernel(tla.Float8E4M3FN, tla.Float8E5M2), ()
    elif case == "fp8-vector-copy":
        kernel, type_args = _fp8_vector_copy_kernel, _fp8_vector_copy_args()
    elif case == "mx":
        kernel, type_args = _mx_mmad_kernel(tla.Float8E4M3FN, tla.Float8E4M3FN), ()
    elif case == "mx-mixed":
        kernel, type_args = _mx_mmad_kernel(tla.Float8E5M2, tla.Float8E4M3FN), ()
    elif case == "mx-misaligned":
        # 512 is legal for every other L0 user, so the frontend accepts it; only
        # the matmul knows this operand is mxfp8 and needs 1024.
        kernel, type_args = _mx_mmad_kernel(tla.Float8E4M3FN, tla.Float8E4M3FN, 512), ()
    elif case == "mxfp4":
        kernel, type_args = _mxfp4_mmad_kernel("f4e2m1"), ()
    elif case == "mxfp4-e1m2":
        kernel, type_args = _mxfp4_mmad_kernel("f4e1m2"), ()
    elif case == "mx-half-scaled":
        kernel, type_args = _mx_half_scaled_mmad_kernel(), ()
    elif case == "mx-op-without-scale":
        kernel, type_args = (
            _mx_half_scaled_mmad_kernel(scale_lhs=False, use_mx_op=True),
            (),
        )
    elif case == "mxfp4-mixed-encodings":
        # lhs e1m2, rhs e2m1. Legal: each operand names its own encoding, and the
        # cube has a mad_mx for every pairing.
        kernel, type_args = _mxfp4_mmad_kernel("f4e1m2", rhs_fmt="f4e2m1"), ()
    elif case == "mxfp4-unscaled":
        kernel, type_args = _mxfp4_mmad_kernel("f4e2m1", scale=False), ()
    else:
        kernels = {
            "dynamic-init": _dynamic_init_mmad_kernel,
            "dynamic-unit": _dynamic_unit_mmad_kernel,
            "dynamic-both": _dynamic_init_unit_mmad_kernel,
        }
        kernel, type_args = kernels[case], _f32_mmad_args()
    print(kernel.dump_mlir(type_args=type_args))


if __name__ == "__main__":
    main()
