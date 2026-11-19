"""Layout-derived batch geometry without a new copy parameter object."""

import inspect

import pytest

from catlass import tla
from catlass.core_api import copy
from catlass.execution_lowering import TlaLoweringError
from catlass.runtime import TlaCoreAPIError


def _copy_kernel(
    *,
    dtype=tla.Float8E4M3FN,
    count=3,
    dst_count=3,
    dst_step=1024,
    src_step=4096,
    dst_rank=3,
    cache_policy=None,
):
    @tla.kernel
    def kernel():
        source = tla.make_tensor(
            tla.make_ptr(dtype, 4096, mem_space=tla.AddressSpace.gm),
            tla.make_layout(
                tla.make_shape(count, 17, 64), tla.make_stride(src_step, 64, 1)
            ),
        )
        ptr = tla.allocate(5 * 32 * 64, dtype, tla.AddressSpace.l1, 512)
        if tla.const_expr(dst_rank == 3):
            destination = tla.make_tensor(
                ptr,
                tla.make_layout(
                    tla.make_shape(dst_count, (16, 2), (32, 2)),
                    tla.make_stride(dst_step, (32, 512), (1, 5120)),
                    origin_shape=tla.make_shape(dst_count, 17, 64),
                    layoutTag=tla.arch.zN,
                ),
            )
        else:
            destination = tla.make_tensor(
                ptr, tla.make_layout_with_tag((17, 64), dtype, tla.arch.zN)
            )
        with tla.cube():
            tla.copy(destination, source, cache_policy=cache_policy)

    return kernel


def test_copy_signature_is_unchanged():
    assert tuple(inspect.signature(copy).parameters) == (
        "dst",
        "src",
        "params",
        "scale",
        "cache_policy",
        "loc",
    )
    assert (
        inspect.signature(copy).parameters["scale"].kind
        == inspect.Parameter.KEYWORD_ONLY
    )


def test_batch_copy_keeps_independent_layout_steps():
    text = _copy_kernel().dump_mlir(type_args=())
    assert "!tla.stride<4096,64,1>" in text
    assert "!tla.stride<1024,(32,512),(1,5120)>" in text
    assert "!tla.shape<3,17,64>" in text
    assert "tla.copy " in text


def test_batch_copy_preserves_cache_policy():
    text = _copy_kernel(cache_policy=tla.params.CachePolicy.DISABLE).dump_mlir(
        type_args=()
    )
    assert "src_cache_policy = #tla.cache_policy<disable>" in text


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"dst_count": 2}, "matching logical extents"),
        ({"dst_step": 1000}, "32-byte aligned"),
        ({"count": 65536, "dst_count": 65536}, "unsigned 16-bit"),
        ({"dtype": tla.Int8}, "supports FP8 and E8M0"),
        ({"dst_rank": 2}, "two logical rank-3"),
    ],
)
def test_batch_copy_rejects_invalid_geometry(kwargs, message):
    with pytest.raises((TlaCoreAPIError, TlaLoweringError), match=message):
        _copy_kernel(**kwargs).dump_mlir(type_args=())


@pytest.mark.parametrize(
    "tag,shape,pitch,inner",
    [
        (tla.arch.RowMajor, (5, 32, 33), 2048, "64,1"),
        (tla.arch.ColumnMajor, (5, 33, 32), 2048, "1,64"),
        (tla.arch.zN, (5, 17, 33), 2048, "(32,512),(1,1024)"),
    ],
)
def test_default_batch_pitch_includes_matrix_padding(tag, shape, pitch, inner):
    @tla.kernel
    def kernel():
        ptr = tla.allocate(5 * 2048, tla.Float8E4M3FN, tla.AddressSpace.l1, 512)
        tla.make_tensor(ptr, tla.make_layout_with_tag(shape, tla.Float8E4M3FN, tag))

    text = kernel.dump_mlir(type_args=())
    assert f"!tla.stride<{pitch},{inner}>" in text


def test_get_tile_preserves_parent_batch_pitch_and_accumulates_coordinates():
    @tla.kernel
    def kernel():
        source = tla.make_tensor(
            tla.make_ptr(tla.Float8E4M3FN, 4096, mem_space=tla.AddressSpace.gm),
            tla.make_layout(tla.make_shape(5, 32, 64), tla.make_stride(4096, 64, 1)),
        )
        tile = tla.get_tile(source, tla.make_coord(1, 0, 0), tla.make_shape(4, 32, 64))
        tla.get_tile(tile, tla.make_coord(1, 1, 0), tla.make_shape(2, 17, 64))

    text = kernel.dump_mlir(type_args=())
    assert "!tla.shape<2,17,64>" in text
    assert "!tla.coord<2,1,0>" in text
    assert "!tla.stride<4096,64,1>" in text


def _layout_copy_kernel(dtype, src_tag, dst_tag):
    scale_layouts = {
        tla.arch.RowMajorMxScaleA: ((2, (1, 32), (2, 32)), (2048, (1, 64), (1, 2))),
        tla.arch.ColMajorMxScaleA: ((2, (1, 32), (2, 32)), (2048, (1, 2), (1, 64))),
        tla.arch.RowMajorMxScaleB: ((2, (2, 16), (1, 64)), (2048, (1, 128), (1, 2))),
        tla.arch.ColMajorMxScaleB: ((2, (2, 16), (1, 64)), (2048, (1, 2), (1, 32))),
    }
    scale_shape, scale_stride = scale_layouts.get(src_tag, ((), ()))

    @tla.kernel
    def kernel():
        source_ptr = tla.make_ptr(dtype, 4096, mem_space=tla.AddressSpace.gm)
        destination_ptr = tla.allocate(8192, dtype, tla.AddressSpace.l1, 512)
        if tla.const_expr(dtype == tla.Float8E8M0):
            source = tla.make_tensor(
                source_ptr,
                tla.make_layout(
                    tla.make_shape(*scale_shape),
                    tla.make_stride(*scale_stride),
                    origin_shape=tla.make_shape(2, 32, 64),
                    layoutTag=src_tag,
                ),
            )
        else:
            source = tla.make_tensor(
                source_ptr, tla.make_layout_with_tag((2, 32, 64), dtype, src_tag)
            )
        destination = tla.make_tensor(
            destination_ptr, tla.make_layout_with_tag((2, 32, 64), dtype, dst_tag)
        )
        with tla.cube():
            tla.copy(destination, source)

    return kernel


@pytest.mark.parametrize(
    "tag",
    [tla.arch.zNUnAlign, tla.arch.zN, tla.arch.nZ],
)
def test_batch_copy_rejects_layouts_without_batch_runtime(tag):
    with pytest.raises(
        (TlaCoreAPIError, TlaLoweringError),
        match="unsupported batch-matrix copy layout",
    ):
        _layout_copy_kernel(tla.Float8E4M3FN, tag, tag).dump_mlir(type_args=())


@pytest.mark.parametrize(
    "dtype,src_tag,dst_tag",
    [
        (dtype, src, dst)
        for dtype in (tla.Float8E4M3FN, tla.Float8E5M2)
        for src, dst in (
            (tla.arch.RowMajor, tla.arch.zN),
            (tla.arch.ColumnMajor, tla.arch.nZ),
        )
    ]
    + [
        (tla.Float8E8M0, src, dst)
        for src, dst in (
            (tla.arch.RowMajorMxScaleA, tla.arch.zZMxScale),
            (tla.arch.ColMajorMxScaleA, tla.arch.zZMxScale),
            (tla.arch.RowMajorMxScaleB, tla.arch.nNMxScale),
            (tla.arch.ColMajorMxScaleB, tla.arch.nNMxScale),
        )
    ],
)
def test_batch_copy_supported_layout_routes(dtype, src_tag, dst_tag):
    text = _layout_copy_kernel(dtype, src_tag, dst_tag).dump_mlir(type_args=())
    assert "tla.copy " in text
    assert "!tla.shape<2,32,64>" in text
