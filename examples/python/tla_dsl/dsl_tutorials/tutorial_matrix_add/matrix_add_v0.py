import catlass.tla as tla

from matrix_add_common import Init


def ceil_div(a, b):
    return (a + b - 1) // b


@tla.kernel
def matrix_add_v0(
    gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor, config: tla.Constexpr
):
    # 8192行平分到core_num，A5上是56个vector core，每个vector core分8192//56 = 146, 16

    # 获取这个core在gm_a, gm_b, gm_c上的tensor
    gm_a_core, gm_b_core, gm_c_core, core_data_len = Init(gm_a, gm_b, gm_c, config)

    # UB里面申请空间，每次搬入一个data块
    ub_ptr_a = tla.allocate(config.dataBlockLen, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ptr_b = tla.allocate(config.dataBlockLen, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ptr_c = tla.allocate(config.dataBlockLen, tla.Float32, tla.AddressSpace.ub, 256)

    totalBlocks = ceil_div(core_data_len, config.dataBlockLen)

    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.SCALAR)
    scalar_done = tla.flag("scalar_done", tla.arch.SCALAR, tla.arch.MTE3)
    ub_stored = tla.flag("ub_stored", tla.arch.MTE3, tla.arch.MTE2)
    with tla.vector():
        # 数据切分成多块从GM搬运到UB
        for i in tla.range(totalBlocks):
            gm_a_tile = tla.tile_view(
                gm_a_core, tla.make_shape(config.dataBlockLen), tla.make_coord(i)
            )
            ub_a_tile = tla.make_tensor_like(ub_ptr_a, gm_a_tile, tla.arch.RowMajor)

            gm_b_tile = tla.tile_view(
                gm_b_core, tla.make_shape(config.dataBlockLen), tla.make_coord(i)
            )
            ub_b_tile = tla.make_tensor_like(ub_ptr_b, gm_b_tile, tla.arch.RowMajor)

            gm_c_tile = tla.tile_view(
                gm_c_core, tla.make_shape(config.dataBlockLen), tla.make_coord(i)
            )
            ub_c_tile = tla.make_tensor_like(ub_ptr_c, gm_c_tile, tla.arch.RowMajor)

            tla.copy(ub_a_tile, gm_a_tile)
            tla.copy(ub_b_tile, gm_b_tile)
            # Scalar loads must wait for both MTE2 copies to finish.
            tla.set_flag(ub_loaded)
            tla.wait_flag(ub_loaded)

            for j in tla.range(gm_a_tile.origin_shape[0]):
                ub_c_tile[j] = ub_a_tile[j] + ub_b_tile[j]

            # MTE3 must wait for the scalar stores before copying the result.
            tla.set_flag(scalar_done)
            tla.wait_flag(scalar_done)

            tla.copy(gm_c_tile, ub_c_tile)
            tla.set_flag(ub_stored)
            tla.wait_flag(ub_stored)

        tla.pipe_barrier(tla.pipes.ALL)
    return
