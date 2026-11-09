import catlass.tla as tla

from matrix_add_common import Init


def ceil_div(a, b):
    return (a + b - 1) // b


@tla.kernel
def matrix_add_v1(
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

    mutex_ub_a = tla.mutex(resource="ub_a", id=0)
    mutex_ub_b = tla.mutex(resource="ub_b", id=1)
    mutex_ub_c = tla.mutex(resource="ub_c", id=2)

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

            # copy in
            mutex_ub_a.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_a_tile, gm_a_tile)
            mutex_ub_a.unlock(pipe=tla.arch.MTE2)

            mutex_ub_b.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_b_tile, gm_b_tile)
            mutex_ub_b.unlock(pipe=tla.arch.MTE2)

            # compute
            mutex_ub_a.lock(pipe=tla.arch.VECTOR)
            mutex_ub_b.lock(pipe=tla.arch.VECTOR)
            mutex_ub_c.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                valid_len = gm_a_tile.origin_shape[0]
                for j in range(ceil_div(valid_len, config.VL_LEN)):
                    # reg load
                    ub_vl_a = tla.tile_view(
                        ub_a_tile, tla.make_shape(config.VL_LEN), tla.make_coord(j)
                    )
                    ub_vl_b = tla.tile_view(
                        ub_b_tile, tla.make_shape(config.VL_LEN), tla.make_coord(j)
                    )
                    ub_vl_c = tla.tile_view(
                        ub_c_tile, tla.make_shape(config.VL_LEN), tla.make_coord(j)
                    )

                    reg_a = ub_vl_a.load()
                    reg_b = ub_vl_b.load()

                    # reg calc
                    reg_c = tla.add(reg_a, reg_b)

                    # reg store
                    ub_vl_c.store(reg_c)

            mutex_ub_c.unlock(pipe=tla.arch.VECTOR)
            mutex_ub_b.unlock(pipe=tla.arch.VECTOR)
            mutex_ub_a.unlock(pipe=tla.arch.VECTOR)

            # copy out

            mutex_ub_c.lock(pipe=tla.arch.MTE3)
            tla.copy(gm_c_tile, ub_c_tile)
            mutex_ub_c.unlock(pipe=tla.arch.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)
    return
