import catlass.tla as tla

from matrix_add_common import Init


def ceil_div(a, b):
    return (a + b - 1) // b


@tla.kernel
def matrix_add_v2(
    gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor, config: tla.Constexpr
):
    # 定义pingpong的两组六个信号量
    mutex_ub_a = []
    mutex_ub_b = []
    mutex_ub_c = []

    # 这里不能用mutex_ub_a = []然后append的方式来初始化mutex变量
    # list.append是python内置方法，没有提供可供检查的callable function
    mutex_ub_a = [None] * config.buffer_num
    mutex_ub_b = [None] * config.buffer_num
    mutex_ub_c = [None] * config.buffer_num

    for i in tla.range_constexpr(config.buffer_num):
        mutex_ub_a[i] = tla.mutex(resource="ub_a", id=3 * i)
        mutex_ub_b[i] = tla.mutex(resource="ub_b", id=3 * i + 1)
        mutex_ub_c[i] = tla.mutex(resource="ub_c", id=3 * i + 2)

    # 定义gm上的tensor
    gm_a_core, gm_b_core, gm_c_core, core_data_len = Init(gm_a, gm_b, gm_c, config)

    # 申请ub上的空间
    ub_a_ptr = [None] * config.buffer_num
    ub_b_ptr = [None] * config.buffer_num
    ub_c_ptr = [None] * config.buffer_num

    for buffer_idx in tla.range_constexpr(config.buffer_num):
        ub_a_ptr[buffer_idx] = tla.allocate(
            config.UB_STAGE_BYTES // config.ELEM_TYPE_LEN,
            tla.Float32,
            tla.AddressSpace.ub,
            256,
        )
        ub_b_ptr[buffer_idx] = tla.allocate(
            config.UB_STAGE_BYTES // config.ELEM_TYPE_LEN,
            tla.Float32,
            tla.AddressSpace.ub,
            256,
        )
        ub_c_ptr[buffer_idx] = tla.allocate(
            config.UB_STAGE_BYTES // config.ELEM_TYPE_LEN,
            tla.Float32,
            tla.AddressSpace.ub,
            256,
        )
    gm_tile_count = ceil_div(core_data_len, config.dataBlockLen)
    with tla.vector():
        # group_idx表示gm上的tile切块有多少组buffer
        for group_idx in tla.range(ceil_div(gm_tile_count, config.buffer_num)):
            # buffer_idx表示在一个buffer里面的编号，比如db下就是0 or 1
            for buffer_idx in tla.range_constexpr(config.buffer_num):
                # 这里不能重复使用i，因为前面i已经作为编译期整数使用过，后面再给给同样的i赋值
                # DSL不允许在动态控制流内部把已有的编译期变量改成运行期变量
                slot_idx = group_idx * config.buffer_num + buffer_idx
                if slot_idx < gm_tile_count:
                    gm_a_tile = tla.tile_view(
                        gm_a_core,
                        tla.make_shape(config.dataBlockLen),
                        tla.make_coord(slot_idx),
                    )
                    ub_a_tile = tla.make_tensor_like(
                        ub_a_ptr[buffer_idx], gm_a_tile, tla.arch.RowMajor
                    )

                    gm_b_tile = tla.tile_view(
                        gm_b_core,
                        tla.make_shape(config.dataBlockLen),
                        tla.make_coord(slot_idx),
                    )
                    ub_b_tile = tla.make_tensor_like(
                        ub_b_ptr[buffer_idx], gm_b_tile, tla.arch.RowMajor
                    )

                    gm_c_tile = tla.tile_view(
                        gm_c_core,
                        tla.make_shape(config.dataBlockLen),
                        tla.make_coord(slot_idx),
                    )
                    ub_c_tile = tla.make_tensor_like(
                        ub_c_ptr[buffer_idx], gm_c_tile, tla.arch.RowMajor
                    )

                    mutex_ub_a[buffer_idx].lock(pipe=tla.arch.MTE2)
                    tla.copy(ub_a_tile, gm_a_tile)
                    mutex_ub_a[buffer_idx].unlock(pipe=tla.arch.MTE2)

                    mutex_ub_b[buffer_idx].lock(pipe=tla.arch.MTE2)
                    tla.copy(ub_b_tile, gm_b_tile)
                    mutex_ub_b[buffer_idx].unlock(pipe=tla.arch.MTE2)

                    mutex_ub_a[buffer_idx].lock(pipe=tla.arch.VECTOR)
                    mutex_ub_b[buffer_idx].lock(pipe=tla.arch.VECTOR)
                    mutex_ub_c[buffer_idx].lock(pipe=tla.arch.VECTOR)

                    with tla.vec.func(mode="simd"):
                        valid_len = gm_c_tile.origin_shape[0]
                        for j in tla.range(ceil_div(valid_len, config.VL_LEN)):
                            ub_a_reg_tile = tla.tile_view(
                                ub_a_tile,
                                tla.make_shape(config.VL_LEN),
                                tla.make_coord(j),
                            )
                            ub_b_reg_tile = tla.tile_view(
                                ub_b_tile,
                                tla.make_shape(config.VL_LEN),
                                tla.make_coord(j),
                            )
                            ub_c_reg_tile = tla.tile_view(
                                ub_c_tile,
                                tla.make_shape(config.VL_LEN),
                                tla.make_coord(j),
                            )

                            reg_a = ub_a_reg_tile.load()
                            reg_b = ub_b_reg_tile.load()
                            reg_c = tla.add(reg_a, reg_b)
                            ub_c_reg_tile.store(reg_c)

                    mutex_ub_c[buffer_idx].unlock(pipe=tla.arch.VECTOR)
                    mutex_ub_b[buffer_idx].unlock(pipe=tla.arch.VECTOR)
                    mutex_ub_a[buffer_idx].unlock(pipe=tla.arch.VECTOR)

                    mutex_ub_c[buffer_idx].lock(pipe=tla.arch.MTE3)
                    tla.copy(gm_c_tile, ub_c_tile)
                    mutex_ub_c[buffer_idx].unlock(pipe=tla.arch.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)
    return
