import catlass.tla as tla


@tla.jit
def Init(gm_a, gm_b, gm_c, config: tla.Constexpr):
    blockIdx = tla.arch.block_idx()
    blockNum = tla.arch.block_num()
    baseCoreM = config.m // blockNum
    remainderM = config.m % blockNum
    actualCoreM = tla.as_numeric(0)
    startM = tla.as_numeric(0)
    if blockIdx < remainderM:
        actualCoreM = tla.as_numeric(baseCoreM) + 1
        startM = tla.as_numeric(blockIdx) * tla.as_numeric(actualCoreM)
    else:
        actualCoreM = tla.as_numeric(baseCoreM)
        startM = tla.as_numeric(remainderM) * (tla.as_numeric(baseCoreM) + 1) + (
            tla.as_numeric(blockIdx) - tla.as_numeric(remainderM)
        ) * tla.as_numeric(baseCoreM)

    # 输入为连续RowMajor矩阵，将每核负责的连续行表示成一维元素范围
    core_data_len = actualCoreM * config.n
    core_start = startM * config.n
    core_layout = tla.make_layout(
        tla.make_shape(core_data_len),
        tla.make_stride(1),
        layoutTag=tla.arch.RowMajor,
    )
    core_coord = tla.make_coord(core_start)
    gm_a_core = tla.make_tensor(gm_a.ptr, core_layout, coord=core_coord)
    gm_b_core = tla.make_tensor(gm_b.ptr, core_layout, coord=core_coord)
    gm_c_core = tla.make_tensor(gm_c.ptr, core_layout, coord=core_coord)

    return gm_a_core, gm_b_core, gm_c_core, core_data_len
