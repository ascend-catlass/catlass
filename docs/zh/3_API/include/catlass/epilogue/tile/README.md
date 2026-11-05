# Epilogue/Tile 类模板概述

> [代码位置](../../../../../../../include/catlass/epilogue/tile/)

[TOC]

Epilogue 的 tile 层 API 作为 epilogue block 层的模板参数，一般需要在 kernel 模板组装时做声明。包含搬运（copy）、广播（broadcast）、逐元素计算（elemwise）、类型转换（cast）、反量化（dequant）和 swizzle 遍历策略等组件。

## API 清单

### 搬运组件

| 组件                                        | 风格   | 适用硬件           | 说明                                                            |
| :------------------------------------------ | :----- | :----------------- | :-------------------------------------------------------------- |
| [copy_gm_to_ub](./copy_gm_to_ub/README.md)  | 非 TLA | Atlas A2, Ascend 950 | GM→UB 搬运（CopyGm2Ub, CopyPerTokenScale2Ub, CopyGm2UbAligned） |
| [copy_gm_to_ub_tla](./copy_gm_to_ub_tla.md) | TLA    | Atlas A2, Ascend 950 | GM→UB TLA 搬运（CopyGm2UbTla）                                  |
| [copy_ub_to_gm](./copy_ub_to_gm/README.md)  | 非 TLA | Atlas A2, Ascend 950 | UB→GM 搬运（CopyUb2Gm, CopyUb2GmAligned）                       |
| [copy_ub_to_gm_tla](./copy_ub_to_gm_tla.md) | TLA    | Atlas A2, Ascend 950 | UB→GM TLA 搬运（CopyUb2GmTla）                                  |
| [copy_ub_to_l1_tla](./copy_ub_to_l1_tla.md) | TLA    | Ascend 950          | UB→L1 搬运（zN 格式）                                           |
| [tile_copy](./tile_copy/README.md)          | 组合   | Atlas A2, Ascend 950 | 搬运聚合模板（TileCopy, TileCopyBf16, PerTokenDequant 等）      |

### 广播组件

| 组件                                                                      | 风格         | 说明                               |
| :------------------------------------------------------------------------ | :----------- | :--------------------------------- |
| [tile_broadcast_add](./tile_broadcast_add.md)                             | 非 TLA       | 行广播加法（In0 + broadcast(In1)） |
| [tile_broadcast_mul](./tile_broadcast_mul/README.md)                      | 非 TLA + TLA | 广播乘法（行广播/列广播）          |
| [tile_broadcast_one_blk](./tile_broadcast_one_blk/README.md)              | 非 TLA + TLA | One-block 广播（scalar→block）     |
| [tile_broadcast_inplace_by_column](./tile_broadcast_inplace_by_column.md) | 非 TLA       | 列广播原地拷贝（原地修改）         |
| [tile_broadcast_inplace_by_row](./tile_broadcast_inplace_by_row.md)       | 非 TLA       | 行广播原地拷贝（原地修改）         |

### 逐元素组件

| 组件                                          | 风格   | 说明                   |
| :-------------------------------------------- | :----- | :--------------------- |
| [tile_elemwise_add](./tile_elemwise_add.md)   | 非 TLA | 逐元素加法（Add）      |
| [tile_elemwise_mul](./tile_elemwise_mul.md)   | 非 TLA | 逐元素乘法（Mul）      |
| [tile_elemwise_muls](./tile_elemwise_muls.md) | 非 TLA | 逐元素乘以标量（Muls） |
| [tile_elemwise_gelu](./tile_elemwise_gelu.md) | 非 TLA | GELU 激活函数          |
| [tile_elemwise_silu](./tile_elemwise_silu.md) | 非 TLA | SiLU / Swish 激活函数  |

### 转换与反量化

| 组件                                                | 风格   | 适用硬件           | 说明                         |
| :-------------------------------------------------- | :----- | :----------------- | :--------------------------- |
| [tile_cast](./tile_cast.md)                         | 非 TLA | Atlas A2, Ascend 950 | 类型转换（Cast）             |
| [tile_pertoken_dequant](./tile_pertoken_dequant.md) | TLA    | Ascend 950          | Per-Token 反量化（int32→fp） |

### Swizzle

| 组件                                     | 说明                                   |
| :--------------------------------------- | :------------------------------------- |
| [tile_swizzle](./tile_swizzle/README.md) | Tile 遍历策略（Identity / Horizontal） |
