# Gemv/Tile 类模板概述

> [代码位置](../../../../../../../include/catlass/gemv/tile/)

[TOC]

GEMV 的 tile 层 API 作为 GEMV block 层的模板参数，负责矩阵-向量乘法场景下的数据搬运和向量计算。根据芯片类型分为 AIV（GM↔UB）和 AIC（GM↔L1↔L0↔GM）两类数据通路。

## API 清单

### 搬运组件

| 组件                                              | 适用硬件           | 说明                                                |
| :------------------------------------------------ | :----------------- | :-------------------------------------------------- |
| [vec_copy_gm_to_ub](./vec_copy_gm_to_ub.md)       | 全架构             | 向量 GM→UB 搬运                                     |
| [vec_copy_ub_to_gm](./vec_copy_ub_to_gm.md)       | Atlas A2            | 向量 UB→GM 搬运（含 atomic add 模式）               |
| [matrix_copy_gm_to_ub](./matrix_copy_gm_to_ub.md) | Atlas A2            | 矩阵 GM→UB 搬运（RowMajor/ColumnMajor，三级自适应） |
| [tile_copy](./tile_copy/README.md)                | Atlas A2, Ascend 950 | 搬运聚合模板（TileCopyGemvAiv / TileCopyGemvAic）   |

### 向量计算

| 组件                          | 适用硬件 | 说明                                                     |
| :---------------------------- | :------- | :------------------------------------------------------- |
| [tile_vmuls](./tile_vmuls.md) | 全架构   | 向量乘以标量（Muls）                                     |
| [tile_vmad](./tile_vmad.md)   | Atlas A2  | 向量-矩阵乘加（Y += A * X），支持 RowMajor / ColumnMajor |
