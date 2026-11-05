# TileCopy（搬运模板集合）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 概述

`TileCopy` 系列模板是 GEMM Tile 层的**搬运模板集合**，通过模板参数推导出 GM→L1、L1→L0A/B、L0C→GM(Dst/UB)、Bias、Scale 等所有数据搬运子组件的类型，供 blockMmad 在组装时选用。

所有模板只提供 `using` 类型别名，不做任何实际计算。

## API 清单

### 非 TLA 风格

| 模板                                                                       | 适用硬件            | 继承关系      | 说明                           |
| :------------------------------------------------------------------------- | :------------------ | :------------ | :----------------------------- |
| [TileCopy](./tile_copy.md)                                                 | Atlas A2 + Ascend 950 | 基类          | 基础搬运集合                   |
| [TileCopyWithPrologueDeqPerTensor](./tile_copy_prologue_deq_per_tensor.md) | Atlas A2             | —             | +Prologue + per-tensor dequant |
| [TileCopyWithPrologue](./tile_copy_prologue.md)                            | Atlas A2             | —             | +Prologue 预处理               |
| [TileCopyGemm](./tile_copy_gemm.md)                                        | Atlas A2 + Ascend 950 | —             | GEMM 专用选择器                |
| [ConvTileCopy](./conv_tile_copy.md)                                        | Atlas A2 + Ascend 950 | —             | Conv 专用                      |
| [ReluTileCopy](./relu_tile_copy.md)                                        | Atlas A2 + Ascend 950 | 继承 TileCopy | +ReLU 写回                     |
| [QuantTileCopy](./quant_tile_copy.md)                                      | Atlas A2             | 继承 TileCopy | +Scale/FP 通道 + 随路量化      |

### TLA 风格

| 模板                                                          | 适用硬件            | 继承关系               | 说明             |
| :------------------------------------------------------------ | :------------------ | :--------------------- | :--------------- |
| [SparseTileCopyTla](./sparse_tile_copy_tla.md)                | Atlas A2             | —                      | 稀疏 GEMM（TLA） |
| [PackedTileCopyTla](./packed_tile_copy_tla.md)                | Atlas A2 + Ascend 950 | —                      | 核心 TLA 集合    |
| [PaddingPackedTileCopyTla](./padding_packed_tile_copy_tla.md) | Atlas A2 + Ascend 950 | —                      | +Padding 支持    |
| [PackedTileCopyTlaToUB](./packed_tile_copy_tla_to_ub.md)      | Ascend 950           | 继承 PackedTileCopyTla | +UB 目标         |
| [PackedMxTileCopyTla](./packed_mx_tile_copy_tla.md)           | Atlas A2             | 继承 PackedTileCopyTla | +MX Scale 通道   |
| [PackedMxA8W4TileCopyTla](./packed_mx_a8w4_tile_copy_tla.md)  | Atlas A2             | 继承 PackedTileCopyTla | +MX Scale + A8W4 |

## 模板继承关系

```cpp
TileCopy
├── ReluTileCopy            (重写 CopyL0CToGm → ReLU)
├── QuantTileCopy           (重写 CopyL0CToGm + 新增 Scale/FP 通道)
│
PackedTileCopyTla
├── PackedTileCopyTlaToUB   (重写 CopyL0CToDst → UB)
├── PackedMxTileCopyTla     (新增 MxScale 通道)
└── PackedMxA8W4TileCopyTla (新增 MxScale + 重写 L1→L0B)
```

## 模板选择指南

| 场景                         | 推荐模板                           | 风格        | 架构      |
| :--------------------------- | :--------------------------------- | :---------- | :-------- |
| 通用 GEMM（FP16 / BF16）     | `TileCopy` / `PackedTileCopyTla`   | 非TLA / TLA | 均可      |
| GEMM 带显式 L1/L0 推导       | `TileCopyGemm`                     | 非TLA       | 均可      |
| INT8 量化推理                | `QuantTileCopy`                    | 非TLA       | Atlas A2   |
| Prologue 反量化              | `TileCopyWithPrologueDeqPerTensor` | 非TLA       | Atlas A2   |
| Prologue 预处理（INT4→INT8） | `TileCopyWithPrologue`             | 非TLA       | Atlas A2   |
| ReLU 激活                    | `ReluTileCopy`                     | 非TLA       | 均可      |
| Conv GEMM（Im2Col）          | `ConvTileCopy`                     | 非TLA       | 均可      |
| 稀疏 GEMM                    | `SparseTileCopyTla`                | TLA         | Atlas A2   |
| Padding Block GEMM           | `PaddingPackedTileCopyTla`         | TLA         | 均可      |
| L0C→UB（Ascend 950）          | `PackedTileCopyTlaToUB`            | TLA         | Ascend 950 |
| FP8 MX Scale                 | `PackedMxTileCopyTla`              | TLA         | Atlas A2   |
| FP8 MX Scale + INT4 weight   | `PackedMxA8W4TileCopyTla`          | TLA         | Atlas A2   |
