# TileMmad / TileMmadTla（Tile 层 Mmad 计算）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_mmad.hpp)

[TOC]

## 概述

Tile 层 Mmad 模块完成 L0A*L0B → L0C 的矩阵乘加 `C += A * B`，直接调用 `AscendC::Mmad` 硬件指令。L0A/L0B/L0C 数据排布固定为 zZ / nZ / zN。

提供非 TLA（`TileMmad`）和 TLA（`TileMmadTla`）两套风格。

## API 清单

| API                               | 风格   | 支持 Bias | L0 Batch | 自动维度 | 架构                | 说明                           |
| :-------------------------------- | :----- | :-------- | :------- | :------- | :------------------ | :----------------------------- |
| [TileMmad](./tile_mmad.md)        | 非 TLA | ✓         | —        | —        | Atlas A2 + Ascend 950 | 直接操作 AscendC::LocalTensor  |
| [TileMmadTla](./tile_mmad_tla.md) | TLA    | ✓         | ✓        | ✓        | Atlas A2 + Ascend 950 | tla::Tensor 封装，自动维度提取 |

## 功能对比

| 特性              | TileMmad                  | TileMmadTla                        |
| :---------------- | :------------------------ | :--------------------------------- |
| 操作数类型        | `AscendC::LocalTensor<T>` | `tla::Tensor<LocalTensor<T>, ...>` |
| 无 Bias mmad      | ✓                         | ✓                                  |
| 带 Bias mmad      | ✓                         | ✓                                  |
| L0 Batch mmad     | —                         | ✓                                  |
| 自动维度提取      | —                         | ✓（模式 4）                        |
| unitFlag 并行搬运 | ✓                         | ✓                                  |
| kDirectionAlign   | Atlas A2 float + nZ L1A    | 同左                               |
| GEMV 模式控制     | Ascend 950 `disableGemv`   | Ascend 950 `disableGemv`            |
| GEMV 自动规避     | —                         | Atlas A2 M=1→M=16（模式 4）         |

## 调用示例

### 非 TLA

```cpp
#include "catlass/gemm/tile/tile_mmad.hpp"

using MmadOp = Tile::TileMmad<Arch::AtlasA2,
    Gemm::GemmType<half, layout::zZ>,
    Gemm::GemmType<half, layout::nZ>,
    void>;

MmadOp mmadOp;
mmadOp(l0CTensor, l0ATensor, l0BTensor, 64, 64, 32);
```

### TLA（推荐）

```cpp
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/tensor.hpp"

auto l0cTensor = tla::MakeTensor(l0c, l0cLayout, Arch::PositionL0C{});
auto l0aTensor = tla::MakeTensor(l0a, l0aLayout, Arch::PositionL0A{});
auto l0bTensor = tla::MakeTensor(l0b, l0bLayout, Arch::PositionL0B{});

Tile::TileMmadTla<Arch::AtlasA2, half, layout::zN> mmadOp;
mmadOp(l0cTensor, l0aTensor, l0bTensor);  // 自动提取 m/n/k
```

## 模板选择指南

| 场景                                      | 推荐                             |
| :---------------------------------------- | :------------------------------- |
| 传统 blockMmad 组装                       | `TileMmad`                       |
| TLA 风格 kernel（PackedTileCopyTla 配合） | `TileMmadTla`（模式 1 或 4）     |
| 需要 Bias 累加                            | `TileMmadTla`（模式 2）          |
| Flash Attention L0 Batch                   | `TileMmadTla`（模式 3）          |
| 追求代码简洁                              | `TileMmadTla`（模式 4 自动提取） |
