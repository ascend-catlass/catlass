# TileCopyTla 系列（TLA 搬运模板基类）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## 概述

`tile_copy_tla.hpp` 定义 CATLASS Tile 层所有 TLA 搬运模板的**基类声明**，包括 `TileCopyTla`、`TileCopyTlaExt`、`TileCopySparseTla`、`CopyL1ToL0BSparseTla`、`TileCopyFAQTla`。具体实现位于 `atlasa2/` 和 `ascend950/` 子目录下的对应 `copy_*.hpp` 文件中。

## API 清单

| 模板                                                   | 分发方式       | 适用硬件            | 说明                           |
| :----------------------------------------------------- | :------------- | :------------------ | :----------------------------- |
| [TileCopyTla](./tile_copy_tla.md)                      | SFINAE trait   | Atlas A2 + Ascend 950 | 核心 TLA 搬运，自动匹配 layout |
| [TileCopyTlaExt](./tile_copy_tla_ext.md)               | 显式 LayoutTag | Atlas A2             | 扩展版，支持 Padding layout    |
| [TileCopySparseTla](./tile_copy_sparse_tla.md)         | SFINAE trait   | Atlas A2             | 稀疏 GEMM 搬运                 |
| [CopyL1ToL0BSparseTla](./copy_l1_to_l0b_sparse_tla.md) | SFINAE trait   | Atlas A2             | 稀疏 B L1→L0B（带 index）      |
| [TileCopyFAQTla](./tile_copy_faq_tla.md)               | 固定匹配       | Atlas A2             | FA LoadQ GM→L1 zN              |

## 模板关系图

```cpp
tile_copy_tla.hpp
├── TileCopyTla           → 9 个偏特化（GM→L1, L1→L0A/B, GM→UB, UB→GM, L1→BT）
├── TileCopyTlaExt        → 3 个偏特化（PaddingRowMajor, PaddingColumnMajor）
├── TileCopySparseTla     → 3 个偏特化（GM→L1A, GM→L1B, L1→L0A）
├── CopyL1ToL0BSparseTla  → 1 个偏特化（L1→L0B + index）
└── TileCopyFAQTla        → 1 个偏特化（GM RowMajor → L1 zN）
```

## 分模块实现索引

各偏特化实现分布如下：

| 实现文件                       | 包含的偏特化                                                           |
| :----------------------------- | :--------------------------------------------------------------------- |
| `atlasa2/copy_gm_to_l1.hpp`    | TileCopyTla*3, TileCopyTlaExt*2, TileCopySparseTla*2, TileCopyFAQTla*1 |
| `atlasa2/copy_gm_to_ub.hpp`    | TileCopyTla*1                                                          |
| `atlasa2/copy_l1_to_l0a.hpp`   | TileCopyTla*2, TileCopySparseTla*1                                     |
| `atlasa2/copy_l1_to_l0b.hpp`   | TileCopyTla*2, CopyL1ToL0BSparseTla*1                                  |
| `atlasa2/copy_ub_to_gm.hpp`    | TileCopyTla*1, TileCopyTlaExt*1                                        |
| `ascend950/copy_gm_to_l1.hpp`  | TileCopyTla*2                                                          |
| `ascend950/copy_l1_to_l0a.hpp` | TileCopyTla*2                                                          |
| `ascend950/copy_l1_to_l0b.hpp` | TileCopyTla*1                                                          |
| `ascend950/copy_l1_to_bt.hpp`  | TileCopyTla*1                                                          |

## 模板选择指南

| 场景                                       | 推荐                                         |
| :----------------------------------------- | :------------------------------------------- |
| 常规 GM→L1 搬运                            | `TileCopyTla`                                |
| L1→L0A/L0B 搬运                            | `TileCopyTla`                                |
| GM 源为 PaddingRowMajor/PaddingColumnMajor | `TileCopyTlaExt`                             |
| 稀疏 GEMM                                  | `TileCopySparseTla` + `CopyL1ToL0BSparseTla` |
| Flash Attention Q 加载                      | `TileCopyFAQTla`                             |

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"   // 拉入偏特化
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

// 构造 TLA tensor
auto gmTensor = tla::MakeTensor(gmData,
    tla::MakeLayout<half, layout::RowMajor>(M, K), Arch::PositionGM{});
auto l1Tensor = tla::MakeTensor(l1Data,
    tla::MakeLayout<half, layout::RowMajor>(M, K), Arch::PositionL1{});

// SFINAE 自动分发
TileCopyTla<Arch::AtlasA2, decltype(gmTensor), decltype(l1Tensor)> copyOp;
copyOp(l1Tensor, gmTensor);
```
