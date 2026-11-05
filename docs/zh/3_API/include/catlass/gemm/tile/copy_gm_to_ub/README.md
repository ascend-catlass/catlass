# CopyGm2Ub / TileCopyTla（GM → UB）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_ub.hpp)

[TOC]

## 概述

GM→UB 搬运模块，负责将数据从 GM（Global Memory）搬运到 UB（Unified Buffer）。VectorLayout 的一维向量数据使用 `CopyGm2Ub`，RowMajor 的二维矩阵数据使用 `TileCopyTla`。

> **限制**：仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## API 清单

| API                               | 风格   | 适用硬件 | Layout       | 说明                      |
| :-------------------------------- | :----- | :------- | :----------- | :------------------------ |
| [CopyGm2Ub](./copy_gm_to_ub.md)   | 非 TLA | Atlas A2  | VectorLayout | GM 一维向量 → UB          |
| [TileCopyTla](./tile_copy_tla.md) | TLA    | Atlas A2  | RowMajor     | GM RowMajor → UB RowMajor |

## 调用示例

### 非 TLA

```cpp
#include "catlass/gemm/tile/copy_gm_to_ub.hpp"

using CopyOp = CopyGm2Ub<Arch::AtlasA2,
    Gemm::GemmType<half, layout::VectorLayout>>;

auto layoutSrc = layout::VectorLayout(len);
auto layoutDst = layout::VectorLayout(len);

CopyOp copyOp;
copyOp(dstUB, srcGm, layoutDst, layoutSrc);
```

### TLA

```cpp
#include "catlass/gemm/tile/copy_gm_to_ub.hpp"
#include "tla/tensor.hpp"

auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto srcTensor = tla::MakeTensor(srcGm, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstUB, dstLayout, Arch::PositionUB{});

TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

## 模板选择指南

| 场景                       | 推荐          | 风格   |
| :------------------------- | :------------ | :----- |
| 一维向量搬运（Bias/Scale） | `CopyGm2Ub`   | 非 TLA |
| 二维矩阵搬运               | `TileCopyTla` | TLA    |
