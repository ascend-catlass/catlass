# CopyUb2Gm / TileCopyTla（UB → GM）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_ub_to_gm.hpp)

[TOC]

## 概述

UB→GM 搬运模块，负责将数据从 UB（Unified Buffer）写回 GM（Global Memory）。RowMajor 排布数据使用 `CopyUb2Gm`（非 TLA）、`TileCopyTla`（TLA RowMajor）或 `TileCopyTlaExt`（TLA PaddingRowMajor）。

> **限制**：仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## API 清单

| API                                      | 风格    | 适用硬件 | Layout          | 说明                           |
| :--------------------------------------- | :------ | :------- | :-------------- | :----------------------------- |
| [CopyUb2Gm](./copy_ub_to_gm.md)          | 非 TLA  | Atlas A2  | RowMajor        | UB RowMajor → GM RowMajor      |
| [TileCopyTla](./tile_copy_tla.md)        | TLA     | Atlas A2  | RowMajor        | TLA 封装，RowMajor 目标        |
| [TileCopyTlaExt](./tile_copy_tla_ext.md) | TLA Ext | Atlas A2  | PaddingRowMajor | TLA 封装，PaddingRowMajor 目标 |

## 调用示例

### 非 TLA

```cpp
#include "catlass/gemm/tile/copy_ub_to_gm.hpp"

using CopyOp = CopyUb2Gm<Arch::AtlasA2,
    Gemm::GemmType<half, layout::RowMajor>>;

auto layoutSrc = layout::RowMajor::MakeLayout<half>(M, N);
auto layoutDst = layout::RowMajor::MakeLayout<half>(M, N);

CopyOp copyOp;
copyOp(dstGm, srcUB, layoutDst, layoutSrc);
```

### TLA

```cpp
#include "catlass/gemm/tile/copy_ub_to_gm.hpp"
#include "tla/tensor.hpp"

auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);
auto srcTensor = tla::MakeTensor(srcUB, srcLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(dstGm, dstLayout, Arch::PositionGM{});

TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA Ext（PaddingRowMajor）

```cpp
auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::PaddingRowMajor>(M, N);
auto srcTensor = tla::MakeTensor(srcUB, srcLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(dstGm, dstLayout, Arch::PositionGM{});

TileCopyTlaExt<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor),
    layout::RowMajor, layout::PaddingRowMajor> copyOp;
copyOp(dstTensor, srcTensor);
```

## 模板选择指南

| 场景                       | 推荐             | 风格    |
| :------------------------- | :--------------- | :------ |
| 普通 RowMajor 写回         | `CopyUb2Gm`      | 非 TLA  |
| TLA 风格 RowMajor 写回     | `TileCopyTla`    | TLA     |
| Padding 后的 RowMajor 写回 | `TileCopyTlaExt` | TLA Ext |
