# CopyL0CToGm / CopyL0CToGmTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_l0c_to_gm.hpp)

[TOC]

## 概述

L0C→GM 搬运模块，负责将矩阵乘累加结果从 L0C（Accumulator Buffer, `CO1`）搬运到 GM（Global Memory）。支持类型转换、per-tensor / per-channel 量化/反量化和 ReLU 激活。

## API 清单

| API                                                    | 风格   | 适用硬件            | 说明                                  |
| :----------------------------------------------------- | :----- | :------------------ | :------------------------------------ |
| [CopyL0CToGm](./copy_l0c_to_gm.md)                     | 非 TLA | Atlas A2 / Ascend 950 | L0C→GM，支持 RowMajor / zN / NDC1HWC0 |
| [CopyL0CToGmTla](./tile_copy_tla.md)                   | TLA    | Atlas A2 / Ascend 950 | L0C→GM TLA 封装，SFINAE 自动派发      |
| [CopyL0CToGmSparseTla](./copy_l0c_to_gm_sparse_tla.md) | TLA    | Atlas A2             | Sparse GEMM L0C→GM 搬运，Fixpipe v220 |

## 适用硬件型号

| 架构                | 非 TLA                   | TLA                                      |
| :------------------ | :----------------------- | :--------------------------------------- |
| Atlas A2（`2201`）   | RowMajor / zN / NDC1HWC0 | RowMajor / zN (NO_QUANT)                 |
| Ascend 950（`3510`） | RowMajor / zN / NDC1HWC0 | RowMajor / zN + PER_TENSOR / PER_CHANNEL |

## 调用示例

### 非 TLA：NO_QUANT

```cpp
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"

using CopyOp = CopyL0CToGm<Arch::AtlasA2, float,
    Gemm::GemmType<half, layout::RowMajor>>;

auto dstLayout = layout::RowMajor::MakeLayout<half>(M, N);
auto srcLayout = layout::zN::MakeLayout<float>(M, N);

CopyOp copyOp;
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```

### 非 TLA：PER_TENSOR

```cpp
using CopyOp = CopyL0CToGm<Arch::AtlasA2, int32_t,
    Gemm::GemmType<half, layout::RowMajor>, ScaleGranularity::PER_TENSOR>;

CopyOp::Params params(0.5f);
CopyOp copyOp(params);
copyOp(dstGmTensor, srcL0CTensor, dstLayout, srcLayout);
```

### TLA：NO_QUANT

```cpp
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "tla/tensor.hpp"

auto srcLayout = tla::MakeLayout<float, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);
auto srcTensor = tla::MakeTensor(srcL0C, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGm, dstLayout, Arch::PositionGM{});

CopyL0CToGmTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

## 模板选择指南

| 场景                  | 推荐                        | 风格   |
| :-------------------- | :-------------------------- | :----- |
| 常规 GM 写回          | `CopyL0CToGm`               | 非 TLA |
| GM + per-tensor 量化  | `CopyL0CToGm` + PER_TENSOR  | 非 TLA |
| GM + per-channel 量化 | `CopyL0CToGm` + PER_CHANNEL | 非 TLA |
| GM + ReLU             | `CopyL0CToGm` + ReluEnable  | 非 TLA |
| TLA 风格 GM 写回      | `CopyL0CToGmTla`            | TLA    |
