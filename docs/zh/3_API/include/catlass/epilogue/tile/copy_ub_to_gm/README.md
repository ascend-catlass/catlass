# copy_ub_to_gm

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/copy_ub_to_gm.hpp)

[TOC]

## 概述

`copy_ub_to_gm` 模块实现 epilogue 阶段从 UB（Unified Buffer）到 GM（Global Memory）的数据搬运，将 epilogue 计算结果写回 Global Memory。

TLA 风格的 UB→GM 搬运为独立模块 [CopyUb2GmTla](../copy_ub_to_gm_tla.md)。

## API 清单

| API                                            | 风格   | 适用硬件           | 说明                                       |
| :--------------------------------------------- | :----- | :----------------- | :----------------------------------------- |
| [CopyUb2Gm](./copy_ub_to_gm.md)                | 非 TLA | Atlas A2, Ascend 950 | 基础 UB→GM 搬运（RowMajor / VectorLayout） |
| [CopyUb2GmAligned](./copy_ub_to_gm_aligned.md) | 非 TLA | Atlas A2            | 对齐优化搬运（自动处理大 stride 场景）     |

## 适用硬件

| 硬件型号  | CopyUb2Gm              | CopyUb2GmAligned |
| :-------- | :--------------------- | :--------------- |
| Atlas A2   | RowMajor, VectorLayout | RowMajor         |
| Ascend 950 | RowMajor               | -                |

## 调用示例

### CopyUb2Gm

```cpp
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
using LayoutTagDst = layout::RowMajor;

uint32_t rows = 128;
uint32_t cols = 256;

auto layoutSrc = LayoutTagDst::MakeLayout<Element>(rows, cols);
auto layoutDst = LayoutTagDst::MakeLayout<Element>(rows, cols);

AscendC::LocalTensor<Element> srcTensor;
AscendC::GlobalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagDst>;
using CopyOp = CopyUb2Gm<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
