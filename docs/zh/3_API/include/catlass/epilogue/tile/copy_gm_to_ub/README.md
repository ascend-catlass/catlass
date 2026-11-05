# copy_gm_to_ub

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub.hpp)

[TOC]

## 概述

`copy_gm_to_ub` 模块实现 epilogue 阶段从 GM（Global Memory）到 UB（Unified Buffer）的数据搬运，包含基础搬运、per-token scale 搬运和对齐优化搬运三种结构体。

TLA 风格的 GM→UB 搬运为独立模块 [CopyGm2UbTla](../copy_gm_to_ub_tla.md)。

## API 清单

| API                                                     | 风格   | 适用硬件           | 说明                                                         |
| :------------------------------------------------------ | :----- | :----------------- | :----------------------------------------------------------- |
| [CopyGm2Ub](./copy_gm_to_ub.md)                         | 非 TLA | Atlas A2, Ascend 950 | 基础 GM→UB 搬运（RowMajor / VectorLayout）                   |
| [CopyPerTokenScale2Ub](./copy_per_token_scale_to_ub.md) | 非 TLA | Atlas A2            | Per-token scale 专用搬运（ColumnMajor→RowMajor，带 padding） |
| [CopyGm2UbAligned](./copy_gm_to_ub_aligned.md)          | 非 TLA | Atlas A2            | 对齐优化搬运（自动处理大 stride 场景）                       |

## 适用硬件

| 硬件型号  | CopyGm2Ub              | CopyPerTokenScale2Ub | CopyGm2UbAligned |
| :-------- | :--------------------- | :------------------- | :--------------- |
| Atlas A2   | RowMajor, VectorLayout | ColumnMajor→RowMajor | RowMajor         |
| Ascend 950 | RowMajor, VectorLayout | -                    | -                |

## 调用示例

### CopyGm2Ub

```cpp
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
using LayoutTagSrc = layout::RowMajor;

uint32_t rows = 128;
uint32_t cols = 256;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(rows, cols);
auto layoutDst = LayoutTagSrc::MakeLayout<Element>(rows, cols);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;
using CopyOp = CopyGm2Ub<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
