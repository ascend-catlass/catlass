# Copy L1 To BT 模块概述

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_bt.hpp)

[TOC]

## 概述

`copy_l1_to_bt` 模块提供将 Bias Table（一维向量）从 L1（Local Memory，A1 Buffer）搬运到 BT（Bias Table Buffer，C2 Buffer）的模板类。Bias Table 用于矩阵乘的 Bias 加法和量化反量化操作。

由于 Bias 数据是一维向量，该模块固定使用 `VectorLayout`（rank=1、stride=1 的一维排布）作为唯一的数据排布格式。根据架构不同，实现分为两套：

- **Atlas A2**（ARCH 2201）：[atlasa2/copy_l1_to_bt.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_bt.hpp)
- **Ascend 950**（ARCH 3510）：[ascend950/copy_l1_to_bt.hpp](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_bt.hpp)

模块包含 **非 TLA 风格**（直接操作 `LocalTensor`）和 **TLA 风格**（通过 `tla::Tensor` 封装）两套 API。

## API 清单

| 组件名                            | 风格   | 适用硬件            | 说明                                        |
| :-------------------------------- | :----- | :------------------ | :------------------------------------------ |
| [CopyL1ToBT](./copy_l1_to_bt.md)  | 非 TLA | Atlas A2 / Ascend 950 | 基础 L1→BT 一维向量搬运，使用 DataCopy 指令 |
| [TileCopyTla](./tile_copy_tla.md) | TLA    | Ascend 950           | TLA 风格 L1→BT 搬运，通过 tla::Tensor 封装  |

> **说明**：该模块通常不直接使用，而是作为 [TileCopy](../tile_copy/README.md) 的成员类型（`CopyL1ToBT`），由 [blockMmad](../../block/block_mmad.md) 自动管理。仅在需要自定义 kernel 模板组装时显式声明。

## 适用硬件型号说明

| 硬件型号   | 架构标识          | ARCH 宏                | 支持的非 TLA 模板 | 支持的 TLA 模板 |
| :--------- | :---------------- | :--------------------- | :---------------- | :-------------- |
| Atlas A2   | `Arch::AtlasA2`   | `CATLASS_ARCH == 2201` | CopyL1ToBT        | —               |
| Ascend 950 | `Arch::Ascend950` | `CATLASS_ARCH == 3510` | CopyL1ToBT        | TileCopyTla     |

### 架构差异

| 特性              | Atlas A2             | Ascend 950              |
| :---------------- | :------------------ | :--------------------- |
| 目标 Buffer       | C2（Bias Table）    | C2（Bias Table）       |
| 搬运指令          | `AscendC::DataCopy` | `AscendC::DataCopy`    |
| blockLen 对齐基准 | `BYTE_PER_C2`       | `BYTE_PER_C0`          |
| B32 类型对齐处理  | 不处理              | `RoundUp(blockLen, 2)` |
| 源/目标元素类型   | 可不同              | 可不同                 |
| TLA 支持          | 不支持              | 支持                   |

## 接口调用示例

### 非 TLA 风格（CopyL1ToBT）

```cpp
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float;
using ElementDst = half;
using L1Type = Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>;

uint32_t vecLen = 256;

auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstBTTensor;

using CopyOp = CopyL1ToBT<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstBTTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### TLA 风格（TileCopyTla，Ascend 950）

```cpp
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float;
using ElementDst = half;

const uint32_t vecLen = 256;

auto layoutSrc = tla::MakeLayout<ElementSrc, layout::VectorLayout>(1, vecLen);
auto layoutDst = tla::MakeLayout<ElementDst, layout::VectorLayout>(1, vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstBTTensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstBTTensor, layoutDst, Arch::PositionBias{});

TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

## 模板选择指南

| 场景                       | 推荐模板                                      |
| :------------------------- | :-------------------------------------------- |
| 通用 Bias Table L1→BT 搬运 | `CopyL1ToBT`（非 TLA）                        |
| Atlas A2 Bias Table 搬运    | `CopyL1ToBT`（非 TLA）                        |
| Ascend 950 Bias Table 搬运  | `CopyL1ToBT`（非 TLA）或 `TileCopyTla`（TLA） |
| 已使用 TLA 编程范式        | `TileCopyTla`（统一风格，仅 Ascend 950）       |
