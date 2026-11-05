# Copy L1 To L0A 模块概述

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_l0a.hpp)

[TOC]

## 概述

`copy_l1_to_l0a` 模块提供将 A 矩阵 tile 块从 L1（Local Memory，A1 Buffer）搬运到 L0A（A2 Buffer）的模板类，支持多种数据排布格式（layout）转换。根据架构不同，实现分为两套：

- **Atlas A2**（ARCH 2201）：[atlasa2/copy_l1_to_l0a.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp)
- **Ascend 950**（ARCH 3510）：[ascend950/copy_l1_to_l0a.hpp](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0a.hpp)

模块包含 **非 TLA 风格**（直接操作 `LocalTensor`）和 **TLA 风格**（通过 `tla::Tensor` 封装）两套 API。

## API 清单

| 组件名                                         | 风格   | 适用硬件            | 说明                                                |
| :--------------------------------------------- | :----- | :------------------ | :-------------------------------------------------- |
| [CopyL1ToL0A](./copy_l1_to_l0a.md)             | 非 TLA | Atlas A2 / Ascend 950 | 基础 L1→L0A 搬运模板，支持多种 layout 转换          |
| [TileCopyTla](./tile_copy_tla.md)              | TLA    | Atlas A2 / Ascend 950 | TLA 风格 L1→L0A 搬运，通过 tla::Tensor 封装简化调用 |
| [TileCopySparseTla](./tile_copy_sparse_tla.md) | TLA    | Atlas A2             | Sparse GEMM L1→L0A 搬运，zN→zZ LoadData3D v2        |

> **说明**：该模块通常不直接使用，而是作为 [TileCopy](../tile_copy/README.md) 的成员类型（`CopyL1ToL0A`），由 [blockMmad](../../block/block_mmad.md) 自动管理。仅在需要自定义 kernel 模板组装时显式声明。

## 适用硬件型号说明

| 硬件型号   | 架构标识          | ARCH 宏                | 支持的非 TLA 模板 | 支持的 TLA 模板 |
| :--------- | :---------------- | :--------------------- | :---------------- | :-------------- |
| Atlas A2   | `Arch::AtlasA2`   | `CATLASS_ARCH == 2201` | CopyL1ToL0A       | TileCopyTla     |
| Ascend 950 | `Arch::Ascend950` | `CATLASS_ARCH == 3510` | CopyL1ToL0A       | TileCopyTla     |

### 架构差异

| 特性              | Atlas A2    | Ascend 950                 |
| :---------------- | :--------- | :------------------------ |
| 目标 L0A layout   | zZ         | zN                        |
| 基础搬运指令      | LoadData2D | LoadData2DParamsV2        |
| l0Batch 批量搬运  | 不支持     | 支持（`operator()` 重载） |
| MX Scale 浮点量化 | 不支持     | 支持（`operator()` 重载） |
| Vector layout     | 不支持     | 支持                      |

## 接口调用示例

### 非 TLA 风格（CopyL1ToL0A）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using L1Type = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;

uint32_t row = 256;
uint32_t col = 256;

// 构造 L1 上的 zN layout 和 L0A 上的 zZ layout
auto layoutSrc = layout::zN::MakeLayout<Element>(row, col);
auto layoutDst = layout::zZ::MakeLayout<Element>(row, col);

AscendC::LocalTensor<Element> srcL1Tensor;
AscendC::LocalTensor<Element> dstL0ATensor;

// 实例化并调用
using CopyOp = CopyL1ToL0A<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0ATensor, srcL1Tensor, layoutDst, layoutSrc);
```

### TLA 风格（TileCopyTla）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;

// 通过 tla::MakeLayout 创建 Layout
auto layoutSrc = tla::MakeLayout<half, layout::zN>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zZ>(M, K);

// 通过 tla::MakeTensor 构造 TLA Tensor
AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0ATensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// 实例化并调用（SFINAE 根据 src/dst layout trait 自动匹配偏特化）
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA 风格 - 转置搬运（Atlas A2）

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zZ>(M, K);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// isnZ<LayoutSrc> && iszZ<LayoutDst> → 自动匹配转置偏特化
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA 风格 - Ascend 950 基础搬运

```cpp
// Ascend 950 目标 layout 为 zN
auto layoutSrc = tla::MakeLayout<half, layout::zN>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zN>(M, K);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// Ascend950: zN L1 → zN L0A
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA 风格 - Ascend 950 l0Batch 批量搬运

```cpp
uint32_t l0Batch = 4;

// l0Batch 重载：多 batch 连续搬运
copyOp(dstTensor, srcTensor, l0Batch);
```

### TLA 风格 - Ascend 950 MX Scale 搬运

```cpp
using ElementSrc = float8_e4m3_t;
using ElementDst = AscendC::mx_fp8_e4m3_t;
using ElementMxScale = float8_e8m0_t;

// MX Scale 的 K 方向维度：每 MX_SCALE_GROUP_NUM（32）个元素共享一个 scale 值
const uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(K);

// 源数据 layout（L1 zN）
auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(M, K);
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});

// 目标数据 layout（L0A zN，元素类型为 mx_fp8）
auto layoutDst = tla::MakeLayout<ElementDst, layout::zN>(M, K);
auto dstTensor = tla::MakeTensor(dstL0ATensor, layoutDst, Arch::PositionL0A{});

// MX Scale layout（L1 zZ，使用 MakeMxScaleLayout 构造）
auto layoutScaleL1 = tla::MakeMxScaleLayout<ElementMxScale, layout::zZ, false>(M, mxScaleK);

AscendC::LocalTensor<ElementMxScale> scaleL1Tensor;
auto scaleTensor = tla::MakeTensor(scaleL1Tensor, layoutScaleL1, Arch::PositionL1{});

// MX Scale 重载：L1 zN 源数据 + L1 zZ scale → L0A zN mx 数据
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor, scaleTensor);
```

## 模板选择指南

| 场景                          | 推荐模板                                       |
| :---------------------------- | :--------------------------------------------- |
| 通用矩阵乘 tile L1→L0A 搬运   | `CopyL1ToL0A`（非 TLA）或 `TileCopyTla`（TLA） |
| 转置搬运（nZ → zZ / nZ → zN） | `CopyL1ToL0A` 或 `TileCopyTla`（自动匹配）     |
| Ascend 950 多 batch 搬运       | `TileCopyTla`（l0Batch 重载）                  |
| Ascend 950 MX 浮点量化         | `TileCopyTla`（MX Scale 重载）                 |
| 卷积场景 NDC1HWC0 搬运        | `CopyL1ToL0A`（非 TLA，NDC1HWC0 偏特化）       |
| 已使用 TLA 编程范式           | `TileCopyTla`（统一风格）                      |
