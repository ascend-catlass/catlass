# Copy Gm To L1 模块概述

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## 概述

`copy_gm_to_l1` 模块提供将 tile 块从 GM（Global Memory）搬运到 L1（Local Memory）的模板类，支持多种数据排布格式（layout）转换。根据架构不同，实现分为两套：

- **Atlas A2**（ARCH 2201）：[atlasa2/copy_gm_to_l1.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp)
- **Ascend 950**（ARCH 3510）：[ascend950/copy_gm_to_l1.hpp](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp)

模块包含 **非 TLA 风格**（直接操作 `LocalTensor` / `GlobalTensor`）和 **TLA 风格**（通过 `tla::Tensor` 封装）两套 API。

## API 清单

| 组件名                                                              | 风格   | 适用硬件            | 说明                                                                    |
| :------------------------------------------------------------------ | :----- | :------------------ | :---------------------------------------------------------------------- |
| [CopyGmToL1](./copy_gm_to_l1.md)                                    | 非 TLA | Atlas A2 / Ascend 950 | 基础 GM→L1 搬运模板，支持多种 layout 转换                               |
| [CopyGmToL1IntervalDataCopy](./copy_gm_to_l1_interval_data_copy.md) | 非 TLA | Atlas A2             | 基于 strided DataCopy 的逐行/逐列搬运，适用于矮宽/高窄数据块            |
| [CopyGmToL1GMMPTD](./copy_gm_to_l1_gmmptd.md)                       | 非 TLA | Atlas A2 / Ascend 950 | GMM PTD 场景专用搬运，含单行优化和手动 stride 接口                      |
| [CopyGmToL1DynamicOptimized](./copy_gm_to_l1_dynamic_optimized.md)  | 非 TLA | Atlas A2 / Ascend 950 | 运行时动态选择最优搬运策略（小矩阵用 strided DataCopy，大矩阵用 Nd2Nz） |
| [TileCopyTla](./tile_copy_tla.md)                                   | TLA    | Atlas A2 / Ascend 950 | TLA 风格 GM→L1 搬运，通过 tla::Tensor 封装简化调用                      |
| [TileCopyTlaExt](./tile_copy_tla_ext.md)                            | TLA    | Atlas A2             | TLA 扩展搬运，支持 ActualShape 部分搬运和 Padding layout                |
| [TileCopySparseTla](./tile_copy_sparse_tla.md)                      | TLA    | Atlas A2             | Sparse GEMM GM→L1 搬运，支持 RowMajor/ColumnMajor/zN/nZ→zN/nZ           |
| [TileCopyFAQTla](./tile_copy_faq_tla.md)                            | TLA    | Atlas A2             | FlashAttention LoadQ 搬运，支持 3D 多矩阵 GM→L1 zN 转换                 |

## 适用硬件型号说明

| 硬件型号   | 架构标识          | ARCH 宏                | 支持的非 TLA 模板                                                                       | 支持的 TLA 模板              |
| :--------- | :---------------- | :--------------------- | :-------------------------------------------------------------------------------------- | :--------------------------- |
| Atlas A2   | `Arch::AtlasA2`   | `CATLASS_ARCH == 2201` | CopyGmToL1 / CopyGmToL1IntervalDataCopy / CopyGmToL1GMMPTD / CopyGmToL1DynamicOptimized | TileCopyTla / TileCopyTlaExt |
| Ascend 950 | `Arch::Ascend950` | `CATLASS_ARCH == 3510` | CopyGmToL1 / CopyGmToL1GMMPTD / CopyGmToL1DynamicOptimized                              | TileCopyTla                  |

## 接口调用示例

### 非 TLA 风格（CopyGmToL1）

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;
using ElementSrc = half;
using ElementDst = half;

// 定义 GM 上的 RowMajor 数据（A 矩阵）
using GmType = Gemm::GemmType<ElementSrc, LayoutTagSrc>;
// 定义 L1 上的 zN 数据
using L1Type = Gemm::GemmType<ElementDst, LayoutTagDst, AscendC::TPosition::A1>;

uint32_t row = 256;
uint32_t col = 256;

// 构造 GM 上的 RowMajor layout
auto layoutSrc = LayoutTagSrc::MakeLayout<ElementSrc>(row, col);
// 构造 L1 上的 zN layout
auto layoutDst = LayoutTagDst::MakeLayout<ElementDst>(row, col);

AscendC::GlobalTensor<ElementSrc> srcTensor;
AscendC::LocalTensor<ElementDst> dstTensor;

// 实例化并调用
using CopyOp = CopyGmToL1<Arch::AtlasA2, GmType, L1Type>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### TLA 风格（TileCopyTla）

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;

// 通过 tla::MakeLayout 创建 Layout（由 LayoutTag + Element + 维度自动推导 Shape/Stride）
auto layoutSrc = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zN>(M, K);

// 通过 tla::MakeTensor 构造 TLA Tensor
AscendC::GlobalTensor<half> srcGmTensor;
AscendC::LocalTensor<half> dstL1Tensor;
auto srcTensor = tla::MakeTensor(srcGmTensor, layoutSrc, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, layoutDst, Arch::PositionL1{});

// 实例化并调用
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA 风格（TileCopyTlaExt）

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;
const uint32_t actualM = 128;
const uint32_t actualK = 128;

// 通过 tla::MakeLayout 创建 Layout
auto layoutSrc = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zN>(M, K);

// 通过 tla::MakeTensor 构造 TLA Tensor
AscendC::GlobalTensor<half> srcGmTensor;
AscendC::LocalTensor<half> dstL1Tensor;
auto srcTensor = tla::MakeTensor(srcGmTensor, layoutSrc, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, layoutDst, Arch::PositionL1{});

// 实例化 TileCopyTlaExt（LayoutTagSrc/LayoutTagDst 决定搬运策略，与 tensor 的 layout 无关）
TileCopyTlaExt<Arch::AtlasA2,
    decltype(srcTensor), decltype(dstTensor),
    layout::RowMajor, layout::zN> copyOp;

// 指定实际搬运的数据块形状（可小于 tensor 的完整 shape）
tla::Shape<uint32_t, uint32_t> actualShape(actualM, actualK);
copyOp(dstTensor, srcTensor, actualShape);
```

### 动态优化风格（CopyGmToL1DynamicOptimized）

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;
using ElementDst = half;

// 定义 GM 上的 Gemm 类型
using GmType = Gemm::GemmType<ElementDst, LayoutTagSrc>;
// 定义 L1 上的 Gemm 类型
using L1Type = Gemm::GemmType<ElementDst, LayoutTagDst, AscendC::TPosition::A1>;

uint32_t row = 256;
uint32_t col = 256;

// 构造 layout
auto layoutSrc = LayoutTagSrc::MakeLayout<ElementDst>(row, col);
auto layoutDst = LayoutTagDst::MakeLayout<ElementDst>(row, col);

AscendC::GlobalTensor<ElementDst> srcTensor;
AscendC::LocalTensor<ElementDst> dstTensor;

// 实例化 CopyGmToL1DynamicOptimized
// 内部会根据 row/col 自动选择 Nd2Nz 或 strided DataCopy
using CopyOp = CopyGmToL1DynamicOptimized<Arch::AtlasA2, GmType, L1Type>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### GMM PTD 风格（CopyGmToL1GMMPTD）

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;
using ElementDst = half;

// GMM PTD 场景只需指定 GmType（L1Type 默认为 void，由偏特化自动推导）
using GmType = Gemm::GemmType<ElementDst, LayoutTagSrc>;

uint32_t row = 256;
uint32_t col = 256;

// 构造 layout
auto layoutSrc = LayoutTagSrc::MakeLayout<ElementDst>(row, col);
auto layoutDst = LayoutTagDst::MakeLayout<ElementDst>(row, col);

AscendC::GlobalTensor<ElementDst> srcTensor;
AscendC::LocalTensor<ElementDst> dstTensor;

// 基础调用
using CopyOp = CopyGmToL1GMMPTD<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);

// 扩展调用：手动指定 stride（多矩阵搬运场景）
// copyOp(dstTensor, srcTensor, layoutDst, layoutSrc,
//        ndNum, srcNdMatrixStride, dstNzNStride, dstNzMatrixStride, dstNzC0Stride);
```

### 间隔搬运风格（CopyGmToL1IntervalDataCopy）

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;

// CopyGmToL1IntervalDataCopy 当前仅支持 half 类型和 Atlas A2 架构
using GmType = Gemm::GemmType<half, LayoutTagSrc>;

uint32_t row = 256;
uint32_t col = 256;

// 构造 layout
auto layoutSrc = LayoutTagSrc::MakeLayout<half>(row, col);
auto layoutDst = LayoutTagDst::MakeLayout<half>(row, col);

AscendC::GlobalTensor<half> srcTensor;
AscendC::LocalTensor<half> dstTensor;

// 使用 strided DataCopy 逐行搬运，适用于矮宽/高窄数据块
using CopyOp = CopyGmToL1IntervalDataCopy<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

## 模板选择指南

| 场景                               | 推荐模板                                      |
| :--------------------------------- | :-------------------------------------------- |
| 通用矩阵乘 tile 搬运               | `CopyGmToL1`（非 TLA）或 `TileCopyTla`（TLA） |
| 数据块形状不确定，需要运行时自适应 | `CopyGmToL1DynamicOptimized`                  |
| GMM PTD 场景，需要手动控制 stride  | `CopyGmToL1GMMPTD`                            |
| 矮宽/高窄数据块（仅 half 类型）    | `CopyGmToL1IntervalDataCopy`                  |
| 需要部分搬运或 Padding 场景        | `TileCopyTlaExt`                              |
