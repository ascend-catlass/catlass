# Copy L1 To L0B 模块概述

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_l0b.hpp)

[TOC]

## 概述

`copy_l1_to_l0b` 模块提供将 B 矩阵 tile 块从 L1（Local Memory，B1 Buffer）搬运到 L0B（B2 Buffer）的模板类，支持多种数据排布格式（layout）转换。根据架构不同，实现分为两套：

- **Atlas A2**（ARCH 2201）：[atlasa2/copy_l1_to_l0b.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp)
- **Ascend 950**（ARCH 3510）：[ascend950/copy_l1_to_l0b.hpp](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp)

模块包含 **非 TLA 风格**（直接操作 `LocalTensor`）和 **TLA 风格**（通过 `tla::Tensor` 封装）两套 API。

## API 清单

| 组件名                                                 | 风格   | 适用硬件            | 说明                                                  |
| :----------------------------------------------------- | :----- | :------------------ | :---------------------------------------------------- |
| [CopyL1ToL0B](./copy_l1_to_l0b.md)                     | 非 TLA | Atlas A2 / Ascend 950 | 基础 L1→L0B 搬运模板，支持多种 layout 转换            |
| [TileCopyTla](./tile_copy_tla.md)                      | TLA    | Atlas A2 / Ascend 950 | TLA 风格 L1→L0B 搬运，通过 tla::Tensor 封装简化调用   |
| [CopyL1ToL0BSparseTla](./copy_l1_to_l0b_sparse_tla.md) | TLA    | Atlas A2             | Sparse L1→L0B 搬运（仅 Atlas A2），需传入 index tensor |

> **说明**：该模块通常不直接使用，而是作为 [TileCopy](../tile_copy/README.md) 的成员类型（`CopyL1ToL0B`），由 [blockMmad](../../block/block_mmad.md) 自动管理。仅在需要自定义 kernel 模板组装时显式声明。

## 适用硬件型号说明

| 硬件型号   | 架构标识          | ARCH 宏                | 支持的非 TLA 模板 | 支持的 TLA 模板                    |
| :--------- | :---------------- | :--------------------- | :---------------- | :--------------------------------- |
| Atlas A2   | `Arch::AtlasA2`   | `CATLASS_ARCH == 2201` | CopyL1ToL0B       | TileCopyTla / CopyL1ToL0BSparseTla |
| Ascend 950 | `Arch::Ascend950` | `CATLASS_ARCH == 3510` | CopyL1ToL0B       | TileCopyTla                        |

### 架构差异

| 特性              | Atlas A2                                       | Ascend 950                                          |
| :---------------- | :-------------------------------------------- | :------------------------------------------------- |
| 主要搬运方向      | zZ→nZ（转置）、zN→nZ（转置）、nZ→nZ（直传）   | zN→nZ（转置）、nZ→nZ（直传）                       |
| 基础搬运指令      | LoadData2D / LoadData2dTranspose / LoadData3D | LoadData2DParamsV2                                 |
| l0Batch 批量搬运  | 不支持                                        | 支持（`operator()` 重载）                          |
| MX Scale 浮点量化 | 不支持                                        | 支持（`operator()` 重载，B 侧 scale layout 为 nN） |
| B8/B4 窄类型      | 支持（int8_t/float8_ 等）                     | 支持（含 MX Scale）                                |
| Sparse 搬运       | 支持（CopyL1ToL0BSparseTla）                  | 不支持                                             |
| GEMV 场景         | 支持（zN→zN、nN→zN、nZ→zN）                   | 不支持                                             |

## 接口调用示例

### 非 TLA 风格（CopyL1ToL0B）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using L1Type = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>;
using L0Type = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;

uint32_t k = 256;
uint32_t n = 256;

auto layoutSrc = layout::zZ::MakeLayout<Element>(k, n);
auto layoutDst = layout::nZ::MakeLayout<Element>(k, n);

AscendC::LocalTensor<Element> srcL1Tensor;
AscendC::LocalTensor<Element> dstL0BTensor;

using CopyOp = CopyL1ToL0B<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0BTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### TLA 风格 — zN → nZ 转置搬运（Atlas A2）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t K = 256;
const uint32_t N = 256;

auto layoutSrc = tla::MakeLayout<half, layout::zN>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0BTensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// iszN<LayoutSrc> && isnZ<LayoutDst> → 自动匹配转置偏特化
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA 风格 — nZ → nZ 直传搬运（Atlas A2，Transpose B）

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// isnZ<LayoutSrc> && isnZ<LayoutDst> → 自动匹配直传偏特化
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA 风格 — Ascend 950 基础搬运（zN → nZ）

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::zN>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// Ascend950: zN L1 → nZ L0B
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA 风格 — Ascend 950 nZ → nZ 直传

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA 风格 — Ascend 950 l0Batch 批量搬运

```cpp
uint32_t l0Batch = 4;

copyOp(dstTensor, srcTensor, l0Batch);
```

### TLA 风格 — Ascend 950 MX Scale 搬运（B 侧）

```cpp
using ElementSrc = float8_e4m3_t;
using ElementDst = AscendC::mx_fp8_e4m3_t;
using ElementMxScale = float8_e8m0_t;

const uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(K);

// 源数据 layout（L1 zN）
auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(K, N);
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});

// 目标数据 layout（L0B nZ，元素类型为 mx_fp8）
auto layoutDst = tla::MakeLayout<ElementDst, layout::nZ>(K, N);
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// MX Scale layout（L1 nN，B 侧 isMxScaleB=true，使用 MakeMxScaleLayout 构造）
auto layoutScaleL1 = tla::MakeMxScaleLayout<ElementMxScale, layout::nN, true>(mxScaleK, N);

AscendC::LocalTensor<ElementMxScale> scaleL1Tensor;
auto scaleTensor = tla::MakeTensor(scaleL1Tensor, layoutScaleL1, Arch::PositionL1{});

// MX Scale 重载：L1 zN 源数据 + L1 nN scale → L0B nZ mx 数据
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor, scaleTensor);
```

## 模板选择指南

| 场景                          | 推荐模板                                        |
| :---------------------------- | :---------------------------------------------- |
| 通用矩阵乘 tile L1→L0B 搬运   | `CopyL1ToL0B`（非 TLA）或 `TileCopyTla`（TLA）  |
| zZ→nZ 转置搬运（Atlas A2）     | `CopyL1ToL0B` 或 `TileCopyTla`（自动匹配）      |
| zN→nZ 转置搬运                | `CopyL1ToL0B` 或 `TileCopyTla`（自动匹配）      |
| nZ→nZ 直传（Transpose B）     | `CopyL1ToL0B` 或 `TileCopyTla`（自动匹配）      |
| Ascend 950 多 batch 搬运       | `TileCopyTla`（l0Batch 重载）                   |
| Ascend 950 MX 浮点量化（B 侧） | `TileCopyTla`（MX Scale 重载，scale layout nN） |
| Atlas A2 Sparse 搬运           | `CopyL1ToL0BSparseTla`                          |
| Atlas A2 GEMV 场景             | `CopyL1ToL0B`（非 TLA）                         |
| 已使用 TLA 编程范式           | `TileCopyTla`（统一风格）                       |
