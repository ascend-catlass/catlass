# TileCopyTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## 功能说明

`TileCopyTla` 是 TLA（Tensor Layout Abstraction）风格的 GM 到 L1 数据搬运模板。与 `CopyGmToL1`（非 TLA）不同，`TileCopyTla` 使用 `tla::Tensor` 封装源和目的操作数，通过 TLA 的 layout/coord 系统自动推导搬运参数，简化调用接口。

支持 `Arch::AtlasA2` 和 `Arch::Ascend950` 两种架构。

## 模板原型

```cpp
template <
    class ArchTag,                                  // 架构标签
    class TensorSrc,                                // 源操作数 TLA Tensor 类型
    class TensorDst,                                // 目的操作数 TLA Tensor 类型
    class Enable = void                             // SFINAE 条件
>
struct TileCopyTla
```

其中 `TensorSrc` 和 `TensorDst` 的期望形式为：

```cpp
tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>   // 源
tla::Tensor<AscendC::LocalTensor<ElementDst>,  LayoutDst, CoordDst, AscendC::TPosition::A1>   // 目的
```

### 模板参数说明

| 参数        | 说明                                                                 |
| :---------- | :------------------------------------------------------------------- |
| `ArchTag`   | 架构标签，可选 `Arch::AtlasA2` 或 `Arch::Ascend950`                  |
| `TensorSrc` | 源 TLA Tensor，封装 GM GlobalTensor、layout、coord 和 TPosition::GM  |
| `TensorDst` | 目的 TLA Tensor，封装 L1 LocalTensor、layout、coord 和 TPosition::A1 |
| `Enable`    | SFINAE 条件，通过 `std::enable_if_t` 限制合法的 layout 组合          |

## 偏特化实现

### Atlas A2 偏特化

所有偏特化的 `Enable` 条件通过 `std::enable_if_t<cond>` 约束。

| 源 Layout 条件                | 目的 Layout 条件              | 说明                |
| :---------------------------- | :---------------------------- | :------------------ |
| `isRowMajor<LayoutSrc>`       | `iszN<ElementDst, LayoutDst>` | RowMajor → zN       |
| `isColumnMajor<LayoutSrc>`    | `isnZ<ElementDst, LayoutDst>` | ColumnMajor → nZ    |
| `iszN<ElementSrc, LayoutSrc>` | `iszN<ElementDst, LayoutDst>` | zN → zN（保持格式） |
| `isnZ<ElementSrc, LayoutSrc>` | `isnZ<ElementDst, LayoutDst>` | nZ → nZ（保持格式） |

### Ascend 950 偏特化

| 源 Layout 条件                                    | 目的 Layout 条件                        | 说明                        |
| :------------------------------------------------ | :-------------------------------------- | :-------------------------- |
| `isRowMajor<LayoutSrc>`                           | `iszN<ElementDst, LayoutDst>`           | RowMajor → zN               |
| `iszN<ElementSrc, LayoutSrc>`                     | `iszN<ElementDst, LayoutDst>`           | zN → zN（保持格式）         |
| `isColumnMajor<LayoutSrc>`                        | `isnZ<ElementDst, LayoutDst>`           | ColumnMajor → nZ            |
| `isnZ<ElementSrc, LayoutSrc>`                     | `isnZ<ElementDst, LayoutDst>`           | nZ → nZ（保持格式）         |
| `isVector<LayoutSrc>`                             | `isVector<LayoutDst>`                   | Vector → Vector（保持格式） |
| `isMxScaleForRowMajorA<fp8_e8m0_t, LayoutSrc>`    | `isMxScaleForzZ<fp8_e8m0_t, LayoutDst>` | MX Scale RowMajor A → zZ    |
| `isMxScaleForColumnMajorA<fp8_e8m0_t, LayoutSrc>` | `isMxScaleForzZ<fp8_e8m0_t, LayoutDst>` | MX Scale ColumnMajor A → zZ |
| `isMxScaleForRowMajorB<fp8_e8m0_t, LayoutSrc>`    | `isMxScaleFornN<fp8_e8m0_t, LayoutDst>` | MX Scale RowMajor B → nN    |
| `isMxScaleForColumnMajorB<fp8_e8m0_t, LayoutSrc>` | `isMxScaleFornN<fp8_e8m0_t, LayoutDst>` | MX Scale ColumnMajor B → nN |

## 调用接口

### 基础调用接口（Atlas A2 全部偏特化 + Ascend 950 zN/nZ/Vector/MX Scale）

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,     // 目的 TLA Tensor
    TensorSrc const &srcTensor      // 源 TLA Tensor
)
```

| 参数        | 说明                                 |
| :---------- | :----------------------------------- |
| `dstTensor` | 目的 TLA Tensor（L1, TPosition::A1） |
| `srcTensor` | 源 TLA Tensor（GM, TPosition::GM）   |

### 扩展调用接口（Ascend 950 RowMajor/ColumnMajor 偏特化）

Ascend 950 的 RowMajor → zN 和 ColumnMajor → nZ 偏特化额外支持多矩阵搬运参数：

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,         // 目的 TLA Tensor
    TensorSrc const &srcTensor,         // 源 TLA Tensor
    uint32_t ndNum = 1,                 // ND 矩阵数量
    uint32_t srcNdMatrixStride = 0,     // 源 ND 矩阵间 stride
    uint32_t dstNzMatrixStride = 0      // 目的矩阵间 stride
)
```

| 参数                | 说明                                  |
| :------------------ | :------------------------------------ |
| `ndNum`             | 连续搬运的 ND 矩阵数量，默认为 1      |
| `srcNdMatrixStride` | 源端相邻 ND 矩阵间的 stride，默认为 0 |
| `dstNzMatrixStride` | 目的端相邻矩阵间的 stride，默认为 0   |

## 调用示例

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

// 实例化并调用（SFINAE 根据 src/dst layout trait 自动匹配偏特化）
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
