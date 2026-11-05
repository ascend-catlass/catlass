# CopyGmToL1

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## 功能说明

`CopyGmToL1` 是非 TLA 风格的 GM（Global Memory）到 L1（Local Memory）数据搬运模板，负责将 tile 块从 GlobalTensor 搬运到 LocalTensor，并在搬运过程中完成数据排布格式（layout）的转换。

该模板支持多种源/目的 layout 组合，覆盖矩阵乘（Gemm）和向量乘（Gemv）场景。根据架构不同，偏特化实现分布在：

- `Arch::AtlasA2`（ARCH 2201）：[atlasa2/copy_gm_to_l1.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp)
- `Arch::Ascend950`（ARCH 3510）：[ascend950/copy_gm_to_l1.hpp](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp)

## 模板原型

```cpp
template <
    class ArchTag,          // 架构标签，如 Arch::AtlasA2 / Arch::Ascend950
    class GmType,           // GM 上操作数的 Gemm 类型
    class L1Type = void     // L1 上操作数的 Gemm 类型（默认 void 表示由偏特化自动推导）
>
struct CopyGmToL1
```

### 模板参数说明

| 参数      | 说明                                                                                                   |
| :-------- | :----------------------------------------------------------------------------------------------------- |
| `ArchTag` | 架构标签，决定使用哪套硬件指令。可选 `Arch::AtlasA2` 或 `Arch::Ascend950`                              |
| `GmType`  | GM 上源操作数的 Gemm 类型，封装了数据类型和 layout 信息                                                |
| `L1Type`  | L1 上目的操作数的 Gemm 类型，封装了数据类型、layout 和 TPosition 信息。默认为 `void`，由偏特化自动推导 |

## 偏特化实现

### Atlas A2 偏特化

以下偏特化适用于 `Arch::AtlasA2`。

#### 简化版（仅指定 `GmType`，`L1Type` 自动推导）

仅指定 `GmType`（2 参数），无需指定目的 Layout 和 TPosition，由偏特化自动推导最优目标格式。省去重复声明，适用于最常用搬运场景。其中 `RowMajor → zN` 额外提供手动指定 stride 的扩展接口。

| 源 Layout          | 目标 Layout | 说明                                               |
| :----------------- | :---------- | :------------------------------------------------- |
| RowMajor           | zN          | 含双调用接口（基础 + 手动 stride），见下方调用接口 |
| ColumnMajor        | nZ          | 常用于 B 矩阵搬运                                  |
| PaddingRowMajor    | zN          | 带 Padding 的 RowMajor，用于非对齐矩阵乘           |
| PaddingColumnMajor | nZ          | 带 Padding 的 ColumnMajor，用于非对齐矩阵乘        |
| zN                 | zN          | 保持 zN 格式不变                                   |
| nZ                 | nZ          | 保持 nZ 格式不变                                   |

#### Gemm 场景

| 源 Layout   | 目标 Layout    | 说明                   |
| :---------- | :------------- | :--------------------- |
| RowMajor    | zN（A1）       | A 矩阵搬运并转 zN 格式 |
| RowMajor    | zZ（B1）       | B 矩阵搬运并转 zZ 格式 |
| RowMajor    | zN（B1）       | B 矩阵搬运并转 zN 格式 |
| RowMajor    | RowMajor（A1） | 保持 RowMajor 格式不变 |
| ColumnMajor | nN（A1）       | A 矩阵搬运并转 nN 格式 |
| ColumnMajor | nZ（A1）       | A 矩阵搬运并转 nZ 格式 |
| ColumnMajor | nZ（B1）       | B 矩阵搬运并转 nZ 格式 |
| ColumnMajor | nN（B1）       | B 矩阵搬运并转 nN 格式 |

#### Gemv 场景

| 源 Layout          | 目标 Layout        | 说明                 |
| :----------------- | :----------------- | :------------------- |
| VectorLayout       | zN（A1）           | 向量搬运并转 zN 格式 |
| VectorLayout（GM） | VectorLayout（A1） | 向量搬运保持格式不变 |

#### 卷积场景

| 源 Layout            | 目标 Layout | 说明             |
| :------------------- | :---------- | :--------------- |
| NDC1HWC0（GM）       | NDC1HWC0    | 保持格式不变     |
| KDC1KHKWN1N0C0（GM） | nZ          | 搬运并转 nZ 格式 |

### Ascend 950 偏特化

以下偏特化适用于 `Arch::Ascend950`。

| 源 Layout          | 目标 Layout | 说明                                       |
| :----------------- | :---------- | :----------------------------------------- |
| RowMajor           | zN          | 简化版，含双调用接口（基础 + 手动 stride） |
| ColumnMajor        | nZ          | 简化版                                     |
| zN                 | zN          | 保持 zN 格式不变                           |
| nZ                 | nZ          | 保持 nZ 格式不变                           |
| RowMajor           | zZ（A1）    | MX Scale 专用，仅 `fp8_e8m0_t` 类型        |
| PaddingRowMajor    | zN          | 带 Padding 的 RowMajor                     |
| PaddingColumnMajor | nZ          | 带 Padding 的 ColumnMajor                  |

## 调用接口

### 基础调用接口（所有偏特化通用）

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,   // 目的操作数 LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,  // 源操作数 GlobalTensor
    LayoutDst const &layoutDst,                       // 目的操作数 layout
    LayoutSrc const &layoutSrc                        // 源操作数 layout
)
```

| 参数        | 说明                                                |
| :---------- | :-------------------------------------------------- |
| `dstTensor` | 目的 L1 LocalTensor                                 |
| `srcTensor` | 源 GM GlobalTensor                                  |
| `layoutDst` | 目的操作数的 layout 描述，包含 shape 和 stride 信息 |
| `layoutSrc` | 源操作数的 layout 描述，包含 shape 和 stride 信息   |

### 扩展调用接口（手动指定 stride）

以下偏特化额外提供手动指定搬运 stride 的重载：

- `AtlasA2, RowMajor`（简化版）
- `AtlasA2, RowMajor → zN, A1`（通用版）
- `Ascend950, RowMajor`

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,   // 目的操作数 LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,  // 源操作数 GlobalTensor
    LayoutDst const &layoutDst,                       // 目的操作数 layout
    LayoutSrc const &layoutSrc,                       // 源操作数 layout
    uint32_t ndNum,                                   // ND 矩阵数量
    uint32_t srcNdMatrixStride,                       // 源 ND 矩阵间 stride
    uint32_t dstNzNStride,                            // 目的 n 方向 stride
    uint32_t dstNzMatrixStride,                       // 目的矩阵间 stride
    uint32_t dstNzC0Stride                            // 目的 C0 方向 stride
)
```

| 参数                | 说明                                            |
| :------------------ | :---------------------------------------------- |
| `ndNum`             | 连续搬运的 ND 矩阵数量                          |
| `srcNdMatrixStride` | 源端相邻 ND 矩阵间的 stride                     |
| `dstNzNStride`      | 目的端 n 方向的 stride（覆盖 layout 默认值）    |
| `dstNzMatrixStride` | 目的端相邻矩阵间的 stride（覆盖 layout 默认值） |
| `dstNzC0Stride`     | 目的端 C0 方向的 stride（覆盖 layout 默认值）   |

## 调用示例

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
