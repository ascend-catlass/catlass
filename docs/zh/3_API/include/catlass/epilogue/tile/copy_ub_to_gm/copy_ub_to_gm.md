# CopyUb2Gm

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/copy_ub_to_gm.hpp)

[TOC]

## 功能说明

`CopyUb2Gm` 实现 epilogue 阶段从 UB 到 GM 的数据搬运操作。用于将 epilogue 处理完成后的最终结果从 Unified Buffer 写回到 Global Memory。

- 适用范围：Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品
- 风格：非 TLA，直接操作 `AscendC::GlobalTensor` / `AscendC::LocalTensor`
- 通过 `AscendC::DataCopyPad` 实现带 stride 的数据搬运

## 模板原型

```cpp
template <
    class ArchTag,
    class GmType      // Gemm::GemmType<Element, Layout>
>
struct CopyUb2Gm;
```

| 模板参数  | 说明                                                                    |
| :-------- | :---------------------------------------------------------------------- |
| `ArchTag` | 架构标签：`Arch::AtlasA2` 或 `Arch::Ascend950`                          |
| `GmType`  | GM 数据类型，`Gemm::GemmType<Element, Layout>`，Layout 决定 GM 输出格式 |

## 偏特化实现

| 架构      | GM Layout      | UB Layout      | 说明                               |
| :-------- | :------------- | :------------- | :--------------------------------- |
| Atlas A2   | `RowMajor`     | `RowMajor`     | 二维矩阵搬运，源 stride 按 C0 对齐 |
| Atlas A2   | `VectorLayout` | `VectorLayout` | 一维向量搬运                       |
| Ascend 950 | `RowMajor`     | `RowMajor`     | 二维矩阵搬运                       |

## 调用接口

```cpp
void operator()(
    AscendC::GlobalTensor<Element> const &dstTensor,    // 目的 GM GlobalTensor
    AscendC::LocalTensor<Element> const &srcTensor,     // 源 UB LocalTensor
    LayoutDst const &layoutDst,                         // 目的 GM layout 描述
    LayoutSrc const &layoutSrc                          // 源 UB layout 描述
)
```

| 参数        | 说明                                    |
| :---------- | :-------------------------------------- |
| `dstTensor` | 目的 GM GlobalTensor                    |
| `srcTensor` | 源 UB LocalTensor                       |
| `layoutDst` | 目的 GM 的 layout，包含 shape 和 stride |
| `layoutSrc` | 源 UB 的 layout，包含 shape 和 stride   |

## 调用示例

### RowMajor（二维矩阵）

```cpp
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
using LayoutTagDst = layout::RowMajor;

uint32_t rows = 128;
uint32_t cols = 256;

auto layoutDst = LayoutTagDst::MakeLayout<Element>(rows, cols);
auto layoutSrc = LayoutTagDst::MakeLayout<Element>(rows, cols);

AscendC::LocalTensor<Element> srcTensor;
AscendC::GlobalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagDst>;
using CopyOp = CopyUb2Gm<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### VectorLayout（一维向量）

```cpp
using Element = half;
using LayoutTagDst = layout::VectorLayout;

uint32_t length = 256;

auto layoutDst = LayoutTagDst::MakeLayout<Element>(length, 1);
auto layoutSrc = LayoutTagDst::MakeLayout<Element>(length, 1);

AscendC::LocalTensor<Element> srcTensor;
AscendC::GlobalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagDst>;
using CopyOp = CopyUb2Gm<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
