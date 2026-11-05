# CopyGm2Ub

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub.hpp)

[TOC]

## 功能说明

`CopyGm2Ub` 实现 epilogue 阶段从 GM 到 UB 的数据搬运操作。用于将最终输出矩阵的 C/X/Y 矩阵从 Global Memory 搬运到 Unified Buffer，供 epilogue 后续计算使用。

- 适用范围：Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR&950DT系列产品
- 风格：非 TLA，直接操作 `AscendC::LocalTensor` / `AscendC::GlobalTensor`
- 通过 `AscendC::DataCopyPad` 实现带 stride 的数据搬运

## 模板原型

```cpp
template <
    class ArchTag,
    class GmType      // Gemm::GemmType<Element, Layout>
>
struct CopyGm2Ub;
```

| 模板参数  | 说明                                                                  |
| :-------- | :-------------------------------------------------------------------- |
| `ArchTag` | 架构标签：`Arch::AtlasA2` 或 `Arch::Ascend950`                        |
| `GmType`  | GM 数据类型，`Gemm::GemmType<Element, Layout>`，Layout 触发不同偏特化 |

## 偏特化实现

| 架构      | GM Layout      | UB Layout      | 说明                        |
| :-------- | :------------- | :------------- | :-------------------------- |
| Atlas A2   | `RowMajor`     | `RowMajor`     | 二维矩阵搬运，`DataCopyPad` |
| Atlas A2   | `VectorLayout` | `VectorLayout` | 一维向量搬运                |
| Ascend 950 | `RowMajor`     | `RowMajor`     | 二维矩阵搬运，`DataCopyPad` |
| Ascend 950 | `VectorLayout` | `VectorLayout` | 一维向量搬运                |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,     // 目的 UB LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,    // 源 GM GlobalTensor
    LayoutDst const &layoutDst,                         // 目的 UB layout 描述
    LayoutSrc const &layoutSrc                          // 源 GM layout 描述
)
```

| 参数        | 说明                                    |
| :---------- | :-------------------------------------- |
| `dstTensor` | 目的 UB LocalTensor                     |
| `srcTensor` | 源 GM GlobalTensor                      |
| `layoutDst` | 目的 UB 的 layout，包含 shape 和 stride |
| `layoutSrc` | 源 GM 的 layout，包含 shape 和 stride   |

## 调用示例

### RowMajor（二维矩阵）

```cpp
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
using LayoutTagSrc = layout::RowMajor;
using GmType = Gemm::GemmType<Element, LayoutTagSrc>;

uint32_t rows = 128;
uint32_t cols = 256;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(rows, cols);
auto layoutDst = LayoutTagSrc::MakeLayout<Element>(rows, cols);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using CopyOp = CopyGm2Ub<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### VectorLayout（一维向量）

```cpp
using Element = half;
using LayoutTagSrc = layout::VectorLayout;

uint32_t length = 256;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(length, 1);
auto layoutDst = LayoutTagSrc::MakeLayout<Element>(length, 1);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;
using CopyOp = CopyGm2Ub<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
