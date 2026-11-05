# CopyPerTokenScale2Ub

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub.hpp)

[TOC]

## 功能说明

`CopyPerTokenScale2Ub` 实现 per-token scale 从 GM 到 UB 的特殊搬运。将 GM 上 (m, 1) 形状的 scale 数据搬运到 UB 上 (m, n) 矩阵的第一列，并按 block 边界做 padding。

典型场景：per-token dequant 中，将 per-token scale 从 GM ColumnMajor 搬运到 UB RowMajor 矩阵的第一列。

- 适用范围：Atlas A2（无架构限制但有 Layout 静态断言）
- 风格：非 TLA，直接操作 `AscendC::LocalTensor` / `AscendC::GlobalTensor`
- GM layout 仅支持 `ColumnMajor`
- 通过 `AscendC::DataCopyPad` + padding 实现

## 模板原型

```cpp
template <
    class ArchTag,
    class GmType       // Gemm::GemmType<Element, layout::ColumnMajor>
>
struct CopyPerTokenScale2Ub;
```

| 模板参数  | 说明                                           |
| :-------- | :--------------------------------------------- |
| `ArchTag` | 架构标签                                       |
| `GmType`  | GM 数据类型，`Layout` 静态断言为 `ColumnMajor` |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,     // 目的 UB LocalTensor（RowMajor, m*n）
    AscendC::GlobalTensor<Element> const &srcTensor,    // 源 GM GlobalTensor（ColumnMajor, m*1）
    LayoutDst const &layoutDst,                         // 目的 Ub RowMajor layout
    LayoutSrc const &layoutSrc                          // 源 GM ColumnMajor layout
)
```

| 参数        | 说明                                                             |
| :---------- | :--------------------------------------------------------------- |
| `dstTensor` | 目的 UB LocalTensor，RowMajor 布局，scale 写入每行首列并 padding |
| `srcTensor` | 源 GM GlobalTensor，ColumnMajor 布局 (m, 1)                      |
| `layoutDst` | 目的 UB layout，`layoutDst.shape(1)` 用于计算 dstStride          |
| `layoutSrc` | 源 GM layout，`layoutSrc.shape(0)` 为 m                          |

内部使用 `DataCopyPad` 的 padding 参数 `isPad = true` 确保每行首列数据填充到完整 block。

## 调用示例

```cpp
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
using LayoutTagSrc = layout::ColumnMajor;
using LayoutTagDst = layout::RowMajor;

uint32_t m = 128;
uint32_t n = 256;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(m, 1);
auto layoutDst = LayoutTagDst::MakeLayout<Element>(m, n);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;
using CopyOp = CopyPerTokenScale2Ub<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
