# TileCopyTlaExt（UB → GM, PaddingRowMajor）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_ub_to_gm.hpp)

[TOC]

## 功能说明

`TileCopyTlaExt` 的 UB→GM 偏特化负责将 RowMajor 二维数据从 UB（`VECCALC`）搬运到 GM，目标排布为 PaddingRowMajor。PaddingRowMajor 在逻辑维度上与原 RowMajor 相同，但 stride 可能因 padding 而更大，常用于矩阵分块后确保对齐的中间输出场景。

与 [TileCopyTla](./tile_copy_tla.md) 的普通 RowMajor 目标不同，本模板的目标 Layout 为 PaddingRowMajor。

> **限制**：仅支持 Atlas A2 架构。

## 模板原型

```cpp
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTlaExt<Arch::AtlasA2,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::GM>,
    layout::RowMajor, layout::PaddingRowMajor>;
```

- `LayoutTagSrc = layout::RowMajor`：源 layout 标签（仅用于偏特化分发，与 tensor 物理 Layout 无关）
- `LayoutTagDst = layout::PaddingRowMajor`：目标 layout 标签

## 偏特化实现

| 架构    | 源位置  | 目标位置 | LayoutTagSrc | LayoutTagDst    | 搬运指令               |
| :------ | :------ | :------- | :----------- | :-------------- | :--------------------- |
| Atlas A2 | VECCALC | GM       | RowMajor     | PaddingRowMajor | `AscendC::DataCopyPad` |

与普通 RowMajor 目标版的关键区别：维度计算使用 `tla::get<1, 1>(dstTensor.shape())`（逻辑行数）*`tla::get<1, 0>(dstTensor.shape())`（逻辑列数），并基于 PaddingRowMajor 的 stride 计算偏移。

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // 目标 tensor（GM, PaddingRowMajor）
    TensorSrc const &srcTensor     // 源 tensor（UB, VECCALC, RowMajor）
);
```

## 调用示例

```cpp
#include "catlass/gemm/tile/copy_ub_to_gm.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using Element = half;

const int M = 128;
const int N = 256;

auto srcLayout = tla::MakeLayout<Element, layout::RowMajor>(M, N);
auto dstLayout = tla::MakeLayout<Element, layout::PaddingRowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcUBTensor, srcLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

TileCopyTlaExt<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor),
    layout::RowMajor, layout::PaddingRowMajor> copyOp;
copyOp(dstTensor, srcTensor);
```
