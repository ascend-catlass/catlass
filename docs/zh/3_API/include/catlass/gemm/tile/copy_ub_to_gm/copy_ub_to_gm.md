# CopyUb2Gm

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_ub_to_gm.hpp)

[TOC]

## 功能说明

`CopyUb2Gm` 模板负责将二维矩阵数据从 UB（Unified Buffer，`VECCALC`）搬运到 GM（Global Memory），常见于 Vector 引擎后处理完成后将结果写回 GM 的场景。

与 TLA 风格 [TileCopyTla](./tile_copy_tla.md) / [TileCopyTlaExt](./tile_copy_tla_ext.md) 不同，非 TLA 版本仅支持 RowMajor 排布，使用 `AscendC::DataCopyPad` 完成搬出。

> **限制**：仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## 模板原型

```cpp
template <
    class ArchTag,                    // 架构标签：仅 Arch::AtlasA2
    class GmType                      // GM 数据描述：Gemm::GemmType<Element, layout::RowMajor>
>
struct CopyUb2Gm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm.");
};
```

## 偏特化实现

| 架构    | Layout              | 搬运指令               | 说明                                       |
| :------ | :------------------ | :--------------------- | :----------------------------------------- |
| Atlas A2 | RowMajor → RowMajor | `AscendC::DataCopyPad` | 逐行拷贝，行数 `shape(0)`，行长 `shape(1)` |

stride 计算：

- `srcStride = (layoutSrc.stride(0) - shape(1)) / ELE_NUM_PER_C0`
- `dstStride = (layoutDst.stride(0) - shape(1)) * sizeof(Element)`

## 调用接口

```cpp
void operator()(
    AscendC::GlobalTensor<Element> const &dstTensor,     // GM 目标 tensor
    AscendC::LocalTensor<Element> const &srcTensor,      // UB 源 tensor（VECCALC）
    layout::RowMajor const &layoutDst,                   // GM RowMajor layout
    layout::RowMajor const &layoutSrc                    // UB RowMajor layout
);
```

## 调用示例

```cpp
#include "catlass/gemm/tile/copy_ub_to_gm.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using GmType = Gemm::GemmType<Element, layout::RowMajor>;

const int M = 128;
const int N = 256;
auto layoutSrc = layout::RowMajor::MakeLayout<Element>(M, N);
auto layoutDst = layout::RowMajor::MakeLayout<Element>(M, N);

AscendC::LocalTensor<Element> srcUBTensor;
AscendC::GlobalTensor<Element> dstGmTensor;

using CopyOp = CopyUb2Gm<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstGmTensor, srcUBTensor, layoutDst, layoutSrc);
```
