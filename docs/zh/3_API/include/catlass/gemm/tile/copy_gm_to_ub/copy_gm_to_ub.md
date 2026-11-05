# CopyGm2Ub

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_ub.hpp)

[TOC]

## 功能说明

`CopyGm2Ub` 模板负责将一维向量数据从 GM（Global Memory）搬运到 UB（Unified Buffer，`VECCALC`），常见于 Bias、Scale 等辅助数据的搬入场景。

与 TLA 风格的 [TileCopyTla](./tile_copy_tla.md) 不同，非 TLA 版本仅支持 VectorLayout（一维向量），搬运指令为 `AscendC::DataCopyPad`。

> **限制**：仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`）。

## 模板原型

```cpp
template <
    class ArchTag,                    // 架构标签：仅 Arch::AtlasA2
    class GmType                      // GM 数据描述：Gemm::GemmType<Element, layout::VectorLayout>
>
struct CopyGm2Ub {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub.");
};
```

## 偏特化实现

| 架构    | Layout                      | 搬运指令               | 说明                             |
| :------ | :-------------------------- | :--------------------- | :------------------------------- |
| Atlas A2 | VectorLayout → VectorLayout | `AscendC::DataCopyPad` | 一维向量，单行拷贝，`blockLen=1` |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,      // UB 目标 tensor（VECCALC）
    AscendC::GlobalTensor<Element> const &srcTensor,     // GM 源 tensor
    layout::VectorLayout const &layoutDst,               // UB VectorLayout
    layout::VectorLayout const &layoutSrc                // GM VectorLayout
);
```

## 调用示例

```cpp
#include "catlass/gemm/tile/copy_gm_to_ub.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using GmType = Gemm::GemmType<Element, layout::VectorLayout>;

const uint32_t vecLen = 256;
auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::GlobalTensor<Element> srcGmTensor;
AscendC::LocalTensor<Element> dstUBTensor;

using CopyOp = CopyGm2Ub<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstUBTensor, srcGmTensor, layoutDst, layoutSrc);
```
