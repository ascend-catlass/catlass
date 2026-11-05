# CopyL1ToFP

> [代码位置](../../../../../../../include/catlass/gemm/tile/copy_l1_to_fp.hpp)

[TOC]

## 功能说明

`CopyL1ToFP` 模板负责将数据（通常为量化 Scale 等辅助信息）从 L1（Local Memory，A1 Buffer）搬运到 FP（FixPipe Buffer，C2PIPE2GM），通过 FixPipe 通道将数据直接写入 GM。

FixPipe 是 Atlas A2 架构上的一种特殊数据通路，允许将 Core 内部的计算结果或辅助数据直接发送到外部存储（GM），绕过常规的 L0C→GM 搬运路径，常用于 per-token/per-channel 反量化场景中的 Scale 数据回写。

该模板通常不直接使用，而是作为 [TileCopy](./tile_copy/README.md) 的成员类型（`CopyL1ToFP`），由 `blockMmad` 自动管理。仅在需要自定义 kernel 模板组装时显式声明。

> **限制**：该模板仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`），Ascend 950 不支持。

## 模板原型

```cpp
template <
    class ArchTag,                    // 架构标签：Arch::AtlasA2
    class L1Type,                     // L1 数据描述：Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>
    class L0Type = void               // FP 数据描述：Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>
>
struct CopyL1ToFP {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to fixpipe buffer, can not find the specialization.");
};
```

- `ArchTag`：架构标签，仅支持 `Arch::AtlasA2`
- `L1Type`：L1 上的一维向量数据类型，固定使用 `Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>`
- `L0Type`：FixPipe Buffer 的数据类型，固定使用 `Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>`

## 偏特化实现

| 架构    | 源 Layout    | 目标 Layout  | 位置           | 说明                                                                 |
| :------ | :----------- | :----------- | :------------- | :------------------------------------------------------------------- |
| Atlas A2 | VectorLayout | VectorLayout | A1 → C2PIPE2GM | 一维向量拷贝，使用 `AscendC::DataCopy`，块大小基于 `BYTE_PER_BLK_FP` |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementDst> dstTensor,   // FixPipe 目标 tensor（C2PIPE2GM）
    AscendC::LocalTensor<ElementSrc> srcTensor,   // L1 源 tensor（A1）
    LayoutDst layoutDst,                          // FixPipe 数据 layout（VectorLayout）
    LayoutSrc layoutSrc                           // L1 数据 layout（VectorLayout）
);
```

- `srcTensor`：L1 上的一维源 tensor
- `dstTensor`：FixPipe Buffer 上的一维目标 tensor
- `srcTensor` 元素类型为 `ElementSrc`，`dstTensor` 元素类型为 `ElementDst`，两者可以不同

## 调用示例

### 基础 FixPipe 搬运（Atlas A2）

```cpp
#include "catlass/gemm/tile/copy_l1_to_fp.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = uint64_t;
using ElementDst = uint64_t;
using L1Type = Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>;

uint32_t vecLen = 256;

auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstFPTensor;

using CopyOp = CopyL1ToFP<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstFPTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### 类型转换 FixPipe 搬运

源和目标元素类型可以不同，支持在搬运过程中进行类型转换：

```cpp
using ElementSrc = float;
using ElementDst = uint64_t;
using L1Type = Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>;

auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstFPTensor;

using CopyOp = CopyL1ToFP<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstFPTensor, srcL1Tensor, layoutDst, layoutSrc);
```
