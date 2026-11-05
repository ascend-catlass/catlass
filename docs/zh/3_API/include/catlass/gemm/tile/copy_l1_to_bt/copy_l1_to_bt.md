# CopyL1ToBT

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_bt.hpp)

[TOC]

## 功能说明

`CopyL1ToBT` 模板负责将 Bias Table（一维向量）从 L1（Local Memory，A1 Buffer）搬运到 BT（Bias Table Buffer，C2 Buffer）。

Bias Table 用于矩阵乘的 Bias 加法和量化反量化操作。由于 Bias 数据是一维向量，该模板固定使用 `VectorLayout`（rank=1、stride=1 的一维排布），通过 `AscendC::DataCopy` 指令以 `blockLen` 为粒度进行连续搬运。

该模板通常不直接使用，而是作为 [TileCopy](../tile_copy/README.md) 的成员类型（`CopyL1ToBT`），由 `blockMmad` 自动管理。仅在需要自定义 kernel 模板组装时显式声明。

## 模板原型

```cpp
template <
    class ArchTag,                    // 架构标签：Arch::AtlasA2 或 Arch::Ascend950
    class L1Type,                     // L1 数据描述：Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>
    class L0Type = void               // BT 数据描述：Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::C2>
>
struct CopyL1ToBT {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to biasTable buffer, can not find the specialization.");
};
```

- `ArchTag`：架构标签，`Arch::AtlasA2` 或 `Arch::Ascend950`
- `L1Type`：L1 上 Bias Table 的数据类型，固定使用 `Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>`
- `L0Type`：BT Buffer 上 Bias Table 的数据类型，固定使用 `Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::C2>`

## 偏特化实现

| 架构      | 源 Layout    | 目标 Layout  | 位置    | 说明                                                                               |
| :-------- | :----------- | :----------- | :------ | :--------------------------------------------------------------------------------- |
| Atlas A2   | VectorLayout | VectorLayout | A1 → C2 | 一维向量拷贝，使用 `AscendC::DataCopy`，块大小基于 `BYTE_PER_C2`                   |
| Ascend 950 | VectorLayout | VectorLayout | A1 → C2 | 一维向量拷贝，使用 `AscendC::DataCopy`，块大小基于 `BYTE_PER_C0`，B32 类型自动对齐 |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementDst> dstTensor,   // BT buffer 目标 tensor（C2）
    AscendC::LocalTensor<ElementSrc> srcTensor,   // L1 源 tensor（A1）
    LayoutDst layoutDst,                          // BT 数据 layout（VectorLayout）
    LayoutSrc layoutSrc                           // L1 数据 layout（VectorLayout）
);
```

- `srcTensor`：L1 上的一维 Bias Table tensor
- `dstTensor`：C2 Buffer（BT）上的一维 Bias Table tensor
- `srcTensor` 元素类型为 `ElementSrc`，`dstTensor` 元素类型为 `ElementDst`，两者可以不同（支持类型转换）

## 调用示例

### Atlas A2

```cpp
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float;
using ElementDst = half;
using L1Type = Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>;

uint32_t vecLen = 256;

auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstBTTensor;

using CopyOp = CopyL1ToBT<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstBTTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### Ascend 950

```cpp
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float;
using ElementDst = half;
using L1Type = Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>;

uint32_t vecLen = 256;

auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstBTTensor;

using CopyOp = CopyL1ToBT<Arch::Ascend950, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstBTTensor, srcL1Tensor, layoutDst, layoutSrc);
```
