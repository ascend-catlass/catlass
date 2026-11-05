# VecCopyUBToGm

> [代码位置](../../../../../../../include/catlass/gemv/tile/vec_copy_ub_to_gm.hpp)

[TOC]

## 功能说明

`VecCopyUBToGm` 实现 GEMV 场景下从 UB 到 GM 的向量数据搬运。通过 `AscendC::DataCopyPad` 将一维向量从 Unified Buffer 写回 Global Memory。

- 适用范围：Atlas A2
- 支持 `atomic_add` 模式：当 `is_atoadd = true` 时，先调用 `AscendC::SetAtomicAdd<Element>()` 再搬运

## 模板原型

```cpp
template <class ArchTag, class GmType, bool is_atoadd = false>
struct VecCopyUBToGm;
```

| 模板参数    | 说明                                    |
| :---------- | :-------------------------------------- |
| `ArchTag`   | 架构标签                                |
| `GmType`    | `Gemm::GemmType<Element, VectorLayout>` |
| `is_atoadd` | 是否启用 atomic add 模式                |

## 偏特化实现

| 架构    | GmType         | is_atoadd       | 说明                           |
| :------ | :------------- | :-------------- | :----------------------------- |
| Atlas A2 | `VectorLayout` | `false`（默认） | 标准 `DataCopyPad` 搬运        |
| Atlas A2 | `VectorLayout` | `true`          | `SetAtomicAdd` + `DataCopyPad` |

## 调用接口

```cpp
void operator()(
    AscendC::GlobalTensor<Element> dstTensor,             // GM 上的目的 GlobalTensor
    AscendC::LocalTensor<Element> srcTensor,              // UB 上的源 LocalTensor
    layout::VectorLayout const &layoutDst,
    layout::VectorLayout const &layoutSrc
)
```

## 调用示例

### 默认模式

```cpp
#include "catlass/gemv/tile/vec_copy_ub_to_gm.hpp"

using namespace Catlass::Gemv::Tile;

using Element = half;
using GmType = Gemm::GemmType<Element, layout::VectorLayout>;

uint32_t length = 64;
auto layoutSrc = layout::VectorLayout::MakeLayout<Element>(length, 1);
auto layoutDst = layout::VectorLayout::MakeLayout<Element>(length, 1);

AscendC::LocalTensor<Element> srcTensor;
AscendC::GlobalTensor<Element> dstTensor;

using CopyOp = VecCopyUBToGm<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### Atomic Add 模式

```cpp
using CopyOp = VecCopyUBToGm<Arch::AtlasA2, GmType, true>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
