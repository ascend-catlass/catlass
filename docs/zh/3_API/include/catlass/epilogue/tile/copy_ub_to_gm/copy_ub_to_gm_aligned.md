# CopyUb2GmAligned

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/copy_ub_to_gm.hpp)

[TOC]

## 功能说明

`CopyUb2GmAligned` 实现 epilogue 阶段从 UB 到 GM 的对齐搬运操作。相比 `CopyUb2Gm`，额外处理 stride 较大无法直接用 `DataCopyPad` 的场景，自动拆分为多次 `DataCopy` 或逐行搬运。

- 适用范围：Atlas A2
- 风格：非 TLA，直接操作 `AscendC::GlobalTensor` / `AscendC::LocalTensor`
- 内部逻辑：连续内存→`DataCopy`；小 stride→分块 `DataCopy`；大 stride→逐行搬运

## 模板原型

```cpp
template <
    class ArchTag,
    class GmType      // Gemm::GemmType<Element, layout::RowMajor>
>
struct CopyUb2GmAligned;
```

| 模板参数  | 说明                                |
| :-------- | :---------------------------------- |
| `ArchTag` | 仅 `Arch::AtlasA2` 有特化           |
| `GmType`  | GM 数据类型，Layout 固定 `RowMajor` |

## 调用接口

```cpp
void operator()(
    AscendC::GlobalTensor<Element> const &dstTensor,    // 目的 GM GlobalTensor
    AscendC::LocalTensor<Element> const &srcTensor,     // 源 UB LocalTensor
    layout::RowMajor const &layoutDst,                  // 目的 GM layout
    layout::RowMajor const &layoutSrc                   // 源 UB layout
)
```

| 参数        | 说明                       |
| :---------- | :------------------------- |
| `dstTensor` | 目的 GM GlobalTensor       |
| `srcTensor` | 源 UB LocalTensor          |
| `layoutDst` | 目的 GM 的 RowMajor layout |
| `layoutSrc` | 源 UB 的 RowMajor layout   |

内部根据 stride 和 dimension 自动选择最优搬运策略：连续内存直接 `DataCopy`；小 stride 分块 `DataCopy`；大 stride 逐行搬运。

## 调用示例

```cpp
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
uint32_t rows = 128;
uint32_t cols = 256;

auto layoutSrc = layout::RowMajor::MakeLayout<Element>(rows, cols);
auto layoutDst = layout::RowMajor::MakeLayout<Element>(rows, cols);

AscendC::LocalTensor<Element> srcTensor;
AscendC::GlobalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, layout::RowMajor>;
using CopyOp = CopyUb2GmAligned<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
