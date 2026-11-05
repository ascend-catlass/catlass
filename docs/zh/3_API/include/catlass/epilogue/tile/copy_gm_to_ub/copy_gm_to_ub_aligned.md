# CopyGm2UbAligned

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub.hpp)

[TOC]

## 功能说明

`CopyGm2UbAligned` 实现 epilogue 阶段从 GM 到 UB 的对齐搬运操作。相比 `CopyGm2Ub`，额外处理 stride 较大无法直接用 `DataCopyPad` 的场景，自动拆分为多次 `DataCopy` 或逐行搬运。

- 适用范围：Atlas A2
- 风格：非 TLA，直接操作 `AscendC::LocalTensor` / `AscendC::GlobalTensor`
- 内部逻辑：连续内存→`DataCopy`；小 stride→分块 `DataCopy`；大 stride→逐行搬运

## 模板原型

```cpp
template <
    class ArchTag,
    class GmType      // Gemm::GemmType<Element, layout::RowMajor>
>
struct CopyGm2UbAligned;
```

| 模板参数  | 说明                                |
| :-------- | :---------------------------------- |
| `ArchTag` | 仅 `Arch::AtlasA2` 有特化           |
| `GmType`  | GM 数据类型，Layout 固定 `RowMajor` |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,     // 目的 UB LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,    // 源 GM GlobalTensor
    layout::RowMajor const &layoutDst,                  // 目的 UB layout
    layout::RowMajor const &layoutSrc                   // 源 GM layout
)
```

| 参数        | 说明                       |
| :---------- | :------------------------- |
| `dstTensor` | 目的 UB LocalTensor        |
| `srcTensor` | 源 GM GlobalTensor         |
| `layoutDst` | 目的 UB 的 RowMajor layout |
| `layoutSrc` | 源 GM 的 RowMajor layout   |

内部根据 stride 和 dimension 自动选择最优搬运策略：

- 无 stride 且 dst 无 stride：直接 `DataCopy(dst, src, rows * cols)`
- 小 stride（< 65536）且 cols/blk < 65536：分块 `DataCopy` 带 `DataCopyParams`
- 大 stride：逐行 `DataCopy`

## 调用示例

```cpp
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
uint32_t rows = 128;
uint32_t cols = 256;

auto layoutSrc = layout::RowMajor::MakeLayout<Element>(rows, cols);
auto layoutDst = layout::RowMajor::MakeLayout<Element>(rows, cols);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, layout::RowMajor>;
using CopyOp = CopyGm2UbAligned<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
