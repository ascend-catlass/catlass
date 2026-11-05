# MatrixCopyGmToUB

> [代码位置](../../../../../../../include/catlass/gemv/tile/matrix_copy_gm_to_ub.hpp)

[TOC]

## 功能说明

`MatrixCopyGmToUB` 实现 GEMV 场景下矩阵数据从 GM 到 UB 的搬运。根据 stride 和元素数量自动选择最优搬运策略（`DataCopy` 连续块 / 跨步块 / 单行搬运）。

- 适用范围：Atlas A2
- 支持 RowMajor 和 ColumnMajor 两种矩阵布局
- 当 `stride >= STRIDE_LIMIT(65536)` 时回退到单行逐条搬运

## 模板原型

```cpp
template <class ArchTag, class GmType>
struct MatrixCopyGmToUB;
```

| 模板参数  | 说明                                                                    |
| :-------- | :---------------------------------------------------------------------- |
| `ArchTag` | 架构标签                                                                |
| `GmType`  | `Gemm::GemmType<Element, RowMajor>` 或 `GemmType<Element, ColumnMajor>` |

## 偏特化实现

| 架构    | GmType        | 搬运策略                             |
| :------ | :------------ | :----------------------------------- |
| Atlas A2 | `RowMajor`    | 三级自适应（连续块 / 跨步块 / 单行） |
| Atlas A2 | `ColumnMajor` | 三级自适应（连续块 / 跨步块 / 单行） |

**三级搬运策略**：

| 策略   | 触发条件                                        | 方式                                |
| :----- | :---------------------------------------------- | :---------------------------------- |
| 连续块 | 长度对齐 C0 且 stride 对齐 C0 且 stride < 65536 | 一次 `DataCopy`（blockCount = m/n） |
| 跨步块 | 长度对齐 C0 且 stride*C0 < 65536                | C0 条 `DataCopy`，每条 stride 间隔  |
| 单行   | 兜底                                            | 逐条 `DataCopy`                     |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,            // UB 目的
    AscendC::GlobalTensor<Element> srcTensor,           // GM 源
    LayoutDst const &layoutDst,                         // UB 上的 layout（含 round）
    LayoutSrc const &layoutSrc                          // GM 上的 layout（实际尺寸）
)
```

## 调用示例

### RowMajor

```cpp
#include "catlass/gemv/tile/matrix_copy_gm_to_ub.hpp"

using namespace Catlass::Gemv::Tile;

using Element = half;
using LayoutTagSrc = layout::RowMajor;

uint32_t m = 64, n = 128;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(m, n);
auto layoutDst = LayoutTagSrc::MakeLayout<Element>(m, n);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;
using CopyOp = MatrixCopyGmToUB<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
