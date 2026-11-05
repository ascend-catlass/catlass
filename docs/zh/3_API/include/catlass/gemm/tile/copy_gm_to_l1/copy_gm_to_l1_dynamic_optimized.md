# CopyGmToL1DynamicOptimized

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## 功能说明

`CopyGmToL1DynamicOptimized` 是非 TLA 风格的 GM 到 L1 数据搬运模板。与 `CopyGmToL1` 相比，该模板在运行时根据数据块形状动态选择最优搬运策略：

- 当矩阵行数/列数较少（≤16）时，使用逐行/逐列的 strided `DataCopy` 接口，避免 `Nd2Nz` 指令开销
- 当矩阵规模较大时，回退到 `Nd2Nz` 指令进行高效搬运

对于 zN → zN 和 nZ → nZ 等格式保持不变的场景，直接继承自 `CopyGmToL1` 对应偏特化。

支持 `Arch::AtlasA2` 和 `Arch::Ascend950` 两种架构。

## 模板原型

```cpp
template <
    class ArchTag,          // 架构标签
    class GmType,           // GM 上操作数的 Gemm 类型
    class L1Type = void     // L1 上操作数的 Gemm 类型（默认 void）
>
struct CopyGmToL1DynamicOptimized
```

### 模板参数说明

| 参数      | 说明                                                |
| :-------- | :-------------------------------------------------- |
| `ArchTag` | 架构标签，可选 `Arch::AtlasA2` 或 `Arch::Ascend950` |
| `GmType`  | GM 上源操作数的 Gemm 类型                           |
| `L1Type`  | L1 上目的操作数的 Gemm 类型，默认为 `void`          |

## 偏特化实现

### Atlas A2 偏特化

| GmType                                  | 目的 Layout | 实现方式                                                          |
| :-------------------------------------- | :---------- | :---------------------------------------------------------------- |
| `GemmType<Element, RowMajor>`           | `zN`        | 自主实现，动态选择策略                                            |
| `GemmType<Element, ColumnMajor>`        | `nZ`        | 自主实现，动态选择策略                                            |
| `GemmType<Element, zN>`                 | `zN`        | 继承自 `CopyGmToL1<AtlasA2, GmType<Element, zN>>`                 |
| `GemmType<Element, nZ>`                 | `nZ`        | 继承自 `CopyGmToL1<AtlasA2, GmType<Element, nZ>>`                 |
| `GemmType<Element, PaddingRowMajor>`    | `zN`        | 继承自 `CopyGmToL1<AtlasA2, GmType<Element, PaddingRowMajor>>`    |
| `GemmType<Element, PaddingColumnMajor>` | `nZ`        | 继承自 `CopyGmToL1<AtlasA2, GmType<Element, PaddingColumnMajor>>` |

### Ascend 950 偏特化

| GmType                           | 目的 Layout | 实现方式                                            |
| :------------------------------- | :---------- | :-------------------------------------------------- |
| `GemmType<Element, RowMajor>`    | `zN`        | 自主实现，动态选择策略                              |
| `GemmType<Element, ColumnMajor>` | `nZ`        | 自主实现，动态选择策略                              |
| `GemmType<Element, zN>`          | `zN`        | 继承自 `CopyGmToL1<Ascend950, GmType<Element, zN>>` |
| `GemmType<Element, nZ>`          | `nZ`        | 继承自 `CopyGmToL1<Ascend950, GmType<Element, nZ>>` |

## 调用接口

所有偏特化使用统一的调用接口：

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,   // 目的操作数 LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,  // 源操作数 GlobalTensor
    LayoutDst const &layoutDst,                       // 目的操作数 layout
    LayoutSrc const &layoutSrc                        // 源操作数 layout
)
```

| 参数        | 说明                     |
| :---------- | :----------------------- |
| `dstTensor` | 目的 L1 LocalTensor      |
| `srcTensor` | 源 GM GlobalTensor       |
| `layoutDst` | 目的操作数的 layout 描述 |
| `layoutSrc` | 源操作数的 layout 描述   |

## 调用示例

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;
using ElementDst = half;

// 定义 GM 上的 Gemm 类型
using GmType = Gemm::GemmType<ElementDst, LayoutTagSrc>;
// 定义 L1 上的 Gemm 类型
using L1Type = Gemm::GemmType<ElementDst, LayoutTagDst, AscendC::TPosition::A1>;

uint32_t row = 256;
uint32_t col = 256;

// 构造 layout
auto layoutSrc = LayoutTagSrc::MakeLayout<ElementDst>(row, col);
auto layoutDst = LayoutTagDst::MakeLayout<ElementDst>(row, col);

AscendC::GlobalTensor<ElementDst> srcTensor;
AscendC::LocalTensor<ElementDst> dstTensor;

// 实例化 CopyGmToL1DynamicOptimized
// 内部会根据 row/col 自动选择 Nd2Nz 或 strided DataCopy
using CopyOp = CopyGmToL1DynamicOptimized<Arch::AtlasA2, GmType, L1Type>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
