# CopyL1ToL0B

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_l0b.hpp)

[TOC]

## 功能说明

`CopyL1ToL0B` 模板负责将 B 矩阵的 tile 块从 L1（Local Memory，B1 Buffer）搬运到 L0B（B2 Buffer），支持多种数据排布格式（layout）转换。

根据源 layout 和目标 layout 的不同，内部会选择合适的硬件搬运指令：

- **zZ → nZ**：转置拷贝（Transpose B），`ifTranspose = true`，对于 float 使用 `LoadDataWithTranspose`
- **zN → nZ**：转置拷贝，对于 int8_t 使用 `LoadDataWithTranspose`，float 使用 `LoadData3D + SetFmatrix`
- **nZ → nZ**：非转置拷贝（直传），`ifTranspose = false`
- **zN → zN / nN → zN**：GEMV 专用路径

该模板通常不直接使用，而是作为 [TileCopy](../tile_copy/README.md) 的成员类型，由 `blockMmad` 自动管理。仅在需要自定义 kernel 组装时显式声明。

## 模板原型

```cpp
template <
    class ArchTag,                    // 架构标签：Arch::AtlasA2 或 Arch::Ascend950
    class L1Type,                     // L1 数据描述：Gemm::GemmType<Element, Layout, AscendC::TPosition::B1>
    class L0Type = void               // L0B 数据描述：Gemm::GemmType<Element, Layout, AscendC::TPosition::B2>（可省略，自动推导）
>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};
```

- `ArchTag`：架构标签，`Arch::AtlasA2` 或 `Arch::Ascend950`
- `L1Type`：L1 上 B 矩阵的数据类型，封装了 Element、Layout、TPosition
- `L0Type`：L0B 上 B 矩阵的数据类型，默认为 `void`，大多数偏特化会自动推导

## 偏特化实现

### Atlas A2 — Gemm 场景（zZ/zN → nZ）

| 源 Layout | 目标 Layout | 元素类型                | 说明                                              |
| :-------- | :---------- | :---------------------- | :------------------------------------------------ |
| zZ （B1） | nZ （B2）   | 任意                    | 基础转置拷贝，使用 LoadData2D（ifTranspose=true） |
| zZ （B1） | nZ （B2）   | float                   | float 专用，使用 LoadData2dTranspose              |
| zN （B1） | nZ （B2）   | int8_t                  | int8_t 转置，使用 LoadDataWithTranspose           |
| zN （A1） | nZ （B2）   | int8_t                  | int8_t zN→nZ 转置（单参数 L1Type 偏特化）         |
| zN （A1） | nZ （B2）   | float                   | float zN→nZ 转置，使用 LoadData3D + SetFmatrix    |
| zN （A1） | nZ （B2）   | 任意（非 int8_t/float） | 通用 zN→nZ 转置                                   |
| zN （A1） | nZ （B2）   | AscendC::int4b_t        | int4b_t zN→nZ 转置，使用 LoadDataWithTranspose    |
| nZ （B1） | nZ （B2）   | 任意                    | nZ→nZ 非转置拷贝（直传）                          |
| nZ （A1） | nZ （B2）   | 任意                    | nZ→nZ 直传（单参数 L1Type 偏特化）                |

### Atlas A2 — GEMV 场景（zN/nN → zN）

| 源 Layout | 目标 Layout | 元素类型 | 说明                      |
| :-------- | :---------- | :------- | :------------------------ |
| zN （B1） | zN （B2）   | 任意     | GEMV 用 zN→zN 拷贝        |
| nN （B1） | zN （B2）   | 任意     | GEMV 用 nN→zN 转置拷贝    |
| nN （B1） | zN （B2）   | float    | float GEMV 用 nN→zN 转置  |
| nZ （B1） | zN （B2）   | int8_t   | int8_t GEMV 用 nZ→zN 转置 |

### Ascend 950

| 源 Layout           | 目标 Layout | 元素类型                    | 说明                                                           |
| :------------------ | :---------- | :-------------------------- | :------------------------------------------------------------- |
| nZ （A1）           | nZ （B2）   | 任意                        | nZ→nZ 非转置拷贝，使用 LoadData2DParamsV2                      |
| zN （A1），非 B8/B4 | nZ （B2）   | 非 int8_t/float8_/float4 等 | zN→nZ 转置拷贝，使用 LoadData2DParamsV2                        |
| zN （A1），B8/B4    | nZ （B2）   | int8_t/float8_/float4 等    | B8/B4 zN→nZ 转置拷贝，根据 L0N 对齐情况选择单次或分步 LoadData |

> **注意**：Ascend 950 的 L0Type 目标 layout 为 nZ（非 zZ），且 L1Type 的 Position 为 A1（非 B1），与 Atlas A2 不同。

## 调用接口

### 基础接口（所有偏特化通用）

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,   // L0B 目标 tensor
    AscendC::LocalTensor<Element> srcTensor,   // L1 源 tensor
    LayoutDst layoutDst,                       // L0B 数据 layout
    LayoutSrc layoutSrc                        // L1 数据 layout
);
```

## 调用示例

### zZ → nZ 转置搬运（Atlas A2，非 TLA）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using L1Type = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>;
using L0Type = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;

uint32_t k = 256;
uint32_t n = 256;

auto layoutSrc = layout::zZ::MakeLayout<Element>(k, n);
auto layoutDst = layout::nZ::MakeLayout<Element>(k, n);

AscendC::LocalTensor<Element> srcL1Tensor;
AscendC::LocalTensor<Element> dstL0BTensor;

using CopyOp = CopyL1ToL0B<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0BTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### zN → nZ 转置搬运（Atlas A2，单参数 L1Type）

```cpp
using L1Type = Gemm::GemmType<half, layout::zN, AscendC::TPosition::A1>;

auto layoutSrc = layout::zN::MakeLayout<half>(k, n);
auto layoutDst = layout::nZ::MakeLayout<half>(k, n);

// 单参数 L1Type 偏特化，L0Type 自动推导
using CopyOp = CopyL1ToL0B<Arch::AtlasA2, L1Type>;
CopyOp copyOp;
copyOp(dstL0BTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### nZ → nZ 直传搬运（Atlas A2，非 TLA）

```cpp
using L1Type = Gemm::GemmType<half, layout::nZ, AscendC::TPosition::B1>;
using L0Type = Gemm::GemmType<half, layout::nZ, AscendC::TPosition::B2>;

auto layoutSrc = layout::nZ::MakeLayout<half>(k, n);
auto layoutDst = layout::nZ::MakeLayout<half>(k, n);

using CopyOp = CopyL1ToL0B<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0BTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### nZ → nZ 搬运（Ascend 950，非 TLA）

```cpp
using L1Type = Gemm::GemmType<half, layout::nZ, AscendC::TPosition::A1>;

auto layoutSrc = layout::nZ::MakeLayout<half>(k, n);
auto layoutDst = layout::nZ::MakeLayout<half>(k, n);

// Ascend950 L1Type Position 为 A1，L0Type 可省略
using CopyOp = CopyL1ToL0B<Arch::Ascend950, L1Type>;
CopyOp copyOp;
copyOp(dstL0BTensor, srcL1Tensor, layoutDst, layoutSrc);
```
