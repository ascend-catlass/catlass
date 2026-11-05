# CopyL1ToL0A

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_l0a.hpp)

[TOC]

## 功能说明

`CopyL1ToL0A` 模板负责将 A 矩阵的 tile 块从 L1（Local Memory，A1 Buffer）搬运到 L0A（A2 Buffer），支持多种数据排布格式（layout）转换。

根据源 layout 和目标 layout 的不同，内部会选择合适的硬件搬运指令：

- **zN → zZ**：Nd 转置拷贝，`ifTranspose = false`
- **nZ → zZ**：转置拷贝（Transpose A），`ifTranspose = true`，对于 int8_t 使用 `LoadDataWithTranspose`
- **NDC1HWC0 → zZ**：卷积专用路径，使用 `LoadData3Dv2`

该模板通常不直接使用，而是作为 [TileCopy](../tile_copy/README.md) 的成员类型，由 `blockMmad` 自动管理。仅在需要自定义 kernel 组装时显式声明。

## 模板原型

```cpp
template <
    class ArchTag,                    // 架构标签：Arch::AtlasA2 或 Arch::Ascend950
    class L1Type,                     // L1 数据描述：Gemm::GemmType<Element, Layout, AscendC::TPosition::A1>
    class L0Type = void               // L0A 数据描述：Gemm::GemmType<Element, Layout, AscendC::TPosition::A2>（可省略，自动推导）
>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};
```

- `ArchTag`：架构标签，`Arch::AtlasA2` 或 `Arch::Ascend950`
- `L1Type`：L1 上 A 矩阵的数据类型，封装了 Element、Layout、TPosition
- `L0Type`：L0A 上 A 矩阵的数据类型，默认为 `void`，大多数偏特化会自动推导

## 偏特化实现

### Atlas A2

| 源 Layout | 目标 Layout | 元素类型          | 说明                                            |
| :-------- | :---------- | :---------------- | :---------------------------------------------- |
| zN        | zZ          | 任意              | 基础 Nd 拷贝，使用 LoadData2D                   |
| zN        | zZ          | float             | float 专用，使用 LoadData3D 路径                |
| nZ        | zZ          | 任意（非 int8_t） | 转置拷贝，使用 LoadData2D（ifTranspose=true）   |
| nZ        | zZ          | int8_t            | 转置拷贝，使用 LoadDataWithTranspose            |
| nZ        | zZ          | float             | 转置拷贝，使用 LoadData3D（含 SetFmatrix 对齐） |
| nN        | zZ          | 任意              | nN 转 zZ 拷贝，使用 LoadData2D                  |
| nN        | zZ          | float             | float 专用，使用 LoadData2dTranspose            |
| NDC1HWC0  | zZ          | 任意              | 卷积专用，使用 LoadData3Dv2                     |

### Ascend 950

| 源 Layout | 目标 Layout | 元素类型                                                 | 说明                                               |
| :-------- | :---------- | :------------------------------------------------------- | :------------------------------------------------- |
| zN        | zN          | 任意                                                     | 基础 Nd 拷贝，使用 LoadData2DParamsV2              |
| nZ        | zN          | 非 B8/B4（int8_t/float8_e4m3_t/float8_e5m2_t/float4 等） | 转置拷贝，使用 LoadData2DParamsV2                  |
| nZ        | zN          | B8/B4（int8_t/float8_e4m3_t/float8_e5m2_t/float4 等）    | 转置拷贝，根据 L0M 对齐情况选择单次或分步 LoadData |

> **注意**：Ascend 950 的 `L0Type` 目标 layout 均为 zN（非 zZ），与 Atlas A2 不同。

## 调用接口

### 基础接口（所有偏特化通用）

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,   // L0A 目标 tensor
    AscendC::LocalTensor<Element> srcTensor,   // L1 源 tensor
    LayoutDst layoutDst,                       // L0A 数据 layout
    LayoutSrc layoutSrc                        // L1 数据 layout
);
```

### 卷积接口（NDC1HWC0 偏特化）

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,
    AscendC::LocalTensor<Element> srcTensor,
    LayoutDst layoutDst, LayoutSrc layoutSrc,
    uint32_t kStartPt, uint32_t mStartPt,
    uint32_t l1H, uint32_t l1W, uint8_t* padList
);
```

该偏特化通过静态工厂方法构造实例：

```cpp
static CopyL1ToL0A MakeCopyL1ToL0A(
    uint32_t strideW = 0, uint32_t strideH = 0,
    uint32_t filterW = 0, uint32_t filterH = 0,
    uint32_t dilationFilterW = 0, uint32_t dilationFilterH = 0
);
```

## 调用示例

### 基础 zN → zZ 搬运（Atlas A2，非 TLA）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using L1Type = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>;

uint32_t row = 256;
uint32_t col = 256;

// 构造 L1 上的 zN layout 和 L0A 上的 zZ layout
auto layoutSrc = layout::zN::MakeLayout<Element>(row, col);
auto layoutDst = layout::zZ::MakeLayout<Element>(row, col);

AscendC::LocalTensor<Element> srcL1Tensor;
AscendC::LocalTensor<Element> dstL0ATensor;

// 实例化并调用
using CopyOp = CopyL1ToL0A<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0ATensor, srcL1Tensor, layoutDst, layoutSrc);
```

### nZ → zZ 转置搬运（Atlas A2，非 TLA）

```cpp
using L1Type = Gemm::GemmType<half, layout::nZ, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<half, layout::zZ, AscendC::TPosition::A2>;

auto layoutSrc = layout::nZ::MakeLayout<half>(row, col);
auto layoutDst = layout::zZ::MakeLayout<half>(row, col);

using CopyOp = CopyL1ToL0A<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0ATensor, srcL1Tensor, layoutDst, layoutSrc);
```

### zN → zN 搬运（Ascend 950，非 TLA）

```cpp
using L1Type = Gemm::GemmType<half, layout::zN, AscendC::TPosition::A1>;

auto layoutSrc = layout::zN::MakeLayout<half>(row, col);
auto layoutDst = layout::zN::MakeLayout<half>(row, col);

// Ascend 950 的 L0Type 可省略，自动推导对应 layout
using CopyOp = CopyL1ToL0A<Arch::Ascend950, L1Type>;
CopyOp copyOp;
copyOp(dstL0ATensor, srcL1Tensor, layoutDst, layoutSrc);
```
