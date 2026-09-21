# TileVmad

> [代码位置](../../../../../../../include/catlass/gemv/tile/tile_vmad.hpp)

[TOC]

## 功能说明

`TileVmad` 实现 GEMV 场景下的向量-矩阵乘加运算（`Y += A * X`）。将矩阵 A（m×n）的每行与向量 X（n 维）做内积，结果累加到向量 Y（m 维）。

- 适用范围：AtlasA2
- 两种实现：RowMajor 矩阵（MulAddDst + WholeReduceSum）+ ColumnMajor 矩阵（Axpy 逐列）

## 模板原型

```cpp
template <class ArchTag, class AType, class XType, class YType, class BiasType = void>
struct TileVmad;
```

| 模板参数   | 说明                                                  |
| :--------- | :---------------------------------------------------- |
| `ArchTag`  | 架构标签                                              |
| `AType`    | A 矩阵类型 `GemmType<ElementA, RowMajor/ColumnMajor>` |
| `XType`    | X 向量类型 `GemmType<ElementX, VectorLayout>`         |
| `YType`    | Y 向量类型 `GemmType<ElementY, VectorLayout>`         |
| `BiasType` | 偏置类型，默认 `void`                                 |

## 偏特化实现

| 架构    | AType         | 实现策略                                              | 特殊版本                      |
| :------ | :------------ | :---------------------------------------------------- | :---------------------------- |
| AtlasA2 | `RowMajor`    | `Duplicate`→`MulAddDst`→`WholeReduceSum`→`Cast`→`Add` | float 版本：`Mul`+`MulAddDst` |
| AtlasA2 | `ColumnMajor` | `Duplicate`→scalar 抖动→`Axpy`逐列→`Cast`→`Add`       | float 版本同步                |

**RowMajor 实现流程**：

1. `Duplicate` 初始化累加缓冲区 temp 为 0
2. 分块 `MulAddDst` 计算 `A[i:*n] * X[i:*n]`，累加到 temp
3. `WholeReduceSum` 归约 temp 到列向量
4. `Cast` 转回 ElementA 类型
5. `Add` 累加到 Y

**ColumnMajor 实现流程**：

1. `Duplicate` 初始化 temp 为 0
2. 通过 `SetFlag/WaitFlag` 将向量 X 的值加载到 scalar 寄存器
3. 逐列 `Axpy`（`temp += A[:,i] * pix[i]`）
4. `Cast` + `Add` 累加到 Y

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementY> dstTensor,           // Y 向量（UB，含累加结果）
    AscendC::LocalTensor<ElementX> srcTensor_v,         // X 向量（UB）
    AscendC::LocalTensor<ElementA> srcTensor_m,         // A 矩阵（UB）
    AscendC::LocalTensor<ElementAccumulator> temp,      // 临时缓冲（UB）
    LayoutDst const &layoutDst,                         // A 矩阵的实际 layout
    LayoutSrc const &layoutSrc                          // A 矩阵的 round layout
)
```

## 调用示例

### RowMajor

```cpp
#include "catlass/gemv/tile/tile_vmad.hpp"

using namespace Catlass::Gemv::Tile;

using ElementA = half;
using ElementX = half;
using ElementY = half;

using LayoutTagSrc = layout::RowMajor;

uint32_t m = 64, n = 128;

auto layoutSrc = LayoutTagSrc::MakeLayout<ElementA>(m, n);
auto layoutDst = LayoutTagSrc::MakeLayout<ElementA>(m, n);

AscendC::LocalTensor<ElementY> dstTensor;
AscendC::LocalTensor<ElementX> srcTensor_v;
AscendC::LocalTensor<ElementA> srcTensor_m;
AscendC::LocalTensor<float> temp;

using AType = Gemm::GemmType<ElementA, LayoutTagSrc>;
using XType = Gemm::GemmType<ElementX, layout::VectorLayout>;
using YType = Gemm::GemmType<ElementY, layout::VectorLayout>;

using VmadOp = TileVmad<Arch::AtlasA2, AType, XType, YType>;
VmadOp vmad;
vmad(dstTensor, srcTensor_v, srcTensor_m, temp, layoutDst, layoutSrc);
```
