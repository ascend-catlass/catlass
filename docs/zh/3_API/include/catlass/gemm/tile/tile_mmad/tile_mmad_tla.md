# TileMmadTla

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_mmad.hpp)

[TOC]

## 功能说明

`TileMmadTla` 是 [TileMmad](./tile_mmad.md) 的 TLA 版本，使用 `AscendC::Mmad` 完成 `C += A * B`。所有操作数通过 `tla::Tensor` 封装（`l0ATensor.data()` + `.layout().originShape()` 自动提取维度）。

支持四种调用模式：

1. 标准矩阵乘加（无 Bias）
2. 带 Bias 矩阵乘加
3. L0 Batch 批量 Mmad（多 batch 矩阵）
4. 自动维度提取（从 `tla::Tensor.layout().originShape()` 推导 m/n/k）

### 架构差异

| 架构             | `kDirectionAlign`         | `disableGemv`               | 自动 GEMV 规避       |
| :--------------- | :------------------------ | :-------------------------- | :------------------- |
| Atlas A2 (2201)   | `float` + L1A `nZ` 时开启 | —                           | 模式 4 中 M=1 → M=16 |
| Ascend 950 (3510) | —                         | L1A `VectorLayout` 时 false | —                    |

## 模板原型

```cpp
template <
    class ArchTag,            // 架构标签
    class ElementA,           // A 矩阵元素类型
    class LayoutTagL1A        // A 矩阵 L1 layout tag（用于架构差异判断）
>
struct TileMmadTla;
```

## 调用接口

### 模式 1：标准无 Bias

```cpp
template <class TensorC, class TensorA, class TensorB>
void operator()(
    TensorC const &l0CTensor,    // L0C tensor（zZ）
    TensorA const &l0ATensor,    // L0A tensor（zZ）
    TensorB const &l0BTensor,    // L0B tensor（nZ）
    uint32_t m, uint32_t n, uint32_t k,
    bool initC = true,
    uint8_t unitFlag = 0
);
```

### 模式 2：带 Bias

```cpp
template <class TensorC, class TensorA, class TensorB, class TensorBias>
void operator()(
    TensorC const &l0CTensor,
    TensorA const &l0ATensor,
    TensorB const &l0BTensor,
    TensorBias const &l0BiasTensor,    // BT Bias tensor
    uint32_t m, uint32_t n, uint32_t k,
    bool initC = true,
    uint8_t unitFlag = 0
);
```

与模式 1 的区别：`cmatrixInitVal = false`（Bias 已预设到 L0C），`disableGemv = true`（Ascend 950）。

### 模式 3：L0 Batch（批量 Mmad）

```cpp
template <class TensorC, class TensorA, class TensorB>
void operator()(
    TensorC const &l0CTensor,
    TensorA const &l0ATensor,
    TensorB const &l0BTensor,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t l0Batch                // batch 数量
);
```

每个 batch 的偏移量从 `tla::get<x,y>(tensor.shape())` 乘积计算，`cmatrixInitVal = true` 且 batch 间互不累加。

### 模式 4：自动维度提取

```cpp
template <class TensorC, class TensorA, class TensorB>
void operator()(
    TensorC const &l0CTensor,    // m,n 从 originShape 提取
    TensorA const &l0ATensor,    // k 从 originShape[1] 提取
    TensorB const &l0BTensor,
    bool initC = true,
    uint8_t unitFlag = 0
);
```

维度推导：

- `m = tla::get<0>(l0CTensor.layout().originShape())`
- `n = tla::get<1>(l0CTensor.layout().originShape())`
- `k = tla::get<1>(l0ATensor.layout().originShape())`

Atlas A2 下 `m=1` 时自动提升为 16，避免进入低效 GEMV 模式。

## 调用示例

### 模式 1：标准无 Bias

```cpp
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm;

using ElementA = half;
using ElementB = half;
using ElementC = float;

auto l0cLayout = tla::MakeLayout<ElementC, layout::zZ>(64, 64);
auto l0aLayout = tla::MakeLayout<ElementA, layout::zZ>(64, 32);
auto l0bLayout = tla::MakeLayout<ElementB, layout::nZ>(32, 64);

auto l0cTensor = tla::MakeTensor(l0cData, l0cLayout, Arch::PositionL0C{});
auto l0aTensor = tla::MakeTensor(l0aData, l0aLayout, Arch::PositionL0A{});
auto l0bTensor = tla::MakeTensor(l0bData, l0bLayout, Arch::PositionL0B{});

Tile::TileMmadTla<Arch::AtlasA2, ElementA, layout::zN> mmadOp;
mmadOp(l0cTensor, l0aTensor, l0bTensor, 64, 64, 32);
```

### 模式 2：带 Bias

```cpp
using ElementBias = float;
auto btLayout = tla::MakeLayout<ElementBias, layout::VectorLayout>(64);
auto btTensor = tla::MakeTensor(btData, btLayout, Arch::PositionBT{});

mmadOp(l0cTensor, l0aTensor, l0bTensor, btTensor, 64, 64, 32);
```

### 模式 4：自动维度提取（推荐）

```cpp
// 无需手动传 m/n/k，由 originShape 推导
mmadOp(l0cTensor, l0aTensor, l0bTensor);

// 后续 mmad：原子累加
mmadOp(l0cTensor, l0aTensor, l0bTensor, false);
```

### 模式 3：L0 Batch

```cpp
mmadOp(l0cTensor, l0aTensor, l0bTensor, 64, 64, 32, 4);  // 4 个 batch
```
