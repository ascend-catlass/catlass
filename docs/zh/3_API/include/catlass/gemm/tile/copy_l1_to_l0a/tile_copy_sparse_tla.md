# TileCopySparseTla（L1 → L0A，Sparse）

> [代码位置](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp)

[TOC]

## 功能说明

`TileCopySparseTla` 在 `copy_l1_to_l0a.hpp` 中定义的偏特化负责 Sparse GEMM 场景下将稀疏化的 A 矩阵 tile 块从 L1（A1 Buffer）搬运到 L0A（A2 Buffer）。通过 `LoadData3DParamsV2Pro` + `LoadData` 指令实现 zN→zZ 的格式转换。

> **限制**：该模板仅支持 Atlas A2 架构（`CATLASS_ARCH == 2201`），Ascend 950 不支持。

## 模板原型

```cpp
template <
    class ArchTag,        // 架构标签
    class TensorSrc,      // 源 Tensor（L1）
    class TensorDst,      // 目标 Tensor（L0A）
    class Enable = void   // SFINAE 分发
>
struct TileCopySparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopySparseTla, can not find the specialization.");
};
```

| 模板参数    | 说明                                                                |
| :---------- | :------------------------------------------------------------------ |
| `ArchTag`   | 架构标签，仅支持 `Arch::AtlasA2`                                    |
| `TensorSrc` | 源 Tensor：`tla::Tensor<LocalTensor<Element>, Layout, Coord, A1>`   |
| `TensorDst` | 目标 Tensor：`tla::Tensor<LocalTensor<Element>, Layout, Coord, A2>` |
| `Enable`    | SFINAE 条件，根据 Layout 自动派发偏特化                             |

## 偏特化实现

### Atlas A2

| 源 Layout | 目标 Layout | SFINAE 条件                          | 说明                       |
| :-------- | :---------- | :----------------------------------- | :------------------------- |
| zN        | zZ          | `iszN<LayoutSrc> && iszZ<LayoutDst>` | LoadData3D v2 Pro，16 对齐 |

硬件参数：`HW_N0 = 16`, `HW_M0 = 16`，通过 `Load3DSetFMatrixCal` 设置矩阵计算参数，`LoadData3DParamsV2Pro.extConfig` 携带 M/K 坐标偏移和步长。

## 调用接口

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

- `dstTensor`：L0A 上的目标 Tensor（zZ 格式）
- `srcTensor`：L1 上的源 Tensor（zN 格式）

## 调用示例

### zN L1 → zZ L0A（Atlas A2）

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;

auto srcLayout = tla::MakeLayout<half, layout::zN>(M, K);
auto dstLayout = tla::MakeLayout<half, layout::zZ>(M, K);

AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0ATensor;

auto srcTensor = tla::MakeTensor(srcL1Tensor, srcLayout, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, dstLayout, Arch::PositionL0A{});

TileCopySparseTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor);
```
