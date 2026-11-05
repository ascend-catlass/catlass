# TilePerTokenDequant

> [代码位置](../../../../../../../include/catlass/epilogue/tile/tile_pertoken_dequant.hpp)

[TOC]

## 功能说明

`TilePerTokenDequant` 实现 epilogue 阶段的 per-token 反量化操作。将 UB 上的 int32 累加结果乘以 per-channel scale 和 per-token scale，反量化为目标浮点类型。

- 适用范围：仅 `Arch::Ascend950`
- 风格：TLA（使用 `tla::Tensor` 封装操作数，内部使用微架构 intrinsic 指令）
- 计算流程：`dst[i,j] = (int32)src[i,j] * (float)scale[j] * (float)perToken[i]`
- 使用 `__simd_vf__` 内联汇编级优化，在 register 上完成 int32→float32 类型转换

## 模板原型

```cpp
template <
    class ArchTag_,          // 架构标签（静态断言仅 Ascend 950）
    class ElementSrc_,       // 源元素类型（静态断言 int32_t）
    class ElementScale_,     // per-channel scale 元素类型（half/bfloat16_t/float）
    class ElementPerToken_,  // per-token scale 元素类型（half/bfloat16_t/float）
    class ElementDst_,       // 目标元素类型（half/bfloat16_t/float）
    class TileShape_         // Tile 形状类型（含 COLUMN）
>
struct TilePerTokenDequant;
```

| 模板参数           | 说明                                                        |
| :----------------- | :---------------------------------------------------------- |
| `ArchTag_`         | 仅支持 `Arch::Ascend950`，编译期断言检查                    |
| `ElementSrc_`      | 源元素类型，仅支持 `int32_t`                                |
| `ElementScale_`    | per-channel scale 元素类型：`half` / `bfloat16_t` / `float` |
| `ElementPerToken_` | per-token scale 元素类型：`half` / `bfloat16_t` / `float`   |
| `ElementDst_`      | 目标元素类型：`half` / `bfloat16_t` / `float`               |
| `TileShape_`       | Tile 形状，`TileShape_::COLUMN` 用于确定 N_BASE_SIZE        |

## 调用接口

```cpp
template <class TensorDst, class TensorSrc, class TensorScale, class TensorPerToken>
void operator()(
    TensorDst const &ubOut,               // 目标 UB TLA Tensor（RowMajor）
    TensorSrc const &ubIn,                // 源 UB TLA Tensor（RowMajor, int32_t）
    TensorScale const &ubScale,           // per-channel scale TLA Tensor（VectorLayout）
    TensorPerToken const &ubPerToken      // per-token scale TLA Tensor（VectorLayout）
)
```

| 参数         | 说明                                                                               |
| :----------- | :--------------------------------------------------------------------------------- |
| `ubOut`      | UB TLA Tensor，类型 `ElementDst`，布局 `RowMajor`                                  |
| `ubIn`       | UB TLA Tensor，类型 `int32_t`，布局 `RowMajor`，MMAD 累加输出                      |
| `ubScale`    | per-channel scale，类型 `ElementScale`，布局 `VectorLayout`（status=0），长度 = n  |
| `ubPerToken` | per-token scale，类型 `ElementPerToken`，布局 `VectorLayout`（status=0），长度 = m |

静态断言确保所有 Tensor 的 position 为 `UB`，且布局匹配。

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_pertoken_dequant.hpp"

using namespace Catlass::Epilogue::Tile;

constexpr uint32_t M = 128;
constexpr uint32_t N = 256;

using TileShape = Shape<M, N>;

using DequantOp = TilePerTokenDequant<Arch::Ascend950, int32_t, half, half, half, TileShape>;

auto srcLayout = tla::MakeLayout<int32_t, layout::RowMajor>(M, N);
auto scaleLayout = tla::MakeLayout<half, layout::VectorLayout>(N, 1);
auto perTokenLayout = tla::MakeLayout<half, layout::VectorLayout>(M, 1);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);

AscendC::LocalTensor<int32_t> ubIn;
AscendC::LocalTensor<half> ubScale, ubPerToken, ubOut;

auto srcTensor = tla::MakeTensor(ubIn, srcLayout, Arch::PositionUB{});
auto scaleTensor = tla::MakeTensor(ubScale, scaleLayout, Arch::PositionUB{});
auto perTokenTensor = tla::MakeTensor(ubPerToken, perTokenLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(ubOut, dstLayout, Arch::PositionUB{});

DequantOp dequantOp;
dequantOp(dstTensor, srcTensor, scaleTensor, perTokenTensor);
```
