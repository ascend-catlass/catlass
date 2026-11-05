# CopyL0CToGm

> [代码位置](../../../../../../../../include/catlass/conv/tile/copy_l0c_to_gm.hpp)

[TOC]

## 功能说明

`CopyL0CToGm` 实现 Conv 场景下累加结果从 L0C（zN 格式）写回 GM（NC1HWC0 格式）的非 TLA 版本。通过 `AscendC::Fixpipe` 直接通路完成数据搬运、类型转换（F322F16/F322BF16）和可选 ReLU 激活。

- 适用范围：Atlas A2
- 风格：非 TLA

## 模板原型

```cpp
template <class ArchTag, class ElementAccumulator, class GmType,
          ScaleGranularity DEQUANT_GRANULARITY = NO_QUANT, bool ReluEnable = false>
struct CopyL0CToGm;
```

| 模板参数              | 说明                                                           |
| :-------------------- | :------------------------------------------------------------- |
| `ArchTag`             | 架构标签                                                       |
| `ElementAccumulator`  | 累加元素类型，如 `float`                                       |
| `GmType`              | `Gemm::GemmType<ElementDst, NC1HWC0>`                          |
| `DEQUANT_GRANULARITY` | 量化模式：`NO_QUANT`、`PER_TENSOR`、`PER_CHANNEL`、`PER_GROUP` |
| `ReluEnable`          | 是否启用 ReLU，默认 `false`                                    |

## 偏特化实现

| 偏特化       | GmType                          | 说明                         |
| :----------- | :------------------------------ | :--------------------------- |
| A2, NO_QUANT | `GemmType<ElementDst, NC1HWC0>` | zN→NC1HWC0，逐 Ho 行 Fixpipe |

## 调用接口

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> const &dst,    // (Batch, Cout1, Ho, Wo, C0) NC1HWC0
    AscendC::LocalTensor<ElementSrc> const &src,     // L0C zN 格式
    LayoutDst const &dstLayout,
    uint8_t unitFlag = 0
)
```
