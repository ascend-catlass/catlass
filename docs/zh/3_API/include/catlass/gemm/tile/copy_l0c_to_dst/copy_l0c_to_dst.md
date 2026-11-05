# CopyL0CToDstQuantMode

> [代码位置](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp)（Ascend 950） / [atlasa2/copy_l0c_to_gm.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp)（Atlas A2 等效 `CopyL0CToGmQuantMode`）

[TOC]

## 功能说明

`copy_l0c_to_dst` 模块定义了 L0C 数据搬运到目标缓冲区的**共享基础设施**，包括量化模式映射表 `CopyL0CToDstQuantMode`、Scale 粒度枚举 `ScaleGranularity`、UB 搬运模式枚举 `CopyL0CToUBMode`，以及 `CopyL0CToGmTla`、`CopyL0CToUBTla` 的模板声明。

这些类型被 [copy_l0c_to_gm](../copy_l0c_to_gm/README.md) 和 [copy_l0c_to_ub](../copy_l0c_to_ub.md) 模块 include 引用，并不直接暴露给最终用户。

> **注意**：Atlas A2 架构中的等效类型名为 `CopyL0CToGmQuantMode`（定义在 `atlasa2/copy_l0c_to_gm.hpp`），命名不同但功能一致。Ascend 950 额外支持更多量化模式。

## ScaleGranularity

```cpp
enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};
```

| 粒度          | 说明         | scale 数据形式                  | 典型场景                                |
| :------------ | :----------- | :------------------------------ | :-------------------------------------- |
| `NO_QUANT`    | 不量化       | 无                              | int32→int32、float→half/bf16 纯类型转换 |
| `PER_TENSOR`  | 单一 scale   | 1 个 `float` 标量               | 粗粒度量化                              |
| `PER_CHANNEL` | 逐通道 scale | `uint64_t` 向量（FixPipe 旁路） | 细粒度量化                              |
| `PER_GROUP`   | per-group    | —                               | 保留                                    |

## CopyL0CToDstQuantMode（Ascend 950）

将 `(ElementSrc, ElementDst, ScaleGranularity)` 映射为 AscendC `QuantMode_t`。

### NO_QUANT

| ElementSrc | ElementDst   | VALUE      |
| :--------- | :----------- | :--------- |
| `float`    | `float`      | `NoQuant`  |
| `float`    | `half`       | `F322F16`  |
| `float`    | `bfloat16_t` | `F322BF16` |
| `int32_t`  | `int32_t`    | `NoQuant`  |

### PER_TENSOR

| ElementSrc | ElementDst           | VALUE           |
| :--------- | :------------------- | :-------------- |
| `float`    | `uint8_t` / `int8_t` | `QF322B8_PRE`   |
| `int32_t`  | `half`               | `DEQF16`        |
| `int32_t`  | `uint8_t` / `int8_t` | `REQ8`          |
| `int32_t`  | `bfloat16_t`         | `QS322BF16_PRE` |
| `float`    | `half`               | `QF322F16_PRE`  |
| `float`    | `bfloat16_t`         | `QF322BF16_PRE` |
| `float`    | `float`              | `QF322F32_PRE`  |

### PER_CHANNEL

| ElementSrc | ElementDst           | VALUE            |
| :--------- | :------------------- | :--------------- |
| `float`    | `uint8_t` / `int8_t` | `VQF322B8_PRE`   |
| `int32_t`  | `half`               | `VDEQF16`        |
| `int32_t`  | `uint8_t` / `int8_t` | `VREQ8`          |
| `int32_t`  | `bfloat16_t`         | `VQS322BF16_PRE` |
| `float`    | `half`               | `VQF322F16_PRE`  |
| `float`    | `bfloat16_t`         | `VQF322BF16_PRE` |
| `float`    | `float`              | `VQF322F32_PRE`  |

> Atlas A2 的 `CopyL0CToGmQuantMode` 不支持 `QS322BF16`、`QF322F16`、`QF322BF16`、`QF322F32` 等模式。

## CopyL0CToUBMode

UB 搬运模式枚举，控制 L0C→UB 搬运时的 M/N 维度 split 策略：

```cpp
enum class CopyL0CToUBMode {
    NO_SPLIT = 0,
    SPLIT_M,
    SPLIT_N,
    RESERVED
};
```

| 模式       | M 要求          | N 要求           | dualDstCtl |
| :--------- | :-------------- | :--------------- | :--------- |
| `NO_SPLIT` | —               | —                | —          |
| `SPLIT_M`  | `RoundUp(M, 2)` | —                | `1`        |
| `SPLIT_N`  | —               | `RoundUp(N, 32)` | `2`        |

## 模板声明

以下声明在本模块中定义，偏特化实现在 [copy_l0c_to_gm](../copy_l0c_to_gm/README.md) 和 [copy_l0c_to_ub](../copy_l0c_to_ub.md) 中：

```cpp
template <class ArchTag, class TensorSrc, class TensorDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false, class Enable = void>
struct CopyL0CToGmTla;

template <class ArchTag, class TensorSrc, class TensorDst,
    CopyL0CToUBMode CopyMode = CopyL0CToUBMode::NO_SPLIT,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false, class Enable = void>
struct CopyL0CToUBTla;
```
