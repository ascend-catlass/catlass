# CopyL0CToDstQuantMode

> [Code Location](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp) (Ascend 950)/[atlasa2/copy_l0c_to_gm.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp) (Equivalent Type `CopyL0CToGmQuantMode` in Atlas A2)

[TOC]

## Function

The `copy_l0c_to_dst` module defines the **shared infrastructure** for moving L0C data to the destination buffer, including the quantization mode mapping table `CopyL0CToDstQuantMode`, scale granularity enumeration `ScaleGranularity`, UB movement mode enumeration `CopyL0CToUBMode`, and template declarations of `CopyL0CToGmTla` and `CopyL0CToUBTla`.

These items are included and referenced by the [copy_l0c_to_gm](../copy_l0c_to_gm/README.md) and [copy_l0c_to_ub](../copy_l0c_to_ub.md) modules and are not directly exposed to end users.

> **Note**: On the Atlas A2 architecture, the equivalent type is named `CopyL0CToGmQuantMode` (defined in `atlasa2/copy_l0c_to_gm.hpp`). Although the naming differs, its functionality is identical. Ascend 950 supports additional quantization modes.

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

| Granularity| Description| Scale Data Format| Typical Scenario|
| :------ | :------ | :------ | :------ |
| `NO_QUANT` | No quantization| None| Pure type conversion: int32 → int32, float → half/bf16|
| `PER_TENSOR` | Single scale| One `float` scalar| Coarse-grained quantization|
| `PER_CHANNEL` | Per-channel scale| `uint64_t` vector (FixPipe bypass)| Fine-grained quantization|
| `PER_GROUP` | per-group | — | Reserved|

## CopyL0CToDstQuantMode (Ascend 950)

The CopyL0CToDstQuantMode struct maps `(ElementSrc, ElementDst, ScaleGranularity)` to a corresponding `AscendC QuantMode_t` value.

### NO_QUANT

| ElementSrc | ElementDst | VALUE |
| :------ | :------ | :------ |
| `float` | `float` | `NoQuant` |
| `float` | `half` | `F322F16` |
| `float` | `bfloat16_t` | `F322BF16` |
| `int32_t` | `int32_t` | `NoQuant` |

### PER_TENSOR

| ElementSrc | ElementDst | VALUE |
| :------ | :------ | :------ |
| `float` | `uint8_t` / `int8_t` | `QF322B8_PRE` |
| `int32_t` | `half` | `DEQF16` |
| `int32_t` | `uint8_t` / `int8_t` | `REQ8` |
| `int32_t` | `bfloat16_t` | `QS322BF16_PRE` |
| `float` | `half` | `QF322F16_PRE` |
| `float` | `bfloat16_t` | `QF322BF16_PRE` |
| `float` | `float` | `QF322F32_PRE` |
| `float` | `fp8_e4m3fn_t` | `QF322FP8_PRE` |
| `float` | `hifloat8_t` | `QF322HIF8_PRE` |

### PER_CHANNEL

| ElementSrc | ElementDst | VALUE |
| :------ | :------ | :------ |
| `float` | `uint8_t` / `int8_t` | `VQF322B8_PRE` |
| `int32_t` | `half` | `VDEQF16` |
| `int32_t` | `uint8_t` / `int8_t` | `VREQ8` |
| `int32_t` | `bfloat16_t` | `VQS322BF16_PRE` |
| `float` | `half` | `VQF322F16_PRE` |
| `float` | `bfloat16_t` | `VQF322BF16_PRE` |
| `float` | `float` | `VQF322F32_PRE` |
| `float` | `fp8_e4m3fn_t` | `VQF322FP8_PRE` |
| `float` | `hifloat8_t` | `VQF322HIF8_PRE` |

> `CopyL0CToGmQuantMode` of Atlas A2 does not support the `QS322BF16`, `QF322F16`, `QF322BF16` or `QF322F32` mode.

## CopyL0CToUBMode

An enumeration that controls the M/N dimension split strategy when moving data from L0C to UB

```cpp
enum class CopyL0CToUBMode {
    NO_SPLIT = 0,
    SPLIT_M,
    SPLIT_N,
    RESERVED
};
```

| Mode| M Requirement| N requirement| dualDstCtl |
| :------ | :------ | :------ | :------ |
| `NO_SPLIT` | — | — | — |
| `SPLIT_M` | `RoundUp(M, 2)` | — | `1` |
| `SPLIT_N` | — | `RoundUp(N, 32)` | `2` |

## Template Declaration

The following templates are declared in this module; their partial specializations are implemented in [copy_l0c_to_gm](../copy_l0c_to_gm/README.md) and [copy_l0c_to_ub](../copy_l0c_to_ub.md).

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
