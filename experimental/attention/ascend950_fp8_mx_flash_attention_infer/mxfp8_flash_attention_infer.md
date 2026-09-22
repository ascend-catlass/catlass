# CATLASS MXFP8 FlashAttention Infer设计文档

## 概述

CATLASS MXFP8 FlashAttention Infer 是基于 [49_ascend950_flash_attention_infer](../../../examples/49_ascend950_flash_attention_infer/flash_attention_infer.md) 样例，额外适配以下两个独立特性的推理算子：

- **MXFP8（Microscaled FP8）量化**：Q、K、V 矩阵量化为 `float8_e4m3_t` 格式，并为每个分块（block size=32）提供 `float8_e8m0_t` 格式的 MX Scale 缩放因子，在 Cube 矩阵乘的 L1→L0 阶段完成反量化。同时，Online Softmax 在 FP32 精度下完成计算后，P 矩阵必定 cast 为 `float8_e4m3_t` 以配合后续 P×V 的 MXFP8 矩阵乘。
- **P矩阵静态量化（可选）**：在 P 矩阵 cast 为 fp8 之前，额外乘以一个全局静态量化系数（pScale），实现 `P_fp8 = quantize(P_fp32 × pScale)`。由模板参数 `ENABLE_P_SCALE` 和命令行参数 `usePscale` 控制是否启用。

核心公式与样例49一致：

```text
O = FlashAttention(Q, K, V, mask)
  = softmax(Q * K^T / sqrt(d)) * V
```

---

## 与样例49的总体差异概览

下图对比展示了样例49与样例62在数据流层面的主要差异：

```text
       样例49 FP16/BF16                         样例62 MXFP8 + P 量化
       =================                        =========================

  Q(FP16/BF16)  K(FP16/BF16)           Q(fp8_e4m3)  K(fp8_e4m3)     mxScaleQ(e8m0)  mxScaleK(e8m0)
       |              |                      |              |              |              |
       v              v                      v              v              v              v
  +----------+  +----------+          +-----------+  +-----------+  +-----------+  +-----------+
  | GM → L1  |  | GM → L1  |          | GM → L1   |  | GM → L1   |  |GM→L1      |  |GM→L1      |
  +----------+  +----------+          +-----------+  +-----------+  |mxScaleA   |  |mxScaleB   |
       |              |                      |              |       +-----------+  +-----------+
       v              v                      v              v              |              |
  +----------+  +----------+          +------------------------+    +------------------------+
  | L1 → L0  |  | L1 → L0  |          |         L1→L0A         |    |         L1→L0B         |
  +----------+  +----------+          +------------------------+    +------------------------+
       |              |                      |                              |
       v              v                      v                              v
  +---------------------+             +-------------------------------------------+
  |  Cube Mmad (FP16)   |             |           Cube Mmad (mx_fp8_e4m3)         |
  +---------------------+             +-------------------------------------------+
       |                                               |
       v                                               v
  S = Q*K^T (float)                              S = Q*K^T (float)
       |                                               |
       v                                               v
  +-----------------------------+          +-----------------------------------+
  | Online Softmax              |          | Online Softmax (FP32计算)         |
  | P = cast(softmax(S))        |          | P_fp8 = cast(softmax(S) * pScale) |
  | P: FP16/BF16                |          | P: fp8_e4m3                       |
  +-----------------------------+          +-----------------------------------+
       |                                               |
       v                                               v
  P(FP16/BF16)  V(FP16/BF16)                P(fp8_e4m3)  V(fp8_e4m3)    mxScaleV(e8m0)
       |              |                          |              |              |
       v              v                          v              v              v
  +---------------------+          +-----------------------------------------------+
  |  Cube Mmad (FP16)   |          |  Cube Mmad (mx_fp8_e4m3)                      |
  +---------------------+          |  P: L1→L0A + 全1scale（P无scale输入）          |
       |                           |  V: L1→L0B + mxScaleV                         |
       v                           +-----------------------------------------------+
  O = P*V (float)                              O = P*V (float)
       |                                                 |
       v                                                 v
  +---------------------+                      +---------------------+
  | RescaleO            |                      | RescaleO            |
  +---------------------+                      +---------------------+
       |                                                 |
       v                                                 v
  Output (FP16/BF16)                             Output (FP16/BF16)
```

---

## MXFP8 量化特性

### MXFP8 数据格式

MXFP8（Microscale FP8）是一种基于分块的浮点量化方案：

- **数据格式**：矩阵元素采用 `float8_e4m3_t`，每个元素占1字节
- **缩放因子**：沿量化轴方向，每32个连续元素共享一个 `float8_e8m0_t` 格式的缩放因子（MX Scale）

```cpp
量化轴方向

  [e0][e1][e2]...[e31] | [e32][e33]...[e63] | [e64]...
  <--- block_size=32 -->  <--- block_size=32 -->
        |                        |
     scale_0                  scale_1        (float8_e8m0)
```

### 算子中的数据布局

| 输入     | 数据类型        | Shape              | 说明                             |
| -------- | --------------- | ------------------ | -------------------------------- |
| Q        | `float8_e4m3_t` | [B, N, S, D]       | Query矩阵                        |
| K        | `float8_e4m3_t` | [B, N, S, D]       | Key矩阵                          |
| V        | `float8_e4m3_t` | [B, N, S, D]       | Value矩阵                        |
| mxScaleQ | `float8_e8m0_t` | [B, N, S, D/64, 2] | Q的MX缩放因子（沿embed维度分块） |
| mxScaleK | `float8_e8m0_t` | [B, N, S, D/64, 2] | K的MX缩放因子（沿seq维度分块）   |
| mxScaleV | `float8_e8m0_t` | [B, N, S/64, D, 2] | V的MX缩放因子（沿seq维度分块）   |

> **注意**：MX Scale 的量化轴方向由矩阵乘的语义决定。Q×K^T 中，Q 沿 embed 维度分块，K 沿 embed 维度分块；P×V 中，V 沿 seq 维度分块。

### Q×K^T（BMM1）的 MXFP8 矩阵乘

Q×K^T 的 BlockMmad 从样例49的 [block_mmad_fai_qk_tla.hpp](../../../include/catlass/gemm/block/block_mmad_fai_qk_tla.hpp) 切换到 MX 版本的 [block_mmad_fai_qk_mx_tla.hpp](../../../include/catlass/gemm/block/block_mmad_fai_qk_mx_tla.hpp)，核心变化如下：

| 对比项            | 样例49                     | 样例62 (MX版本)                                |
| ----------------- | -------------------------- | ---------------------------------------------- |
| 元素类型 A/B      | FP16/BF16                  | `float8_e4m3_t`                                |
| L1新增 MX Scale A | —                          | `L1_TILE_M × L1_TILE_K / 32 × sizeof(uint8_t)` |
| L1新增 MX Scale B | —                          | `L1_TILE_N × L1_TILE_K / 32 × sizeof(uint8_t)` |
| BLOCK_L1_SIZE     | `(L1A + L1B) × STAGES`     | `(L1A + L1B + L1ScaleA + L1ScaleB) × STAGES`   |
| GM→L1搬运         | 只搬运A/B                  | A/B + mxScaleA/mxScaleB                        |
| L1→L0搬运         | `copyL1ToL0A(L0, L1_tile)` | `copyL1ToL0A(L0, L1_tile, L1_scale_tile)`      |
| DispatchPolicy    | `MmadFAIQK`                | `MmadFAIQKMx`（继承自 `MmadFAIQK`）            |

---

## P矩阵静态量化特性

### 设计动机

在 FlashAttention 推理场景中，Online Softmax 产出的中间结果 P 矩阵需要传递给后续的 P×V 矩阵乘。在样例49中，P 矩阵以 FP16/BF16 格式存储，占用 L1 带宽和存储较大。

P 矩阵 cast 到 `float8_e4m3_t` 是 MXFP8 路径下**必定执行**的步骤——Softmax 计算本身在 FP32 精度下完成，最终输出的 P 矩阵被 cast 为 fp8 以配合后续 P×V 的 MXFP8 矩阵乘，从而将 P 矩阵的 L1 占用减半。

在此基础上，模板参数 `ENABLE_P_SCALE` 和命令行参数 `usePscale` 控制的是：是否在 cast 之前额外对 P 矩阵乘一个全局静态量化系数（pScale），即 **P矩阵静态量化**。这是一个独立于 cast 的可选特性——当 `usePscale=1` 时，`P_fp8 = quantize(P_fp32 × pScale)`；当 `usePscale=0` 时，`P_fp8 = cast(P_fp32)`。

### Softmax + P量化流程

```text
Q×K^T 结果 (UB, float32)
         |
         v
  +---------------------------+
  | ComputeMaskandScale       |  ← FP32 精度
  | S = S × scaleValue + mask |
  | nowMax = max(S, lastMax)  |
  +---------------------------+
         |
         v
  +------------------------------+
  | UpdateMax                    |  ← FP32 精度
  | newMax = max(nowMax, lastMax)|
  +------------------------------+
         |
         v
  +---------------------------+        +---------------------------------+
  | 样例49 (非P量化):          |        | 样例62 (P量化):                  |
  | ComputeExpSubSum          |        | ComputeExpSubSumFp8             |
  | P_fp16 = exp(S - newMax)  |        | P_fp8 = quantize(exp(S-newMax)) |
  +---------------------------+        +---------------------------------+
         |                                      |
         v                                      v
  P(FP16/BF16) → CopyUb2L1              P(fp8_e4m3) → CopyUb2L1
```

> **概念区分**：P 矩阵 cast 到 fp8 是 MXFP8 路径下为配合后续 P×V 矩阵乘的必选步骤（由 `ElementP` 类型为 `float8_e4m3_t` 决定），而 P 矩阵静态量化（`ENABLE_P_SCALE`/`usePscale`）是在 cast 之前额外乘一个 pScale 系数的可选特性。两者是不同层级的概念：cast 决定 P 的输出精度，静态量化决定 cast 前是否做 scale 调整。

实现上，当 `ElementP` 为 `float8_e4m3_t` 类型时，[block_epilogue_fa_softmax_ascend950.hpp](../../../include/catlass/epilogue/block/block_epilogue_fa_softmax_ascend950.hpp) 中的 Softmax 通过 `if constexpr` 选择 `ComputeExpSubSumFp8` 路径，直接在 UB 上将 exp 结果量化为 fp8 后输出到 L1：

```cpp
if constexpr (AscendC::IsSameType<ElementP, float8_e4m3_t>::value) {
    ComputeExpSubSumFp8<ElementP, ElementS, ...>(outputAddr, inputAddr, ...);
} else {
    ComputeExpSubSum<ElementP, ElementS, ...>(outputAddr, inputAddr, ...);
}
```

---

## P×V（BMM2）的 MXFP8 矩阵乘及全1 Scale 初始化

### P 矩阵的特殊性

P×V 的 BlockMmad 从样例49的 [block_mmad_fai_pv_tla.hpp](../../../include/catlass/gemm/block/block_mmad_fai_pv_tla.hpp) 切换到 MX 版本的 [block_mmad_fai_pv_mx_tla.hpp](../../../include/catlass/gemm/block/block_mmad_fai_pv_mx_tla.hpp)。

与 Q×K^T 不同，P×V 矩阵乘中：

- **矩阵B（V）**：V(fp8) 和其 MX Scale（mxScaleV）从 GM 搬运到 L1，L1→L0 时一并搬运到 L0，Cube Mmad 中按 MXFP8 格式完成计算，输出 fp32。
- **矩阵A（P）**：P 矩阵由 AIV 端的 Softmax 分 tile 写入 L1，**不来自 GM**，没有 MX Scale 输入。但 MXFP8 矩阵乘要求 A 侧必须传入 scale 参与计算，因此在 BlockMmadPV 构造时，L1 上预先初始化一份**全1的 MX Scale**（L1 常驻）作为 P 的缩放因子。

### 全1 Scale 初始化

在 [block_mmad_fai_pv_mx_tla.hpp](../../../include/catlass/gemm/block/block_mmad_fai_pv_mx_tla.hpp) 的构造函数中调用 `InitOneInL1MxScaleA()`：

```cpp
CATLASS_DEVICE
void InitOneInL1MxScaleA()
{
    AscendC::InitConstValueParams<int16_t> initConstValueParams;
    initConstValueParams.repeatTimes = 1;
    initConstValueParams.blockNum = L1SCALEA_TILE_SIZE / BYTE_PER_BLK;
    initConstValueParams.initValue = 0x7f7f;    // 全1，float8_e8m0 中表示 scale = 1.0
    initConstValueParams.dstGap = 0;
    auto l1MxScaleAInt16 = l1MxScaleATensor_.template ReinterpretCast<int16_t>();
    InitConstValue(l1MxScaleAInt16, initConstValueParams);
}
```

`0x7f7f` 在 `float8_e8m0_t` 格式中表示数值 `1.0`。Cube Mmad 计算时 `P(fp8) × 1.0` 等价于 P 保持原始 fp8 数值参与 MXFP8 矩阵乘。

该全1 Scale 在 L1 上只分配**一份**（L1 常驻，不参与多级流水），因为 P 矩阵的缩放因子固定为 1，不随 KV 循环更新。

### P×V BlockMmad 数据流

```text
    A侧：P矩阵（来自AIV）                B侧：V矩阵（来自GM）
    =======================               =======================

  L1: l1MxScaleATensor_                  GM
  全1 scale (e8m0, 常驻)                  |
  (构造函数初始化)                         +----------+----------+
                                          |                     |
  AIV → L1 (每次传入一个tile)              v                     v
  P(fp8_e4m3_t)                     V(fp8_e4m3_t)         mxScaleV(e8m0)
        |                                 |                     |
        |                                 v                     v
        |                          +-----------+         +-----------+
        |                          |CopyGmToL1B|         |CopyGmToL1 |
        |                          +-----------+         | MxScaleB  |
        |                                 |              +-----------+
        |                                 v                     |
        |                          l1BTensorList_[S]            v
        |                           (STAGES)             l1MxScaleB
        |                                 |            TensorList_[S]
        |                                 |              (STAGES)
        |                                 +---------+---------+
        |                                           |
        |                                           v
        |                                   +---------------+
        |                                   | CopyL1ToL0B   |
        |                                   | V(fp8) +      |
        |                                   | mxScaleV(e8m0)|
        |                                   +---------------+
        |                                           |
        v                                           v
  +-----------------+                        +--------------+
  | CopyL1ToL0A     |                        | L0B          |
  | P(fp8) +        |                        | (mx_fp8_e4m3)|
  | 全1scale(e8m0)  |                        +--------------+
  +-----------------+                              |
        |                                          |
        v                                          |
  +--------------+                                 |
  | L0A          |                                 |
  | (mx_fp8_e4m3)|                                 |
  +--------------+                                 |
        |                                          |
        +-------------------+----------------------+
                            |
                            v
                     +--------------+
                     |  Cube Mmad   |
                     |  (MXFP8)     |
                     +--------------+
                            |
                            v
                     +-----------+
                     | L0C(fp32) |
                     +-----------+
```

| 对比项         | 样例49                           | 样例62 (MX版本)                          |
| -------------- | -------------------------------- | ---------------------------------------- |
| 头文件         | `block_mmad_fai_pv_tla.hpp`      | `block_mmad_fai_pv_mx_tla.hpp`           |
| P的来源        | L1（AIV Softmax写入，FP16/BF16） | L1（AIV Softmax写入，fp8_e4m3）          |
| P的scale       | —                                | L1常驻全1 scale（`InitOneInL1MxScaleA`） |
| V的来源        | GM→L1，FP16/BF16                 | GM→L1，fp8_e4m3 + mxScaleV               |
| DispatchPolicy | `MmadFAIPV`                      | `MmadFAIPVMx`（继承自 `MmadFAIPV`）      |

---
