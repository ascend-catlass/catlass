---
nav_order: 45
---

# Kernel 入口 SoC 状态初始化

[核心概念](index.md) | [Kernel 入口](../../api/runtime/kernel_entry.md)

## 概述

每个 device kernel 入口会**自动**完成 SoC / 控制状态初始化，用于消除前序算子可能留下的脏状态。用户无需、也无法单独调用初始化 API；但入口会改写下列状态，开发者编写 kernel 时应按本文理解入口后的默认硬件约定。

整体顺序：

```text
1) SoC 状态复位（原子模式、Mask、反量化 scale、CTRL、padding / 向量 mask 等）
2) 用户 kernel body
3) 出口：恢复 mmad 相关 CTRL，并保证尾部流水同步
```

适用架构：Ascend 950。

---

## AIC 入口初始化

| 顺序 | 初始化动作 | 修改的状态 / 效果 |
| --- | --- | --- |
| 1 | 关闭随路原子模式 | 原子模式关闭时，GM 写入覆盖原值 |
| 2 | Mask 模式归一 | Mask 工作模式设为 normal |
| 3 | 反量化默认 scale | 默认 scale 置为 1（half）；影响依赖该默认值的反量化 / 相关 cast 路径 |
| 4 | bulk CTRL 复位 | 整寄存器赋值 `0x1000000000000008 \| (旧CTRL & bit48)`：**CTRL[60]=1**，**CTRL[48]=旧值（0 或 1）**，**CTRL[3]=1**，其余位为 0 |
| 5 | padding 清零 | `padding = 0`；避免脏 padding 影响 LoadData |
| 6 | store-atomic 配置复位 | `st_atomic_cfg = 0b00100100` |

AIC **不**修改：向量 mask、UB↔OUT loop-mode 参数。

---

## AIV 入口初始化

| 顺序 | 初始化动作 | 修改的状态 / 效果 |
| --- | --- | --- |
| 1 | 关闭随路原子模式 | 原子模式关闭时，UB→GM 写入覆盖原值 |
| 2 | Mask 模式归一 | Mask 工作模式设为 normal |
| 3 | 反量化默认 scale | 默认 scale 置为 1（half） |
| 4 | bulk CTRL 复位 | 同 AIC：CTRL[60]=1；CTRL[48] 保留旧值；其余位按常量重写 |
| 5 | 向量 mask 全开 | `vector_mask = (-1, -1)`；避免部分 mask 残留导致漏元素 |
| 6 | UB↔OUT loop 参数复位 | `loop_size_ubtoout` / `loop_size_outtoub = (1<<21)\|1` |
| 7 | store-atomic 配置复位 | `st_atomic_cfg = 0b00100100` |

AIV **不**修改：padding（仅 AIC）。

---

## 与 DSL 接口相关的隐式 CTRL 设置

编译器根据 `tla.mmad` 和 `tla.mmad_mx` 的参数配置相应的 CTRL 位。函数内取值一致的设置可在入口初始化之后写入一次。取值不一致时，在各条 MMAD 前写入。函数存在显式 CTRL 访问或可能修改 CTRL 的调用时，各条 MMAD 分别设置所需的位。

| DSL 接口 | 隐式 CTRL 效果 |
| --- | --- |
| `tla.mmad` / `tla.mmad_mx` `compute_order=M_FIRST` | 清 CTRL[51] |
| `tla.mmad` / `tla.mmad_mx` `compute_order=N_FIRST` | 置 CTRL[51]=1 |
| `tla.mmad` `hf32_mode=HF32_DISABLE` | 清 CTRL[46] 和 CTRL[47] |
| `tla.mmad` `hf32_mode=HF32_NEAREST_ZERO` | 置 CTRL[46]=1、CTRL[47]=1 |
| `tla.mmad` `hf32_mode=HF32_NEAREST_EVEN` | 置 CTRL[46]=1，清 CTRL[47] |

`tla.cast` 保留当前饱和策略（CTRL[60]）。要使用指令的饱和模式，请通过 `tla.arch.set_ctrl` 显式选择 `USE_API`。使用全局饱和策略时，CTRL[48] 保留入口初始化前的值，直到显式写入。

在显式访问 CTRL 的函数中，两个输入均为 E4M3 的 `tla.mmad` 选择 E4M3 输入格式（CTRL[45]）。

使用默认 `atomic_mode=NONE` 的 GM 拷贝沿用当前原子状态。通过 `tla.arch.set_ctrl` 开启原子写入后，若要覆盖 GM 原值，须先设置 `atomic_format=AtomicFormat.DISABLED`。

---

## 出口（AIC / AIV 相同）

| 动作 | 修改的状态 / 效果 |
| --- | --- |
| 清 CTRL[51] | 恢复 mmad M/N 计算方向优先级为硬件默认 |
| 清 CTRL[46] | 关闭 mmad HF32 rounding-mode enable |
| 清 CTRL[47] | 关闭 mmad HF32 rounding-mode select |
| 尾部 `PIPE_ALL` barrier | 保证出口流水同步 |
