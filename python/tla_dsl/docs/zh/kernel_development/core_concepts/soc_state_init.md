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
| 1 | 关闭随路原子模式 | 原子模式设为 none；后续未声明原子语义的 GM 写按覆盖执行，不会累加 |
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
| 1 | 关闭随路原子模式 | 原子模式设为 none；普通 UB→GM 写为覆盖而非累加 |
| 2 | Mask 模式归一 | Mask 工作模式设为 normal |
| 3 | 反量化默认 scale | 默认 scale 置为 1（half） |
| 4 | bulk CTRL 复位 | 同 AIC：CTRL[60]=1；CTRL[48] 保留旧值；其余位按常量重写 |
| 5 | 向量 mask 全开 | `vector_mask = (-1, -1)`；避免部分 mask 残留导致漏元素 |
| 6 | UB↔OUT loop 参数复位 | `loop_size_ubtoout` / `loop_size_outtoub = (1<<21)\|1` |
| 7 | store-atomic 配置复位 | `st_atomic_cfg = 0b00100100` |

AIV **不**修改：padding（仅 AIC）。

---

## 与 DSL 接口相关的隐式 CTRL 设置

入口 InitSoc 提供默认 CTRL。下列接口在语义依赖 CTRL 时，由编译器**自动**调整对应位；用户无需、也无法手写 CTRL 接口。若入口默认已满足该次调用的需求，则不会额外改写。同一段内连续相同需求、且中间不需要相反状态时，只会生效一次。

| DSL 接口 | 隐式 CTRL 效果 |
| --- | --- |
| `tla.cast` / `CastParams.sat_mode`（SAT / NOSAT / UNKNOWN） | 清 CTRL[60]，使指令上的饱和操作数生效（入口默认 CTRL[60]=1 为全局覆盖） |
| `tla.mmad` / `tla.mmad_mx` `compute_order=M_FIRST` | 需要 CTRL[51]=0（与入口默认一致，通常无需改写） |
| `tla.mmad` / `tla.mmad_mx` `compute_order=N_FIRST` | 置 CTRL[51]=1 |
| `tla.mmad` `hf32_mode=HF32_DISABLE` | 需要 CTRL[46]=0、CTRL[47]=0（与入口默认一致，通常无需改写） |
| `tla.mmad` `hf32_mode=HF32_NEAREST_ZERO` | 置 CTRL[46]=1、CTRL[47]=1 |
| `tla.mmad` `hf32_mode=HF32_NEAREST_EVEN` | 置 CTRL[46]=1；CTRL[47] 保持 0 |

与上述接口无关、已由入口 InitSoc 覆盖的位：

- **CTRL[48]**：bulk 中保留旧值；cast 在 CTRL[60]=0 后走指令饱和，不再依赖它
- **CTRL[56]**：由 InitSoc 的 mask_norm / vector_mask 取代，无对应 DSL 参数

---

## 出口（AIC / AIV 相同）

| 动作 | 修改的状态 / 效果 |
| --- | --- |
| 清 CTRL[51] | 恢复 mmad M/N 计算方向优先级为硬件默认 |
| 清 CTRL[46] | 关闭 mmad HF32 rounding-mode enable |
| 清 CTRL[47] | 关闭 mmad HF32 rounding-mode select |
| 尾部 `PIPE_ALL` barrier | 保证出口流水同步 |
