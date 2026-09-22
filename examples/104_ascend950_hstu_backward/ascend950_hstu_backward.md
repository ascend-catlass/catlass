# HSTU Backward 算子设计方案

## 一、设计概述

HSTU backward 面向 Ascend 950，采用 packed TND 布局，支持 `fp16`、`bf16`、可选 RAB，以及 no-mask 和 bottom-right causal 两种 MASK 模式。整体执行收敛为一次 `MIX_AIC_1_2` Kernel launch，由 AIC 完成矩阵计算，AIV 完成 Score、MASK、激活梯度及结果整理。

设计的核心思路是以 `(batch, head, K Tile)` 为任务单元，按 K Tile 外层、Q Tile 内层组织计算。K/V Tile 跨多个 Q Tile 常驻 L1，dK/dV 在 L0C 中持续累加；各 K Tile 产生的 dQ 局部贡献则直接以 fp32 AtomicAdd 累加到单份 `q_share`，最后在同一 Kernel 尾部转换并写回 `qGrad`，不再使用 `partial_dQ` 和独立的 `dQ_reduce kernel`。

## 二、整体架构

方案沿用 CATLASS 的分层结构：

| 层级 | 主要职责 |
|---|---|
| `Gemm::Device` | 参数校验、profile 选择、workspace 规划、tiling 和单次 Kernel 启动 |
| `Gemm::Kernel` | Scheduler、K-outer/Q-inner 循环、AIC/AIV 协同和尾部 finalize |
| `Gemm::Block` | QK、GV、dV、dK、dQ 五个 MMAD |
| `Epilogue::Block` | RAB、MASK、SiLU、dP、dRAB 和 `qGrad` finalize |

所有 Block 共享同一个 `Arch::Resource` 及其 L1/L0A/L0B/L0C 物理资源，不重复分配片上存储。

## 三、计算流程

对每个 batch 和 head，主要计算关系如下：

```text
P    = Q @ K^T
X    = alpha * (P + RAB)              # 无 RAB 时省略 RAB
A    = silu_scale * MASK * SiLU(X)
GV   = dO @ V^T
dP   = alpha * silu_scale * MASK * SiLU'(X) * GV
dQ   = dP @ K
dK   = dP^T @ Q
dV   = A^T @ dO
dRAB = dP
```

单个 K Tile 的执行过程为：

```text
1. K/V 从 GM 搬入 L1，并在当前 K Tile 的 Q 循环中保持常驻
2. Q/dO 按双 stage 搬入 L1
3. AIC 执行 QK、GV，并将结果分发给两个 AIV
4. AIV 完成 RAB、MASK、SiLU、dP 和 dRAB 计算
5. AIV 将 activation/dP 写入 L1，并通知 AIC
6. AIC 执行 dV、dK、dQ
7. dV/dK 在 L0C 中持续累加，dQ 直接 AtomicAdd 到 q_share
8. 当前 K Tile 完成后，将 dK/dV 有效区域写回 GM
```

## 四、任务调度与 Tile 配置

Scheduler 的最小任务单元为：

```text
(batch_id, head_id, k_tile_id)
```

当 `batch==1` 时，各 AIC 按连续区间静态分配任务；多 batch 场景按有效 Q Tile 数量估算工作量，并采用连续 `PersistentSplit`。Scheduler 仅在 batch 变化时更新并缓存 offsets、序列长度和 Tile 数量，保持原有 work 顺序、分段方式及 AIC/AIV 映射不变。

当前 Tile profile 为：

| profile | 条件 | Mq | Nk | Ma | BLOCK_K |
|---|---|---:|---:|---:|---:|
| D128 | `max(Dqk,Dv) <= 128` | 128 | 128 | 64 | 64 |
| D256 | `max(Dqk,Dv) <= 256` | 128 | 64 | 64 | 256 |

D256 内部按固定 M64 分两段完成前端 MMAD，并跨两段复用 K/V 的 L0B 数据。no-mask 路径按 K Tile 奇偶反转 Q 遍历方向，在边界一致时复用相邻 K Tile 的 Q/dO L1 数据；causal 路径保持正向遍历。

## 五、片上资源与流水

- L1 固定划分 K/V、Q/dO、activation 和 dP 区域；Q/dO 与 activation/dP 均采用两级 stage。
- L0A/L0B 使用 ping-pong，L0C 划分为 `DV_ACC`、`DK_ACC`、`TMP0` 和 `TMP1`。
- dV/dK 在一个 K Tile 的有效 Q Tile 循环中保持 L0C resident，首 Tile 初始化、末 Tile 写回。
- 每个 AIV 使用两份 Score union arena，并配置两份独立的 RAB input buffer。

AIC 与 AIV 通过双 stage 流水协同：AIC 发布 `FRONT_READY` 后继续准备下一 Tile；AIV 完成 Score 计算并将 activation/dP 写入 L1 后发布 `L1_READY`；AIC 消费对应 stage 后再释放资源。

## 六、RAB 与 MASK

RAB 为可选输入，布局为 `[B,H,maxLq,maxLk]`。AIV 按当前 Q/K 有效矩形直接从 GM 搬入 UB，在寄存器融合过程中完成 RAB add、alpha、sigmoid、activation、导数和输出计算。`dRAB=dP` 的每个元素只有一个生产者，因此可直接写回 GM，无需 AtomicAdd 或额外归约。

当前支持两种 MASK 策略：

```text
window_size_left=-1, window_size_right=-1
    -> no-mask

window_size_left=-1, window_size_right=0
    -> bottom-right causal
```

bottom-right causal 条件为：

```text
k_local <= q_local + (Lk - Lq)
```

Tile 分为 `AllInvalid`、`AllValid` 和 `Boundary`。`AllInvalid` 跳过无效计算，`AllValid` 不生成 MASK，`Boundary` 在寄存器融合计算后应用 causal mask。MASK 不从 GM 读取，也不跨 Tile 保存。

## 七、dQ 归约与 Kernel 尾部处理

`q_share` 是 Device 内部使用的 fp32 workspace：

```text
logical shape = [totalQ,H,Dqk]
internal view = [H,totalQ,Dqk]
dtype         = fp32
```

每个 `(Q_tile,K_tile)` 的 dQ 局部贡献在 AIC 的 fp32 L0C 中产生，并直接累加到 `q_share`：

```text
dQ_contribution = dP @ K
L0C -> GM q_share, fp32 AtomicAdd
```

所有 AIC 完成任务后设置 `Q_TRANS_READY_FLAG`。AIV 等待配对 AIC 完成，再通过全体 `SyncAll()` 保证所有 AtomicAdd 对 `q_share` 可见，随后按 packed Q token 范围分工，将 head-major fp32 `q_share` 转换为 `fp16`/`bf16`，重排为 TND 布局并写回 `qGrad`。
