# FMHA Backward 算子设计方案

## 一、设计概述

FMHA Backward面向 Ascend 950，采用 packed TND 布局，支持 `fp16`、`bf16`、headSize 64/128/192/256、qHeads:kvHeads = N:1，以及 no-mask 和 causal 两种 MASK 模式。使用场景为大模型训练/微调的反向传播，与 FMHA 前向配对（输入含前向给出的 LSE）。

整体执行分两次 MIX_AIC_1_2 Kernel launch：主 kernel 由 AIC 完成矩阵计算、AIV 完成向量段（D/P/dS），把 per-qhead 的 dQ/dK/dV 部分和以 fp32 AtomicAdd 累加进 GM workspace；独立 Post kernel 读 workspace 完成 GQA 归约、scale、cast 并写回 GM，先后关系由 host launch 顺序天然保证。

调度以 `(batch, qHead, S2o)` 外层块为任务单元（KV 外 Q 内），1 个 AIC + 2 个 AIV 配对一个核组，核组内按块完成 Cube → Vector → Cube 流水。

## 二、整体架构

| 层级 | 主要职责 |
|---|---|
| host（fab.cpp） | 参数校验、workspace/tiling 计算、两次 Kernel launch、host fp64 参考比对 |
| Tiling（fab_tiling.h） | BN2GS1S2 任务切分、工作量加权贪心分核 |
| Kernel（fab_kernel.h） | AIC 块循环、AIV 块循环、跨核 flag 同步；FA950Post6A 独立 kernel |
| Block | mm1/mm2 复用 `BlockMmadTla<MmadFAIQK / MmadFAIPV>`；mm3/4/5 手工组装 `BlockMmadFAGradDQ / BlockMmadFAGradKV`（输出需原子累加、dK/dV 转置 A 侧，不继承通用模板，复用其 TileCopy 子组件） |
| Tile | CopyGmToL1 / CopyL1ToL0A/B / tileMmad / CopyL0CToUB（SPLIT_M）/ CopyL0CToGm 复用组件 |

所有 Block 共享同一 `Arch::Resource` 的 L1/L0A/L0B/L0C 物理资源。




## 三、原型设计

### 输入输出张量

| 张量 | 布局 | Shape | 语义 |
|---|---|---|---|
| Q / K / V | TND | [totalQ/Kv, heads, D] | 注意力三输入 |
| O / dO | TND RowMajor | [totalQ, qHeads, D] | 前向输出 / 输出梯度 |
| LSE | RowMajor | [totalQ, qHeads] | 前向全局 log-sum-exp，在线重算 P |
| scale | 标量 | — | QK 缩放（默认 1/√D） |
| dQ/dK/dV | TND RowMajor | [totalQ/Kv, heads, D] | 输出梯度（fp16/bf16） |

边界寻址：`cu_seqlens_q / cu_seqlens_kv` 前缀和——第 b 个序列 token 行区间 = [cu_seqlens[b], cu_seqlens[b+1])；


约束：dtype ∈ {half, bf16}；D ∈ {64,128,192,256}；qHeads/kvHeads ∈ {1/1, 8/1, 16/1}；mask ∈ {无, causal}；


## 四、计算流程

### 1、数学定义

```text
dp = dO @ V^T                  (mm1)
S  = Q @ K^T                   (mm2)
D  = rowsum(dO ⊙ O)            (v1)
P  = exp(S*scale + mask − LSE) (v2)
dS = P ⊙ (dp − D)              (v3)
dQ = dS @ K                    (mm3)
dK = dS^T @ Q                  (mm4)
dV = P^T @ dO                  (mm5)
dQ,dK *= scale；GQA 归约        (Post)
```

- LSE 前向给出 → P 在线重算：不存 P（省一次 128×128 中间量的存储与搬运）；
- scale 分两处乘：V2 提前乘入 S，Post 再对 dQ/dK 补乘、dV 不乘——与梯度公式的 scale 归属对齐；

### 2、单块执行过程

```text
1. AIC 执行 mm2(S) / mm1(dp)，L0C 经 FixPipe SPLIT_M 按行拆半双写配对 AIV UB，发 C2/C1
2. AIV 收 C2/C1 后执行 v1(D) → v2(P) → v3(dS)，dS/P Cast fp16 写 L1，发 C3/C5
3. AIC 等 C3/C5 后执行 mm3(dQ) / mm4(dK) / mm5(dV)
4. mm3/4/5 输出 SetAtomicAdd 累加进 workspace
```

## 五、任务调度与 Tile 配置

```text
for 每个 batch b:
  for 每个 qHead n:
    for 每个 KV 块 s2o:            # 外层扁平 f=(b,n,s2o)，核区间按 f 切分
      for 每个 Q 块 s1o (>= s2o):  # 内层；causal 下 s1o < s2o 跳过
```

- 负载均衡：每个 (b,n,s2o) 权重 = 有效 S1o 块数（causal 三角），贪心切分 `target = totalValid / coreNum`；


## 六、MASK 与 GQA

causal 三角由 device 块内生成，不读 GM、不跨块保存。块分类：`AllValid` 整块有效不生成 mask；Boundary 只写无效列；AllValid 整块零开销不生成 mask；AllInvalid 跳过整块。

GQA：dQ 按 qHead 独立写 workspace；dK/dV 同 kvHead 的多 qHead 部分和由 Post 按 headSize 分组归约。

## 七、归约与 Post

主 kernel 中 mm3/4/5 的 L0C 结果以 FixPipe 原子模式（`SetAtomicAdd<float>`）直接累加到 workspace 对应区。

主 kernel 完成后 host launch Post kernel：读 fp32 workspace → 按 kvHead 归约 GQA 各 qHead 部分和 → dQ/dK 乘 scale → cast fp16/bf16 → 按原始 cu_seqlens 写 unpadded GM 布局。后续优化过程考虑使用fixpipe随路量化，避免额外的 cast 操作。



