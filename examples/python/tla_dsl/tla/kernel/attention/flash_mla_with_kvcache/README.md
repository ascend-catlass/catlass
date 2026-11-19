# FlashMlaWithKvcache Example Readme

## 功能说明

- 算子功能：基于CATLASS Python的 Flash MLA（Multi-head Latent Attention）解码/短序列预填充注意力，KV cache 以单个 576 维 latent 头共享（`KV_HEAD_NUM=1`），支持两类 paged KV cache 物理布局、因果与非因果掩码、变长（TND）输入与批量解码。
- 计算公式：

  $$
  \text{MLA}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{qk}}}\right) V
  $$

  其中 $Q \in \mathbb{R}^{S_q \times H \times d_{qk}}$，KV cache 只存一份 latent 头（$d_{qk} = 576$，$d_v = 512$）：$K$ 使用 latent 的全部 576 维，$V$ 使用其前 512 维（二者是同一段页内数据的不同视图，KV 头共享即 MLA 的压缩收益来源）。分块后单块分数矩阵（FP32）：

  $$
  S_{b} = \text{scale} \cdot (Q K_b^{T}) \in \mathbb{R}^{B_r \times B_c}
  $$

  $\text{scale} = 1/\sqrt{d_{qk}}$，$B_r$ 为 Q 基块（相邻两个 token 合算时取 128，单 token 时等于 Q 头数 64 或 96），$B_c$ 为 KV 基块大小（`KV_BLOCK` = 128）。KV cache 按 tile 流式读入，Online Softmax 增量更新：

  $$
  \begin{aligned}
    m_{\text{new}} &= \max(m,\ \text{rowmax}(S_{b})) \\
    P_{b} &= e^{S_{b} - m_{\text{new}}} \\
    O &= O \cdot e^{m - m_{\text{new}}} + P_{b} \cdot V_{b} \\
    l &= l \cdot e^{m - m_{\text{new}}} + \text{rowsum}(P_{b}) \\
    m &= m_{\text{new}}
  \end{aligned}
  $$

  循环结束后归一化 $O_{\text{final}} = O / l$，转回 FP16/BF16 写回 GM；同时输出 $lse = m + \log l$（FP32），供上层做多段归并。

- 支持产品型号：Ascend 950PR&950DT系列产品。

### Mask 模式

`--mask` 选择两种掩码语义，共用同一个 kernel（编译期常量特化）：

- `0`（全可见）：每行可见全部 `cache_seqlens` 个 KV；
- `3`（right-down causal）：按 `cache_seqlens` 右对齐的因果掩码，要求 `kv_len >= q_len`，第 $i$ 行可见前 $kv\_len - q\_len + i + 1$ 个 KV。

```plain
       K V  →                     K V  →
      0 1 2 3 4                  0 1 2 3 4 5 6
Q  0  ■ ■ ■ ■ ■             Q  0  ■ ■ ■ · · · ·
   1  ■ ■ ■ ■ ■                1  ■ ■ ■ ■ · · ·
↓  2  ■ ■ ■ ■ ■             ↓  2  ■ ■ ■ ■ ■ · ·
   3  ■ ■ ■ ■ ■                3  ■ ■ ■ ■ ■ ■ ·
   （mask=0）                   （mask=3，kv_len > q_len）
```

### PagedAttention

KV cache 一律按 128 token 一页组织，host 生成块表、kernel 侧每 tile 查表寻址，支持两种页内排布：

- `PA_BBND`：页内自然序 `(num_blocks, 128, 1, 576)`，即 `[token][d]`；
- `PA_NZ`：页内分形排布 `(num_blocks, 1, 36, 128, 16)`，即 `[36 个 16 维组][128 token][16 维]`，与 NPU 的 nZ 格式对齐，搬运可整页扁平拷贝。

两种排布的计算语义一致，仅影响 kernel 的 K/V 搬运路径。

### 变长输入

`cu_seqlens_q` 给出逐请求的 Q 序列前缀和，`cache_seqlens` 给出逐请求的 KV 长度，因此一个 batch 内各请求可以长短不一（含 `q_len` 与 `kv_len` 不同）。退化请求（`q_len <= 0` 或 `kv_len <= 0`）在 host 侧被跳过语义处理：对应输出保持 0，`lse` 保持 `-inf`。

### 实现要点

- **单核内全流水**：Q 常驻 L1，K/V 按 tile 流入；QK 计算 $Q_iK_{t+1}^{T}$ 与 PV 计算 $P_tV_t$ 在 Cube Core 上交替，fixpipe 搬运 $S_{t+1}$ 与 PV 并行，Softmax 在 AI Core 的 Vector 部分上与 Cube 并行。
- **Q 基块选择**：`heads = 64` 且 `total_q` 不小于 `8 × AI Core 数` 时，把相邻两个 token 合成一个基块（$B_r = 128$）；其余情况单 token 一个基块（$B_r$ 等于 Q 头数，即 64 或 96）。
- **头拆分与 KV-split 归并**：不满足上述配对条件时，`total_q <= 4` 先把 Q 头按 16 头的粒度拆到多核（head-split）；仍在空核、且单请求 tile 数不小于所需倍率时，再把 KV 按 tile 均分到多核（split 数取 2、4、8）。主 kernel 输出 FP32 partial 与 partial `lse`，再由归并 kernel 按 $\text{out} = \sum_c o_c e^{l_c - l_{\max}} / \sum_c e^{l_c - l_{\max}}$、$lse = l_{\max} + \log \sum_c e^{l_c - l_{\max}}$ 合并（两个 `@tla.kernel` 在同一进程内顺序启动）。
- **host 侧缓存**：按执行器键（dtype、设备、布局、头数、Q 基块、head-split/ksplit 参数）缓存已编译 kernel，并提供 ACLGraph 捕获入口 `capture_graph()`。

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | Input | TND `(ΣS_q, H, 576)` | `float16/bfloat16` | Q 序列，`H` 取 64 或 96 |
| `k_cache` | Input | `PA_BBND`：`(num_blocks, 128, 1, 576)`；`PA_NZ`：`(num_blocks, 1, 36, 128, 16)` | `float16/bfloat16` | 单头 KV latent cache（K 用全部 576 维、V 用前 512 维），按块表物理页寻址 |
| `block_table` | Input | `(B, num_pages_per_req)` | `int32` | 每个请求的物理页号，按 `cache_seqlens` 截取有效页 |
| `cu_seqlens_q` | Input | `(B + 1,)` | `int32` | Q 序列前缀和，末元素等于 `ΣS_q` |
| `cache_seqlens` | Input | `(B,)` | `int32` | 每个请求的 KV 长度 |
| `softmax_scale` | Input | 标量 | `float32` | softmax 缩放，常规取 $1/\sqrt{576}$ |
| `mask_mode` | Input | 标量 | `int32` | `0` 全可见，`3` right-down causal |
| `cache_layout` | Input | 标量 | `str` | `"PA_BBND"` 或 `"PA_NZ"` |
| `attentionOut` | Output | NTD `(H, ΣS_q, 512)` | 同输入 | 注意力输出，`lse` 之外的唯一输出 |
| `lse` | Output | `(H, ΣS_q)` | `float32` | 每行 `max + log(sum)`，无效行为 `-inf` |

shape 中各简写含义：

| 简写 | 含义 | 对应命令行参数 |
| --- | --- | --- |
| `B` | batch 数 | `--batch` |
| `S_q` | 单个请求的 Q 序列长度 | `--total-q` |
| `kv_len` | 单个请求的 KV 长度 | `--kv-len` |
| `ΣS_q` | 各 batch Q 序列长度之和（TND 打包） | — |
| `H` | Q 头数（KV 头数恒为 1，故天然 MQA） | `--heads` |
| `d_qk` | QK 维度，固定 576 | — |
| `d_v` | V 维度，固定 512 | — |
| `num_blocks` | KV cache 物理页总数（= `B * ceil(kv_len / 128)`） | — |
| `num_pages_per_req` | 块表列数，即每请求可用页数上限 | — |

中间计算（S/PV/acc、`lse`）恒为 FP32。

## 使用范围说明

- 输入输出数据类型支持如下组合，中间计算恒为 FP32：

| 输入 (Q/KV/O) | 中间计算 (S/PV/acc/lse) |
|:-:|:-:|
| fp16 | f32 |
| bf16 | f32 |

- `q` 的最后一维固定 576，`attentionOut` 的最后一维固定 512，`lse` 恒为 FP32。
- `H`（Q 头数）只支持 64 与 96 两档；KV 头数固定 1。
- `mask_mode` 只支持 `0` 与 `3`；`3` 要求每个请求满足 `kv_len >= q_len`。
- PagedAttention 强制开启，页大小固定 128 token；要求 `cache_seqlens` 的最大值不超过 `block_table.shape[1] * 128`。
- `cu_seqlens_q` 必须是前缀和且末元素等于 Q 的总行数（`q.shape[0]`）。
- `q_len <= 0` 或 `kv_len <= 0` 的请求输出为 0、`lse` 为 `-inf`；校验此类请求需放宽 `allow_negative_inf`。

## 使用示例

### 命令行参数

```text
flash_mla_with_kvcache.py [-h] [--device DEVICE] [--batch BATCH]
                          [--total-q TOTAL_Q] [--kv-len KV_LEN]
                          [--heads {64,96}] [--dtype {float16,bfloat16}]
                          [--layout {PA_BBND,PA_NZ}] [--mask {0,3}]
                          [--scale SCALE] [--seed SEED] [--golden-device {npu,cpu}]
                          [--atol ATOL] [--cases]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--batch` | `1` | batch 数，`--batch` 不得超过 `--total-q`。 |
| `--total-q` | `512` | 单个请求的 Q 序列长度。 |
| `--kv-len` | `1024` | 单个请求的 KV 长度。 |
| `--heads` | `64` | Q 头数，可选 64 或 96。 |
| `--dtype` | `"float16"` | 输入/输出数据类型，中间计算恒为 FP32；该值触发重新编译。 |
| `--layout` | `"PA_BBND"` | KV cache 页内排布。 |
| `--mask` | `0` | 掩码模式，`0` 全可见、`3` right-down causal。 |
| `--scale` | `0.0` | softmax 缩放，`0` 表示取 $1/\sqrt{576}$。 |
| `--seed` | `2718` | 输入与块表的随机种子，可复现。 |
| `--golden-device` | `"npu"` | golden 运行位置，`cpu` 为完全独立的慢速交叉验证。 |
| `--atol` | `0.0` | 绝对误差阈值，`0` 表示 fp16 取 `0.0025`、bf16 取 `0.01`。 |
| `--cases` | 关闭 | 跑内置的 8 条边界用例（覆盖两种布局、两种掩码、两档头数、两档批量与 bf16），否则只跑命令行指定的单例。 |
| `--perf` | 关闭 | 精度校验之外再测端到端时延，输出 `latency_us_mean/min/max`、`bytes_moved`、`effective_gbps`、`tokens_per_s`。 |
| `--perf-only` | 关闭 | 只测端到端时延、跳过 golden 校验；此时 `passed` 为 `null`。 |
| `--perf-iters` | `20` | 计时阶段连续下发的次数。 |
| `--perf-warmup` | `5` | 计时前的预热次数（含首次编译与缓存填充）。 |

### 执行示例

1. 部署 CATLASS Python 环境后，进入 `python/tla_dsl` 目录（可参考[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)）。

2. 执行算子样例程序：

    ```bash
    # 默认：b1 t512 kv1024 h64 fp16 PA_BBND mask=0（device 0）
    python examples/python/tla_dsl/tla/kernel/attention/flash_mla_with_kvcache/flash_mla_with_kvcache.py

    # 内置边界用例全套（两种布局 × 两种掩码 × 两档头数 × fp16/bf16）
    python examples/python/tla_dsl/tla/kernel/attention/flash_mla_with_kvcache/flash_mla_with_kvcache.py --cases

    # 指定形状与布局：h96 + PA_NZ + causal，kv 留 383 的尾页
    python examples/python/tla_dsl/tla/kernel/attention/flash_mla_with_kvcache/flash_mla_with_kvcache.py \
      --device 1 --batch 2 --total-q 130 --kv-len 383 --heads 96 --dtype bfloat16 \
      --layout PA_NZ --mask 3
    ```

3. 执行结果如下，说明样例执行成功，精度通过（`passed=True`）：

    ```text
    {"name": "B2_T130_KV383_H96_bfloat16_PA_NZ_M3", "batch": 2, "total_q": 130, "kv_len": 383,
     "heads": 96, "dtype": "bfloat16", "cache_layout": "PA_NZ", "mask_mode": 3, "atol": 0.01,
     "matched_ratio": 1.0, "out_max_abs_error": 0.00390625, "lse_max_abs_error": 9.5367431640625e-07,
     "passed": true}
    ```

## 性能说明

`--perf` 在同一个输入上先预热 `--perf-warmup` 次，再用 `torch.npu.Event` 逐次记录
`--perf-iters` 次连续调用，输出端到端时延 `latency_us_mean/min/max`、最小搬运量
`bytes_moved`、有效带宽 `effective_gbps` 与吞吐 `tokens_per_s`。这些字段的
`latency_scope` 为 `eager_end_to_end_including_host`：每次调用都包含 Python 侧的张量封装、
任务划分与下发，所以端到端时延由 host 主导，不能当作设备时延使用。

设备时延需用 profiling 工具测量。下表是 8 条内置用例在 Ascend 950PR&950DT 单卡上的实测值，
设备时延取 `step_trace_time.Computing` 的均值（预热 1 次、采样 2 次），端到端时延取
`--perf --perf-iters 20` 的结果：

| 用例 | 设备时延 / ms | 端到端时延 / us |
| --- | --- | --- |
| B1_T1_KV512_H64_float16_PA_BBND_M0 | 0.0109 | 455 |
| B1_T8_KV128_H64_float16_PA_BBND_M0 | 0.0080 | 385 |
| B1_T64_KV512_H64_float16_PA_BBND_M0 | 0.0250 | 387 |
| B2_T512_KV1024_H64_float16_PA_BBND_M0 | 0.2026 | 403 |
| B1_T64_KV512_H96_float16_PA_NZ_M0 | 0.0286 | 396 |
| B1_T64_KV512_H64_float16_PA_BBND_M3 | 0.0233 | 384 |
| B2_T130_KV383_H96_bfloat16_PA_NZ_M3 | 0.0439 | 390 |
| B2_T512_KV2048_H64_bfloat16_PA_BBND_M0 | 0.3860 | 415 |

在更宽的集合上（24 例开发集与 50 例扩展集，覆盖两种布局、两种掩码、两档头数与两档
dtype），设备时延均值分别为 0.103 ms 与 0.078 ms。当 `total_q ≤ 4` 且单个请求的任务数少于
AI Core 数时，样例按 KV 维度切分任务并由归并 kernel 合并，`B1_T1_KV512_H64_float16_PA_BBND_M0`
的设备时延为 0.011 ms。
