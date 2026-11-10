# FlashAttnFp8Fullquant Example Readme

## 功能说明

- 算子功能：计算推理阶段的 FP8 全量化分页 GQA 注意力，将 QK、
  online softmax、P 量化与 PV 融合到一个 CATLASS DSL AIC/AIV mixed
  kernel 中。支持 causal 开关、变长 batch、GQA/MQA、128 行分页 cache 和
  可调 P 量化幅度。
- 计算公式：记逻辑 KV token 为 `t`，物理页为
  `phys(t) = block_table[b, t / 128]`，页内偏移为 `off(t) = t mod 128`，
  KV head 为 `kh = h / (Hq / Hkv)`。对每个 query 行 `r` 和 head `h`：

  $$
  \begin{aligned}
  s_{r,h,t} &=
    q_s[r] \cdot q[r,h,:] \cdot
    k_s[phys(t), off(t)] \cdot
    K[phys(t), off(t), kh,:] \cdot scale \\
  m_{r,h} &= \max_t s_{r,h,t} \\
  e_{r,h,t} &= \exp(s_{r,h,t} - m_{r,h}) \\
  P_{r,h,t} &=
    \operatorname{round\_e4m3}(e_{r,h,t} \cdot p\_scale) \\
  O_{r,h,:} &=
    \frac{\sum_t P_{r,h,t}V[phys(t),off(t),kh,:]
          \cdot v_s[kh]}
         {p\_scale \cdot \sum_t e_{r,h,t}}
  \end{aligned}
  $$

  QK/PV 使用 E4M3 输入和 FP32 累加。P 写入 PV 前按 `p_scale` 放大并舍入到
  E4M3FN，最终分母使用量化前 FP32 指数和乘以 `p_scale`。causal mask 采用
  right-bottom 对齐：query token 的绝对位置为
  `query_index + kv_len - q_len`。

- 支持产品型号：Ascend 950PR/950DT 系列产品；当前编译目标为
  `--npu-arch 3510`。

## 参数说明

```python
output = flash_attn_fp8_fullquant.run(
    q, q_scale, k_cache, k_scale, v_cache, v_scale,
    block_table, seq_lens_q, seq_lens_kv,
    softmax_scale, causal, p_scale,
)
```

记 `T` 为所有 batch 的 query token 总数，`P` 为物理页数，`Hq` 为 query
head 数，`Hkv` 为 KV head 数，`D=128`。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | Input | `[T, Hq, D]` | `uint8` | E4M3FN 位模式 query，TND 布局 |
| `q_scale` | Input | `[T]` | `float32` | Q 逐 token descale |
| `k_cache` | Input | `[P, 128, Hkv, D]` | `uint8` | E4M3FN 位模式分页 K cache |
| `k_scale` | Input | `[P, 128]` | `float32` | K 逐 cache row descale |
| `v_cache` | Input | `[P, 128, Hkv, D]` | `uint8` | E4M3FN 位模式分页 V cache |
| `v_scale` | Input | `[Hkv]` | `float32` | V 逐 KV head descale |
| `block_table` | Input | `[batch, max_blocks]` | `int32` | 逻辑页到物理页映射，未使用项可填 `-1` |
| `seq_lens_q` | Input | `[batch]` | `int32` | 每个请求的 query 长度 |
| `seq_lens_kv` | Input | `[batch]` | `int32` | 每个请求的 KV 长度 |
| `softmax_scale` | Input | 标量 | Python `float` | softmax score 缩放 |
| `causal` | Input | 标量 | Python `int` | `1` 表示 causal，`0` 表示全量注意力 |
| `p_scale` | Input | 标量 | Python `float` | P 量化幅度，必须为正数 |
| `output` | Output | `[T, Hq, D]` | `float16` | 注意力输出，TND 布局 |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 维度 | `D=128`；物理页大小 `page_size=128` |
| Head | `Hq % Hkv == 0`；支持 MHA、GQA 和 MQA |
| FP8 | `q`、`k_cache`、`v_cache` 以 `uint8` 承载 E4M3FN 位模式；应避开 NaN 编码 `0x7F/0xFF` |
| 序列 | `sum(seq_lens_q) == T`；`seq_lens_kv[b] <= max_blocks * 128` |
| 页表 | 每个请求实际使用的页号必须在 `[0, P)` 内 |
| Causal | `causal=1` 时采用 right-bottom 对齐，允许 `kv_len >= q_len` |
| Tail | 最后一页按 `seq_lens_kv` 截断，mask 后不会读取有效行之外的数据参与归一化 |
| 数值 | `softmax_scale` 必须有限；`p_scale > 0` |
| 设备 | 输入张量位于同一 NPU |

## Kernel 组织

- 一个 task 处理 `(batch, query_head, 64-row query block)`，按 grid-stride
  分布在所有 Cube core 上。
- 每个物理 KV page 一次完成 `QK -> online softmax/P quantize -> PV`；
  不为每个 V block 重复扫描 K。
- mixed kernel 使用两个 AIV 子块各处理最多 32 个 query row；Q/K/V/P 的
  L1/L0 和 UB 缓冲区通过 same-core flag 与 cross-core flag 形成
  producer/consumer 流水。
- causal task 只加载到该 query block 最后一行可见的 KV 范围；完全可见页
  跳过 mask，边界页按 64 列 chunk 掩码。
- 每个 head 的 query block 使用蛇形顺序，交错长短 causal task，降低
  多核尾部不均衡。

## 使用示例

准备好 CANN 9.2.1、PyTorch、torch_npu 与 CATLASS DSL 2.2.0 环境，导入样例
模块后调用 `run`：

```python
import torch
import torch_npu  # noqa: F401

import flash_attn_fp8_fullquant

batch, tokens, hq, hkv, pages, dim = 1, 1, 8, 2, 1, 128
q = torch.randint(0, 126, (tokens, hq, dim), dtype=torch.uint8, device="npu:0")
q_scale = torch.rand(tokens, dtype=torch.float32, device="npu:0")
k_cache = torch.randint(0, 126, (pages, 128, hkv, dim), dtype=torch.uint8, device="npu:0")
k_scale = torch.rand(pages, 128, dtype=torch.float32, device="npu:0")
v_cache = torch.randint(0, 126, (pages, 128, hkv, dim), dtype=torch.uint8, device="npu:0")
v_scale = torch.rand(hkv, dtype=torch.float32, device="npu:0")
block_table = torch.arange(pages, dtype=torch.int32, device="npu:0").reshape(batch, pages)
seq_lens_q = torch.tensor([tokens], dtype=torch.int32, device="npu:0")
seq_lens_kv = torch.tensor([128], dtype=torch.int32, device="npu:0")

output = flash_attn_fp8_fullquant.run(
    q, q_scale, k_cache, k_scale, v_cache, v_scale,
    block_table, seq_lens_q, seq_lens_kv,
    softmax_scale=dim ** -0.5,
    causal=1,
    p_scale=256.0,
)
print(output.shape, output.dtype)
```

预期输出：

```text
torch.Size([1, 8, 128]) torch.float16
```

本地开发验证使用 `workload.jsonl` 中 40 条 case，覆盖 decode/prefill、
MHA/GQA/MQA、causal/noncausal、变长 batch、tail page 和不同 `p_scale`。
