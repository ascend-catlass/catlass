# SparseFlashAttention Example Readme

## 功能说明

- 算子功能：MLA-absorb 稀疏注意力前向，入口为 `sparse_flash_attention` 模块的 `run`
  函数。单个 `@tla.kernel` 的 AIC/AIV mixed kernel 完成 QK（NoPE 512 维 + 可选 RoPE
  64 维）、带 sinks 的 online softmax、P 按输入 dtype 舍入后的 PV，并可写回 softmax
  的 max / sum。支持 token 级稀疏索引、分页 KV cache、right-down causal 掩码与 BSND / TND 布局。
- 计算公式：对每个 query row `r` 与 head `h`，按 `sparse_indices` 选出该行的有效
  token 集合 `j`（遇 `-1` 终止、越界跳过、重复不去重）：

  $$
  \begin{aligned}
  s_j &= scale \cdot \left(q[r,h,:] \cdot k_j + q_{rope}[r,h,:] \cdot k_{rope,j}\right) \\
  m &= \max_j s_j \quad (\text{有 sinks 时 } m = \max(m,\ sink_h)) \\
  p_j &= e^{s_j - m} \\
  O[r,h,:] &= \frac{\sum_j \operatorname{round}_{io}(p_j)\, k_j}
                   {\sum_j p_j + [\text{有 sinks}]\,e^{sink_h - m}}
  \end{aligned}
  $$

  其中 `value` 与 `key` 是同一份字节（MLA-absorb，V ≡ K），`value` 只做参数校验、不参与
  计算；sinks 只进入 max 与分母，不参与加权（无 sinks 时分母不含 sink 项）。稀疏阈值：
  `sparse_mode=0` 为 `kv_len`，`sparse_mode=3` 为 `kv_len - q_len + row + 1`。空行输出 0。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
attention_out, softmax_max, softmax_sum = run(
    query, key, value, sparse_indices,
    block_table=None, query_rope=None, key_rope=None, sinks=None,
    cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None,
    scale_value=1.0, sparse_block_size=1,
    layout_query="BSND", layout_kv="BSND", sparse_mode=0,
    pre_tokens=2**63 - 1, next_tokens=2**63 - 1, attention_mode=2,
    return_softmax_lse=False,
)
```

记 `B` 为 batch，`S1` / `T1` 为 query 行数，`S2` / `T2` 为 KV token 数，`N1` 为 query head 数，`K` 为每行稀疏索引宽度，`P` 为分页页大小。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `query` | Input | BSND `[B, S1, N1, 512]` / TND `[T1, N1, 512]` | `float16` / `bfloat16` | 连续行优先张量，head_dim 固定 512 |
| `key` | Input | BSND `[B, S2, 1, 512]` / TND `[T2, 1, 512]` / PA `[blocks, P, 1, 512]` | 与 `query` 相同 | MLA 只有 1 个 KV head |
| `value` | Input | 与 `key` 相同 | 与 `query` 相同 | MLA-absorb 下 V ≡ K，仅校验，不参与计算 |
| `sparse_indices` | Input | BSND `[B, S1, 1, K]` / TND `[T1, 1, K]` | `int32` | 每行选中的 KV token；`-1` 终止，越界跳过 |
| `block_table` | Input，可选 | `[B, pages]` | `int32` | 仅 `layout_kv="PA_BSND"` 时使用 |
| `query_rope` | Input，可选 | 与 `query` 前缀相同，末维 64 | 与 `query` 相同 | 与 `key_rope` 必须同时给出或同时省略 |
| `key_rope` | Input，可选 | 与 `key` 前缀相同，末维 64 | 与 `query` 相同 | RoPE 部分拼接在 NoPE 之后参与 QK |
| `sinks` | Input，可选 | `[N1]` | `float32` | 每 head 的 sink 值，只进 max 与分母 |
| `cu_seqlens_q` | Input，可选 | `[B + 1]` | `int32` | 仅 TND query 需要 |
| `cu_seqlens_kv` | Input，可选 | `[B + 1]` | `int32` | 仅 TND KV 需要 |
| `seqused_q` | Input，可选 | `[B]` | `int32` | 每 batch 的实际 query 长度 |
| `seqused_kv` | Input，可选 | `[B]` | `int32` | 每 batch 的实际 KV 长度；分页路径必需 |
| `scale_value` | Input | 标量 | Python `float` | QK 缩放因子 |
| `sparse_block_size` | Input | 标量 | Python `int` | 当前仅支持 1 |
| `layout_query` | Input | 标量 | Python `str` | `"BSND"` 或 `"TND"` |
| `layout_kv` | Input | 标量 | Python `str` | `"BSND"`、`"TND"` 或 `"PA_BSND"` |
| `sparse_mode` | Input | 标量 | Python `int` | `0` 稠密阈值 / `3` right-down causal |
| `pre_tokens` / `next_tokens` | Input | 标量 | Python `int` | 保持默认值，非默认会被拒绝 |
| `attention_mode` | Input | 标量 | Python `int` | 固定为 2 |
| `return_softmax_lse` | Input | 标量 | Python `bool` | 是否写回 `softmax_max` / `softmax_sum` |
| `attention_out` | Output | 与 `query` 相同 | 与 `query` 相同 | attention 结果 |
| `softmax_max` | Output | BSND `[B, 1, S1, N1]` / TND `[1, T1, N1]` | `float32` | online softmax 的行最大值；关闭时为空张量 |
| `softmax_sum` | Output | 与 `softmax_max` 相同 | `float32` | online softmax 的行分母；关闭时为空张量 |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 布局组合 | `BSND/BSND`、`BSND/PA_BSND`、`TND/TND`、`TND/PA_BSND` |
| head 维度 | NoPE 固定 512；RoPE 可选且固定 64 |
| head 数 | `N1 ∈ [1, 128]`，KV head 数固定为 1 |
| dtype | `query`、`key`、`value` 必须同为 `float16` 或 `bfloat16`；索引与长度张量为 `int32`；`sinks` 为 `float32` |
| 分页 KV | 页大小为 16 的倍数且 `∈ [16, 1024]`；`block_table` 与 `seqused_kv` 必需；不支持 `return_softmax_lse` |
| 稀疏索引 | `K > 0`；值域按行阈值过滤，`-1` 终止扫描 |
| 标量参数 | `attention_mode=2`、`sparse_block_size=1`、`pre_tokens` / `next_tokens` 保持默认 |
| 张量布局 | `query` 与 `value` 必须连续；`value` 必须与 `key` 同 shape、同 dtype |
| 执行形态 | 每次 `run` 只 launch 一个 kernel，host 侧不做 torch 计算替代 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/attention:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

from sparse_flash_attention.sparse_flash_attention import run

B, S1, S2, N1, D, K = 1, 128, 1024, 32, 512, 64
device = "npu:0"
query = torch.randn(B, S1, N1, D, dtype=torch.bfloat16, device=device)
key = torch.randn(B, S2, 1, D, dtype=torch.bfloat16, device=device)
value = key.clone()
sparse_indices = torch.arange(K, dtype=torch.int32, device=device)
sparse_indices = sparse_indices.view(1, 1, 1, K).expand(B, S1, 1, K).contiguous()

attention_out, _, _ = run(
    query, key, value, sparse_indices,
    scale_value=D ** -0.5,
)
print(attention_out.shape, attention_out.dtype)
PY
```

预期输出：

```text
torch.Size([1, 128, 32, 512]) torch.bfloat16
```
