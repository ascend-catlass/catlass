# SplitQkvRmsnormRope Example Readme

## 功能说明

- 算子功能：对已融合的 QKV 输入按 `[q_hidden_size, kv_hidden_size, kv_hidden_size]` 切分为 Q、K、V；对 Q、K 逐 head 做 RMSNorm、乘 head 权重（可选加 bias），再对每 head 前 `rope_dim` 维做 NeoX 风格 RoPE；V 直接拷出。入口为 `split_qkv_rmsnorm_rope` 模块的 `split_qkv_rmsnorm_rope` 函数，基于 CATLASS DSL 实现的单个纯 AIV kernel。
- 计算公式：

  $$
  \text{var} = \frac{1}{d}\sum_{i=0}^{d-1} x_i^2,\qquad
  y_i = \frac{x_i}{\sqrt{\text{var} + \epsilon}} \cdot w_i + b_i
  $$

  归一化结果按 bfloat16 舍入后做旋转，`half = rope_dim // 2`，`cos, sin = cache[pos, :half], cache[pos, half:]`：

  $$
  y'_i = y_i \cos_i - y_{i+half} \sin_i, \qquad
  y'_{i+half} = y_{i+half} \cos_i + y_i \sin_i \qquad i \in [0, half)
  $$

  `rope_dim < head_dim` 时（partial RoPE），`[rope_dim, head_dim)` 原样保留。其中 $x$ 为单个 head 的输入向量，$d$ 为 `head_dim`，$w$ / $b$ 为该 head 的权重与偏置。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
split_qkv_rmsnorm_rope(
    input, cos_sin_cache, positions, q_weight, k_weight,
    q_hidden_size, kv_hidden_size, head_dim, eps,
    q_bias=None, k_bias=None, block_num=None,
)
```

记 `T` 为 token 数，返回 `(q, k, v)`；`rope_dim` 由 `cos_sin_cache.shape[-1]` 推出。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `input` | Input | `(T, q_hidden_size + 2 * kv_hidden_size)` | `bfloat16` | QKV 融合输入。 |
| `cos_sin_cache` | Input | `(max_positions, rope_dim)` | `bfloat16` | 每行 `[cos | sin]`。 |
| `positions` | Input | `(T,)` | `int64` | 每个 token 的位置索引。 |
| `q_weight`、`k_weight` | Input | `(head_dim,)` | `bfloat16` | 逐 head 的 RMSNorm 权重。 |
| `q_bias`、`k_bias` | Input，可选 | `(head_dim,)` | `bfloat16` | 归一化后的偏置，默认 `None`。 |
| `q_hidden_size` | 属性 | 标量 | Python `int` | `q_heads * head_dim`。 |
| `kv_hidden_size` | 属性 | 标量 | Python `int` | `kv_heads * head_dim`。 |
| `head_dim` | 属性 | 标量 | Python `int` | 单个 head 宽度，样例仅支持 `128`。 |
| `eps` | 属性 | 标量 | Python `float` | RMSNorm 数值稳定项。 |
| `block_num` | 属性，可选 | 标量 | Python `int` / `None` | 启动核数；默认 `None` 表示取 `vector_core_num`。 |
| `q` | Output | `(T, q_hidden_size)` | `bfloat16` | RMSNorm 与 RoPE 后的 Q。 |
| `k` | Output | `(T, kv_hidden_size)` | `bfloat16` | RMSNorm 与 RoPE 后的 K。 |
| `v` | Output | `(T, kv_hidden_size)` | `bfloat16` | 原样拷贝的 V。 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入维度 | `input` 为 2 维张量，最后一维等于 `q_hidden_size + 2 * kv_hidden_size`；`positions` 长度等于 `input` 行数。 |
| head 切分 | `q_hidden_size % head_dim == 0`、`kv_hidden_size % head_dim == 0`；`q_hidden_size` 必须是 `kv_hidden_size` 的整数倍（GQA / MQA 几何）。 |
| RoPE 宽度 | `rope_dim` 由 `cos_sin_cache` 最后一维给出，须为正偶数且不超过 `head_dim`，并满足 `head_dim % (rope_dim // 2) == 0`（head 起点须 tile 对齐）。 |
| 位置 | `positions` 的每个元素须落在 `[0, cos_sin_cache.shape[0])` 内。 |
| dtype | 仅支持 `bfloat16` 输入；`input`、`cos_sin_cache`、`positions`、权重与偏置须位于同一 NPU 且连续。 |
| 偏置 | `q_bias`、`k_bias` 必须同时提供或同时省略，否则入口抛 `ValueError`。 |
| 数值行为 | 设备侧在 FP32 下完成 RMSNorm 与 RoPE 乘加，归一化结果与旋转结果各做一次 bfloat16 舍入。 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from split_qkv_rmsnorm_rope.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

torch.npu.set_device(0)
t, head_dim, q_hidden, kv_hidden = 8, 128, 32 * 128, 8 * 128
dtype = torch.bfloat16
input = torch.randn(t, q_hidden + 2 * kv_hidden, dtype=dtype, device="npu:0")
q_weight = torch.ones(head_dim, dtype=dtype, device="npu:0")
k_weight = torch.ones_like(q_weight)
cos_sin_cache = torch.randn(1024, head_dim, dtype=dtype, device="npu:0")
positions = torch.arange(t, dtype=torch.int64, device="npu:0")

q, k, v = split_qkv_rmsnorm_rope(
    input, cos_sin_cache, positions, q_weight, k_weight,
    q_hidden, kv_hidden, head_dim, 1e-6,
)
print(q.shape, k.shape, v.shape)
PY
```

```text
torch.Size([8, 4096]) torch.Size([8, 1024]) torch.Size([8, 1024])
```
