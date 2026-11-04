# QwenSparseAttention Example Readme

## 功能说明

- 算子功能：计算带分页 KV cache、逐行稀疏选择的 GQA 注意力前向，将 QK、掩码在线 softmax 与 PV 融合到一个 CATLASS DSL AIC/AIV mixed kernel 中。
- 计算公式：对每个 query row `r`，按 `logical_indices` 选出有效逻辑 token 集合，再经 `token_to_req` 与 `block_table` 映射到分页 cache 的物理位置：

  $$
  \begin{aligned}
  phys(t) &= block\_table[token\_to\_req[r],\ t \,/\, page\_size] \\
  off(t)  &= t \bmod page\_size \\
  s_j     &= scale \cdot q[r, h, :] \cdot K[phys(t_j), off(t_j), kv\_head, :] \\
  O[r, h, :] &= \operatorname{softmax}_j(s_j) \cdot V[phys(t_j), off(t_j), kv\_head, :]
  \end{aligned}
  $$

  softmax 在该行全部有效逻辑 token 上归一化；`kv_head = h // (query_heads / kv_heads)`。该行无有效 token 时输出 0。

- 支持产品型号：Ascend 950PR&950DT 系列产品

## 参数说明

```python
output = qwen_sparse_attention.run(
    q, k_cache, v_cache, logical_indices, block_table, token_to_req,
    softmax_scale=None,
)
```

记 `rows` 为 query token 数，`heads` 为 query head 数，`W` 为每行的选择宽度，`P` 为页大小。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `q` | Input | `[rows, heads, 256]` | `float16` / `bfloat16` | 连续行优先张量，head_dim 维最内 |
| `k_cache` | Input | `[cache_blocks, P, 1, 256]` | 与 `q` 相同 | 分页 Key cache；允许非连续 split 视图，按 stride 寻址 |
| `v_cache` | Input | 与 `k_cache` 相同 | 与 `q` 相同 | 分页 Value cache，shape、dtype 与 stride 必须和 `k_cache` 一致 |
| `logical_indices` | Input | `[rows, W]` | `int32` | 每行选中的逻辑 KV token，负值表示无效；`W ≤ 2051` |
| `block_table` | Input | `[requests, pages]` | `int32` | 逻辑页到物理 cache block 的映射 |
| `token_to_req` | Input | `[rows]` | `int32` | 每行所属请求编号 |
| `softmax_scale` | Input，可选 | 标量 | Python `float` | score 缩放因子，缺省为 `256 ** -0.5 = 0.0625` |
| `output` | Output | `[rows, heads, 256]` | 与 `q` 相同 | 稀疏注意力结果；无有效 token 的行为 0 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| head 维度 | `head_dim` 固定为 256 |
| head 数 | `heads % kv_heads == 0`，当前 `kv_heads = 1`；`heads ≤ 128`，kernel 按 128 个 head 一组切分 task |
| 选择宽度 | `W ≤ 2051`；`logical_indices` 中的有效下标应满足 `0 ≤ t < pages × P`，越界行为未定义 |
| dtype | `q`、`k_cache`、`v_cache` 必须为同一 `float16` / `bfloat16`；三个索引张量必须为 `int32` |
| 张量布局 | `q` 必须连续；`k_cache` 与 `v_cache` 分布在同一个 NPU，允许 `split` 产生的非连续视图，但两者 stride 必须一致 |
| 数值行为 | 空行（全部 index 为负）输出精确 0；`softmax_scale` 必须为有限数值 |

## 使用示例

准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境，导入样例模块后调用 `run`。脚本根据输入编译 kernel、执行 NPU 计算并返回同 shape 的输出：

```python
import torch
import torch_npu  # noqa: F401

import qwen_sparse_attention

rows, heads, width, page_size, pages = 8, 8, 64, 16, 8
kv = torch.randn(pages, page_size, 1, 256, dtype=torch.float16, device="npu:0")
k_cache, v_cache = kv.split(256, dim=-1)
q = torch.randn(rows, heads, 256, dtype=torch.float16, device="npu:0")
logical_indices = torch.randint(0, pages * page_size, (rows, width), dtype=torch.int32, device="npu:0")
block_table = torch.arange(pages, dtype=torch.int32, device="npu:0").reshape(1, pages)
token_to_req = torch.zeros(rows, dtype=torch.int32, device="npu:0")

output = qwen_sparse_attention.run(
    q, k_cache, v_cache, logical_indices, block_table, token_to_req,
    softmax_scale=0.0625,
)
print(output.shape, output.dtype)
```

执行结果如下，说明样例执行成功：

```text
torch.Size([8, 8, 256]) torch.float16
```

与 Torch 参考实现比较（FP16、`rows=8`、`heads=8`、`width=64`、`P=16`，以及 FP16 非连续 split cache、BF16 `width=2051` 生产宽度），精度检测通过：

```text
fp16 contiguous      max_abs=4.70e-04 pass=True
fp16 split cache     max_abs=6.32e-04 pass=True
bf16 width=2051      max_abs=9.21e-04 pass=True
```
