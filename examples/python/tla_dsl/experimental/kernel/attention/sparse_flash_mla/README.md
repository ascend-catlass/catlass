# SparseFlashMLA Example Readme

## 功能说明

- 算子功能：稀疏 MLA（Multi-head Latent Attention）注意力前向，支持 ori / cmp 双路 KV（compressed KV 按 `cmp_ratio` 压缩）、token 级稀疏索引、分页 KV cache、dense / causal / 滑窗掩码，并可写回 FP32 的 lse。
- 计算公式：对每个 query row `r` 与 head `h`，可见列集合由 ori 段与 cmp 段拼接而成：

  ```text
  dense ori 段：ori_mask_mode=0 [0, T2)；=3 [0, min(T2 - S1 + r + 1, T2))；=4 [max(thr - win_left - 1, 0), min(thr + win_right, T2))
  sparse ori 段：按 ori_sparse_indices 前缀取，-1 终止，按段边界过滤
  cmp 段：[0, thr_c)，causal 时 thr_c = min((restored_cmp - S1 + r + 1) // cmp_ratio, T2c)
  ```

  $$
  \begin{aligned}
  s_j &= scale \cdot q[r,h,:] \cdot k_j \\
  m &= \max_j s_j \quad (\text{有 sinks 时 } m = \max(m,\ sink_h)) \\
  p_j &= e^{s_j - m} \\
  O[r,h,:] &= \frac{\sum_j \operatorname{round}_{io}(p_j)\, k_j}{\sum_j p_j + e^{sink_h - m}}
  \end{aligned}
  $$

  P 在 PV 前按输入 dtype 舍入、PV 用 FP32 累加；sinks 只进入 max 与分母。可见集为空的行输出 0 且不写 lse；lse 为 `m + log(denom + 1e-10)`。
- 支持产品型号：Ascend 950PR / 950DT（`--npu-arch 3510`）

## 参数说明

```python
out, lse = run(
    q, ori_kv, cmp_kv,
    ori_sparse_indices, cmp_sparse_indices,
    ori_block_table, cmp_block_table,
    cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
    seqused_q, seqused_ori_kv, seqused_cmp_kv,
    cmp_residual_kv, ori_topk_length, cmp_topk_length,
    sinks, metadata, softmax_scale, cmp_ratio,
    ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
    layout_q, layout_kv, return_softmax_lse, *, compile_only=False,
)
```

记 `B` 为 batch，`T1` / `S1` 为 query 行数，`T2` / `S2` 为 ori KV token 数，`T2c` / `S2c` 为 cmp KV token 数，`N1` 为 query head 数，`D = 512`；PA 指分页 KV（`layout_kv="PA_BBND"`）。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | Input | BSND `[B, S1, N1, 512]` / TND `[T1, N1, 512]` | `float16` / `bfloat16` | 连续行优先张量 |
| `ori_kv` | Input | BSND `[B, S2, 1, 512]` / TND `[T2, 1, 512]` / PA `[blocks, page, 1, 512]` | 与 `q` 相同 | 原始 KV；MLA 只有 1 个 KV head |
| `cmp_kv` | Input，可选 | 与 `ori_kv` 同布局 | 与 `q` 相同 | compressed KV；省略时不得给 cmp 掩码与 cmp 索引 |
| `ori_sparse_indices` | Input，可选 | 行数与 query 行数一致，末维为索引宽度 | `int32` | 每行选中的 ori token；`-1` 终止，越界跳过 |
| `cmp_sparse_indices` | Input，可选 | 行数与 query 行数一致，末维为索引宽度 | `int32` | 每行选中的 cmp token；要求给出 `cmp_kv` |
| `ori_block_table` | Input，可选 | `[B, pages]` | `int32` | 分页 ori KV 的页映射 |
| `cmp_block_table` | Input，可选 | `[B, pages]` | `int32` | 分页 cmp KV 的页映射 |
| `cu_seqlens_q` | Input，可选 | `[B + 1]` | `int32` | TND query 必需（可由 `seqused_q` 代替） |
| `cu_seqlens_ori_kv` | Input，可选 | `[B + 1]` | `int32` | 非分页 TND ori KV 必需 |
| `cu_seqlens_cmp_kv` | Input，可选 | `[B + 1]` | `int32` | 非分页 TND cmp KV 必需 |
| `seqused_q` | Input，可选 | `[B]` | `int32` | 每 batch 的实际 query 长度 |
| `seqused_ori_kv` | Input，可选 | `[B]` | `int32` | 每 batch 的实际 ori KV 长度；分页路径必需 |
| `seqused_cmp_kv` | Input，可选 | `[B]` | `int32` | 每 batch 的实际 cmp KV 长度；分页路径必需 |
| `cmp_residual_kv` | Input，可选 | `[B]` | `int32` | cmp causal 阈值用的残差长度，仅在 `cmp_ratio > 1` 且 `cmp_mask_mode != 0` 时生效 |
| `ori_topk_length` | Input，可选 | BSND `[B, S1]` / TND `[T1]` | `int32` | 每行 ori 索引的有效前缀长度 |
| `cmp_topk_length` | Input，可选 | BSND `[B, S1]` / TND `[T1]` | `int32` | 每行 cmp 索引的有效前缀长度 |
| `sinks` | Input，可选 | `[N1]` | `float32` | 每 head 的 sink 值，只进 max 与分母 |
| `metadata` | Input（必需） | `[L]`，L 随上游版本（1024 / 2048 等） | `int32` | 预留元数据；本实现不参与计算，仅校验 1-D int32 |
| `softmax_scale` | Input | 标量 | Python `float` | QK 缩放因子 |
| `cmp_ratio` | Input | 标量 | Python `int` | compressed KV 压缩比，正整数 |
| `ori_mask_mode` | Input | 标量 | Python `int` | `0` 全量 / `3` causal / `4` 滑窗 |
| `cmp_mask_mode` | Input | 标量 | Python `int` | `0` 全量 / `3` causal |
| `ori_win_left` / `ori_win_right` | Input | 标量 | Python `int` | 滑窗左右宽度，`-1` 表示不限制 |
| `layout_q` | Input | 标量 | Python `str` | `"BSND"` 或 `"TND"` |
| `layout_kv` | Input | 标量 | Python `str` | `"BSND"`、`"TND"` 或 `"PA_BBND"` |
| `return_softmax_lse` | Input | 标量 | Python `bool` | 是否写回 `lse` |
| `out` | Output | 与 `q` 相同 | 与 `q` 相同 | attention 结果 |
| `lse` | Output | TND `[1, T1, N1]` / BSND `[B, 1, S1, N1]` | `float32` | `max + log(sum + 1e-10)`；关闭时为空张量 |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 维度 | `D` 固定 512；`N1 ≥ 1`，KV head 数固定为 1 |
| dtype | `q`、`ori_kv`、`cmp_kv` 必须同为 `float16` 或 `bfloat16`；索引与长度张量为 `int32`；`sinks` 为连续 `float32` 且元素数等于 `N1` |
| 布局 | `layout_q ∈ {BSND, TND}`；`layout_kv ∈ {BSND, TND, PA_BBND}`；分页 KV 与 query 布局相互独立，非分页时 `layout_q == layout_kv`；`q` 必须连续，所有张量位于同一 NPU |
| 分页 | 分页 KV 需要对应 `block_table` 与 `seqused_*`；ori / cmp 两侧页大小各自独立，分别由对应 KV 张量的第 1 维给出 |
| 掩码 | `ori_mask_mode ∈ {0, 3, 4}`，`cmp_mask_mode ∈ {0, 3}`；无 `cmp_kv` 时 `cmp_mask_mode` 必须为 0 |
| 压缩比例 | `cmp_ratio ≥ 1`；cmp causal 阈值按 `restored_cmp = T2c * cmp_ratio + residual` 计算 |
| 稀疏路由 | `cmp_sparse_indices` 必须配合 `cmp_kv`；给出 `ori_sparse_indices` 与 `cmp_kv` 却缺 `cmp_sparse_indices` 时直接报错，不静默退化 |
| topk 长度 | 稀疏路由必须给对应 `*_topk_length`：仅 cmp 稀疏需 `cmp_topk_length`，仅 ori 稀疏需 `ori_topk_length`，双稀疏两者都要；缺失即报错 |
| 执行形态 | 按输入组合选路，每次 `run` 只执行一个分支：无 `cmp_kv` 且无稀疏索引 → 稠密 / causal / 滑窗；有 `cmp_kv` 无稀疏索引 → ori 连续 + cmp 连续；仅 ori 稀疏索引 → 仅 ori 段；仅 cmp 稀疏索引 → ori 连续 + cmp 稀疏；两者都有 → 双稀疏。`N1 > 64` 时 head 维按两组切分执行 |
| ragged BSND | `layout_q == BSND` 且 `seqused_q` 有小于 `S1` 的 batch 时，按有效行成包送入、跑完散射回物理位置，`out` / `lse` 的 padding 行写 0；非分页 `BSND` KV 的 `seqused_ori_kv` / `seqused_cmp_kv` 同理 |
| lse | `return_softmax_lse=True` 时写回 FP32 的 `lse`；关闭时返回空张量，内部仍按同形缓冲分配 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/attention:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401  -- registers torch.npu
from sparse_flash_mla.sparse_flash_mla import run

q = torch.randn(192, 32, 512, dtype=torch.bfloat16, device="npu:0")
ori_kv = torch.randn(1024, 1, 512, dtype=torch.bfloat16, device="npu:0")
cu_q = torch.tensor([0, 192], dtype=torch.int32, device="npu:0")
cu_kv = torch.tensor([0, 1024], dtype=torch.int32, device="npu:0")
metadata = torch.zeros(1024, dtype=torch.int32, device="npu:0")

out, lse = run(
    q, ori_kv, None,        # cmp_kv
    None, None,             # ori / cmp_sparse_indices
    None, None,             # ori / cmp_block_table
    cu_q, cu_kv, None,      # cu_seqlens_q / ori_kv / cmp_kv
    None, None, None,       # seqused_q / ori_kv / cmp_kv
    None, None, None,       # cmp_residual_kv / topk lengths
    None, metadata, 512 ** -0.5,  # sinks / metadata / softmax_scale
    1, 0, 0, -1, -1,        # cmp_ratio / mask modes / win
    "TND", "TND", False,    # layout_q / layout_kv / return_softmax_lse
)
print(out.shape, out.dtype)
PY
```

```text
torch.Size([192, 32, 512]) torch.bfloat16
```
