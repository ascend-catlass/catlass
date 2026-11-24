# MixedQuantSparseFlashMla Example Readme

## 功能说明

- 算子功能：混合量化稀疏 MLA（Multi-head Latent Attention）前向。KV 为 per-token-group 量化记录，kernel 内完成反量化；支持 SWA、HCA、CSA、ORI_SPARSE、ORI_CMP_SPARSE 五种形态，KV 支持 `BSND` / `TND` / `PA_BBND` 三种布局。
- 计算公式：对每个 query row，可见列集合由 ori 段与 cmp 段拼接，softmax 后加权求和：

  ```text
  s_j = scale * q[r, :] · k_j
  m   = max_j s_j        （有 sinks 时并入 m）
  p_j = exp(s_j - m)
  O[r, :] = Σ_j p_j k_j / (Σ_j p_j + exp(sink - m))
  ```

  P 在 PV 前按输入 dtype 舍入，PV 用 FP32 累加；lse 为 `m + log(denom)`。
- 支持产品型号：Ascend 950PR / 950DT（`--npu-arch 3510`）。

## 参数说明

```python
out, lse = run(
    q, quant_mode,
    ori_kv, cmp_kv,
    ori_sparse_indices, cmp_sparse_indices,
    ori_block_table, cmp_block_table,
    cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
    seqused_q, seqused_ori_kv, seqused_cmp_kv,
    cmp_residual_kv, ori_topk_length, cmp_topk_length,
    sinks, metadata, rope_head_dim, softmax_scale,
    cmp_ratio, ori_mask_mode, cmp_mask_mode,
    ori_win_left, ori_win_right,
    layout_q, layout_kv, topk_value_mode, return_softmax_lse,
)
```

记 `B` 为 batch，`T1` / `S1` 为 query 行数，`T2` / `S2` 为 ori KV token 数，`T2c` / `S2c` 为 cmp KV token 数，`N1` 为 query head 数，`D = 512`；PA 指分页 KV（`layout_kv="PA_BBND"`）。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | Input | BSND `[B, S1, N1, 512]` / TND `[T1, N1, 512]` | `float16` / `bfloat16` | 连续行优先张量 |
| `quant_mode` | Attr | - | `int` | 量化模式，当前实现支持 `1` |
| `ori_kv` | Input | BSND `[B, S2, 1, 512]` / TND `[T2, 1, 512]` / PA `[blocks, page, 1, 512]` | 与 `q` 相同 | 原始 KV；MLA 只有 1 个 KV head |
| `cmp_kv` | Input，可选 | 与 `ori_kv` 同布局 | 与 `q` 相同 | compressed KV；省略时不得给 cmp 掩码与 cmp 索引 |
| `ori_sparse_indices` | Input，可选 | 行数与 query 行数一致，末维为索引宽度 | `int32` | 每行选中的 ori token；`-1` 终止 |
| `cmp_sparse_indices` | Input，可选 | 行数与 query 行数一致，末维为索引宽度 | `int32` | 每行选中的 cmp token；要求给出 `cmp_kv` |
| `ori_block_table` | Input，可选 | `[B, pages]` | `int32` | 分页 ori KV 的页映射 |
| `cmp_block_table` | Input，可选 | `[B, pages]` | `int32` | 分页 cmp KV 的页映射 |
| `cu_seqlens_*` | Input，可选 | `[B + 1]` | `int32` | TND 布局的累积序列长度 |
| `seqused_*` | Input，可选 | `[B]` | `int32` | 每 batch 的实际长度 |
| `cmp_residual_kv` | Input，可选 | `[B]` | `int32` | cmp causal 阈值用的残差长度 |
| `ori_topk_length` / `cmp_topk_length` | Input，可选 | 与 query 行对应 | `int32` | 每行索引的有效前缀长度 |
| `sinks` | Input，可选 | `[N1]` | `float32` | 每 head 的 sink logit |
| `metadata` | Input，可选 | `[2048]` | `int32` | 预生成的调度元数据；省略时由本实现内部生成 |
| `softmax_scale` | Attr | - | `float` | 省略时取 `D ** -0.5` |
| `cmp_ratio` | Attr | - | `int` | cmp KV 的压缩比 |
| `ori_mask_mode` / `cmp_mask_mode` | Attr | - | `int` | 掩码模式 |
| `ori_win_left` / `ori_win_right` | Attr | - | `int` | 滑窗左右宽度 |
| `layout_q` / `layout_kv` | Attr | - | `str` | `BSND` / `TND` / `PA_BBND` |
| `return_softmax_lse` | Attr | - | `bool` | 为 `True` 时返回 `lse` |

输出 `out` 与 `q` 同 shape、同 dtype；`lse` 为 `float32`。

## 运行

```python
import torch
import torch_npu
from mixed_quant_sparse_flash_mla import run

torch.npu.set_device(0)
out, lse = run(
    q, 1,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    cmp_sparse_indices=cmp_sparse_indices,
    ori_block_table=ori_block_table,
    cmp_block_table=cmp_block_table,
    cu_seqlens_q=cu_seqlens_q,
    seqused_ori_kv=seqused_ori_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    sinks=sinks,
    rope_head_dim=64,
    cmp_ratio=4,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=True,
)
```
