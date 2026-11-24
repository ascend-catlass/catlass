# QuantSparseLightningIndexerDsl Example Readme

## 功能说明

- 算子功能：对打包 MXFP4 Q/K 执行稀疏 Lightning Indexer 打分，在 candidate
  8-token block 中完成 TND/PA_BBND 寻址、可见性裁剪、32-head 加权 ReLU 归约和
  确定性 TopK。入口为 `quant_sparse_lightning_indexer_dsl.py` 的 `run` 函数。
- 计算公式：记 `Q_s[i,h]`、`K_s[j]` 为 E8M0 scale，`W[i,h]` 为 head 权重，
  `candidate(i)` 为 query `i` 的有效 candidate token 集合：

  $$
  \begin{aligned}
  qk_{i,h,j} &= \operatorname{BF16}\left(
    Q_s[i,h] \cdot Q[i,h,:] \cdot K_s[j] \cdot K[j,:]
  \right) \\
  score_{i,j} &=
    \sum_{h=0}^{31} \operatorname{BF16}\left(
      score_{i,j} + \operatorname{BF16}(W[i,h]) \cdot
      \max(qk_{i,h,j}, 0)
    \right) \\
  indices_i &= \operatorname{TopK}_{desc(score), token\_asc}
    \left(score_{i,j},\ j \in candidate(i) \cap visible(i)\right)
  \end{aligned}
  $$

  TopK 按 score 降序输出；同分时逻辑 token 下标升序。Padding 输出为 `-1`/`-inf`。
- 支持产品型号：Ascend 950PR&950DT系列产品。

## 代码组织

`quant_sparse_lightning_indexer_dsl.py` 同时包含 Device kernel 和 host 调用入口：

| 组件 | 说明 |
| ---- | ---- |
| `prepare_candidates_kernel` | 在 UB bitmap 中去重 candidate block，生成升序 block 列表和连续 packed-K token 视图。 |
| `qk_kernel` / `qk2_kernel` | 使用 MX MMAD 计算单行或两行 query 的 QK，L1/L0/L0C 由显式 mutex 管理。 |
| `reduce_topk_kernel` | 32-head 顺序 BF16 加权归约、可见性过滤和确定性 TopK。 |
| `reduce_chunk_topk_kernel` | 对大 TopK shape 按 token chunk 并行归约和排序。 |
| `merge_topk_kernel` / `merge_stage_topk_kernel` | 保留 top-K 前缀的串行或分层并行 merge。 |
| `run` | 校验 host 参数，复用 workspace/compiled artifact，并按 shape 选择 kernel 组合。 |

主要性能设计：

- candidate 去重不依赖全局排序，并在同一 row 内按 AIV split 并行物化 packed-K。
- 两行 query 合并到一次 QK launch，使两个 row 共享同一次 Cube wave。
- 大 TopK 先并行生成 chunk TopK，再执行 top-K-preserving merge tree。
- host 侧缓存 shape-stable workspace、DLPack binding template 和 compiled artifact。

## 参数说明

```python
indices, values = run(
    q, k, w, descale_q, descale_k,
    candidate_block_indices, candidate_block_length,
    cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
    cmp_residual_k, block_table, output_idx_offset,
    topk, candidate_block_size, mask_mode, cmp_ratio, return_value,
)
```

记 `T1` 为 query token 总数，`T2` 为 TND K token 总数，`B` 为请求数，
`topk=K`，candidate block size 固定为 8。

| 参数 | 属性 | shape | dtype | 说明 |
| ---- | ---- | ----- | ----- | ---- |
| `q` | Input | `[T1,32,64]` | `uint8` | row-major packed MXFP4 Q，逻辑 `D=128` |
| `k` | Input | TND `[T2,1,64]`；PA `[pages,page_size/8,544]` | `uint8` | packed K；PA record 内含 8-token K 和 E8M0 scale |
| `w` | Input | `[T1,32]` | `float32` | Lightning head 权重，计算时舍入到 BF16 |
| `descale_q` | Input | `[T1,32,2,2]` | `uint8` | Q E8M0 scale，共 4 个 group32 |
| `descale_k` | Input | TND `[T2,1,2,2]`；PA `null` | `uint8` | TND K scale |
| `candidate_block_indices` | Input | `[T1,1,2048]` | `int32` | candidate block 下标；长度之后的槽位忽略 |
| `candidate_block_length` | Input | `[T1,1]` | `int32` | 每行有效 candidate 前缀长度 |
| `cu_seqlens_q` | Input | `[B+1]` | `int32` | Q TND 累计长度 |
| `cu_seqlens_k` | Input | TND `[B+1]`；PA `null` | `int32` | K TND 累计长度 |
| `seqused_q`、`seqused_k` | Input | `[B]` | `int32` | 每请求有效 Q/K 长度 |
| `cmp_residual_k` | Input | `[B]` 或 `null` | `int32` | causal 压缩残差 |
| `block_table` | Input | PA `[B,max_pages]`；TND `null` | `int32` | 逻辑页到物理页映射 |
| `output_idx_offset` | Input | `[T1,1]` 或 `null` | `int32` | 输出 token 偏移 |
| `topk` | 属性 | 标量 | Python `int` | TopK 上限；有效值由输出 shape、UB 和 sort size 共同约束 |
| `candidate_block_size` | 属性 | 标量 | Python `int` | 固定为 8 |
| `mask_mode` | 属性 | 标量 | Python `int` | `0` 或 right-down causal `3` |
| `cmp_ratio` | 属性 | 标量 | Python `int` | causal 压缩比 |
| `return_value` | 属性 | 标量 | `bool` / `int` | 是否返回 BF16 score |
| `indices`、`values` | Output | `[T1,1,K]`；关闭 value 时 `values` 为 `[0]` | `int32`、`bfloat16` | 确定性 TopK 输出 |

## 约束说明

| 项目 | 约束 |
| ---- | ---- |
| 固定维度 | `Q_HEADS=32`、逻辑 `HEAD_DIM=128`、candidate block size 为 8 |
| candidate | 每行最多 2048 个 block；仅 `candidate_block_length` 前缀有效 |
| 可见性 | `mask_mode=0` 使用 `seqused_k`；`mask_mode=3` 使用 right-down causal 公式 |
| 页表 | PA 页号必须位于 `[0,pages)`；尾页读取不得超过 `seqused_k` |
| 数值 | MXFP4 解码、BF16 QK、BF16 sequential weighted sum 和 tie-break 均为精确合同 |
| 设备 | 所有输入位于同一 NPU；shape 变化会生成新的 compiled artifact |

## 使用示例

以下示例在 CATLASS 仓库根目录执行，构造一个 `rows=3`、`seqlen_k=65`、
`topk=16` 的 TND 用例：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/indexer:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from quant_sparse_lightning_indexer_dsl.quant_sparse_lightning_indexer_dsl import run

rows = 3
key_length = 65
topk = 16
candidate_blocks = 9
i32 = {"dtype": torch.int32, "device": "npu:0"}
u8 = {"dtype": torch.uint8, "device": "npu:0"}

q = torch.randint(0, 256, (rows, 32, 64), **u8)
k = torch.randint(112, 128, (key_length, 1, 64), **u8)
w = torch.rand((rows, 32), dtype=torch.float32, device="npu:0")
descale_q = torch.full((rows, 32, 2, 2), 127, **u8)
descale_k = torch.full((key_length, 1, 2, 2), 127, **u8)
candidate_ids = (
    torch.arange(candidate_blocks, **i32)
    .reshape(1, 1, candidate_blocks)
    .expand(rows, 1, candidate_blocks)
    .contiguous()
)
candidate_length = torch.full((rows, 1), candidate_blocks, **i32)
cu_seqlens_q = torch.tensor([0, rows], **i32)
cu_seqlens_k = torch.tensor([0, key_length], **i32)
seqused_q = torch.tensor([rows], **i32)
seqused_k = torch.tensor([key_length], **i32)

indices, values = run(
    q, k, w, descale_q, descale_k,
    candidate_ids, candidate_length,
    cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
    None, None, None,
    topk, 8, 0, 1, True,
)
torch.npu.synchronize()
print(indices.shape, indices.dtype)
print(values.shape, values.dtype)
PY
```

```text
torch.Size([3, 1, 16]) torch.int32
torch.Size([3, 1, 16]) torch.bfloat16
```
