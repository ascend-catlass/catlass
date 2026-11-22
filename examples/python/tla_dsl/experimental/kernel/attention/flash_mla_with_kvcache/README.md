# FlashMlaWithKvcache Example Readme

## 功能说明

- 算子功能：基于 CATLASS Python 的 Flash MLA（Multi-head Latent Attention）解码注意力，KV cache 仅保存单头 576 维 latent，支持两种分页 KV cache 物理布局、因果/非因果掩码以及变长（TND）输入；入口函数为 `flash_mla_with_kvcache`。
- 计算公式：

  $$
  \text{MLA}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{qk}}}\right) V
  $$

  其中 $Q \in \mathbb{R}^{S_q \times H \times d_{qk}}$，KV cache 共享一份 latent 头（$d_{qk} = 576$，$d_v = 512$）：$K$ 取 latent 的全部 576 维，$V$ 取前 512 维；缩放因子由 `softmax_scale` 给出，未指定时取 `1.0`，通常传入 $1/\sqrt{d_{qk}} = 1/\sqrt{576}$。

- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cu_seqlens_q,
    cache_seqlens,
    softmax_scale=None,
    mask_mode=0,
    cache_layout="PA_BBND",
)
```

返回 `(attention_out, lse)`。shape 简写：`B` 为 batch 数，`S_q` 为单请求 Q 序列长度，`kv_len` 为单请求 KV 长度，`ΣS_q` 为各请求 Q 长度之和，`H` 为 Q 头数，`d_qk` 为 QK 维度（576），`d_v` 为 V 维度（512），`num_blocks` 为 KV cache 物理页总数，`num_pages_per_req` 为块表列数。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | Input | TND `(ΣS_q, H, 576)` | `float16/bfloat16` | Q 序列，`H` 取 64 或 96，需 contiguous |
| `k_cache` | Input | `PA_BBND`：`(num_blocks, 128, 1, 576)`；`PA_NZ`：`(num_blocks, 1, 36, 128, 16)` | 同 `q` | 单头 KV latent cache，按块表寻址 |
| `block_table` | Input | `(B, num_pages_per_req)` | `int32` | 每个请求的物理页号，需 contiguous |
| `cu_seqlens_q` | Input | `(B + 1,)` | `int32` | Q 序列前缀和，末元素等于 `ΣS_q`，需 contiguous |
| `cache_seqlens` | Input | `(B,)` | `int32` | 每个请求的 KV 长度，需 contiguous |
| `softmax_scale` | Input | 标量 | `float32` | softmax 缩放，默认 `None` 等价于 `1.0` |
| `mask_mode` | Input | 标量 | `int32` | `0` 全可见（默认），`3` right-down causal |
| `cache_layout` | Input | 标量 | `str` | `"PA_BBND"`（默认）或 `"PA_NZ"` |
| `attention_out` | Output | NTD `(H, ΣS_q, 512)` | 同 `q` | 注意力输出 |
| `lse` | Output | `(H, ΣS_q)` | `float32` | 每行 `max + log(sum)`，无效行为 `-inf` |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 维度 | `q` 末维固定 576，`attention_out` 末维固定 512；KV 头数固定 1 |
| 头数 | `H` 只支持 64 与 96 两档 |
| dtype | `q` 与 `k_cache` 只能为 `float16` 或 `bfloat16` 且需一致；`attention_out` 同 `q`；`lse` 恒为 FP32，中间计算（S/PV/acc）恒为 FP32 |
| 内存布局 | `q`、`block_table`、`cu_seqlens_q`、`cache_seqlens` 必须 contiguous |
| 页表 | PagedAttention 强制开启，页大小固定 128 token；`block_table` 为 `int32`；`cache_seqlens` 最大值不得超过 `block_table.shape[1] * 128` |
| 序列长度 | `cu_seqlens_q` 必须是前缀和且末元素等于 `q.shape[0]`；`cu_seqlens_q` 长度 `B+1`、`cache_seqlens` 长度 `B` |
| 掩码 | `mask_mode` 只支持 `0` 与 `3`；`3` 要求每个请求满足 `kv_len >= q_len` |
| cache_layout | 只支持 `"PA_BBND"` 与 `"PA_NZ"`，两种排布计算语义一致 |
| 边界行为 | 请求 `q_len <= 0` 或 `kv_len <= 0` 时对应输出保持 0、`lse` 为 `-inf`；`ΣS_q == 0` 或 `B == 0` 时直接返回空结果 |
| 返回值 | `lse` 为 `(H, ΣS_q)` 的转置视图（底层按 TND 写入），非 contiguous；需要打包内存时可自行调用 `.contiguous()` |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/attention:$PYTHONPATH python - <<'PY'
import torch
import torch_npu
from flash_mla_with_kvcache.flash_mla_with_kvcache import flash_mla_with_kvcache

torch.npu.set_device(0)
batch, total_q, kv_len, heads = 1, 8, 512, 64
pages = batch * ((kv_len + 127) // 128)
q = torch.randn(total_q, heads, 576, dtype=torch.float16, device="npu")
k_cache = torch.randn(pages, 128, 1, 576, dtype=torch.float16, device="npu")
block_table = torch.arange(pages, dtype=torch.int32, device="npu").unsqueeze(0)
cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device="npu")
cache_seqlens = torch.tensor([kv_len], dtype=torch.int32, device="npu")
with torch.inference_mode():
    attention_out, lse = flash_mla_with_kvcache(
        q, k_cache, block_table, cu_seqlens_q, cache_seqlens,
        softmax_scale=1.0 / 576 ** 0.5, mask_mode=0, cache_layout="PA_BBND",
    )
torch.npu.synchronize()
PY
```
