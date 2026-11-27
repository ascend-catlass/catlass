# TLA 算子样例

本目录按算子类别收录基于 **CATLASS Python** TLA 抽象的端到端算子样例。每个子目录是一个可独立运行的样例，其 README 说明功能、文件组织、可调参数与约束。

## 矩阵乘（matmul）

| 样例 | 概述 |
|------|------|
| [`basic_mmad/`](matmul/basic_mmad/README.md) | 基础矩阵乘，包含 flag 同步、mutex、mutex_with、atomic_add 与指针偏移等多条路径。 |
| [`basic_mixed/`](matmul/basic_mixed/README.md) | Cube + Vector 混合流水，含 mutex、fixpipe NZ→DN、store zN 与 UB→L1 等变体。 |
| [`basic_mmad_epilogue/`](matmul/basic_mmad_epilogue/README.md) | 矩阵乘后处理融合：add、bias、leaky_relu、sigmoid、silu、tanh。 |
| [`basic_mmad_streamk/`](matmul/basic_mmad_streamk/README.md) | StreamK 调度矩阵乘：常规 tile 全 K 计算，尾轮 tile 多核切 K 归约。 |
| [`basic_syrk/`](matmul/basic_syrk/README.md) | 计算输入矩阵与其转置的乘积（SYRK）。 |
| [`batched_matmul/`](matmul/batched_matmul/README.md) | 批量矩阵乘。 |
| [`mm_batchinvariant/`](matmul/mm_batchinvariant/README.md) | 满足 batch 一致性的二维矩阵乘，对齐 `torch.mm`。 |
| [`matmul_batchinvariant/`](matmul/matmul_batchinvariant/README.md) | 满足 batch 一致性的矩阵乘，对齐 `torch.matmul`，支持 1～6 维与 batch 广播。 |
| [`multi_core_splitk_matmul/`](matmul/multi_core_splitk_matmul/README.md) | 多核切 K 矩阵乘，含尾块切分变体。 |
| [`mx_mmad/`](matmul/mx_mmad/README.md) | 微缩放（MX FP8/FP4）矩阵乘，演示 scale 搬运与 `tla.mmad_mx`。 |
| [`fused_ffn/`](matmul/fused_ffn/README.md) | 融合 FFN：`Up Matmul → Act → SyncAll → Down Matmul`（SiLU / GELU / SwiGLU，可选 Cube BT bias）。 |

## 注意力（attention）

| 样例 | 概述 |
|------|------|
| [`flash_attention_infer/`](attention/flash_attention_infer/README.md) | FlashAttention 推理样例，覆盖 KV 分页寻址、causal mask、GQA 与尾块场景。 |
| [`block_mean_attention_score/`](attention/block_mean_attention_score/README.md) | 按 16-token Query/Key 块计算平均注意力概率，支持动态有效长度与尾块。 |
| [`flash_attention_score_batch_invariant/`](attention/flash_attention_score_batch_invariant/README.md) | 训练前向 FlashAttentionScore 样例，输出 `softmax_max` / `softmax_sum`，覆盖 BNSD/BSH/TND/BSND 与 sparse_mode 0/1/2/3，保证 batch 一致性。 |

## 卷积（conv）

| 样例 | 概述 |
|------|------|
| [`basic_conv2d/`](conv/basic_conv2d/README.md) | 以隐式 GEMM（implicit GEMM）方式实现的 2D 卷积。 |

## 分组矩阵乘（gmm）

| 样例 | 概述 |
|------|------|
| [`grouped_matmul_slice_m/`](gmm/grouped_matmul_slice_m/README.md) | 按 M 轴切分的分组矩阵乘。 |

## 纯向量（vec）

| 样例 | 概述 |
|------|------|
| [`basic_vadd/`](vec/basic_vadd/README.md) | 不涉及 Cube 的纯 Vector 算子：向量加法，含 flag 同步、mutex、atomic_add 变体。 |
| [`reduce_mean_batch_invariant/`](vec/reduce_mean_batch_invariant/README.md) | 纯 Vector 归约：`[batch, m, n]` 沿 `axis ∈ {1, 2}` 求均值，覆盖 `float32` / `float16` / `bfloat16`，输出与 `batch` 划分逐位无关。 |

## 运行方式

在 CATLASS 仓库根目录执行，例如：

```bash
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py --device 0
python examples/python/tla_dsl/tla/kernel/vec/basic_vadd/basic_vadd.py --device 0
```

传入 `--help` 可查看样例支持的全部参数。
