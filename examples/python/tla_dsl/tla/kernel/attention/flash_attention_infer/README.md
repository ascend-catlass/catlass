# FlashAttentionInfer Example Readme

## 功能说明

- 算子功能：基于CATLASS Python的推理 FlashAttention（chunk-prefill 底对齐语义），支持 causal掩码与全可见两种模式、GQA、`BSND/TND` 输入布局与 PagedAttention。
- 计算公式：

  $$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
  $$

  通过分块（Tiling）与 Online Softmax 避免物化完整注意力分数矩阵，将内存复杂度从$O(N^2)$ 降至 $O(N)$。分块后单个 Q 基块 $Q_i$ 与 K 基块 $K_j$ 的注意力分数矩阵$S_{ij}$（FP32）：

  $$
  S_{ij} = \text{scale} \cdot (Q_i K_j^T) \in \mathbb{R}^{B_r \times B_c}
  $$

  其中 $\text{scale} = 1/\sqrt{d_k}$，$B_r$、$B_c$ 分别为 Q/KV 基块大小（`Q_BLOCK`、`KV_BLOCK`，均适配为 128）。增量更新过程（Online Softmax）：

  $$
  \begin{aligned}
    m_{\text{new}} &= \max(m,\ \text{rowmax}(S_{ij})) \\
    P_{ij} &= e^{S_{ij} - m_{\text{new}}} \\
    O &= O \cdot e^{m - m_{\text{new}}} + P_{ij} \cdot V_j \\
    l &= l \cdot e^{m - m_{\text{new}}} + \text{rowsum}(P_{ij}) \\
    m &= m_{\text{new}}
    \end{aligned}
  $$

  内层循环结束后归一化输出 $O_{\text{final}} = O / l$，结果转回 FP16/BF16 写回 GM。

- 支持产品型号：Ascend 950PR&950DT系列产品。

### Mask 模式

样例包含两个相互独立的 kernel（分别实现、独立编译），host 按 mask 需求选择调用：

- `flash_attention_nomask_kernel`：全可见（`--mask none`），无掩码代码路径；
- `flash_attention_mask_kernel`：causal 掩码（`--mask causal`，默认）。

`causal` 模式的可见性示意（■ = 可见，· = 遮蔽），chunk-prefill 底对齐语义：

```plain
       K V  →                     K V  →
      0 1 2 3 4                  0 1 2 3 4 5 6
Q  0  ■ · · · ·             Q  0  ■ ■ ■ · · ·
   1  ■ ■ · · ·                1  ■ ■ ■ ■ · ·
↓  2  ■ ■ ■ · ·             ↓  2  ■ ■ ■ ■ ■ ·
   3  ■ ■ ■ ■ ·                3  ■ ■ ■ ■ ■ ■
   4  ■ ■ ■ ■ ■
   （q = kv）                   （q < kv，prefix 区双向可见）
```

### PagedAttention

默认开启（`--no-paged` 切回连续 KV 存储）。开启后 host 将 K/V 按页打散进页式cache 并生成随机块表，kernel 侧每 tile 查表进行物理页寻址，避免对连续大块 KV的内存申请。分页与连续路径的计算语义一致。

### 变长输入

host 通过 `--varlen` 支持逐 batch 变长（仅 TND 布局，BSND 为定长语义）：长度以命令行 `--qseqlen/--kvseqlen` 为区间上限逐 batch 随机采样（`[1, 上限]`，固定种子，可复现）；causal 模式逐 batch 保证 `q <= kv`（chunk-prefill 底对齐）；

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `query` | Input | BSND `(B, Sq, H, D)` / TND `(ΣSq, H, D)` | `float16/bfloat16` | Q 序列 |
| `key` | Input | BSND `(B, Sk, Hkv, D)` / TND `(ΣSk, Hkv, D)` / PA `(numBlocks, Hkv, bs, D)` | `float16/bfloat16` | K cache；PA 开启时为页式布局（页大小 = `KV_BLOCK` = 128），按块表物理页寻址 |
| `value` | Input | 同 `key` | `float16/bfloat16` | V cache，布局与 `key` 一致 |
| `attentionOut` | Output | BSND `(B, Sq, H, D)` / TND `(ΣSq, H, D)` | `float16/bfloat16` | 注意力输出，dtype 与输入一致 |

shape 中各简写含义：

| 简写 | 含义 | 对应命令行参数 |
| --- | --- | --- |
| `B` | batch 数 | `--batch` |
| `Sq` | Q 序列长度 | `--qseqlen` |
| `Sk` | KV 序列长度 | `--kvseqlen` |
| `H` | Q 头数（`H` 须为 `Hkv` 的整数倍，GQA） | `--headnum` |
| `Hkv` | KV 头数 | `--kvheadnum` |
| `D` | 头维度，固定 128 | — |
| `ΣSq` / `ΣSk` | 各 batch Q/KV 序列长度之和（TND 变长打包） | — |
| `bs` | 分页块大小（= `KV_BLOCK` = 128） | — |
| `numBlocks` | KV cache 物理页总数（= `B*ceil(Sk / bs)`） | — |

中间计算（S/PV/acc）恒为 FP32。

## 使用范围说明

- 输入输出数据类型支持如下组合，中间计算恒为 FP32：

| 输入 (Q/K/V/O) | 中间计算 (S/PV/acc) |
|:-:|:-:|
| fp16 | f32 |
| bf16 | f32 |

- GQA 约束：`HEAD_NUM` 必须是 `KV_HEAD_NUM` 的整数倍。
- `HEAD_DIM` 固定 128。
- causal 模式下 `Q_SEQ` 须不大于 `KV_SEQ`（chunk-prefill 底对齐语义）。
- 变长输入通过 `--varlen` 支持（仅 TND，实现见"功能说明-变长输入"），采样下界为 1。
- 为避免误差累加和计算超时，建议 Q 与 K/V 的总元素数不超过 `31457280`，超出时减小 batch。

## 使用示例

### 命令行参数

```text
flash_attention_infer.py [-h] [--device DEVICE] [--dtype {fp16,bf16}]
                         [--batch BATCH] [--headnum HEADNUM] [--kvheadnum KVHEADNUM]
                         [--qseqlen QSEQLEN] [--kvseqlen KVSEQLEN]
                         [--block-num BLOCK_NUM] [--sentinel SENTINEL]
                         [--format {BSND,TND}] [--mask {none,causal}]
                         [--paged | --no-paged] [--perf-only]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--dtype` | `"fp16"` | 输入/输出数据类型，中间计算恒为 FP32。该值触发重新编译。 |
| `--batch` | `1` | batch 数。 |
| `--headnum` | `8` | Q 头数，须被 `--kvheadnum` 整除。 |
| `--kvheadnum` | `1` | KV 头数。 |
| `--qseqlen` | `117` | Q 序列长度。 |
| `--kvseqlen` | `512` | KV 序列长度。 |
| `--block-num` | `-1` | 启用的核数，`-1` 表示满核。 |
| `--sentinel` | `-7.0` | O 的初始值，用于检测 kernel 是否真正写入。 |
| `--format` | `"BSND"` | 输入张量布局，可选 `"BSND"`（定长）或 `"TND"`（变长打包）。 |
| `--mask` | `"causal"` | 掩码模式：`"causal"` → mask kernel，`"none"` → nomask kernel。 |
| `--paged` / `--no-paged` | 开启 | PagedAttention 默认开启，`--no-paged` 切回连续 KV。 |
| `--varlen` | 关闭 | 变长模式：以 `qseqlen`/`kvseqlen` 为区间上限逐 batch 随机采样长度（`[1, 上限]`，固定种子可复现）；仅支持 TND；causal 逐 batch 保证 `q <= kv`。 |
| `--perf-only` | 关 | 跳过精度校验（性能采集用）。 |

### 执行示例

1. 部署 CATLASS Python 环境后，进入 `python/tla_dsl` 目录（可参考[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)）。

2. 执行算子样例程序：

    ```bash
    # 默认：causal + PA（全核、device 0）
    python examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/flash_attention_infer.py

    # 指定 NPU ID 以及核数
    python examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/flash_attention_infer.py --block-num 1 --device 1

    # 覆盖形状及数据类型, 含 mask / PagedAttention / TND
    python examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/flash_attention_infer.py \
      --batch 1 --qseqlen 5678 --kvseqlen 10000 --headnum 8 --kvheadnum 1 --dtype bf16 \
      --mask causal --format TND
    ```

3. 执行结果如下，说明样例执行成功，精度通过（`passed=True`）：

    ```text
    --- BATCH=(1,117,512) HEAD=(8,1) HEAD_DIM=128 dtype=fp16 mask=causal sentinel=-7.0 ---
    host=torch_npu BATCH=1 Q_SEQ=117 KV_SEQ=512 HEAD_NUM=8 KV_HEAD_NUM=1 HEAD_DIM=128
    O unchanged (sentinel)? False changed_count=... / ...
    passed=True
    ```
