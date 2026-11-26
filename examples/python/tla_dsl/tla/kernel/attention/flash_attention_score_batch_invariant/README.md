# FlashAttentionScoreBatchInvariant Example Readme

## 功能说明

- 算子功能：训练前向 FlashAttentionScore，输出注意力结果与 Online Softmax 的逐行统计量 `softmax_max` / `softmax_sum`，供反向注意力算子使用；输出与 batch 划分方式无关（batch invariant）。
- 计算公式：标准缩放点积注意力为

  $$
  \begin{aligned}
  S &= \text{scale\_value} \cdot (QK^T) \\
  O &= \text{softmax}(\text{masked\_fill}(S,\ \text{MIN\_VALUE}))\, V
  \end{aligned}
  $$

  其中 `scale_value` 为 softmax 缩放系数（先乘后掩码），`MIN_VALUE` 为掩码填充值，即 FP32 可表示的最小负有限值。

  softmax 的逐行计算流程为：

  $$
  \begin{aligned}
  m_i &= \text{rowmax}(S_i) \\
  P_{ij} &= \exp(S_{ij} - m_i) \\
  l_i &= \text{rowsum}(P_i) \\
  O_i &= (P_i V) / l_i
  \end{aligned}
  $$

  按 `Q_BLOCK` = `KV_BLOCK` = 128 分块后，使用 Online Softmax 增量更新：

  $$
  \begin{aligned}
    m_{\text{new}} &= \max(m,\ \text{rowmax}(S_{ij})) \\
    P_{ij} &= e^{S_{ij} - m_{\text{new}}} \\
    O &= O \cdot e^{m - m_{\text{new}}} + P_{ij} \cdot V_j \\
    l &= l \cdot e^{m - m_{\text{new}}} + \text{rowsum}(P_{ij})
  \end{aligned}
  $$

  内层循环结束后归一化输出 $O_{\text{final}} = O / l$，`m`、`l` 分别写入 `softmax_max`、`softmax_sum` 并写回 GM。每行统计量广播到 8 列，即 `softmax_max[..., k] = m`、`softmax_sum[..., k] = l`（`k = 0..7`），与反向算子接口对齐。

- 支持产品型号：Ascend 950PR&950DT系列产品。

### Mask 模式

样例只有一个 kernel `flash_attention_score_kernel`，`sparse_mode` 作为编译期常量注入，按模式特化生成代码：

- `sparse_mode=0`：全可见，编译结果不含掩码处理代码；
- `sparse_mode=1`（`atten_mask`）、`2`（leftUpCausal）、`3`（rightDownCausal）：编译期打开掩码路径，`1` 从 GM 读取 `[S1, S2]` 掩码，`2`/`3` 在 Vector 侧按行列下标生成掩码。

掩码模式下，TileMask（`tile_range` / `tile_compute` / `fine_mask`）用于跳过整块不可见的 KV 基块。全掩码行（所有 KV 均被遮蔽）输出精确为 0，且 `softmax_max = MIN_VALUE`、`softmax_sum = 0`，无 NaN/Inf。

### Batch 一致性

本样例保证 **batch 一致性（batch invariant）**：同一份数据，无论整批运行还是拆分运行，输出逐位一致。该性质由任务划分方式保证，不依赖特殊调度：

1. 任务以 `(batch, head, Q 基块)` 为单位划分，每行的 max/sum 归约只在本行的 KV 范围内完成，不存在跨 batch 的归一化或共享缓冲；
2. 固定布局下，`B=N` 整批运行与 `N` 次 `B=1` 运行的输出切片逐位一致；
3. TND 变长布局下，打包运行与逐序列拆包运行逐位一致；每个序列只访问自己的 `[q_start, q_end) × [kv_start, kv_end)` 段，因果掩码按序列内部下标对齐。

典型用途：推理/训练框架中要求数值不随 batch size 变化（可复现、可对齐）的场景。

## 样例参数说明

```python
attention_out, softmax_max, softmax_sum = flash_attention_score_batch_invariant.run(
    query, key, value, atten_mask, actual_seq_qlen, actual_seq_kvlen,
    scale_value, head_num, input_layout, sparse_mode,
)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `query` | Input | BNSD `(B, Nq, S1, D)` / BSH `(B, S1, Nq*D)` / TND `(T, Nq, D)` / BSND `(B, S1, Nq, D)` | `float16/bfloat16` | Q 序列 |
| `key` | Input | 同 `query`，头数为 `Nkv` | `float16/bfloat16` | K 序列，`Nq` 须被 `Nkv` 整除（GQA 映射 `kv_head = q_head // (Nq/Nkv)`） |
| `value` | Input | 同 `key` | `float16/bfloat16` | V 序列 |
| `atten_mask` | Input | `(S1, S2)` | `uint8` | 仅 `sparse_mode=1` 有效，其余模式传 `None`；非 0 表示遮蔽 |
| `actual_seq_qlen` | Input | `(B,)` | `int` 列表 | 仅 TND 布局使用：逐 batch Q 长度的前缀和（不含前导 0），如 `[100, 250]` 表示两序列长度 `100`、`150` |
| `actual_seq_kvlen` | Input | `(B,)` | `int` 列表 | 仅 TND 布局使用：同 `actual_seq_qlen` 语义 |
| `scale_value` | Input | 标量 | `float` | softmax 缩放系数，先乘后掩码 |
| `head_num` | Input | 标量 | `int` | query 头数 `Nq`（BSH 布局下由 `Nq*D` 拆出 `D`） |
| `input_layout` | Input | 标量 | `int` | `0`=BNSD，`1`=BSH，`2`=TND，`3`=BSND |
| `sparse_mode` | Input | 标量 | `int` | `0`=不掩码，`1`=使用 `atten_mask`，`2`=leftUpCausal，`3`=rightDownCausal |
| `attention_out` | Output | BNSD `(B, Nq, S1, D)` / BSH `(B, S1, Nq*D)` / TND `(T, Nq, D)` / BSND `(B, S1, Nq, D)` | 同 `query` | 注意力输出，布局与 `query` 一致 |
| `softmax_max` | Output | `(B, Nq, S1, 8)` / TND `(T, Nq, 8)` | `float32` | 每行 Online Softmax 的 max 值，广播到 8 列 |
| `softmax_sum` | Output | 同 `softmax_max` | `float32` | 每行 Online Softmax 的 sum 值，广播到 8 列 |

shape 中各简写含义：

| 简写 | 含义 | 对应命令行参数 |
| --- | --- | --- |
| `B` | batch 数 | `--batch` |
| `S1` | Q 序列长度 | `--qseqlen` |
| `S2` | KV 序列长度 | `--kvseqlen` |
| `Nq` | Q 头数 | `--headnum` |
| `Nkv` | KV 头数 | `--kvheadnum` |
| `D` | 头维度，最大 128 | `--headdim` |
| `T` | 各 batch Q 长度之和（TND 变长打包） | — |

中间结果（S/PV/acc）恒为 FP32。

## 使用范围说明

输入输出数据类型支持如下组合，中间结果（S/PV/acc）与 `softmax_max` / `softmax_sum` 恒为 FP32：

| 输入 (Q/K/V/O) | 中间结果 (S/PV/acc) |
|:-:|:-:|
| fp16 | f32 |
| bf16 | f32 |

- 头维度 `D <= 128`；`D < 128` 时 host 自动零填充到 128（`D` 超过 128 抛出 `NotImplementedError`）。
- GQA：`Nq` 必须是 `Nkv` 的整数倍。
- causal：`sparse_mode=2` 仅保留 `j <= i` 的位置；`sparse_mode=3` 仅保留 `j <= i + (S2 - S1)` 的位置（右下对齐）；TND 下按每序列内部长度对齐。
- 掩码：`sparse_mode=1` 的 `atten_mask` 按 `[S1, S2]` 广播到所有 batch/head，不支持逐 batch 不同掩码。
- 变长：TND 需提供 `actual_seq_qlen` / `actual_seq_kvlen`；变长仅支持 TND，非 TND 布局传变长（或与张量维度不一致）的 `actual_seq_*` 会直接报错；`sparse_mode=1` 不支持 TND。
- 全掩码行：输出精确为 0，`softmax_max = MIN_VALUE`，`softmax_sum = 0`。
- 布局与设备：所有输入张量须位于同一 NPU。

## 使用示例

### 调用示例

CATLASS Python DSL 环境搭建见 [TLA 样例说明](../../../../README.md)。在 CATLASS 仓库根目录下执行：

`run()` 对首次出现的 `(shape, dtype, 布局, 掩码模式)` 组合，用 `tla.compile`（`--npu-arch 3510`）编译对应 kernel；同配置的后续调用命中进程内缓存，不重复编译。

```python
import torch
import torch_npu  # noqa: F401

import sys

sys.path.append("examples/python/tla_dsl/tla/kernel/attention"
                "/flash_attention_score_batch_invariant")
import flash_attention_score_batch_invariant as fa_score

b, s1, s2, nq, nkv, d = 2, 333, 777, 8, 2, 128
query = torch.randn(b, nq, s1, d, dtype=torch.bfloat16).npu()
key = torch.randn(b, nkv, s2, d, dtype=torch.bfloat16).npu()
value = torch.randn(b, nkv, s2, d, dtype=torch.bfloat16).npu()

# sparse_mode=3：右下对齐 causal，BNSD 布局，GQA
out, smax, ssum = fa_score.run(
    query, key, value, None, None, None,
    scale_value=d ** -0.5, head_num=nq, input_layout=0, sparse_mode=3,
)
print(out.shape, smax.shape, ssum.shape)

# batch 一致性：整批运行与逐 batch 运行逐位一致
out1 = fa_score.run(
    query[:1], key[:1], value[:1], None, None, None,
    scale_value=d ** -0.5, head_num=nq, input_layout=0, sparse_mode=3,
)[0]
assert torch.equal(out[0], out1[0])
```

### 命令行参数

```text
flash_attention_score_batch_invariant.py [-h] [--device DEVICE]
                                        [--dtype {fp16,bf16}] [--batch BATCH]
                                        [--qseqlen QSEQLEN] [--kvseqlen KVSEQLEN]
                                        [--headnum HEADNUM] [--kvheadnum KVHEADNUM]
                                        [--headdim HEADDIM]
                                        [--layout {BNSD,BSH,TND,BSND}]
                                        [--sparse {0,1,2,3}] [--seed SEED]
                                        [--skip-batch-check]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--dtype` | `"bf16"` | 输入/输出数据类型，可选 `fp16`、`bf16`，中间结果恒为 FP32。改变该值会触发重新编译。 |
| `--batch` | `2` | batch 数。 |
| `--qseqlen` | `333` | Q 序列长度；TND 下为各 batch Q 长度之和。 |
| `--kvseqlen` | `777` | KV 序列长度；TND 下为各 batch KV 长度之和。 |
| `--headnum` | `8` | Q 头数，须被 `--kvheadnum` 整除。 |
| `--kvheadnum` | `2` | KV 头数。 |
| `--headdim` | `128` | 头维度，最大 128。 |
| `--layout` | `"BNSD"` | 输入张量布局，可选 `BNSD`、`BSH`、`TND`（变长打包）、`BSND`。 |
| `--sparse` | `0` | 掩码模式：`0` 不掩码，`1` 使用 `atten_mask`，`2` leftUpCausal，`3` rightDownCausal。 |
| `--seed` | `0` | 随机种子。 |
| `--skip-batch-check` | 关闭 | 跳过整批与逐 batch 的逐位一致性比对。 |

### 执行示例

在 CATLASS 仓库根目录执行。脚本内置 float32 参考实现、逐 batch 误差输出、按 dtype 的绝对误差阈值判定（`attention_out`：bf16 1.5e-2 / fp16 2e-3；`softmax_max`、`softmax_sum` 均为 FP32，阈值 1e-4 / 5e-3），以及整批/逐 batch 的逐位一致性比对；任一 batch 误差超阈值或一致性不一致时，末行输出 `RESULT FAIL` 且进程返回非 0：

```bash
# 默认：BNSD、sparse_mode=0、bf16、B=2、S1=333、S2=777
python examples/python/tla_dsl/tla/kernel/attention/flash_attention_score_batch_invariant/flash_attention_score_batch_invariant.py --device 0

# 稠密 GM mask（sparse_mode=1）与 TND 变长（sparse_mode=3）
python examples/python/tla_dsl/tla/kernel/attention/flash_attention_score_batch_invariant/flash_attention_score_batch_invariant.py --device 0 --layout BSND --sparse 1
python examples/python/tla_dsl/tla/kernel/attention/flash_attention_score_batch_invariant/flash_attention_score_batch_invariant.py --device 0 --layout TND --sparse 3 --batch 3 --qseqlen 629 --kvseqlen 1341
```

末行输出 `RESULT PASS` 表示样例执行成功、精度与 batch 一致性均通过：

```text
batch 0: out_err=... max_err=... sum_err=... OK
batch 1: out_err=... max_err=... sum_err=... OK
tolerance: out<=1.5e-02 max<=1.0e-04 sum<=5.0e-03
batch_invariance_mismatch=0
layout=BNSD sparse=0 dtype=bf16 shape=(2,333,777,8,128)
RESULT PASS
```
