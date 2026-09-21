# FlashAttentionInfer端到端示例

本目录下的样例演示 **CATLASS DSL** 下基于 Ascend950 的推理 FlashAttention 算子实现，对齐 C++ 参考实现 `examples/70_ascend950_flash_attention_chunk_prefill`。

## 功能说明

FlashAttention 算子通过分块（Tiling）策略和 Online Softmax 技术，避免物化完整的 $N \times N$ 注意力分数矩阵，将内存复杂度从 $O(N^2)$ 降至 $O(N)$，实现等价的 attention 计算。计算公式为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

分块后单个 Q 基块 $Q_i$ 与 K 基块 $K_j$ 的注意力分数矩阵 $S_{ij}$（FP32）：

$$
S_{ij} = \text{scale} \cdot (Q_i K_j^T) \in \mathbb{R}^{B_r \times B_c}
$$

其中 $\text{scale} = 1/\sqrt{d_k}$，$B_r$、$B_c$ 分别为 Q/KV 基块大小（`Q_BLOCK`、`KV_BLOCK`，均适配为 128）。

增量更新过程（Online Softmax）：

$$
m_{\text{new}} = \max(m,\ \text{rowmax}(S_{ij}))
$$

$$
P_{ij} = e^{S_{ij} - m_{\text{new}}}
$$

$$
O = O \cdot e^{m - m_{\text{new}}} + P_{ij} \cdot V_j
$$

$$
l = l \cdot e^{m - m_{\text{new}}} + \text{rowsum}(P_{ij})
$$

$$
m = m_{\text{new}}
$$

内层循环结束后归一化输出 $O_{\text{final}} = O / l$，结果转回 FP16/BF16 写回 GM。

### Mask 模式

支持两种 mask 模式（`--mask` 参数选择）：

| 模式 | 说明 |
|------|------|
| `none` | 全可见（无 mask） |
| `causal` | 上三角因果 mask |

`causal` 模式的可见性示意（■ = 可见，· = 遮蔽）：

```plain
       K V  →
      0 1 2 3 4
Q  0  ■ · · · ·
   1  ■ ■ · · ·
↓  2  ■ ■ ■ · ·
   3  ■ ■ ■ ■ ·
   4  ■ ■ ■ ■ ■
```

当 `qseqlen < kvseqlen` 时，Q 按右下对齐的 causal 对角线对应：

```plain
       K V  →
      0 1 2 3 4
Q  0  ■ ■ · · ·
   1  ■ ■ ■ · ·
↓  2  ■ ■ ■ ■ ·
   3  ■ ■ ■ ■ ■
```

### PagedAttention

支持 KV cache 分页（`--paged` 开启）。开启后 host 自动将 K/V 按页打散并生成随机块表，kernel 侧按块表进行物理页寻址，避免对连续大块 KV 的内存申请。

## 代码组织

本目录组织结构如下所示：

```plain
./flash_attention_infer
├── flash_attention_infer.py     # @tla.kernel + Host：构造输入、编译、调用 kernel、精度校验
├── fa_tiling.py                 # Tiling 参数计算与打包
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`flash_attention_infer.py`**](flash_attention_infer.py) | 设备侧 `@tla.kernel` 与 host 侧运行/校验逻辑同文件。编译期 shape 参数集中于文件顶部，CLI 可覆盖，kernel 在 `tla.compile` trace 时读取最新值。 |
| [**`fa_tiling.py`**](fa_tiling.py) | Tilingdata 计算与打包。 |

## 约束说明

- 输入输出数据类型支持如下组合，中间计算恒为 FP32：

| 输入 (Q/K/V/O) | 中间计算 (S/PV/acc) |
|:-:|:-:|
| fp16 | f32 |
| bf16 | f32 |

- GQA 约束：`HEAD_NUM` 必须是 `KV_HEAD_NUM` 的整数倍。

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../docs/zh/quick_start.md)完成部署。

### 命令行参数

```text
flash_attention_infer.py [-h] [--device DEVICE] [--dtype {fp16,bf16}]
                         [--batch BATCH] [--headnum HEADNUM] [--kvheadnum KVHEADNUM]
                         [--qseqlen QSEQLEN] [--kvseqlen KVSEQLEN]
                         [--block-num BLOCK_NUM] [--sentinel SENTINEL]
                         [--format {BSND,TND}] [--mask {none,causal}] [--paged]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--dtype` | `"fp16"` | 输入/输出数据类型，可选 `"fp16"` 或 `"bf16"`，中间计算恒为 FP32。该值触发重新编译。 |
| `--batch` | `1` | batch 数，覆盖编译期 `BATCH`。该值触发重新编译。 |
| `--headnum` | `8` | Q 头数，覆盖 `HEAD_NUM`，须被 `--kvheadnum` 整除。 |
| `--kvheadnum` | `1` | KV 头数，覆盖 `KV_HEAD_NUM`。 |
| `--qseqlen` | `117` | Q 序列长度，覆盖 `Q_SEQ`。 |
| `--kvseqlen` | `512` | KV 序列长度，覆盖 `KV_SEQ`。 |
| `--block-num` | `-1` | 启用的核数，`-1` 表示自动探测可用核数（满核）。 |
| `--sentinel` | `-7.0` | O 的初始值，用于检测 kernel 是否真正写入。 |
| `--format` | `"BSND"` | 输入张量布局格式，可选 `"BSND"` 或 `"TND"`。 |
| `--mask` | `"causal"` | mask 模式，可选 `"none"`（全可见）或 `"causal"`（上三角因果）。 |
| `--paged` | 关闭 | 开启 PagedAttention（KV cache 分页）。 |

### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# 默认参数（全核、device 0）
python examples/end_to_end/flash_attention_infer/flash_attention_infer.py

# 指定 NPU ID 以及核数
python examples/end_to_end/flash_attention_infer/flash_attention_infer.py --block-num 1 --device 1

# 覆盖编译期形状及数据类型, 含 mask / PagedAttention / TND
python examples/end_to_end/flash_attention_infer/flash_attention_infer.py \
  --batch 1 --qseqlen 5678 --kvseqlen 10000 --headnum 8 --kvheadnum 1 --dtype bf16 \
  --mask causal --paged --format TND
```

执行测试后，预期输出：
```plain
--- BATCH=(1,117,512) HEAD=(8,1) HEAD_DIM=128 dtype=fp16 mask=causal sentinel=-7.0 ---
host=torch_npu BATCH=1 Q_SEQ=117 KV_SEQ=512 HEAD_NUM=8 KV_HEAD_NUM=1 HEAD_DIM=128
O unchanged (sentinel)? False changed_count=... / ...
passed=True
```

其中 `passed` 结果为 `True` 或 `False` 表明 NPU 计算结果与精度校验是否通过。
