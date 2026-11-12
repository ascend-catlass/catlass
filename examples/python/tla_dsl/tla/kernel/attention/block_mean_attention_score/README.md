# BlockMeanAttentionScore（块平均注意力评分）端到端示例

## 功能说明

基于 CATLASS Python，计算有效 token 的注意力概率，并输出每对 16-token Query/Key 块的平均概率。Q、K 使用相同的有效长度：每个块的前 `counts[b]` 个 token 有效，其余位置不参与计算。

令 `R = S // 16`，`I_b` 为第 `b` 个块中的有效 token 索引，则：

$$
P_{h,q,k} = \operatorname{softmax}_{k\in\cup_b I_b}
\left(\frac{Q_{h,q}\cdot K_{h,k}}{\sqrt D}\right),\qquad
O_{h,i,j} = \frac{\sum_{q\in I_i}\sum_{k\in I_j} P_{h,q,k}}
{|I_i|\,|I_j|}.
$$

任一侧块为空时，对应输出为 NaN。有效 logits 中的 NaN、正无穷及全负无穷按 PyTorch Softmax 语义传播；无效位置的 NaN/Inf 被屏蔽。

支持产品型号：Ascend 950PR&950DT系列产品。

## 代码组织

```text
block_mean_attention_score/
├── block_mean_attention_score.py  # 输入 bundle、编译启动、API 与命令行
├── kernel.py                      # Cube/Vector 流水和任务划分
├── stage_qk.py                    # K @ Q.T、驻留 Query 与分片搬运
├── stage_vector.py                # 掩码、在线统计、延迟归一化与原生 mutex 同步
├── golden.py                      # 分块 CPU FP32 golden 与 BF16 校验
├── constants.py                   # 固定算法块大小与整数对齐工具，不依赖 DSL
└── README.md
```

入口参考相邻 `flash_attention_infer` 的组织方式：`_build_inputs` 保留 CPU 原始输入供 golden 使用，将 NPU Q/K 展平为 `[H*S,D]`；`_compile_launch` 编译并绑定缓冲；`run` 执行一次并校验。公共参数和启动核数工具复用 `examples/python/tla_dsl/helpers`。

`constants.BLOCK_TOKENS=16` 定义固定的算法块大小，由入口、CPU golden 和 kernel 共用。块索引移位数、块内索引掩码和 counts 取值数由它派生。SPLIT_N 的 Query 对齐使用 `QUERY_HALF_ALIGNMENT_TOKENS`，BF16 输出行对齐使用 `OUTPUT_ROW_ALIGNMENT_ELEMS`，谓词存储使用 `PREDICATE_WORDS` 和 `QUERY_MASK_WORDS`；这些布局参数分别定义，不共用算法块常量。

内部按 `224 Key * 256 Query * 256 D` 分块，L0 K 子块为 64。每个 Cube 配合两个 AIV，沿 Query 轴分工；每次只计算一次 QK，保存块指数和与 tile 最大值，再恢复全序列 Softmax 并做两侧块平均。每次调用独立分配 FP32 workspace：

```text
launch_blocks * (S//16 + ceil(S/224)) * 256 * 4 bytes
```

counts 在运行时根据长度选择缓存或流式读取，两条路径位于同一个 kernel 中。counts 的 UB 空间固定按 `UB_CAPACITY_BYTES - vector_fixed_bytes()` 向下对齐到 256 字节分配，包含向量读取所需的保护区；超过缓存容量时分片读取。每个 AIV 的 UB 分配量不超过 248 KiB。S 和 counts 内容不参与编译特化。

块指数和通过完整的 64-lane FP32 视图写回 staging UB，视图覆盖每个已分配物理行的两个分片，省去动态 tile 视图的重复边界掩码计算。

最终块归约先使用原生块内 FP32 归约计算每 8 个 Query 元素之和，再合并相邻两组完成 16-token 块求和，在寄存器中重排输出行，最后统一转换为 BF16 并写入输出 UB。该阶段使用一个 Vector 函数，归约结果无需先写回 staging UB 再通过第二个函数 gather；统计、staging 和输出缓冲仍按原生 mutex 协议交接。

### 静态与动态参数

| 参数 | 编译与运行时行为 |
| --- | --- |
| `H`、`D` | 编译时静态参数；不同 H/D 使用不同特化。 |
| `S` | 运行时动态参数，从 counts 的长度推导为 `counts.shape[0]*16`；仍要求输入 S 为正的 16 的整数倍。 |
| Q/K 与 counts | 张量数据在每次 launch 时读取；Q/K 的序列维、counts 长度以及输出和 workspace 的长度均使用动态 Tensor 元数据。 |
| `block_num` | 运行时启动参数；host 按当前 S、H 和设备可用核数选择，不参与编译缓存键。 |
| 算法块大小、计算 tile、对齐与掩码存储 | 编译时固定配置。 |

在同一设备上，H/D 相同的合法输入复用同一编译产物，包括不同 S、不同 counts 内容和跨越 counts 缓存容量的调用。host 每次按当前 S 分配 BF16 `[H,S//16,S//16]` 输出，并按当前 S 和启动核数分配 workspace。当前接口仍要求 Q/K shape 相同，counts 由两侧及全部 head 共用。

## 样例参数说明

| 参数 | 方向 | shape | dtype | 含义 |
| --- | --- | --- | --- | --- |
| `q`、`k` | 输入 | `[H, S, D]` | BF16 | Query、Key |
| `variable_block_size_16` | 输入 | `[R]` | INT32 | Q/K 共用的块内有效 token 数，取值 `[0,16]` |
| `block_num` | 属性 | 标量 | int | 默认 `-1` 自动调度，或指定启用的 Cube Core 数 |
| 返回值 | 输出 | `[H, R, R]` | BF16 | Query 块为行、Key 块为列的平均概率 |

## 使用范围说明

推荐范围：

令 `R=S//16`，`C` 为当前 NPU 可用的 Cube Core 数，由 `get_block_num(-1, device, kind="mix")` 查询。公开 NPU 接口的约束如下：

| 参数 | 合法范围与约束 |
| --- | --- |
| `H` | 正整数，`H>=1`。 |
| `S` | `S>=16` 且 `S%16==0`；支持 224-token Key tile 和 256-token Query tile 的尾块。 |
| `D` | 正整数，`D>=1`；支持特征维度尾块，无额外的 16/64/128 对齐要求。 |
| `q`、`k` | shape 完全相同的 BF16 `[H,S,D]`，连续存储，位于同一 NPU。接口没有额外设置元素值域或有限性检查；非有限值的传播与掩码语义见[功能说明](#功能说明)。 |
| `variable_block_size_16` | INT32 `[R]`，连续且与 Q/K 位于同一 NPU；每个元素为 `[0,16]` 内的整数，表示该块的有效前缀长度，Q/K 及全部 head 共用。允许部分块或所有块为 0。 |
| `block_num` | 整数 `-1`，或 `1<=block_num<=C`，不接受布尔值。`-1` 自动取 `min(C,H*ceil(S/256))`；显式指定时允许部分启动组没有 Query 任务。 |
| 返回值 | BF16 `[H,R,R]`；Query/Key 任一侧块为空时，对应元素为 NaN。 |

当前代码未设置 H/S/D 的固定最大值，也未定义经过系统验证的最大 shape 规格或覆盖全部 BF16 数值的精度保证范围。实际可运行尺寸受设备内存、框架与底层实现能力限制。Q/K 合计占用 `4*H*S*D` 字节，输出占用 `2*H*(S//16)**2` 字节，此外还需 counts、workspace、框架及 CPU golden 的内存。扩展尺寸或数值区间时需要单独验证。

Cube 使用 BF16 乘法和 FP32 累加，Vector 使用 FP32 归一化，最终输出转换为 BF16。CPU golden 保持 FP32；校验保留原有 `atol=1e-7, rtol=1e-4`，仅为最终 BF16 转换增加逐元素半 ULP 误差界，并严格检查 NaN/Inf 位置。

## 使用示例

先按[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)安装 CATLASS Python，并加载 CANN 环境。

### 命令行参数

```text
block_mean_attention_score.py [-h] [--device DEVICE]
                              [--headnum HEADNUM] [--seqlen SEQLEN]
                              [--head-dim HEAD_DIM]
                              [--pattern {full,partial,mixed,empty,first_tile_empty}]
                              [--block-num BLOCK_NUM] [--seed SEED]
                              [--amplitude AMPLITUDE] [--sentinel SENTINEL]
                              [--query-chunk-size QUERY_CHUNK_SIZE]
                              [--poison-padding] [--cases] [--perf-only]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `0` | 当前可见设备的有效逻辑 NPU 编号，满足 `0<=device<torch.npu.device_count()`。 |
| `--headnum` / `--h` | `2` | Query/Key 的 head 数 `H`，须为正整数。 |
| `--seqlen` / `--s` | `512` | 序列 token 数 `S`，须为正的 16 的整数倍。 |
| `--head-dim` / `--d` | `128` | 每个 head 的特征维度 `D`，须为正整数，支持尾块。 |
| `--pattern` | `mixed` | counts 模式，可选 `full`、`partial`、`mixed`、`empty`、`first_tile_empty`；最后一项在单场景下要求 `S>=272`。 |
| `--block-num` | `-1` | `-1` 时取设备 Cube Core 数与 Query 任务数的较小值；也可指定不超过设备 Cube Core 数的正整数。 |
| `--seed` | `2718` | Q/K 及 `partial` counts 的整数随机种子，PyTorch 接受闭区间 `[-2^63,2^64-1]`。 |
| `--amplitude` | `1.0` | Q/K 正态随机输入的缩放幅度，须为有限非负数。 |
| `--sentinel` | `-7.0` | kernel 启动前填充输出的哨兵值，须为有限负数，用于检查输出覆盖。 |
| `--query-chunk-size` | `256` | CPU golden 每次处理的 Query token 数，须为正的 16 的整数倍，可大于 S；只控制参考实现的内存开销，不改变全序列 Softmax。 |
| `--poison-padding` | `False` | 将无效 Q/K 分别置为 NaN/Inf，验证无效位置的掩码。 |
| `--cases` | `False` | 使用内置的 H/S/D 和 pattern 运行固定的 12 个边界场景；与 `--perf-only` 互斥。 |
| `--perf-only` | `False` | 仅在 MSTX range 内启动一次完整 kernel，不预热、不执行 golden，输出 JSON 中 `passed` 为 `null`。 |
| `-h` / `--help` | — | 打印命令行帮助并退出；`--h` 是 `--headnum` 的别名。 |

`--amplitude` 和 `--sentinel` 的范围检查分别为有限非负数与有限负数；当前 CLI 没有进一步检查 BF16 可表示范围或设定经过精度验证的幅度上限。它们仅用于样例的输入构造与输出预填充。

#### 有效长度模式

五种 `--pattern` 模式仅用于命令行构造 counts。每块的前 `counts[b]` 个 token 有效，该 counts 向量由 Q/K 两侧及全部 head 共用。

| 模式 | counts 的构造方式与含义 |
| --- | --- |
| `full` | 每个块的 counts 均为 16，全部 token 有效。 |
| `partial` | 从均值 8、标准差 4 的正态分布取样，经 `torch.round` 取整（半整数取偶）后截断到 `[0,16]`；每块的有效长度随样本变化。 |
| `mixed` | 块编号 `b` 从 0 开始，先令 `counts[b]=(7*b+3)%(16+1)`，再将 `counts[::3]` 置为 0，最后将末块置为 1。 |
| `empty` | 每个块的 counts 均为 0，整个输出为 NaN。 |
| `first_tile_empty` | 当前 Query tile 为 256 token：前 16 个块的 counts 置为 0，其余块均为 7。单场景要求 `S>256`，结合 16-token 对齐，最小 `S=272`。 |

公开接口也可直接接收任意满足上述形状与 `[0,16]` 取值约束的 counts 向量。`--poison-padding` 在模式生成完成后，将无效 Q/K 分别填为 NaN/Inf，不改变有效前缀长度。

### 执行示例

以下命令从仓库根目录执行：

```bash
# 默认 H=2、S=512、D=128，随机 BF16 Q/K，混合有效长度
python examples/python/tla_dsl/tla/kernel/attention/block_mean_attention_score/block_mean_attention_score.py --device 3

# 典型模型场景；partial counts 由均值 8、标准差 4 的正态分布经 torch.round 取整并截断到 [0,16] 生成
python examples/python/tla_dsl/tla/kernel/attention/block_mean_attention_score/block_mean_attention_score.py \
  --device 3 --headnum 32 --seqlen 11264 --head-dim 128 --pattern partial

# 12 个小型边界用例，包含空块、空序列有效域、尾块与 D 尾部
python examples/python/tla_dsl/tla/kernel/attention/block_mean_attention_score/block_mean_attention_score.py \
  --device 3 --cases --poison-padding

# 单卡上板回归
python -m pytest tests/dsl_battery --run-battery --device 3 -p no:randomly \
  -k block-mean-attention-score
```

每个场景输出一行 JSON，精度校验通过时 `passed` 为 `true`；校验失败时抛出异常。固定 `--seed` 可重建相同输入。

上板回归使用 `--device` 指定的单张 NPU，在 `tests/dsl_battery/test_battery.py` 中登记样例入口；该入口以 `--cases --poison-padding` 运行上述 12 个边界用例，并逐例校验精度。

## 补充说明

### Python 接口

从仓库根目录可直接导入：

```python
from examples.python.tla_dsl.tla.kernel.attention.block_mean_attention_score.block_mean_attention_score import prepare

runner = prepare(H=32, D=128, device="npu:3")

# q1/k1: BF16 [32,512,128]；counts1: INT32 [32]
output1 = runner(q1, k1, counts1)
# q2/k2: BF16 [32,1024,128]；counts2: INT32 [64]
output2 = runner(q2, k2, counts2)
```

`prepare(H,D,device)` 校验静态配置并取得编译产物，返回可调用的 runner，不绑定输入或分配 output/workspace。H/D 须为正整数；device 接受非负 NPU 编号、`"npu:i"` 字符串或 `torch.device`，不带编号的 `"npu"` 使用准备时的当前 NPU。H/D 或设备改变时重新调用 `prepare`。

每次 `runner(q,k,counts,block_num=-1)` 接收本次输入地址和动态 shape，要求 H/D、设备与准备时一致，S 和 counts 满足[使用范围说明](#使用范围说明)。S、counts 内容或输入地址改变时直接再次调用同一个 runner。Q/K/counts 由调用方提供；runner 每次按本次 S 和启动核数独立分配 output 和 workspace，不维护容量池或扩容，后续调用不会覆盖之前返回的 output。`block_num` 是每次调用的可选关键字参数。

每次调用在目标 NPU 的当前 stream 上异步启动；host 校验 counts 值域时会同步。每次调用具有独立的 workspace，可在不同 stream 使用同一个 runner；输入生产与输出消费的跨 stream 依赖由调用方安排。

同一模块仍提供一次性接口，内部复用相同的编译缓存：

```python
from examples.python.tla_dsl.tla.kernel.attention.block_mean_attention_score.block_mean_attention_score import block_mean_attention_score

output = block_mean_attention_score(q, k, counts)
```

### 性能采集

`--perf-only` 在准备输入、workspace 和编译产物后，仅执行一次完整 kernel，应用无预热、无循环，MSTX range 名为 `block_mean_attention_score`。此模式不执行 golden，JSON 中 `passed` 为 `null`；采集前使用相同形状完成精度校验。

```bash
msprof op --aic-metrics=PipeUtilization --warm-up=0 --launch-count=1 \
  --replay-mode=range --mstx=true --mstx-include=block_mean_attention_score \
  --kernel-name=block_mean_attention_score_kernel --kill=false --output=./block_mean_attention_score_profile \
  python examples/python/tla_dsl/tla/kernel/attention/block_mean_attention_score/block_mean_attention_score.py \
    --device 3 --headnum 32 --seqlen 11264 --head-dim 128 --pattern partial --perf-only
```

完整设备调用包含 QK、counts 读取、统计初始化、所有 workspace 读写、归一化和输出。输入生成、H2D、分配、编译及 host 参数校验在采集范围外。读取 profiler 的设备耗时、AIC/AIV 累计 cycles、参与核数和流水利用率；单样本不表示中位数或分位数。
