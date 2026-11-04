# GatherTokenLogits Example Readme

## 功能说明

- 算子功能：对词表分片 `hidden @ weightᵀ` 的 logits 做片上 top-k 取回，只把命中的 top-k logits 写回 GM，不落 `[M, V]` 的完整 logits；基于 CATLASS DSL 实现的 AIC/AIV mixed kernel。
- 计算公式：记 `rank` 负责的词表分片起点为 `vocab_start = rank × V`，片上 logits 为

  $$
  z[i, v] = \frac{X[i, :] \cdot W[v, :]}{temperature}
  $$

  输出只保留每行指定的 top-k 词表下标，未命中本分片的槽位写 0：

  $$
  out[i, k] =
  \begin{cases}
  z[i,\ ids[i, k] - vocab_start] & vocab\_start \le ids[i, k] < vocab\_start + V \\
  0 & \text{否则}
  \end{cases}
  $$

  其中 $X$ 为 `[M, H]` 的 hidden，$W$ 为 `[V, H]` 的本分片权重，$ids$ 为 `[M, K]` 的全局词表下标。

- 支持产品型号：Ascend 950PR&950DT 系列产品

## 参数说明

```python
logits = gather_token_logits.run(hidden, weight, topk_ids, temperature, rank)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `hidden` | Input | `[M, H]` | `bfloat16` | 行优先连续张量，`M` 为 token 数，`H` 为 hidden 维度 |
| `weight` | Input | `[V, H]` | `bfloat16` | 本 rank 负责的词表分片，行优先；host 侧会转置并缓存为 `[H, V]` 操作数 |
| `topk_ids` | Input | `[M, K]` | `int64` | 全局词表下标；不属于本分片或为负值的槽位输出 0 |
| `temperature` | Input | 标量 | `float` | logits 缩放温度，必须为正数 |
| `rank` | Input | 标量 | `int` | 当前分片编号，决定 `vocab_start = rank × V` |
| `logits` | Output | `[M, K]` | `float32` | top-k logits；未命中槽位为 0 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `M`、`H`、`V`、`K` 为正整数；`M`、`H`、`V` 可非 tile 整数倍，host 侧自动补零 |
| top-k 宽度 | `K` 建议为 8 的倍数，保证 int32 索引按 32B 对齐搬运；其他取值未验证 |
| 分片语义 | `topk_ids` 为全局词表下标，`vocab_start = rank × V`，`rank` 从 0 开始 |
| dtype | `hidden` 与 `weight` 必须同为 `bfloat16`；输出固定 `float32` |
| 设备 | `hidden`、`weight`、`topk_ids` 位于同一 NPU |
| 编译 | 每个 `(M, H, V, K)` 组合编译一个 kernel，编译产物按 shape 缓存 |

## 使用示例

准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境，导入样例模块后调用 `run`。脚本根据输入编译 kernel、执行 NPU 计算并返回 top-k logits：

```python
import torch
import torch_npu  # noqa: F401

import gather_token_logits

m, h, v, k = 1024, 1024, 4096, 128
hidden = torch.randn(m, h, dtype=torch.bfloat16, device="npu:0")
weight = torch.randn(v, h, dtype=torch.bfloat16, device="npu:0")
topk_ids = torch.randint(0, 4 * v, (m, k), dtype=torch.int64, device="npu:0")

logits = gather_token_logits.run(hidden, weight, topk_ids, temperature=1.0, rank=0)
print(logits.shape, logits.dtype)
```

执行结果如下，说明样例执行成功：

```text
torch.Size([1024, 128]) torch.float32
```

与拆分前的合并算子逐例对比 50 个用例（覆盖 `M=33..512`、`H=128..1024`、`V=512..3000`、`K=8/16/32/64/128/10`、`rank=0..3`、`temperature=0.7..2.0`），top-k logits 逐位一致：

```text
split vs original: 50/50 exact=True, max|dlogits|=0.00e+00
```
