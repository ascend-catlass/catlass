# MatmulSoftmaxStats Example Readme

## 功能说明

- 算子功能：在词表分片的 `hidden @ weightᵀ` 投影上直接完成逐行 softmax 统计，只把每行的最大值与 `exp` 求和写回 GM，不落 `[M, V]` 的完整 logits；基于 CATLASS DSL 实现的 AIC/AIV mixed kernel。
- 计算公式：记本分片的片上 logits 为

  $$
  z[i, v] = \frac{X[i, :] \cdot W[v, :]}{temperature}
  $$

  输出为每个 token 在本分片内的 softmax 统计量：

  $$
  local\_max[i] = \max_{v} z[i, v], \qquad
  exp\_sum[i] = \sum_{v} e^{\,z[i, v] - local\_max[i]}
  $$

  其中 $X$ 为 `[M, H]` 的 hidden，$W$ 为 `[V, H]` 的本分片权重。kernel 先按 V tile 在线维护分段 max / sum，再由归并 kernel 折成本分片的最终统计量；多分片时由调用方合并各 rank 的 `local_max` / `exp_sum`。

- 支持产品型号：Ascend 950PR&950DT 系列产品

## 参数说明

```python
local_max, exp_sum = matmul_softmax_stats.run(hidden, weight, temperature)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `hidden` | Input | `[M, H]` | `bfloat16` | 行优先连续张量，`M` 为 token 数，`H` 为 hidden 维度 |
| `weight` | Input | `[V, H]` | `bfloat16` | 本 rank 负责的词表分片，行优先；host 侧会转置并缓存为 `[H, V]` 操作数 |
| `temperature` | Input | 标量 | `float` | logits 缩放温度，必须为正数 |
| `local_max` | Output | `[M]` | `float32` | 每行 logits 在本分片内的最大值 |
| `exp_sum` | Output | `[M]` | `float32` | 每行 `exp(z - local_max)` 在本分片内的求和 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `M`、`H`、`V` 为正整数；可非 tile 整数倍，host 侧自动补零，补零列不参与 max / sum |
| dtype | `hidden` 与 `weight` 必须同为 `bfloat16`；输出固定 `float32` |
| 设备 | `hidden`、`weight` 位于同一 NPU |
| 数值范围 | `local_max` 为有限值；越界 lane 求 max 时掩 `-3.0e38`、求和时掩 0，空分片不产生 NaN |
| 编译 | 每个 `(M, H, V)` 组合编译一个 main kernel 与一个 merge kernel，编译产物按 shape 缓存 |

## 使用示例

准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境，导入样例模块后调用 `run`。脚本根据输入编译 kernel、执行 NPU 计算并返回两个统计量：

```python
import torch
import torch_npu  # noqa: F401

import matmul_softmax_stats

m, h, v = 1024, 1024, 4096
hidden = torch.randn(m, h, dtype=torch.bfloat16, device="npu:0")
weight = torch.randn(v, h, dtype=torch.bfloat16, device="npu:0")

local_max, exp_sum = matmul_softmax_stats.run(hidden, weight, temperature=1.0)
print(local_max.shape, exp_sum.shape, local_max.dtype)
```

执行结果如下，说明样例执行成功：

```text
torch.Size([1024]) torch.Size([1024]) torch.float32
```

与拆分前的合并算子逐例对比 50 个用例（覆盖 `M=33..512`、`H=128..1024`、`V=512..3000`、`rank=0..3`、`temperature=0.7..2.0`），`local_max` 与 `exp_sum` 逐位一致：

```text
split vs original: 50/50 exact=True, max|dmax|=0.00e+00, max|dsum|=0.00e+00
```
