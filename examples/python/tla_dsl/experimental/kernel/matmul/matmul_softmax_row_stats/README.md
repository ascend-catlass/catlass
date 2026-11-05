# MatmulSoftmaxRowStats Example Readme

## 功能说明

- 算子功能：对 `x @ wᵀ` 的结果直接完成逐行 softmax 统计，只把每行的最大值与 `exp` 求和写回 GM，不落 `[M, V]` 的中间结果；基于 CATLASS DSL 实现的 AIC/AIV mixed kernel。
- 计算公式：记矩阵乘结果为

  $$
  z[i, n] = \frac{X[i, :] \cdot W[n, :]}{temperature}
  $$

  输出为逐行 softmax 统计量：

  $$
  row\_max[i] = \max_{n} z[i, n], \qquad
  row\_sum[i] = \sum_{n} e^{\,z[i, n] - row\_max[i]}
  $$

  其中 $X$ 为 `[M, H]`、$W$ 为 `[V, H]`，分别对应入参 `x` 与 `w`。kernel 按 V tile 在线维护分段 max / sum，再由归并 kernel 合并为最终统计量。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 参数说明

```python
row_max, row_sum = matmul_softmax_row_stats.run(x, w, temperature)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, H]` | `bfloat16` | 左操作数，行优先连续张量；`M` 为行数，`H` 为内积维度 |
| `w` | Input | `[V, H]` | `bfloat16` | 右操作数，行优先；host 侧会转置并缓存为 `[H, V]` 形式 |
| `temperature` | Input | 标量 | `float` | 结果缩放温度，必须为正数 |
| `row_max` | Output | `[M]` | `float32` | 每行的最大值 |
| `row_sum` | Output | `[M]` | `float32` | 每行 `exp(z - row_max)` 的求和 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `M`、`H`、`V` 为正整数；可非 tile 整数倍，host 侧自动补零，补零列不参与 max / sum |
| dtype | `x` 与 `w` 必须同为 `bfloat16`；输出固定 `float32` |
| 设备 | `x`、`w` 位于同一 NPU |
| 数值范围 | `row_max` 为有限值；越界 lane 求 max 时掩 `-3.4e38`、求和时掩 0，空分片不产生 NaN |
| 编译 | 每个 `(M, H, V)` 组合编译一个 main kernel 与一个 merge kernel，产物按 shape 缓存；`V` 分段数 `NV` 由 shape 与核数推导后一并参与编译 |

## 使用示例

准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境，导入样例模块后调用 `run`。脚本根据输入编译 kernel、执行 NPU 计算并返回两个统计量：

```python
import torch
import torch_npu  # noqa: F401

import matmul_softmax_row_stats

m, h, v = 1024, 1024, 4096
x = torch.randn(m, h, dtype=torch.bfloat16, device="npu:0")
w = torch.randn(v, h, dtype=torch.bfloat16, device="npu:0")

row_max, row_sum = matmul_softmax_row_stats.run(x, w, temperature=1.0)
print(row_max.shape, row_sum.shape, row_max.dtype)
```

执行结果如下，说明样例执行成功：

```text
torch.Size([1024]) torch.Size([1024]) torch.float32
```
