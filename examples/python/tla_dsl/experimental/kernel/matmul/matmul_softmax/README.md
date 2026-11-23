# MatmulSoftmax Example Readme

## 功能说明

- 算子功能：完成 `y = softmax(x @ wᵀ)`，即先做矩阵乘，再对每行做 softmax，输出 `[M, V]` 的 `bfloat16` 结果；基于 CATLASS DSL 实现的 AIC/AIV mixed kernel，入口为 `matmul_softmax` 模块的 `run` 函数。
- 计算公式：记矩阵乘结果为

  $$
  z[i, n] = X[i, :] \cdot W[n, :]
  $$

  输出为逐行 softmax：

  $$
  row\_max[i] = \max_{n} z[i, n], \qquad
  y[i, n] = \frac{e^{\,z[i, n] - row\_max[i]}}{\sum_{k} e^{\,z[i, k] - row\_max[i]}}
  $$

  其中 $X$ 为 `[M, H]`、$W$ 为 `[V, H]`，分别对应入参 `x` 与 `w`。实现分三步：main kernel 按 V 分段完成 `z` 的分块矩阵乘，把 fp32 结果写回 workspace，同时在线维护每个分段的部分 max / exp 求和；merge kernel 合并分段统计量得到逐行 `row_max` 与归一化分母；normalize kernel 读取 workspace 计算 `exp(z - row_max) / denominator` 并转为 `bfloat16` 输出。normalize 之前需要一块 `[M_pad, V_pad]` 的 fp32 workspace 暂存 `z`，额外显存为 `4 * M_pad * V_pad` 字节。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
y = matmul_softmax.run(x, w)
```

`M` 为行数，`H` 为内积维度，`V` 为 `w` 的行数（即逻辑输出列数）；`M_pad`、`V_pad` 分别为 `M`、`V` 向上补齐到 256 的结果。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, H]` | `bfloat16` | 左操作数，行优先连续张量；`M` 为行数，`H` 为内积维度 |
| `w` | Input | `[V, H]` | `bfloat16` | 右操作数，行优先；host 侧会转置并缓存为 `[H, V]` 形式 |
| `y` | Output | `[M, V]` | `bfloat16` | softmax 结果，取自 `[M_pad, V_pad]` 输出缓冲区的左上角视图，`V` 非 256 整数倍时可能不连续 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `M`、`H`、`V` 为正整数；可非 tile 整数倍，host 侧自动补零，补零列不参与 max / exp 求和 |
| dtype | `x` 与 `w` 必须同为 `bfloat16`；输出固定 `bfloat16` |
| 设备 | `x`、`w` 与输出位于同一 NPU |
| 输出缓冲区 | 内部按 `[M_pad, V_pad]` 分配，返回 `[:M, :V]` 视图；需要连续结果时调用方自行 `contiguous()` |
| 数值范围 | 每行以 `row_max` 做减法和归一化，`exp` 输入不超过 0；越界 lane 求 max 时掩 `-3.4e38`、求和时掩 0，补零行不产生 NaN |
| 编译 | 每个 `(M, H, V)` 组合编译一个 main kernel、一个 merge kernel 与一个 normalize kernel，产物按 shape 缓存；`V` 分段数 `NV` 由 shape 与核数推导后一并参与编译 |

## 使用示例

导入样例模块后调用 `run`，脚本根据输入编译 kernel、执行 NPU 计算并返回 softmax 结果：

```python
import torch
import torch_npu  # noqa: F401

from matmul_softmax.matmul_softmax import run

m, h, v = 1024, 1024, 4096
x = torch.randn(m, h, dtype=torch.bfloat16, device="npu:0")
w = torch.randn(v, h, dtype=torch.bfloat16, device="npu:0")

y = run(x, w)
print(y.shape, y.dtype)
```

执行结果如下，说明样例执行成功：

```text
torch.Size([1024, 4096]) torch.bfloat16
```
