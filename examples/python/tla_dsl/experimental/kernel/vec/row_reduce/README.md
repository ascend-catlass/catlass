# RowReduce Example Readme

## 功能说明

- 对 `[M, N]` 的FP32张量按行归约，支持 `sum`、`max`、`mean` 和 `logsumexp`，输出形状为 `[M]` 的张量。通过 `row_reduce` 模块的 `run` 函数调用。
- 第 $r$ 行的输出为 $y_r=\operatorname{reduce}(x_{r,0},\ldots,x_{r,N-1})$。其中 `logsumexp` 采用以下数值稳定的计算方式。

  $$
  a_r = \max_j x_{r,j}, \qquad
  y_r = a_r + \log\left(\sum_j e^{x_{r,j}-a_r}\right)
  $$

- 支持产品型号为Ascend 950PR&950DT系列产品。

## 参数说明

```python
y = row_reduce.run(x, op="sum")
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, N]` | `float32` | 待归约的输入张量。 |
| `op` | Input | 标量 | `str` | 归约模式，可选 `sum`、`max`、`mean` 或 `logsumexp`，默认为 `sum`。 |
| `y` | Output | `[M]` | `float32` | 每行的归约结果。 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入维度 | `x` 为二维张量，`N >= 64`。非空输入的列宽上限为63424，由单行补齐后的输入和结果暂存所需的UB容量决定。尾块只归约有效列。 |
| 设备与布局 | 非空输入位于NPU，非连续输入由 `run` 转为行优先连续布局。 |
| 空输入 | `M = 0` 时返回形状为 `[0]` 的张量。 |
| 编译 | 按输入设备、`M`、`N` 和 `op` 缓存编译产物。 |

## 使用示例

以下命令在CATLASS仓库根目录执行。

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from row_reduce.row_reduce import run

m, n = 37, 1000
x = torch.randn(m, n, dtype=torch.float32, device="npu:0")

for op in ("sum", "max", "mean", "logsumexp"):
    y = run(x, op)
    print(op, y.shape, y.dtype)
PY
```

```text
sum torch.Size([37]) torch.float32
max torch.Size([37]) torch.float32
mean torch.Size([37]) torch.float32
logsumexp torch.Size([37]) torch.float32
```
