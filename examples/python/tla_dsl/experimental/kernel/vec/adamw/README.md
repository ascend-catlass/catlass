# AdamW Example Readme

## 功能说明

- 单步更新AdamW的参数、一阶矩和二阶矩，返回三个新张量。通过 `adamw` 模块的 `run` 函数调用。
- 设 $p$、$g$、$m$、$v$ 分别为参数、梯度、一阶矩和二阶矩，$t$ 为当前步数。更新公式如下。

  $$
  \begin{aligned}
  m_t &= \beta_1 m_{t-1} + (1-\beta_1)g, \\
  v_t &= \beta_2 v_{t-1} + (1-\beta_2)g^2, \\
  p_t &= p_{t-1} - \eta\left(
  \frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}
  + \lambda p_{t-1}\right),
  \end{aligned}
  $$

  其中 $\eta$ 为学习率，$\epsilon$ 为分母修正项，$\lambda$ 为权重衰减系数。
- 支持产品型号为Ascend 950PR&950DT系列产品。

## 参数说明

```python
p, m, v = adamw.run(
    param, grad, exp_avg, exp_avg_sq,
    lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `param` | Input | 任意 | `float32` / `float16` / `bfloat16` | 待更新的参数。 |
| `grad` | Input | 同 `param` | 同 `param` | 当前梯度。 |
| `exp_avg` | Input | 同 `param` | `float32` | 更新前的一阶矩。 |
| `exp_avg_sq` | Input | 同 `param` | `float32` | 更新前的二阶矩。 |
| `lr` | Input | 标量 | `float` | 学习率。 |
| `beta1`、`beta2` | Input | 标量 | `float` | 一阶矩和二阶矩的衰减系数。 |
| `eps` | Input | 标量 | `float` | 分母修正项。 |
| `weight_decay` | Input | 标量 | `float` | 权重衰减系数。 |
| `step` | Input | 标量 | `int` | 当前步数，从1开始。 |
| `p` | Output | 同 `param` | 同 `param` | 更新后的参数。 |
| `m`、`v` | Output | 同 `param` | `float32` | 更新后的一阶矩和二阶矩。 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入形状 | 四个输入的形状相同，支持标量和多维张量。空输入返回对应形状和类型的空张量。 |
| 设备与布局 | 非空输入位于同一NPU，按逻辑顺序展平为连续数据。 |
| 编译 | 按设备、元素数、参数类型和编译期系数缓存编译产物。 |

## 使用示例

以下命令在CATLASS仓库根目录执行。

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from adamw.adamw import run

p = torch.randn(4096, dtype=torch.float16, device="npu:0")
g = torch.randn_like(p)
m = torch.zeros(4096, dtype=torch.float32, device=p.device)
v = torch.zeros_like(m)

p2, m2, v2 = run(p, g, m, v, lr=1e-2)
print(p2.dtype, m2.dtype, v2.dtype)
PY
```

```text
torch.float16 torch.float32 torch.float32
```
