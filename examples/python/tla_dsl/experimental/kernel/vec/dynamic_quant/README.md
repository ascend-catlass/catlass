# DynamicQuant Example Readme

## 功能说明

- 对二维FP32张量按行执行对称INT8动态量化，输出量化编码、行缩放因子和反量化结果。通过 `dynamic_quant` 模块的 `run` 函数调用。
- 计算公式如下。设量化截断上界 `q_max` 为 $Q$，第 $r$ 行的最大绝对值为 $a_r$，缩放因子为 $s_r$。

  $$
  \begin{aligned}
  a_r &= \max_j |x_{r,j}|, \qquad s_r = a_r / Q, \\
  q_{r,j} &= \operatorname{round}_{\mathrm{even}}\!\left(
    \operatorname{clamp}\!\left(x_{r,j}\cdot\frac{Q}{\max(a_r, 10^{-12})}, -Q, Q\right)\right), \\
  \mathrm{deq}_{r,j} &= q_{r,j}\,s_r.
  \end{aligned}
  $$

  量化值舍入到最近整数，半整数舍入到相邻的偶数。全零行的三项输出均为零。
- 支持产品型号为Ascend 950PR&950DT系列产品。

## 参数说明

```python
q, scale, deq = dynamic_quant.run(x, q_max=127.0)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, N]` | `float32` | 每行对应一个 token。 |
| `q_max` | Input | 标量 | `float` | 舍入前的对称截断上界，默认为127.0。 |
| `q` | Output | `[M, N]` | `int8` | 舍入后的量化编码。 |
| `scale` | Output | `[M]` | `float32` | 每行的缩放因子。 |
| `deq` | Output | `[M, N]` | `float32` | 量化编码乘以对应行的缩放因子。 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入 | `x` 为二维FP32张量，非空输入位于NPU。非连续输入由 `run` 转为连续布局。 |
| 列宽 | `N` 为64的倍数。 |
| 量化截断上界 | `0 < q_max <= 127`。 |
| 空输入 | `M = 0` 时返回空张量。`N = 0` 时 `q` 和 `deq` 的形状为 `[M, 0]`，`scale` 各元素为零。 |

## 使用示例

以下命令在CATLASS仓库根目录执行。

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from dynamic_quant.dynamic_quant import run

x = torch.randn(64, 512, dtype=torch.float32, device="npu:0")
q, scale, deq = run(x)
print(q.shape, q.dtype, scale.shape, deq.shape)
PY
```

```text
torch.Size([64, 512]) torch.int8 torch.Size([64]) torch.Size([64, 512])
```
