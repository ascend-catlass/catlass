# RowTopK Example Readme

## 功能说明

- 对 `[M, N]` 的FP32张量逐行选择前 `k` 个不同的最大值，返回降序排列的值和对应的列下标。`k = 1` 时返回最大值及其下标。通过 `row_topk` 模块的 `run` 函数调用。
- 每轮取当前行的最大值和最左匹配下标，再将行内所有等于该值的元素替换为FP32最小有限值。不同值不足 `k` 个时，剩余轮次继续按替换后的行归约。
- 支持产品型号为Ascend 950PR&950DT系列产品。

## 参数说明

```python
val, idx = row_topk.run(x, k=4)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, N]` | `float32` | 待选择的输入张量。 |
| `k` | Input | 标量 | `int` | 每行的输出项数，默认为1。 |
| `val` | Output | `[M, k]` | `float32` | 每轮选出的值。 |
| `idx` | Output | `[M, k]` | `int32` | 当前轮中对应值的最左列下标，从0开始。 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入维度 | `x` 为二维张量，`64 <= N <= 63360`。`N` 不是64的倍数时，尾块只处理有效列。 |
| 选择项数 | `k` 支持1、2、4、8。 |
| 设备与布局 | 非空输入位于NPU，非连续输入由 `run` 转为行优先连续布局。 |
| 空输入 | `M = 0` 时返回两个形状为 `[0, k]` 的张量。 |
| 编译 | 按输入设备、`M`、`N` 和 `k` 缓存编译产物。 |

## 使用示例

以下命令在CATLASS仓库根目录执行。

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from row_topk.row_topk import run

x = torch.randn(37, 1000, dtype=torch.float32, device="npu:0")
val, idx = run(x, k=4)
print(val.shape, val.dtype, idx.shape, idx.dtype)
PY
```

```text
torch.Size([37, 4]) torch.float32 torch.Size([37, 4]) torch.int32
```
