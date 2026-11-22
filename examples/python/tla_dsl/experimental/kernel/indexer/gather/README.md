# Gather Example Readme

## 功能说明

- 算子功能：按 `[M, K]` 的索引从 `[M, V]` 的 `x` 中逐行取回元素并写回 `[M, K]`；入口为 `gather` 模块的 `run` 函数，基于 CATLASS DSL 实现的 AIV SIMT kernel。
- 计算公式：

  $$
  out[i, k] =
  \begin{cases}
  x[i,\ index[i, k]] & 0 \le index[i, k] < V \\
  0 & \text{否则}
  \end{cases}
  $$

  其中 `index` 为 `[M, K]` 的列下标，负值表示该槽位不使用。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
out = gather.run(x, index)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, V]` | `float32` | 行优先连续张量。 |
| `index` | Input | `[M, K]` | `int64` | 列下标；负值或越界槽位输出 0。 |
| `out` | Output | `[M, K]` | `float32` | 取回的取值；未命中槽位为 0。 |

缩放等后处理由调用方在 `[M, K]` 输出上完成，算子只做取回。

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `M`、`V`、`K` 为正整数；`M`、`V` 无需按 tile 补齐。 |
| dtype | `x` 必须为 `float32`（样例不做隐式类型转换，避免额外的 `[M, V]` 拷贝）；输出固定 `float32`；`index` 在下标转换后按 `int32` 参与计算。 |
| 下标范围 | `index` 中小于 0 或大于等于 `V` 的槽位不写入，输出保持 0。 |
| 设备 | `x`、`index` 位于同一 NPU。 |
| 编译 | 每个 `(M, V, K)` 组合编译一个 kernel，编译产物按 shape 缓存。 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/indexer:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from gather.gather import run

m, v, k = 1024, 4096, 128
x = torch.randn(m, v, dtype=torch.float32, device="npu:0")
index = torch.randint(0, v, (m, k), dtype=torch.int64, device="npu:0")

out = run(x, index)
print(out.shape, out.dtype)
PY
```

```text
torch.Size([1024, 128]) torch.float32
```
