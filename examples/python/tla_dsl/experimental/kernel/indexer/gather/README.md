# Gather Example Readme

## 功能说明

- 算子功能：按 `[M, K]` 的索引从 `[M, V]` 的 `x` 中逐行取回元素，写回 `[M, K]`；基于 CATLASS DSL 实现的 AIV SIMT kernel。
- 计算公式：

  $$
  out[i, k] =
  \begin{cases}
  x[i,\ index[i, k]] & 0 \le index[i, k] < V \\
  0 & \text{否则}
  \end{cases}
  $$

  其中 `index` 为 `[M, K]` 的列下标，负值表示该槽位不使用。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 参数说明

```python
out = gather.run(x, index)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, V]` | `float32` | 行优先连续张量 |
| `index` | Input | `[M, K]` | `int64` | 列下标；负值或越界槽位输出 0 |
| `out` | Output | `[M, K]` | `float32` | 取回的取值；未命中槽位为 0 |

缩放等后处理由调用方在 `[M, K]` 输出上完成，算子只做取回。

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `M`、`V`、`K` 为正整数；`M`、`V` 无需按 tile 补齐 |
| dtype | `x` 必须为 `float32`（样例不做隐式类型转换，避免额外的 `[M, V]` 拷贝）；输出固定 `float32` |
| 设备 | `x`、`index` 位于同一 NPU |
| 编译 | 每个 `(M, V, K)` 组合编译一个 kernel，编译产物按 shape 缓存 |

## 实现说明

- 每个线程负责一行，行内按 `K` 做标量随机读，总访存量为 `O(M * K)` 个 `float32`。
- 不使用 UB 分块搬运，也没有跨核同步：整个算子只有一条 AIV 流水，block 之间按行切分。
- 下标在 host 侧转为 `int32`；kernel 只判断上下界，负值槽位直接跳过。

## 使用示例

准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境，导入样例模块后调用 `run`：

```python
import torch
import torch_npu  # noqa: F401

import gather

m, v, k = 1024, 4096, 128
x = torch.randn(m, v, dtype=torch.float32, device="npu:0")
index = torch.randint(0, v, (m, k), dtype=torch.int64, device="npu:0")

out = gather.run(x, index)
print(out.shape, out.dtype)
```

执行结果如下，说明样例执行成功：

```text
torch.Size([1024, 128]) torch.float32
```
