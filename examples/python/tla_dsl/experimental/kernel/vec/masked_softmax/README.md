# MaskedSoftmax Example Readme

## 功能说明

- 算子功能：对 `[M, N]` 的 FP32 logits 逐行计算数值稳定的 softmax，支持乘性缩放和
  可选加性 mask/bias；基于 CATLASS Python 实现纯 AIV 单 kernel，入口为
  `masked_softmax` 模块的 `run` 函数。
- 计算公式：

  $$
  \begin{aligned}
  z_{r,i} &= \text{scale} \cdot \left(x_{r,i} + \text{add}_{r,i}\right) \\
  y_{r,i} &= \frac{e^{z_{r,i} - \max_j z_{r,j}}}
                   {\sum_j e^{z_{r,j} - \max_j z_{r,j}}}
  \end{aligned}
  $$

  加性 mask 用较大负数关闭指定位置；`add=None` 时跳过加性项。`scale` 与是否存在
  `add` 都是编译期常量，由 host 缓存对应编译产物。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
y = masked_softmax.run(x, add=None, scale=1.0)
```

`M` 为行数，`N` 为每行元素数。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, N]` | `float32` | 输入 logits，行优先连续张量。 |
| `add` | Optional Input | `[M, N]` 或 `None` | `float32` | 加性 bias/mask；`None` 表示不加。 |
| `scale` | Input | 标量 | Python `float` | softmax 前的乘性缩放。 |
| `y` | Output | `[M, N]` | `float32` | 逐行归一化结果，每行有效值之和为 1。 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `x` 必须为二维；`N > 0` 且 `N % 64 == 0`；`M` 可以为 0。 |
| dtype | `x`、`add` 与输出均为 `float32`。 |
| 设备 | 输入与输出位于同一 NPU。 |
| mask 语义 | 通过加性 bias 表达；极大负值位置的 `exp` 可自然下溢为 0，不引入显式 MaskSSA。 |
| UB | 常驻一块 `N` 元素 FP32 行缓冲；仅 `add` 存在时额外使用一块。 |
| 并行度 | 一行由一个 AIV 完整处理，避免跨 block 归约；小 `M` 时并行度受行数限制。 |
| 编译 | 按 `(M, N, scale, has_add)` 缓存 kernel；`scale` 或 mask 分支变化会重新编译。 |
| 硬件 | 编译选项固定 `--npu-arch 3510`，block 数取设备 `vector_core_num`。 |

## 高性能设计

- **行独占与无跨核归约**：block 以 `block_idx + k * block_num` 的方式认领完整行，
  每行的 max 和 sum 在一个 AIV 内闭合，不需要部分统计量或第二次 GM 往返。
- **单次搬入、片上三遍复用**：`x` 只从 GM 搬入一次；缩放+bias、`exp`+求和、归一化
  三遍都在同一 UB 行缓冲上完成，中间结果不落 GM。
- **寄存器驻留归约**：每遍以 64 个 FP32 lane 的向量 tile 展开，max 与 sum 在向量
  寄存器中分块累计，最后只做一次 64-lane 归约。
- **乘法替代除法**：归一化先计算一次 `1/sum`，随后每个元素只做乘法。
- **精确 UB 所有权同步**：`ub_x` 和 `ub_add` 各有一个独立 `tla.mutex`，按 MTE2
  fill、VECTOR consume、MTE3 drain 的顺序在对应 pipe 上 lock/unlock；释放行缓冲
  即授权下一次搬入，避免使用全局 pipe barrier。
- **host 常量复用**：输入 DLPack 视图按 tensor 身份及 pointer/shape/stride/dtype/
  device 签名缓存；缓存项使用 weakref 自动回收，形状切换不会累积常驻显存。

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
python - <<'PY'
import math
import torch
import torch_npu  # noqa: F401

from examples.python.tla_dsl.experimental.kernel.vec.masked_softmax.masked_softmax import run

m, n = 64, 256
x = torch.randn(m, n, dtype=torch.float32, device="npu:0")
add = torch.zeros(m, n, dtype=torch.float32, device="npu:0")
add[:, 200:] = -1.0e9

y = run(x, add, 1.0 / math.sqrt(n))
torch.npu.synchronize()
print(y.shape, y.dtype)
PY
```

执行结果如下，说明样例执行成功：

```text
torch.Size([64, 256]) torch.float32
```

模块也提供独立精度自检：

```bash
python examples/python/tla_dsl/experimental/kernel/vec/masked_softmax/masked_softmax.py \
    --device 0 --m 64 --n 256
```
