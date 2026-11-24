# StreamingMaskedSoftmax Example Readme

## 功能说明

- 算子功能：对 `[M, N]` 的 FP32 logits 逐行计算数值稳定的 softmax，支持乘性缩放和
  可选加性 mask/bias；长行无需整体驻留 UB。
- 计算公式：

  $$
  \begin{aligned}
  z_{r,i} &= \text{scale} \cdot \left(x_{r,i} + \text{add}_{r,i}\right) \\
  y_{r,i} &= \frac{e^{z_{r,i} - \max_j z_{r,j}}}
                   {\sum_j e^{z_{r,j} - \max_j z_{r,j}}}
  \end{aligned}
  $$

  加性 mask 通过较大负值表达；`add=None` 时完全跳过该输入。
- 实现方式：Device 侧使用 CATLASS Python，HOST 侧负责输入检查、DLPack 绑定、编译缓存、
  workspace 管理和 kernel 下发。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
y = streaming_masked_softmax.run(x, add=None, scale=1.0)
```

`M` 为行数，`N` 为每行元素数。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, N]` | `float32` | 输入 logits；HOST 会转换为连续张量。 |
| `add` | Optional Input | `[M, N]` 或 `None` | `float32` | 加性 bias/mask，必须与 `x` 同 shape、同 dtype。 |
| `scale` | Input | 标量 | Python `float` | softmax 前的乘性缩放，必须为有限值。 |
| `y` | Output | `[M, N]` | `float32` | 逐行归一化结果。 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `x` 必须为二维；`N >= 64`；`M` 可以为 0。 |
| dtype | `x`、`add` 与输出均为 `float32`。 |
| 设备 | 输入、输出和 workspace 位于同一 NPU。 |
| mask 语义 | 加性大负值位置的 `exp` 可自然下溢为 0，不引入独立布尔 mask。 |
| `scale` | 有限；允许 0、正数和负数。 |
| 非 32B 对齐 | 支持任意 `N >= 64`，不要求 `N % 8 == 0` 或 `N % 64 == 0`。尾块按 64-lane 分块，最后一次 max/sum 归约使用动态 mask，padding lane 不参与归约。 |
| 编译 | HOST 按 shape、切片、`scale`、`has_add` 和设备缓存编译产物。 |
| 硬件 | 编译目标固定为 `--npu-arch 3510`。 |

## 高性能设计

- **形状自适应路径**：`N <= 2048` 且 64 对齐时整行驻留 UB，`x` 只从 GM 读取一次；
  一般长行按 2048 元素切片维护 online `(max, sum)`；低并行超长行使用单 launch 分核。
- **单 launch 分核**：超长低并行行在同一个 kernel 内依次执行并行 chunk 统计、
  逐行统计归约和并行归一化。阶段之间使用 device-wide SyncAll，避免多次 host launch。
- **两遍数据流**：一般长行第一遍计算缩放后 logits 的 online 统计，第二遍重新读取
  `x`/`add` 并直接写回 `y`，不产生 GM 概率中间张量。
- **mutex 资源所有权**：`ub_x`、`ub_add`、partial statistics 等片上资源分别由独立
  `tla.mutex` 管理，按 MTE2 fill、VECTOR consume、MTE3 drain 的次序 lock/unlock。
  不包含 `set_flag` 或 `wait_flag`。
- **寄存器归约**：每 64 个 FP32 lane 组成一个向量 tile，max/sum 在寄存器内累计，
  每片只执行一次归约。
- **倒数替代除法**：归一化只计算一次 `1/sum`，后续元素执行乘法。
- **HOST 轻量化**：稳定输入的 DLPack 描述符和编译产物会缓存；已编译 artifact 直接接收
  Torch 张量，避免每次调用重复构造输入输出包装。

## 验证结果

在 Ascend950PR、CANN 9.2.0、`ascend-catlass-dsl 2.2.1` 上验证：

- 完整 50 个 fp32 用例全部通过。
- 24 个非 32B 字节对齐专项用例全部通过，覆盖尾长 1～63、跨 64 边界和长行尾块，
  逐元素匹配率为 100%。
- 目标长行 `M=1, N=1048576`：candidate `0.0762 ms`，Torch NPU reference
  `0.1298 ms`，speedup `1.70x`。

## 使用示例

以下示例在 catlass 仓库根目录执行：

```bash
python - <<'PY'
import torch
import torch_npu  # noqa: F401

from examples.python.tla_dsl.experimental.kernel.vec.streaming_masked_softmax.streaming_masked_softmax import run

m, n = 16, 8192
x = torch.randn(m, n, dtype=torch.float32, device="npu:0")
add = torch.zeros(m, n, dtype=torch.float32, device="npu:0")
add[:, n - 100 :] = -1.0e9

y = run(x, add, scale=1.0 / n**0.5)
torch.npu.synchronize()
print(y.shape, y.dtype)
PY
```

期望输出：

```text
torch.Size([16, 8192]) torch.float32
```

模块也提供独立精度自检，覆盖 resident、单块尾部、多块尾部、长行流式和 split-row
路径，并分别验证 `add=None` 与存在加性项：

```bash
python examples/python/tla_dsl/experimental/kernel/vec/streaming_masked_softmax/streaming_masked_softmax.py \
    --device 0
```
