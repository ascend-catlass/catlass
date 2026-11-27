# ReduceMeanBatchInvariant Example Readme

## 功能说明

- 算子功能：对 `[batch, m, n]` 的 `x` 沿 `axis` 求均值，`batch` 维不参与归约；基于 CATLASS DSL 实现的 AIV kernel，覆盖 `float32` / `float16` / `bfloat16`。
- batch 一致性：每个输出元素只由单行（`axis = 2`）或单 `batch` 元素的同名列块（`axis = 1`）决定，且归约顺序固定，因此同一行数据在任意 `batch` 大小与 `batch` 划分下结果逐位一致，与 `batch` 无关。
- 计算公式：

  $$
  out[b, i] = \frac{1}{m} \sum_{j=0}^{m-1} x[b, j, i] \quad (axis = 1)
  $$

  $$
  out[b, j] = \frac{1}{n} \sum_{i=0}^{n-1} x[b, j, i] \quad (axis = 2)
  $$

  其中 `axis = 1` 时输出为 `[batch, n]`，`axis = 2` 时输出为 `[batch, m]`。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 参数说明

```python
out = reduce_mean_batch_invariant.run(x, axis)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[batch, m, n]` | `float32` / `float16` / `bfloat16` | 行优先连续张量；非连续输入在 host 侧先做 `contiguous()` |
| `axis` | Input | 标量 | `int` | 归约轴，只支持 `1` 或 `2`；`batch` 维（`0`）不允许归约 |
| `out` | Output | `axis = 1`：`[batch, n]`<br>`axis = 2`：`[batch, m]` | 与 `x` 一致 | 均值结果，kernel 内部以 `float32` 累加后转回输入 dtype |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | 必须为 3 维 `[batch, m, n]`；`batch` 与被归约维必须为正 |
| axis | 仅支持 `1` 或 `2`；`axis = 0` 未实现 |
| dtype | `float32` / `float16` / `bfloat16`；输出 dtype 与输入一致，不做隐式类型转换 |
| 归约长度 | `axis = 2` 时 `n <= 16384`（整行需 stage 进 UB）；`axis = 1` 按列分块，无额外上限 |
| 设备 | `x` 与输出位于同一 NPU |
| 编译 | 源码只定义一个 `@tla.kernel`。编译时按 `(axis, n, dtype)` 特化并缓存，行数变化可复用编译结果。每次调用下发一个 kernel |
| batch 一致性 | 同一行数据在任意 `batch` 划分下输出逐位一致；kernel 内无跨 `batch` 累加、无原子操作 |

## 实现说明

- 只对输出元素维度并行：`axis = 2` 按行（`batch * m`）切任务，`axis = 1` 按 `(batch, 列块)` 切任务；网格大小由 host 按规模计算并封顶 48 个 block。
- 任务划分只沿输出元素展开，行与行、`batch` 与 `batch` 之间没有数据依赖：同一行无论和哪些行同批下发、落在哪个 block 上，归约顺序都固定不变，因此结果与 `batch` 划分无关。
- `axis = 2`：一次 stage `row_tile` 整行到 UB，逐行在 64 lane 的 `float32` 寄存器中累加，`row_tile` 个标量结果先写进 UB 的一行再一次连续搬出；`row_tile` 除不尽的行由尾循环逐行处理。
- `axis = 1`：一次处理一个 `batch` 元素的一个列块（`float32` 64 列，`float16` / `bfloat16` 128 列），每次 stage 32 行输入，列累加器驻留 UB。
- 两个归约轴和低精度计算路径通过编译期分支选择。kernel 从 `mem_x.ptr.dtype` 取得 UB 分配和输出转换的数据类型。
- `float16` / `bfloat16` 的块长为 128，用 `CastParams(reg_slot=ZERO/ONE)` 分离偶/奇 lane 分别加宽到 `float32` 累加；收窄时两半相加，还原连续的 128 lane 布局。
- 累加与归约全程 `float32`，`1 / n`、`1 / m` 由 host 预先算成标量传入，避免 kernel 内除法。
- UB 缓冲由 `tla.mutex` 的 lock/unlock 保护，输入搬运（MTE2）、向量计算（VECTOR）、结果搬出（MTE3）三者的同步关系在 kernel 内显式表达。

## 使用示例

准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境，导入样例模块后调用 `run`：

```python
import torch
import torch_npu  # noqa: F401

import reduce_mean_batch_invariant

batch, m, n = 8, 2048, 2048
x = torch.randn(batch, m, n, dtype=torch.float16, device="npu:0")

out = reduce_mean_batch_invariant.run(x, 2)
print(out.shape, out.dtype)

ref = x.float().mean(dim=2)
print(torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2))
```

执行结果如下，说明样例执行成功：

```text
torch.Size([8, 2048]) torch.float16
True
```
