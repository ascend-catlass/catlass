# ResidualAddRmsNorm Example Readme

## 功能说明

- 算子功能：先做逐元素残差相加，再按行执行 RMSNorm，可选逐列仿射。入口为
  `residual_add_rmsnorm` 模块的 `run` 函数，基于 CATLASS Python 实现单个纯 AIV
  kernel。
- 计算公式：对每一行 `r`：

  $$
  \begin{aligned}
  t_{r,i} &= x_{r,i} + \mathrm{residual}_{r,i} \\
  \mathrm{ms}_r &= \frac{1}{N}\sum_{i} t_{r,i}^2 \\
  y_{r,i} &= \frac{t_{r,i}}{\sqrt{\mathrm{ms}_r + \epsilon}} \cdot w_i
  \end{aligned}
  $$

  全程 FP32 计算；`weight=None` 时不做仿射，相关加载与乘法在编译期消除。
- 支持产品型号：Ascend 950PR 与 950DT 系列产品。

## 参数说明

```python
y = run(x, residual, weight=None, eps=1e-6)
```

`M` 为行数，`N` 为每行元素数。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, N]` | `float32` | 主输入；非连续输入内部自动 `contiguous()`。 |
| `residual` | Input | `[M, N]` | `float32` | 残差分支，shape 与 dtype 必须与 `x` 相同。 |
| `weight` | Optional Input | `[N]` 或 `None` | `float32` | 可选逐列仿射权重。 |
| `eps` | Input | 标量 | Python `float` | 非负有限稳定项，默认 `1e-6`。 |
| `y` | Output | `[M, N]` | `float32` | 融合结果。 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `x`、`residual` 必须为二维且 shape 完全一致。 |
| 设备 | `x`、`residual` 与可选 `weight` 必须位于同一个 NPU 设备。 |
| 行数 | `M >= 0`；`M=0` 时返回空张量，不启动 kernel。 |
| 行宽 | `N >= 64`；`N` 不要求 32B 对齐，非 64 整数倍走 masked tail 路径。 |
| dtype | 输入、输出和可选权重均为 `float32`。 |
| 权重 | `weight=None` 或 shape `[N]` 的 FP32 张量。 |
| UB 占用 | `BUFS * 2 * TILE_ROWS * N_PAD * 4 + N_PAD * 4` bytes，另留约 8 KiB 余量；单 AIV UB 上限按 256 KiB 设计。 |
| tile 选择 | `TILE_ROWS` 从 4、2、1 中选择，且需整除 `M`；`BUFS` 取剩余 UB 可容纳的块数，上限 4。 |
| 最宽输入 | `N=16384` 时使用 `TILE_ROWS=1, BUFS=1`，三份 FP32 行缓冲约 192 KiB。 |
| 同步 | 核内只使用 `tla.mutex` 表达 MTE2、VECTOR、MTE3 之间的资源所有权。 |
| 编译 | 按 `(M, N, eps, 是否仿射, TILE_ROWS, BUFS, device)` 缓存编译产物。 |
| Host 缓存 | 按输入地址与编译参数缓存 TLA descriptor、完整 launch plan 和预打包 launcher；不跨调用隐式复用输出。 |
| 设备流 | launcher 缓存绑定当前 stream；切换 stream 后需调用 `reset_stream_cache()`。 |

## 高性能设计

- **连续多行 tile**：连续 `TILE_ROWS` 行组成一个 tile，每个 AIV block 认领连续 tile，
  每个 tile 一次连续 MTE2 搬入、一次连续 MTE3 搬出。
- **三段流水**：`BUFS` 个 tile 缓冲各用一个 mutex，按 MTE2 载入、VECTOR 计算、MTE3
  写回的顺序传递所有权，并形成缓冲复用背压。
- **权重常驻 UB**：可选权重在每个 block 开头载入一次，在该 block 处理的所有行间复用。
- **归约摊薄**：行内平方和先保存在 64 lane 向量中，chunk 循环结束后每行只执行一次
  reduction。
- **尾块专门化**：GM 保持稠密 stride `N`，UB 行按
  `N_PAD = ceil(N / 64) * 64` 补齐；`update_mask(N % 64)` 只处理有效 lane，
  padding 不参与归约，也不写回 GM。
- **Host 快速路径**：当分配器复用输出地址时，复用完整 launch plan 和 CATLASS advanced
  launcher，避免重复构造参数树和 launch payload；输出存储本身不跨调用隐式池化。

## 精度与性能验证

- 在 Ascend950PR（A5）上执行 31 个模型场景 case，全部通过。
- 覆盖 `N=65..127` 的全部 63 种尾长度，plain 与 affine 共 126 个 case，全部通过；
  其中 112 个 case 的 `N * 4` 非 32B 对齐。
- 特殊场景覆盖空 Tensor、全零行、近 `eps`、零权重、大数输入和最大
  `N=16384`。
- 在 31 个稳定场景的逐例配对测试中，所有 case 均快于 `torch_npu` reference；
  平均 speedup 为 `1.59x`。
- `M=257, N=1024` 的 kernel duration 为 `2.673 us`，AIV active 时间为 `2.44 us`。

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
python - <<'PY'
import torch
import torch_npu  # noqa: F401

from examples.python.tla_dsl.experimental.kernel.vec.residual_add_rmsnorm.residual_add_rmsnorm import run

m, n = 37, 1000
x = torch.randn(m, n, dtype=torch.float32, device="npu:0")
residual = torch.randn_like(x)
weight = torch.randn(n, dtype=torch.float32, device="npu:0")

y = run(x, residual, weight)
torch.npu.synchronize()
print(y.shape, y.dtype)
PY
```

执行结果如下，说明样例执行成功：

```text
torch.Size([37, 1000]) torch.float32
```

模块也提供独立精度自检：

```bash
python examples/python/tla_dsl/experimental/kernel/vec/residual_add_rmsnorm/residual_add_rmsnorm.py \
    --device 0 --m 64 --n 512
```
