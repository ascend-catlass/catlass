# LayerNorm Example Readme

## 功能说明

- 算子功能：按行（per-token）做 LayerNorm，入口为 `layernorm` 模块的 `run` 函数，基于 CATLASS DSL 实现的单个纯 AIV kernel。行按连续 `TILE_ROWS` 行切成一个 tile，每个 AIV 块认领一段连续 tile（一个 tile 就是一次连续 MTE2 搬入 + 一次 MTE3 搬出），`BUFS` 个 tile 缓冲用 per-buffer mutex 串成「搬入 / 计算 / 搬出」三段流水，让 MTE2、VECTOR、MTE3 三个 pipe 互相重叠。
- 计算公式：对每一行 `r`（`eps = EPS`，仿射可选）：

  $$
  \begin{aligned}
  \mu_r &= \frac{1}{N}\sum_{i} x_{r,i} \\
  \sigma_r^2 &= \frac{1}{N}\sum_{i} \left(x_{r,i} - \mu_r\right)^2 \\
  y_{r,i} &= \frac{x_{r,i} - \mu_r}{\sqrt{\sigma_r^2 + \epsilon}} \cdot w_i + b_i
  \end{aligned}
  $$

  全程 FP32 计算；`w`/`b` 缺省时只做归一化（`HAS_AFFINE = 0` 分支在编译期折叠）。方差走「先去均值再平方和」的两趟算法（与 torch 的 Welford 精度同量级），不用 `E[x^2] - mean^2` 的单趟写法（均值远大于方差时 fp32 相消会失真）。
- 高性能设计：
  - 行摊薄：`TILE_ROWS = 4` 时四行的求和 / 平方和 / 缩放分别合并在同一个动态 chunk 循环里，四条独立累加链提供指令级并行；缩放趟的 `w`、`b` lane 从「每行一次 load」降为「每 tile 一次 load」。
  - 归约摊薄：行内统计量先按 64 lane 向量累加，循环结束后每行只做一次 `reduce`，把 reduce 的固定开销摊到全部 chunk 上。
  - `w`/`b` 常驻 UB：每个块只在开头搬一次，整个 tile 循环复用，用 mutex 保护 MTE2 -> VECTOR 交接；无仿射时这条同步在编译期消除。
  - `TILE_ROWS`、`BUFS` 由 UB 预算在 host 侧选择（4 / 2 / 1 行，至多 4 块缓冲），保证至少两块 tile 在飞。
- 非对齐 `N`（`N % 64 != 0`）：尾块路径，UB 行按 `N_PAD = ceil(N/64) * 64` 补齐，末块按完整 64-lane 向量读写，用 `update_mask(N % 64)` 只把前 `N % 64` 个 lane 计入统计并落盘；对齐时整条路径在编译期消除。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
y = run(x, weight=None, bias=None, eps=1e-6)
```

`M` 为行数，`N` 为每行长度。`weight` 与 `bias` 要么都给、要么都不给；只给 `weight` 时 `bias` 视为全 0。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, N]` | `float32` | 输入；`N` 任意（非 64 倍数走尾块路径）；非连续输入内部自动 `contiguous()`。 |
| `weight` | Input，可选 | `[N]` | `float32` | 仿射缩放，默认 `None` 表示不带仿射。 |
| `bias` | Input，可选 | `[N]` | `float32` | 仿射偏置，仅在给出 `weight` 时生效。 |
| `eps` | 属性 | 标量 | Python `float` | 方差修正项，默认 `1e-6`。 |
| `y` | Output | `[M, N]` | `float32` | 归一化结果。 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入维度 | `x` 必须为 2 维；`M`、`N` 为正整数，`N` 不限（非 64 倍数时 UB 行按 64 补齐）。 |
| dtype | 输入、输出、仿射参数均为 `float32`。 |
| UB 占用 | `BUFS` 块 `TILE_ROWS * N_PAD` FP32 tile（`N_PAD = ceil(N/64) * 64`）+ `w`/`b` 各 `N_PAD`；受 256 KiB UB 限制，`N` 上限约 16K。 |
| tile 选择 | `TILE_ROWS` 取能放下两块在飞 tile 的最大值（4 → 2 → 1），且需 `M % TILE_ROWS == 0`；`BUFS` 取剩余 UB 能容纳的块数，上限 4。 |
| 数值行为 | 设备侧全程 FP32；方差为两趟算法，不做单趟 `E[x^2] - mean^2`。 |
| 编译 | 按 `(M, N, eps, 是否带仿射)` 缓存编译产物，形状或参数变化会重编译。 |
| 启动缓存 | host 侧按张量地址缓存绑定张量与 launch payload；切换 `torch.npu.stream` 后需调用 `reset_stream_cache()`。 |
| 执行 | 纯 AIV；block 数取 `vector_core_num` 并缓存。 |
| 硬件 | 编译选项固定 `--npu-arch 3510`。 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from layernorm.layernorm import run

torch.npu.set_device(0)
torch.manual_seed(0)
m, n = 128, 256
x = torch.randn(m, n, dtype=torch.float32, device="npu:0")
w = torch.randn(n, dtype=torch.float32, device="npu:0")
b = torch.randn(n, dtype=torch.float32, device="npu:0")

y = run(x, w, b, 1e-6)
mu = x.mean(-1, keepdim=True)
var = x.var(-1, unbiased=False, keepdim=True)
ref = (x - mu) / torch.sqrt(var + 1e-6) * w + b
print(y.shape, y.dtype, f"max abs err {(y - ref).abs().max().item():.2e}")
PY
```

```text
torch.Size([128, 256]) torch.float32 max abs err 1.43e-06
```
