# GroupNorm Example Readme

## 功能说明

- 算子功能：GroupNorm 前向（可融合 Swish/silu），入口为 `group_norm` 模块的 `run` 函数，由单个纯 AIV kernel 实现。`x` 为 `[N, C, S...]` 且 `C % G == 0`，同一组的 channel 在内存上连续，因此整块张量被看成一个 `[N * G, L]` 的二维问题（`L = (C / G) * S`），每个块认领若干行，单行处理链是
  `MTE2(整行) -> VECTOR(求和) -> VECTOR(去均值平方和) -> VECTOR(归一 + 仿射 + 可选 swish) -> MTE3(整行)`。
- 计算公式：对每个 `(n, g)` 行（`c` 为该行的 channel 下标，`CPG = C / G`）：

  $$
  \begin{aligned}
  \mu &= \frac{1}{L}\sum_{i} x_i \\
  \sigma^2 &= \frac{1}{L}\sum_{i} (x_i - \mu)^2 \\
  y_i &= \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot w_c + b_c \\
  y_i &\gets y_i \cdot \mathrm{sigmoid}(y_i) \quad (\text{SWISH} = 1)
  \end{aligned}
  $$

  全程 FP32 计算；`weight` / `bias` 必须同时给或同时不给（只给任一者都直接报错），无仿射时整段在编译期消除；`swish=True` 时在归一仿射之后原地融合。
- 高性能设计：
  - **按 channel 切 chunk，天然覆盖非对齐 `S`**：组内每个 channel 的 `S` 个元素单独成段，UB 里每段按 `S_PAD = ceil(S/64) * 64` 补齐。`S % 64 == 0` 时整行在 UB 里仍是一段连续的 `L` 个元素，行搬入/搬出各只有**一条连续 DMA 描述符**；`S % 64 != 0` 时每段末尾仍是一条完整 64-lane 向量、只放行 `S % 64` 个 lane。统计量只累加有效 lane，落盘侧 GM 保持稠密 `S`，padding lane 既不参与统计也不出 UB，**不需要 `UnalignStoreParams`，也不要求 `S` 是 32 B 对齐**。
  - **一条平坦的动态 chunk 循环 + 有上限的 trace 期展开**：不需要 channel 下标的遍历（两个统计 pass 的对齐路径、无仿射的缩放 pass）都写成一条 trip 为 `L / 64` 的平坦循环；按「每 channel 一条 `S / 64` 次的动态循环」写会退化成几十条两迭代的短循环，每条都付一次循环固定开销（小 channel 时实测慢 2.5 倍）。必须按 channel 走的路径（尾块统计、仿射缩放）把 channel 内的 chunk 在 trace 期展开，但展开深度上限 `K_UNROLL = 16`：一个基本块里塞 32 个 chunk 会让 256 B 向量寄存器溢出（`S = 2048` 时实测慢 3 倍），更长的 channel 因此降级为 `n_grp` 次动态迭代 × `k_unroll` 个展开 chunk 加余数。
  - **每 buffer 一个 mutex 的三段流水**：`BUFS` 个行缓冲各挂一个 mutex，同一时刻「MTE2 搬第 `r+1` 行 / VECTOR 算第 `r` 行 / MTE3 写第 `r-1` 行」三段重叠；缓冲复用由 mutex 而不是程序序决定，`BUFS` 按 UB 预算自动在 `1..6` 之间取，保证至少两块在飞。仿射表用单独的 mutex（`MTE2 -> VECTOR`）保护，无仿射时该同步在编译期消除。
  - **每行只做两次归约**：lane 内按 64-lane 向量累加，循环结束后只做一次 `reduce` 得到 mean、一次得到 var，归约的固定开销被 `L / 64` 次累加摊薄。
  - **方差用「先去均值再平方和」**：精度优于单趟写法，不用 `E[x^2] - mean^2`（均值远大于方差时 fp32 相消会失真）。
  - **仿射参数按 channel 折算成 scale/shift**：每个 channel 只取一次 `w[c]` / `b[c]`（`DIST_BRC_B32` 把单个 f32 元素直接广播成 64-lane 寄存器，是向量 load 而不是带依赖停顿的标量 load），并预折成 `scale = rstd * w[c]`，于是每个 chunk 只剩 `(x - mean) * scale + b[c]`，比非仿射路径只多一条 FMA：没有逐 chunk 的整数除法，也没有逐 chunk 的表查找。先减均值再缩放（不折成 `x * A + B`）是为了保住精度：行均值远大于方差时 `x * A + B` 会在两个大数相减处丢有效位。
  - **全程 FP32 计算**：fp16/bf16 的 UB 缓冲按原 dtype 存放，`DIST_UNPACK_B16` 载入即转 fp32，`DIST_PACK_B32` 落盘时转回，UB 占用减半、向量计算不降精度。
  - **Swish 原地融合**：`y * sigmoid(y)` 复用同一份寄存器，不额外读写 UB。
  - **整行只读一遍的统计与缩放**：统计与缩放各只读一遍行、中间量不出 UB、行 DMA 走单描述符、仿射无逐 chunk 分支。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
y = run(x, num_groups, weight=None, bias=None, eps=1e-5, swish=False)
```

`N` 为 batch，`C` 为通道数，`S...` 为任意多阶空间维度（内部展平为 `S`），`G` 为 `num_groups`，`CPG = C / G` 为每组的通道数，`L = CPG * S` 为单行长度。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[N, C, S...]` | `float32` / `float16` / `bfloat16` | 至少 3 维；`S...` 可为任意取值（非 64 倍数走尾块路径）；非连续输入内部自动 `contiguous()`。 |
| `num_groups` | 属性 | 标量 | Python `int` | 分组数 `G`，需 `C % G == 0`；必须是 `int`（`bool` / `float` 直接报错），且 `1 <= G <= C`。 |
| `weight` | Input | `[C]` | 同 `x` 或 `float32` | 仿射缩放；内部转成 fp32 连续张量，并按「张量身份 + `_version`」缓存。常规 in-place 修改会 bump `_version` 触发重建；**绕过版本计数的写入（`weight.data.copy_()`、自定义 kernel 直接改内存）不会被感知**，此时需传入新建的张量。fp32 且连续的 `weight` 零拷贝，不付 cast。 |
| `bias` | Input | `[C]` | 同 `x` 或 `float32` | 仿射偏移；**必须与 `weight` 同时给**，只给 `weight` 或只给 `bias` 都会直接报错。 |
| `eps` | 属性 | 标量 | Python `float` | 方差平滑项，默认 `1e-5`。 |
| `swish` | 属性 | 标量 | Python `bool` | 融合 `silu`，默认 `False`。 |
| `y` | Output | 同 `x` | 同 `x` | 归一（+ 仿射 + 可选 swish）结果，新分配的且与 `x` 同形状的张量。 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入维度 | `x` 至少 3 维 `[N, C, S...]`；`S...` 各维乘积为 `S`，`S` 只要求是正整数。 |
| 分组 | 必须 `C % G == 0`，且 `1 <= G <= C`；`CPG = C / G >= 1`。 |
| 仿射 | `weight` 与 `bias` 必须同时给或同时不给，且 `numel() == C`；只给 `weight`、只给 `bias` 均直接报错，避免把缺失的 `weight` 静默当作 1。 |
| dtype | `x` 为 `float32` / `float16` / `bfloat16`；`weight` / `bias` 可为同 dtype 或 `float32`（内部统一成 fp32）；`y` 与 `x` 同 dtype。设备侧全程 FP32。 |
| 设备 | `x` 必须是 `npu` 张量，`weight` / `bias` 必须与 `x` 同设备。校验放在首次调用（缓存 miss 建 plan）路径：命中启动缓存时张量已在 key 中被 pin 住，热路径不付任何校验开销。 |
| 尾块 | `S % 64 != 0` 时走尾块路径：UB 内每 channel 补到 `S_PAD`，尾块用 `update_mask(TAIL)` 只放行前 `S % 64` 个 lane，不产生跨行越界访问；不使用 `UnalignStoreParams`。`S` 不要求 32 B 对齐（示例里 `S = 100` / `S = 65` 都可执行）。 |
| UB 占用 | 每个在飞行缓冲 `CPG * S_PAD * sizeof(dtype)` 字节，另加两张常驻仿射表 `2 * C_PAD * 4` 字节（`C_PAD` 为 `C` 的 64 对齐，**仅 `has_affine` 时分配**）；`slack = 8192 + (2 * C_PAD * 4 if has_affine else 0)`，`BUFS = max(1, min(6, (262144 - slack) // row_bytes))`。**单行必须能放进一个 UB 缓冲**，即 `CPG * S_PAD * sizeof(dtype) <= 262144 - slack`，否则编译期越界。宽行会自动退化成单缓冲（流水变浅）。 |
| 任务划分 | `rows = N * G` 行按 `r = block_idx + k * vector_core_num` 跨步分配，块数取 `vector_core_num`；每块内 `BUFS` 行为一组做三段流水。 |
| 数值行为 | 归约在 fp32 下两遍完成（均值、去均值平方和）：fp32 输出误差约 `1e-6`，fp16 / bf16 受输出量化限制（约半个 ulp）。 |
| 编译 | 按 `(rows, CPG, C, S, has_affine, swish, eps, dtype)` 缓存编译产物，形状或参数变化会重编译（编译选项固定 `--npu-arch 3510`）。`C` 必须参与缓存键：它决定仿射表大小与搬运长度，并影响 `BUFS`。 |
| 启动缓存 | host 侧按「5 个张量地址 + 形状 + 参数」缓存绑定张量与 launch payload；仿射 fp32 副本、channel 基址表、block 数均按形状缓存。切换 `torch.npu.stream` 后需调用 `reset_stream_cache()`。 |
| 执行 | 纯 AIV，无 AIC / 无跨核同步；`weight` / `bias` 视为每次 launch 不变的常量表（常规 in-place 修改由 `_version` 感知并重建 fp32 副本；绕过版本计数的写入不会被感知）。 |

## 使用示例

以下命令在仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

from group_norm.group_norm import run

torch.npu.set_device(0)
torch.manual_seed(0)
n, c, s, g = 3, 32, 100, 8          # S = 100 不是 64 的倍数，走尾块路径
dt = torch.float16
x = torch.randn(n, c, s, dtype=dt, device="npu:0")
w = torch.randn(c, dtype=dt, device="npu:0")
b = torch.randn(c, dtype=dt, device="npu:0")

y = run(x, g, w, b)                                  # 也可 run(x, g) 走无仿射路径
ref = F.group_norm(x.cpu().float(), g, w.cpu().float(), b.cpu().float(), 1e-5)
print(y.shape, y.dtype)
print(f"err {(y.cpu().float() - ref).abs().max().item():.2e}")

y_sw = run(x, g, w, b, swish=True)
ref_sw = F.silu(ref)
print(f"swish err {(y_sw.cpu().float() - ref_sw).abs().max().item():.2e}")
PY
```

```text
torch.Size([3, 32, 100]) torch.float16
err 1.93e-03
swish err 1.94e-03
```
