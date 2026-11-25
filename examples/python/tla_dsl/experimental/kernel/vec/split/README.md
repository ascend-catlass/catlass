# Split Example Readme

## 功能说明

- 算子功能：将二维 `float32` 张量 `x(M, N)` 沿最后一维等分，返回 `parts` 个行内稠密的
  `(M, N / parts)` 分片；所有分片来自同一块连续的 `(parts, M, W)` 缓冲。
- 单次 launch 完成全部分片，不随 `parts` 增加 host dispatch 或 device launch 次数。
- 支持产品型号：Ascend 950PR&950DT 系列产品。
- 计算公式：

  $$
    Y[p, i, j] = X[i, p \times W + j], \quad
    0 \leq p < parts,\ 0 \leq i < M,\ 0 \leq j < W
  $$

  其中 `W = N / parts`。

## 参数说明

```python
parts = split(x, 4, dim=-1)
packed = run(x, 4, dim=-1)
```

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `x` | Input | `(M, N)` | `float32` | 输入张量，非连续输入会转为连续 |
| `parts` | Input | 标量 | `int` | 分片数，`parts >= 1` |
| `dim` | Input | 标量 | `int` | 仅支持 `-1` 或 `1` |
| `parts` 返回值 | Output | `parts` 个 `(M, W)` | `float32` | 连续 packed 缓冲的稠密切片 |
| `run` 返回值 | Output | `(parts, M, W)` | `float32` | 完整 packed 缓冲 |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 输入维度 | `x.dim() == 2` |
| dtype | 输入和输出均为 `float32` |
| 整除关系 | `N % parts == 0` |
| 分片宽度 | `W <= 24576`，保证双 UB 缓冲中每个槽位可容纳一整行 |
| `dim` | 仅支持最后一维，允许 `-1` 或 `1` |
| 空输入 | `M == 0` 或 `N == 0` 时返回空缓冲，不启动 kernel |
| GM 对齐 | 任意 `N`、`W` 和 `p * W` 均可，无 32 B 字节对齐要求 |
| UB 对齐 | 行 stride 内部补齐到 32 B，逻辑 shape 仍为 `(BR, W)` |
| 设备 | 输入位于 NPU；同一 shape 在同一 device 上复用编译产物和 descriptor |

## 高性能设计

- **单次 launch：** 全部分片写入同一块连续 GM 输出，只下发一次 kernel。
- **任务展平：** `(part, row_group)` 和 `(part, tail_row)` 分别展平为连续任务域，所有
  AIV 核跨步处理，避免外层 part 串行和尾行空闲。
- **双 UB 槽位：** 两个槽位交替承载相邻 tile，使下一次 MTE2 搬入与上一次 MTE3 写回重叠。
- **资源 mutex：** 每个槽位使用独立 mutex，通过 MTE2/MTE3 的 lock/unlock 表达 RAW/WAR，
  不同槽位不共享依赖。
- **热点无全流水 barrier：** 所有 tile 完成后仅执行一次 `pipe_barrier(ALL)` 收尾。
- **非对齐 DMA：** UB 行 stride 补齐到 32 B，而 GM 源和目的分别使用 `N`、`W` stride，
  直接支持非对齐列宽与列偏移。

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
python - <<'PY'
import torch
import torch_npu  # noqa: F401

from examples.python.tla_dsl.experimental.kernel.vec.split.split import split

x = torch.randn(37, 1001, dtype=torch.float32, device="npu:0")
outputs = split(x, 7)
references = [part.contiguous() for part in torch.split(x, 143, dim=-1)]

for output, reference in zip(outputs, references):
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
print([tuple(output.shape) for output in outputs])
PY
```

运行模块自检。自检会对 10 组 shape 做逐位比较并检查分片连续性，失败时返回非零退出码：

```bash
python examples/python/tla_dsl/experimental/kernel/vec/split/split.py --device 0
```

## 性能

在 Ascend 950PR 上，将完整连续分片结果与以下等价 Torch NPU 实现配对计时：

```python
torch.stack(torch.split(x, x.shape[-1] // parts, dim), dim=0)
```

覆盖 25 条 shape、warmup 1 次、采集 5 次的端到端平均结果：

| 用例 | 本样例 | Torch NPU | 加速 |
| --- | ---: | ---: | ---: |
| 全部用例平均 | 0.05922 ms | 0.07526 ms | **1.27x** |
| `(8192, 8192) / 4` | 0.4148 ms | 0.7804 ms | **1.88x** |

极小 shape 时 host 启动固定开销占主导；`split_kernel` 本体通常只有几微秒。仅返回视图的
`torch.split` 不搬运数据，因此不作为产生连续分片结果的性能基线。
