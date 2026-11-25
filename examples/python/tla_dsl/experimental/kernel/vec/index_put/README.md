# IndexPut Example Readme

## 功能说明

- 一维 `index_put` / scatter-add：先复制底张量，再按索引写入或累加 `values`。通过 `index_put` 模块的 `run` 函数调用。
- 计算公式：

  $$
  \begin{aligned}
  out &= x.\text{clone}() \\
  out_{idx_i} &\mathrel{+}= v_i \quad (\text{accumulate=True}) \\
  out_{idx_i} &= v_i \quad (\text{accumulate=False})
  \end{aligned}
  $$

  其中 `idx` 对应 `index`，`v` 对应 `values`；`accumulate=True` 时重复下标由设备端硬件原子加合并，累加顺序不确定，只产生 fp32 尾差。

- 支持产品型号为 Ascend 950PR&950DT 系列产品（编译选项固定 `--npu-arch 3510`）。

## 参数说明

```python
out = index_put.run(x, index, values, accumulate=True)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[N]` | `float32` | 被写入的底张量，只读。 |
| `index` | Input | `[M]` | `int32` / `int64` | 目标下标，必须落在 `[0, N)`，可重复。 |
| `values` | Input | `[M]` | `float32` | 写入/累加值，shape 必须与 `index` 一致。 |
| `accumulate` | Input | 标量 | `bool` | `True` 累加，`False` 覆盖。 |
| `out` | Output | `[N]` | `float32` | 结果，不改写 `x`。 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入维度 | 仅支持 1-D `x`；`values.shape` 必须等于 `index.shape`，否则抛 `ValueError`。 |
| dtype | `x`、`values` 必须为 `float32`；`index` 必须为 `int32` / `int64`，其它 dtype 抛 `ValueError`。 |
| 下标范围 | `index` 必须在 `[0, N)`；本实现不做负索引回绕与越界检查，传负值属于未定义行为。 |
| 重复索引 | `accumulate=True` 由硬件原子加合并，顺序不确定；`accumulate=False` 重复索引最后一次写生效（与 torch 一致）。 |
| 精度 | 累加为 fp32 原子加，相对 torch 的尾差实测 ≤ 3.8e-6（`atol=1e-4` / `rtol=1e-5`）。 |
| 大规模 N | `N < 2^31` 走多核原子路径；`N >= 2^31` 回退单 block 顺序路径（正确但慢），避免 32 位线性索引溢出。 |
| 设备 | `x`、`index`、`values` 位于同一 NPU；调用前需 `torch.npu.set_device`。 |
| 编译 | 按 `(M, N, dtype, CHUNK, 路径)` 缓存编译产物；host 按 `(data_ptr, numel, dtype)` 缓存 tensor 描述符。 |
| 内存 | 每 block 的 stage 为 `CHUNK * 8` 个 fp32（≤128 KiB），`CHUNK ≤ 4096`；`M` 更大时增加 block 数。 |

## 实现说明

四条路径共用一个 `run` 入口：

| 路径 | 条件 | 做法 |
| ----- | ----- | ----- |
| `scatter_overwrite_kernel` | `accumulate=False` | 64 个 AI vector core × SIMT 线程按 `gid/stride` 并行写 `out[index[i]]`。 |
| `scatter_add_atomic_kernel` | `accumulate=True`，`N < 2^31`，`M > 512`，`CHUNK` 非 128 倍数 | 每 block 处理 `CHUNK` 个元素，64 个 value 一批：`tla.scatter` 把每个值写到独立 32B 槽的 lane 0，再以 `shape(1)` + `AtomicMode.ADD` 做 4B 原子加。 |
| `scatter_add_atomic_pipe_kernel` | 同上且 `CHUNK` 为 128 倍数（大 M） | 同上，value 双缓冲，`ub_val0/1`、`ub_stage0/1` 四把 mutex 交接缓冲区所有权，MTE3 与下一批 DMA/vector 重叠。 |
| `scatter_add_small_kernel` | `M ≤ 512` 或 `N >= 2^31` | 单 block 顺序累加兜底。 |

高性能设计点：

- **4B 原子加而不是 32B 扇区**：`tla.copy` 的 `AtomicMode.ADD` 降为 `SetAtomicAdd + DataCopyPad + SetAtomicNone`，`DataCopyPad` 只搬 tile 的字节数。单元素 tile 就是一次 4B 原子加，索引的原子写流量从 32B 降到 4B。
- **原子 tile 必须由 vector pipe 生成**：每个值独占一个 32B UB 槽的 lane 0，由一次 `tla.scatter` 写入。vector 生成的 tile 与 MTE3 原子读之间有 2.2.1 的 mutex 排序；若用标量 UB 写填 tile，实测会丢加/多加。
- **stage 不复用**：每个元素在 block 内拥有独立的 32B 槽，不做“清零后复用”，从根上消除 clear 与在途原子读的竞争，多波次调度也安全。
- **按核切连续区间、索引不进 UB**：每 block 处理 `[lo, lo + CHUNK)`，索引只做标量读，value 每批一次 64 元素 DMA；尾部补齐到 64 的倍数并用 `gidx < m` mask，补零的加法无害。
- **mutex 流水（大 M）**：原实现每 64 元素一次 `pipe_barrier(ALL)` 会把在途原子 DMA 一起等掉；流水版把 value 批次双缓冲，用 `ub_val0/1`（MTE2↔VECTOR）和 `ub_stage0/1`（VECTOR↔MTE3）四把 mutex 交接所有权，MTE3 不再阻塞下一批的 DMA 与 scatter。
- **host 只留必要开销**：输入/输出描述符按指针缓存；padding 用缓存的零缓冲 + 一次 `copy_`（顺带完成 dtype 转换），每 call 的设备算子只剩 `x.clone()` + scatter kernel。

## 使用示例

以下命令在 CATLASS 仓库根目录执行。

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from index_put.index_put import run

torch.npu.set_device(0)
x = torch.randn(4096, dtype=torch.float32, device="npu:0")
index = torch.randint(0, 4096, (3000,), dtype=torch.int64, device="npu:0")
values = torch.randn(3000, dtype=torch.float32, device="npu:0")

out = run(x, index, values, accumulate=True)
print(out.shape, out.dtype)
PY
```

```text
torch.Size([4096]) torch.float32
```

在一维 `index_put` 的 13 条覆盖用例（重复索引、奇偶 shape、int32 索引）上，端到端均值相对原版单 block 实现由 3.6371 ms 降到 0.1693 ms，逐例 1.73×~118.63×；与 torch 原生 `index_put_` 的端到端均值同一量级（0.1483 vs 0.1324 ms），5/13 例反超。
