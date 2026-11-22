# QsaIndexer Example Readme

## 功能说明

- 算子功能：计算稀疏注意力的索引表。先在分页压缩 KV 上完成 `q·k` 打分，再在每行可见压缩块内做 top-k 选择，最后把选中的压缩块展开成原始 token 下标序列写回 GM；入口为 `qsa_indexer` 模块的 `qsa_indexer` kernel，基于 CATLASS DSL 实现的 AIC/AIV mixed kernel。`run_topk=False` 时跳过打分与 top-k，直接输出全可见 token 下标。
- 计算公式：记第 `i` 个 query token 的位置为 `p_i`，压缩块大小 `CR = compress_ratio`，页大小 `P = page_size`，保留块数 `K = topk_limit`。压缩块 `b` 对应物理页 `block_table[0, b / P]` 的第 `b % P` 个槽位，取 `compressed_k` 的第 0 个 KV head：

  $$
  \begin{aligned}
  k_c[b, :] &= compressed\_k\left[block\_table[0,\ b / P],\ b \bmod P,\ 0,\ :\right] \\
  s[i, b] &= \sum_{h=0}^{QH-1} \max\left(q[i, h, :] \cdot k_c[b, :],\ 0\right) \\
  visible_i &= \left\lfloor \frac{p_i + 1}{CR} \right\rfloor, \qquad
  tail_i = (p_i + 1) \bmod CR, \qquad keep_i = \min(visible_i,\ K)
  \end{aligned}
  $$

  `visible_i ≤ K` 时输出全部可见 token；否则按 `s[i, :visible_i]` 从大到小取 `K` 个块（同分按块下标升序补足），块 `b` 展开为 `b*CR … b*CR + CR-1`，尾部不足一块的 token 以原始下标追加。`W = CR × (K + 1)` 时输出布局为：全可见路径 `out[i, j] = j`（`j ≤ p_i`）；top-k 路径第 `t` 个选中块 `b` 的第 `l` 项写入 `out[i, t*CR + l] = b*CR + l`，尾部 token 写入 `out[i, j] = visible_i*CR + (j - keep_i*CR)`（`keep_i*CR ≤ j < keep_i*CR + tail_i`）；其余槽位为 `-1`；每行最后槽位写入有效索引个数 `out[i, W-1] = keep_i*CR + tail_i`。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
qsa_indexer(
    q, compressed_k, block_table, actual_seq, query_positions, output,
    raw_workspace, score_workspace, compress_ratio, topk_limit,
    query_heads, head_dim, page_size, kv_heads, run_topk, rows, output_width,
)
```

记 `rows` 为 query token 数，`QH/KVH` 为 query/KV head 数，`D` 为 head 维度，`pages` 为 `block_table` 页数，`max_blocks = pages × page_size`。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `q` | Input | `[rows, QH, D]` | `float16` / `bfloat16` | 行优先连续；分数在每个 token 的全部 query head 上求和。 |
| `compressed_k` | Input | `[pages, P, KVH, D]` | 同 `q` | 分页压缩 KV；只用第 0 个 KV head，所有 query head 共享。 |
| `block_table` | Input | `[1, pages]` | `int32` | 逻辑页到物理页映射，只读取第 0 行。 |
| `actual_seq` | Input | 任意 | 任意 | 预留参数，当前实现不读取。 |
| `query_positions` | Input | `[rows]` | `int32` | 每个 query token 的绝对位置 `p_i`。 |
| `output` | Output | `[rows, W]` | `int32` | 稀疏索引表：前 `keep_i*CR + tail_i` 项为 token 下标，其余 `-1`，末项为有效个数。 |
| `raw_workspace` | Workspace | `[rows, max_blocks]` | `float32` | 仅提供 shape/stride 描述（原始分数经 L0C→UB 交给 AIV，不落该 GM 区）。 |
| `score_workspace` | Workspace | `[rows, max_blocks]` | `int16` | 存 AIV 归约后的逐块分数，按 FP16/BF16 位模式承载，作为 top-k 输入。 |
| `compress_ratio`、`topk_limit` | 属性 | 标量 | Python `int` | 每块覆盖 token 数 `CR`；每行保留块数上限 `K`。 |
| `query_heads`、`head_dim`、`page_size`、`kv_heads` | 属性 | 标量 | Python `int` | `QH`、`D`、`P`；`QH` 取 4 或 8 时有专门求和路径，`kv_heads` 仅描述 `compressed_k` 步长。 |
| `run_topk`、`rows`、`output_width` | 属性 | 标量 | `bool` / `int` | `True` 走打分 + top-k；`rows`、`output_width` 仅在 `run_topk=False` 时使用。 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 请求数 | 单请求；`block_table` 只读第 0 行，逻辑块 `b` 映射到 `block_table[0, b / P]` 的第 `b % P` 个槽位。 |
| 位置 | `query_positions` 非负且按行单调不减；需满足 `(p_i + 1) / CR ≤ pages × P`。 |
| 输出宽度 | `W = compress_ratio × (topk_limit + 1)`；top-k 路径写入前先把整行置 `-1`，`W` 需覆盖 `keep_i*CR + tail_i`；`run_topk=False` 时要求 `p_i + 1 ≤ W - 1`，否则下标越界，行为未定义。 |
| dtype | `q` 与 `compressed_k` 同为 `float16` 或 `bfloat16`；`block_table`、`query_positions`、`output` 为 `int32`。 |
| 分数单调性 | 分数经 ReLU 后非负，选择阶段的 FP16/BF16 位模式按无符号比较与数值大小等价。 |
| 片上容量 | 分数 tile 固定为 `32 × page_size` 块、`32 × QH` 行 FP32；top-k 输入复用该 UB 区，要求 `ceil(max_blocks / 256) × 256 ≤ 64 × QH × page_size`；选中块暂存区固定 640 个 FP16 槽位，要求 `topk_limit ≤ 640`。 |
| 设备与编译 | 所有张量与工作区位于同一 NPU；shape 与属性均为编译期常量，变化需重新 `tla.compile`。 |

## 使用示例

本目录只包含 kernel 定义，调用方需自行准备输入、工作区并编译执行。`q` 与 `compressed_k` 需以二维视图传入（`from_dlpack` 无法从三维/四维形状推导 RowMajor 布局）；`score_workspace` 按 `int16` 绑定，承载分数的位模式。

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/indexer:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

import catlass.tla as tla
from qsa_indexer.qsa_indexer import qsa_indexer

rows, qh, kvh, d = 127, 8, 1, 128
cr, p, topk, pages = 64, 4, 2, 8
mb, w = pages * p, cr * (topk + 1)
i32 = dict(dtype=torch.int32, device="npu:0")
f16 = dict(dtype=torch.float16, device="npu:0")
buffers = [
    torch.randn(rows * qh, d, **f16), torch.randn(pages * p * kvh, d, **f16),
    torch.arange(pages, **i32).reshape(1, pages), torch.full((1,), rows, **i32),
    torch.arange(rows, **i32) * 2, torch.empty((rows, w), **i32),
    torch.empty((rows, mb), dtype=torch.float32, device="npu:0"),
    torch.empty((rows, mb), dtype=torch.int16, device="npu:0"),
]
tensors = [tla.from_dlpack(t, layout_tag=tla.arch.RowMajor) for t in buffers]
artifact = tla.compile(qsa_indexer, *tensors, cr, topk, qh, d, p, kvh, True,
                       rows, w, options="--npu-arch 3510")
artifact(*tensors, block_num=20)
torch.npu.synchronize()
print(buffers[5].shape, buffers[5].dtype)
PY
```

```text
torch.Size([127, 192]) torch.int32
```
