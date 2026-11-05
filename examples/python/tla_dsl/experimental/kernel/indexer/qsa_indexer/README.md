# QsaIndexer Example Readme

## 功能说明

- 算子功能：计算稀疏注意力的索引表（indexer）。先在分页压缩 KV 上完成 `q·k` 打分，再在每行可见压缩块内做 top-k 选择，最后把选中的压缩块展开成原始 token 下标序列写回 GM；基于 CATLASS DSL 实现的 AIC/AIV mixed kernel。`run_topk=False` 时跳过打分与 top-k，直接输出全可见 token 下标。
- 计算公式：记第 `i` 个 query token 的位置为 `p_i`，压缩块大小为 `CR = compress_ratio`，页大小为 `P = page_size`，保留块数为 `K = topk_limit`。压缩块 `b` 对应物理页 `block_table[0, b / P]` 内的第 `b % P` 个槽位，取 `compressed_k` 的第 0 个 KV head：

  $$
  k_c[b, :] = compressed\_k\left[block\_table[0,\ b / P],\ b \bmod P,\ 0,\ :\right]
  $$

  indexer 分数为各 query head 上 `q·k` 的非负部分之和（ReLU 后求和）：

  $$
  s[i, b] = \sum_{h=0}^{QH-1} \max\left(q[i, h, :] \cdot k_c[b, :],\ 0\right)
  $$

  每行可见的完整压缩块数与尾部 token 数为

  $$
  visible_i = \left\lfloor \frac{p_i + 1}{CR} \right\rfloor, \qquad
  tail_i = (p_i + 1) \bmod CR, \qquad
  keep_i = \min(visible_i,\ K)
  $$

  `visible_i ≤ K` 时输出全部可见 token；否则按 `s[i, :visible_i]` 从大到小取 `K` 个压缩块（同分按块下标升序补足），每个选中块 `b` 展开为 `b*CR + 0 … b*CR + CR-1`。尾部不足一个压缩块的 token 以原始下标追加。输出布局为：

  $$
  out[i, j] =
  \begin{cases}
  j & visible_i \le K,\ j \le p_i \\
  b*CR + l & j = t*CR + l,\ \text{第 } t \text{ 个选中块为 } b \\
  visible_i*CR + (j - keep_i*CR) & keep_i*CR \le j < keep_i*CR + tail_i \\
  -1 & \text{其余槽位}
  \end{cases}
  $$

  每行最后一个槽位 `out[i, W-1]` 写入该行有效索引个数：

  $$
  out[i, W-1] = keep_i*CR + tail_i
  $$

  其中 `W = output_width = CR × (K + 1)`。

- 支持产品型号：Ascend 950PR&950DT 系列产品

## 参数说明

kernel 直接以静态 shape 编译，入参如下：

```python
qsa_indexer(
    q, compressed_k, block_table, actual_seq, query_positions, output,
    raw_workspace, score_workspace,
    compress_ratio, topk_limit, query_heads, head_dim, page_size, kv_heads,
    run_topk, rows, output_width,
)
```

记 `rows` 为 query token 数，`QH/KVH` 为 query/KV head 数，`D` 为 head 维度，`pages` 为 `block_table` 的页数，`max_blocks = pages × page_size`。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `q` | Input | `[rows, QH, D]` | `float16` / `bfloat16` | 行优先连续张量；indexer 分数在每个 token 的全部 query head 上求和 |
| `compressed_k` | Input | `[pages, P, KVH, D]` | 与 `q` 相同 | 分页压缩 KV；kernel 只读取第 0 个 KV head，所有 query head 共享 |
| `block_table` | Input | `[1, pages]` | `int32` | 逻辑页到物理页映射，kernel 只读取第 0 行 |
| `actual_seq` | Input | 任意 | 任意 | 预留参数，当前实现不读取；长度信息由 `query_positions` 决定 |
| `query_positions` | Input | `[rows]` | 整数 | 每个 query token 的绝对位置 `p_i`，决定可见块数与尾部 token 数 |
| `output` | Output | `[rows, W]` | `int32` | 稀疏索引表；前 `keep_i*CR + tail_i` 项为 token 下标，其余为 `-1`，末项为有效索引个数 |
| `raw_workspace` | Workspace | `[rows, max_blocks]` | `float32` | `32 × QH` 行 × `page_size` 列原始分数 tile 的布局参考；kernel 只使用其 shape/stride 描述，原始分数经 L0C→UB 直接交给 AIV，不落该 GM 区 |
| `score_workspace` | Workspace | `[rows, max_blocks]` | 与 `q` 相同 | AIV 归约后的逐块分数，作为 top-k 选择输入；每行占 `max_blocks` 个连续元素 |

| 属性 | 类型 | 说明 |
| ----- | ----- | ----- |
| `compress_ratio` | `int` | 每个压缩块覆盖的原始 token 数 `CR` |
| `topk_limit` | `int` | 每行保留的压缩块数上限 `K` |
| `query_heads` | `int` | `QH`；`4` 与 `8` 有专门展开的求和路径，其他取值走通用循环 |
| `head_dim` | `int` | `D` |
| `page_size` | `int` | 每页压缩块数 `P` |
| `kv_heads` | `int` | 描述 `compressed_k` 的 head 步长，仅第 0 个 KV head 参与计算 |
| `run_topk` | `bool` | `True` 走打分 + top-k；`False` 走全可见索引 |
| `rows` | `int` | output 行数，仅 `run_topk=False` 使用 |
| `output_width` | `int` | output 列数，仅 `run_topk=False` 使用，应为 `CR × (K + 1)` |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 请求数 | 单请求；`block_table` 只读取第 0 行，逻辑块 `b` 映射到 `block_table[0, b / P]` 的第 `b % P` 个槽位 |
| 位置 | `query_positions` 非负且按行单调不减；需满足 `visible_i ≤ max_blocks`，即 `(p_i + 1) / CR ≤ pages × P` |
| 输出宽度 | `W = compress_ratio × (topk_limit + 1)`；top-k 路径写入前会先把整行置 `-1`，因此 `W` 需覆盖 `keep_i*CR + tail_i` |
| 全可见路径 | `run_topk=False` 时要求 `p_i + 1 ≤ W - 1`，否则下标越界，行为未定义 |
| dtype | `q` 与 `compressed_k` 必须同为 `float16` 或 `bfloat16`；`block_table`、`query_positions`、`output` 为 `int32` |
| 分数单调性 | 分数经 ReLU 后非负，选择阶段的 FP16/BF16 位模式按无符号比较与数值大小等价 |
| 片上容量 | 分数 tile 固定为 `32 × page_size` 个块、`32 × QH` 个 query 行，`page_size` 需使该 FP32 分数 tile 落在 UB 容量内；top-k 输入复用该 UB 区，要求 `ceil(max_blocks / 256) × 256 ≤ 64 × QH × page_size` |
| top-k 暂存 | 选中块暂存区固定 640 个 FP16 槽位，要求 `topk_limit ≤ 640` |
| 设备 | 所有张量与工作区位于同一 NPU |
| 编译 | 所有 shape 与属性均为编译期常量；`rows`、`max_blocks`、`P`、`CR`、`K`、`QH`、`D` 或 `run_topk` 变化需重新 `tla.compile` |

## 使用示例

本目录只包含 kernel 定义，没有内置的命令行样例或 `run` 包装，调用方需自行准备输入、工作区并编译执行。准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境后，可按下面的方式调用：

```python
import torch
import torch_npu  # noqa: F401

import catlass.tla as tla
from qsa_indexer import qsa_indexer

rows, query_heads, kv_heads, head_dim = 127, 8, 1, 128
compress_ratio, page_size, topk_limit = 64, 4, 2
pages = 8
max_blocks = pages * page_size
output_width = compress_ratio * (topk_limit + 1)

q = torch.randn(rows, query_heads, head_dim, dtype=torch.float16, device="npu:0")
compressed_k = torch.randn(
    pages, page_size, kv_heads, head_dim, dtype=torch.float16, device="npu:0"
)
block_table = torch.arange(pages, dtype=torch.int32, device="npu:0").reshape(1, pages)
actual_seq = torch.full((1,), rows, dtype=torch.int32, device="npu:0")
query_positions = (torch.arange(rows, dtype=torch.int32, device="npu:0") * 2)
output = torch.empty((rows, output_width), dtype=torch.int32, device="npu:0")
raw_workspace = torch.empty(
    (rows, max_blocks), dtype=torch.float32, device="npu:0"
)
score_workspace = torch.empty(
    (rows, max_blocks), dtype=torch.float16, device="npu:0"
)


def to_tla(tensor):
    return tla.from_dlpack(tensor, layout_tag=tla.arch.RowMajor)


tensors = [
    to_tla(t)
    for t in (
        q,
        compressed_k,
        block_table,
        actual_seq,
        query_positions,
        output,
        raw_workspace,
        score_workspace,
    )
]
artifact = tla.compile(
    qsa_indexer,
    *tensors,
    compress_ratio,
    topk_limit,
    query_heads,
    head_dim,
    page_size,
    kv_heads,
    True,
    rows,
    output_width,
    options="--npu-arch 3510",
)
artifact(*tensors, block_num=20)
torch.npu.synchronize()

print(output.shape, output.dtype)
print(output[:, -1])  # 每行有效索引个数
```

预期输出如下，说明样例执行成功（具体数值随输入与设备不同）：

```text
torch.Size([127, 192]) torch.int32
tensor([...], device='npu:0', dtype=torch.int32)
```

`output[:, -1]` 为每行有效索引个数，等于 `keep_i*CR + tail_i`：当 `visible_i ≤ K` 时为 `p_i + 1`，否则为 `K*CR + tail_i`。`output[i, :count]` 内是 token 下标序列：全可见路径按位置升序，top-k 路径按选中块的分数顺序、块内按 token 升序；`output[i, count:W-1]` 为 `-1`。

`block_num` 按可用 AI Core 数量设置；AIC 与 AIV 同时参与，取值过小会退化为少核串行执行，但不影响结果。
