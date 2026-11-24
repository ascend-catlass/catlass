# QuantLightningIndexer Example Readme

## 功能说明

- 算子功能：quant_lightning_indexer 是稀疏注意力的索引器。对每个 query token，先在分页的 MXFP4 压缩 KV 上计算 32 个 query head 的 `q·k` 分数，按 head 做 ReLU 加权 BF16 归约得到每个 key 的 indexer 分数；再在可见 key 范围内取 Top-K，输出 key 下标与分数。本目录是 CATLASS DSL 实现，两个 kernel 定义在同一个文件 `quant_lightning_indexer.py` 中：S1 打分（AIC/AIV mixed kernel）与 S2 Top-K（单个融合 AIV kernel）。S2 的 key 构建、两级阈值、计数与发射四段在同一个 kernel 内以 3 次 `tla_sync_all()` 分隔，不再拆成多个 kernel。
- 计算公式：记第 `r` 个 query 行属于 batch `b(r)`，行内序号为 `l(r)`，query/key 长度为 `lq_b`、`lk_b`，压缩比 `CR = cmp_ratio`，页大小 `P = page_size`。

  MXFP4（E2M1）解码：每个 uint8 打包 2 个 4 bit code，低半字节在前；E8M0 scale 每 32 个逻辑特征共用一个字节。

  $$
  v_d = \operatorname{sgn}(c_d)\,m_{c_d \bmod 8}\,2^{e_{\lfloor d/32 \rfloor}-127},
  \qquad
  \operatorname{sgn}(c)=
  \begin{cases}
  +1, & c < 8 \\
  -1, & c \ge 8
  \end{cases}
  $$

  其中 `c_d` 为第 `d` 个 4 bit code，`m = [0, 0.5, 1, 1.5, 2, 3, 4, 6]`，`e` 为对应分组的 E8M0 字节。

  每个 query head 的 `q·k` 在 FP32 累加后舍入到 BF16；随后按 head 依次做「ReLU → 乘 BF16 权重 → 累加并按 BF16 舍入」：

  $$
  qk_{r,h,s} = \mathrm{bf16}\!\left(\sum_{d=0}^{127} v^{q}_{r,h,d}\,v^{k}_{s,0,d}\right)
  $$

  $$
  a^{(0)}_{r,s}=0,\qquad
  a^{(h+1)}_{r,s}=\mathrm{bf16}\!\left(a^{(h)}_{r,s}+\max(qk_{r,h,s},0)\cdot \mathrm{bf16}(w_{r,h})\right),
  \qquad
  score_{r,s}=a^{(32)}_{r,s}
  $$

  可见 key 数由 `mask_mode` 决定，`mask_mode=0` 时全部可见，`mask_mode=3` 时为压缩因果可见：

  $$
  visible_r =
  \begin{cases}
  lk_{b(r)}, & mask\_mode = 0 \\
  \min\!\left(lk_{b(r)},\ \max\!\left(0,\left\lfloor
  \dfrac{CR\cdot lk_{b(r)} + res_{b(r)} - lq_{b(r)} + l(r) + 1}{CR}
  \right\rfloor\right)\right), & mask\_mode = 3
  \end{cases}
  $$

  Top-K 只在前 `visible_r` 个 key 内选择（`block_table` 中无效页也不参与）：

  $$
  take_r = \min(topk,\ visible_r),\qquad
  S_r = \operatorname*{arg\,topk}_{s < visible_r} score_{r,s}
  $$

  `S_r` 按分数从大到小写入 `out_idx[r, 0:take_r]`（同分时索引集合任意），提供 `offsets` 时逐元素相加；不足 `topk` 的槽位写 `-1`，对应 `out_val` 写 `-inf`，其余写对应 BF16 分数。

- 支持产品型号：Ascend 950PR&950DT 系列产品（DAV_3510 / arch35）。

## 参数说明

kernel 直接以静态 shape 编译，两个入口如下：

```python
qli_scores(
    q, k, sq, sk, w, block_table, scores,
    page_size, rows, query_heads, head_dim, tile_rows,
)

qli_topk(
    scores, visible, offsets, keys, rpart, scal, cpart, out_idx, out_val,
    rows, max_k, topk,
)
```

记 `rows` 为 query token 数，`pages` 为 `block_table` 的页数，`max_k = pages × page_size`。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `q` | Input | `[rows*32, 128]` | `float4_e2m1` | MXFP4 打包的 Q；每行 32 head × 64 B |
| `k` | Input | `[128, pages*page]` | `float4_e2m1` | 列优先打包的 K；只有 1 个 KV head |
| `sq` / `sk` | Input | `[rows*32, 4]` / `[4, pages*page]` | `float8_e8m0` | Q/K 的 E8M0 分组 scale |
| `w` | Input | `[rows, 32]` | `float32` | head 权重，kernel 内先按 BF16 舍入 |
| `block_table` | Input | `[1, pages]` | `int32` | 逻辑页到物理页映射；只读第 0 行 |
| `scores` | Output | `[rows, pages*page]` | `bfloat16` | S1 写、S2 读的分数矩阵 |
| `page_size` | `int` | — | — | 每页 key 数 `P` |
| `rows` | `int` | — | — | query 行数 |
| `query_heads` | `int` | — | — | 固定 32 |
| `head_dim` | `int` | — | — | 固定 128（逻辑维，打包 64 B） |
| `tile_rows` | `int` | — | — | 每个 cube tile 的 query 行数，固定 4 |
| `visible` | Input | `[rows]` | `int32` | 每行可见 key 数，由 `mask_mode`/`cmp_ratio` 在调用前算出 |
| `offsets` | Input | `[rows]` | `int32` | 索引偏移；无偏移时传全 0 |
| `keys` | Workspace | `[rows, max_k]` | `uint16` | 单调 key workspace |
| `rpart` | Workspace | `[rows, 8, 512]` | `int32` | 两级 CDF 的分片 partial |
| `scal` | Workspace | `[rows, 8]` | `int32` | 阈值与计数标量 |
| `cpart` | Workspace | `[rows, 2*max_k/128]` | `int16` | 每 chunk 的 `>thr` / `==thr` 计数 |
| `out_idx` / `out_val` | Output | `[rows, topk]` | `int32` / `bfloat16` | Top-K 索引与分数 |
| `max_k` | `int` | — | — | 分数矩阵列数，`max_k % 128 == 0` |
| `topk` | `int` | — | — | 每行保留的 key 数，`≤ max_k` |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 产品 | Ascend 950PR&950DT 系列产品（DAV_3510） |
| batch | 单 batch；`block_table` 只读第 0 行 |
| Q 布局 | 仅 TND 等价的行优先打包，`q` 的逻辑 shape 为 `[rows, 32, 64]` 字节 |
| K 布局 | 仅分页 PA；逻辑块 `b` 映射到 `block_table[0, b / P]` 的第 `b % P` 个槽位 |
| 页大小 | `page_size ≤ 128`；epilogue 按 64 列链展开，支持 `(page_size+63)//64 ∈ {1, 2}`（64/128 已验证） |
| head 与 D | 固定 32 个 query head、1 个 K head、逻辑 `head_dim = 128` |
| 参与核数 | S1 要求 `block_num ≥ ceil(rows / tile_rows)` 且 `groups = block_num // tiles_q ≥ 1`；S2 要求 `rows ≤ 参与 AIV 数` |
| S1 UB | 两个 `raw_ub` 槽共 `2 × 128 × page_size × 4 B`（`page_size=128` 时为 128 KiB），加上 `score_ub` 与权重需落在单 AIV UB 内 |
| `max_k` | `max_k % 128 == 0`；`max_k > 4096` 时须为 `4096` 的整数倍，`max_k > 16384` 时须为 `16384` 的整数倍，否则尾部分块不参与统计 |
| `topk` | `≤ max_k`；`visible < topk` 时只写 `visible` 个有效索引，其余槽位 `-1` / `-inf` |
| `visible` | 每行 `0 ≤ visible[row] ≤ max_k`；`visible=0` 时该行全部索引 `-1`、value `-inf` |
| `block_table` 无效页 | 项为 `-1` 或越界时该页不读取，其 key 不参与 Top-K |
| 同分 key | 允许索引集合内任意选择；验收按「允许集合 + 第 K 名边界分数」比较，不要求顺序 |
| candidates / return_value | 本 DSL 实现只输出 `indices` 与 `values`，不输出 candidates/lengths，也没有 `return_value` 开关 |
| 编译期常量 | 所有 shape 与属性均为编译期常量；`rows`、`page_size`、`max_k`、`topk`、`tile_rows` 变化需重新 `tla.compile` |

### 同步约束

| 项目 | 约束 |
| ----- | -------- |
| 核内同步 | S1 关闭 `auto_sync="v0"`，每个片上 buffer 用显式 `tla.mutex` 保护（`id` 必须显式写 `0..31`，S1 使用 0..9） |
| 互斥范围 | `tla.mutex` 只在单个 AIC 或 AIV 内有效；AIC↔AIV 的页交接必须用 `tla.cross_flag(mode=4)` 并显式给 `aiv_id=0/1` |
| 事件配对 | cross flag 是一次性事件，每次 `set` 必须对应一次 `wait`；S1 在循环前只预置两个 `free` 槽，页面循环按 `for slot in (0, 1)` 在编译期展开，保证 flag 与 mutex 不依赖运行期下标 |
| 不能混用 | `auto_sync="v0"` 不能与手写 `mutex`/`mutex_guard` 或 `tla.call_extern` 共存；融合 S2 因使用 `call_extern(SyncAll)` 必须关闭 `auto_sync` |
| `mutex_guard` | 只识别 `tla.copy`/`tla.mmad`/`tla.vec.func`，`mmad_mx` 不在此列，CUBE 侧使用显式 `lock`/`unlock` |
| 段间交接 | 融合 S2 用 3 次 `tla_sync_all()`（`AscendC::SyncAll`）代替 launch 边界；`SyncAll` 之后跨 block 的标量 GM 写不可见，`rpart`/`cpart`/`scal` 的交接必须走 DMA（`tla.copy`），不能直接读标量 GM |
| 段内顺序 | 融合 S2 每处 `copy`↔`vec`、段间 DMA 前后都有显式 `tla.pipe_barrier(tla.pipes.ALL)` |

## 使用示例

本目录只包含 kernel 定义，没有内置的命令行样例；调用方需自行准备输入、workspace 并编译执行。准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境后，可按下面的方式调用：

```python
import torch
import torch_npu  # noqa: F401

import catlass.tla as tla

from quant_lightning_indexer import qli_scores, qli_topk

rows, query_heads, head_dim, page_size, topk = 12, 32, 128, 128, 512
lk = 131072
pages = lk // page_size
max_k = pages * page_size

# 输入：MXFP4 打包字节 + E8M0 scale；这里只演示形状与 dtype，实际取真实数据。
q = torch.randint(0, 256, (rows * 32, 64), dtype=torch.uint8, device="npu:0")
k = torch.randint(0, 256, (pages * page_size, 64), dtype=torch.uint8, device="npu:0")
sq = torch.full((rows * 32, 4), 127, dtype=torch.uint8, device="npu:0")
sk = torch.full((pages * page_size, 4), 127, dtype=torch.uint8, device="npu:0")
w = torch.rand(rows, 32, dtype=torch.float32, device="npu:0")
table = torch.arange(pages, dtype=torch.int32, device="npu:0").reshape(1, pages)
scores = torch.empty(rows, max_k, dtype=torch.bfloat16, device="npu:0")

q_t = tla.from_dlpack(
    q, layout_tag=tla.arch.RowMajor,
    origin_shape=(rows * 32, 128), element_type=tla.Float4E2M1,
).mark_layout_dynamic()
k_t = tla.from_dlpack(
    k, layout_tag=tla.arch.ColumnMajor,
    origin_shape=(128, max_k), element_type=tla.Float4E2M1,
).mark_layout_dynamic()
sq_t = tla.from_dlpack(
    sq, layout_tag=tla.arch.RowMajorMxScaleA,
    origin_shape=(rows * 32, 4), element_type=tla.Float8E8M0,
).mark_layout_dynamic()
sk_t = tla.from_dlpack(
    sk, layout_tag=tla.arch.ColMajorMxScaleB,
    origin_shape=(4, max_k), element_type=tla.Float8E8M0,
).mark_layout_dynamic()
w_t = tla.from_dlpack(w, layout_tag=tla.arch.RowMajor).mark_layout_dynamic()
table_t = tla.from_dlpack(table, layout_tag=tla.arch.RowMajor).mark_layout_dynamic()
scores_t = tla.from_dlpack(scores, layout_tag=tla.arch.RowMajor).mark_layout_dynamic()

s1 = tla.compile(
    qli_scores, q_t, k_t, sq_t, sk_t, w_t, table_t, scores_t,
    page_size, rows, query_heads, head_dim, 4,
    options="--npu-arch 3510",
)
s1(q_t, k_t, sq_t, sk_t, w_t, table_t, scores_t, block_num=28)

# S2 workspace / 输出
visible = torch.full((rows,), max_k, dtype=torch.int32, device="npu:0")
offsets = torch.zeros(rows, dtype=torch.int32, device="npu:0")
keys = torch.empty(rows, max_k, dtype=torch.uint16, device="npu:0")
rpart = torch.empty(rows, 8, 512, dtype=torch.int32, device="npu:0")
scal = torch.empty(rows, 8, dtype=torch.int32, device="npu:0")
cpart = torch.empty(rows, 2 * max_k // 128, dtype=torch.int16, device="npu:0")
out_idx = torch.full((rows, topk), -1, dtype=torch.int32, device="npu:0")
out_val = torch.full((rows, topk), float("-inf"), dtype=torch.bfloat16, device="npu:0")

vis_t = tla.from_dlpack(visible, layout_tag=tla.arch.RowMajor)
off_t = tla.from_dlpack(offsets, layout_tag=tla.arch.RowMajor)
keys_t = tla.from_dlpack(keys, layout_tag=tla.arch.RowMajor)
rpart_t = tla.from_dlpack(rpart, layout_tag=tla.arch.RowMajor)
scal_t = tla.from_dlpack(scal, layout_tag=tla.arch.RowMajor)
cpart_t = tla.from_dlpack(cpart, layout_tag=tla.arch.RowMajor)
idx_t = tla.from_dlpack(out_idx, layout_tag=tla.arch.RowMajor)
val_t = tla.from_dlpack(out_val, layout_tag=tla.arch.RowMajor)

s2 = tla.compile(
    qli_topk, scores_t, vis_t, off_t, keys_t, rpart_t, scal_t, cpart_t,
    idx_t, val_t, rows, max_k, topk,
    options="--npu-arch 3510",
)
s2(scores_t, vis_t, off_t, keys_t, rpart_t, scal_t, cpart_t, idx_t, val_t, block_num=56)
torch.npu.synchronize()

print(scores.shape, scores.dtype)
print(out_idx[:, :8], out_val[:, :8])
```

预期输出如下，说明两段 kernel 均已执行完成（具体数值随输入与设备不同）：

```text
torch.Size([12, 131072]) torch.bfloat16
tensor([[...]], device='npu:0', dtype=torch.int32) tensor([[...]], device='npu:0', dtype=torch.bfloat16)
```

`out_idx` 每行前 `take_r` 项为 Top-K key 下标（已加 `offsets`），其余为 `-1`；`out_val` 为对应 BF16 分数，其余为 `-inf`。

## 性能数据

本机实测（Ascend 950，`rows=12`、`page=128`、PA、`mask_mode=0`、`topk=512`，wall/iter，
含下发）：

| 配置 | S1 打分 | S2 Top-K | S1+S2 |
| ----- | -----: | -----: | -----: |
| 8K keys | 16 us | — | — |
| 32K keys | 27 us | — | — |
| 128K keys | 85 us | 95 us | 180 us |

S1 的同步方式对比（128K keys）：`auto_sync="v0"` 自动 mutex 为 131 us，改为显式
`tla.mutex` 后为 85 us。收益来自两点：每页不再需要一次 `pipe_barrier(ALL)`（改由
`score_ub` 槽位 mutex 表达 VECTOR→MTE3 的 RAW/WAR），以及 `raw_ub` 在 SIMD 段结束
后立即释放，使 AIC 可以提前写下一页。

## 实现与目录

| 文件 | 说明 |
| ----- | ----- |
| `quant_lightning_indexer.py` | S1：MXFP4 QK + 逐 head ReLU 加权 BF16 归约（AIC/AIV mixed）；S2：单调 key + 两级 CDF 阈值 + 计数/发射，融合为单个纯 AIV kernel `qli_topk`（3 次 `SyncAll` 串联四段） |
