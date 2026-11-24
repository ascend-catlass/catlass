# PagedKvGather Example Readme

## 功能说明

- 算子功能：按 paged KV Cache 的 block table，把 K/V 从分块缓存搬运到连续的 decode 输出布局。
  入口为 `paged_kv_gather` 模块的 `run` 函数，纯 AIV kernel：host 选取 tile 大小，kernel 以
  多 KiB DMA 搬整个 tile，尾块逐行处理，未命中位置由 kernel 写 0。
- 计算公式（逐个 `(batch b, position t)`）：

  $$
  blk = block\_table[b, \lfloor t / BS \rfloor], \qquad off = t \bmod BS
  $$

  $$
  out\_k[b, t, :] = cache\_k[blk, off, :], \qquad
  out\_v[b, t, :] = cache\_v[blk, off, :]
  $$

- 高性能设计：

  1. **Tile DMA**：一个 block table 表项覆盖 `BS` 个连续位置，tile 行数 `TR` 同时整除 `BS`
     与 `max_seq`，因此 tile 内源行在 cache 中连续、目的行在输出中连续。K/V 各用一条 DMA
     搬完 `TR x HD`，DMA 发射次数与 block table 标量装载次数降到 1/`TR`；`TR` 在 UB 允许时取最大。
  2. **多缓冲流水**：`BUFS` 组独立 K/V tile buffer，每组由自己的 mutex 保护。MTE2 装填
     slot `i+1` 与 MTE3 写出 slot `i` 重叠；slot 被 MTE3 释放后才会被下一次装填复用。
  3. **仅 UB 内补齐**：UB 行按 `N_PAD = ceil(HD / 64) * 64` 存放，尾块逐行搬运时行地址仍满足
     32Byte 对齐；GM 侧保持 dense，不产生额外补齐流量，`HD` 可取任意值。
  4. **尾块逐行回退**：每个序列只有最后一个 tile 可能部分有效，该 tile 逐行判断
     `t0 + r < seq_lens[b]` 后搬运，其余行从全零 UB tile 写出。
  5. **Kernel 内清零**：负表项、整块越界、尾块空洞都由 kernel 写 0，host 直接分配未初始化输出，
     省掉一次与输出等大的 memset。
  6. **核内同步全部用 mutex**：每个 tile slot、零 tile 各一组 `tla.mutex`；零 tile 的
     VECTOR→MTE3 交接由同一 mutex 的 lock/unlock 完成，不使用 `set_flag`/`wait_flag`。
  7. **按核均分 tile**：每核处理一段连续的 tile id，核内按 `BUFS` 组展开；`total` 不足核数时
     由边界判断自然跳过空核，不额外起核。

- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
out_k, out_v = paged_kv_gather.run(cache_k, cache_v, block_table, seq_lens, max_seq)
```

记 `NB` 为物理块数，`BS` 为每块行数，`HD` 为行宽（`num_kv_heads * head_dim`），`B` 为
batch 数，`MB` 为 block table 宽度，`S` 为输出序列长度 `max_seq`。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `cache_k` | Input | `[NB, BS, HD]` | `float32` | 分页 K 缓存，`HD` 为 `num_kv_heads * head_dim` |
| `cache_v` | Input | `[NB, BS, HD]` | `float32` | 分页 V 缓存，与 `cache_k` 形状一致 |
| `block_table` | Input | `[B, MB]` | `int32` | 每个 batch 的逻辑块到物理块映射，负值表示无效 |
| `seq_lens` | Input | `[B]` | `int32` | 每个 batch 的有效长度 |
| `max_seq` | 属性 | 标量 | Python `int` | 输出序列维长度，参与编译 |
| `out_k` | Output | `[B, max_seq, HD]` | `float32` | 连续的 K 结果 |
| `out_v` | Output | `[B, max_seq, HD]` | `float32` | 连续的 V 结果 |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 维度 | `cache_k`/`cache_v` 必须 3-D 且形状相同；`block_table` 2-D；`seq_lens` 1-D |
| dtype | 缓存与输出为 `float32`；`block_table`/`seq_lens` 为 `int32`（host 自动转换） |
| 序列长度 | `0 < max_seq <= MB * BS`；`seq_lens[b]` 可小于或大于 `max_seq`，超过部分不写 |
| `HD` | 任意正整数，包括 `HD * 4` 不是 32Byte 倍数的形状（100、33、7 等）；UB 行按 64 lane 补齐，GM 侧 dense |
| 尾块 | `TR` 由 host 取 `BS` 与 `max_seq` 的公约数，tile 不跨 block table 表项、不跨序列末尾；不满足时退化为逐行（`TR = 1`） |
| 未命中语义 | `t >= seq_lens[b]`、`block_table` 为负、整块超出序列的位置输出 0，由 kernel 写入 |
| 精度 | 输出与逐位置参考实现逐位一致：命中行为精确拷贝，未命中位置为 0 |
| UB 预算 | 零 tile + 一组 K/V tile 必须放得进 UB；`HD` 过大时 host 直接报错 |
| 同步模型 | tile slot 与零 tile 各一组 mutex，无 `set_flag`/`wait_flag`；跨核无通信 |
| 编译 | 按 `(NB * BS, HD, B, MB, max_seq)` 缓存 artifact；`TR/BUFS/N_PAD` 由形状推导 |
| 执行 | 纯 AIV，`block_num = vector_core_num` |
| 硬件 | 编译选项固定 `--npu-arch 3510` |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from paged_kv_gather.paged_kv_gather import run

nb, bs, hd = 8, 16, 128
bsz, max_seq = 4, 40
cache_k = torch.randn(nb, bs, hd, dtype=torch.float32, device="npu:0")
cache_v = torch.randn(nb, bs, hd, dtype=torch.float32, device="npu:0")
block_table = torch.randint(0, nb, (bsz, 3), dtype=torch.int32, device="npu:0")
seq_lens = torch.tensor([40, 17, 33, 1], dtype=torch.int32, device="npu:0")

out_k, out_v = run(cache_k, cache_v, block_table, seq_lens, max_seq)
print(out_k.shape, out_v.shape)
PY
```

样例文件自带自检（同上路径下执行 `python .../paged_kv_gather/paged_kv_gather.py`），
预期输出：

```text
k max abs err: 0.0
v max abs err: 0.0
```

## 验证方法

- 数值基准：与 CPU 逐位置参考实现对照（按 block table 与 `seq_lens` 计算每个 `(b, t)` 的源行），
  输出 K/V 逐位一致。
- 专项用例：`HD = 1/7/33/63/100/129/1000/4096`（含非 32Byte 对齐与超大行宽）、
  `BS = 1/2/3/8/16/32/128/1024`、`max_seq` 不整除 `BS`、`seq_lens` 为 0/部分/全量、
  `block_table` 部分为负与全为负、`B = 1/64`（B 大于核数）。
- 确定性：同一输入连续执行，输出逐位一致。

## 性能结果

### 测量配置与口径

- 设备：Ascend 950PR，56 个 AIV；`Profiler` 采集 kernel 设备耗时。
- 形状：`B=16`，`BS=128`，dtype 为 FP32；`S=8192` 时约 128K 行。
- 下表只统计 kernel 设备耗时，不含 host 侧参数校验与编译时间。

| 形状 | 设备耗时（μs） |
| --- | ---: |
| S=8192, HD=128 | 155 |
| S=8192, HD=100 | 116 |
| S=8192, HD=576 | 805 |
| S=4096, HD=128 | 21 |
| S=4096, HD=128, `seq_lens` 递减 | 30 |

### 结论

- 带宽主导：`HD=576` 时 K/V 读入与写出合计约 1.2 GB，805 μs 折合等效带宽约 1.5 TB/s。
- tile 化收益：`BS=128`、`S=8192` 时 `TR` 取 128（`HD=128`）或 32（`HD=576`），DMA 发射与
  block table 标量装载次数相应降到 1/128 或 1/32。
- 非 32Byte 对齐的 `HD`（如 100）只在 UB 内补齐，不产生额外 GM 流量。
