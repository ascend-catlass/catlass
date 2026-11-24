# EmbeddingBag Example Readme

## 功能说明

- 算子功能：提供行级索引取行与定长 bag 求和两个入口，入口为 `embedding_bag` 模块的
  `run_index_select` / `run_embedding_bag`，纯 AIV kernel。两个入口共用同一个
  `row_gather_kernel`：编译期 `ACC=False` 时每个输出行直接搬一行，`ACC=True` 时先在 UB
  累加 `L` 行再写出。两者都先把行号当标量读入，再以整行 DMA 完成搬运。

- 计算公式：

  - `index_select`：按索引取行

    $$
    y[i, :] = x[idx[i], :]
    $$

  - `embedding_bag`：定长 bag 的嵌入求和

    $$
    out[b, :] = \sum_{l < L} weight[indices[b, l], :]
    $$

- 高性能设计：

  1. **整行 DMA 取代逐元素标量循环**：标量读一次行号，每个输出行发起一次 GM→UB 搬运
     （`index_select` 再发起一次 UB→GM 写出），访存由延迟受限转为带宽受限。
  2. **多缓冲流水**：`BUFS` 个行缓冲，每个由独立的 MTE2/VECTOR(/MTE3) mutex 保护；行 `i+1`
     的搬运与行 `i` 的累加/写出重叠，缓冲被消费 pipe 释放后才会被下一次 MTE2 复用。
  3. **UB 累加器**：`embedding_bag` 的第 0 行直接 DMA 进累加器，其余 `L-1` 行流水载入后
     在向量 pipe 上累加，每个 bag 只做一次输出 DMA，`L` 行源数据只读一遍，不产生中间 GM 张量。
  4. **跨 bag 复用累加器**：一个核处理多个 bag 时，VECTOR 在最后一次累加后释放累加器 mutex，
     MTE3 等到该释放后才开始输出 DMA，并在 DMA 后释放同一 mutex；下一个 bag 的 MTE2 重填
     因此排在上一次写出之后，整个 bag 循环无额外同步对象、无竞态。
  5. **尾部对齐**：UB 行按 64 lane 补齐，`D % 64 != 0` 时最后一个 64-lane chunk 用
     `update_mask` 只处理有效 lane，写回仍按 `D` 长度，GM 侧没有 padding 流量。
  6. **同步全部用 mutex**：行缓冲与累加器各一组 `tla.mutex`，不使用 `set_flag`/`wait_flag`。

- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
y   = run_index_select(x, idx)              # x[idx]
out = run_embedding_bag(weight, indices)    # 每个 bag 的定长求和
```

记 `R` 为表行数，`D` 为行宽，`M` 为取行数，`B` 为 bag 数，`L` 为每个 bag 的索引数。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `x` | Input | `[R, D]` | `float32` | index_select 源表，行优先连续 |
| `idx` | Input | `[M]` | 整型 | 行索引，host 统一转 `int32` |
| `y` | Output | `[M, D]` | `float32` | 取行结果 |
| `weight` | Input | `[R, D]` | `float32` | embedding_bag 嵌入表，行优先连续 |
| `indices` | Input | `[B, L]` | 整型 | 每个 bag 的 `L` 个索引，host 统一转 `int32` |
| `out` | Output | `[B, D]` | `float32` | 每个 bag 的求和结果 |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 维度 | `x`/`weight` 必须 2-D；`idx` 1-D；`indices` 2-D |
| dtype | 数据为 `float32`；索引为整型（host 转 `int32`） |
| 行宽 | `D` 任意正整数，包括 `D % 64 != 0`、`D * 4` 非 8/32 字节倍数的形状（100、33、7 等） |
| 索引范围 | `[0, R)`，host 侧校验，越界直接抛错 |
| bag 形态 | 只支持定长 bag（形状 `[B, L]`）；变长 / `offsets` 形态需调用方先做前缀和后发射 |
| 任务切分 | `index_select` 按行切给各核；`embedding_bag` 按 bag 轮转，`B` 可大于核数 |
| 累加顺序 | 行内按 64 lane 分块向量累加、行间按 `l = 0..L-1` 顺序；固定顺序保证同一输入结果可复现 |
| 精度 | `index_select` 与 CPU 参考实现逐位一致；`embedding_bag` 最大绝对误差 `L=8` 时 ≤ 5e-7、`L=17` 时 ≤ 2e-6 |
| UB 预算 | 累加器行 + 至少一个流水行必须放得进 UB；`D` 过大时 host 直接报错 |
| 同步模型 | 行缓冲与累加器各一组 mutex，无 `set_flag`/`wait_flag`；跨核无通信 |
| 编译 | 按 `(ACC, N, L, D)` 缓存 artifact |
| 执行 | 纯 AIV，`block_num = vector_core_num` |
| 硬件 | 编译选项固定 `--npu-arch 3510` |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from embedding_bag.embedding_bag import run_index_select, run_embedding_bag

x = torch.randn(128, 96, dtype=torch.float32, device="npu:0")
idx = torch.randint(0, 128, (37,), dtype=torch.int32, device="npu:0")
y = run_index_select(x, idx)
print(y.shape, y.dtype)

w = torch.randn(64, 48, dtype=torch.float32, device="npu:0")
ind = torch.randint(0, 64, (16, 5), dtype=torch.int32, device="npu:0")
out = run_embedding_bag(w, ind)
print(out.shape, out.dtype)
PY
```

样例文件自带自检（同上路径下执行 `python .../embedding_bag/embedding_bag.py`），
预期输出：

```text
index_select D=96 err: 0.0
embedding_bag D=96 err: 0.0
index_select D=100 err: 0.0
embedding_bag D=100 err: 4.76837158203125e-07
```

## 验证方法

- 数值基准：与 CPU 逐元素参考实现对照；`index_select` 逐位一致，`embedding_bag`
  最大绝对误差 `L=8` 时 ≤ 5e-7、`L=17` 时 ≤ 2e-6（浮点累加顺序差异）。
- 专项用例：`D = 1/7/8/31/32/33/63/64/65/96/100/127/128/129/300`、
  `M = 1/7/56/57/111/1000/4096`、`B = 1/16/17/55/56/57/64/112`、`L = 1/2/5/8/9/17`，
  覆盖索引重复、`B` 大于核数（一个核多个 bag）、尾 chunk 掩码。
- 确定性：同一输入连续执行 6 次，输出逐位一致。

## 性能结果

### 测量配置与口径

- 设备：Ascend 950PR，56 个 AIV；`Profiler` 采集 kernel 设备耗时。
- dtype 为 FP32；`R=50000`；下表只统计 kernel 设备耗时，不含 host 侧参数校验与编译时间。

| 场景 | 设备耗时（μs） |
| --- | ---: |
| embedding_bag B=1024, L=8, D=128 | 10.9 |
| embedding_bag B=1024, L=8, D=576 | 13.4 |
| embedding_bag B=16, L=8, D=128 | 1.8 |
| embedding_bag B=1, L=8, D=576 | 1.5 |
| index_select M=8192, D=128 | 5.4 |
| index_select M=2048, D=576 | 3.0 |
| index_select M=1000, D=100 | 2.5 |

### 结论

- 大 shape 由带宽主导：`index_select M=8192, D=128` 搬运 8 MiB，5.4 μs 折合约 1.55 TB/s；
  `embedding_bag B=1024, L=8, D=576` 搬运约 21 MiB，13.4 μs 折合约 1.66 TB/s。
- 小 shape（`B=1`、`B=16`）耗时 1.5–1.8 μs，已接近 kernel 启动下限，继续加深流水不再带来收益。
