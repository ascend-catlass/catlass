# Pad Example Readme

## 功能说明

本样例实现二维 `float32` 张量最后一维的常量填充，入口为 `pad.py` 的 `pad` 函数：

```text
out[m, n] = value,              0 <= n < left
out[m, n] = x[m, n-left],       left <= n < left+N
out[m, n] = value,              left+N <= n < left+N+right
```

输出为稠密的 `[M, N+left+right]` 张量，等价于：

```python
torch.nn.functional.pad(x, (left, right), value=value)
```

- 支持任意非负 `M`、`left`、`right`；非空输入需满足下面的 UB 行宽约束。
- `M == 0` 或 `N == 0` 时由 host 直接生成输出，不启动 kernel。
- GM 写支持任意元素偏移和任意字节长度，不要求 `N`、`left`、`right`、`W` 为 32 Byte
  对齐。
- 支持 Ascend 950PR&950DT 系列产品，编译目标为 `--npu-arch 3510`。

## 高性能设计

- **按行组批量 DMA**：每个 AIV block 一次处理最多 32 行，payload 和两侧 pad 都按行组
  批量搬运，减少小 DMA 和同步次数。
- **payload 双缓冲**：两个独立的 `BR × N` UB 槽按行组交替使用；后一槽的 MTE2 load
  可与前一槽的 MTE3 store 重叠。
- **三段不重叠 GM 写**：不重建整条输出行，而是分别写左 pad、payload 和右 pad，避免
  非 32 Byte 对齐的 UB vector store。
- **pad 缓冲一次填充**：左右 pad 缓冲在每个 block 开头各填充一次，之后所有行组复用。
- **非对齐尾块**：UB pad 行宽向上取整到 64 个 FP32 lane；尾行单独处理，不读写有效区外
  的 GM。
- **连续行组划分**：每个 block 领取一段连续行组，再在 block 内两两配对，减少
  HBM 长距离交错；各 block 无跨核归约。
- **复杂度**：kernel 为 `O(M * N)`；每行 payload 只读一次、写一次。

## 参数说明

```python
out = pad(x, left, right, value=0.0)
```

| 参数 | 方向 | 形状 / 类型 | 说明 |
| --- | --- | --- | --- |
| `x` | Input | `[M, N]`, `float32` | 输入；非连续输入在内部转为连续张量 |
| `left` | Input | Python `int` | 左侧填充宽度，必须 `>= 0` |
| `right` | Input | Python `int` | 右侧填充宽度，必须 `>= 0` |
| `value` | Input | Python `float` | 填充值，参与编译期特化 |
| 返回值 | Output | `[M, N+left+right]`, `float32` | 稠密填充结果 |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 维度 | 仅支持二维张量，只填充最后一维 |
| dtype | 输入和输出仅支持 `float32` |
| padding | `left`、`right` 必须为非负整数 |
| 行宽 | `N > 0` 时必须满足 `N + PLA + PRA <= 30720`，其中 `PLA/PRA` 分别是 64 lane 向上取整后的填充缓冲行宽；超出时抛 `ValueError` |
| 对齐 | GM 偏移和长度无需 32 Byte 对齐；UB 起始地址按 256 Byte 对齐 |
| UB 双槽 | 总 staging 预算 240 KiB，每槽约 120 KiB，另留 256 元素 guard |
| 行组 | `BR <= 32`；低 `M` 时按 AIV 数自动减小 `BR` |
| 编译 | 按 `(M, N, W, left, right, value, BR)` 缓存编译产物 |
| 设备 | 纯 AIV；block 数取 `vector_core_num` 并按形状裁剪 |

## 代码组织

```text
pad
├── README.md
├── bench_pad.py  # 与 torch.nn.functional.pad 配对计时
└── pad.py        # kernel、host wrapper 和精度自检
```

## 使用示例

以下命令在仓库根目录执行：

```bash
source <CANN_INSTALL_PATH>/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
timeout -k 5 60s python \
    examples/python/tla_dsl/experimental/kernel/vec/pad/pad.py \
    --device 0 --shard 0/5
```

完整自检包含 10 组形状 × 20 个随机种子，共 200 例；可拆成 5 片执行：

```bash
for shard in 0 1 2 3 4; do
  timeout -k 5 60s python \
      examples/python/tla_dsl/experimental/kernel/vec/pad/pad.py \
      --device 0 --shard "${shard}/5"
done
```

在 Python 中调用：

```python
import torch
import torch_npu  # noqa: F401

from examples.python.tla_dsl.experimental.kernel.vec.pad.pad import pad

x = torch.randn(37, 1001, dtype=torch.float32, device="npu:0")
out = pad(x, 3, 5, value=-2.5)
```

性能对比：

```bash
timeout -k 5 60s python \
    examples/python/tla_dsl/experimental/kernel/vec/pad/bench_pad.py \
    --device 0 --m 16384 --n 2048 --left 3 --right 5
```

## 实现说明

- 每个 block 领取连续行组区间，输出行之间没有数据依赖，因此不需要跨 block 归约或
  全局屏障。
- `n_groups = M // BR` 的整行组走双槽流水；`M % BR` 的尾行由各 block 交错领取并单独
  搬运，避免尾行集中到单个 block。
- payload 两个 UB 槽分别对应 `payload_stage0` 和 `payload_stage1` mutex。后一槽的
  MTE2 load 可以排在前一槽的 MTE3 store 之后提交，二者不需要共用同一把锁。
- 左右 pad 各自的 VECTOR 生产者和 MTE3 消费者使用独立 mutex。消费者锁覆盖整个只读
  生命周期，不在每个行组重复加锁。
- `left == 0` 或 `right == 0` 时，对应 pad 缓冲、填充循环和 DMA 在编译期消除。
- host 按完整调用参数缓存编译产物和 launch plan；重复调用只更新输入、输出地址。

## 验证结果

在 Ascend 950PR、单卡、`ascend-catlass-dsl 2.2.1`、`torch_npu 2.10.0.post7`
环境完成以下验证：

- 精度：10 组形状 × 20 个随机种子，共 200/200 例与
  `torch.nn.functional.pad` 逐元素一致。形状覆盖 `M=1/5/9/11/13/31/33/37`、
  `N=1/7/63/999/1000/1001/4096`，padding 覆盖零填充、单侧填充、两侧填充、
  `left=65`、`right=66` 和多字节非对齐尾块。
- 非对齐专项：12/12 例通过，包括 `N=1/3/7/9/63/65/127/999/1001`、奇数 padding
  偏移，以及 `M=64/65/67` 的整行组与尾行边界。
- 性能：`(16384, 2048), left=3, right=5`、warmup 1、trials 5 的官方 paired benchmark
  为 candidate `0.18616 ms`、`torch.nn.functional.pad` `0.18582 ms`，约 `0.9982x`；
  相较优化前 `0.20230 ms` 提升约 `8.0%`。profiling 中 MTE3 利用率约 `90%`，主流
  搬运已接近停顿下限。单卡计时仍未稳定超过 Torch，性能结论仅限上述环境。

## 验证要求

- 精度使用 `torch.equal` 校验，要求与 Torch reference 逐元素一致。
- 至少覆盖零 padding、单侧 padding、双侧 padding、空行、空列和非对齐行宽。
- 单例执行必须使用 `timeout -k 5 60s`；超过 60 秒按失败处理。
- 更换目标架构时必须同步修改 `--npu-arch`。
