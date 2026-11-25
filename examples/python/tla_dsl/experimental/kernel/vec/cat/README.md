# Cat Example Readme

## 功能说明

- 算子功能：`cat(xs, dim)` 等价 `torch.cat(xs, dim)`，支持二维、行主序的
  `float32` 张量，`dim` 为 0 或 1。
- 计算公式：

  $$
  \begin{aligned}
  \operatorname{cat}_0([x_0,\ldots,x_{k-1}])
    &= \begin{bmatrix} x_0 \\ \vdots \\ x_{k-1} \end{bmatrix}, \\
  \operatorname{cat}_1([x_0,\ldots,x_{k-1}])
    &= \begin{bmatrix} x_0 & \cdots & x_{k-1} \end{bmatrix}.
  \end{aligned}
  $$

  `dim=0` 要求所有输入列数相同，`dim=1` 要求所有输入行数相同。

- 实现方式：输入先由 MTE2 搬入 UB，再由 MTE3 写入输出。对于不超过 3 个输入，
  每个方向只启动一次融合 kernel；超过 3 个输入时退化为逐输入启动。
- 同步方式：每个 UB staging buffer 绑定一个 mutex，MTE2 完成搬入后交给 MTE3
  搬出，不使用 `set_flag`、`wait_flag` 或全 pipe barrier。
- 支持产品型号：Ascend 950PR、950DT 系列产品。当前在 Ascend 950PR、CANN 9.2.0
  环境完成验证。

## 参数说明

```python
from cat.cat import run

out = run([a, b], dim=0)   # [M1 + M2, N]
out = run([a, b], dim=1)   # [M, N1 + N2]
```

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `xs` | Input | `k` 个二维张量 | `float32` | 非空列表；`dim=0` 时列数相同，`dim=1` 时行数相同 |
| `dim` | Input | 标量 | Python `int` | 仅支持 0 或 1 |
| `out` | Output | `[ΣM, N]` 或 `[M, ΣN]` | `float32` | 连续行主序张量 |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 输入列表 | `xs` 至少包含一个张量 |
| 维度 | 仅支持二维张量 |
| dtype | 输入和输出均为 `float32` |
| `dim=0` shape | 所有输入为 `[M_i, N]`，列数 `N` 相同；`M_i` 可为 0 |
| `dim=1` shape | 所有输入为 `[M, N_i]`，行数 `M` 相同；`M`、`N_i` 可为 0 |
| `dim=1` 单行宽度 | 每个输入的 `N_i <= 49152` 个 FP32 元素，即一行不超过 192 KiB UB 空间 |
| `dim=0` 宽度 | 列数不受 UB 空间限制，输入按连续元素分块搬运 |
| 空张量 | 零元素输入不启动 kernel；全部为空时直接返回空输出 |
| 非连续输入 | 入口自动执行 `.contiguous()`，拷贝成本计入调用耗时 |
| 设备 | 所有输入必须位于同一 NPU |
| 对齐 | GM 偏移、DMA 长度和行跨度无需 32 B 对齐；UB 暂存区按 256 B 对齐 |
| launch | `k <= 3` 时一次 launch；`k > 3` 时每个非空输入一次 launch |
| 多核 | 按连续块或行组使用 `tla.range(gid, work, aiv_num)` 切分到 AIV 核 |
| 同步 | 核内使用 UB mutex 的 MTE2→MTE3 接力；无跨核同步和原子写 |
| 编译 | 按 shape、段数、chunk 和行批参数缓存编译产物 |
| 流切换 | host 缓存绑定到创建 plan 时的 stream；切换 `torch.npu.stream` 后调用 `reset_host_cache()` |

## 代码组织

```text
cat/
├── cat.py
└── README.md
```

| 文件 | 说明 |
| --- | --- |
| [`cat.py`](cat.py) | kernel、host 编排、launch 缓存、`run()` 入口和自检 |
| [`README.md`](README.md) | 功能、约束、使用方式和性能说明 |

## 高性能设计

- **单 launch 多输入**：`k <= 3` 时多个输入共享一次 AIV launch，避免 host launch
  次数随输入数线性增长。
- **32 KiB DMA**：`dim=0` 每块搬运 8192 个 FP32 元素，兼顾 AIV wave 数、DMA
  在途字节和尾块成本。
- **二维跨步搬运**：`dim=1` 一次搬入 `BR` 行，再按输出行跨度写回，把多次逐行
  DMA 合并为一次二维搬运。
- **行批上限**：
  `BR = max(1, min(64, 49152 // N_i, M))`，同时受 UB、行数和固定批量上限约束。
- **尾块复用**：`dim=0` 的不足一块部分由单核补短 DMA；`dim=1` 的 `M % BR`
  行逐行处理，并与主路径复用同一 UB buffer 和 mutex。
- **快速 host 路径**：张量地址、版本和 device 命中后直接使用预打包 launch
  payload；TLA binding 和 payload 只在缓存 miss 时构建。
- **无 GM 中间量**：每个输入只读一次、输出只写一次，理论流量为
  `2 × sum(x_i.nbytes)`。

## 使用示例

在 CATLASS 仓库根目录执行：

```bash
export PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH
python examples/python/tla_dsl/experimental/kernel/vec/cat/cat.py --device 0
```

Python 调用示例：

```python
import torch
import torch_npu  # noqa: F401

from cat.cat import run

a = torch.randn(37, 1001, device="npu:0")
b = torch.randn(23, 1001, device="npu:0")
out0 = run([a, b], dim=0)

c = torch.randn(37, 250, device="npu:0")
d = torch.randn(37, 1001, device="npu:0")
out1 = run([c, d], dim=1)
```

内置自检同时检查结果与 `torch.cat` 是否逐位一致，示例输出：

```text
cat0 ((37, 1024), (23, 1024)) passed=True
cat0 ((37, 1001), (23, 1001)) passed=True
cat0 ((1, 3), (2, 3)) passed=True
cat0 ((0, 8), (5, 8)) passed=True
cat0 ((13, 4097), (7, 4097)) passed=True
cat1 ((37, 512), (37, 512)) passed=True
cat1 ((37, 250), (37, 1001)) passed=True
cat1 ((5, 1), (5, 63)) passed=True
cat1 ((9, 999), (9, 3), (9, 17)) passed=True
cat1 ((13, 4096), (13, 1)) passed=True
```

## 精度结果

- 开发验证覆盖 14 个基础用例和 18 个对齐/尾块专项用例，全部与 `torch.cat`
  逐位一致。
- 对齐专项包含 15 个总字节数非 32 B 倍数的用例和 14 个实际尾块用例。
- 覆盖 4/8/12/32/40 B 总长度、所选 32 KiB chunk 的精确边界、`+1/+3` 元素尾块、
  非对齐列偏移、空行和最大单行宽度。

## 性能结果

测量设备为 Ascend950PR，参考实现为 NPU 输入上的 `torch.cat`。warmup 20、
trials 3，每个 trial 连续执行 500 次后求均值，计时包含 `torch.empty`、输入检查、
当前 stream 查询和 host launch。

在最终的多流安全版本上，14 个开发用例的 suite 为：本算子 `0.007191 ms`，
`torch.cat` `0.008217 ms`，加速比 **1.143×**。其中 9/14 个用例快于参考，
其余用例受固定 host 开销影响，单例结果会随设备和系统负载波动。

性能设计重点是把 GM 往返次数、launch 次数和无效 scalar 地址计算压低；
样例本身不作为跨设备性能承诺。开发过程中大 shape kernel 的 profile 约为
`6.98 µs`，混合宽度小 shape kernel 约为 `3.19 µs`。
