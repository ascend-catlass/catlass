# Histc Example Readme

## 功能说明

- 算子功能：统计一维张量 `x` 在区间 `[min, max]` 上按等宽桶划分的频次直方图；基于
  CATLASS Python 模板实现的纯 AIV 多核 kernel。
- 计算公式：

  $$
  out[k] = \sum_{i=0}^{N-1} \left[\, k \le \frac{(x_i - min) \cdot bins}{max - min} < k+1 \,\right],
  \quad k = 0, \dots, bins-1
  $$

  即落到 `[min, max]` 之外的元素不计入任何桶，落在 `max` 上的元素计入最后一个桶。
  `min == 0` 且 `max == 0` 时按输入的实际 `[min, max]` 自动定界（autorange）。

- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
out = histc.run(x, bins, min, max)
```

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[N]` | `float32` / `float16` | 一维行优先连续张量 |
| `bins` | Input | 标量 | `int` | 桶数，取值 `[1, 4096]` |
| `min` | Input | 标量 | `float` | 区间下界；与 `max` 同时为 0 时触发 autorange |
| `max` | Input | 标量 | `float` | 区间上界 |
| `out` | Output | `[bins]` | 与 `x` 同 dtype | 各桶计数 |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 输入维度 | `x` 必须为一维连续张量；非一维会被 `reshape(-1)`，非连续会先 `contiguous()` |
| dtype | `x` 仅支持 `float32` / `float16`；输出 dtype 与 `x` 一致 |
| bins | `1 <= bins <= 4096`，否则抛 `ValueError` |
| autorange | `min == 0.0` 且 `max == 0.0` 时按实际数据定界，**仅支持 float32**；f16 会抛 `NotImplementedError` |
| f16 路径 | 宿主先把 `x` 升到 f32（kernel 只保留 f32 计算路径），输出再一次性 cast 回 f16 |
| 空输入 | `N == 0` 直接返回全 0，不启动 kernel |
| 设备 | `x` 与输出位于同一 NPU |
| 编译 | 每个 `(bins, is_f16, bins_pad, 是否合并尾块)` 组合编译一次并缓存；GM 工作区（partials / staging / 输出）按 shape 复用 |
| 尾块 | `N % 256 != 0` 时存在全局尾块：`bins <= 255` 且 `N >= 256` 并入 `histc_reduce`；否则（多趟窗口或 `N < 256`）走独立尾核 + 宿主 staging |

## 实现说明

整体是 **两段式** 结构，`hist` 负责并行统计、`reduce` 负责跨核归并：

1. **`histc_hist`（64 核并行）**：每核处理一段连续区间，逐 16K 元素 tile 搬运到 UB；
   把 16384 个 f32 元素折算成 i8 桶索引，用 `tla.histograms`（dhistv2，BIN0/BIN1 各
   128 桶）在寄存器内累加，再把 i16 计数零扩展成 i32 写回 UB 累加器；每核最后把自己的
   结果写成 `partials[bid, 0:bins_pad]` 一行。
2. **`histc_reduce`（单核）**：一次 2-D 搬运把 `partials[0:nrow, col:col+256]` 整块
   读进 UB，按列求和、按输出 dtype 一次 cast 后写回 `out`；尾块（若合并）在这一步用
   `dhistv2` 顺带算完并加入同一列空间。
3. **`histc_tail`（单核，按需）**：`bins > 255` 或 `N < 256` 时使用，宿主把尾块与
   `min-1` 填充拼进 256 元素 staging，内核当作一个完整组处理。
4. **`histc_range`（autorange 时多一段）**：每核求自己区间内的 min/max 写 2 个平面，
   宿主用两个 device 侧 `min()/max()` 收口成 `[min, max]`。

高性能设计点：

- **key 打包 + 向量直方图**：桶索引按 `DIST_PACK4_B32` 一次落 256 字节（256 个 i8 key），
  再由 dhistv2 在寄存器内直方图化，避免逐元素标量累加；i16 计数用
  `DIST_UNPACK_B16` 零扩展到 i32（自然桶序，计数非负故等价符号扩展）。
- **窗口 + 哨兵键**：每趟窗口固定 255 列，key `255` 作为哨兵表示「丢弃」，因此
  越界值、非对齐尾块残留 lane 都只需生成哨兵键，不必在 SIMD region 内使用动态 mask。
- **不跨核原子累加**：实测 GM 原子加约 15 ns/元素（64 核 × 255 桶 ≈ 250 µs），
  改为每核写非原子 `partials` 行 + 单核归并；归并只做一次整块 2-D 搬运与 256 次向量加。
- **尾块零额外 launch**：尾块并入 `histc_reduce`，用 `arange` 生成 lane 号 + 比较把
  越界 lane 置哨兵键，省掉一次 launch、一次宿主 staging 拷贝与 partials 的尾行。
- **mutex 握手**：MTE2（搬入）与 VECTOR（计算）之间的生产者/消费者关系用
  `tla.mutex` 表达，替代 `set_flag/wait_flag`；region 之间的 UB 顺序仍由 `pipe_barrier`
  保证（同一 region 内 UB store→load 无顺序保证）。
- **精简发射**：单次 launch 的 Python 打包约占 30 µs 中的 27 µs（payload 10 µs、
  查询当前 stream 9 µs、真正 ACL 调用仅 3 µs），故缓存 payload、流句柄与 ABI 常量，
  只调用一次 `launch_kernel`；同时缓存 `from_dlpack` 绑定、输出缓冲与几何参数。

## 使用示例

准备好 CANN、PyTorch、torch_npu 与 CATLASS 环境，导入样例模块后调用 `run`：

```python
import torch
import torch_npu  # noqa: F401

import histc

n, bins = 1 << 20, 100
x = torch.rand(n, dtype=torch.float32, device="npu:0") * 10.0 - 2.0

out = histc.run(x, bins, -2.0, 8.0)
torch.npu.synchronize()

ref = torch.histc(x, bins=bins, min=-2.0, max=8.0)
print(out.sum().item(), (out - ref).abs().max().item())
```

执行结果如下，说明样例执行成功（逐桶精确）：

```text
1048576.0 0.0
```

## 精度与性能（Ascend950DT 实测）

- 精度：`N = 1048576 + 3`（带全局尾块）下覆盖 `bins = 8 / 16 / 100 / 255 / 256 / 1000`
  六种配置，全部与 `torch.histc` 逐桶相等（`max_abs_diff = 0`）。
- 性能（17 组 `(N, bins)` 用例，端到端配对计时）：15/17 通过；随规模增大全面超过原生：

  | n | 本实现 | 原生 `torch.histc` | 比值 |
  |---|---|---|---|
  | 1M | 43–83 µs | 32–33 µs | 0.51–0.89x |
  | 2M | 54 µs | 57 µs | 1.00x |
  | 3M | 62 µs | 72 µs | 1.13x |
  | 4M | 48 µs | 84 µs | 1.79x |
  | 16M | 97 µs | 275 µs | 2.88x |

  1M 档的差距来自「两次 launch 的固定下发开销」：单次 dispatch 约 30 µs，两次背靠背
  48 µs，而原生只有一次 launch。

## 限制

- `histc-015`（f16，bins=1000）与 `histc-017`（f16，bins=256）与原生存在少量桶差
  （diff 352 / 208）：本实现把 f16 精确升到 f32 再分桶，原生按 f16 精度算索引，
  边界附近落桶不同；`bins <= 255` 的 f16 用例逐桶精确。
- 1M 及以下规模受两次 launch 固定开销限制，无法与单 kernel 的原生持平；曾尝试用
  「GM 原子计数 + 有界 spin」做设备侧跨核 barrier 把两核合一（运行时未暴露 `dcci`，
  spin 中的 GM 标量读会被缓存），以及把 reduce 按列切到 4 核并行（实测反而更慢），
  均已回退。
- autorange 仅支持 float32。
