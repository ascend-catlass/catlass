# RopeKvCache Example Readme

## 功能说明

- 算子功能：对 Q 施加 NeoX 旋转位置编码（RoPE），同时把当前 token 的 K/V 按
  `slot_mapping` 追加写入分页 KV cache，入口为 `rope_kv_cache` 模块的 `run` 函数。
  基于 CATLASS DSL 的纯 AIV kernel：MTE2 批量搬入 → UB 内 SIMD 旋转 → MTE3 批量
  写回 Q、逐 token 散播 K/V。`head_dim % 16 != 0` 时 token 行不足 32 Byte DMA
  粒度，自动回退标量 SIMT 路径。
- 计算公式：对 token `t`，记 `hd = head_dim / 2`，`slot = slot_mapping[t]`，对每个 Q head
  的 `d < hd`：

  $$
  \begin{aligned}
  q\_out[t, d] &= q[t, d] \cdot \cos[t, d] - q[t, d + hd] \cdot \sin[t, d] \\
  q\_out[t, d + hd] &= q[t, d] \cdot \sin[t, d] + q[t, d + hd] \cdot \cos[t, d] \\
  cache\_k[slot, :] &= k[t, :] \quad (\text{仅当 } slot \ge 0) \\
  cache\_v[slot, :] &= v[t, :] \quad (\text{仅当 } slot \ge 0)
  \end{aligned}
  $$

  旋转对为前后半区配对（NeoX），不是 GPT-J 的交错配对；`slot < 0` 表示该 token 不写
  cache（例如 padding）。

- 高性能设计：

  1. 批量 DMA 取代标量访存：每 `rows_per_group` 行（默认 4）对 q/cos/sin/k/v/q_out
     各发一次连续 MTE2/MTE3 搬移（1–4 KB/次）；标量 GM 访问次数从 O(元素数) 降到
     O(组数 × 张量数)，访存由延迟受限转为带宽受限。
  2. 三 pipe 流水 + UB 双缓冲：`slots=2` 组缓冲轮转，下一组的 MTE2 与当前组的
     VECTOR、MTE3 并行；UB 预算决定 `rows_per_group`，保证两组常驻。
  3. 向量化旋转：NeoX 前后半区各自连续，按 64 lane 直接取 tile 配对，无需跨 lane
     shuffle；同一行的 cos/sin 只读一次，在 `HQ` 个 head 间复用。
  4. K/V 直通：K/V 只做 UB 中转、不参与向量运算，每 token 一次 1 KB 级 UB→GM 散播，
     与 Q 的写回共用同一条 MTE3。
  5. 幂等尾组：末尾组回拉至以 `T` 结束，重复行重复写同值，省掉尾块特判分支。
  6. 按需起核：`block_num = min(vector_core_num, 组数)`，小 shape 不唤醒空核。
  7. 核内同步全部用 mutex：每槽 `m_in`（MTE2→VECTOR，q/cos/sin）、`m_out`
     （VECTOR→MTE3，q_out 暂存）、`m_kv`（MTE2→MTE3，k/v），producer 与 consumer
     一对一；不使用 `set_flag`/`wait_flag`。kernel 末尾 `pipe_barrier(ALL)` 只负责
     退出前排空在途 DMA。
  8. q_out 使用独立暂存区而非原地改写 q，避免同一块 UB 跨 MTE2→VECTOR→MTE3
     串成三段锁链，让每个 mutex 保持一对一、便于验证。

- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
q_out = run(
    q, k, v, cos, sin, slot_mapping, cache_k, cache_v,
    num_q_heads, num_k_heads, head_dim,
)
```

记 `T` 为 token 总数，`S` 为 cache 的 slot 容量，`HQ`/`HK` 为 Q/KV head 数，`D` 为
head dim。`cache_k` / `cache_v` 原地更新，不需要额外输出。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | Input | `[T, HQ * D]` | `float32` | 待旋转的 Q，行优先连续 |
| `k` | Input | `[T, HK * D]` | `float32` | 待写入 cache 的 K |
| `v` | Input | `[T, HK * D]` | `float32` | 待写入 cache 的 V |
| `cos` | Input | `[T, D // 2]` | `float32` | 每个 token 每个旋转对的 cos 系数 |
| `sin` | Input | `[T, D // 2]` | `float32` | 每个 token 每个旋转对的 sin 系数 |
| `slot_mapping` | Input | `[T]` | `int32` | 目标 cache slot，负数表示不写 |
| `cache_k` | Input/Output | `[S, HK * D]` | `float32` | 分页 K cache，原地写入 |
| `cache_v` | Input/Output | `[S, HK * D]` | `float32` | 分页 V cache，原地写入 |
| `num_q_heads` | 属性 | 标量 | Python `int` | Q head 数 `HQ` |
| `num_k_heads` | 属性 | 标量 | Python `int` | KV head 数 `HK` |
| `head_dim` | 属性 | 标量 | Python `int` | 头维 `D`，必须为偶数 |
| `q_out` | Output | `[T, HQ * D]` | `float32` | 旋转后的 Q |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 维度 | `q`/`k`/`v` 必须 2-D；`head_dim % 2 == 0` |
| 旋转形式 | NeoX（前后半区配对），不支持 GPT-J 交错配对 |
| 形状一致性 | `q.shape[1] == HQ * D`，`k.shape == v.shape` 且宽度 `== HK * D`；`cos`/`sin` 均为 `[T, D // 2]` |
| dtype | `q`/`k`/`v`/`cos`/`sin`/`cache_k`/`cache_v` 均为 `float32` 且连续；`slot_mapping` 为 `int32`（host 侧自动转换） |
| Cache 写入 | 仅 `slot >= 0` 时写；`slot` 必须小于 `S`（kernel 不做越界检查）；同一批内 `slot` 应唯一，否则写结果取决于调度顺序 |
| DMA 对齐 | 主路径要求 `head_dim % 16 == 0`（token 行 ≥ 32 Byte）；不满足时走标量 SIMT 回退，结果仍正确 |
| 同步模型 | 每 slot 一组 mutex（`m_in`/`m_out`/`m_kv`），无 `set_flag`/`wait_flag`；`slots=2`、`rows_per_group <= 4` |
| UB 预算 | 双槽缓冲总量 ≤ 224 KB；`rows_per_group` 按 `(HQ*D + D/2 + HK*D) * 8` Byte/行自动收缩，下限 1 |
| 起核 | `block_num = min(vector_core_num, ceil(T / rows_per_group))`，与 `T` 无关地复用同一 artifact |
| 编译 | 按 `(T, D, HQ, HK, rows_per_group)` 缓存 artifact；回退路径按 `(T, D, HQ, HK)` |
| 精度 | 主路径 FP32 逐位一致（`q` 误差 0，cache 为精确拷贝）；回退路径 `q` 最大绝对误差约 5e-7 |
| 硬件 | 编译选项固定 `--npu-arch 3510` |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from rope_kv_cache.rope_kv_cache import run

T, HQ, HK, D, S = 32, 4, 2, 64, 128
torch.manual_seed(0)
q = torch.randn(T, HQ * D, dtype=torch.float32, device="npu:0")
k = torch.randn(T, HK * D, dtype=torch.float32, device="npu:0")
v = torch.randn(T, HK * D, dtype=torch.float32, device="npu:0")
cos = torch.rand(T, D // 2, dtype=torch.float32, device="npu:0")
sin = torch.rand(T, D // 2, dtype=torch.float32, device="npu:0")
slot = torch.arange(T, dtype=torch.int32, device="npu:0")
cache_k = torch.zeros(S, HK * D, dtype=torch.float32, device="npu:0")
cache_v = torch.zeros(S, HK * D, dtype=torch.float32, device="npu:0")

q_out = run(q, k, v, cos, sin, slot, cache_k, cache_v, HQ, HK, D)
print(q_out.shape, q_out.dtype)
PY
```

样例文件自带精度自检（同上路径下执行 `python .../rope_kv_cache/rope_kv_cache.py`），
预期输出：

```text
q max abs err: 0.0
cache_k err: 0.0
cache_v err: 0.0
```

## 验证方法

- 数值基准：与 PyTorch 参考实现（NeoX 前后半区配对旋转 + 按 `slot` 写入 cache）
  对照；主路径旋转结果逐位一致（`q` 误差 0，cache 为精确拷贝），标量回退路径
  `q` 最大绝对误差约 5e-7。
- 专项用例：`T` 非 `rows_per_group` 整数倍（尾组重叠）、`slot` 全跳过 / 混合 /
  全写入、`D=2…256`、`HQ/HK` 组合（1/1、4/2、8/8）、`D % 16 != 0` 的标量回退。
- 确定性：同一输入连续两次执行，`q_out`、`cache_k`、`cache_v` 逐位一致。
- 性能口径见下节。

## 性能结果

### 测量配置与口径

- 设备：Ascend 950PR，56 个 AIV；`torch.profiler` 采集 kernel 设备耗时，同机同进程
  连续采样 30 次取均值；不计 host 参数校验与编译时间。
- 形状：`HQ=4`，`HK=2/4`，`D=64/128`，`T` 从 32 到 2048；cache 容量 `S=4096`，
  工作集约 18 MB（大部分常驻 L2）。dtype 为 FP32。
- 「设备耗时」为 `run()` 内 kernel 的执行时间；「端到端」额外包含 9 次
  `from_dlpack` 与参数校验的 host 开销。

### 设备耗时

| 形状 | 设备耗时（us） | 端到端（us） |
| --- | ---: | ---: |
| T=32, HQ=4, HK=2, D=64 | 1.14 | 94 |
| T=128, HQ=4, HK=2, D=128 | 1.86 | 96 |
| T=512, HQ=4, HK=2, D=128 | 2.96 | 95 |
| T=513, HQ=4, HK=2, D=128 | 3.07 | — |
| T=1024, HQ=4, HK=2, D=128 | 3.56 | — |
| T=2048, HQ=4, HK=2, D=128 | 4.75 | 97 |
| T=2048, HQ=4, HK=4, D=128 | 5.57 | — |
| T=2048, HQ=4, HK=2, D=64 | 4.22 | — |
| T=512, HQ=4, HK=4, D=128 | 3.17 | — |

### 结论

- token 数从 32 增到 2048（64×）时，设备耗时仅 1.14 → 4.75 us（约 4×），吞吐随规模
  上升，说明主路径由批量 DMA 的带宽与启动开销主导，而不是逐元素访存。
- 端到端时延稳定在 94–97 us，shape 变化几乎不影响 host 侧；这段固定开销主要来自
  9 次 `from_dlpack` 与参数校验，kernel 本身已不是端到端瓶颈。
