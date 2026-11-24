# Scan Example Readme

## 功能说明

- 算子功能：按行（per-row）沿最后一维做包含式前缀和（inclusive scan），入口为 `scan` 模块的 `run` 函数，基于 CATLASS DSL 实现的单个纯 AIV kernel。可选的分段模式用 `head` 掩码在指定位置清零累加器，等价于 `torch.cumsum` 的分段版本。
- 计算公式：对每一行 `r`：

  $$
  y_{r,j} = \sum_{k=\mathrm{seg\_start}(j)}^{j} x_{r,k}
  $$

  其中 `seg_start(j) = 0`（普通前缀和，或无段首时），或为满足 `k <= j` 且 `head[r, k] != 0` 的最后一个位置 `k`（分段前缀和，该位置累加器清零后从 `x_{r,k}` 重新累加，因此段首元素本身计入结果）。全程 FP32 计算；`HAS_HEAD = 0` 时整条分段分支在编译期消除。
- 高性能设计：
  - **行 tile + per-buffer mutex 的三段流水**：每个块认领一段连续行，每行一次 MTE2 搬入、一次 MTE3 搬出；`BUFS` 个行缓冲各挂一个 mutex，搬入 / 计算 / 搬出由 mutex 而不是程序序决定何时执行，MTE2 搬第 `r+1` 行、VECTOR 扫第 `r` 行、MTE3 写第 `r-1` 行三段重叠，纯向量算子不再被 DMA 串行化。
  - **64 lane Hillis-Steele 树 + 寄存器内移位**：一个 64-lane chunk 原地扫描，六步 `shift up by d`（`d = 1,2,4,8,16,32`）全部用 `(lane - d) mod 64` 的寄存器 `gather` + 恒等填充完成，不走 UB 往返，也没有共享内存冲突。
  - **跨 chunk 只做一次归约**：chunk 按序推进，把上一块 lane 63 的值折叠进下一块 lane 0，跨块传播的代价是每个 chunk 一次 `reduce`，而不是整行重扫。
  - **尾块按完整 64-lane 向量处理**：UB 行按 `N_PAD = ceil(N/64) * 64` 补齐，GM 侧保持稠密 `N`；末块先清零 padding lane 再用全 lane mask 扫描（收窄 mask 会让向量长度变化、破坏 64-lane `gather`），最后用 `update_mask(TAIL)` 只落盘前 `TAIL` 个 lane。`N < 64` 时 `n_chunks == 0`，整个算子退化成这一条尾块路径，无需额外分支。
  - **GM 起始地址不要求 32 B 对齐**：一行的 GM 基址是 `r * N * 4` 字节，尾块路径只对 UB 侧做 256 B 对齐，GM 侧的 `N` 可以是任意正整数，不对齐的行首由 DMA 自己处理。
  - **段式扫描零额外遍历**：分段变体把 `(value, head-flag)` 作为算子 `op((a,ha),(b,hb)) = hb ? (b,1) : (a+b,ha)` 的二元组，与普通扫描共用同一棵扫描树，每个 chunk 只多一组 `where/cmp`，不增加内存往返。
  - **相对 `torch_npu` 实现**：普通模式的基线是 `torch.cumsum(x, -1)`，分段模式是 `cumsum` 加掩码定位段首的组合实现；本算子一次 launch 完成整行扫描、跨 chunk 只做一次归约，短行多行场景下不再被 kernel 启动与调度开销主导，长行纯带宽场景则与基线接近内存带宽上限。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
y = run(x, head=None)
```

`M` 为行数，`N` 为每行长度。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `x` | Input | `[M, N]` | `float32` | 输入；`N` 任意（非 64 倍数走尾块路径）；非连续输入内部自动 `contiguous()`。 |
| `head` | Input，可选 | `[M, N]` | `float32` | 分段掩码，非 0 处重新开始累加（该位置的累加值即 `x` 本身）；取值不要求恰为 `1`，任意非 0 值都视为段首。默认 `None` 表示普通前缀和；非 `float32` 输入内部自动转成 `0/1` 掩码。 |
| `y` | Output | `[M, N]` | `float32` | 前缀和结果。 |

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 输入维度 | `x` 必须为 2 维；`M`、`N` 为正整数，`N` 不限（非 64 倍数时 UB 行按 64 补齐）。 |
| dtype | 输入、输出、`head` 均为 `float32`。 |
| UB 占用 | `BUFS` 块 `N_PAD` FP32 行缓冲（`N_PAD = ceil(N/64) * 64`），分段模式下再各配一块 `N_PAD` 掩码；受 256 KiB UB 限制，普通模式 `N` 上限约 62K，分段模式约 31K。 |
| 流水深度 | `BUFS` 由 UB 预算在 host 侧选择：`max(1, min(4, (UB_BYTES - 8192) / (4 * N_PAD * [2])))`，行越宽流水越浅。 |
| 数值行为 | 设备侧全程 FP32；行内扫描用 64-lane 树形结构，跨 chunk 只折叠一次进位，误差随 `N` 线性累积（见使用示例）。 |
| 编译 | 按 `(M, N, 是否分段)` 缓存编译产物，形状或模式变化会重编译。 |
| 启动缓存 | host 侧按张量地址缓存绑定张量与 launch payload；切换 `torch.npu.stream` 后需调用 `reset_stream_cache()`。 |
| 执行 | 纯 AIV；block 数取 `vector_core_num` 并缓存，每个块认领一段连续行。 |
| 硬件 | 编译选项固定 `--npu-arch 3510`。 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from scan.scan import run

torch.npu.set_device(0)
torch.manual_seed(0)
m, n = 64, 1000
x = torch.randn(m, n, dtype=torch.float32, device="npu:0")

y = run(x)
print(y.shape, y.dtype, f"max abs err {(y - torch.cumsum(x, -1)).abs().max().item():.2e}")

# 分段前缀和：head 非 0 处重新开始累加
head = torch.zeros_like(x)
head[:, 256] = 1
head[:, 768] = 1
ys = run(x, head)
ref = torch.empty_like(x)
start = 0
for j in range(n):
    if head[0, j] != 0 or j == 0:
        start = j
    ref[:, j] = x[:, start:j + 1].sum(-1)
print(ys.shape, f"segmented max abs err {(ys - ref).abs().max().item():.2e}")
PY
```

```text
torch.Size([64, 1000]) torch.float32 max abs err 4.58e-05
torch.Size([64, 1000]) segmented max abs err 1.91e-05
```
