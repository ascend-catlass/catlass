# GmmSwiGluMxQuant Example Readme

## 功能说明

- 算子功能：将按 M 轴分组的 MX 量化矩阵乘（Grouped Matmul，slice-M）、SwiGLU 激活与
  Dynamic MX 量化融合为一个 CATLASS DSL AIC/AIV mixed kernel。Cube 侧对每个分组完成
  MXFP8 / MXFP4 矩阵乘与 FP32 累加（激活半与门控半共享 A 操作数），Vector 侧直接对 FP32
  累加结果做 SwiGLU、BF16 舍入与 32 元素分组的 Dynamic MX 量化，激活中间结果不落 GM。
- 计算公式：记 $A$ 按 `group_list` 分组的第 $g$ 段为 $A_g$，$s^{A}$、$s^{B}$ 为每 32 元素一个的
  E8M0 分块缩放因子：

  $$
  \begin{aligned}
  C_{g} &= \left(A_g\, s^{A}\right) \left(B_g\, s^{B}\right) \quad \text{(FP32 累加)}\\
  S_{g} &= \operatorname{silu}(C_{g}[:,\,:N/2])\cdot C_{g}[:,\,N/2:] \quad \text{(SwiGLU)}\\
  S^{16} &= \operatorname{round\_bf16\_rne}(S_{g})\\
  (Y_{g}, S^{Y}_{g}) &= \operatorname{dynamic\_mx\_quant}(S^{16})
  \end{aligned}
  $$

  $C_g$ 为 FP32 累加结果，SwiGLU 在 FP32 上计算，量化输入舍入为 BF16；$Y$ 与 $S^{Y}$ 按
  32 元素分组共享 E8M0 指数，输出统一为 MXFP8。

- 输入精度组合：a8w8（FP8×FP8）、a8w4（FP8 激活×FP4 权重，B 展宽到 FP8）、a4w4（原生
  FP4×FP4 cube）；输出 MXFP8（E4M3，激活为 E5M2 时输出 E5M2）及其 E8M0 缩放因子。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
q, q_scale = run(a, b, mx_scale_a, mx_scale_b, group_list, transB=True)
```

FP8 张量使用 `torch.float8_e4m3fn` / `torch.float8_e5m2`，FP4 张量使用打包的
`torch.float4_e2m1fn_x2`（沿 K 打包，第 $2i$ 个元素在低 4 位、第 $2i+1$ 个元素在高 4 位），
E8M0 缩放因子使用 `torch.float8_e8m0fnu`。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `a` | Input | `[M, K]`（FP8）或 `[M, K/2]`（FP4） | `float8_e4m3fn` / `float8_e5m2` / `float4_e2m1fn_x2` | 左操作数，行优先，按 `group_list` 沿 M 分组 |
| `b` | Input | `transB=True`：`[G, N, K]`（FP8）或 `[G, N, K/2]`（FP4）；`transB=False`：`[G, K, N]`（FP8）或 `[G, K, N/2]`（FP4） | 同 `a` 的 FP8/FP4 | 右操作数，rank-3，第 0 维为分组数 `G` |
| `mx_scale_a` | Input | `[M, ceil(K/64), 2]` | `float8_e8m0fnu` | 左操作数 E8M0 缩放因子，逻辑 shape `[M, ceil(K/32)]` |
| `mx_scale_b` | Input | `transB=True`：`[G, N, ceil(K/64), 2]`；`transB=False`：`[G, ceil(K/64), N, 2]` | `float8_e8m0fnu` | 右操作数 E8M0 缩放因子，逻辑 K 组数 `ceil(K/32)` |
| `group_list` | Input | `[G]` | `int64` | 每组消耗的 A 行数，非负且总和 ≤ M |
| `transB` | Input | 标量 | `bool` | `True` 时 B 为 `[G, N, K]`，否则 `[G, K, N]` |
| `q` | Output | `[M, N/2]` | `float8_e4m3fn`（激活 E5M2 时为 `float8_e5m2`） | SwiGLU + 量化后的 MXFP8 激活结果 |
| `q_scale` | Output | `[M, ceil(N/64)]` | `float8_e8m0fnu` | 激活结果的 E8M0 缩放因子，逻辑 shape `[M, ceil((N/2)/32)]` |

## 约束说明

| 项目 | 约束 |
| ----- | -------- |
| 维度 | `M`、`K`、`N` 为正整数；`K % 64 == 0`，`N % 128 == 0`；分组数 `G ∈ [1, 1024]` |
| dtype | a8w8：`a`、`b` 为同一 FP8 dtype（E4M3 或 E5M2）；a8w4：`a` 为 E4M3、`b` 为 FP4；a4w4：`a`、`b` 均为 FP4 |
| 布局 | `a` 行优先；`b` 按 `transB` 取 `[G, N, K]` 或 `[G, K, N]`；FP4 沿 K（transB）或 N（nontransB）打包 |
| 设备 | 所有张量位于同一 NPU 且连续（contiguous） |
| 分组 | 仅前 `sum(group_list)` 行有效，其余尾行置零；`group_list` 为 INT64 |
| 尾块 | `M`、`N` 无需按 tile 对齐，kernel 只拷贝有效 extent；`K` 尾块在片内补零 |
| 数值 | 缩放因子按 32 元素分组共享；SwiGLU 左半为 silu 分支、右半为门控分支 |

## 使用示例

准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境。样例模块 `gmm_swiglu_mx_quant`
位于 `examples/python/tla_dsl/experimental/kernel/matmul/` 下，不是可安装的顶层包，需先把该
目录加入 `PYTHONPATH`。以下命令在 CATLASS 仓库根目录执行，脚本根据 shape 与输入输出格式
编译 kernel、执行 NPU 计算并返回激活量化结果及其缩放因子：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/matmul:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from gmm_swiglu_mx_quant.gmm_swiglu_mx_quant import run

m, k, n = 256, 1024, 512
a = torch.randn(m, k, device="npu:0").to(torch.float8_e4m3fn)
b = torch.randn(1, n, k, device="npu:0").to(torch.float8_e4m3fn)
mx_scale_a = torch.full((m, k // 64, 2), 127, dtype=torch.uint8, device="npu:0").view(torch.float8_e8m0fnu)
mx_scale_b = torch.full((1, n, k // 64, 2), 127, dtype=torch.uint8, device="npu:0").view(torch.float8_e8m0fnu)
group_list = torch.tensor([m], dtype=torch.int64, device="npu:0")

q, q_scale = run(a, b, mx_scale_a, mx_scale_b, group_list, transB=True)
print(q.shape, q_scale.shape, q.dtype, q_scale.dtype)
PY
```

执行结果如下，说明样例执行成功：

```text
torch.Size([256, 256]) torch.Size([256, 8]) torch.float8_e4m3fn torch.float8_e8m0fnu
```

## 同步实现

核内同步分两类：

- **L1/L0 双缓冲：按 buffer 拆分的 flag，而不是单个 mutex。** 为让下一块 `GM→L1` 与
  `L1→L0` 能和当前块的 MMAD 重叠，L1/L0 各维护两个槽位；二值 mutex 只能表达"占用/空闲"
  一种状态，表达不了 p0/p1 两组 buffer 各自的状态，因此用按槽位拆分的 flag：
  `l1_free_p0` / `l1_free_p1`（MTE1→MTE2，归还 L1 槽）、`l0_free_p0` / `l0_free_p1`
  （CUBE→MTE1，归还 L0 槽）；配合 `ab_ready`（MTE2→MTE1）、`l0_ready`（MTE1→CUBE）、
  `l0_drain`（CUBE→FIX）。这是与上游 `tla.mutex` 版本的**有意结构差异**，不是回退：
  单 mutex 无法表达双缓冲，合并会丢失重叠。
- **L0C 与输出 UB：各一个二元 mutex**（`mtx_l0c`、`mtx_ub_out`），生产者在对应 pipe 上
  `lock`/`unlock` 包住写入，消费者在对应 pipe 上 `lock`/`unlock` 包住读取。

AIC↔AIV 的跨核同步使用 `tla.cross_flag`：`ub_ready` / `ub_free`（FIX↔VECTOR），以及
解码路径的 `a_full` / `a_free` / `b_full` / `b_free`（MTE3↔MTE1，`mode=4` 按 AIV sub-id
分别计数）。transB 的两级解码 staging 还带一组按奇偶拆分的 `dcd_a_*` / `dcd_g_*`
（fill/widen/store/release 各两条）。a8w4 transB 满 tile 的 FP4→FP8 展宽使用硬件
`tla.experimental.mx_to`（unit E8M0 scale=127 无损，真实 per-32 MX scale 由 cube 侧
`gm_sb` 施加）；nontransB 满 tile 走字宽 SIMT 解码（Int32 一次加载 4 个 packed 字节、
解码 8 个 nibble 后以 Int16 打包存储，摊薄循环归纳与地址运算），partial tile 保留逐字节
SIMT nibble 解码。

**本样例不使用 `tla.extern`。** 外部 Ascend C 写回版本实测净亏——把转置搬到 VEC 后
`aiv_vec` 由 11.77 µs 升到 31.46 µs；且 L1→L0 的融合 `LoadData` 被 `tla.mmad_mx` 的
provenance 校验（L0 操作数必须由 `tla.copy(..., scale=...)` 产生）拦截。

### a8w4 窄 transB 解码门（TK256-W）

对窄 transB 解码 shape，取更宽的 L1 K tile 并把整个 K tile 作为一个 L0 操作数，可减少
`L1→L0` 拷贝次数与 `UB→L1` 解码写回的事务数：

- host 侧 `_tiling_for`：当 `decode_b and transB and tn>64 and tm<=128 and n_out<=512` 时
  放宽 `sk_max`，使 `tk` 可选到 256；
- kernel 侧：同一条件下令 `lk = tk`（取消 tk=256 时默认的 lk=128 解耦，`nsub_l0=1`），
  每个 K tile 的 `L1→L0` 拷贝由 3×16 KiB 变为 3×32 KiB、K 步数减半。

该门控在 063（g16 m2048 n512 k1024）上把 `aic_mte1` 由 12.9 µs 压到 5.0 µs、`aiv_mte3`
由 19.4 µs 压到 8.8 µs，`Duration` −12%，输出与改前逐位一致；`tm=256` 的大例（如 069/070）
被 `tm<=128` 排除，保持原有解耦路径。

## Launch block 钳制（A8LAT）

小 shape 的 `Duration` 有一块与工作量无关的固定开销。实测空 kernel：**28 block 4.71 µs、
2 block 1.35 µs**，约 **3.4 µs 是本代硬件派发满核的斜坡**（块数越多，ramp 越长），
对只有 2–8 个有效 block 的小 shape 直接构成主导项。

做法：`run()` 里按形状算出实际需要的 block 数

```
total_blocks = ceil(n_out / tiling.tn) × Σ_g ceil(group_size_g / tiling.tm)
```

以 `min(core_num, total_blocks)` 作为 `block_num` 启动。**tiling 仍按满核计算**、工作划分不变，
只是不启动没有任何 block 的空闲核；`total_blocks <= core_num` 时每个核至多一个 block，
删掉空闲块不改变任何核的负载（kernel 在运行期读 `block_num()`，映射对任意值自洽）。

- **边界**：`total_blocks >= core_num` 时返回 `core_num` 不变，**大 shape 零影响**。
- **实测**：055 8.637 → 4.948 µs（**−43%**）；小 shape 普遍 **−12% ~ −45%**。

## 性能数据

测量环境：Ascend 950PR（DAV_3510）、CANN 9.3.0、ascend-catlass-dsl 2.2.1、
torch-npu 2.10.0.post4。
口径为**纯 kernel `Duration`**（`kernel_details.csv` 中计算 kernel 的 Duration）；
**逐例配对**测量：同一个用例内先跑本样例、紧接着跑对标算子，两侧各 3 次取最小，
共享同一负载窗口，不依赖两次独立运行之间的负载一致。

对标对象为 CANN 已交付融合算子：a8w4 / a4w4 用
`aclnnGroupedMatmulSwigluQuantWeightNzV2`，a8w8 用
`aclnnGroupedMatmulSwigluQuantWeightNzV3`（E4M3）与
`aclnnGroupedMatmulSwigluQuantV2`（E5M2 非 NZ）。

- **正确性**：200 个模型场景用例（a8w8 / a8w4 / a4w4 × transB / nontransB，含 ragged 分组、
  partial tile、zeros、sparse、M/N/K 非对齐）按用例自带判据（`max_atol = max_rtol = 0.125`、
  `required_matched_ratio = 0.999`）**200/200 通过**；其中 **196 例**的 `q` 与 `q_scale` 与
  PyTorch 标杆**逐位相同**，其余 4 例（170 / 172 / 173 / 175，均为 a8w8 E5M2 nontransB）各有
  1 个元素落在容差带外（`q_max_abs_diff` 为 1~16，即 E5M2 上的 1~2 个量化刻度），
  `q_scale` 仍逐位相同。
- **特殊值 / 确定性**：按本 PR 描述「测试 → 特殊值检测 / 确定性检测」两节已回填的结果
  （`tools/special_det_test.py` 与双进程逐位比对，测试脚本未随 PR 提供），NaN、Inf（E5M2
  激活）、极大值（FP8/FP4 最大有限值 + E8M0 scale 取 2^127 / 2^-126）、空 Tensor（M=0）、
  空分组（组长度 0）等 6 类用例全部通过、跨进程输出逐位一致；NaN/Inf 输入下 payload 保持有限
  （epilogue 在 FP8 cast 前 clamp），不传播 NaN。

**本样例 / 对标算子**（几何平均，越小越好，200 例全部纳入）：

| 族 | 例数 | 比值 | 胜出例数 |
|---|---|---|---|
| a8w8（E4M3） | 38 | 0.5732 | 36/38 |
| a8w8（E5M2） | 37 | 0.6513 | 31/37 |
| a4w4 | 62 | 0.6885 | 50/62 |
| a8w4 | 63 | 0.9387 | 35/63 |
| 全部 | 200 | 0.7256 | 152/200 |

按本样例 `Duration` 规模拆开（该算子为带宽/解码受限，派发与解码的固定开销在小 shape 上占比高）：

| 族 | < 16 µs | 16 ~ 40 µs | ≥ 40 µs |
|---|---|---|---|
| a8w8（E4M3） | 0.469（24） | 0.752（9） | 0.922（5） |
| a8w8（E5M2） | 0.547（25） | 0.806（6） | 1.090（6） |
| a4w4 | 0.567（40） | 0.902（12） | 1.083（10） |
| a8w4 | 0.799（36） | 1.175（13） | 1.153（14） |

即：小 shape 三族全部领先；a8w8 E4M3 的 ≥ 40 µs 桶已追平到 0.922；仍落后于对标的是 a8w4 的
中大型 shape（落后 15%~18%）与 a8w8 E5M2、a4w4 的 ≥ 40 µs 桶（落后 8%~9%），差距来源是权重
解码的 `aiv_mte3`（085 这类半 N tile 用例：本样例 11.4 µs，对标 4.1 µs）与 `aic_mte2` 上每单位
MMAD 的搬运粒度。

- **收益来源**：融合避免激活中间结果 GM 往返；`ZNUB` / `NTB-ZN` / `AGSHARE` 收拢 a8w4 与
  a8w8 transB 的权重解码链路，`TK256` / `TK512` / `TK256-W` / `TK256-PIN` 把 L0 操作数宽度与
  解码事务粒度对齐（窄 transB 每 K 步的 `L1→L0` 搬运与 `UB→L1` 事务各减半），`A8TK` 与
  `CB-PIPE` 保住 a8w8 大例的 L0 双缓冲和跨块重叠，`A8LAT` 去掉小 shape 上 3.4 µs 的派发斜坡。
- **口径说明**：本表是纯 kernel 口径，两侧都只计计算 kernel 的 `Duration`；端到端口径的
  比值会因两侧 host 开销不同而偏离，本文不采用。
