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
q, q_scale = gmm_swiglu_mx_quant.run(a, b, mx_scale_a, mx_scale_b, group_list, transB=True)
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

准备好 CANN、PyTorch、torch_npu 与 catlass DSL 环境，在 CATLASS 仓库根目录导入样例模块后
调用 `run`。脚本根据 shape 与输入输出格式编译 kernel、执行 NPU 计算并返回激活量化结果及其
缩放因子：

```python
import torch
import torch_npu  # noqa: F401

import gmm_swiglu_mx_quant

m, k, n = 256, 1024, 512
a = torch.randn(m, k, device="npu:0").to(torch.float8_e4m3fn)
b = torch.randn(1, n, k, device="npu:0").to(torch.float8_e4m3fn)
mx_scale_a = torch.full((m, k // 64, 2), 127, dtype=torch.uint8, device="npu:0").view(torch.float8_e8m0fnu)
mx_scale_b = torch.full((1, n, k // 64, 2), 127, dtype=torch.uint8, device="npu:0").view(torch.float8_e8m0fnu)
group_list = torch.tensor([m], dtype=torch.int64, device="npu:0")

q, q_scale = gmm_swiglu_mx_quant.run(a, b, mx_scale_a, mx_scale_b, group_list, transB=True)
print(q.shape, q_scale.shape, q.dtype, q_scale.dtype)
```

执行结果如下，说明样例执行成功：

```text
torch.Size([256, 256]) torch.Size([256, 8]) torch.float8_e4m3fn torch.float8_e8m0fnu
```

## 同步实现

核内流水同步使用 `tla.mutex` 与 `tla.flag`：L0C 与输出 UB 各一个二元 mutex（`mtx_l0c`、
`mtx_ub_out`），生产者在对应 pipe 上 `lock`/`unlock` 包住写入，消费者在对应 pipe 上
`lock`/`unlock` 包住读取；L1/L0 阶段用 `ab_ready` / `l0_ready` / `l1_avail` / `l0_avail`
flag 配对。AIC↔AIV 的 `ub_ready` / `ub_free` 以及解码路径的 `a_full` / `a_free` /
`b_full` / `b_free` 属于跨核同步，使用 `tla.cross_flag`。a8w4 transB 满 tile 的 FP4→FP8
展宽使用硬件 `tla.experimental.mx_to`（unit E8M0 scale=127 无损，真实 per-32 MX scale 由
cube 侧 `gm_sb` 施加）；nontransB 满 tile 走字宽 SIMT 解码（Int32 一次加载 4 个 packed 字节、
解码 8 个 nibble 后以 Int16 打包存储，摊薄循环归纳与地址运算），partial tile 保留逐字节
SIMT nibble 解码。

## 性能数据

测量环境：Ascend950PR（DAV_3510）、ascend-catlass-dsl 2.2.1、torch-npu 2.10.0.post7；
基准一与基准二的同卡配对测量均在 CANN 9.2.0-beta.2 下完成，官方标杆基线文件
（`baseline.jsonl` / `baseline_kernel.jsonl`）由任务提供方在 CANN 9.3.0 下测得。

- 正确性：150 个模型场景用例（a8w8 / a8w4 / a4w4 × transB / nontransB，含 partial tile 与
  多分组）全量通过，`q` 匹配率 ≥ 0.99999994、输出 finite、`q_scale` 与 PyTorch 标杆逐位一致。
- 性能基准一（强化标杆，全量 150 例）：把官方标杆链的 epilogue（SwiGLU → BF16 舍入 →
  约 20 个 elementwise MX 量化算子）替换为 CANN 融合算子 `torch_npu.npu_swiglu_mx_quant`，
  反量化与 FP32 `torch.matmul` 保持官方实现不变，得到"已融合 epilogue 的强标杆"。同卡同进程
  逐例配对：纯 kernel 口径（一次调用提交的全部 NPU kernel 时间之和）候选 0.1118 ms、强化
  标杆 0.7283 ms，几何平均 **6.18x**、算术平均 **7.20x**；端到端口径（官方协议：3 次 warmup +
  100 样本，均值 ≤1.5×中位数守门）候选 0.1704 ms、强化标杆 1.4630 ms，几何平均 **7.56x**、
  算术平均 **9.69x**。纯 kernel 分族几何：a8w8 6.35x、a8w4 4.03x、a4w4 9.22x。
- 性能基准二（官方标杆 `reference.py` 小算子链，任务定义口径）：同一官方协议、同卡、全量
  150 例同进程配对，候选 0.1704 ms、标杆 3.2737 ms，几何平均 **21.15x**、算术平均 **26.64x**
  （任务提供的 `baseline.jsonl` 标杆均值 3.4778 ms，与同卡复测一致）。纯 kernel 口径对官方
  `baseline_kernel.jsonl` 的 100 个代表用例（a8w8 / a8w4）几何平均 **12.00x**、算术平均
  **14.59x**（a8w8 几何 15.95x / a8w4 几何 9.02x）。
- 口径说明：端到端比值里官方标杆链的 Python 逐算子调度与 GM 往返占比很高，纯 kernel 口径
  去掉了两侧的 host 开销，因此数值低于端到端口径；两口径的候选与标杆都由同一脚本、同一
  设备、同一进程配对测得。
- 收益来源：融合避免了激活中间结果的 GM 往返；其余差距主要来自以 FP8/FP4 cube 取代官方
  标杆的 PyTorch FP32 反量化与 FP32 matmul。该算子为带宽/解码受限而非计算受限，窄 N 与
  a8w4 nontransB（SIMT 解码）场景加速比较低；nontransB 满 tile 采用字宽解码（Int32 加载 +
  Int16 存储）后，焦点子集纯 kernel 时间较官方标杆再降 15.9%（case100 单例 -30.8%）。
- 范围说明：CANN 全融合单算子 `npu_grouped_matmul_swiglu_quant_v2` 与手写 C++ 样例
  `examples/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant` 在 a8w8 大 shape 上
  快于本 kernel（021：0.050 / 0.059 ms 对 0.117 ms；041：0.084 / 0.094 ms 对 0.218 ms），
  本示例为 DSL 写法演示，不作为该链路的性能上限。
