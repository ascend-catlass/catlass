# GmmSwiGluMxQuant Example Readme

## 功能说明

- 算子功能：将按 M 轴分组的 MX 量化矩阵乘（Grouped Matmul，slice-M）、SwiGLU 激活与 Dynamic MX 量化融合为一个 CATLASS DSL AIC/AIV mixed kernel，入口为 `gmm_swiglu_mx_quant` 模块的 `run` 函数。Cube 侧对每个分组完成 MXFP8 / MXFP4 矩阵乘与 FP32 累加（激活半与门控半共享 A 操作数），Vector 侧直接对 FP32 累加结果做 SwiGLU、BF16 舍入与 32 元素分组的 Dynamic MX 量化，激活中间结果不落 GM。
- 计算公式：记 $A$ 按 `group_list` 分组的第 $g$ 段为 $A_g$，$s^{A}$、$s^{B}$ 为每 32 元素一个的 E8M0 分块缩放因子：

  $$
  \begin{aligned}
  C_{g} &= \left(A_g\, s^{A}\right) \left(B_g\, s^{B}\right) \quad \text{(FP32 累加)}\\
  S_{g} &= \operatorname{silu}(C_{g}[:,\,:N/2])\cdot C_{g}[:,\,N/2:] \quad \text{(SwiGLU)}\\
  S^{16} &= \operatorname{round\_bf16\_rne}(S_{g})\\
  (Y_{g}, S^{Y}_{g}) &= \operatorname{dynamic\_mx\_quant}(S^{16})
  \end{aligned}
  $$

  $C_g$ 为 FP32 累加结果，SwiGLU 在 FP32 上计算，量化输入舍入为 BF16；$Y$ 与 $S^{Y}$ 按 32 元素分组共享 E8M0 指数，输出统一为 MXFP8（激活为 E5M2 时输出 E5M2）。支持 a8w8（FP8×FP8）、a8w4（FP8 激活×FP4 权重，B 展宽到 FP8）、a4w4（原生 FP4×FP4 cube）三类输入组合。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
q, q_scale = gmm_swiglu_mx_quant.run(a, b, mx_scale_a, mx_scale_b, group_list, transB=True)
```

`M` 为 A 的行数（按 `group_list` 分组），`K` 为内积维度，`N` 为 B 的输出宽度，`G` 为分组数。FP8 张量使用 `torch.float8_e4m3fn` / `torch.float8_e5m2`，FP4 张量使用打包的 `torch.float4_e2m1fn_x2`（沿 K 打包，第 $2i$ 个元素在低 4 位、第 $2i+1$ 个元素在高 4 位），E8M0 缩放因子使用 `torch.float8_e8m0fnu`。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `a` | Input | `[M, K]`（FP8）或 `[M, K/2]`（FP4） | `float8_e4m3fn` / `float8_e5m2` / `float4_e2m1fn_x2` | 左操作数，行优先，按 `group_list` 沿 M 分组 |
| `b` | Input | `transB=True`：`[G, N, K]`（FP8）或 `[G, N, K/2]`（FP4）；`transB=False`：`[G, K, N]`（FP8）或 `[G, K, N/2]`（FP4） | 同 `a` 的 FP8/FP4 | 右操作数，rank-3，第 0 维为分组数 `G` |
| `mx_scale_a` | Input | `[M, ceil(K/64), 2]` | `float8_e8m0fnu` | 左操作数 E8M0 缩放因子，逻辑 shape `[M, ceil(K/32)]` |
| `mx_scale_b` | Input | `transB=True`：`[G, N, ceil(K/64), 2]`；`transB=False`：`[G, ceil(K/64), N, 2]` | `float8_e8m0fnu` | 右操作数 E8M0 缩放因子，逻辑 K 组数 `ceil(K/32)` |
| `group_list` | Input | `[G]` | `int64` | 每组消耗的 A 行数，非负且总和 ≤ M |
| `transB` | Input | 标量 | `bool` | `True` 时 B 为 `[G, N, K]`，否则 `[G, K, N]`；默认 `True` |
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

导入样例模块后调用 `run`，脚本根据 shape 与输入输出格式编译 kernel、执行 NPU 计算并返回激活量化结果及其缩放因子：

以下命令在 CATLASS 仓库根目录执行：

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
