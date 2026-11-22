# FlashKdaFwd Example Readme

## 功能说明

- 算子功能：实现 KDA 线性注意力的分块前向计算，逐 chunk 完成 score、三角矩阵求逆、chunk 入口状态投影与跨 chunk 状态递推，输出 out、最终状态及可选的训练缓存。
- 计算公式：记每个 V 头的状态 `H` 按 `[Dv,Dk]` 存放，逐 token 定义为

  ```text
  H_decay = H_prev * exp(ell_t)[None, :]
  delta_t = b_t * (v_t - H_decay @ k_t)
  H_t     = H_decay + delta_t[:, None] * k_t[None, :]
  out_t   = scale * (H_t @ q_t)
  ```

  chunk 内从零开始累计 `G_i = sum(ell_0...ell_i)`、`Gamma_i = exp(G_i)`，则

  ```text
  L[i,j]   = b_i * sum_d(k_i[d] * k_j[d] * exp(G_i[d]-G_j[d]))，i>j 否则 0
  Akk      = inverse(I + L)
  Aqk[i,j] = scale * sum_d(q_i[d] * k_j[d] * exp(G_i[d]-G_j[d]))，i>=j 否则 0
  w      = Akk @ (b[:,None] * k * Gamma)
  u      = Akk @ (b[:,None] * v)
  qg     = q * Gamma
  kg     = k * exp(G_last[None,:] - G)
  v_new  = u - w @ H_chunk_in.T
  out    = (scale * qg) @ H_chunk_in.T + Aqk @ v_new
  H_next = H_chunk_in * exp(G_last)[None,:] + v_new.T @ kg
  ```

- 支持产品型号：Ascend 950 系列产品

## 参数说明

```python
def flash_kda_fwd(q, k, v, g, beta, metadata=None, initial_state=None, A_log=None,
                  dt_bias=None, cu_seqlens=None, seqused=None, *, scale=128**-0.5,
                  output_final_state=False, return_intermediate_states=False,
                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False,
                  disable_recompute=False, lower_bound=None, layout="TND"):
```

记 `Dk=Dv=128`、`CHUNK=64`；`B` 为逻辑序列数（TND 由 `cu_seqlens` 元素数减一得到），`S` 为 padded 存储长度，`T` 为 packed token 总数，`Nqk` 为共享 Q/K 头数，`Nv` 为 V/状态头数，`J=ceil(S/64)`（TND 为 `ceil(T/64)+B`）为 chunk 槽位数。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | Input | BNSD `[B,Nqk,S,128]`；BSND `[B,S,Nqk,128]`；TND `[T,Nqk,128]` | bfloat16 | query 张量。 |
| `k` | Input | 与 `q` 相同 | bfloat16 | key 张量。 |
| `v` | Input | BNSD `[B,Nv,S,128]`；BSND `[B,S,Nv,128]`；TND `[T,Nv,128]` | bfloat16 | value 张量。 |
| `g` | Input | 与 `v` 相同 | bfloat16 / float32 | 逐维 log decay 输入；内核先转 BF16 再升 FP32。 |
| `beta` | Input | 与 `v` 去掉最后一维相同，如 `[B,Nv,S]`、`[B,S,Nv]`、`[T,Nv]` | bfloat16 / float32 | beta 输入；内核转为 FP32。 |
| `metadata` | Input，可选 | 标量 | Tensor | 仅为调用位置兼容保留，本版本只接受 `None`。 |
| `initial_state` | Input，可选 | `[B,Nv,128,128]`，末两维为 `[Dv,Dk]` | float32 | 缺省为全零状态，不被原地更新。 |
| `A_log` | Input | `[Nv]` | float32 | gate 参数，`a=exp(A_log)`；`use_gate_in_kernel=True` 时必需。 |
| `dt_bias` | Input，可选 | `[Nv,128]` | float32 | gate 偏置，缺省按零偏置。 |
| `cu_seqlens` | Input，可选 | `[B+1]` | int32 | 累计序列边界；TND 必需。 |
| `seqused` | Input，可选 | `[B]` | int32 | padded 布局各序列有效长度。 |
| `scale` | Input，关键字 | 标量 | Python `float` | 输出缩放，默认 `128**-0.5`；`None` 等价默认值。 |
| `output_final_state` | 关键字 | 标量 | `bool` | 是否返回最终状态，默认 `False`。 |
| `return_intermediate_states` | 关键字 | 标量 | `bool` | 是否返回各 chunk 入口状态 `h`，默认 `False`。 |
| `use_qk_l2norm_in_kernel` | 关键字 | 标量 | `bool` | 是否在 kernel 内做 Q/K L2 归一化，默认 `True`。 |
| `use_gate_in_kernel` | 关键字 | 标量 | `bool` | 是否在 kernel 内计算 gate，默认 `True`。 |
| `use_beta_sigmoid_in_kernel` | 关键字 | 标量 | `bool` | 是否在 kernel 内做 beta 激活，默认 `True`。 |
| `allow_neg_eigval` | 关键字 | 标量 | `bool` | beta 激活用 `2*sigmoid(beta)`，默认 `False`。 |
| `disable_recompute` | 关键字 | 标量 | `bool` | 是否保存六项训练重算缓存，默认 `False`。 |
| `lower_bound` | 关键字 | 标量 | Python `float` | 有界 gate 下界，默认 `None`；取值须满足 `-5 <= lower_bound < 0`。 |
| `layout` | 关键字 | 标量 | Python `str` | 输入布局，支持 `BNSD`、`BSND`、`TND`，默认 `TND`。 |

返回值固定 16 项，顺序为 `out, final_state, h, Aqk, Akk, w, u, qg, kg, v_new, g_cumsum, q_norm, q_rstd, k_norm, k_rstd, beta_sigmoid`；未启用的槽位返回 `None`，顺序不变。训练模式指未启用 `torch.inference_mode()`。

| 返回值 | 返回条件 | shape / dtype |
| --- | --- | --- |
| `out` | 始终返回 | BF16，与 `v` 同 shape、同布局 |
| `final_state` | `output_final_state=True` | FP32 `[B,Nv,128,128]` |
| `h` | `return_intermediate_states=True` | FP32 `[B,J,Nv,128,128]`；TND `[J,Nv,128,128]` |
| `Aqk`、`Akk` | 训练模式 | BF16 `[B,J,Nv,64,64]`；TND `[J,Nv,64,64]` |
| `w`、`u`、`qg`、`kg`、`v_new` | 训练模式且 `disable_recompute=True` | BF16，与 `v` 同 shape、同布局 |
| `g_cumsum` | 训练模式且 `disable_recompute=True` | FP32 `[B,J,Nv,64,128]`；TND `[J,Nv,64,128]` |
| `q_norm`、`k_norm` | 训练模式且 `use_qk_l2norm_in_kernel=True` | BF16，`[B,Nqk,S,128]` 或 TND `[T,Nqk,128]` |
| `q_rstd`、`k_rstd` | 同上一项 | FP32，`[B,Nqk,S]` 或 TND `[T,Nqk]` |
| `beta_sigmoid` | 训练模式且 `use_beta_sigmoid_in_kernel=True` | FP32 `[B,S,Nv]`；TND `[T,Nv]` |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 产品型号 | Ascend 950 系列产品，编译选项 `--npu-arch 3510`。 |
| chunk 大小 | `CHUNK` 固定为 64，不支持其它取值。 |
| 维度 | `Dk=Dv=128` 固定，`q`、`k`、`v` 的最后一维必须为 128。 |
| 头数 | `1 <= Nqk <= Nv <= 96` 且 `Nv % Nqk == 0`；V 头 j 使用 Q/K 头 `floor(j/(Nv/Nqk))`。 |
| 布局 | 支持 BNSD、BSND、TND；BNSD/BSND 输入为 rank 4，TND 为 rank 3；`q` 与 `k` shape 必须一致。 |
| dtype | `q`/`k`/`v` 为 BF16；`g`、`beta` 为 BF16 或 FP32；`initial_state`、`A_log`、`dt_bias` 为 FP32；`cu_seqlens`、`seqused` 为 int32。 |
| 张量属性 | 输入须为连续 Tensor 且位于同一 NPU；不修改输入，输出不与输入别名。 |
| 长度 | `B`、`S`、`T` 及每条序列有效长度为正整数；支持长度不是 64 整数倍的尾块。 |
| 长度向量 | `cu_seqlens` 首项为 0、严格递增，TND 末项必须等于 T，padded 各段差分不大于 S；与 `seqused` 同时提供时逐序列长度必须相等，缺省 padded 长度为 S。 |
| gate 开关 | 启用时 `A_log` 必需，`lower_bound=None` 用 `ell=-a*softplus(x)`，否则 `ell=lower_bound*sigmoid(a*x)`；关闭时忽略 `A_log`、`dt_bias`、`lower_bound`。 |
| beta 开关 | `allow_neg_eigval=True` 要求 `use_beta_sigmoid_in_kernel=True`；关闭 beta 激活时 `beta` 直接作为已激活系数。 |
| metadata | 仅接受 `None`，非 `None` 在 Host 抛出 `ValueError`。 |
| 输入语义 | `g` 先转 BF16 再升 FP32 参与计算；关闭 gate 的 `g` 应为非正 log decay；模型输入要求有限数。 |
| 边界行为 | padded 布局 `out` 的无效 token 置零；尾块只处理有效 token，padding 不参与状态更新；TND 序列之间不共享状态；`initial_state` 缺省为 FP32 零，`h` 的首个有效 chunk 等于该序列初态。 |
| 非法输入 | 非法布局/dtype/shape/stride/device、`D != 128`、头数不整除、非法 `lower_bound`、非有限 `scale`、非 bool 开关在启动计算内核前抛出 `ValueError`。 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/attention:$PYTHONPATH python - <<'PY'
import torch
import torch_npu
from flash_kda_fwd.flash_kda_fwd import flash_kda_fwd

torch.npu.set_device(0)
q = torch.randn((1, 2, 128, 128), device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn((1, 4, 128, 128), device="npu", dtype=torch.bfloat16)
g = torch.randn(v.shape, device="npu", dtype=torch.float32)
beta = torch.randn(v.shape[:-1], device="npu", dtype=torch.float32)
A_log = torch.zeros(4, device="npu", dtype=torch.float32)
with torch.inference_mode():
    out, final_state, *cache = flash_kda_fwd(
        q, k, v, g, beta, A_log=A_log, layout="BNSD",
        lower_bound=-5.0, output_final_state=True,
    )
torch.npu.synchronize()
print(out.shape, final_state.shape)
PY
```
