# KdaBwd Example Readme

## 功能说明

- 算子功能：KDA（Kimi Delta Attention）分块（WY）反向，按 BSND 布局对 Q/K/V、逐 K 通道真对数衰减与 beta 求五个梯度，返回 `dq/dk/dv/dg/dbeta`；入口为 `kda_bwd.kda_bwd` 模块中的 `kda_bwd`。
- 计算公式：记 chunk 长 `C = 64`、`lg = exp(cumsum(g))`、`kh = lg*k`、`kg = beta*k/lg`、`qg = scale*lg*q`、`A = inverse(I + strict_lower(kh @ kg^T))`，则

  ```text
  逐 chunk： c = A kh        u = A v        x = u - c S_in
            intra = tril(qg kg^T)          o = qg S_in + intra x
            S_out = lg[-1] S_in + (lg[-1] kg)^T x

  反向（dS 为该 chunk 的 dS_out）：
            dx   = intra^T dO + w dS        dv = A^T dx
            dkh  = -(A^T dx) S_in^T         dL = -dv x^T
            dkg  = dL^T kh + tril(dO x^T)^T qg + lg[-1] dS^T x
            dqg  = dO S_in^T + tril(dO x^T) kg
            dq   = scale lg dqg             dk = lg dkh + beta/lg dkg
            dbeta = sum_K(k/lg dkg)         dg = reverse_cumsum(lg dlg)
  ```

  其中 `lg dlg = dqg qg + dkh kh - dkg kg`，末位再加 `sum_V(dS S_in) + sum_C((lg[-1] dS^T x) kg)`。
- 支持产品型号：Ascend 950PR。

## 参数说明

```python
def kda_bwd(
    q, k, v, g, beta, do,
    scale=None,
):
    return dq, dk, dv, dg, dbeta
```

记号：`B` 为批量，`T` 为序列长度，`HK` 为 Q/K 头数，`HV` 为 V 头数，`K=V=128` 为 head dim。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | 输入 | `[B, T, HK, 128]` | BF16 | Q，与 K 共用 `HK` 个头 |
| `k` | 输入 | `[B, T, HK, 128]` | BF16 | K，shape 必须与 `q` 完全相同 |
| `v` | 输入 | `[B, T, HV, 128]` | BF16 | V |
| `g` | 输入 | `[B, T, HV, 128]` | BF16 / FP32 | 已激活的逐 K 通道自然对数衰减，`g ≤ 0` |
| `beta` | 输入 | `[B, T, HV]` | BF16 / FP32 | 已激活的 beta 系数 |
| `do` | 输入 | `[B, T, HV, 128]` | BF16 | 输出梯度，shape 必须等于 `v` |
| `scale` | 输入（可选） | 标量 | float / None | Q 的缩放；默认 `None` 取 `128**-0.5` |
| `dq` | 输出 | `[B, T, HK, 128]` | FP32 | 已按 GQA 组求和回 Q 头 |
| `dk` | 输出 | `[B, T, HK, 128]` | FP32 | 已按 GQA 组求和回 K 头 |
| `dv` | 输出 | `[B, T, HV, 128]` | FP32 | 按 V 头 |
| `dg` | 输出 | `[B, T, HV, 128]` | FP32 | 按 V 头，chunk 内反向累加、chunk 边界重开 |
| `dbeta` | 输出 | `[B, T, HV]` | FP32 | 按 V 头 |

返回值固定为五元组 `(dq, dk, dv, dg, dbeta)`。

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 布局 | 全部 Tensor 为 4 维连续 BSND，位于同一 NPU，不允许非连续 |
| 维度 | `B ≥ 1`、`T ≥ 1`、`HK ≥ 1`；支持非整块尾部 |
| 头映射 | `HV/HK` 必须整除且 `∈ {1,2,4}`（GQA） |
| K/V | 必须均为 128；`q` 与 `k` shape 相同，`v` 与 `do` 为 `[B, T, HV, 128]` |
| dtype | `q`/`k`/`v`/`do` 必须 BF16；`g`/`beta` 只能 BF16 或 FP32；输出固定 FP32 |
| 取值 | `g ≤ 0`、`0 ≤ beta ≤ 1`；`scale` 必须是有限实数或 `None`；Host 不回读 Tensor 检查值域，也不执行 gate 激活、beta sigmoid 或 Q/K 归一化 |
| chunk | 计算块长固定为 64；`T` 不是 64 的整数倍时按零填充计算并切回原长度，填充行不参与状态递推 |
| 数值行为 | `beta = 0` 的 token 不出现 `0/0`：`dbeta` 的链式因子取 `k/lg`，结果有限（其余四路为 0，该 token 不更新状态）；chunk 内累积衰减按 `lg = max(exp(cumsum(g)), 1e-30)` 截断使 `1/lg` 有界，但每步衰减达 `-30` 量级时 `beta/lg` 仍会溢出，`dq/dk/dg` 可能非有限；输入取 BF16 最大有限值时中间量也可能溢出 |
| 其他 | 不修改输入、输出与输入不别名；使用输入设备的当前 stream；地址按 signed 64-bit 检查字节数 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/attention:$PYTHONPATH python - <<'PY'
import torch
import torch_npu
from kda_bwd.kda_bwd import kda_bwd

B, T, HK, HV = 1, 128, 2, 4
device = "npu:0"
torch.npu.set_device(device)

q = torch.randn(B, T, HK, 128, dtype=torch.bfloat16, device=device)
k = torch.randn(B, T, HK, 128, dtype=torch.bfloat16, device=device)
v = torch.randn(B, T, HV, 128, dtype=torch.bfloat16, device=device)
do = torch.randn(B, T, HV, 128, dtype=torch.bfloat16, device=device)
g = -torch.nn.functional.softplus(                                      # 已激活衰减，g <= 0
    torch.randn(B, T, HV, 128, dtype=torch.float32, device=device))
beta = torch.sigmoid(torch.randn(B, T, HV, dtype=torch.float32, device=device))

dq, dk, dv, dg, dbeta = kda_bwd(q, k, v, g, beta, do)

print(dq.shape, dq.dtype)        # torch.Size([1, 128, 2, 128]) torch.float32
print(dv.shape, dbeta.shape)     # torch.Size([1, 128, 4, 128]) torch.Size([1, 128, 4])
PY
```
