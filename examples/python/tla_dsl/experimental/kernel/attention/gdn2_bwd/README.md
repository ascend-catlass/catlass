# GDN2Bwd Example Readme

## 功能说明

- 算子功能：Gated DeltaNet 2（GDN2）定长反向，按连续 BNSD 布局对 Q/K/V、逐通道 gate 与输出梯度 `do` 做门控线性注意力反向，返回 `dq/dk/dv/db/dw/dg` 六个梯度及进入序列的初始状态梯度 `dh0`；入口为 `gdn2_bwd.gdn2_bwd` 模块中的 `gdn2_bwd`。
- 计算公式：单 chunk（`L≤64`，K-first 状态 `H∈R^{K×V}`）上 `gc=cumsum(g)`、`gl=gc[L-1]`、`kbg=b*k*exp(gc)`、`A=(I+strict_tril(kbg@(k*exp(-gc))^T))^{-1}`、`W=A@kbg`、`Aq=tril((q*exp(gc))@(k*exp(-gc))^T)`，前向与反向核心式：

```text
v_new = A@(w_gate*v) - W@H,        H' = exp(gl)*H + (k*exp(gl-gc))^T@v_new
dh    = exp(gl)*dh_out + scale*(q*exp(gc))^T@do - W^T@dv_new
dv_new= scale*Aq^T@do + (k*exp(gl-gc))@dh_out
dq    = scale*exp(gc)*(do@H^T) + exp(gc)*(dAq@(k*exp(-gc)));  dk = exp(-gc)*(dAq^T@(q*exp(gc))) + exp(gl-gc)*(v_new@dh_out^T)
dv    = w_gate*(A^T@dv_new);  dw = v*(A^T@dv_new);  db = dkbg*(k*exp(gc));  dg = reverse_cumsum(dgc)
```

其中 `dh_out` 为下一 chunk 传入的状态梯度，`dAq`/`dkbg`/`dgc` 为对应的局部梯度部分和（多 chunk 时按逆序把 `dh` 传回）；`dg` 由 `dgc` 在 chunk 内逆累计得到。

- 支持产品型号：Ascend 950PR。

## 参数说明

```python
def gdn2_bwd(q, k, v, g, b, w_gate, do, scale=None, initial_state=None, layout="BNSD"):
    return dq, dk, dv, db, dw, dg, dh0
```

记号：`B` 为批量，`T` 为序列长度，`HK` 为 Q/K 头数，`HV` 为 V 与输出头数，`HG`/`HB` 分别为 `g`/`b` 的头数，`K=V=128` 为 head dim；`L` 为 chunk 内有效 token 数（`L ≤ 64`），`H` 为 K-first 状态 `[K,V]`。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | 输入 | `[B, HK, T, 128]` | BF16 / FP16 | Q，与 K 共用 `HK` 个头 |
| `k` | 输入 | `[B, HK, T, 128]` | 同 `q` | K，shape 必须与 `q` 完全相同 |
| `v` | 输入 | `[B, HV, T, 128]` | BF16 / FP16 | V，可与 `q`/`k` 不同 dtype |
| `g` | 输入 | `[B, HG, T, 128]` | 同 `q`/`k` 或 FP32 | 已激活的逐 K 通道自然对数衰减，`HG ∈ {HK, HV}` |
| `b` | 输入 | `[B, HG, T, 128]` | 同 `g` | 逐 K 通道 erase gate，`HB == HG` |
| `w_gate` | 输入 | `[B, HV, T, 128]` | 同 `v` | 逐 V 通道 write gate，shape 必须等于 `v` |
| `do` | 输入 | `[B, HV, T, 128]` | 同 `v` | 输出梯度，shape 必须等于 `v` |
| `dq` | 输出 | `[B, HK, T, 128]` | 同 `q` | Q 的梯度 |
| `dk` | 输出 | `[B, HK, T, 128]` | 同 `k` | K 的梯度 |
| `dv` | 输出 | `[B, HV, T, 128]` | 同 `v` | V 的梯度 |
| `db` | 输出 | `[B, HG, T, 128]` | 同 `b` | erase gate 的梯度 |
| `dw` | 输出 | `[B, HV, T, 128]` | 同 `w_gate` | write gate 的梯度 |
| `dg` | 输出 | `[B, HG, T, 128]` | 同 `g` | 逐 K 通道衰减的梯度 |
| `dh0` | 输出 | `[B, HV, 128, 128]` | FP32 | 进入序列的 K-first 初始状态梯度 |

`scale` 为输出缩放，默认 `None` 取 `128**-0.5`；`initial_state` 为可选 FP32 K-first 初态 `[B, HV, 128, 128]`，默认 `None` 表示全零；`layout` 只接受 `"BNSD"`（默认）与 `"BSND"`。返回值固定为七元组 `(dq, dk, dv, db, dw, dg, dh0)`。

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 布局 | 7 个输入与 6 个 token-major 输出为连续 4 维 BNSD；`layout="BSND"` 时按 `[B, T, H, 128]` 传入/返回，由 Host 侧 `transpose(1,2).contiguous()` 转置（不进 kernel）；全部 Tensor 位于同一 NPU |
| 维度 | `K=V=128`，`B ≥ 1`、`T ≥ 1`；支持非整块尾部（`valid=min(T-64n,64)`），尾块按 padding 区语义处理 |
| 头映射 | `HV/HK` 必须整除且 `∈ {1,2,3,4}`（GVA），`1 ≤ HV ≤ 96`，Q/K 共用 `HK` 个头；`g`/`b` 必须同头数 `HG == HB`，整体取 `HK` 或 `HV`；不支持任意 GQA 头配置 |
| dtype | `q`/`k` 同 dtype（BF16 或 FP16）；`v`/`w_gate`/`do` 同 dtype（BF16 或 FP16，可与 `q`/`k` 不同）；`g`/`b` 同 dtype，为 `q`/`k` 的 dtype 或 FP32；`initial_state` 仅 FP32 |
| 取值域 | 所有值有限：`g ≤ 0`、`b ∈ [0,2]`、`w_gate ∈ [0,1]`，且单 64-token chunk 内 `|Σg| ≤ 88.7`；`scale` 为有限实数或 `None`。Host 不回读 Tensor 检查值域，也不执行 gate 激活或 Q/K 归一化 |
| 不支持项 | 变长 packed 序列、Context Parallel、V-first 状态、标量 `beta` 自动广播、任意 GQA 头配置 |
| 其他 | 不修改输入；七个输出与输入及彼此均不别名；不构建反向计算图；使用输入设备的当前 stream，调用方按需 synchronize |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/attention:$PYTHONPATH python - <<'PY'
import torch
import torch_npu
from gdn2_bwd.gdn2_bwd import gdn2_bwd

B, HK, HV, T = 1, 16, 32, 1024
device = "npu:0"

q = torch.randn(B, HK, T, 128, dtype=torch.bfloat16, device=device) * 0.1
k = torch.randn(B, HK, T, 128, dtype=torch.bfloat16, device=device) * 0.1
v = torch.randn(B, HV, T, 128, dtype=torch.bfloat16, device=device)
g = -torch.rand(B, HK, T, 128, dtype=torch.bfloat16, device=device) * 0.5   # 已激活衰减，g <= 0
b = torch.rand(B, HK, T, 128, dtype=torch.bfloat16, device=device)          # erase gate, [0, 2]
w_gate = torch.rand(B, HV, T, 128, dtype=torch.bfloat16, device=device)     # write gate, [0, 1]
do = torch.randn(B, HV, T, 128, dtype=torch.bfloat16, device=device)        # 输出梯度

dq, dk, dv, db, dw, dg, dh0 = gdn2_bwd(
    q, k, v, g, b, w_gate, do,
    scale=None,
    initial_state=None,
)

for name, t in zip(("dq", "dk", "dv", "db", "dw", "dg", "dh0"),
                   (dq, dk, dv, db, dw, dg, dh0)):
    print(name, tuple(t.shape), t.dtype)
PY
```
