# GDN2Fwd Example Readme

## 功能说明

- 算子功能：Gated DeltaNet 2（GDN2）定长前向，按连续 BNSD 布局对 Q/K/V 与逐通道 gate 做门控线性注意力状态递推，返回输出 `o` 及可选最终状态；入口为 `gdn2_fwd.gdn2_fwd` 模块中的 `gdn2_fwd`。
- 计算公式：逐 token 递推，`D = exp(g)[:, None] * H_prev`；`erase = (b * k)^T @ D`；`delta = w_gate * v - erase`；`H = D + k[:, None] * delta[None, :]`；`o = scale * q^T @ H`。
- 支持产品型号：Ascend 950PR。

## 参数说明

```python
def gdn2_fwd(
    q, k, v, g, b, w_gate,
    scale=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
):
    return o, final_state
```

记号：`B` 为批量，`T` 为序列长度，`HK` 为 Q/K 头数，`HV` 为 V 与输出头数，`HG`/`HB` 分别为 `g`/`b` 的头数，`K=V=128` 为 head dim。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `q` | 输入 | `[B, HK, T, 128]` | BF16 | Q，与 K 共用 `HK` 个头 |
| `k` | 输入 | `[B, HK, T, 128]` | BF16 | K，shape 必须与 `q` 完全相同 |
| `v` | 输入 | `[B, HV, T, 128]` | BF16 | V |
| `g` | 输入 | `[B, HG, T, 128]` | BF16 / FP32 | 已激活的逐 K 通道自然对数衰减，`HG ∈ {HK, HV}` |
| `b` | 输入 | `[B, HB, T, 128]` | BF16 / FP32 | 逐 K 通道 erase gate，`HB ∈ {HK, HV}` |
| `w_gate` | 输入 | `[B, HV, T, 128]` | BF16 / FP32 | 逐 V 通道 write gate，shape 必须等于 `v` |
| `scale` | 输入（可选） | 标量 | float / None | 输出缩放；默认 `None` 取 `128**-0.5` |
| `initial_state` | 输入（可选） | `[B, HV, 128, 128]` | FP32 | K-first 初态；默认 `None` 表示全零初态 |
| `output_final_state` | 输入（可选） | 标量 | bool | 是否返回最终状态，默认 `False` |
| `chunk_size` | 输入（可选） | 标量 | int | 逻辑块长，仅支持 64 或 128，默认 64 |
| `o` | 输出 | `[B, HV, T, 128]` | 同 `v`（BF16） | 前向输出，按 round-to-nearest-even 转 BF16 |
| `final_state` | 输出 | `[B, HV, 128, 128]` | FP32 | K-first 最终状态；`output_final_state=False` 时为 `None` |

返回值固定为二元组 `(o, final_state)`。

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 布局 | 全部 Tensor 为 4 维连续 BNSD，位于同一 NPU，不允许非连续 |
| 维度 | `B ≥ 1`、`T ≥ 1`、`HK ≥ 1`；支持非整块尾部 |
| 头映射 | `HV/HK` 必须整除且 `∈ {1,2,4}`（GVA），`1 ≤ HV ≤ 96`；不支持任意 GQA 头配置 |
| K/V | 必须均为 128；`q` 与 `k` shape 相同，`v` 为 `[B, HV, T, 128]` |
| g/b 头数 | 各自只能取 `HK` 或 `HV` |
| dtype | `q`/`k`/`v` 必须 BF16；`g`/`b`/`w_gate` 只能 BF16 或 FP32；`initial_state` 只能 FP32；`output_final_state` 必须 bool |
| 取值 | 所有值有限，`g ≤ 0`、`b ∈ [0,2]`、`w_gate ∈ [0,1]`；`scale` 必须是有限实数或 `None`；Host 不回读 Tensor 检查值域，也不执行 gate 激活或 Q/K 归一化 |
| chunk_size | 仅 64 或 128；设备物理递推块固定为 64，逻辑块 128 由两个 64 块组成 |
| 数值行为 | 状态以 FP32 乘法与加法更新，中间量用 BF16 分量近似；不支持反向、变长 packed 序列、CP、V-first 状态与标量 beta 自动广播 |
| 其他 | 不修改输入、输出与输入不别名、不构建反向计算图；使用输入设备的当前 stream；地址按 signed 64-bit 检查字节数 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/attention:$PYTHONPATH python - <<'PY'
import torch
import torch_npu
from gdn2_fwd.gdn2_fwd import gdn2_fwd

B, HK, HV, T = 1, 16, 32, 1024
device = "npu:0"

q = torch.randn(B, HK, T, 128, dtype=torch.bfloat16, device=device)
k = torch.randn(B, HK, T, 128, dtype=torch.bfloat16, device=device)
v = torch.randn(B, HV, T, 128, dtype=torch.bfloat16, device=device)
g = torch.zeros(B, HK, T, 128, dtype=torch.bfloat16, device=device)      # 已激活衰减，g <= 0
b = torch.rand(B, HK, T, 128, dtype=torch.bfloat16, device=device)       # erase gate, [0, 2]
w_gate = torch.rand(B, HV, T, 128, dtype=torch.float32, device=device)   # write gate, [0, 1]

o, final_state = gdn2_fwd(
    q, k, v, g, b, w_gate,
    scale=None,
    initial_state=None,
    output_final_state=True,
    chunk_size=64,
)

print(o.shape, o.dtype)          # torch.Size([1, 32, 1024, 128]) torch.bfloat16
print(final_state.shape)         # torch.Size([1, 32, 128, 128])
PY
```
