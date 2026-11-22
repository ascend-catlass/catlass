# SplitQkvRmsNormMrope Example Readme

## 功能说明

- 算子功能：把打包的 QKV 按 head 拆分，对 Q、K 执行 RMSNorm 与多模态旋转位置编码（MRoPE），原样复制 V 和可选 gate；入口为 `split_qkv_rmsnorm_mrope` 模块的 `run` 函数，基于 CATLASS DSL 实现的 AIV kernel。
- 计算公式：对每个 Q 或 K head，先做 RMSNorm 加权

  $$
  y_i = \frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=0}^{D-1}x_j^2+\epsilon}}w_i+b_i
  $$

  再对前 `R` 维按前后半区配对旋转，其余维度保留 `y`；未提供偏置时 `b=0`。中间计算使用 FP32，输出转换为输入 dtype。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
run(
    qkv, q_weight, k_weight, cos_sin,
    num_q_heads, num_kv_heads, head_size, eps,
    mrope_section, is_interleaved,
    rope_dim=None, q_bias=None, k_bias=None, has_gate=False,
)
```

记 `T` 为 token 数，`HQ`、`HK` 分别为 Q、KV head 数，`D` 为 head 维度，`R` 为旋转维度，`P = ((2 if has_gate else 1) * HQ + 2 * HK) * D`；返回 `(q, k, v, gate)`。

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `qkv` | Input | `[T, P]` | `float16` / `bfloat16` | 无 gate 时按 `[Q, K, V]` 打包；有 gate 时为 `[q0, gate0, q1, gate1, ..., K, V]`。 |
| `q_weight` | Input | `[D]` | 同 `qkv` | Q 的 RMSNorm 权重，各 Q head 共用。 |
| `k_weight` | Input | `[D]` | 同 `qkv` | K 的 RMSNorm 权重，各 KV head 共用。 |
| `cos_sin` | Input | `[3, T, R]` | 同 `qkv` | 时间、高度、宽度三个轴；各轴前 `R/2` 项为 cos，后 `R/2` 项为 sin。 |
| `num_q_heads` | Input | 标量 | Python `int` | `HQ`。 |
| `num_kv_heads` | Input | 标量 | Python `int` | `HK`。 |
| `head_size` | Input | 标量 | Python `int` | head 维度 `D`。 |
| `eps` | Input | 标量 | Python `float` / `int` | RMSNorm 的正有限常数。 |
| `mrope_section` | Input | 长度 3 的序列 | Python `list` / `tuple` | 三个轴的分段配置 `(st, sh, sw)`，非负整数，和为 `R/2`。 |
| `is_interleaved` | Input | 标量 | Python `bool` | 是否交错选择 MRoPE 轴，不改变旋转元素的配对方式。 |
| `rope_dim` | Input，可选 | 标量 | Python `int` / `None` | 旋转维度 `R`；默认 `None`，即取 `D`。 |
| `q_bias`、`k_bias` | Input，可选 | `[D]` | 同 `qkv` | 旋转前加上的偏置；默认 `None`，须同时提供或同时省略。 |
| `has_gate` | Input，可选 | 标量 | Python `bool` | 是否拆分 gate，默认 `False`。 |
| `q` | Output | `[T, HQ*D]` | 同 `qkv` | RMSNorm 和 MRoPE 后的 Q。 |
| `k` | Output | `[T, HK*D]` | 同 `qkv` | RMSNorm 和 MRoPE 后的 K。 |
| `v` | Output | `[T, HK*D]` | 同 `qkv` | 原样复制的 V。 |
| `gate` | Output | `[T, HQ*D]` 或 `[T, 0]` | 同 `qkv` | gate；`has_gate=False` 时为空张量。 |

对频率下标 `0 ≤ j < R/2`：`is_interleaved=False` 时 `j < st` 用时间轴、`st ≤ j < st+sh` 用高度轴、其余用宽度轴；`is_interleaved=True` 时 `j % 3 == 1` 且 `j < 3*sh` 用高度轴、`j % 3 == 2` 且 `j < 3*sw` 用宽度轴、其余用时间轴。

## 约束说明

| 项目 | 约束 |
| ----- | ----- |
| 维度 | `T > 0`；`HQ`、`HK`、`D` 为正的 2 的幂，且 `HQ % HK == 0`；`D` 为 128 或 256。 |
| 旋转维度 | `R` 为 2 的幂且 `2 ≤ R ≤ D`，实现仅支持 `R` 取 64、128、256。 |
| 分段 | `mrope_section` 为 3 个非负整数，且 `2 * sum(mrope_section) == R`。 |
| dtype | `qkv` 为 `float16` 或 `bfloat16`，权重、偏置、`cos_sin` 与之相同。 |
| 布局与设备 | 全部输入连续，位于当前 NPU；调用前设置 `torch.npu.set_device`。 |
| 偏置 | `q_bias`、`k_bias` 必须同时提供或同时省略，否则入口抛 `ValueError`。 |
| 片上容量 | head 数受编译目标的 UB 容量限制，超限配置无法编译。 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/vec:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from split_qkv_rmsnorm_mrope.split_qkv_rmsnorm_mrope import run

torch.npu.set_device(0)
t, hq, hk, d = 32, 8, 2, 128
qkv = torch.randn(t, (hq + 2 * hk) * d, dtype=torch.float16, device="npu:0")
q_weight = torch.ones(d, dtype=qkv.dtype, device=qkv.device)
k_weight = torch.ones_like(q_weight)
cos_sin = torch.randn(3, t, d, dtype=qkv.dtype, device=qkv.device)

q, k, v, gate = run(
    qkv, q_weight, k_weight, cos_sin,
    hq, hk, d, 1e-6, (24, 20, 20), False,
)
print(q.shape, k.shape, v.shape, gate.shape)
PY
```

```text
torch.Size([32, 1024]) torch.Size([32, 256]) torch.Size([32, 256]) torch.Size([32, 0])
```
