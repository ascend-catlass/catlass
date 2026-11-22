# SageAttention Example Readme

## 功能说明

- 算子功能：计算 causal 的 Sage 风格注意力，入口为 `sage_attention` 模块的 `run`
  函数。Q/K 使用 INT8 量化，PV 使用 E4M3FN FP8 量化，QK GEMM、FP32 online
  softmax、P 的 FP8 量化与 PV GEMM 融合在一个 CATLASS DSL AIC/AIV mixed kernel
  中，支持定长与变长 batch、GQA/MQA、HND/NHD 布局和可选的 base-2 LSE。
- 计算公式：记 K 按 64 token 分块，query 按 128 行分块，`kh = h / (Hq / Hkv)`，
  `qb = r / 128`，`kb = t / 64`。对每个 query 行 `r`、head `h`：

  $$
  \begin{aligned}
  s_{r,h,t} &=
    q\_scale[b,h,qb] \cdot k\_scale[b,kh,kb] \cdot
    \sum_d Q[r,h,d] \cdot K[t,kh,d] \\
  m_{r,h} &= \max_{t \le r} s_{r,h,t} \\
  e_{r,h,t} &= 2^{\,s_{r,h,t} - m_{r,h}} \\
  P_{r,h,t} &= \operatorname{round\_e4m3}(448 \cdot e_{r,h,t}) \\
  O_{r,h,:} &=
    \frac{\sum_t P_{r,h,t} V[t,kh,:]}{\sum_t e_{r,h,t}} \cdot
    \frac{v\_scale[b,kh,:]}{448} \\
  LSE_{r,h} &= m_{r,h} + \log_2 \sum_t e_{r,h,t}
  \end{aligned}
  $$

  `q_scale` 已包含 `softmax_scale * log2(e)`，因此 `s` 位于 log2 域；kernel 内部再乘
  `ln2` 后用自然指数 `exp` 实现 `2^x`。Q/K 以 INT8 输入、INT32 累加，P 与 V 以
  E4M3FN 相乘、FP32 累加，分母始终使用量化前的 FP32 指数和。空 KV 时输出为 NaN。
- 支持产品型号：Ascend 950PR&950DT 系列产品。

## 参数说明

```python
out, lse = run(
    q, k, v, q_scale, k_scale, v_scale,
    cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None,
    cu_seqlens_q_scale=None, cu_seqlens_k_scale=None,
    tensor_layout="HND", output_dtype="float16", return_lse=False,
)
```

五个 varlen 参数（`cu_seqlens_*`、`max_seqlen_q`）必须同时给出或同时省略；给出时进入
变长模式并只返回 `out`。记 `B` 为 batch，`T` 为所有序列的 token 总数，`Hq`/`Hkv` 为
query/KV head 数，`N` 为定长序列长度，`D` 为 head dim，`Pq = sum(ceil(Nq / 128))`、
`Pk = sum(ceil(Nk / 64))` 为变长模式下的 scale 块总数。

| 参数 | 属性 | 定长 shape | 变长 shape | dtype | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | Input | `[B, Hq, N, D]`（HND）或 `[B, N, Hq, D]`（NHD） | `[T, Hq, D]` | `int8` | 预量化 query |
| `k` | Input | `[B, Hkv, N, D]`，布局同 `q` | `[T, Hkv, D]` | `int8` | 预量化 key |
| `v` | Input | 与 `k` 相同 | 与 `k` 相同 | `float8_e4m3fn` | 预量化 value，逻辑布局与 K 相同 |
| `q_scale` | Input | `[B, Hq, ceil(N/128)]` | `[Pq, Hq]` | `float32` | per128 query 反量化，已含 `softmax_scale * log2(e)` |
| `k_scale` | Input | `[B, Hkv, ceil(N/64)]` | `[Pk, Hkv]` | `float32` | per64 key 反量化 |
| `v_scale` | Input | `[B, Hkv, D]` | `[B, Hkv, D]` | `float32` | 逐通道 value 反量化 |
| `cu_seqlens_q` | Input | - | `[B+1]` | `int32`/`int64` | query 前缀和，从 0 开始 |
| `cu_seqlens_k` | Input | - | `[B+1]` | `int32`/`int64` | key 前缀和 |
| `max_seqlen_q` | Input | - | 标量 | Python `int` | 不小于实际最大 query 长度 |
| `cu_seqlens_q_scale` | Input | - | `[B+1]` | `int32`/`int64` | query scale 块前缀和 |
| `cu_seqlens_k_scale` | Input | - | `[B+1]` | `int32`/`int64` | key scale 块前缀和 |
| `tensor_layout` | Input | 标量 | 固定 `"HND"` | Python `str` | 定长可选 `"HND"`/`"NHD"` |
| `output_dtype` | Input | 标量 | 标量 | Python `str` | `"float16"`/`"bfloat16"`/`"float32"` |
| `return_lse` | Input | 标量 | 不支持 | Python `bool` | 仅定长可用 |
| `out` | Output | 与 `q` 相同 | 与 `q` 相同 | 同 `output_dtype` | 注意力输出 |
| `lse` | Output | `[B, Hq, N]` 或 CPU 空张量 | 不返回 | `float32` | `return_lse=True` 时返回 base-2 LSE；定长返回 `(out, lse)`，变长只返回 `out` |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 维度 | `D ∈ {64, 128}` |
| Head | `Hq % Hkv == 0`，支持 MHA、GQA 和 MQA |
| 注意力类型 | 仅 causal；定长模式要求 `Nq == Nk`，变长模式允许 `qlen != klen` |
| 布局 | 定长支持 HND/NHD；变长仅支持 TND（`[T, head, D]`），且不支持 LSE |
| Scale | `q_scale` 含 `softmax_scale * log2(e)`；`k_scale` 按 64 token 分块；`v_scale` 必须有限且非负 |
| 量化 | `q`/`k` 必须是 INT8，`v` 必须是 E4M3FN；V 不在 kernel 内做量化或平滑 |
| 内存 | 所有输入张量连续、位于同一 NPU；元素数与 task 数需可按有符号 int32 寻址 |
| 空 KV | `klen == 0` 时输出为 NaN，分母为 0 |
| 分块 | 固定 `bm=128`（query 行）、`bn=512`（key 行）、`kc=256`（L0 GEMM 子块），P 在整个 512 key tile 上统一量化 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/attention:$PYTHONPATH python - <<'PY'
import torch
import torch_npu  # noqa: F401

from sage_attention.sage_attention import run

B, Hq, Hkv, N, D = 2, 8, 2, 1024, 128
q = torch.randint(-16, 16, (B, Hq, N, D), dtype=torch.int8, device="npu:0")
k = torch.randint(-16, 16, (B, Hkv, N, D), dtype=torch.int8, device="npu:0")
v = torch.randn(B, Hkv, N, D, dtype=torch.float32, device="npu:0").to(torch.float8_e4m3fn)
q_scale = torch.rand(B, Hq, N // 128, dtype=torch.float32, device="npu:0") * (D ** -0.5) * 1.442695
k_scale = torch.rand(B, Hkv, N // 64, dtype=torch.float32, device="npu:0")
v_scale = torch.rand(B, Hkv, D, dtype=torch.float32, device="npu:0")

out, lse = run(
    q, k, v, q_scale, k_scale, v_scale,
    tensor_layout="HND",
    output_dtype="float16",
    return_lse=True,
)
print(out.shape, out.dtype, lse.shape)
PY
```

预期输出：

```text
torch.Size([2, 8, 1024, 128]) torch.float16 torch.Size([2, 8, 1024])
```
