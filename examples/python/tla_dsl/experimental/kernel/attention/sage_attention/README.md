# SageAttention Example Readme

## 功能说明

- 算子功能：计算 causal 的 Sage 风格注意力，QK 使用 INT8 量化，PV 使用
  E4M3FN FP8 量化。将 QK GEMM、FP32 online softmax、P 的 FP8 量化与 PV
  GEMM 融合到一个 CATLASS DSL AIC/AIV mixed kernel 中，支持定长与变长
  batch、GQA/MQA、HND/NHD 布局和可选的 base-2 LSE。
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

  `q_scale` 已包含 `softmax_scale * log2(e)`，因此 `s` 位于 log2 域；kernel
  内部再乘 `ln2` 后用自然指数 `exp` 实现 `2^x`。Q/K 以 INT8 输入、INT32
  累加，P 与 V 以 E4M3FN 相乘、FP32 累加，分母始终使用量化前的 FP32 指数和。
  空 KV 时输出为 NaN，与参考实现一致。

- 支持产品型号：Ascend 950PR&950DT系列产品；当前编译目标为
  `--npu-arch 3510`。

## 参数说明

定长模式（不传 `cu_seqlens_*` 与 `max_seqlen_q`）：

```python
out, lse = sage_attention.run(
    q, k, v, q_scale, k_scale, v_scale,
    tensor_layout="HND", output_dtype="float16", return_lse=False,
)
```

变长模式（五个 varlen 参数必须同时给出）：

```python
out = sage_attention.run(
    q, k, v, q_scale, k_scale, v_scale,
    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    cu_seqlens_q_scale=cu_seqlens_q_scale,
    cu_seqlens_k_scale=cu_seqlens_k_scale,
    output_dtype="float16",
)
```

记 `B` 为 batch，`T` 为所有序列的 token 总数，`Hq`/`Hkv` 为 query/KV head
数，`N` 为定长序列长度，`D` 为 head dim，`Pq = sum(ceil(Nq / 128))`、
`Pk = sum(ceil(Nk / 64))` 为 packed 模式下的 scale 块总数。

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
| `lse` | Output | `[B, Hq, N]` 或 CPU 空张量 | 不返回 | `float32` | `return_lse=True` 时返回 base-2 LSE；定长模式返回 `(out, lse)` 二元组，变长模式只返回 `out` |

## 约束说明

| 项目 | 约束 |
| --- | --- |
| 维度 | `D ∈ {64, 128}` |
| Head | `Hq % Hkv == 0`，支持 MHA、GQA 和 MQA |
| 注意力类型 | 仅 causal；定长模式要求 `Nq == Nk`，变长模式允许 `qlen != klen` |
| 布局 | 定长支持 HND/NHD；变长仅支持 TND（`[T, head, D]`），且不支持 LSE |
| Scale | `q_scale` 含 `softmax_scale * log2(e)`；`k_scale` 按 64 token 分块；`v_scale` 必须有限且非负 |
| 量化 | `q`/`k` 必须是 INT8，`v` 必须是 E4M3FN；V 不在 kernel 内做量化或平滑 |
| 内存 | 所有输入张量连续、位于同一 NPU；张量元素数与 task 数需可按有符号 int32 寻址 |
| 空 KV | `klen == 0` 时输出为 NaN，分母为 0，与参考实现对齐 |
| 分块 | 固定 `bm=128`（query 行）、`bn=512`（key 行）、`kc=256`（L0 GEMM 子块），P 在整个 512 key tile 上统一量化 |

## Kernel 组织

- 一个 task 处理 `(batch, query_head, 128-row query block)`，按 grid-stride
  分布在所有 Cube core 上；每个 task 内 AIC 负责 QK/PV，两个 AIV 各负责
  64 行 query（Fixpipe 的 SPLIT_N/SPLIT_M 分别切 P 列与输出行）。
- Q 在 task 开始时一次搬入 L1（nZ），K/V 按 512 key tile 搬入；每个 tile 的
  QK 拆成两个 256 key 的 L0 子块，两个 L0C score chunk 各自独立，可在
  MMAD 与 Fixpipe 之间流水。
- softmax 全部为纯 TLA SIMD 实现（无 `tla.extern`）：先用 INT32→FP32 缩放、
  causal/补齐 mask 与 running max 将分数原位写回，再计算 `alpha` 与分母；
  第二遍按 4 行一组做 `exp`、累加分母、乘 448 转 E4M3FN，用 pp0..pp3 四个
  cast slot 合成 256Byte 寄存器，再经两级 `deinterleave` 拆成两个 32Byte
  平面写入 UB，最后 MTE3 搬入 L1 的 zN P^T。
- PV 以 256 key 为一块做 FP8×FP8→FP32，Fixpipe 输出到 UB 后由 AIV 完成
  `acc = acc * alpha + o` 的在线累加；全部 key 处理完后按 `1/分母 * v_scale/448`
  归一化并按 `output_dtype` 写回，可选写 LSE。
- 同步：核内各 buffer 用 mutex 表达归属（Cube 侧
  `q1/q0/k1/k0/v1/v0/p0/s0/s1`，Vector 侧 `vscale/pp/out`）；核间仍用 cross
  flag（`qready`、`pready`、`pready_alt`、`oready`、`vzero_ready`）完成
  AIC↔AIV 的 QK 分数、P 与输出交接。

## 使用示例

准备与本仓库 `python/tla_dsl` 匹配的 CATLASS DSL、CANN 与 torch_npu 环境，
导入样例模块后调用 `run`：

```python
import torch
import torch_npu  # noqa: F401

import sage_attention

B, Hq, Hkv, N, D = 2, 8, 2, 1024, 128
q = torch.randint(-16, 16, (B, Hq, N, D), dtype=torch.int8, device="npu:0")
k = torch.randint(-16, 16, (B, Hkv, N, D), dtype=torch.int8, device="npu:0")
v = torch.randn(B, Hkv, N, D, dtype=torch.float32, device="npu:0").to(torch.float8_e4m3fn)
q_scale = torch.rand(B, Hq, N // 128, dtype=torch.float32, device="npu:0") * (D ** -0.5) * 1.442695
k_scale = torch.rand(B, Hkv, N // 64, dtype=torch.float32, device="npu:0")
v_scale = torch.rand(B, Hkv, D, dtype=torch.float32, device="npu:0")

out, lse = sage_attention.run(
    q, k, v, q_scale, k_scale, v_scale,
    tensor_layout="HND",
    output_dtype="float16",
    return_lse=True,
)
print(out.shape, out.dtype, lse.shape)
```

预期输出：

```text
torch.Size([2, 8, 1024, 128]) torch.float16 torch.Size([2, 8, 1024])
```

变长模式（各序列 qlen 可以大于 klen）：

```python
q_len, k_len = 1023, 1001
q = torch.randint(-16, 16, (q_len, Hq, D), dtype=torch.int8, device="npu:0")
k = torch.randint(-16, 16, (k_len, Hkv, D), dtype=torch.int8, device="npu:0")
v = torch.randn(k_len, Hkv, D, dtype=torch.float32, device="npu:0").to(torch.float8_e4m3fn)
q_scale = torch.rand((q_len + 127) // 128, Hq, dtype=torch.float32, device="npu:0")
k_scale = torch.rand((k_len + 63) // 64, Hkv, dtype=torch.float32, device="npu:0")
v_scale = torch.rand(1, Hkv, D, dtype=torch.float32, device="npu:0")
cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device="npu:0")
cu_seqlens_k = torch.tensor([0, k_len], dtype=torch.int32, device="npu:0")

out = sage_attention.run(
    q, k, v, q_scale, k_scale, v_scale,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=q_len,
    cu_seqlens_q_scale=torch.tensor([0, (q_len + 127) // 128], dtype=torch.int32, device="npu:0"),
    cu_seqlens_k_scale=torch.tensor([0, (k_len + 63) // 64], dtype=torch.int32, device="npu:0"),
)
print(out.shape, out.dtype)
```

预期输出：

```text
torch.Size([1023, 8, 128]) torch.float16
```

在 Ascend 950 上以 FP16 与 BF16 输出验证：定长 `N ∈ {1023, 1024, 8191,
8192, 16383, 16384}`，变长序列长度 1k、8k、16k（含 `qlen > klen`），以及
`return_lse=True`。以上用例输出与参考实现的元素匹配率均不低于 0.9999。
