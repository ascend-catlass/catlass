# Mega Recurrent KDA Example Readme

## 功能说明

- 算子功能：在一次混合 AIC/AIV kernel 中融合 QKV、decay A/B、beta、gate 投影，causal Conv1D + SiLU，Q/K 归一化、recurrent KDA、RMSNorm、sigmoid gating 与输出投影；入口函数为 `mega_recurrent_kda`。
- 计算公式：按每个 batch、head、token 表述，state 存储方向为 `[V,K]`，两维均为 128。
  - `raw_qkv=FP32(x@Wqkv.T)`；Q/K/V 各做 4-tap causal depthwise Conv，再取 `SiLU(a)=a/(1+exp(-a))`，转 BF16。
  - `a=BF16(x@Wa.T)`，`g=BF16(a@Wb.T)`，`raw_beta=BF16(x@Wbeta.T)`，`z=BF16(x@Wgate.T)`。
  - `q=FP32(q)/sqrt(sum(q*q)+1e-6)*scale`，k 用同一归一化但不乘 `scale`。
  - `log_decay=lower_bound*sigmoid(exp(A_log)*(g+dt_bias))`，`decay=exp(log_decay)`，`beta=sigmoid(raw_beta)`。
  - `Md=M*decay[None,:]`，`delta=beta*(v-Md@k)`，`Mnew=Md+delta[:,None]*k[None,:]`，`o=Mnew@q`。
  - `norm=o/sqrt(mean(o*o)+rms_norm_eps)*output_norm_weight`，`gated=BF16(norm*sigmoid(z))`，`out=BF16(gated@Wo.T)`。

- 支持产品型号：Ascend 950 系列产品（实测 Ascend 950PR_957c，npu-arch 3510）。

## 参数说明

```python
def mega_recurrent_kda(
    hidden_states, qkv_projection_weight, decay_projection_a_weight,
    decay_projection_b_weight, beta_projection_weight,
    output_gate_projection_weight, output_norm_weight, output_projection_weight,
    conv1d_weight, conv_state, recurrent_state, conv_state_indices,
    ssm_state_indices, conv_num_accepted_tokens, num_accepted_tokens,
    A_log, dt_bias, scale, lower_bound, rms_norm_eps=1e-5,
):
```

记号：`H=7168`、`S=8`、`D=128`、`TP∈{4,8,16}`、`N=96/TP`、`P=N*D`、`B=model_batch*TP`（model_batch 为 1～32）、`C/L` 为 Conv 状态池槽数与历史长度、`R` 为 recurrent 状态池槽数。权重逻辑布局均为输出通道在前，矩阵乘使用 `x @ weight.T`。

| 参数 | 属性 | shape | dtype | 说明 |
|---|---|---|---|---|
| hidden_states | 输入 | `[B,8,7168]` | BF16 / ND | BSH 布局输入 |
| qkv_projection_weight | 输入 | `[3*P,7168]` | BF16 / FRACTAL_NZ | Q、K、V 顺序 |
| decay_projection_a_weight | 输入 | `[128,7168]` | BF16 / FRACTAL_NZ | decay 第一级投影 |
| decay_projection_b_weight | 输入 | `[P,128]` | BF16 / FRACTAL_NZ | decay 第二级投影 |
| beta_projection_weight | 输入 | `[N,7168]` | BF16 / FRACTAL_NZ | beta 投影，保留物理 padding |
| output_gate_projection_weight | 输入 | `[P,7168]` | BF16 / FRACTAL_NZ | gating 投影 |
| output_norm_weight | 输入 | `[128]` | FP32 / ND | RMSNorm 权重，各 head 共享 |
| output_projection_weight | 输入 | `[7168,P]` | BF16 / FRACTAL_NZ | 输出投影 |
| conv1d_weight | 输入 | `[3,N,4,128]` | BF16 / ND | 标准 Conv 布局；兼容 `[4,3*P]`、`[3,4,128]` |
| conv_state | 输入（原地更新） | `[C,L,3*P]` | BF16 / ND | Conv 历史状态池 |
| recurrent_state | 输入（原地更新） | `[R,N,128,128]` | FP32 / ND | recurrent 状态池 |
| conv_state_indices | 输入 | `[B]` | INT32 / ND | 每个 batch 选用的 Conv 槽位 |
| ssm_state_indices | 输入 | `[B*8]` | INT32 / ND | 8 个 token 的 state 槽位，batch-major |
| conv_num_accepted_tokens | 输入 | `[B]` | INT32 / ND | 选择 Conv 历史起点 |
| num_accepted_tokens | 输入 | `[B]` | INT32 / ND | 选择 recurrent 初始状态 |
| A_log | 输入 | `[N]` | FP32 / ND | decay 的 log 参数 |
| dt_bias | 输入 | `[N,128]` | FP32 / ND | decay 偏置 |
| scale | 输入 | 标量 | FP32 | 有限，常用 `128**-0.5` |
| lower_bound | 输入 | 标量 | FP32 | 取值 `[-5,0]` |
| rms_norm_eps | 输入 | 标量 | FP32 | 有限且大于 0，默认 `1e-5` |
| 返回值 | 输出 | `[B,8,7168]` | BF16 / ND | 新分配的连续 Tensor |

## 约束说明

| 项目 | 约束 |
|---|---|
| 维度 | `H=7168`、`S=8`、`D=128`、`N∈{6,12,24}`（对应 TP16/TP8/TP4），model batch 为 1～32；`B` 必须为 `96//N` 的整数倍，不接受空 batch |
| dtype 与设备 | 全部 Tensor 连续且位于同一 NPU；hidden、Conv 权重、Conv state 为 BF16，六个投影权重为 BF16，recurrent state、`output_norm_weight`、`A_log`、`dt_bias` 为 FP32，索引为 INT32 |
| 权重布局 | 六个投影权重必须为 FRACTAL_NZ（format=29）allocation root，不能是 view；Parameter 需先 `.detach()`；NZ 转换在模型初始化时完成 |
| Conv 权重布局 | 标准 `[3,N,4,128]` 为零拷贝单 kernel 路径；`[4,3*P]` 与 `[3,4,128]` 由入口转换 |
| Conv 状态池 | `conv_state.shape=[C,L,3*P]`，`C>=1`、`L>=10`；槽 0 为空槽，对应 Conv 输出全零；同一调用中非零 Conv 索引互不重复 |
| Conv 历史 | 起点 `offset=max(conv_num_accepted_tokens[b]-1,0)`，需满足 `offset+3<=L`；未选槽与未写行保持逐元素不变 |
| recurrent 状态池 | `recurrent_state.shape=[R,N,128,128]`，`R>=B*8`；槽 0 有效；初始 state 索引为 `ssm_state_indices[b*8+max(num_accepted_tokens[b]-1,0)]`，偏移需在 `[0,8)` 内 |
| 索引与 batch 互斥 | 索引取值范围分别是 Conv `[0,C)`、recurrent `[0,R)`，由调用方保证有效；不同 batch 的 recurrent 槽集合不能相交；8 个 token 全部计算，accepted 计数不截断输出 |
| 配额 | 当前 stream 的 AIC 配额截断到 32 后必须属于 `{8,16,24,28,32}`，AIV 配额不少于 AIC 的两倍 |
| 副作用 | 仅原地更新 `conv_state` 与 `recurrent_state`；每次调用独立分配 workspace，共享 workspace 的调用需调用方串行化；不包含跨卡通信 |

## 使用示例

以下命令在 CATLASS 仓库根目录执行：

```bash
PYTHONPATH=examples/python/tla_dsl/experimental/kernel/attention:$PYTHONPATH python - <<'PY'
import torch
import torch_npu
from mega_recurrent_kda.mega_recurrent_kda import mega_recurrent_kda

dev, TP, mb = "npu:0", 16, 1
B = mb * TP; N = 96 // TP; D, H, S, P = 128, 7168, 8, N * 128
C, L, R = B + 1, 16, B * S
bf16 = dict(dtype=torch.bfloat16, device=dev)
f32 = dict(dtype=torch.float32, device=dev)
i32 = dict(dtype=torch.int32, device=dev)


def nz(*shape):  # 初始化期转换为 FRACTAL_NZ
    return torch_npu.npu_format_cast(torch.randn(shape, **bf16), 29)


out = mega_recurrent_kda(
    torch.randn(B, S, H, **bf16),  # hidden_states
    nz(3 * P, H), nz(128, H), nz(P, 128), nz(N, H), nz(P, H),
    torch.randn(D, **f32),  # output_norm_weight
    nz(H, P),  # output_projection_weight
    torch.randn(3, N, 4, D, **bf16),  # conv1d_weight
    torch.zeros(C, L, 3 * P, **bf16),  # conv_state，原地更新
    torch.zeros(R, N, D, D, **f32),  # recurrent_state，原地更新
    torch.arange(1, B + 1, **i32),  # conv_state_indices
    torch.arange(B * S, **i32),  # ssm_state_indices
    torch.ones(B, **i32),  # conv_num_accepted_tokens
    torch.ones(B, **i32),  # num_accepted_tokens
    torch.zeros(N, **f32),  # A_log
    torch.zeros(N, D, **f32),  # dt_bias
    scale=128**-0.5, lower_bound=-5.0, rms_norm_eps=1e-5,
)

assert out.shape == (B, S, H) and out.dtype == torch.bfloat16
PY
```
