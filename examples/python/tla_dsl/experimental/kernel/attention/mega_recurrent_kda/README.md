# Mega Recurrent KDA 前向样例

本样例使用纯 `catlass.tla`，在一次 C1V2 mixed AIC/AIV kernel 中完成 QKV、decay A/B、beta、gate 投影，Conv1D + SiLU，Q/K 归一化、recurrent KDA、RMSNorm、sigmoid gating 和输出投影。标准 Conv 权重布局支持该单 kernel 路径。

当前实现为 C25，包含静态检查修复。已测 48 个性能点中，9 点达到至少 1.0 倍，39 点未达标，等权算术平均加速比为 0.948971 倍；这是尚未通过性能目标的实验性样例。逐 case 结果保留在 [PR #1725](https://gitcode.com/cann/catlass/merge_requests/1725)，优化机制与适用边界见 [优化方案](docs/design.md#优化方案)。

## 支持范围

- `H=7168`、`S=8`、`D=128`、`TP=4/8/16`、`N=96/TP`，模型 batch 为 1～32，输入 `B=model_batch*TP`。
- hidden 为 BF16 / ND；六个投影权重为 BF16 / FRACTAL_NZ allocation root；recurrent state 为 FP32，Conv state 为 BF16。
- 标准 Conv 权重为 `[3,N,4,128]`。兼容 `[4,3*N*128]`、`[3,4,128]` 布局需要入口转换，严格单 kernel 路径采用标准布局。
- 已验证 AIC 配额为 8/16/24/28，AIV 配额至少为两倍。32 AIC 保留接口支持，但未实测。
- 每次调用写出八个 FP32 state snapshot；索引有效性、状态槽互斥和输入布局约束见 [接口说明](docs/design.md#接口契约)。算子不包含跨卡通信。

## 代码组织

| 路径 | 作用 |
| --- | --- |
| `mega_recurrent_kda.py`、`__init__.py` | 公开入口、包初始化、输入校验、workspace、stream 配额与路由 |
| `kernel/mega_recurrent_kda.py` | 全部 AIC/AIV 计算与调度：投影、Conv、recurrent、snapshot、归一化、gating、输出权重常驻和融合入口 |
| [接口、设计与优化](docs/design.md) | 20 参数契约、阶段分工、内存与同步约束，以及 C25 优化方案 |

## 环境

实测环境为 Ascend 950PR_957c、CANN toolkit 9.2.0、Python 3.12、torch / torch_npu 2.13.0、`ascend-catlass-dsl==2.2.1.dev20261112`。设备核数通过运行时接口与 `npu-smi` 核对为 28 AIC / 56 AIV。该 wheel 对应 CATLASS 提交 `b62ee850362b4ff649c1e07984ce767d7d88f2b3`，来自 [CATLASS DSL wheel 索引](https://ascend-catlass.github.io/simple/ascend-catlass-dsl/)。

初始化已安装的 CANN toolkit 和 Python 环境后，安装实测版本并准备编译依赖：

```bash
python -m pip install --pre --index-url https://ascend-catlass.github.io/simple/ ascend-catlass-dsl==2.2.1.dev20261112
python -m catlass.bc_compile
```

## 使用示例

将 `mega_recurrent_kda` 目录的父目录加入 `PYTHONPATH`。在 CATLASS 仓库根目录运行时可设置：

```bash
export PYTHONPATH="$PWD/examples/python/tla_dsl/experimental/kernel/attention:${PYTHONPATH:-}"
```

调用方按接口说明准备 NPU Tensor，并在当前 stream 上设置支持的 AIC/AIV 配额，再调用公开入口：

```python
from mega_recurrent_kda.mega_recurrent_kda import mega_recurrent_kda

output = mega_recurrent_kda(
    hidden_states, qkv_projection_weight, decay_projection_a_weight,
    decay_projection_b_weight, beta_projection_weight,
    output_gate_projection_weight, output_norm_weight,
    output_projection_weight, conv1d_weight, conv_state, recurrent_state,
    conv_state_indices, ssm_state_indices, conv_num_accepted_tokens,
    num_accepted_tokens, A_log, dt_bias, scale, lower_bound, rms_norm_eps=1e-5)
```

返回连续 BF16 `[B,8,7168]`，原地更新两个状态池。NZ 权重在模型初始化时完成转换，传入 `.detach()` 后的原分配 Tensor。每次调用独立分配 workspace；第一次调用包含编译。算子运行依赖 CATLASS DSL、torch 与 torch_npu，不依赖测试脚本、CPU reference 或 cannbot-dsl 基线包。
