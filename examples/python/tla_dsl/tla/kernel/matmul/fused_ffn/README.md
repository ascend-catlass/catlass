# Fused FFN 端到端示例

本目录下的样例演示 **CATLASS Python** 下融合 FFN（Feed-Forward Network）的实现。

## 功能说明

- 算子功能：在一次 kernel launch 内完成 **Up Matmul → Activation → SyncAll → Down Matmul**（MIX：AIC Cube + AIV Vector），覆盖常见 Linear FFN 流水形态。
- 计算公式（以 SiLU 为例）：

$$
\begin{aligned}
H &= \mathrm{SiLU}(X W_1^\mathsf{T}) \\
Y &= H W_2^\mathsf{T}
\end{aligned}
$$

  GELU 将 SiLU 替换为 erf 定义的 GELU；SwiGLU 使用 \(W_1\) 形如 \((2H, K)\)（gate|up），隐藏态为 \(\mathrm{SiLU}(\mathrm{gate}) \cdot \mathrm{up}\)。
- 支持产品型号：Ascend 950PR&950DT系列产品。

### 激活与通路

| `--activation` | Up 通路 | 说明 |
| --- | --- | --- |
| `silu` | `l0c2ub` | L0C→UB FixPipe + VF SiLU |
| `gelu` | `l0c2ub` | L0C→UB + VF erf-GELU |
| `swiglu` | `l0c2ub` | 单路 interleaved pack（`SWIGLU_SINGLE`）；容量不足时回退 dual（dual 为 acc+gate 两块 FP32 compact 槽，host 按 `acc_slots=2` 预留 out） |

Up 产品路径固定为 L0C→UB。Down 侧按 shape / tiling 自动选择 Basic、StreamK、A-full 或 B-full（fullLoad 与 StreamK 互斥）。

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `X` | Input | `(M, K)` | `float16/bfloat16` | 激活输入，layout 为 `RowMajor` |
| `W1` | Input | `(H, K)` / SwiGLU `(2H, K)` | `float16/bfloat16` | Up 权重；`linear` 为 `(H,K)`（或 `2H,K`），`canonical` 为 `(K,H)`（SwiGLU 仅 `linear`） |
| `W2` | Input | `(N, H)` | `float16/bfloat16` | Down 权重；`linear` 为 `(N,H)`，`canonical` 为 `(H,N)` |
| `bias1` | Input（可选） | `(H,)` / SwiGLU `(2H,)` | `float32` | Up Cube BT bias；与 `bias2` 成对启用 |
| `bias2` | Input（可选） | `(N,)` | `float32` | Down Cube BT bias；与 `bias1` 成对启用 |
| `Y` | Output | `(M, N)` | `float16/bfloat16` | FFN 输出，dtype 与 `X` 一致 |

中间累加（L0C / epi）恒为 FP32。

## 使用范围说明

推荐范围：

- `M` / `K` / `H` / `N` 均为正整数；`M ≤ 8192`，`K` / `H` / `N` ≤ `65536`；且 `M·K`、`M·H`、`M·N`、`H·K`、`N·H` 均不超过 `67108864`。SwiGLU 时 Up 宽为 `2H`，对应 `M·(2H)`、`(2H)·K` 同受该积上限约束。
- Up 为 `(M, K) × (K, H)`（或线性布局下等价的 `(H, K)` 权重），Down 为 `(M, H) × (H, N)`。
- 推荐 `H`、`N` 对 16 对齐，以减少 Cube / Vector 尾块分支；未对齐 shape 仍可运行，由 host tiling 自动裁剪。
- StreamK / A-full / B-full 由 host 按 shape 自动选择，无需手动指定。

## 代码组织

```plain
./fused_ffn
├── fused_ffn.py          # host 入口（编译 / 启动 / CPU golden）
├── golden.py             # CPU 参考（torch；默认设备同构）
├── swiglu_pack.py        # SwiGLU epi SIMD lane 选择
├── README.md
└── ffn/                  # kernel 包
    ├── config.py
    ├── tiling.py
    ├── tiling_aswt.py
    ├── buffers.py
    ├── sync_hard.py
    ├── kernels/          # MIX kernel 入口
    ├── policies/         # act / bias / swiglu 策略
    └── stages/           # Up / epi / Down / StreamK stages
```

| 文件 | 概述 |
| --- | --- |
| [**`fused_ffn.py`**](fused_ffn.py) | Host 编译 / 启动 / 冒烟精度校验入口。 |
| [**`golden.py`**](golden.py) | CPU 参考：默认 `device_iso=True`（Hidden 先物化为输出 dtype 后再 Down）。 |
| [**`ffn/`**](ffn/) | MIX kernel、tiling、stages 与 SoftFlag / SyncAll 封装。 |

Host 侧 TLA tensor / UB slot 辅助在 `host/tensor_utils.py`。

## 约束说明

- 不接受 `x=f32`；输入 / 权重 / 输出 dtype 与 bias / 中间累加约定见「样例参数说明」。
- 权重布局：`linear` / `canonical`；**SwiGLU 仅支持 `linear`**。
- Bias：Up/Down **仅首 K** 注入；StreamK **仅 K-slice0**。SwiGLU single 的 `bias1=[gate|up]` 与 W1 同款 tile interleave pack；dual 则半宽分两次 Up。Quant 路径仍未启用。
- Up 通路固定 L0C→UB；Host tiling 固定走 ASWT；有 bias 时 ASWT `has_bias` 压 N（`btSize/DB/fp32`）。
- SoftFlag：WithoutQue HardEvent；StreamK 在 SyncAll 后复用同一套 ID，避免 bias+SK 耗尽事件号。

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)完成部署。

### 命令行参数

```text
fused_ffn.py [-h] [--device DEVICE] [--m M] [--k K] [--h H] [--n N]
             [--dtype {f16,bf16}] [--activation {silu,gelu,swiglu}]
             [--layout {linear,canonical}] [--bias] [--block-num BLOCK_NUM]
             [--skip-golden]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号 |
| `--m` / `--k` / `--h` / `--n` | `128` / `256` / `512` / `256` | FFN shape：`M,K,H,N` |
| `--dtype` | `"f16"` | 输入/输出数据类型（见约束说明） |
| `--activation` | `"silu"` | `silu` / `gelu` / `swiglu` |
| `--layout` | `"linear"` | 权重布局；SwiGLU 强制 `linear` |
| `--bias` | off | 启用 Cube BT 成对 bias（fp32 `bias1`/`bias2`） |
| `--block-num` | `-1` | AIC 核数；`-1` 表示自动选择 |
| `--skip-golden` | off | 跳过 CPU 参考与精度校验（仅启动冒烟） |

### 执行示例

在仓库根目录下执行：

```bash
cd /path/to/catlass
# SiLU 默认 shape
python examples/python/tla_dsl/tla/kernel/matmul/fused_ffn/fused_ffn.py --device 0

# GELU / SwiGLU
python examples/python/tla_dsl/tla/kernel/matmul/fused_ffn/fused_ffn.py --device 0 --activation gelu
python examples/python/tla_dsl/tla/kernel/matmul/fused_ffn/fused_ffn.py --device 0 --activation swiglu \
  --m 128 --k 256 --h 512 --n 256

# Cube BT bias（SiLU）
python examples/python/tla_dsl/tla/kernel/matmul/fused_ffn/fused_ffn.py --device 0 --bias
python examples/python/tla_dsl/tla/kernel/matmul/fused_ffn/fused_ffn.py --device 0 --activation gelu --bias
python examples/python/tla_dsl/tla/kernel/matmul/fused_ffn/fused_ffn.py --device 0 --activation swiglu --bias \
  --m 128 --k 256 --h 512 --n 256
```

默认测试条件下，预期输出类似：

```text
shape=(128,256,512,256) dtype=f16 act=silu layout=linear swiglu=- bias=none up_path=1 down_mode=basic \
block_num=... tiling_up=(...) tiling_down=(...)
passed=True matched_ratio=1.000000
```

其中 `passed` 结果为 `True` 或 `False`，表明 NPU 计算结果与设备同构 golden 的精度校验是否通过（`mixed_tolerance_bm_v2`：`matched_ratio ≥ 0.99`，且 abs/ULP 硬门禁）。
