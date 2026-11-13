# CV (Cube + Vector) 融合后处理类 端到端示例

本目录下提供的系列样例演示 **CATLASS Python** 下系列后处理类 CV（Cube + Vector）融合算子的计算过程。

## 功能说明

- 算子功能：实现基础矩阵乘，并融合后续向量后处理计算。
- 计算公式：

  $$
  \begin{aligned}
  D &= f(A \times B)
  \end{aligned}
  $$

  其中 $A$ 和 $B$ 分别是形如 `(m,k)`、`(k,n)` 的输入矩阵，$A \times B$ 是形如 `(m,n)` 的矩阵乘中间结果，$D$ 是形如 `(m,n)` 的最终输出，$f$ 对应后处理操作。
- 支持产品型号：Ascend 950PR/950DT 系列产品

本目录下样例包含的后处理操作如下：

| 文件 | 计算公式 |
| --- | --- |
| `matmul_add.py` | D = A×B + X |
| `matmul_add_ub.py` | D = A×B + X |
| `matmul_bias.py` | D = A×B + bias |
| `matmul_gelu.py` | D = Gelu(A×B)，tanh 近似 |
| `matmul_leaky_relu.py` | D = LeakyRelu(A×B) |
| `matmul_sigmoid.py` | D = Sigmoid(A×B) |
| `matmul_silu.py` | D = Silu(A×B) |
| `matmul_tanh.py` | D = Tanh(A×B) |

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float16/bfloat16/float32` | 左矩阵，layout 支持 `RowMajor`（默认）和 `ColumnMajor` |
| `B` | Input | `(k,n)` | `float16/bfloat16/float32` | 右矩阵，layout 支持 `RowMajor`（默认）和 `ColumnMajor`，数据类型与左矩阵一致 |
| `X` | Input | `(m,n)` | `float16/float32` | `matmul_add.py` / `matmul_add_ub.py` 的加法输入，layout 为 `RowMajor`，数据类型与输出 `D` 一致 |
| `bias` | Input | `(1,n)` | `float16/float32` | `matmul_bias.py` 的偏置，沿 m 轴广播，数据类型与输出 `D` 一致 |
| `alpha` | 常量 | 标量 | `float32` | `matmul_leaky_relu.py` 的负半轴斜率，默认 `0.1` |
| `kScale` / `kCubicCoef` | 常量 | 标量 | `float32` | `matmul_gelu.py` 的 GELU 系数，取 `1.59576912` 与 `0.044715`，与 `examples/common/golden/matmul.hpp` 中 `Gelu` 一致 |
| `D` | Output | `(m,n)` | `float16/float32` | 融合计算结果，layout 仅支持 `RowMajor` |

## 使用范围说明

推荐场景：

- 矩阵的 M/N/K 轴不超过 122880。
- `A`/`B` 均为 `float32` 类型时，建议 `K` 小于 3072 以避免绝对误差超上限。

## 代码组织

```plain
./basic_mmad_epilogue
├── matmul_add.py
├── matmul_add_ub.py
├── matmul_bias.py
├── matmul_gelu.py
├── matmul_leaky_relu.py
├── matmul_sigmoid.py
├── matmul_silu.py
├── matmul_tanh.py
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`matmul_add.py`**](matmul_add.py) | 实现 `D = A@B + X` 计算功能。 |
| [**`matmul_add_ub.py`**](matmul_add_ub.py) | 实现 `D = A@B + X` 计算功能，启用 L0C -> UB 通路将矩阵乘结果搬出到 UB。 |
| [**`matmul_bias.py`**](matmul_bias.py) | 实现 `D = A@B + bias` 计算功能，其中 `bias` 为 `(1,n)` 向量，沿 m 轴广播。 |
| [**`matmul_gelu.py`**](matmul_gelu.py) | 实现 `D = Gelu(A@B)` 计算功能，公式为 `x / (1 + exp(-1.59576912 * (x + 0.044715 * x^3)))`。 |
| [**`matmul_leaky_relu.py`**](matmul_leaky_relu.py) | 实现 `D = LeakyRelu(A@B)` 计算功能，其中 `α` 默认为 `0.1`。 |
| [**`matmul_sigmoid.py`**](matmul_sigmoid.py) | 实现 `D = Sigmoid(A@B)` 计算功能。 |
| [**`matmul_silu.py`**](matmul_silu.py) | 实现 `D = Silu(A@B)` 计算功能。 |
| [**`matmul_tanh.py`**](matmul_tanh.py) | 实现 `D = Tanh(A@B)` 计算功能。 |

## 约束说明

- 左、右矩阵及结果矩阵所支持的数据组合类型如下。

| 算子 | `DTYPE_A` / `DTYPE_B` | `DTYPE_C` |
|------|-------------------|---------|
| add, bias, gelu, leaky_relu, sigmoid, silu | f16 / bf16 / f32 | f16 或 f32 |
| add_ub, tanh | f16 / bf16 / f32 | f32 |

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)完成部署。

### 命令行参数

主要参数如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--m` | `256` | 矩阵乘左矩阵 A 的行数 |
| `--n` | `256` | 矩阵乘右矩阵 B 的列数 |
| `--k` | `256` | 矩阵乘累加轴的大小 |
| `--layout-a` / `--layout-b` | `"row"` / `"row"` | 左、右矩阵 A、B 的数据排布格式，可选 `"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--dtype-a` / `--dtype-b` | `"f32"` / `"f32"` | 左、右矩阵 A、B 的数据类型，可选范围参考约束说明。 |
| `--dtype-c` | `"f32"` | 结果矩阵 D 的数据类型，可选范围参考约束说明（`add_ub` / `tanh` 仅支持 f32）。 |
| `--block-num` | `-1` | 启用的 AIC 核数，`-1` 表示自动探测可用核数（满核）。 |

### 执行示例

在仓库根目录下执行：

```bash
cd /path/to/catlass
# 其余变体替换文件名即可
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad_epilogue/matmul_add.py --device 0 \
  --m 256 --n 256 --k 256 \
  --layout-a row --layout-b row \
  --dtype-a f16 --dtype-b f16 --dtype-c f32
```

默认测试条件下，预期输出：

```text
--- mnk=(256,256,256) layout=row/row dtype=f16/f16/f32 ---
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

其中 `passed` 结果为 `True` 或 `False`，表明 NPU 计算结果与 golden 参考值精度校验是否通过；`cache_dir` 是指定的缓存目录， `cache_key` 是编译缓存的哈希值。
