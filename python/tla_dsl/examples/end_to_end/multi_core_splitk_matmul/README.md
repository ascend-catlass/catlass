# 多核切 K Matmul 端到端示例

本目录下提供的样例演示 **CATLASS DSL** 下多核切 K 矩阵乘的两种实现。

## 功能说明

功能和基础矩阵乘一致，实现形如$(m, k)$和$(k, n)$的两矩阵的乘法运算，输出形如$(m, n)$，计算公式为：
$$
\begin{aligned}
C &= A \times B \\
C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
\end{aligned}
$$

Split-K 将 K 维度的计算切分到多个核上并行执行，各核的计算结果写回 GM，再由 AIV 做 ReduceAdd 归约计算。

## 代码组织

```plain
./multi_core_splitk_matmul
├── multi_core_splitk_matmul.py
├── tail_multi_core_splitk_matmul.py
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`multi_core_splitk_matmul.py`**](multi_core_splitk_matmul.py) | 全部 Tile 块在累加轴 K方向上做切分，并在 AIV 核进行归约（ReduceAdd）写回 GM。 |
| [**`tail_multi_core_splitk_matmul.py`**](tail_multi_core_splitk_matmul.py) | 针对尾轮采取上述多核切K优化，以达成负载均衡。 |

## 约束说明

 - 左、右矩阵及结果矩阵所支持的数据组合类型如下。

| `DTYPE_A` | `DTYPE_B` | `DTYPE_C` |
|---------|---------|------------------|
| f16 | f16 | f16 |
| bf16 | bf16 | bf16 |
| f32 | f32 | f32 |


## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../docs/zh/quick_start.md)完成部署。

### 命令行参数

接收命令行参数如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--m` | `256` | 矩阵乘左矩阵 A 的行数 |
| `--n` | `512` | 矩阵乘右矩阵 B 的列数 |
| `--k` | `1024` | 矩阵乘累加轴的大小 |
| `--layout-a` / `--layout-b` | `"row"` / `"row"` | 左、右矩阵 A、B 的数据排布格式，可选 `"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--dtype-a` / `--dtype-b` / `--dtype-c` | `"f16"` / `"f16"` / `"f16"` | 左、右矩阵 A、B 和结果矩阵 C 的数据类型，可选范围参考约束说明。 |
| `--block-num` | `-1` | 启用的 AIC 核数，`-1` 表示自动探测可用核数（满核）。 |

### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# multi_core_splitk （指定 NPU 设备ID, m/n/k 的值）
python examples/end_to_end/multi_core_splitk_matmul/multi_core_splitk_matmul.py \
  --device 0 --m 256 --n 512 --k 1024

# tail_multi_core_splitk （指定 NPU 设备ID, m/n/k 的值）
python examples/end_to_end/multi_core_splitk_matmul/tail_multi_core_splitk_matmul.py \
  --device 0 --m 2048 --n 1024 --k 2048
```

默认测试条件下，预期输出：

```text
--- mnk=(256,512,1024) layout=row/row dtype=f16/f16/f16 ---
passed=True mismatch=0.0000% (budget=0.1000%) cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

其中 `passed`结果为`True`或`False` 表明 NPU 计算结果与golden参考值精度校验是否通过；`mismatch`为超容差元素占比，不高于 `budget` 即判定通过；`cache_dir` 是指定的缓存目录， `cache_key` 是编译缓存的哈希值。
