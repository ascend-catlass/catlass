# Batch Invariant Matmul 端到端示例

本目录下的样例演示 **CATLASS Python** 下满足 batch 一致性的矩阵乘，计算语义对齐 `torch.matmul`。

## 功能说明

- 算子功能：完成矩阵乘计算。输入秩为 1～6，对最后两维做矩阵乘，并对前面的 batch 维按 `torch.matmul` 规则广播。
- 计算公式：

  $$
  \begin{aligned}
  C &= A \times B \\
  C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
  \end{aligned}
  $$

  其中 $A$、$B$ 的最后两维分别为 `(m, k)`、`(k, n)`，$C$ 的最后两维为 `(m, n)`。一维左操作数在前面补长度为 1 的维，一维右操作数在后面补长度为 1 的维，写回前去掉补上的维。

- 支持产品型号：Ascend 950PR/950DT 系列产品

### batch 一致性

同一组向量无论单独计算，还是放进更大的 $M$、$N$ 或 batch，对应切片的候选结果逐位相同。实现上靠下面几条约束：

- L1/L0 切分只由单个 GEMM 的 $M$、$N$、$K$、dtype 和 Cube 核数决定，batch 计数不进入。
- 每个输出 tile 由一颗 Cube 核沿完整 K 归约，不把 K 分给多核，也不使用原子加。
- 先按 batch 切开 `(m, k)` 与 `(k, n)`，再切 M/N tile。$m$ 不能被 L1 M tile 整除时，尾块停在当前 batch 边界。
- 不使用 HF32。

与 `torch.matmul` 的数值对齐使用 `helpers.compare` 的矩阵乘累加容差。batch 一致性是两次候选启动对应切片的逐位比较，不使用该容差。

## 代码组织

```text
./matmul_batchinvariant
├── matmul_batchinvariant.py
└── README.md
```

| 文件 | 概述 |
| --- | --- |
| [`matmul_batchinvariant.py`](matmul_batchinvariant.py) | 设备侧 `@tla.kernel` 与 host 侧 `run(a, b)`、命令行校验同文件。host 完成 dtype/K/秩校验、batch 广播与空输入返回；kernel 按 `(batch, M tile, N tile)` 分核，L1/L0 双缓冲，核内用 `tla.mutex` 同步。 |

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `a` | Input | 1～6 维 | `float16` / `bfloat16` / `float32` | 左矩阵或左向量，逻辑行主序。非连续输入先 `contiguous()`。 |
| `b` | Input | 与 `a` 的 K 对齐，1～6 维 | 与 `a` 相同 | 右矩阵或右向量，逻辑行主序。 |
| `output` | Output | 与 `torch.matmul` 相同 | 与输入相同 | 矩阵乘结果，行主序。 |

Python 入口为 `run(a, b)`。命令行样例覆盖三维 batch 矩阵乘。

## 约束说明

- 左、右矩阵及结果矩阵所支持的数据组合类型如下。

| `DTYPE_A` | `DTYPE_B` | `DTYPE_C` |
| --- | --- | --- |
| f16 | f16 | f16 |
| bf16 | bf16 | bf16 |
| f32 | f32 | f32 |

- 输入秩在 1～6，K 维相等，batch 维可按 `torch.matmul` 广播。
- 不支持混合 dtype、整数、FP8、HF32、转置标志、bias 与后处理。
- $M$、$N$、$K$ 或 batch 为 0 时由 host 直接返回，不启动 kernel：$M$ 或 $N$ 为 0 返回空张量，$K$ 为 0 返回全 0，batch 为 0 返回 batch 维为 0 的空结果。
- L1/L0 tile 按 DAV_3510 AswTiling 选择，输入是单个 GEMM 的 M、N、K、dtype 和 Cube 核数，batch 计数不进入。

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)完成部署。

### 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--batch` | `4` | batch 数。 |
| `--m` | `256` | 矩阵乘左矩阵 A 的行数。 |
| `--n` | `512` | 矩阵乘右矩阵 B 的列数。 |
| `--k` | `1024` | 矩阵乘累加轴的大小。 |
| `--dtype` | `"f16"` | 左、右矩阵与结果的数据类型，可选 `f16`、`bf16`、`f32`。 |

命令行固定使用行主序输入，并与 `torch.matmul` 比较。

### 执行示例

在仓库根目录下执行：

```bash
cd /path/to/catlass
python examples/python/tla_dsl/tla/kernel/matmul/matmul_batchinvariant/matmul_batchinvariant.py \
  --device 0 --batch 4 --m 256 --n 512 --k 1024
```

默认测试条件下，预期输出：

```text
--- batch=(4) mnk=(256,512,1024) dtype=f16 ---
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

其中 `passed` 结果为 `True` 或 `False`，表明 NPU 计算结果与golden参考值精度校验是否通过；`cache_dir` 是指定的缓存目录，`cache_key` 是编译缓存的哈希值。
