# StreamK MMAD 端到端示例

本目录下的样例演示 **CATLASS DSL** 下 StreamK MatMul 的实现。

## 功能说明

功能和基础矩阵乘一致，实现形如$(m, k)$和$(k, n)$的两矩阵的乘法运算，输出形如$(m, n)$，计算公式为：
$$
\begin{aligned}
C &= A \times B \\
C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
\end{aligned}
$$

StreamK 通过将 K 维度的计算分摊到多个核上以均衡负载：normal tile 直接完成全 K 计算并写回 GM C；尾轮 tile 由多个 AIC 分担 K 切片计算，结果写入 workspace，再由配对的 AIV 做 ReduceAdd 归约写回 GM C。


## 代码组织

```plain
./basic_mmad_streamk
├── basic_mmad_streamk.py
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`basic_mmad_streamk.py`**](basic_mmad_streamk.py) | 设备侧 `@tla.kernel` 与 host 侧运行/校验逻辑同文件。其中 host 侧可配置 GM 布局与元素类型，多 block、K 维分块、L1/L0 双缓冲与 StreamK workspace；默认用 torch + torch_npu 上板并校验精度。问题规模（m/n/k）由 CLI 参数动态指定，dtype、L1/L0 分块等编译期常量直接在源码中定义。 |

## 约束说明

 - 左、右矩阵及结果矩阵所支持的数据组合类型如下。

| `DTYPE_A` | `DTYPE_B` | `DTYPE_C` |
|---------|---------|------------------|
| f16 | f16 | f32 |
| f16 | f16 | f16 |
| bf16 | bf16 | f32 |
| bf16 | bf16 | bf16 |
| f32 | f32 | f32 |


## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../docs/zh/quick_start.md)完成部署。

### 命令行参数

矩阵尺寸、dtype 与 CLI 默认值以源码与 **`--help`** 为准。主要参数如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--m` | `256` | 矩阵乘左矩阵 A 的行数 |
| `--n` | `256` | 矩阵乘右矩阵 B 的列数 |
| `--k` | `512` | 矩阵乘累加轴的大小 |
| `--layout-a` / `--layout-b` | `"row"` / `"row"` | 左、右矩阵 A、B 的数据排布格式，可选 `"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--dtype-a` / `--dtype-b` / `--dtype-c` | `"f16"` / `"f16"` / `"f32"` | 左、右矩阵 A、B 和结果矩阵 C 的数据类型，可选范围参考约束说明。 |
| `--block-num` | `-1` | 启用的 AIC 核数，`-1` 表示自动探测可用核数（满核）。 |
| `--atol` | `1e-3` | 精度比对时的绝对容差。 |

### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# 上板并校验精度
python examples/end_to_end/basic_mmad_streamk/basic_mmad_streamk.py --device 0 \
  --layout-a row --layout-b col \
  --dtype-a f16 --dtype-b f16 --dtype-c f32
```

默认测试条件下，预期输出：

```text
--- mnk=(256,256,512) layout=row/col dtype=f16/f16/f32 ---
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

其中 `passed`结果为`True`或`False` 表明 NPU 计算结果与golden参考值精度校验是否通过；`cache_dir` 是指定的缓存目录， `cache_key` 是编译缓存的哈希值。
