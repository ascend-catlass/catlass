# Batched Matmul 端到端示例

本目录下的样例演示 **CATLASS Python** 下的批量矩阵乘。

## 功能说明

对每个 batch 计算

$$
C_b = A_b @ B_b,\quad b = 0,\ldots,B-1
$$

各 batch 的 `(M, N, K)` 相同。

## 代码组织

```plain
./batched_matmul
├── batched_matmul.py
└── README.md
```

| 文件                                                | 概述                                                                                                |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| [**`batched_matmul.py`**](batched_matmul.py) | 设备侧`@tla.kernel` 与 host 侧运行/校验逻辑同文件。执行数据生成、compile/launch、以及golden比对。 |

## 约束说明

- 左、右矩阵及结果矩阵所支持的数据组合类型如下。

| `DTYPE_A` | `DTYPE_B` | `DTYPE_C` |
| ----------- | ----------- | ----------- |
| f16         | f16         | f32         |
| f16         | f16         | f16         |
| bf16        | bf16        | f32         |
| bf16        | bf16        | bf16        |
| f32         | f32         | f32         |

- 为避免误差累加和计算超时，建议矩阵的 M/N/K 轴不超过 122880，且 GM 总分配不超过 10 GiB（即 `B*(M*K*sizeof(DTYPE_A) + K*N*sizeof(DTYPE_B) + M*N*sizeof(DTYPE_C)) <= 10 GiB`）

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)完成部署。

### 命令行参数

| 参数                                          | 默认值                            | 说明                                                                                 |
| --------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------ |
| `--device`                                  | `0`                             | 上板执行使用的 NPU 设备号。                                                          |
| `--batch`                                   | `5`                             | batch 数 B。                                                                         |
| `--m`                                       | `256`                           | 矩阵乘左矩阵 A 的行数                                                                |
| `--n`                                       | `512`                           | 矩阵乘右矩阵 B 的列数                                                                |
| `--k`                                       | `1024`                          | 矩阵乘累加轴的大小                                                                   |
| `--layout-a` / `--layout-b`               | `"row"` / `"row"`             | 左、右矩阵 A、B 的数据排布格式，可选`"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--dtype-a` / `--dtype-b` / `--dtype-c` | `"f16"` / `"f16"` / `"f32"` | 左、右矩阵 A、B 和结果矩阵 C 的数据类型，可选范围参考约束说明。                      |
| `--block-num`                               | `-1`                            | 启用的核数，`-1` 表示自动探测可用核数（满核）。                                    |

### 执行示例

在仓库根目录下执行：

```bash
cd /path/to/catlass
python examples/python/tla_dsl/tla/kernel/matmul/batched_matmul/batched_matmul.py --device 0 \
  --batch 4 --m 256 --n 256 --k 256
```

默认测试条件下，预期输出：

```text
--- batch=(4) mnk=(256,256,256) layout=row/row dtype=f16/f16/f32 ---
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

其中 `passed`结果为`True`或`False` 表明 NPU 计算结果与golden参考值精度校验是否通过；`cache_dir` 是指定的缓存目录， `cache_key` 是编译缓存的哈希值。
