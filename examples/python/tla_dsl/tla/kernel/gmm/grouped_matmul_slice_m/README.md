# Grouped Matmul Slice-M 端到端示例

本目录下的样例演示 **CATLASS Python** 下按 M 切分的分组矩阵乘。

## 功能说明

实现对M轴切分的分组矩阵乘，其中 A 为 `(M, K)`，按 M 轴切分；B 为 `(G, K, N)` ；C 为 `(M, N)`。`group_list` 为长度 `G+1` 的 Int32 累和式列表（起始为 0）。数学公式如下：

$$
C_g = A_g \times B_g,\quad g = 0..G-1
$$

## 代码组织

```plain
./grouped_matmul_slice_m
├── grouped_matmul_slice_m.py
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`grouped_matmul_slice_m.py`**](grouped_matmul_slice_m.py) | 设备侧 `@tla.kernel` 与 host 侧运行/校验逻辑同文件。执行数据生成、compile/launch、以及golden比对。 |

## 约束说明

 - 左、右矩阵及结果矩阵所支持的数据组合类型如下。

| `DTYPE_A` | `DTYPE_B` | `DTYPE_C` |
|---------|---------|------------------|
| f16 | f16 | f32 |
| f16 | f16 | f16 |
| bf16 | bf16 | f32 |
| bf16 | bf16 | bf16 |
| f32 | f32 | f32 |

- 分组数 `G` 不超过 `min(M, 256)`，且 `group_list` 长度固定为 `G+1`。
- 为避免误差累加和计算超时，建议矩阵的 M/N/K 轴不超过 122880，且 GM 总分配不超过 10 GiB（即 `M*K*sizeof(DTYPE_A) + G*K*N*sizeof(DTYPE_B) + M*N*sizeof(DTYPE_C) <= 10 GiB`）。

## 使用示例

要运行本路径下的样例，请参考[快速开始](../../../../../../../python/tla_dsl/docs/zh/quick_start.md)完成部署。

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--m` | `1024` | 矩阵乘左矩阵 A 的行数 |
| `--n` | `256` | 矩阵乘右矩阵 B 的列数 |
| `--k` | `256` | 矩阵乘累加轴的大小 |
| `--groups` | `4` | 分组数 G。 |
| `--group-mode` | `random` | 组切分模式，可选 `average` 或 `random`。 |
| `--layout-a` / `--layout-b` | `"row"` / `"row"` | 左、右矩阵 A、B 的数据排布格式，可选 `"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--dtype-a` / `--dtype-b` / `--dtype-c` | `"f16"` / `"f16"` / `"f16"` | 左、右矩阵 A、B 和结果矩阵 C 的数据类型，可选范围参考约束说明。 |
| `--block-num` | `-1` | 启用的核数，`-1` 表示自动探测可用核数（满核）。 |

### 执行示例

在仓库根目录下执行：

```bash
cd /path/to/catlass
python examples/python/tla_dsl/tla/kernel/gmm/grouped_matmul_slice_m/grouped_matmul_slice_m.py --device 0 \
  --m 1024 --n 256 --k 256 --groups 4 --group-mode average
```

默认测试条件下，预期输出：

```text
--- groups=(4) mnk=(1024,256,256) layout=row/row dtype=f16/f16/f16 group_mode=average ---
GROUP_CURRENT_M=(256, 256, 256, 256)
GROUP_LIST_PREFIX=(0, 256, 512, 768, 1024)
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

其中 `passed`结果为`True`或`False` 表明 NPU 计算结果与golden参考值精度校验是否通过；`cache_dir` 是指定的缓存目录， `cache_key` 是编译缓存的哈希值。
