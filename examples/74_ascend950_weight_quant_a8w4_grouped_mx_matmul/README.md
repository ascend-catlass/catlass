# WeightQuantA8W4GroupedMxMatmul Example Readme

## 功能说明

- 算子功能：完成 A8W4 权重量化的分组 MX 矩阵乘，把左矩阵 `A` 沿 M 方向按 `GroupList` 分为 `G` 组，第 `g` 组 `A_g` 与第 `g` 个 FP4 权重 `B` 经各自 MX 缩放因子反量化后在 Cube 上完成乘加，各组结果按行拼接为 `C`。`B` 的 FP4 权重在 Prologue 阶段直接 Cast 为 FP8 参与 Cube 计算，省去分形转换。
- 计算公式：

  $$
    \begin{aligned}
    C_g &= \tilde{A}_g \times \tilde{B}_g \\
    C_{i,j} &= \Sigma_{k}\left(a_{i,k}\cdot s^{A}_{i,\lfloor k/32 \rfloor}\right)\left(b_{k,j}\cdot s^{B}_{\lfloor k/32 \rfloor,j}\right)
    \end{aligned}
  $$

  其中 $\tilde{A}_g$、$\tilde{B}_g$ 分别是第 `g` 组 `A` 与对应 `B` 经 MX 缩放因子反量化后的矩阵，$a$、$b$ 为 FP8/FP4 量化值，$s^{A}$、$s^{B}$ 为每 32 个 K 元素共享的缩放因子；$C$ 是形如`(m,n)`的输出矩阵。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float8_e4m3_t` | 左矩阵，按 `GroupList` 分为 `G` 组，layout支持`RowMajor` |
| `B` | Input | `(G,k,n)` | `float4_e2m1x2_t` | 每组对应的右矩阵（FP4 权重），layout支持`Weight4BitnZ`（默认）和`ColumnMajor` |
| `MxScaleA` | Input | `(m,⌈k/32⌉)` | `float8_e8m0_t` | `A` 的 MX 缩放因子，K 方向每 32 个元素共享 |
| `MxScaleB` | Input | `(G,⌈k/32⌉,n)` | `float8_e8m0_t` | `B` 的 MX 缩放因子，K 方向每 32 个元素共享 |
| `GroupList` | Input | `(G,)` | `int64_t` | 每组在 M 方向的行数，各组之和等于 `m` |
| `C` | Output | `(m,n)` | `fp16` | 各分组矩阵乘结果按行拼接后的结果，layout支持`RowMajor` |

## 使用范围说明

本样例在基础矩阵乘之上增加 A8W4 权重量化（`A` 为 FP8、`B` 为 FP4）与分组 MX 能力，`B` 的 FP4 权重以 `Weight4BitnZ` 分形存储并在 Prologue 阶段直接 Cast 为 FP8 参与 Cube 计算。

支持范围：

- `problem_count`（分组数）、`m`、`n`、`k` 均须大于 0。
- `m` 等于各组 `GroupList` 之和（总 M）。
- `B` 布局支持 `Weight4BitnZ`（默认）和 `ColumnMajor`；`Weight4BitnZ` 需按 `k` 对齐 32、`n` 对齐 16 的分形对齐。

推荐范围：

- `k` 推荐为 64 的倍数（`k % 64 == 0`），避免 K 向 32 分组数为奇数（`k % 64 == 32`）时 MX 缩放因子尾块反量化出现 NaN/Inf。

## 代码组织

```text
74_ascend950_weight_quant_a8w4_grouped_mx_matmul
├── README.md
├── gen_data.py  # 生成测试数据（input/ 与 golden/），支持 group_list 与 expect_m_per_group 两种模式
├── weight_quant_a8w4_grouped_mx_matmul.cpp  # 主文件，恒等块 swizzle 分组 MX 矩阵乘
└── CMakeLists.txt
```

## 使用示例

### 命令行参数

执行样例命令如下：

```bash
74_ascend950_weight_quant_a8w4_grouped_mx_matmul [problem_count] [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `problem_count` | 无 | 分组个数 `G`，等于 `GroupList` 的长度 |
| `m` | 无 | 左矩阵的 m 轴大小（总 M），等于各组 `GroupList` 之和 |
| `n` | 无 | 右矩阵的 n 轴大小 |
| `k` | 无 | 左右矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

生成测试数据命令行如下：

```bash
# 模式一：显式指定 group_m_list（各组 M 之和须等于 m）
python3 examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/gen_data.py group_list 128,128 256 256 256 1
# 模式二：按每组期望平均值随机生成 group_m_list
python3 examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/gen_data.py expect_m_per_group 2 128 256 256 256 1
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `group_m_list` | 无 | （模式一）每个专家对应的分组大小，逗号分隔，各组之和须等于 `m` |
| `group_num` | 无 | （模式二）专家数 / 分组数 `G` |
| `expect_m_per_group` | 无 | （模式二）每组期望平均分组大小 |
| `m` | 无 | 总的 M，等于各组 `group_m_list` 之和 |
| `n` | 无 | 矩阵乘的 n 维 |
| `k` | 无 | 矩阵乘的 k 维 |
| `isNz` | 无 | `0` 为 `ColumnMajor`（nd）布局，`1` 为 `Weight4BitnZ`（nZ）布局 |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件（Ascend 950 需指定 `-DCATLASS_ARCH=3510`）。

    ```bash
    bash scripts/build.sh 74_ascend950_weight_quant_a8w4_grouped_mx_matmul -DCATLASS_ARCH=3510
    ```

2. 生成测试数据（`input/` 与 `golden/` 写入样例目录下的 `data/`）。

    ```bash
    python3 examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/gen_data.py group_list 128,128 256 256 256 1
    ```

3. 在项目根目录执行算子样例程序（可执行文件位于 `output/bin`，测试数据从样例目录的 `data/` 读取）。

    ```bash
    ./output/bin/74_ascend950_weight_quant_a8w4_grouped_mx_matmul 2 256 256 256 0
    ```

4. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```

## 补充说明

- 样例未启用 Bias（`ElementBias` 为 `void`）。
- 社区包暂不支持 950 能力，后续支持的版本敬请期待。
- 编译与执行环境准备可参考 [quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)。
- **B 矩阵布局切换**：本案例支持 `B` 矩阵切换 `ColumnMajor` 和 `Weight4BitnZ` 格式的输入，默认为 `Weight4BitnZ` 格式。如需切换，请在 `weight_quant_a8w4_grouped_mx_matmul.cpp` 文件中将：

    ```cpp
    using LayoutPrologueB = layout::Weight4BitnZ
    ```

    修改为：

    ```cpp
    using LayoutPrologueB = layout::ColumnMajor
    ```

    同时生成 nd 布局的 `B` 矩阵时，将 `gen_data.py` 传入参数 `isNz` 置为 `0`：

    ```bash
    python3 examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/gen_data.py group_list 128,128 256 256 256 0
    ```

- **Weight4BitnZ 分形结构**：通常的 FP4 内部分形为 16x64，但这里的 `Weight4BitnZ` 内部分形为 16x32，和 FP8 分形保持一致，目的是为了方便 vector 将 FP4 Cast 到 FP8，省去分形转换，提高算子性能。

- **性能测试**：本例性能对比如下表格所示。

    | group_num | m_per_expert | n | k | catlass(us) | 标杆(us) | 标杆/catlass |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | 48 | 64 | 4096 | 1024 | 89.573 | 164.111 | 1.83214808 |
    | 48 | 80 | 4096 | 1024 | 94.001 | 167.312 | 1.779895959 |
    | 48 | 96 | 4096 | 1024 | 96.111 | 178.953 | 1.861940881 |
    | 48 | 112 | 4096 | 1024 | 101.063 | 181.987 | 1.800728259 |
    | 48 | 128 | 4096 | 1024 | 106.075 | 162.171 | 1.528833373 |

    本例采用了动态 tiling、L2 Cache Hit、Double Buffer 等优化，实现性能提升 10%+。
