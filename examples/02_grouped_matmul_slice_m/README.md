# GroupedMatmulSliceM Example Readme

## 功能说明

- 算子功能：完成分组矩阵乘（Grouped Matmul）计算。
- 计算公式：

  $$
    C = \mathrm{GroupedGEMM}(A,B), \\
    C_i = A_i \times B_i, \quad 0 \le i < g
  $$

  其中 $g$ 为分组数量，分组矩阵乘中的$A_i$、$B_i$、$C_i$ 的形状分别为 `(m_i,k)`、`(k,n)`、`(m_i,n)`。各组沿 m 轴连续存储，共享 n、k 维度。

- 支持产品型号：Atlas A2系列产品、Atlas A3系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float16` | 各组左矩阵，layout 固定为 `RowMajor` |
| `B` | Input | `(g,k,n)` | `float16` | 右矩阵，layout支持`RowMajor` 和 `ColumnMajor`（默认），数据类型与左矩阵一致 |
| `groupList` | Input | `(g,)` | `int64` | m 轴分组长度的前缀和序列 |
| `C` | Output | `(m,n)` | `float16` | 分组矩阵乘结果，当前 layout 为 `RowMajor`；有效行数由 `groupList` 的终止量决定（不大于 `m`） |

## 使用范围说明

推荐范围：

- 分组数量 `g` 满足 `1 ≤ g ≤ 128` 且 `g ≤ m`。

## 使用示例

### 命令行参数

```bash
02_grouped_matmul_slice_m [problemCount] [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `problemCount` | 无 | 分组数量，必须大于 0 |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备 ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 02_grouped_matmul_slice_m
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./02_grouped_matmul_slice_m 128 512 1024 2048 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
