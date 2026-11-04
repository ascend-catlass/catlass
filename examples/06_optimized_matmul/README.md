# OptimizedMatmul Example Readme

## 功能说明

- 算子功能：使用 Preload 预取、ShuffleK 和前置 Padding 优化的矩阵乘计算。
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$

  其中 $A$ 和 $B$ 分别是形如 `(m,k)`、`(k,n)` 的输入矩阵，$C$ 是形如 `(m,n)` 的输出矩阵。

- 支持产品型号：Atlas A2/A3系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float16` | 左矩阵，layout支持`RowMajor`（默认）和`ColumnMajor` |
| `B` | Input | `(k,n)` | `float16` | 右矩阵，layout支持`RowMajor`和`ColumnMajor`（默认） |
| `C` | Output | `(m,n)` | `float16` | 矩阵乘结果，当前 layout 为 `RowMajor` |

表中列出当前样例源码的数据类型和布局，调整配置后需重新编译。

## 使用范围说明

约束：`m`、`n`、`k` 均须大于 0；否则样例在参数解析阶段直接退出。

推荐范围：

- K/N 对齐与否均可，非对齐时自动启用 Padding 前处理。

## 使用示例

### 命令行参数

```bash
06_optimized_matmul [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备 ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 06_optimized_matmul
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./06_optimized_matmul 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
