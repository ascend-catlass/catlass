# BasicMatmul Example Readme

## 功能说明

- 算子功能：基于 TLA 的基础矩阵乘计算。
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$

  其中 $A$ 和 $B$ 分别是形如 `(m,k)`、`(k,n)` 的输入矩阵，$C$ 是形如 `(m,n)` 的输出矩阵。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float32` | 左矩阵，layout 支持 `RowMajor`（默认）和 `ColumnMajor` |
| `B` | Input | `(k,n)` | `float32` | 右矩阵，layout 支持 `RowMajor`（默认）和 `ColumnMajor` |
| `C` | Output | `(m,n)` | `float32` | 矩阵乘结果，当前 layout 为 `RowMajor` |

调整数据类型和布局后需重新编译。

## 使用范围说明

推荐范围：

- `M ≥ 256、N ≥ 256、K ≤ 1024`，M/N 建议为 256 的倍数，K 建议为 128 的倍数。
- `K > 1024` 时，基本块数整除核数，或尾轮基本块数不少于核数的 0.8 倍。
- `A`/`B`/`C` 均为 `float32` 类型时，建议 `K` 小于 3072 以避免绝对误差超上限。

## 使用示例

### 命令行参数

```bash
43_ascend950_basic_matmul [m] [n] [k] [deviceId]
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
    bash scripts/build.sh 43_ascend950_basic_matmul -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./43_ascend950_basic_matmul 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
