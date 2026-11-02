# StreamkMatmul Example Readme

## 功能说明

- 算子功能：完成基础矩阵乘计算，采用切尾轮基本块的多核切K调度，通过切分尾轮基本块的`K`，划分出更多的任务块，从而利用更多的核心参与尾轮的基本块计算
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$

  其中$A$和$B$分别是形如`(m,k)`，`(k,n)`的输入矩阵，$C$是形如`(m,n)`的输出矩阵。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| ---- | ---- | ----- | ----- | ---- |
| `A` | Input | `(m,k)` | `float、float16、bfloat16` | 左矩阵，layout 支持 `RowMajor`（默认）和 `ColumnMajor` |
| `B` | Input | `(k,n)` | `float、float16、bfloat16` | 右矩阵，layout 支持 `RowMajor`（默认）和 `ColumnMajor`，数据类型与左矩阵一致 |
| `C` | Output | `(m,n)` | `float、float16、bfloat16` | 矩阵乘结果，layout 支持 `RowMajor`（默认）和 `ColumnMajor` |

## 使用范围说明

本样例为切尾轮基本块的多核切K模板。与`69_ascend950_tail_multi_core_splitk_matmul`不同的是，本样例不再通过一个切分因子切分`K`，而是将尾轮的计算量直接均分到所有计算核心上。

推荐范围：

- 约定L1上的分块大小为`m1 n1 k1`，设`blocks = ceil(M/m1)*ceil(N/n1)`，推荐`blocks % C ∈ (0, 0.8C) 且 1024 < K < 122880`（C 为 AI Core 核数），即尾轮有残余块且不足 0.8 倍核数的深 K 场景。

## 使用示例

### 命令行参数

```bash
66_ascend950_streamk_matmul [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 66_ascend950_streamk_matmul -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./66_ascend950_streamk_matmul 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
