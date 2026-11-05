# MatmulEvg Example Readme

## 功能说明

- 算子功能：使用 EVG（Epilogue Visitor Graph）完成矩阵乘及后处理（如加法、偏置等）的融合计算。
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    D &= f(C)
    \end{aligned}
  $$

  其中 $A$ 和 $B$ 分别是形如 `(m,k)`、`(k,n)` 的输入矩阵，$C$ 是矩阵乘中间结果，$D$ 是形如 `(m,n)` 的最终输出。$f$ 对应后处理操作。更多说明见 [EVG 设计文档](../../docs/zh/2_Design/03_evg/01_evg_design.md)。

- 支持产品型号：Ascend 950PR&950DT系列产品

| 可执行文件 | 计算公式 |
| --- | --- |
| `64_ascend950_matmul_evg_add` | D = A*B + X |
| `64_ascend950_matmul_evg_leaky_relu` | D = LeakyRelu(A*B) |
| `64_ascend950_matmul_evg_sigmoid` | D = Sigmoid(A*B) |
| `64_ascend950_matmul_evg_silu` | D = Silu(A*B) |
| `64_ascend950_matmul_evg_tanh` | D = Tanh(A*B) |
| `64_ascend950_matmul_evg_bias` | D = A*B + bias |
| `64_ascend950_matmul_evg_add_ub` | D = A*B + X |

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float32` | 左矩阵，当前 layout 为 `RowMajor` |
| `B` | Input | `(k,n)` | `float32` | 右矩阵，当前 layout 为 `RowMajor` |
| `X` | Input | `(m,n)` | `float32` | `add`、`add_ub` 的加法输入，当前 layout 为 `RowMajor` |
| `bias` | Input | `(1,n)` | `float32` | `bias` 的偏置向量，沿 m 轴广播 |
| `leakyReluAlpha` | Input | 标量 | `float32` | `leaky_relu` 的负半轴斜率，默认 `0.1` |
| `D` | Output | `(m,n)` | `float32` | 融合计算结果，当前 layout 为 `RowMajor` |

## 使用范围说明

推荐场景：

- 矩阵的 M/N/K 轴不超过 122880。
- `A`/`B` 均为 `float32` 类型时，建议 `K` 小于 3072 以避免绝对误差超上限。

## 使用示例

### 命令行参数

```bash
64_ascend950_matmul_evg_add [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备 ID |

7 个可执行文件的参数相同，替换文件名即可编译和运行其他样例。

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 64_ascend950_matmul_evg_add -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./64_ascend950_matmul_evg_add 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
