# Ascend950BmmBroadcast Example Readme

## 功能说明

- 算子功能：完成支持 batch 广播、输入转置和可选 Bias 的批量矩阵乘。
- 计算公式：

  $$
    C_b = \operatorname{cast}_T\left(\operatorname{op}(A_{b_A}) \times \operatorname{op}(B_{b_B}) + \operatorname{Bias}_b\right)
  $$

  其中，A、B 的 batch 为 1 时共享输入，否则使用对应 batch 的输入；`op` 表示是否转置。
  Bias 可省略，或为共享的 `(n)`、逐 batch 的 `(batch,n)`，沿 M 维广播。累加类型为 FP32。

- 支持产品型号：Ascend 950PR&950DT系列产品。

## 样例参数说明

下表列出 Kernel 模板及配套参数范围，shape 表示物理存储形状。可执行样例固定使用 FP16、未转置 A/B 和共享 Bias。

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(batchA,m,k)` 或 `(batchA,k,m)` | `float16`、`bfloat16` | 分别对应 `transA=false`、`transA=true`；物理存储连续，batchA 为 1 或 batch。 |
| `B` | Input | `(batchB,k,n)` 或 `(batchB,n,k)` | `float16`、`bfloat16` | 分别对应 `transB=false`、`transB=true`；物理存储连续，batchB 为 1 或 batch。 |
| `Bias` | Input | `(n)` 或 `(batch,n)` | `float16`、`bfloat16` | `bias_mode=0` 时不使用，1 时共享，2 时逐 batch；沿 M 维广播，dtype 与 A/B 一致。 |
| `C` | Output | `(batch,m,n)` | `float16`、`bfloat16` | 矩阵乘结果，连续 RowMajor，dtype 与 A/B 一致。 |

## 使用范围说明

支持范围:

- 本样例的使用范围为 FP16、BF16，累加类型为 FP32。
- 可执行样例固定使用 FP16、未转置输入及共享 Bias；BF16、A/B 转置和三种 Bias 配置由 Kernel 模板覆盖。
- batch、m、n、k 为正整数；m/n/k 不超过 INT32_MAX，总输出分块数不超过 UINT32_MAX，张量字节数不超过 INT64_MAX。
- A/B/Bias 数据类型一致，物理存储连续；输入 batch 为 1 或输出 batch。转置 A/B 的物理形状分别为 `(batchA,k,m)`、`(batchB,n,k)`。
- 不支持空维度、任意外部 stride 或周期广播；调用方须保证内存容量足够且输出不与输入重叠。

## 使用示例

### 命令行参数

```bash
88_ascend950_bmm_broadcast [batch] [batchA] [batchB] [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batch` | 无 | 输出批次数 |
| `batchA` | 无 | A 的批次数，1 或 batch |
| `batchB` | 无 | B 的批次数，1 或 batch |
| `m` | 无 | 输出矩阵的 m 轴大小 |
| `n` | 无 | 输出矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 88_ascend950_bmm_broadcast -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./88_ascend950_bmm_broadcast 3 3 1 67 131 145 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
