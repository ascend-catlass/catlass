# HiFloat8 Quant Batch Matmul Example Readme

## 代码组织

```text
./examples/89_ascend950_hif8_quant_batch_matmul/
├── CMakeLists.txt     // CMake编译文件
├── gen_data
│   ├── gen_data.py    // 数据生成脚本
│   └── data/          // 离线数据文件（运行脚本后生成）
├── quant_batch_matmul.cpp   // 主文件
└── README.md   // 说明文件
```

## 功能说明

- 算子功能：完成量化的批矩阵乘计算
- 计算公式：

  $$
  \begin{aligned}
    out &= (x1@x2 + bias) \cdot \mathrm{trunc}(x1Scale \cdot x2Scale)
    \end{aligned}
  $$

  其中$x1$和$x2$分别是 batch 下形如`(m,k)`，`(k,n)`的输入矩阵，$bias$是形如`(n,)`的
  float 向量（`bias=0` 时无 bias），$x1Scale$是 per-tensor 标量，$x2Scale$是 per-tensor
  标量或 per-channel 向量，$out$是 batch 下形如`(m,n)`的 `bfloat16_t` 输出矩阵。
- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数        | 属性   | shape           | dtype                         | 说明                                                                                                                                                          |
| ----------- | ------ | --------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `x1`      | Input  | `(batch,m,k)` | `hifloat8_t`                | 左矩阵，layout 为`RowMajor`，离线文件以 `uint8` 保存 HiF8 原始编码                                                                                                                              |
| `x2`      | Input  | `(batch,k,n)` | `hifloat8_t`                | 右矩阵，layout 为`RowMajor`，数据类型与左矩阵一致                                                                                                        |
| `bias`    | Input  | `(n,)`        | `float`                     | 偏置向量；`bias=0` 时无 bias                                                                                                                             |
| `x1Scale` | Input  | 标量            | `float`                     | 左矩阵反量化 scale，由 `gen_data.py` 按 `x1quantmode` 生成（`0` 时取 1.0）                                                                              |
| `x2Scale` | Input  | 标量或 `(n,)` | `float` 或 `uint64_t`     | 右矩阵反量化 scale，由 `gen_data.py` 按 `x2quantmode` 生成：per-tensor 为 float 标量，per-channel 为 `(n,)` 向量（device 侧以 `uint64_t` 编码字传入） |
| `out`     | Output | `(batch,m,n)` | `bfloat16_t`                 | 矩阵乘结果，layout 为`RowMajor`                                                                                                                          |

## 使用范围说明

本样例使用 `hifloat8_t` 输入、`float` 累加和 `bfloat16_t` 输出，通过 Fixpipe 执行 per-tensor 或 per-channel 反量化。

离线文件 `tensor_a.bin`、`tensor_b.bin` 保存 HiF8 的 `uint8` 原始编码；`bias.bin`、`dequant_scale.bin`、`golden_out.bin` 使用 `float32`，`quant_scale.bin` 使用 `uint64` 编码字。`golden_out.bin` 的参考精度与算子的 BF16 输出精度不同。

`batch`、`M`、`K`、`N` 均须为正整数。K 按原始长度下发，不在 Host 侧补零。

## 使用示例

### 命令行参数

生成测试数据命令行如下:

```bash
python gen_data/gen_data.py [batch] [M] [K] [N] [bias] [x1quantmode] [x2quantmode]
```

执行样例命令如下:

```bash
89_ascend950_hif8_quant_batch_matmul [batch] [M] [K] [N] [bias] [x1quantmode] [x2quantmode] [deviceId]
```

上述命令行参数具体说明如下：

| 参数            | 默认值 | 参数说明                                                                                                   |
| --------------- | ------ | ---------------------------------------------------------------------------------------------------------- |
| `batch`       | 无     | batch 数                                                                                                   |
| `M`           | 无     | 每 batch 左矩阵的 m 轴大小                                                                                 |
| `K`           | 无     | 左或右矩阵的 k 轴大小                                                                                      |
| `N`           | 无     | 每 batch 右矩阵的 n 轴大小                                                                                 |
| `bias`        | 无     | bias 长度：`0`（无）或 `N`                                                                             |
| `x1quantmode` | 无     | 左矩阵的量化模式：`0`(NO_QUANT) \| `1`(PER_TENSOR)                                                     |
| `x2quantmode` | 无     | 右矩阵的量化模式：`0`(NO_QUANT) \| `1`(PER_TENSOR) \| `2`(PER_CHANNEL)                               |
| `deviceId`    | `0`  | 指定运行设备ID                                                                                             |

其中生成测试数据脚本在 `gen_data/data/` 下生成 `tensor_a.bin`、`tensor_b.bin`、`dequant_scale.bin`、`quant_scale.bin`、`bias.bin`、`golden_out.bin`。

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

   ```bash
   bash scripts/build.sh 89_ascend950_hif8_quant_batch_matmul -DCATLASS_ARCH=3510
   ```
2. 在项目根目录运行数据生成脚本，构造离线数据文件（参数：`batch M K N bias x1quantmode x2quantmode`）。

   ```bash
   python ./examples/89_ascend950_hif8_quant_batch_matmul/gen_data/gen_data.py 2 128 256 128 0 0 2
   ```
3. 在项目根目录执行算子样例程序。

   ```bash
   ./output/bin/89_ascend950_hif8_quant_batch_matmul 2 128 256 128 0 0 2 0
   ```
4. 执行结果如下，说明样例执行成功，精度通过：

   ```text
   Compare success.
   ```
