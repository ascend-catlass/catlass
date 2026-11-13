# BroadcastMatmulPerblockQuant Example Readme

## 功能说明

- 算子功能：完成带 batch 广播的矩阵乘法，并对每个 batch 的 `m×n` 结果块做 E4M3（MXFP8）量化，输出量化结果 `dst` 与逐 batch 缩放系数 `scale`。
- 计算公式：

  $$
    \begin{aligned}
    C_{b,i,j} &= \Sigma_{k} A_{b,i,k}B_{k,j} \\
    scale_{b} &= \frac{\max_{i,j}|C_{b,i,j}|}{448} \\
    dst_{b,i,j} &= \text{quantize}(\frac{C_{b,i,j}}{scale_{b}})
    \end{aligned}
  $$

  其中`A`和`B`分别是形如`(batchCount,m,k)`，`(k,n)`的输入矩阵，`dst`是形如`(batchCount,m,n)`的 E4M3 量化输出矩阵，`scale`是形如`(batchCount)`的缩放系数向量。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(batchCount,m,k)` | `bf16` | 左矩阵，layout支持`RowMajor` |
| `B` | Input | `(k,n)` | `bf16` | 右矩阵，layout支持`RowMajor`，数据类型与左矩阵一致 |
| `dst` | Output | `(batchCount,m,n)` | `float8_e4m3_t` | 矩阵乘结果的 E4M3 量化结果，layout支持`RowMajor` |
| `scale` | Output | `(batchCount)` | `fp32` | 每个 batch 结果块的量化缩放系数，layout支持`VectorLayout` |

## 使用范围说明

本样例在广播矩阵乘的基础上增加逐 batch 的 E4M3 量化输出，不提供 Padding/Preload/切K 等优化；输入与输出的数据类型、layout 均固定为上述单一组合。

支持范围：

- `batchCount` 取值范围为 `[1, 65536]`。
- `m` 取值为 `128` 或 `256`。
- `n` 与 `k` 取值固定为 `128`。

以上取值不满足时，样例在算子校验阶段直接退出。

## 代码组织

```text
62_ascend950_broadcast_matmul_perblock_quant
├── README.md
├── gen_data_compare.py  # 生成测试数据并比对精度
├── broadcast_matmul_perblock_quant_tla.cpp  # 算子调用示例
└── CMakeLists.txt
```

## 使用示例

### 命令行参数

执行样例命令如下：

```bash
62_ascend950_broadcast_matmul_perblock_quant [batchCount] [m] [n] [k] [deviceId]
```

生成测试数据与精度比对命令行如下：

```bash
python3 gen_data_compare.py [batchCount] [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batchCount` | 无 | 广播 batch 数，取值范围 `[1, 65536]` |
| `m` | 无 | A 矩阵的 m 轴大小，取值 `128` 或 `256` |
| `n` | 无 | B 矩阵的 n 轴大小，固定为 `128` |
| `k` | 无 | A/B 矩阵的 k 轴大小，固定为 `128` |
| `deviceId` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 62_ascend950_broadcast_matmul_perblock_quant -DCATLASS_ARCH=3510
    ```

2. 切换到样例目录，运行数据生成与精度比对脚本。

    ```bash
    cd examples/62_ascend950_broadcast_matmul_perblock_quant
    python3 gen_data_compare.py 1024 128 128 128 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    ------ 生成测试数据 ------
    batch_count=1024, m=128, n=128, k=128
    ------ 运行NPU算子 ------
    npu op run log =
    ------ 比对结果 ------
    ------ 计算相对误差 -----
    ------ 综合精度指标 ------
    dst: npu mare=0.1250, golden mare=0.125000
    dst: npu mere=0.0018, golden mere=0.001839
    dst: npu rmse=1.3765, golden rmse=1.377344
    scale: npu mare=0.0030, golden mare=0.003028
    scale: npu mere=0.0013, golden mere=0.001291
    scale: npu rmse=0.0006, golden rmse=0.000632
    ------ 开始比较 ------
    精度指标比较结果：Compare success
    ```
