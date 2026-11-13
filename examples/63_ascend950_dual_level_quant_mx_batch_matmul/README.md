# DualLevelQuantMxBatchMatmul Example Readme

## 功能说明

- 算子功能：对 fp16 输入 `A`/`B` 做两级量化后执行 MX FP4 batch 矩阵乘——第一阶段沿 K 轴按 512 元素块量化到 fp16（LEVEL0），第二阶段再沿 K 轴按 32 元素块做 MX FP4(E2M1) 量化（LEVEL1），最后进行 MX FP4 矩阵乘并输出 bf16 结果。
- 计算公式：

  $$
    \begin{aligned}
    \tilde{A}_{b} &= \text{FP4}_{E2M1}(\text{fp16}(A_{b} / scale0_A) / scale1_A) \\
    \tilde{B}_{b} &= \text{FP4}_{E2M1}(\text{fp16}(B_{b} / scale0_B) / scale1_B) \\
    C_{b,i,j} &= \Sigma_{k} \tilde{A}_{b,i,k}\tilde{B}_{b,j,k}
    \end{aligned}
  $$

  其中`A`和`B`分别是形如`(batchCount,m,k)`，`(batchCount,n,k)`的输入矩阵（`B` 为转置存储，逻辑形状为`(k,n)`），`C`是形如`(batchCount,m,n)`的输出矩阵（转为 bf16）；`scale0`/`scale1` 为沿 K 轴逐块计算的两级缩放系数。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(batchCount,m,k)` | `fp16` | 左矩阵，layout支持`RowMajor` |
| `B` | Input | `(batchCount,n,k)` | `fp16` | 右矩阵，物理按`(batchCount,n,k)` `RowMajor`转置存储（逻辑为`(k,n)` `ColumnMajor`），数据类型与左矩阵一致 |
| `C` | Output | `(batchCount,m,n)` | `bf16` | 二级量化 MX FP4 矩阵乘结果，layout支持`RowMajor` |

## 使用范围说明

本样例固定 `A`/`B` 为 fp16 输入、`C` 为 bf16 输出，layout 固定为上述组合；两级量化均沿 K 轴进行，LEVEL0 块大小为 `512`，LEVEL1 块大小为 `32`。

支持范围：

- `batchCount`、`m`、`n`、`k` 均须大于 `0`。
- `k` 须为偶数，满足 FP4 打包要求。

以上取值不满足时，样例在算子校验阶段直接退出。

## 代码组织

```text
63_ascend950_dual_level_quant_mx_batch_matmul
├── README.md
├── gen_data.py  # 生成测试数据（input/ 与 golden/）
├── dual_level_quant_mx_batch_matmul.cpp  # 二级量化 + MX FP4 batch matmul 调用示例
└── CMakeLists.txt
```

## 使用示例

### 命令行参数

执行样例命令如下：

```bash
63_ascend950_dual_level_quant_mx_batch_matmul [batchCount] [m] [n] [k] [deviceId]
```

生成测试数据命令行如下：

```bash
python3 gen_data.py [batchCount] [m] [n] [k] [trans_a] [trans_b]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `batchCount` | 无 | batch 数，须大于 `0` |
| `m` | 无 | A 矩阵的 m 轴大小，须大于 `0` |
| `n` | 无 | B 矩阵的 n 轴大小，须大于 `0` |
| `k` | 无 | A/B 矩阵的 k 轴大小，须大于 `0` 且为偶数 |
| `trans_a` | `0` | 仅兼容保留，生成布局固定，不参与实际转置 |
| `trans_b` | `1` | 仅兼容保留，生成布局固定，不参与实际转置 |
| `deviceId` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 63_ascend950_dual_level_quant_mx_batch_matmul -DCATLASS_ARCH=3510
    ```

2. 生成测试数据。

    ```bash
    python3 examples/63_ascend950_dual_level_quant_mx_batch_matmul/gen_data.py 1 1024 1024 1024
    ```

3. 执行算子。

    ```bash
    ./output/bin/63_ascend950_dual_level_quant_mx_batch_matmul 1 1024 1024 1024 0
    ```

4. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Result saved to: examples/63_ascend950_dual_level_quant_mx_batch_matmul/data/golden/result.bin
    Compare success.
    ```

## 补充说明

本样例依赖的新增 CATLASS 头文件：

```text
include/catlass/epilogue/block/block_epilogue_dual_level_quant_mx.hpp
include/catlass/epilogue/tile/tile_copy_dual_level_quant_mx.hpp
include/catlass/gemm/kernel/dual_level_quant_mx_batched_matmul_tla.hpp
```

核心组件选型：

| 组件             | 选型                                  |
| ---------------- | ------------------------------------- |
| ArchTag          | `Arch::Ascend950`                     |
| Kernel           | `DualLevelQuantMxBatchedMatmulTla`    |
| AIV BlockQuant   | `BlockQuantDualLevelMx`               |
| TileCopy (Quant) | `TileCopyDualLevelQuantMx`            |
| BlockMmad        | `BlockMmadTla`                        |
| BlockScheduler   | `GemmIdentityBlockSwizzle<3, 0/1>`    |
| DispatchPolicy   | `MmadMx<Ascend950, true, 16>`         |
| L1TileShape      | `Shape<Int<256>, Int<256>, Int<512>>` |
| L0TileShape      | `Shape<Int<256>, Int<256>, Int<256>>` |
| ElementInput     | `float16_t`                           |
| ElementA/B       | `float4_e2m1x2_t`                     |
| ElementMxScale   | `float8_e8m0_t`                       |
| ElementC         | `bfloat16_t`                          |

前置条件：

- 当前实现依赖 mix kernel 中 AIV/AIC 两侧配对调用 `AscendC::SyncAll<false>()`。
- 多流生产场景上线前需要确认 batchmode 已开启，否则 `SyncAll` 可能死锁。
