# MultiCoreSplitkMatmul Example Readme

## 功能说明

- 算子功能：完成矩阵乘计算，通过切分 K 轴增加计算任务，由向量核归并各段结果。
- 计算公式：

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \sum_{l=0}^{k-1} A_{i,l}B_{l,j}
    \end{aligned}
  $$

  其中 $A$ 和 $B$ 分别是形如 `(m,k)`、`(k,n)` 的输入矩阵，$C$ 是形如 `(m,n)` 的输出矩阵。

- 支持产品型号：Ascend 950PR&950DT系列产品。

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float32` | 左矩阵，layout 为 `RowMajor`。 |
| `B` | Input | `(k,n)` | `float32` | 右矩阵，layout 为 `RowMajor`。 |
| `C` | Output | `(m,n)` | `float32` | 矩阵乘结果，layout 为 `RowMajor`。 |

本样例固定使用 FP32 输入、FP32 累加和 FP32 输出，A/B 使用相同类型。PyTorch 入口通过 `transA`、`transB` 解释转置输入的存储形状。本说明不扩展其他输入或输出 dtype。

## 使用范围说明

输入矩阵的 m、n、k 均须为正整数，并满足 uint32 维度、workspace 索引及设备存储限制。本样例保持原有 MMAD 累加和普通 FP32 归并路径，不使用额外的短段累加或补偿求和。

使用范围限定为以下 FP32 推荐负载，K 范围采用保守的精度约定：

- `1025 <= k <= 3072`。
- 设 L1 分块大小为 `m1`、`n1`、`k1`，基本块数为 `blocks = ceil(m/m1)*ceil(n/n1)`，实际 AI Core 核数为 `coreNum`，由 `platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()` 查询。当前样例及 PyTorch 入口的 `m1 = n1 = 256`。
- `blocks <= coreNum/2`，即块少深 K 场景。切 K 因子仍由 `GetSplitkFactor` 按 shape、L1 分块及实际核数计算。

各 K 分段以 FP32 写入 workspace 后，由向量核普通求和归并。workspace 保持原有大小，即 `max(2*1024*1024, m*n*sizeof(float)*splitkFactor)`。

C++ 样例与 PyTorch 入口均通过 `CanImplement` 检查以上条件，超范围输入直接拒绝。该范围是保守的精度使用约定，不代表硬件维度上限或任意输入数值的误差保证。

## 使用示例

### 命令行参数

```bash
68_ascend950_multi_core_splitk_matmul [m] [n] [k] [deviceId]
```

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小。 |
| `n` | 无 | B 矩阵的 n 轴大小。 |
| `k` | 无 | A/B 矩阵的 k 轴大小。 |
| `deviceId` | `0` | 指定运行设备 ID。 |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 68_ascend950_multi_core_splitk_matmul -DCATLASS_ARCH=3510
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。以下 shape 适用于 `coreNum >= 4` 的设备。

    ```bash
    cd output/bin
    ./68_ascend950_multi_core_splitk_matmul 256 512 2048 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
