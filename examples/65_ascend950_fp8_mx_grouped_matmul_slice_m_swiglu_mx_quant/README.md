# FP8MxGroupedMatmulSliceMSwigluMxQuant Example Readme

## 功能说明

- 算子功能：完成分组（grouped）MX FP8 矩阵乘，按 M 轴切片（slice-M）分组，再对结果执行 SwiGLU 激活后做 MX FP8 量化输出。
- 计算公式：

  $$
    \begin{aligned}
    \tilde{A} &= A \odot \text{MxScaleA} \\
    \tilde{B} &= B \odot \text{MxScaleB} \\
    C &= \tilde{A} \times \tilde{B} \\
    \text{act}, \text{gate} &= \text{split}_{N}(C) \\
    S &= \text{Swish}(\text{act}) \odot \text{gate} \\
    Q, \text{QScale} &= \text{MXQuant}(S)
    \end{aligned}
  $$

  其中`A`是形如`(m,k)`的输入矩阵，`B`是形如`(k,n)`的输入矩阵（按`(problem_count,n,k)`转置存储），`A`/`B`经各自的 MX 缩放因子反量化后做分组矩阵乘得到`C`；`C`沿 N 轴均分为`act`与`gate`两半，经 SwiGLU 激活（`Swish(act) * gate`）后按 32 元素块做 MX FP8 量化，得到形如`(m,n/2)`的输出`Q`及其缩放因子`QScale`。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `A` | Input | `(m,k)` | `float8_e4m3_t` | 左矩阵，layout支持`RowMajor` |
| `MxScaleA` | Input | `(m,ceil(k/64),2)` | `float8_e8m0_t` | `A` 的 MX 缩放因子，沿 K 轴按 32 元素块逐块缩放 |
| `B` | Input | `(problem_count,n,k)` | `float8_e4m3_t` | 右矩阵，layout支持`ColumnMajor`（转置存储，逻辑形状`(k,n)`），数据类型与左矩阵一致 |
| `MxScaleB` | Input | `(problem_count,n,ceil(k/64),2)` | `float8_e8m0_t` | `B` 的 MX 缩放因子，沿 K 轴按 32 元素块逐块缩放 |
| `GroupList` | Input | `(problem_count,)` | `int64_t` | 每组在 M 方向的行数，样例按 m 均分生成（余数分摊到前几组） |
| `Q` | Output | `(m,n/2)` | `float8_e4m3_t` | SwiGLU 激活后 MX FP8 量化输出，layout支持`RowMajor` |
| `QScale` | Output | `(m,ceil((n/2)/32))` | `float8_e8m0_t` | `Q` 的 MX 缩放因子 |

## 使用范围说明

本样例相对矩阵乘基线，增加了按 M 轴切片的分组（grouped）执行，并对矩阵乘结果做 SwiGLU 激活与 MX FP8 量化输出。

支持范围：

- `problemCount`（分组数）须满足 `1 ≤ problemCount ≤ 1024`。
- `n` 须为 `128` 的整数倍。

以上取值不满足时，样例在算子校验阶段直接退出。

## 代码组织

```text
65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant
├── README.md
├── gen_data.py  # 生成测试数据（input/ 与 golden/）
├── compare.py  # 计算 golden 并比对精度
├── fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant.cpp  # grouped MX FP8 矩阵乘 + SwiGLU + MX 量化示例
└── CMakeLists.txt
```

## 使用示例

### 命令行参数

执行样例命令如下：

```bash
65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant [problem_count] [m] [n] [k] [deviceId]
```

生成测试数据命令行如下：

```bash
python3 gen_data.py [problem_count] [m] [n] [k]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `problem_count` | 无 | 分组数，须满足 `1 ≤ problem_count ≤ 1024` |
| `m` | 无 | A 矩阵的 m 轴大小，须大于 `0` |
| `n` | 无 | B 矩阵的 n 轴大小，须大于 `0` 且为 `128` 的整数倍 |
| `k` | 无 | A/B 矩阵的 k 轴大小，须大于 `0` |
| `deviceId` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant -DCATLASS_ARCH=3510
    ```

2. 生成测试数据。

    ```bash
    python3 examples/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant/gen_data.py 4 256 512 1024
    ```

3. 执行算子。

    ```bash
    ./output/bin/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant 4 256 512 1024 0
    ```

4. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```

## 补充说明

- 本样例的 `L1TileShape(128,256,256)` 与 `L0TileShape(128,256,128)` 仅为样例 [53_ascend950_fp8_mx_matmul](../53_ascend950_fp8_mx_matmul/README.md) 的一半，原因在于一次循环中需要进行两个 block 的矩阵乘运算，两次结果需同时驻留 UB 供后续计算使用，后续计算还需 UB 存放中间结果，当前场景下 UB 存放数据共约 180K。受 UB 大小限制，本样例的分块仅为 53 样例的一半。
- `groupList` 中未指定的部分不会参与更新。如 `groupList` 为 `(3,4,5)`、`m` 为 `20`，则 `Q[12:]` 与 `QScale[12:]` 的部分不会被更新与初始化，其中数据为显存空间申请时的原数据，即 `Q` 和 `QScale` 的有效数据为 `Q[:12]` 与 `QScale[:12]`。
- `gen_data.py` 会同时生成 B 转置与不转置两份数据；当前样例为 B 转置（layout `ColumnMajor`）的示例。若需改为 B 不转置，需同步修改 `.cpp` 中 B 的 layout 为 `layout::RowMajor` 并改读对应二进制文件。layout 隐式表征转置状态：`layout::RowMajor` 表示不转置，`layout::ColumnMajor` 表示转置。
- 关于 MX 量化矩阵乘的详细特征，参见 [53_ascend950_fp8_mx_matmul](../53_ascend950_fp8_mx_matmul/README.md) 与 [54_ascend950_fp4_mx_matmul](../54_ascend950_fp4_mx_matmul/README.md)。
