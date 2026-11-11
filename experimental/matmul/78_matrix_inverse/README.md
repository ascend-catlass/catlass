# MatrixInverse Example Readme

## 功能说明

- 算子功能：完成方阵的矩阵求逆计算
- 计算方法：基于LU分解与部分选主元（LU decomposition with partial pivoting）

  矩阵求逆计算 $A^{-1}$，满足 $A \times A^{-1} = I$，其中 $A$ 为输入方阵，$I$ 为单位矩阵。

  本算子采用LU分解算法：

  $$
    PA = LU
  $$

  其中 $P$ 为排列矩阵，$L$ 为下三角矩阵，$U$ 为上三角矩阵。通过求解 $UX = L^{-1}P$ 得到逆矩阵。

## 参数说明

以下是本样例的运行参数：

| 参数名     | 描述                   | 约束                      |
| ---------- | ---------------------- | ------------------------- |
| `N`        | 方阵的行/列数          | $0 < N \le 16384$（Atlas A2、`float`），且矩阵必须非奇异 |
| `deviceId` | 使用的NPU卡ID（默认0） | 在设备NPU有效范围内       |

MatrixInverse所涉及的关键模板参数如下:

| 模板参数  | 说明           | 有效范围                            |
| --------- | -------------- | ----------------------------------- |
| `Element` | 矩阵的数据类型 | `float` \| `fp16_t` \| `bfloat16_t` |
| `Layout`  | 矩阵的排布方式 | `layout::RowMajor`                  |

## 约束说明

- 输入矩阵必须是方阵（行数等于列数）
- 输入矩阵必须是非奇异矩阵（行列式不为零）
- `N` 受 UB 容量限制：`PanelGetrf` 在 UB 中叠加 3 段 `RoundUp8(N)` 的 `float` 缓冲，要求 $3 \times \text{RoundUp8}(N) \times 4 \le 192\text{KB}$，即 $N \le 16384$
- 为保证数值稳定性，本样例生成对角占优（diagonally dominant）的随机测试矩阵

## 代码组织

```text
├── 78_matrix_inverse
│   ├── CMakeLists.txt      # CMake编译文件
│   ├── README.md
│   └── matrix_inverse.cpp  # 主文件
```

## 使用示例

1. 编译样例代码，并编译生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 78_matrix_inverse
    ```

2. 切换到可执行文件的编译目录`output/bin`下，执行算子样例程序。测试样例数据随机生成（对角占优矩阵），尺寸从命令行输入。

```bash
cd output/bin
./78_matrix_inverse 128 0
```

- 128：方阵的维度N（128*128矩阵）

- 0：Device ID，可选，默认为0

执行结果如下，说明样例执行成功。

```text
Matrix Inverse: N=128, device=0
Compare success.
```

## 性能说明

矩阵求逆的计算复杂度为 $O(N^3)$，约需 $2N^3$ 次浮点运算。样例代码中包含被注释的精度统计与GFLOPS计算代码，可在需要时启用进行性能分析。
