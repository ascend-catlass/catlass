# DualMatmul SiLU Mul Example

## 代码组织

```text
78_ascend950_dual_matmul_silu_mul
├── 78_ascend950_dual_matmul_silu_mul.md
├── CMakeLists.txt
├── README.md
└── dual_matmul_silu_mul.cpp
```

## 功能介绍

本样例实现共享左矩阵的双 Matmul 与 SiLU/Mul 融合计算：

```text
D0 = X * B0
D1 = X * B1
D  = SiLU(D0) * D1
```

`X` 的逻辑布局为 RowMajor，`B0` 和 `B1` 的逻辑布局为 ColumnMajor，输出 `D` 为 RowMajor。
AIC 完成两路 Matmul，累加结果经 Fixpipe 从 L0C 搬运到 UB，AIV 在 UB 内完成 SiLU、逐元素乘法和输出转换。

该独立样例使用 FP16 输入和 FP16 输出。CATLASS-optest 接口还支持 FP16/BF16 输入与 FP16/BF16 输出组合。

本可执行文件在 Host 侧直接计算 CPU Golden，适合功能演示和中小规模单 case 验证；完整的数据类型组合与泛化测试通过 CATLASS-optest 执行。

## 使用示例

编译 Ascend 950 样例：

```bash
bash scripts/build.sh -DCATLASS_ARCH=3510 78_ascend950_dual_matmul_silu_mul
```

运行样例：

```bash
cd output/bin
./78_ascend950_dual_matmul_silu_mul 128 128 128 0
```

参数依次为 `M N K [device_id]`，其中 `device_id` 可选，默认值为 0。输出 `Compare success.` 表示结果比对通过。
