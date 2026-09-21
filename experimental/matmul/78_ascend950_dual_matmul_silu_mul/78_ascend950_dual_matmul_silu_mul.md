# DualMatmul SiLU Mul 样例

## 功能说明

给定共享左矩阵 `X(M,K)` 和两个权重矩阵 `B0(N,K)`、`B1(N,K)`，计算：

```text
D = SiLU(X * B0^T) * (X * B1^T)
```

两个矩阵乘共享 `X` 的片上数据，结果在 UB 完成 SiLU、乘法和类型转换后写回 `D(M,N)`。

## 参数与约束

命令行参数依次为 `m`、`n`、`k` 和可选的 `deviceId`。输入使用 `fp16`，输出使用 `fp16`；`m`、`n`、`k` 必须为正数，输入设备必须支持 Ascend 950。`B0`、`B1` 的物理形状均为 `(n,k)`，并按转置逻辑参与计算。

optest 接口支持 `float16`/`bfloat16` 输入和输出组合，并要求三个输入 dtype 一致、Tensor 连续且为二维。

## 代码组织

```text
78_ascend950_dual_matmul_silu_mul/
├── CMakeLists.txt
├── README.md
├── 78_ascend950_dual_matmul_silu_mul.md
└── dual_matmul_silu_mul.cpp
```

主机文件负责选择 Arch、DispatchPolicy、TileShape、BlockMmad、BlockEpilogue 和 Scheduler，并通过 `DeviceGemm` 启动 kernel。optest 的 JIT host 负责按输入规模选择合法 tiling，模板文件只实现设备侧执行。

## 使用示例

编译样例：

```bash
bash scripts/build.sh -DCATLASS_ARCH=3510 78_ascend950_dual_matmul_silu_mul
```

运行样例：

```bash
cd output/bin
./78_ascend950_dual_matmul_silu_mul 128 128 128 0
```

输出 `Compare success.` 表示结果通过 golden 校验。
