# PlanarComplexMatmul

本文档用于说明 `examples/77_planar_complex_matmul` 平面复数矩阵乘算子示例所依赖的 Catlass GEMM 模板库能力、外部接口、分层设计方案。

## 功能说明

 - 算子功能：完成平面复数矩阵乘计算。复数矩阵以实部、虚部分离的 planar complex 形式输入，样例输出实部与虚部两路结果。
 - 计算公式：

$$
\begin{aligned}
    A &= A_{real} + iA_{imag} \\
    B &= B_{real} + iB_{imag} \\
    C &= A \times B \\
    C_{real} &= A_{real} \times B_{real} - A_{imag} \times B_{imag} \\
    C_{imag} &= A_{real} \times B_{imag} + A_{imag} \times B_{real}
    \end{aligned}
$$

  其中 `A_real`、`A_imag` 是形如 `(m, k)` 的左矩阵实部和虚部，`B_real`、`B_imag` 是形如 `(k, n)` 的右矩阵实部和虚部，`C_real`、`C_imag` 是形如 `(m, n)` 的输出矩阵实部和虚部。

## 参数说明

以下是本样例可执行文件的运行参数：

| 参数名 | 描述 | 约束 |
| ----- | -------- | ------ |
| `m` | 复数矩阵乘中左矩阵 A 的行 | 正整数 |
| `n` | 复数矩阵乘中右矩阵 B 的列 | 正整数 |
| `k` | 复数矩阵乘中左矩阵 A 的列，也即右矩阵 B 的行 | 正整数 |
| `deviceId` | 使用的 NPU 卡 ID（默认 0） | 在设备 NPU 有效范围内 |
| `--datapath` | 输入数据与输出数据目录 | 可选；未指定时仅随机生成输入并统计 kernel 耗时 |

## 代码组织

```text
├── examples
│   └── 77_planar_complex_matmul
│       ├── CMakeLists.txt              # CMake 编译文件
│       ├── README.md                   # 本文档
│       ├── 77_planar_complex_matmul.md # 设计文档
│       ├── gen_data_compare.py         # NumPy golden 数据生成与精度比对脚本
│       └── planar_complex_matmul.cpp   # 样例主文件（Host 层入口）
└── include
    └── catlass
        ├── gemm
        │   ├── block
        │   │   └── block_mmad_planar_complex_fused_tla.hpp
        │   │       # BlockMmadTla 针对 MmadPlanarComplexFused 的偏特化
        │   │       # 实现 Fused 路径的双 stage K-loop 与 L0C 分时复用
        │   ├── device
        │   │   └── (复用主仓库 device_gemm.hpp)
        │   │       # Device 层薄封装：参数透传 + KernelAdapter 启动
        │   └── kernel
        │       └── planar_complex_gemm_tla.hpp
        │           # PlanarComplexGemm 统一 kernel（Four-Pass / Fused 编排）
        │           # 含 NegateMatrixAiv AIV 取负预处理组件
        └── dispatch_policy.hpp
            # MmadPlanarComplexFused policy（仅 ENABLE_SHUFFLE_K 参数）
```

## 使用示例

1. 编译样例代码，并生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 77_planar_complex_matmul
    ```

2. 切换到可执行文件的编译目录 `output/bin` 下，执行算子样例程序。该方式随机生成输入数据，只输出 kernel 调度路径与平均耗时，不做精度比对。

    ```bash
    cd output/bin
    ./77_planar_complex_matmul 256 512 1024 0
    ```

    • 256：矩阵 m 轴

    • 512：矩阵 n 轴

    • 1024：矩阵 k 轴

    • 0：Device ID，可选，默认为 0

    执行结果中包含如下信息，说明样例执行成功。

    ```text
    PlanarComplexGemm dispatch: M=256 N=512 K=1024 ... -> Fused ...
    PlanarComplexGemm: M=256 N=512 K=1024 variant=Fused gemm=... ms (20 iters)
    No --datapath provided, skipping validation.
    ```

3. 在`catlass`目录下，使用 `gen_data_compare.py` 生成输入数据、运行 NPU 可执行文件并与 NumPy golden 结果进行比对。脚本默认从仓库根目录自动定位 `output/bin/77_planar_complex_matmul`，默认在当前目录下生成 `data`、`golden` 目录并在结束后删除；如需指定保存路径可使用 `--save_path`，如需保留可指定 `--clean false`。

    ```bash
    python examples/77_planar_complex_matmul/gen_data_compare.py 256 512 1024
    ```

    执行结果如下，说明精度比对成功。

    ```text
    Data generated: M=256, N=512, K=1024, BLAS threads=8
    ------计算npu------
    ------ 计算相对误差 -----
    Precision metric: ...
    ------ 开始比较 ------
    比较结果：Compare success
    ```
