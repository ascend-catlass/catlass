# DynamicOptimizedMatmul Example Readme

## Introduction

In the field of high-performance computing and deep learning, matrix multiplication (Matmul) serves as a core operator, making its computational efficiency and generalization capability highly critical. For computational efficiency, the standard evaluation metric is hardware utilization, defined as actual FLOPS divided by theoretical FLOPS. Generalization, however, is difficult to quantify with a single metric. In certain service scenarios, such as recommendation systems, the dimensions of M, N, and K can vary across a range of 10<sup>5</sup>, requiring robust performance across this entire spectrum. Compared to optimization targeting a specific shape or domain, this broad generalization requirement poses a much higher technical challenge.

## 2 Document Index and Constraints

### Project Description

For details about the generalized Matmul project structure, see [Project Structure Description](<./docs/en/Project Structure_en.md>).

By default, the project is compiled into a static library. To compile the project into a dynamic library, change `STATIC` in `CMakeLists.txt` to `SHARED` and manually export the dynamic library path.
Before compilation, the build system invokes a Python script to generate code, which includes the boilerplate wrapper code for each template and `launch_map.h` (which defines the mapping between the tilingKey and specific kernels).

DynamicOptimizedMatmul dynamically determines the tiling parameters based on the runtime shape and attempts to select the best template for execution to maximize performance. However, absolute optimal performance is not guaranteed.

### Template Documents

| Template Name                                                    | Description                                                                   |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [CommonMatmul](./docs/en/CommonMatmul_en.md)                     | Basic template                                                                |
| SmallMatmul                                                      | Doc to be supplemented...                                                     |
| [MultiCoreSplitkMatmul](./docs/en/MultiCoreSplitkMatmul_en.md)   | Multi-core Split-K template (applicable to scenarios where matrix C is small) |
| [StreamkMatmul](./docs/en/StreamkMatmul_en.md)                   | Multi-core Split-K template with more balanced load                           |
| [SingleCoreSplitkMatmul](./docs/en/SingleCoreSplitkMatmul_en.md) | Single-core Split-K template                                                  |

### Related Constraints

1. The data types of matrices A, B, and C support fp16.

2. The memory layouts of matrices A, B, and C support ND formats (RowMajor and ColumnMajor).

## 3 Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compiling a specified case
bash scripts/build.sh 102_dynamic_optimized_matmul
# For dynamic library compilation, you need to manually add the dynamic library path to LD_LIBRARY_PATH.
export LD_LIBRARY_PATH=$PWD/output/shared_lib:$LD_LIBRARY_PATH
cd output/bin
# Executable file name | Matrix M axis | N axis | K axis | LayoutA | LayoutB | Device ID
# 0 is RowMajor, 1 is ColumnMajor
./102_dynamic_optimized_matmul 256 512 1024 0 1 0
```

If the following result is displayed, precision verification is successful.

```bash
Compare success.
```

If you need to perform batch performance benchmarks, comment out the accuracy verification code. Generating the golden data on the CPU during verification is time-consuming.
