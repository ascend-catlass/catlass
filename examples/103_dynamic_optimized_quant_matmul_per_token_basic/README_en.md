# Dynamic Optimized Quant Matmul Per-Token Basic Example Readme

## 1 Background

Based on generalization requirements, this example adds a generalized engineering implementation for quantized Matmul in addition to example 102. This example currently supports the basic template for PerToken-PerChannel quantized Matmul.

Quantization is widely used in deep learning models for modern high-performance computing, especially during inference. Through quantization, models can run more efficiently on hardware, reduce compute resource consumption, accelerate inference, and reduce model storage requirements.

The currently supported quantized compute modes include PerToken quantization and PerChannel quantization. In the following description, the m, n, and k variables represent the sizes of different axes in Tensor computation. The left matrix and right matrix refer to the two input Tensors used for matrix multiplication.

- PerToken quantization: This is usually used to quantize the left matrix. It uses independent quantization parameters for each Token, that is, each row of the left matrix. Assume the left matrix shape is (m, k), and k is the reduce axis. The generated PerToken quantization parameter shape is (m,).

- PerChannel quantization: This is usually used to quantize the right matrix. It uses independent quantization parameters for each Channel, that is, each column of the right matrix. Assume the right matrix shape is (k, n), and k is the reduce axis. The generated PerChannel quantization parameter shape is (n,).

Generally, the left matrix represents activation (A), and the right matrix represents weight (W). This example currently supports the scenario where the input data types of the left and right matrices are int8, PerToken quantization is applied to the left matrix, and PerChannel quantization is applied to the right matrix. This can be abbreviated as the W8A8 PerToken-PerChannel fully quantized Matmul scenario.

## 2 Document Index and Constraints

### Project Description

For the structure of the generalized quantized Matmul project, see [Project Structure Description](<../102_dynamic_optimized_matmul/docs/en/Project Structure_en.md>). This project follows processes similar to example 102, such as template generation, Tiling computation, and template selection, and adapts them based on the compute characteristics of quantized Matmul.

Before project compilation, a Python script is called to generate code, including the wrapper code for each template and launch_map.h, which contains the mapping between tilingKey and specific Kernels.

This project is compiled as a dynamic library by default. Before executing the example after compilation, export the dynamic library path:

```shell
export LD_LIBRARY_PATH=/path/to/catlass/output/shared_lib/lib/:$LD_LIBRARY_PATH
```

### Project Structure

```shell
├── CMakeLists.txt
├── README.md
├── dynamic_optimized_quant_matmul_per_token_basic.cpp
├── impl
│   ├── kernel
│   │   ├── per_token_matmul_kernel.h
│   ├── scripts
│   │   ├── templates
│   │   │   ├── per_token_matmul_template.py
│   │   ├── utils
│   │   │   └── config.py
│   │   └── wrapper_code_gen.py
│   └── wrapper # Automatically generated
│       ├── per_token_matmul_kernel_int8_t_layout00.cpp # Automatically generated
│       ├── per_token_matmul_kernel_int8_t_layout01.cpp # Automatically generated
│       ├── per_token_matmul_kernel_int8_t_layout10.cpp # Automatically generated
│       ├── per_token_matmul_kernel_int8_t_layout11.cpp # Automatically generated
└── include
    ├── do_tiling_b8.h
    ├── dynamic_optimized_matmul_w8a8.h
    ├── launch_map.h # Automatically generated
    ├── platform_info.h
    ├── select_kernel_b8.h
    ├── tiling_params.h
    └── utils.h
```

### Template Documentation

| Template Name       | Description                                            |
| ------------------- | ------------------------------------------------------ |
| PerTokenBasicMatmul | PerToken basic template (documentation to be added...) |

### Constraints

- The data types of matrices A and B support int8.
- The data type of matrix C supports fp16.
- The data formats of matrices A, B, and C support ND (RowMajor and ColumnMajor).

## 3 Compiling the Specified Case

```shell
bash scripts/build.sh 103_dynamic_optimized_quant_matmul_per_token_basic
export LD_LIBRARY_PATH=/path/to/catlass/output/shared_lib/lib/:$LD_LIBRARY_PATH
cd output/bin
# Executable file name | matrix m axis | n axis | k axis | LayoutA | LayoutB | Device ID
# 0 is RowMajor, 1 is ColumnMajor
./103_dynamic_optimized_quant_matmul_per_token_basic 256 512 1024 0 1 0
```

The execution result is as follows, indicating that the precision comparison succeeds.

```text
Compare success.
```

If batch performance testing is required, comment out the precision comparison code, because precision comparison uses the CPU to compute golden data and takes a long time.

---

The current example output data type is `fp16`. To change it to `bf16`, modify the following code and then recompile and execute:

- In `examples/103_dynamic_optimized_quant_matmul_per_token_basic/include/do_tiling_b8.h`, replace `fp16_t` used in all `DoTilingB8LayoutXX` functions with `bfloat16`.
- In `examples/103_dynamic_optimized_quant_matmul_per_token_basic/dynamic_optimized_quant_matmul_per_token_basic.cpp`, search for `fp16_t` and replace it with `bfloat16`.
- In `examples/103_dynamic_optimized_quant_matmul_per_token_basic/impl/scripts/per_token_matmul_template.py`, replace `element_c` from `half` with `bfloat16_t`.
