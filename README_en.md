<h1 align="center">CATLASS</h1>

<p align="center">
  <a href="https://deepwiki.com/ascend-catlass/catlass"><img src="https://deepwiki.com/badge.svg" alt="Ask AI on DeepWiki"></a>
</p>

<p align="center">
  <a href="README.md"><strong>中文</strong></a> |
  <a href="README_en.md"><strong>English</strong></a>
</p>

<p align="center">
  <a href="https://catlass.readthedocs.io/zh-cn/latest/"><strong>Official Documentation</strong></a> |
  <a href="https://www.hiascend.com/cn/developer/operator?tag=catlass&amp;tab=catlass"><strong>Operator Development Journey</strong></a> |
  <a href="https://etherpad-cann.meeting.osinfra.cn/p/sig-catlass"><strong>Community Meetings</strong></a> |
  <a href="https://www.hiascend.com/"><strong>About Ascend</strong></a>
</p>

## Latest News

- [2026/10] [v2.2.0](https://gitcode.com/cann/catlass/releases/v2.2.0) released: expanded SIMD operations and related capabilities in **CATLASS Python** for Ascend 950, including automatic post-transfer address updates with `PostModeUpdate`, Histogram (`FREQUENCY`/`ACCUMULATE`), PER_TENSOR/PER_CHANNEL quantization and scale loading, inline ReLU on L0C stores, L1→BiasTable transfers, and ordinary Matmul with Bias. Officially introduced **SIMT** programming (`tla.vec.func(mode="simt", ...)`, multi-Block task partitioning, and per-thread GM/UB access). CATLASS Python added [Flash Attention Infer](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/flash_attention_infer/README.md) (paged KV and causal Mask), [MX Matmul](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/mx_mmad/README.md), and [SIMT Multi-Block](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/simt/multiple_blocks_simt/README.md) / [Multiple Math Operations](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/simt/multi_ops_simt/README.md) examples. CATLASS C++ added [Ascend 950 HSTU Infer](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/83_ascend950_hstu_infer/README.md). CanImplement, Host input checks, and README constraints were aligned to improve diagnostics for invalid inputs.

- [2026/09] [v2.1.0](https://gitcode.com/cann/catlass/releases/v2.1.0) released: upgraded **CATLASS Python** with low-precision computation, compile-time reuse, and distribution support. Added MXFP8/MXFP4 matrix multiplication for Ascend 950 and extended register transfers and logical operations. [`@tla.kernel` supports Constexpr Callable arguments](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/api/host_api_reference.md#constexpr-callable-%E5%85%A5%E5%8F%82), and the [`ascend-catlass-dsl` wheel package](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/quick_start.md) can be built and installed. Added experimental [AutoSync](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/dsl_development/feature_development/auto_sync_design.md) and [`@tla.extern`](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/api/host_api_reference.md) capabilities to simplify pipeline synchronization and reuse of existing Ascend C implementations, along with a [Vector Op Execution Guide](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/examples/end_to_end/vector_ops/README.md) and [DSL control-flow documentation](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/kernel_development/core_concepts/control_flow.md). CATLASS C++ added [Ascend 950 Basic SYRK](https://gitcode.com/cann/catlass/blob/v2.1.0/examples/82_ascend950_basic_syrk/README.md) and [Atlas A2 X-Attention](https://gitcode.com/cann/catlass/blob/v2.1.0/examples/78_x_attention/README.md) examples.

- [2026/08] [v2.0.0](https://gitcode.com/cann/catlass/releases/v2.0.0) released: officially introduced [**CATLASS Python**](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/README.md), supporting Pythonic Kernels and native Tile programming. Built on the MLIR compilation stack and AscendNPU-IR toolchain, it automatically converts Python descriptions into high-performance executable Kernels, reducing compilation time by over 80% compared with CATLASS C++, with simpler code and matching performance. Provides **eight end-to-end examples**, covering matrix multiplication ([Basic Mmad](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/basic_mmad/README.md), [StreamK](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/basic_mmad_streamk/README.md), [Multi Core SplitK](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/multi_core_splitk_matmul/README.md), [Batched](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/batched_matmul/README.md), [Grouped Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/grouped_matmul_slice_m/README.md), and more) and attention ([Flash Attention Infer](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/flash_attention_infer/README.md)). Added the [catlass_cppgen code generation framework](https://gitcode.com/cann/catlass/blob/v2.0.0/python/catlass_cppgen/README.md) for automatic C++ kernel code generation; six **Ascend 950** examples, including [Svd Quant Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/61_ascend950_svd_quant_matmul/README.md) and [A8W4 Grouped MX Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/README.md); and three Atlas A2 examples: [Symm](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/75_symm/README.md), [Trmm](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/76_trmm/README.md), and [Planar Complex Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/77_planar_complex_matmul/README.md).

<details>
<summary>More news</summary>

- [2026/06] [v1.6.0](https://gitcode.com/cann/catlass/releases/v1.6.0) released: added [**MXFP8/MXFP4 quantization templates**](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/53_ascend950_fp8_mx_matmul/README.md), the [**EVG declarative post-processing framework**](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/64_ascend950_matmul_evg/README.md), and [**BlockMmad based on Mutex synchronization primitives**](https://gitcode.com/cann/catlass/blob/v1.6.0/include/catlass/gemm/block/block_mmad_pingpong_mutex_tla.hpp); added [Ascend 950 Tile components](https://gitcode.com/cann/catlass/tree/v1.6.0/include/catlass/gemm/tile/ascend950), a complete set of [unit tests](https://gitcode.com/cann/catlass/blob/v1.6.0/tests/unittest/catlass/gemm/tile/README.md), and the [operator-level test framework (optest)](https://gitcode.com/cann/catlass/blob/v1.6.0/tests/optest/README.md); and added 16 **Ascend 950** operator examples, including [StreamK Matmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/66_ascend950_streamk_matmul/README.md), [Flash Attention Chunk Prefill](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/70_ascend950_flash_attention_chunk_prefill/README.md), and [Conv2d](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/56_ascend950_basic_conv2d_tla/README.md).

- [2026/04] [v1.5.0](https://gitcode.com/cann/catlass/releases/v1.5.0) released: added **Ascend 950** series examples, such as [Basic Matmul](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/43_ascend950_basic_matmul/README.md), [Flash Attention Inference](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/49_ascend950_flash_attention_infer/README.md), and [Per-Group & Per-Block Quant Matmul TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/51_ascend950_quant_matmul_per_group_per_block_tla/README.md); enhanced **TLA** capabilities, including `origin_shape`, `TileView`, and more; and added [103 Dynamic W8A8 Per-Token Quantization](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/103_dynamic_optimized_quant_matmul_per_token_basic/README.md) to the [Matmul Generalization Project](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/102_dynamic_optimized_matmul/README.md).

- [2026/03] The mainline officially started adding support for the next-generation Ascend hardware Ascend 950PR&950DT series products.

- [2026/02] [v1.4.0](https://gitcode.com/cann/catlass/releases/v1.4.0) released, adding examples such as [StreamK Matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/37_streamk_matmul/README.md), [W4A4 Matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/38_w4a4_matmul_per_token_per_channel_dequant/README.md), and [Sparse Matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/41_sparse_matmul_tla/README.md).

- [2025/12] [v1.3.0](https://gitcode.com/cann/catlass/releases/v1.3.0) released, supporting [`FixPipe` inline quantization](https://gitcode.com/cann/catlass/tree/v1.3.0/include/catlass/gemm/tile/tile_copy.hpp#L373), adding multiple templates to the [Matmul Generalization Project](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/README.md), and adding examples such as [INT4 Dequantization](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/32_w4a8_matmul/README.md) and [2D Convolution](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/33_basic_conv2d/README.md).

- [2025/10] [v1.2.0](https://gitcode.com/cann/catlass/releases/v1.2.0) released, adding examples such as [Matmul Operator Generalization](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/102_dynamic_optimized_matmul/README.md).

- [2025/09] The CATLASS template library was officially open sourced.

</details>

See [CHANGELOG](CHANGELOG_en.md) for detailed updates in current and historical versions.

---

## 📌 Introduction

CATLASS (**CA**NN **T**emplates for **L**inear **A**lgebra **S**ubroutine**s**) is a code repository focused on providing base templates for high-performance matrix multiplication operators.  

CATLASS templates matrix operator code through layered abstraction. Therefore, it enables white-box assembly of operator compute logic and makes operator code reusable, replaceable, and partially modifiable. It is designed for Ascend hardware characteristics and supports complex pipeline layouts for operators such as `Flash Attention`. In addition, it shares upper-layer code logic while supporting specialization for differences in underlying hardware.

The template library enables fast development for custom scenarios. It provides performance optimization modules for different scenarios, so developers can assemble and customize them. Under custom shapes, its performance can reach 0.98 to 1.2 times the benchmark performance of the corresponding operator.

<p align="center">
  <img src="docs/assets/images/homepage_perf_mm.png" alt="Matmul Performance Comparison" width="70%">
</p>

<p align="center">
  <img src="docs/assets/images/homepage_perf_gmm.png" alt="GroupedMatmul Performance Comparison" width="90%">
</p>

This repository is the co-created repository for CATLASS. It combines the strengths of the Ascend ecosystem to jointly design and develop operator templates, and provides high-performance implementation code examples for typical operators. For an overview, see [CATLASS Project Introduction](./docs/en/2_Design/00_project_overview.md#introduction-to-catlass). For details on CATLASS Python, see the [CATLASS Python Introduction](./python/tla_dsl/README.md).

## ⚡️ Quick Start

To quickly try CATLASS operator development and usage, see the following content.

- [Quick Start](./docs/en/1_Practice/01_quick_start.md): Quickly get started with the template library, and compile and run existing operator examples.

- [Basic Development Guide](./docs/en/1_Practice/02_host_example_assembly.md): Uses the basic Matmul operator as an example to introduce CATLASS-based operator development practices.

- [Developer Practices](./docs/en/README.md#1-practices): Provides practice examples from writing code at each operator layer to compilation and testing, then to Tiling tuning and operator optimization, from beginner to advanced levels.

- [CATLASS Python Quick Start](./python/tla_dsl/README.md): Entry point for CATLASS Python environment setup, compilation and testing, and API documentation.

## 📚 Advanced References

The following materials can help you further develop and tune CATLASS operators and implement GEMM-class operators with better performance.

- [Innovative Example Development Guide](./docs/en/1_Practice/10_innovative_example_development_guide.md): Covers the complete process from requirements analysis and solution design to component development, testing, and integration.

- [Matrix Multiplication Template Summary](./docs/en/2_Design/01_kernel_design/04_matmul_summary.md): Introduces Matmul template designs and engineering optimizations to help developers tune Matmul performance.

- [CATLASS Design Summary](./docs/en/README.md#2-design): Summarizes documents such as example algorithm design, swizzle strategies, and TLA design in the CATLASS project.

- [CATLASS API](./docs/en/README.md#3-api-documentation): Introduces the layered features of CATLASS and the general matrix multiplication GEMM API.

## 📁 Directory Structure Description

The key directories are as follows. For the detailed directory structure, see [Project Directory](docs/en/2_Design/00_project_overview.md#project-directory).

```text
catlass
├── 3rdparty                     # Third-party deps
├── cmake                        # Build config
├── docs                         # Documentation
├── examples                     # Operator examples
├── include                      # Template headers
├── python                       # Python frameworks
├── scripts                      # Scripts
├── tests                        # Tests
└── tools                        # Tools
```

## 💻 Software and Hardware Requirements

CATLASS depends on the following software and hardware environments:

- Ascend products:
  - Atlas A2 series products
  - Atlas A3 series products
  - Ascend 950PR&950DT series products
- CPU architecture: `aarch64`/`x86_64`
- System: Linux supported by CANN (perform a [compatibility query](https://www.hiascend.com/hardware/compatibility))
- CATLASS C++ software dependencies:
  - `gcc` >= 7.5, < 13.0
  - `cmake` >= 3.16
  - `python` >= 3.10, < 3.14
- For CATLASS Python software dependencies, see the [CATLASS Python README](./python/tla_dsl/README.md).
- If the build fails with environment-related errors such as missing C++ standard headers or libraries, see the [FAQ](./docs/en/FAQ.md) first.

The hardware platforms supported by different CATLASS releases and the required minimum [CANN](https://www.hiascend.com/en/cann/download) versions are shown in the following table:

| CATLASS Version                                                                                                       | Minimum Supported CANN Package Version                                                                                                                                                                                                            | Supported Ascend Products                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Current                                                                                                                  | [8.5.0](https://www.hiascend.com/en/cann/download?versionId=748)<br>[9.0.0](https://www.hiascend.com/en/cann/download?versionId=734) (Ascend 950PR&950DT series products)<br>[9.2.0](https://www.hiascend.com/en/cann/download) (**CATLASS Python** Ascend 950PR&950DT series products)     | Atlas A2 series products and Atlas A3 series products<br>Ascend 950PR&950DT series products |
| [v2.2.0](https://gitcode.com/cann/catlass/releases/v2.2.0) | [8.5.0](https://www.hiascend.com/en/cann/download?versionId=748)<br>[9.0.0](https://www.hiascend.com/en/cann/download?versionId=734) (Ascend 950PR&950DT series products)<br>[9.2.0](https://www.hiascend.com/en/cann/download) (**CATLASS Python** Ascend 950PR&950DT series products)     | Atlas A2 series products and Atlas A3 series products<br>Ascend 950PR&950DT series products |
| [v2.1.0](https://gitcode.com/cann/catlass/releases/v2.1.0)~[v2.0.0](https://gitcode.com/cann/catlass/releases/v2.0.0) | [8.5.0](https://www.hiascend.com/en/cann/download?versionId=748)<br>[9.0.0](https://www.hiascend.com/en/cann/download?versionId=734) (Ascend 950PR&950DT series products)<br>[9.1.0](https://www.hiascend.com/en/cann/download?versionId=796) (**CATLASS Python** Ascend 950PR&950DT series products)     | Atlas A2 series products and Atlas A3 series products<br>Ascend 950PR&950DT series products |
| [v1.6.0](https://gitcode.com/cann/catlass/releases/v1.6.0)                                                            | [8.5.0](https://www.hiascend.com/en/cann/download?versionId=748)<br>[9.0.0](https://www.hiascend.com/en/cann/download?versionId=734) (Ascend 950PR&950DT series products)     | Atlas A2 series products and Atlas A3 series products<br>Ascend 950PR&950DT series products |
| [v1.5.0](https://gitcode.com/cann/catlass/releases/v1.5.0)                                                            | [8.2.RC1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)<br>[9.0.0](https://www.hiascend.com/en/cann/download?versionId=734) (Ascend 950PR&950DT series products) | Atlas A2 series products and Atlas A3 series products<br>Ascend 950PR&950DT series products |

<details>
<summary>Support for Earlier Versions</summary>

| CATLASS Version                                                                                                       | Minimum Supported CANN Package Version                                                                                                                                                                                                            | Supported Ascend Products                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [v1.4.0](https://gitcode.com/cann/catlass/releases/v1.4.0)~[v1.2.2](https://gitcode.com/cann/catlass/releases/v1.2.2) | [8.2.RC1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)                                                                                                                              | [Atlas A2 series products and Atlas A3 series products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)                              |
| [v1.2.1](https://gitcode.com/cann/catlass/releases/v1.2.1)~[v1.0.0](https://gitcode.com/cann/catlass/releases/v1.0.0) | [8.2.RC1.alpha002](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha002)                                                                                                            | [Atlas A2 series products and Atlas A3 series products](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)                              |

</details>

- Development tasks such as accuracy testing with `TorchNPU` require a `cann-ops` package (such as `cann-910b-ops` / `cann-a3-ops` / `cann-950-ops`) to call its operator implementations. For Ascend 950PR&950DT series products, `cann-950-ops` is available only in the official [CANN 9.0.0](https://www.hiascend.com/en/cann/download?versionId=734) release and later versions.
- Use an [official CANN release](https://www.hiascend.com/en/cann/download) for compatibility; beta versions may have compatibility issues.

The following environments have been tested and support building [current CATLASS](https://gitcode.com/cann/catlass):

| System                                  | `CANN`      | `gcc` | `cmake` | `python` |
| --------------------------------------- | ----------- | ----- | ------- | -------- |
| Ubuntu 20.04.5                          | 8.5.0       | 9.3   | 3.16    | 3.10     |
| Ubuntu 22.04.5                          | 8.5.0       | 11.3  | 3.22    | 3.10     |
| openEuler 22.03 SP4                     | 8.5.0       | 10.3  | 3.22    | 3.10     |
| Ubuntu 22.04.5 (Compiling 950 Examples) | 9.0.0 | 11.3  | 3.22    | 3.10     |

## 👥 Collaborators

### [South China University of Technology Professor Lu Lu's Team](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)

### iFLYTEK Research Institute Engineering Group

### [Ocean University of China Professor Shengke Wang's Team](https://it.ouc.edu.cn/wsk2/main.htm)

## 📝 Related Information

- [Contribution Guide](CONTRIBUTING_en.md)
- [Security Statement](SECURITYNOTE_en.md)
- [License](LICENSE)
