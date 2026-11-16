# CATLASS Project Documentation

## 1 Practices

> Step-by-step practices for learning how to use and develop each layer of CATLASS, covering complete operator development, testing, tuning, and model integration.

### Development Workflow

- `01`[Quick Start](./1_Practice/01_quick_start.md): Prepare CATLASS environment, then build and run operator examples
- `02`[Host-side Matmul Assembly](./1_Practice/02_host_example_assembly.md): Learn how to assemble Matmul on the host side
- `03`[Kernel Development](./1_Practice/03_kernel_development.md): Explore Kernel code, including template assembly, Arguments, Params, and key functions
- `04`[Block Mmad Development](./1_Practice/04_block_mmad_development.md): Explore Block Mmad code, template assembly, and major interfaces
- `05`[Block Scheduler Development](./1_Practice/05_block_scheduler_development.md): Explore Block Scheduler code, template assembly, and major interfaces
- `06`[Tile Development](./1_Practice/06_tile_development.md): Explore Tile Copy and Tile Mmad code, template assembly, and major interfaces
- `07`[Epilogue Adaptation](./1_Practice/07_epilogue_adaptation.md): Adapt GEMM epilogues at the Host and Kernel layers, and develop Epilogue Block and Tile components
- `08`[Debugging and Analysis](./1_Practice/08_evaluation.md): Use debugging tools, locate precision issues, and analyze performance bottlenecks
- `09`[Example Contribution Guide](./1_Practice/09_example_contribution_guide.md): Follow the complete process for designing, developing, testing, and merging an example
- `10`[Innovative Example Development Guide](./1_Practice/10_innovative_example_development_guide.md): Follow the complete workflow for developing an innovative example
- `11`[Matmul Optimization Guide](./1_Practice/11_matmul_optimization.md): Improve performance through Tiling parameter tuning and different Dispatch policies
- `12` Example Integration: Adapt examples and integrate them into full networks (contributions welcome)

### Debugging and Analysis

- [Ascend C Dump](./1_Practice/evaluation/ascendc_dump.md): Use Ascend C operator debugging APIs
- [msDebug](./1_Practice/evaluation/msdebug.md): Use msDebug in a CATLASS example project
- [Performance Analysis Tools](./1_Practice/evaluation/performance_tools.md): Use performance analysis tools such as msProf and Profiling
- [Precision Analysis Basics](./1_Practice/evaluation/precision_analysis_basics.md): Learn the fundamentals of precision analysis
- [Precision Issue Debugging](./1_Practice/evaluation/precision_debug.md): Locate precision issues in examples
- [Performance Bottleneck Analysis and Optimization](./1_Practice/evaluation/bottleneck_analysis_and_optimization.md): Analyze and optimize performance bottlenecks

### Specialized Practices

This section contains specialized practice documents contributed by both internal and external developers.

- TLA example refactoring (contributions welcome)
- [Migrating from Atlas A2 to Ascend 950](./1_Practice/others/migration_from_atlasA2_to_Ascend950_guideline.md): Recommended approach for migrating existing Atlas A2 operators to Ascend 950
- [Conv Kernel Development](./1_Practice/others/conv_kernel_development.md): Develop Conv operators
- [Conv Kernel Optimization](./1_Practice/others/conv_kernel_optimization.md): Optimize the performance of Conv operators
- [FA Kernel Optimization](./1_Practice/others/FA_kernel_optimization.md): Optimize the performance of FA operators
- [Fused Operator Optimization](./1_Practice/others/fused_kernel_optimization.md): Explore performance tuning cases for CV fused operators
- [Direct Kernel Invocation](./1_Practice/others/kernel_execution.md): Invoke newly developed operators directly with `<<<>>>`

## 2 Design

### [Project Overview](./2_Design/00_project_overview.md)

Introduces the project positioning, layered modular design, and repository structure.

### Kernel Design

#### Basics

- [Atlas A2 Hardware](./2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md): Overview of the Atlas A2 hardware architecture
- [Atlas A2 GEMM Instruction Set](./2_Design/01_kernel_design/00_basics/atlasA2_gemm_instruction_set.md): Hardware instructions used by Atlas A2 GEMM examples

#### Core Design

- `01`[Example Design](./2_Design/01_kernel_design/01_example_design.md): Summary and index of example design documents in the repository
- `02`[Swizzle Policies](./2_Design/01_kernel_design/02_swizzle.md): Swizzle policies that affect the execution order of basic compute blocks on AI Cores
- `03`[Dispatch Policies](./2_Design/01_kernel_design/03_dispatch_policies.md): DispatchPolicy, an important template parameter of Block Mmad
- `04`[Matrix Multiplication Template Summary](./2_Design/01_kernel_design/04_matmul_summary.md): Summary of Matmul example templates, theoretical templates, engineering optimizations, and usage
- `05`[Adaptive Sliding Window Tiling](./2_Design/01_kernel_design/05_aswt.md): Adaptive sliding window Tiling policy
- `06` Low-precision Topics (contributions welcome)

### TLA Design

- `01`[Layout](./2_Design/02_tla/01_layout.md): TLA Layout structures and related interfaces
- `02`[LayoutTag](./2_Design/02_tla/02_layout_tag.md): Legacy layout tags and interfaces, including RowMajor, ColumnMajor, zN, and nZ
- `03`[Tensor](./2_Design/02_tla/03_tensor.md): Tensor structures

### EVG Design

- `01`[EVG Design Overview](./2_Design/03_evg/01_evg_design.md): EVG positioning, layering, execution model, and graph organization
- `02`[EVG Extension Guide](./2_Design/03_evg/02_evg_extension.md): When to add a ComputeFn or node and the constraints for implementing them
- `03`[EVG Quick Start](./2_Design/03_evg/03_evg_quick_start.md): Basic EVG integration using `Matmul + Add` as an example

## 3 API Documentation

### CATLASS APIs

- [API Index](./3_API/README.md): Entry point for CATLASS API documentation
- [GEMM API](./3_API/gemm_api.md): General matrix multiplication interfaces
- [EVG API](./3_API/evg_api.md): EVG integration, parameter ordering, and commonly used nodes

### Related APIs

- [Ascend C API](https://www.hiascend.com/en/document/redirect/CannCommercialAscendCApi): Ascend C API reference

## 4 Appendix

### FAQ

- [FAQ](./FAQ.md): Frequently asked questions about build, link and runtime environment issues

### Technical Articles

#### Fundamentals

- [C++ Templates Explained](https://www.runoob.com/w3cnote/c-templates-detail.html)
- [Ascend C Operator Development Documentation](https://www.hiascend.com/en/document/redirect/CannCommunityOpdevAscendC)

#### Advanced Topics

Content covering concepts, troubleshooting, performance optimization, and best practices is to be added.

### Training Videos

#### Ascend Community Courses

- [Ascend C Operator Development (Beginner)](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
- [Ascend C Operator Development (Intermediate)](https://www.hiascend.com/developer/courses/detail/1696414606799486977)
- [Ascend C Operator Development (Advanced)](https://www.hiascend.com/developer/courses/detail/1696690858236694530)

#### CATLASS Courses

- [Code Power Special: Mastering the Fundamentals of CATLASS](https://www.bilibili.com/video/BV1f1BDBMES2): The first CATLASS course, introducing the project, operator quick start, development roadmap, and community collaboration
- [Code Power Special: Hands-on CATLASS Operator Development](https://www.bilibili.com/video/BV1DmBhBNEu8): The second CATLASS course, using a basic Matmul operator to explain NPU-based matrix multiplication theory and implementation
- [Code Power Special: Deep Optimization with CATLASS](https://www.bilibili.com/video/BV1FGi9BrEGH): The third CATLASS course, introducing optimization techniques for operator development with CATLASS
