# CATLASS 项目文档

## 1 开发实践

> 代码实践，指导开发者按步骤上手CATLASS各层级代码开发和使用，逐渐具备完整算子开发、测试、调优、模型使用的能力。

### 开发流程

- `01`[快速开始](./1_Practice/01_quick_start.md)：介绍CATLASS的环境准备，以及算子样例的编译和执行
- `02`[Host 侧 Matmul 组装](./1_Practice/02_host_example_assembly.md)：讲解如何在 Host 侧组装 Matmul
- `03`[Kernel 开发](./1_Practice/03_kernel_development.md)：拆解 Kernel 代码，介绍模板组装机制、Arguments、Params 和关键函数
- `04`[Block Mmad 开发](./1_Practice/04_block_mmad_development.md)：拆解 Block Mmad 代码，介绍模板组装机制和主要接口
- `05`[Block Scheduler 开发](./1_Practice/05_block_scheduler_development.md)：拆解 Block Scheduler 代码，介绍模板组装机制和主要接口
- `06`[Tile 开发](./1_Practice/06_tile_development.md)：拆解 Tile Copy 和 Tile Mmad 代码，介绍模板组装机制和主要接口
- `07`[Epilogue 适配](./1_Practice/07_epilogue_adaptation.md)：介绍 GEMM 算子在 Host、Kernel 层的 Epilogue 适配，以及 Epilogue 的 Block、Tile 开发
- `08`[调测与分析](./1_Practice/08_evaluation.md)：介绍调测工具使用、精度问题定位和性能瓶颈分析
- `09`[样例贡献指南](./1_Practice/09_example_contribution_guide.md)：介绍完整样例的设计、开发、测试和合入流程
- `10`[创新样例开发指南](./1_Practice/10_innovative_example_development_guide.md)：介绍创新样例的完整开发流程
- `11`[Matmul 优化指南](./1_Practice/11_matmul_optimization.md)：介绍如何通过 Tiling 调参、应用不同的 Dispatch 策略快速获得性能提升
- `12` 样例集成：介绍样例适配及接入整网的方式（待贡献）

### 调测与分析

- [Ascend C Dump 工具](./1_Practice/evaluation/ascendc_dump.md)：介绍 Ascend C 算子调试接口的使用方法
- [msDebug 工具](./1_Practice/evaluation/msdebug.md)：介绍如何在 CATLASS 样例工程中使用 msDebug
- [性能分析工具](./1_Practice/evaluation/performance_tools.md)：介绍 msProf、Profiling 等性能分析工具
- [精度分析基础](./1_Practice/evaluation/precision_analysis_basics.md)：介绍精度分析的基础知识
- [精度问题定位](./1_Practice/evaluation/precision_debug.md)：介绍样例精度问题的定位方法
- [性能瓶颈分析与优化](./1_Practice/evaluation/bottleneck_analysis_and_optimization.md)：介绍性能瓶颈分析及优化手段

### 专题实践

这里存放内部和外部贡献的专题实践文档。

- TLA 样例改造（待贡献）
- [Atlas A2 到 Ascend 950 迁移指南](./1_Practice/others/migration_from_atlasA2_to_Ascend950_guideline.md)：介绍 Atlas A2 平台存量算子向 Ascend 950 迁移的推荐方案
- [Conv Kernel 开发](./1_Practice/others/conv_kernel_development.md)：介绍 Conv 类算子的开发方法
- [Conv Kernel 优化](./1_Practice/others/conv_kernel_optimization.md)：介绍 Conv 类算子的性能优化方法
- [FA Kernel 优化](./1_Practice/others/FA_kernel_optimization.md)：介绍 FA 类算子的性能优化方法
- [融合算子优化](./1_Practice/others/fused_kernel_optimization.md)：介绍 CV 融合算子的性能调优案例
- [Kernel 直调](./1_Practice/others/kernel_execution.md)：介绍通过 `<<<>>>` 直调新开发算子

## 2 模块设计

### [项目概览](./2_Design/00_project_overview.md)

介绍项目定位、分层模块化设计和代码仓结构。

### Kernel 设计

#### 基础知识

- [Atlas A2 硬件信息](./2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md)：介绍 Atlas A2 的硬件架构
- [Atlas A2 GEMM 指令集](./2_Design/01_kernel_design/00_basics/atlasA2_gemm_instruction_set.md)：介绍 Atlas A2 GEMM 类样例涉及的硬件指令集

#### 核心设计

- `01`[样例设计](./2_Design/01_kernel_design/01_example_design.md)：汇总和索引仓库中的样例设计文档
- `02`[Swizzle 策略](./2_Design/01_kernel_design/02_swizzle.md)：介绍影响 AI Core 计算基本块执行顺序的 Swizzle 策略
- `03`[Dispatch 策略](./2_Design/01_kernel_design/03_dispatch_policies.md)：介绍 Block Mmad 的重要模板参数 DispatchPolicy
- `04`[矩阵乘模板总结](./2_Design/01_kernel_design/04_matmul_summary.md)：汇总 Matmul 样例模板、理论模板、工程优化和应用方式
- `05`[自适应滑窗 Tiling](./2_Design/01_kernel_design/05_aswt.md)：介绍自适应滑窗 Tiling 策略
- `06` 低精度专题（待贡献）

### TLA 设计

- `01`[Layout](./2_Design/02_tla/01_layout.md)：介绍 TLA 的 Layout 结构和相关接口
- `02`[LayoutTag](./2_Design/02_tla/02_layout_tag.md)：介绍 RowMajor、ColumnMajor、zN、nZ 等旧版布局标签和接口
- `03`[Tensor](./2_Design/02_tla/03_tensor.md)：介绍 Tensor 结构

### EVG 设计

- `01`[EVG 设计概览](./2_Design/03_evg/01_evg_design.md)：介绍 EVG 的定位、分层关系、执行模型和图组织方式
- `02`[EVG 扩展规范](./2_Design/03_evg/02_evg_extension.md)：说明何时增加 ComputeFn、何时增加节点及其实现约束
- `03`[EVG 快速开始](./2_Design/03_evg/03_evg_quick_start.md)：以 `Matmul + Add` 为例介绍 EVG 的基础接入流程

## 3 API 文档

### CATLASS API

- [API 清单](./3_API/README.md)：CATLASS API 文档入口
- [GEMM API](./3_API/gemm_api.md)：介绍通用矩阵乘法接口
- [EVG API](./3_API/evg_api.md)：介绍 EVG 的接入方式、参数顺序和常用节点

### 相关 API

- [Ascend C API](https://www.hiascend.com/document/redirect/CannCommercialAscendCApi)：昇腾社区Ascend C API列表

## 4 附录

### 常见问题

- [常见问题（FAQ）](./FAQ.md)：汇总编译、链接与运行阶段的环境类问题及处理方式

### 技术文章

#### 基础入门

- [C++ Template 详解](https://www.runoob.com/w3cnote/c-templates-detail.html)
- [Ascend C 算子开发文档](https://www.hiascend.com/document/redirect/CannCommunityOpdevAscendC)

#### 进阶主题

概念理解、问题定位、性能优化和优秀实践等内容待补充。

### 培训视频

#### 昇腾社区在线课程

- [Ascend C 算子开发（入门）](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
- [Ascend C 算子开发（进阶）](https://www.hiascend.com/developer/courses/detail/1696414606799486977)
- [Ascend C 算子开发（高级）](https://www.hiascend.com/developer/courses/detail/1696690858236694530)

#### CATLASS 专题课程

- [【码力全开特辑】一站式掌握CATLASS基本概念](https://www.bilibili.com/video/BV1f1BDBMES2)：CATLASS 学习系列课程第一讲，介绍 CATLASS 整体情况、算子快速上手、发展全景和生态共建
- [【码力全开特辑】CATLASS算子开发初体验](https://www.bilibili.com/video/BV1DmBhBNEu8)：CATLASS 学习系列课程第二讲，以基础 Matmul 算子为例介绍基于 NPU 的矩阵乘理论建模和代码实现
- [【码力全开特辑】CATLASS深度优化](https://www.bilibili.com/video/BV1FGi9BrEGH): CATLASS 学习系列课程第三讲，介绍使用 CATLASS 开发算子时的优化技巧
