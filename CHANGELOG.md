# CHANGELOG

## CATLASS 2.X

### CATLASS 2.2.0

#### CATLASS Python

- 关键特性
  - **SIMD**：面向 Ascend 950 扩展 SIMD OP 及配套搬运与矩阵计算能力：补齐对齐连续、对齐非连续（DataBlock 跨步）、非对齐连续三类场景下 Load/Store 的 `PostModeUpdate` 模式，搬运完成后按步长自动更新地址；支持 Histogram 的 `FREQUENCY` 与 `ACCUMULATE` 两种模式；L0C→GM 与 L0C→UB 搬出均支持随路 ReLU（`relu_enable`）；补齐 PER_TENSOR / PER_CHANNEL 量化及配套 scale 搬入、`tla.gather` Reg→Reg、无符号整数相关搬运/Compare/Cast，以及 L1→BiasTable 与普通 Matmul 带 Bias。
  - **SIMT**：正式上线 SIMT 编程能力：通过 `tla.vec.func(mode="simt", thread_block_dim=...)` 描述逐线程标量计算，支持线程坐标、跨 Block 标量捕获与任务划分、GM/UB 逐线程访问，以及算术、比较、选择、类型转换与浮点数学等线程级运算，并可与 Cube/SIMD 同 Kernel 混用。
  - **框架能力**：支持以类实例组织 Kernel：类内可封装 `tla.kernel` 并访问成员变量，`tla.compile` 可编译类实例或其绑定方法，实例属性作为编译期常量参与编排与缓存。
- 更多样例（新增 Ascend 950 Python 样例）
  - [Flash Attention Infer](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/flash_attention_infer/README.md)：在既有 nomask 路径上完善 KV 分页（块表寻址）与独立 causal mask kernel，覆盖 FP16/BF16、BSND/TND、GQA、页边界与尾块场景。
  - [MX Matmul](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/mx_mmad/README.md)：演示 MX FP8/FP4 数据准备、E8M0 缩放块、GM→L1→L0 带 scale 搬运、`tla.mmad_mx` 与 FP32 结果回写。
  - [SIMT 多 Block 协作](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/simt/multiple_blocks_simt/README.md)：通过 3D 线程坐标线性化与全局步长，将任务分配至多个 Block，覆盖不同 Block 数、线程配置及非整除长度。
  - [SIMT 多数学运算](https://gitcode.com/cann/catlass/blob/v2.2.0/python/tla_dsl/examples/end_to_end/simt/multi_ops_simt/README.md)：提供浮点与整数两条运算路径，覆盖算术、比较、`where`、类型转换以及浮点数学函数。
- Bugfix&优化
  - 将 HIVMC 编译默认注入的 `cce-vf-auto-sync=global` 统一覆盖为 `fused`：`global` 由 CCEC 分析并插入同步，`fused` 由开发者自行保证同步正确性；覆盖后可避免 `global` 在 VF 环内插入多余同步屏障。

#### CATLASS C++

- 更多样例（新增 Ascend 950 样例）
  - [Ascend 950 HSTU Infer](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/83_ascend950_hstu_infer/README.md)：实现 `S=QKᵀ`、`P=siluScale×S×sigmoid(S)`、`O=PV` 计算链路，支持可选 causal 乘法掩码，覆盖 NTD/TND 布局、变长序列与 Paged KV Cache。
- 文档资料
  - 对齐 CanImplement、Host 输入检查与 README 约束：建立输入支持矩阵，使数据类型、布局、形状、对齐、分页和 Mask 约束与实现一致；对非法输入在 Kernel 启动前给出明确诊断，并补齐合法边界与非法输入回归用例（首批覆盖含 Basic Matmul、Grouped Matmul、Optimized Matmul、Quant 系列、Ascend 950 相关样例、X-Attention、Basic SYRK 等）。
- Bugfix&优化
  - 优化 [Broadcast Matmul PerBlock Quant](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/62_ascend950_broadcast_matmul_perblock_quant/README.md) 与 [A8W4 Grouped MX Matmul](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/README.md) 的 golden 生成：E4M3 LUT 改为分块向量化 + 二分最近邻，消除 FP4 索引重复重算；产物逐字节一致，Broadcast Matmul PerBlock Quant 单用例耗时由约 37 分钟降至约数秒量级。
  - 修复 [MLA](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/19_mla/README.md) 以加代乘原子重缩放在零值、非规格数与指数下溢边界的精度问题，并补齐 UB 复用同步、移除可能误清合法大数的极值清零逻辑。
  - 修复 Conv3d/Conv2d 输出尺寸计算在 `uint32` 下的无符号下溢，非法 shape（含 `stride=0`）在 CanImplement / Host 入口拦截。
  - 修复 [Ascend 950 Flash Attention Chunk Prefill](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/70_ascend950_flash_attention_chunk_prefill/README.md) 在 `kvHeads != 1` 时的精度问题，并补充约束与 README。
  - 修复 [A8W4 MX Matmul](https://gitcode.com/cann/catlass/blob/v2.2.0/examples/59_ascend950_a8w4_mx_matmul/README.md) optest 与 example 对齐问题。
  - 修复 catlass_cppgen EVG 模板遗漏 Bias 入参的问题。

### CATLASS 2.1.0

#### CATLASS Python

- 关键特性
  - 面向 Ascend 950 扩展低精度计算与寄存器编程能力：
    - 支持 MXFP8/MXFP4 矩阵乘、E8M0 缩放块及 MX 布局，新增 `copy(..., scale=...)` 和 `mmad_mx` 接口。
    - 扩展寄存器数据搬运能力，支持连续/非连续对齐搬入、多种 LoadDist 模式，以及单元素和双寄存器交织搬出。
    - 新增带 Mask 的寄存器向量/标量移位操作，支持有符号右移。
  - 新增[`@tla.extern`](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/api/host_api_reference.md)声明API：支持在 DSL Kernel 中声明和调用用户提供的 Ascend C 函数，复用已有底层实现。
  - 新增[AutoSync](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/dsl_development/feature_development/auto_sync_design.md)：通过 `@tla.kernel(auto_sync="v0")` 自动分析片上内存依赖并处理单个AIC或AIV内部各流水之间的核内同步：
  - 增强编译期复用能力：[`@tla.kernel` 支持 Constexpr Callable 入参](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/api/host_api_reference.md#constexpr-callable-%E5%85%A5%E5%8F%82)，可将后处理、Mask 和激活等逻辑作为编译期策略复用，并纳入特化与编译缓存。
  - 完善 DSL 发布交付能力，支持从源码构建并安装 [`ascend-catlass-dsl` Wheel 包](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/quick_start.md)。
- 文档资料
  - 新增 [Vector Op 执行指南](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/examples/end_to_end/vector_ops/README.md)，覆盖二元、单目、比较、搬运、归约及寄存器控制流等典型操作的编译、运行与结果验证。
  - 新增 [Host API 参考](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/api/host_api_reference.md)，补充 [DSL 控制流](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/kernel_development/core_concepts/control_flow.md)和 [AutoSync 设计与使用约束](https://gitcode.com/cann/catlass/blob/v2.1.0/python/tla_dsl/docs/zh/dsl_development/feature_development/auto_sync_design.md)文档。
- Bugfix&优化
  - 优化 DSL JIT 执行与动态 Tensor 参数封装，重复启动复用已加载的 Binary 和 Function Handle，动态 Tensor 稳态启动开销由 28 μs 降至 15 μs。

#### CATLASS C++

- 更多样例（新增 Ascend 950 样例 1 个，Atlas A2系列产品、Atlas A3系列产品样例 1 个）
  - [Ascend 950 Basic SYRK](https://gitcode.com/cann/catlass/blob/v2.1.0/examples/82_ascend950_basic_syrk/README.md)：支持 `Y = X·Xᵀ` 对称秩更新，仅计算下三角分块，并通过一次计算完成对称位置写回。
  - [Atlas A2 X-Attention](https://gitcode.com/cann/catlass/blob/v2.1.0/examples/78_x_attention/README.md)：提供数据生成、Tiling、Kernel、Host 和 PyTorch 参考实现，融合 shared/unshared KV 注意力并支持两种缓存组合。
- Bugfix&优化
  - 修复 Basic Conv2D 大 padding 与输出通道非对齐场景的越界和精度问题。
  - 修复 MLA Tiling 在部分核数和 Shape 下选择错误、尾段任务未写回的问题。
  - 修复 BlockMmad Shuffle-K 边界除零和 Batched Matmul 大尺寸下的 stride 溢出风险。

### CATLASS 2.0.0

#### CATLASS Python

- 正式上线**CATLASS Python**支持Pythonic Kernel，原生支持Tile编程。在 AscendNPU-IR 的基础上构建 TLA Dialect，将 Python 中描述的 TLA 操作降级为 MLIR，并由 AscendNPU-IR 工具链编译为可执行的核函数产物。
  - 覆盖矩阵乘、向量计算、动态控制流等完整算子表达能力，`tla.print` 支持`GM/UB`上的格式化标量输出及动态控制流内 tensor/scalar 打印
  - Host 侧提供统一的 tensor 构造入口 `from_dlpack` / `make_fake_tensor`，支持 `@dataclass` 结构体作为 kernel 入参
- 样例
  - [Basic Mmad](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/basic_mmad/README.md)：基础矩阵乘示例，basic_matmul、basic_matmul_atomic_add、basic_matmul_mutex、basic_matmul_mutex_with、basic_mmad_ptr（mutex / atomic_add / ptr 多路径）
  - [Basic Mixed](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/basic_mixed/README.md)：基础CV算子示例，basic_mixed、basic_mixed_mutex、basic_mixed_fixpipe_nz2dn、basic_mixed_store_zN、basic_mixed_store_zNUnAlign、basic_mixed_ub2l1（混合 copy+mmad 场景）
  - [Matmul Epilogue](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/basic_mmad_epilogue/README.md)：后处理类CV融合样例，matmul_add、matmul_add_ub、matmul_bias、matmul_leaky_relu、matmul_sigmoid、matmul_silu、matmul_tanh
  - [StreamK Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/basic_mmad_streamk/README.md)：StreamK 调度矩阵乘，normal tile 全 K 计算 + 尾轮 tile 多核切 K 归约
  - [Multi Core SplitK Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/multi_core_splitk_matmul/README.md)：多核切 K 矩阵乘，multi_core_splitk_matmul、tail_multi_core_splitk_matmul
  - [Batched Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/batched_matmul/README.md)：批量矩阵乘
  - [Grouped Matmul Slice M](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/grouped_matmul_slice_m/README.md)：按 M 轴切分的分组矩阵乘
  - [Flash Attention Infer](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/examples/end_to_end/flash_attention_infer/README.md)：推理Flash Attention样例demo，支持非Mask非Paged场景，支持对齐/非对齐输入
- 工具支持
  - 构建体系：基于 AscendNPU-IR 子模块构建，提供 Docker 开发镜像，支持 CANN >= 9.1.0
- 文档资料
  - 提供 [CATLASS Python 文档集](https://gitcode.com/cann/catlass/tree/v2.0.0/python/tla_dsl/docs/)，包含 [Kernel API Reference](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/docs/zh/kernel_api_reference.md)、[DSL Python 语法指南](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/docs/zh/dsl_python_syntax_guide.md)、[BC 后端集成文档](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/docs/zh/bc-backend-integration.md)、[环境变量说明](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/docs/zh/environment_variables.md)及[环境搭建指南](https://gitcode.com/cann/catlass/blob/v2.0.0/python/tla_dsl/docs/zh/dev_guide/00_environment_setup.md)等

#### CATLASS C++

- 关键特性
  - 新增 [CATLASS官方文档站](https://catlass.readthedocs.io/zh-cn/latest/)，为开发者提供一站式学习与开发体验
  - 新增 [catlass_cppgen 代码生成框架](https://gitcode.com/cann/catlass/blob/v2.0.0/python/catlass_cppgen/README.md)：基于 Python 的算子代码生成框架，支持定义算子参数、选择优化策略并自动生成 C++ 核函数代码，覆盖 GEMM 与 GroupGEMM 类算子，配套 EVG、OpTensor 等 API 文档与单元测试
- 更多样例（新增样例 9 个，其中 Ascend 950 样例 6 个、Atlas A2 样例 3 个）
  - [Ascend 950 Svd Quant Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/61_ascend950_svd_quant_matmul/README.md)：SVD 低秩分解量化，主通路 W4A4 量化 + 低秩旁路高精度计算
  - [Ascend 950 Dual-Level Quant MX BatchMatmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/63_ascend950_dual_level_quant_mx_batch_matmul/README.md)：二级量化 + MXFP4 BatchMatmul
  - [Ascend 950 Grouped MXFP8 Matmul SwiGLU MX Quant](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant/README.md)：Grouped MXFP8 矩阵乘 + SwiGLU + MXFP8 量化
  - [Ascend 950 Grouped MXFP8 Matmul Finalize Routing](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/71_ascend950_fp8_mx_grouped_matmul_finalize_routing/README.md)：Grouped MXFP8 矩阵乘 + Finalize Routing 融合，提供确定性与非确定性两个版本
  - [Ascend 950 A8W4 Grouped MX Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/README.md)：伪量化场景的 A8W4 Grouped MX 矩阵乘
  - [Ascend 950 Grouped Matmul Slice M Gelu](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/80_ascend950_grouped_matmul_slice_m_gelu/README.md)：Grouped Matmul SliceM + Gelu 融合
  - [Atlas A2 Symm](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/75_symm/README.md)：对称矩阵乘
  - [Atlas A2 Trmm](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/76_trmm/README.md)：三角矩阵乘
  - [Atlas A2 Planar Complex Matmul](https://gitcode.com/cann/catlass/blob/v2.0.0/examples/77_planar_complex_matmul/README.md)：平面复数矩阵乘，输出实部与虚部两路结果
- 工具支持
  - [optest](https://gitcode.com/cann/catlass/blob/v2.0.0/tests/optest/README.md) 接口扩展至 72 个，新增 Symm、Trmm、PlanarComplexMatmul、GroupedMatmulSliceMGelu、A8W4GroupedMxMatmul 等 10 个接口
  - Agent Skill 新增 [CANNBot CATLASS C++](https://gitcode.com/cann/cannbot/blob/master/plugins-official/catlass-op-generator/quickstart.md)、CANNBot CATLASS Python 算子自动生成工具链，及[A2/A3 向 Ascend 950 迁移工作流](https://gitcode.com/cann/catlass/tree/v2.0.0/.agents/skills/catlass-ascend950-migration)和[Tile 层级单元测试编写技能](https://gitcode.com/cann/catlass/tree/v2.0.0/.agents/skills/catlass-tile-level-ut)
- 文档资料
  - 更新 README 中 gcc 版本约束说明
  - 更新 Gemm/Tile UT 依赖说明
- Bugfix&优化
  - MLA 算子优化：加入 `AMLATp1Spec` 特化Kernel模板，采取KV-split 均衡分核策略，并修复golden类型读取错误问题
  - 修复 Ascend 950 A8W4 MxMatmul 样例在非对齐场景下的精度问题
  - 修复 Ascend 950 MxGroupedMatmulSliceM 样例在 MxFp4 类型和transB场景下的精度问题
  - 修复 optest下的Quant Optimized Matmul Tla 在非对齐 shape 下 JIT 编译失败的问题

## CATLASS 1.X

### CATLASS 1.6.0

- 关键特性
  - 全面支持 **Ascend 950**，新增全套 [Tile层组件](https://gitcode.com/cann/catlass/tree/v1.6.0/include/catlass/gemm/tile/ascend950)（文档见[Gemm/Tile类模板概述](https://gitcode.com/cann/catlass/blob/v1.6.0/docs/zh/3_API/include/catlass/gemm/tile/README.md)），兼容 Atlas A2系列产品、Atlas A3系列产品，并新增对 **Atlas A2/Atlas A3** 及 **Ascend 950** 架构下Gemm类全量Tile层组件的[**单元测试**](https://gitcode.com/cann/catlass/blob/v1.6.0/tests/unittest/catlass/gemm/tile/README.md)
  - 新增 [**MXFP8 / MXFP4 量化模板体系**](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/53_ascend950_fp8_mx_matmul/README.md)，基于Ascend 950架构支持 MXFP8/MXFP4 量化矩阵乘、A8W4 MX矩阵乘等样例
  - 新增 [**基于Mutex 同步原语的 BlockMmad**](https://gitcode.com/cann/catlass/blob/v1.6.0/include/catlass/gemm/block/block_mmad_pingpong_mutex_tla.hpp)，提供更简化的block组件代码实现
  - 新增 [**EVG 声明式后处理框架**](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/64_ascend950_matmul_evg/README.md)，支持以声明式方式描述 Epilogue 后处理逻辑
  - 支持使用 AscendC CMake 构建系统编译
- 更多样例 （新增Ascend 950 样例16个，Atlas A2系列产品、Atlas A3系列产品样例1个）
  - [Ascend 950 Flash Attention Chunk Prefill](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/70_ascend950_flash_attention_chunk_prefill/README.md)
  - [Ascend 950 MXFP8 Matmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/53_ascend950_fp8_mx_matmul/README.md)
  - [Ascend 950 MXFP4 Matmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/54_ascend950_fp4_mx_matmul/README.md)
  - [Ascend 950 A8W4 MX Matmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/59_ascend950_a8w4_mx_matmul/README.md)
  - [Ascend 950 MXFP8 BatchMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/58_ascend950_fp8_mx_batch_matmul/README.md)
  - [Ascend 950 MXFP8/MXFP4 GroupedMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/55_ascend950_mx_grouped_matmul_slice_m/README.md)
  - [Ascend 950 Matmul Full Dequant](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/57_ascend950_matmul_full_dequant/README.md)
  - [Ascend 950 Broadcast Matmul PerBlock Quant](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/62_ascend950_broadcast_matmul_perblock_quant/README.md)
  - [Ascend 950 matmul Evg](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/64_ascend950_matmul_evg/README.md)
  - [Ascend 950 GroupedMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/60_ascend950_grouped_matmul_slice_m/README.md)
  - [Ascend 950 BatchedMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/67_ascend950_batched_matmul/README.md)
  - [Ascend 950 Matmul FullLoadA](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/73_ascend950_matmul_full_loadA/README.md)
  - [Ascend 950 StreamK Matmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/66_ascend950_streamk_matmul/README.md)
  - [Ascend 950 MultiCoreSplitkMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/68_ascend950_multi_core_splitk_matmul/README.md)
  - [Ascend 950 TailMultiCoreSplitkMatmul](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/69_ascend950_tail_multi_core_splitk_matmul/README.md)
  - [Ascend 950 Conv2d](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/56_ascend950_basic_conv2d_tla/README.md)
  - [Atlas A2 Quant Matmul 多核切K](https://gitcode.com/cann/catlass/blob/v1.6.0/examples/52_quant_multi_core_splitk_matmul_tla/README.md)
- 工具支持
  - 新增 [算子级测试框架（optest）](https://gitcode.com/cann/catlass/blob/v1.6.0/tests/optest/README.md)，支持 JIT 与 Prebuilt 两种模式，使用弱符号机制统一 prebuilt kernel 加载，并完成 63 个存量及新增样例的 torch_catlass 测试接口接入，覆盖 Matmul、GroupedMatmul、Flash Attention等算子类型
  - [msTuner 工具增强](https://gitcode.com/cann/catlass/blob/v1.6.0/tools/tuner/README.md)：支持 -xasc 编译及 Ascend 950 架构寻优
  - 新增 [Agent Skill 开发工具](https://gitcode.com/cann/catlass/tree/v1.6.0/.agents)，当前具备支持生成样例torch接口、生成样例optest交付件等场景的自动化功能
  - 新增 CI/构建辅助设施：pre-commit 脚本、依赖描述文件等
- 文档资料
  - 新增 [EVG 声明式后处理框架文档](https://gitcode.com/cann/catlass/tree/v1.6.0/docs/zh/2_Design/03_evg/01_evg_design.md)
  - 新增 [存量非 TLA 算子向 Ascend 950 平台迁移文档](https://gitcode.com/cann/catlass/tree/v1.6.0/docs/zh/1_Practice/others/migration_from_atlasA2_to_Ascend950_guideline.md)
  - [tile 相关 API 文档](https://gitcode.com/cann/catlass/tree/v1.6.0/docs/zh/3_API/README.md) 补充完善
  - 文档新增中英文目录，完成多轮纠错整改及跳转链接修复
- Bugfix&优化
  - 将原本的fftsAddr替换为hardwareSyncAddr，使用aclrtGetHardwareSyncAddr接口获取
  - 修复 EpilogueAtlasA2PerTokenDequant 同步缺少导致的 RAW 内存竞争问题
  - 修复 quant_matmul TLA 版本针对 zN 输入的 Bug
  - 修复 RemovePaddingNDAndCast 中 compute length 可能不对齐的问题
  - 修复部分样例的 uint32/int32 溢出隐患

### CATLASS 1.5.0

- 关键特性
  - 新增支持 **Ascend 950** 架构与配套底层模板组件
  - **TLA** 增强：引入 `origin_shape`，新增 `TileView`、`MakeTensorLike`、Tensor `operator()` 等接口，并完善布局与张量表达
  - **Matmul 泛化工程**扩展：支持 **W8A8 Per-Token + Per-Channel** 动态量化路径、**分批编译**，并补充相关设计文档
  - **FixPipe** 能力延伸：新增/完善 Matmul FixPipe 优化与 **GMM + FixPipe + Dequant** 等组合模板与样例
  - 适配 **CANN 9.0.0.beta2**；在使用 **g++** 与毕昇工具链链接时，需显式链接 **profapi**（编译器非兼容性变更说明）
  - 新增 **单元测试**（unittest）与 **CI** 对 Ascend 950 的适配
- 更多样例
  - [Ascend 950 基础 Matmul](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/43_ascend950_basic_matmul/README.md)
  - [Ascend 950 Matmul FixPipe 优化](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/46_ascend950_matmul_fixpipe_opti/README.md)
  - [Ascend 950 Grouped Matmul SliceM Per-Token Dequant](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/47_ascend950_grouped_matmul_slice_m_per_token_dequant/README.md)
  - [Ascend 950 Grouped Matmul Per-Tensor & Per-Channel Dequant](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant/README.md)
  - [Ascend 950 Flash Attention 推理](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/49_ascend950_flash_attention_infer/README.md)
  - [Ascend 950 基础 Matmul GEMV](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/50_ascend950_basic_matmul_gemv/README.md)
  - [Ascend 950 Quant Matmul Per-Group & Per-Block TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/51_ascend950_quant_matmul_per_group_per_block_tla/README.md)（Per-Group*Per-Block 量化组合）
  - [Quant Optimized Matmul TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/42_quant_optimized_matmul_tla/README.md)
  - [Quant Matmul Full LoadA TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/44_quant_matmul_full_loadA_tla/README.md)
  - [Strided Batched Matmul TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/45_strided_batched_matmul_tla/README.md)
  - [Matmul 泛化工程：动态 W8A8 Per-Token 量化](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/103_dynamic_optimized_quant_matmul_per_token_basic/README.md)
- 工具支持
  - [shared\_lib](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/shared_lib/README.md) 输出产物增加 **soname**，Python 扩展依赖的共享库切换为带版本信息的 `.so`
  - [Python 扩展](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/python_extension/README.md) 支持 `build.sh` 编译选项传入，并支持异步模式；更新设备侧取指针等接入方式
  - [msTuner\_CATLASS](https://gitcode.com/cann/catlass/tree/v1.5.0/tools/tuner/README.md) 扩展 GEMM 配置与搜索空间；[MatmulGelu](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/27_matmul_gelu/README.md) 样例接入寻优示例
- 文档资料
  - 文档目录与资源路径调整（如 **figures** 目录），并做纠错与内容修订
  - 泛化工程补充 **MultiCoreSplitK**、**StreamK**、**单核切 K** 等相关说明文档
  - 修复 [ascendc\_dump 文档](https://gitcode.com/cann/catlass/blob/v1.5.0/docs/1_Practice/evaluation_tools/ascendc_dump.md) 中的错误表述
- Bugfix&优化
  - 修复 TLA **OriginShape** 与 **Flash Attention Golden** 等相关问题；完善 Ascend 950 FA 的 Block/Epilogue 等实现路径
  - 调整 [CopyGmToL1](https://gitcode.com/cann/catlass/blob/v1.5.0/include/catlass/gemm/tile/copy_gm_to_l1.hpp) 中 `blockLen` 计算逻辑，无需再为对齐 **C0\_NUM\_PER\_FRACTAL** 做不必要向上取整
  - **Nan 专项**：在 `exp11` 等路径为 **Ki=0** 场景补充清零，避免脏数据影响模型精度
  - 修复 **CopyL0CToDstQuantMode** 等问题；Ascend 950 架构标识由 **3501** 更正为 **3510**
  - 修复间接头文件引用、License 注释与多处文档笔误；持续消除代码规范告警与风格清理

### CATLASS 1.4.0

- 关键特性
  - [Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.4.0/examples/102_dynamic_optimized_matmul/README.md)新增
    - `LocalPaddingCPaddingCommonMatmul`模板，使用局部workSpace对C矩阵做padding
- 更多样例
  - [StreamK Matmul算子](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/37_streamk_matmul/README.md)
  - [W4A4低精度Matmul算子](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/38_w4a4_matmul_per_token_per_channel_dequant/README.md)
  - [Matmul算子L2层级切分+错位分核](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/39_big_matmul_tla/README.md)
  - [Sparse Matmul算子](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/41_sparse_matmul_tla/README.md)
- 工具支持
  - 增加[shared\_lib](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/shared_lib/basic_matmul_shared_lib.cpp)使用示例
- 文档资料
  - 新增[单核切K优化Matmul算子的详设文档](https://gitcode.com/cann/catlass/blob/v1.4.0/docs/contents/example_design/34_single_splitk_matmul.md)，介绍单核切K矩阵乘的设计思路和代码拆解
  - 新增[主页Matmul/GroupedMatmul算子性能展示数据](https://gitcode.com/cann/catlass/blob/v1.4.0/README.md)
  - [msdebug文档](https://gitcode.com/cann/catlass/blob/v1.4.0/docs/tools/msdebug.md)新增驱动支持检查
- Bugfix&优化
  - 优化[grouped\_matmul\_slice\_m](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/02_grouped_matmul_slice_m/README.md)样例支持`groupList`分段式输入
  - BlockMmad增加`TileShape`的32B对齐约束
  - 修复[w4a8 matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/32_w4a8_matmul/w4a8.cpp)样例构造数据长度问题
  - [Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.4.0/examples/102_dynamic_optimized_matmul/README.md)支持Stride大于Shape的场景
  - [msTuner\_CATLASS工具](https://gitcode.com/cann/catlass/tree/v1.4.0/tools/tuner/README.md)支持`quant Matmul`做tiling寻优
  - 修复device侧对`cmath`函数的不规范使用
  - 修复[MatmulSilu](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/28_matmul_silu/README.md)精度问题和命名错误
  - 修复[cast\_int4\_to\_int8](https://gitcode.com/cann/catlass/blob/v1.4.0/include/catlass/gemm/tile/cast_int4_to_int8.hpp)组件的Vector同步问题
  - 修复[w8a16 Matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/30_w8a16_matmul/w8a16_matmul.cpp)算子half类型使用问题
  - 修复[单核切K Matmul算子](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/34_single_core_splitk_matmul/single_core_splitk.cpp)在`RemovePaddingNDAndCastC`为空时的逻辑问题

### CATLASS 1.3.0

- 关键特性
  - 将`CMake`最低版本要求从3.22降至3.16
  - 支持[`FixPipe`随路量化](https://gitcode.com/cann/catlass/tree/v1.3.0/include/catlass/gemm/tile/tile_copy.hpp#L373)
  - [Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/README.md)新增
    - `PaddingCommonMatmul`
    - `SmallMatmul`
    - `PaddingMultiCoreSplitkMatmul`
    - `PaddingStreamkMatmul`
    - `单核切K系列模板`
    - `动态Swizzle`
- 更多样例
  - [INT4类型反量化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/32_w4a8_matmul/README.md)
  - [2D卷积算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/33_basic_conv2d/README.md)
  - [单核切K优化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/34_single_core_splitk_matmul/README.md)
- 工具支持
  - 新增[msOpGen](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0018.html)工具代码示例[basic_matmul](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/advanced/basic_matmul_aclnn/basic_matmul_aclnn.cpp)和[接入文档](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/advanced/basic_matmul_aclnn/README.md)
  - [msTuner_CATLASS工具](https://gitcode.com/cann/catlass/tree/v1.3.0/tools/tuner/README.md)新增
    - `GroupedMatmulSliceM算子`
    - `OptimizedMatmul算子`
- 文档资料
  - 新增[INT8类型反量化GroupedMatmul算子的详设文档](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/example_design/10_grouped_matmul_slice_m_per_token_dequant.md)，介绍`groupMatmul+后处理`类型的算子的设计思路和代码拆解
  - 新增[矩阵乘模板总结文档](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/matmul_template_summary.md)，介绍模板库已有的Matmul模板设计
  - 新增[CommonMatmul说明文档](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/doc/CommonMatmul.md)，介绍泛化Matmul工程中的基础模板
- Bugfix&优化
  - 修复[Flash Attention推理算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/23_flash_attention_infer/README.md)在softmax拷贝mask时引入的内存问题
  - 修复文档错误
    - [catlass_optimize_guidance.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/catlass_optimize_guidance.md)
    - [api.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/api.md)
    - [quickstart.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/quickstart.md)
    - [tutorials.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/tutorials.md)
  - [Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/README.md)更新
    - 修改`TilingParams`读取方式增强可读性
    - 优化原有的`Splitk ReduceAdd`，UB空间利用更充分
    - 新增`CMakeLists.txt`中对python环境的判断
  - 修复[OptimizedMatmul算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/06_optimized_matmul/README.md)在kernel里没有支持PADDING_NZ的问题
  - 优化重构[FP8类型反量化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/29_a2_fp8_e4m3_matmul/README.md)，使其更符合Prologue范式
  - 修复[MatmulBias算子](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/20_matmul_bias/README.md)精度问题并增加对bf16的校验拦截
  - 优化仿真的编译逻辑以及在A3环境下的编译问题，现在编译`simulator`模式时逻辑与上板模式相同
  - [msTuner_CATLASS工具](https://gitcode.com/cann/catlass/tree/v1.3.0/tools/tuner/README.md)更新
    - 新增接口替换、非法字符、`groupCount`最大值检查等安全校验
    - 修复下发部分算子时默认传入`ffts_addr`被拦截的问题
  - 更改默认的跨核标志位可连续置位次数，避免超过次数后引发的系统卡死问题

### CATLASS 1.2.0

- 关键特性
  - 算子编译时支持传入计算平台架构
  - 新增[Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/102_dynamic_optimized_matmul/README.md)示例
    - 自动依照特征尺寸确定Tiling参数
    - 可在预设的算子模板中择优选取

- 更多样例
  - [Flash Attention推理算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/23_flash_attention_infer/README.md)
  - [3D卷积算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/24_conv_bias/README.md)
  - [A矩阵全加载Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/25_matmul_full_loadA/README.md)
  - [小矩阵优化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/31_small_matmul/README.md)
  - [MatmulRelu算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/26_matmul_relu/README.md)
  - [MatmulGelu算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/27_matmul_gelu/README.md)
  - [MatmulSilu算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/28_matmul_silu/README.md)
  - [FP8类型反量化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/29_a2_fp8_e4m3_matmul/README.md)
  - [INT8类型反量化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/30_w8a16_matmul/README.md)

- 工具支持
  - 更新[Python调用接口](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/python_extension/README.md)内容
    - 调整工程组织结构
    - 支持转置情况
  - 新增[`msTuner_CATLASS`](https://gitcode.com/cann/catlass/tree/v1.2.0/tools/tuner/README.md)工具，用于Tiling自动寻优，在搜索空间内全量运行并获取性能数据
  - 支持启用[`msSanitizer`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0039.html)地址消毒工具（编译选项加入`--enable_mssanitizer`）

- 文档资料
  - 新增[`catlass_optimize_guidance.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/contents/advanced/catlass_optimize_guidance.md)文档，介绍CATLASS赋能下`Gemm`类算子常用的调优方式

- Bugfix&优化
  - 优化[`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/06_optimized_matmul/README.md)算子实现，支持任意Padding方式组合
  - 修复`ASCEND_RT_VISIBLE_DEVICES`环境变量启用下，`msTuner_CATLASS`工具无法取得实际运行`DeviceId`的问题
  - 修复[PFA算子样例](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/19_mla/README.md)在单行数据场景下`Set/Wait`错配的异常情形
  - 修复[`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/06_optimized_matmul/README.md)算子样例计算`Workspace`大小出错
  - 隔离启用`AscendC::Dump`及`AscendC::print`功能的代码段
  - 修复[`GroupedMatmulSliceK`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/05_grouped_matmul_slice_k/README.md)算子在Ki=0特例时的输出清零行为，并将真值比较逻辑调整为全尺寸(M,N,K)比较
  - 修改[`performance_tools.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/tools/performance_tools.md)，[`tutorials.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/tutorials.md)等文档中的错误

### CATLASS 1.1.0

- 关键特性
  - 提供Python调用CATLASS算子的工程组件
    - 可编译[pybind](https://github.com/pybind/pybind11)扩展及[PyTorch](https://pytorch.org/)扩展件
  - 支持算子仿真运行（编译选项启用`--simulator`）
  - 编译过程适配毕昇编译器（[bisheng](https://www.hiascend.com/cann/bisheng)）

- 更多样例
  - [带偏置的MatmulBias算子](https://gitcode.com/cann/catlass/blob/v1.1.0/examples/20_matmul_bias/README.md)
  - [预加载(Preload)优化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/21_basic_matmul_preload_zN/README.md) （科大讯飞联创贡献）
  - [K轴切分(Split-K)优化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/22_padding_splitk_matmul/README.md) （科大讯飞联创贡献）

- 工具支持
  - 支持[`AscendC::Dump`](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0075.html)与[`AscendC::printf`](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0075.html)进行打印调试
    - 编译选项中加入`--enable_ascendc_dump`和`--enable_print`以启用上述功能
    - 请参阅文档: [`ascendc_dump`](https://gitcode.com/cann/catlass/tree/v1.1.0/docs/tools/ascendc_dump.md)和[`print`](https://gitcode.com/cann/catlass/tree/v1.1.0/docs/tools/print.md)

- 文档资料
  - 新增[tutorials快速上手示例](https://gitee.com/ascend/catlass/tree/v1.1.0/docs/tutorials.md)
  - 新增利用[msProf工具](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0082.html)进行算子性能调测的文档：[msProf](https://gitee.com/ascend/catlass/tree/v1.1.0/docs/tools/performance_tools.md)性能调测

- Bugfix&优化
  - 优化`Kernel`层AIC程序，添加`PIPE_ALL`避免整网影响
  - 优化[`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/06_optimized_matmul/README.md)算子实现，在非必要Padding场景下不启动AIV核
  - 修复`Block`层预加载`nextBlock`时的错误
  - 隔离Kernel侧`AscendC`的`inline`定义，避免异构编程时无法使用部分标准库
  - 修改`l2offset`设置的重定义问题

### CATLASS 1.0.0

- [CATLASS](https://gitcode.com/cann/catlass/)模板库正式开源发布

- 关键特性
  - 提供Kernel、Block、Tile、Basic分层算子开发能力

- 样例参考
  - 提供包括基础Matmul及各种不同的优化策略在内的算子样例
