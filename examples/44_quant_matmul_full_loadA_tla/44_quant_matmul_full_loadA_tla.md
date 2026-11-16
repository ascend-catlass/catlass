# CATLASS Quant Matmul Full LoadA Tla 样例介绍

## 原型设计

| 名称/Name     | 类型/Class | 数据类型/Dtype | 维度/Dims | 格式/Format | 描述/Description   |
| ------------- | ---------- | -------------- | --------- | ----------- | ------------------ |
| matA          | inTensor   | int8           | [m, k]    | ND          | 左矩阵             |
| matB          | inTensor   | int8           | [n, k]    | ND          | 右矩阵，支持转置   |
| scale         | inTensor   | float          | [n]       | ND          | perChannel量化系数 |
| perTokenScale | inTensor   | float          | [m]       | ND          | perToken量化系数   |
| matD          | outTensor  | bf16           | [m, n]    | ND          | 输出矩阵           |

## 样例实现

CATLASS [`44_quant_matmul_full_loadA_tla`样例](./README.md)算子是基于CATLASS Gemm API实现的昇腾亲和Matmul算子，针对大尺寸矩阵计算场景优化设计，关键算子组件包括以下几部分:

- **Example组装**：[quant_matmul_full_loadA_tla.cpp](./quant_matmul_full_loadA_tla.cpp)
- **Kernel实现**：
  - 主Kernel文件：[quant_matmul_full_loadA_tla.hpp](../../include/catlass/gemm/kernel/quant_matmul_full_loadA_tla.hpp)

- **Block组件**，包含：
  - 全载专门的mmad组件[block_mmad_pingpong_full_loadA_tla.hpp](../../include/catlass/gemm/block/block_mmad_pingpong_full_loadA_tla.hpp)
  - 反量化后处理组件[block_epilogue_per_token_dequant_tla.hpp](../../include/catlass/epilogue/block/block_epilogue_per_token_dequant_tla.hpp)；
  - 多核全载用的swizzle[GemmIdentityBlockSwizzleL1FullLoad](../../include/catlass/gemm/block/block_swizzle.hpp)

## 方案设计

- 如下图所示，CATLASS矩阵有关键参数 `L1TileShape<M, N, K>`，C矩阵按照`L1TileShape::M`和`L1TileShape::N`参数切分基本块并分核，而后普通Matmul模板会将A矩阵中`L1TileShape::M * L1TileShape::K`大小的矩阵块载入L1，而A矩阵全载模板会直接将`L1TileShape::M * K`大小的矩阵块载入L1，而对于B矩阵，普通模板和A矩阵全载模板都是载入`L1TileShape::K * L1TileShape::N`大小的矩阵块L1。

![A矩阵全载示意图](https://raw.gitcode.com/user-images/assets/7631999/38ccafec-7fd8-4b9f-a4bb-dd57085ffb63/53e318e0feb94c578d888e67365e4c97.png_tplv-a9rns2rl98-image-qvalue.png "53e318e0feb94c578d888e67365e4c97.png~tplv-a9rns2rl98-image-qvalue.png")

- A矩阵全载模板相较普通模板的流水变动如下图所示，使用全载方案，在计算一开始就经MTE2完整搬入A矩阵的数据块，而后pingpong搬入B矩阵执行计算。

![A矩阵全载流水图](https://raw.gitcode.com/user-images/assets/7631999/1de46727-7c46-411e-936c-7a437d951a3a/3e9c799e1de0405d89f07a6bfd7d7c54.png_tplv-a9rns2rl98-image-qvalue.png "3e9c799e1de0405d89f07a6bfd7d7c54.png~tplv-a9rns2rl98-image-qvalue.png")

- 采用A矩阵全载模板时，需要一半的L1空间以放入`L1TileShape::M * problemShape.K`的数据，若L1空间不够A矩阵全载，则返回报错。
- A矩阵全载时，N轴越大，单核越能多次复用L1中的A矩阵、无需再从GM或L2 Cache搬运A矩阵，性能收益就越大。
- A矩阵全载时，N轴较小，无法复用A矩阵，性能收益较00_basic_matmul可能会出现劣化。
- 若`problemShape.M > L1TileShape::M`，可使用`GemmIdentityBlockSwizzleL1FullLoad<SwizzleOffset, SwizzleDirection, AicCoreNum>`策略，使得每个核需要处理的基本块尽可能地连在一起，提升A矩阵分核全载时的块间复用率。
- 若`problemShape.M <= L1TileShape::M`，即M方向不切块分核，此时使用`GemmIdentityBlockSwizzle`策略即可适用，常用的策略参数选取可参考[swizzle_explanation](../../docs/zh/2_Design/01_kernel_design/02_swizzle.md)。
- 以20个cube核为例，常用的`GemmIdentityBlockSwizzle`策略的基本块分核顺序为`0-1-2-...-18-19-0-1-2...-18-19-0-1-2...`，每个核需要处理的基本块跳跃分布，而`GemmIdentityBlockSwizzleL1FullLoad`策略的基本块分核顺序为`0-0...-0-1-1...-1-2-2...-19`，每个核需要处理的基本块连续分布。

## 性能收益

使用相同tileShape和swizzle参数，相比标杆样例12_quant_matmul，具备A矩阵全载特性的样例性能平均提升5%~15%，这是由于全载实现减少了计算过程中的A矩阵搬运，提升了全载矩阵的数据复用率，可参考下表。

| [M,N,K]            | 12_quant_matmul | 44_quant_matmul_full_loadA_tla |
| ------------------ | --------------- | ------------------------------ |
| [512, 4096, 1024]  | 33.26us         | 31.73us                        |
| [128,16384, 1024]  | 38.68us         | 37.05us                        |
| [1024, 4096, 1024] | 42.66us         | 41.96us                        |
| [512,8192,1024]    | 52.36us         | 44.05us                        |
| [128,16384,2048]   | 53.77us         | 52.75us                        |

说明：

- 标杆为[QuantMatmul](../12_quant_matmul/README.md)算子。
- 统计耗时均为核函数总耗时，使用[msprof](https://www.hiascend.com/document/redirect/CannCommunitymsOpProf)工具得到。
- 上述测试例中A、B及输出矩阵均为`layout::RowMajor`排布方式。
- 测试环境说明：NPU型号为910B1，CANN包版本为9.0.0。
