# SvdQuantMatmul Example Readme

## SvdQuant算法原理简介

在量化前有原始的激活X，原始的权重W。给定一个Smooth参数进行变换，可将量化难度从激活转移到权重，该部分离线融合，无在线开销。

$$
X' = X @ diag(1/s) \\
W' = diag(s) @ W
$$

将变换后的权重 W' 进行SVD分解

$$
W' = (US)T = L_{1} L_{2}
$$

其中 L1 的shape为[k, r]，L2 的shape为[r, n]，其中r为SVD分解后取的秩，典型值16/32/64

计算残差R

$$
R = W' - L_{1} L_{2}
$$

$L_{1} L_{2}$ 中吸收了异常值，而残差R中主要为正常值，易于量化。

因此算法可以描述为

$$
\begin{aligned}
Y &= X W = X' W' \\
&= X'(L_{1}L_{2} + R) \\
&= X'L_{1}L_{2} + X'R \\
&\approx X'L_{1}L_{2} + Quant(X')Quant(R) \\
\end{aligned}
$$

因此原始矩阵乘XW可以分解为2路进行计算：

- 主通路$X'R$: W4A4
- 低秩旁路 $X'L_{1}L_{2}$: 高精度计算(fp16/bf16)

计算的数据流如下图所示：

![SvdQuant数据流](https://raw.gitcode.com/lavateinn/images/raw/main/svd_quant_matmul/svd_quant_data_flow.png)

在本样例的实现中，Quant(R) 采用标准的量化实现，scale的计算

$$
scaleR = 2 ** (log_{2} (abs(R)) - emax)
$$

在本算子的实现中, Quant(X')的scale用如下公式计算

$$
scaleX = 2 ** (log_{2} (abs(X') / qmax))
$$

其中qmax是外部传入的参数，后续不用像标准实现一样减去emax。

SvdQuant更多细节参考[SvdQuant论文](https://arxiv.org/abs/2411.05007)

## 输入输出

- 样例中的shape参数分别为 `m, n, k, r`，量化参数`qmax`，约束如下：

| shape | 约束                                                                    |
| ----- | ----------------------------------------------------------------------- |
| m,n,k | 无限制                                                                  |
| r     | 典型值16/32/64，算子实现要求 r<=BlockMmad1::L1_TILE_N, 现有配置即r<=128 |
| qmax  | 取值范围[6.0, 12.0], 典型值 8.0                                         |

涉及的算子输入如下：

| tensor      | dtype           | shape     | layout      | desc                                                     |
| ----------- | --------------- | --------- | ----------- | -------------------------------------------------------- |
| X           | fp16/bf16       | (m, k)    | RowMajor    |                                                          |
| Svd1        | fp16/bf16       | (k, r)    | ColumnMajor | 类型和 X 相同，公式中的L1                                |
| Svd2        | fp16/bf16       | (r, n)    | ColumnMajor | 类型和 X 相同，公式中的L2                                |
| W           | float4_e2m1x2_t | (k, n)    | ColumnMajor | 公式中的残差R量化后的fp4                                 |
| MxScaleW    | float8_e8m0_t   | (k/32, n) | ColumnMajor | 公式中的残差R量化后的scale                               |
| qmax        | float           | (1,)      |             | scalar, 在kernel内部转换为fp16/bf16，和 X 相同           |
| SmoothScale | fp16/bf16       | (k,)      | RowMajor    | 可选输入，类型和 X 相同，传入算子的实际是(1/smoothScale) |
| Bias        | float           | (n,)      | RowMajor    | 可选输入，取决于量化前网络层是否有bias                   |

有些场景下量化前的网络层无法计算得到SmoothScale，因此设置为可选输入，当传入SmoothScale时，由于无法做bf16的除法，因此host侧这里传入1/SmoothScale， 然后在UB上乘上(1/SmoothScale)。

- 样例中涉及的 workspace 变量如下:

| tensor   | dtype           | shape     | layout   | desc          |
| -------- | --------------- | --------- | -------- | ------------- |
| C1       | fp16/bf16       | (m, r)    | RowMajor | 类型和 X 相同 |
| QuantX   | float4_e2m1x2_t | (m, k)    | RowMajor |               |
| MxScaleX | float8_e8m0_t   | (m, k/32) | RowMajor |               |

- 样例中输出为：

| tensor | dtype     | shape  | layout   | desc          |
| ------ | --------- | ------ | -------- | ------------- |
| Y      | fp16/bf16 | (m, n) | RowMajor | 类型和 X 相同 |

- tensor的依赖关系及计算流程如下:

![算子计算流程](https://raw.gitcode.com/lavateinn/images/raw/main/svd_quant_matmul/svd_quant_kernel_overview.png)

## 代码组织

```text
experimental
├── matmul
│   ├── 61_ascend950_svd_quant_matmul
│   │   ├── CMakeLists.txt     # CMake编译文件
│   │   ├── README.md
│   │   ├── svd_quant_matmul.cpp    # 调用样例
│   │   └── gen_data.py             # 数据生成脚本
```

## 编译及运行

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)，本用例为 Ascend 950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`。

```shell
# 编译指定用例
bash scripts/build.sh 61_ascend950_svd_quant_matmul -DCATLASS_ARCH=3510

# 生成测试样例（在 examples/61_ascend950_svd_quant_matmul/data 下生成 input/ 与 golden/）
# 参数对应 m n k r
python examples/61_ascend950_svd_quant_matmul/gen_data.py 256 256 512 32

# 执行测试样例
# 输入参数分别对应 m, n, k, r, deviceId, deviceId可选，默认为0
./output/bin/61_ascend950_svd_quant_matmul 256 256 512 32 0
```

执行结果如下，说明精度比对成功。

```bash
Compare success.
```

查看gen_data脚本可配参数

```bash
python examples/61_ascend950_svd_quant_matmul/gen_data.py -h

positional arguments:
    m
    n
    k
    r

options:
    -h, --help
    --dtype {float16, bfloat16}  default float16
    --smooth {0,1}              default 1
    --bias {0,1}                default 0
    --qmax QMAX                 qmax>0, default 8.0
```

其中dtype、smooth等配置参数需要和svd_quant_matmul.cpp文件中一致。

## 算子使用说明

`SvdQuantMatmulTla` 所使用的模板组件有4个，分别是`Gemm::Kernel::SmoothQuant`, `Gemm::MmadSvd1`, `Gemm::MmadSvd2`, `Gemm::MmadSvd3`,
其中`Gemm::Kernel::SmoothQuant`的定义在`include/catlass/gemm/kernel/svd_quant_matmul_tla.hpp`，3个Mmad定义在`include/catlass/gemm/block/block_mmad_svd_quant_tla.hpp`，这3个Mmad对应的 DispatchPolicy 定义在`include/catlass/gemm/dispatch_policy.hpp`。

3个DispatchPolicy参数顺序与默认值如下：

1、`Gemm::MmadSvd1`

| 模板参数             | 默认值  | 参数说明                                                                     |
| -------------------- | ------- | ---------------------------------------------------------------------------- |
| `ArchTag`            | 无      | 架构标签，例如 `Arch::Ascend950`                                             |
| `ENABLE_L1_RESIDENT` | `false` | 是否开启 L1 常驻                                                             |
| `L0C_STAGES`         | `1`     | L0C 缓冲段数；设为 `2` 可启用 L0C 双缓冲（需与 `ENABLE_UNIT_FLAG` 约束一致） |
| `ENABLE_L1_RESIDENT` | `false` | 是否启用 L1 常驻                                                             |
| `L1A_STAGES`         | `2`     | L1 上加载矩阵 A 的 buffer 数量                                               |
| `L1B_STAGES`         | `2`     | L1 上加载矩阵 B 的 buffer 数量                                               |
| `L0A_STAGES`         | `2`     | L0 上加载矩阵 A 的 buffer 数量                                               |
| `L0B_STAGES`         | `2`     | L0 上加载矩阵 B 的 buffer 数量                                               |

2、`Gemm::MmadSvd2`

| 模板参数           | 默认值  | 参数说明                                                                     |
| ------------------ | ------- | ---------------------------------------------------------------------------- |
| `ArchTag`          | 无      | 架构标签，例如 `Arch::Ascend950`                                             |
| `L0C_STAGES`       | `1`     | L0C 缓冲段数；设为 `2` 可开启 L0C 双缓冲（需与 `ENABLE_UNIT_FLAG` 约束一致） |
| `L0C_STAGES`       | `1`     | L0C 缓冲段数；设为 `2` 可启用 L0C 双缓冲（需与 `ENABLE_UNIT_FLAG` 约束一致） |
| `L1A_STAGES`       | `2`     | L1 上加载矩阵 A 的 buffer 数量                                               |
| `L1B_STAGES`       | `2`     | L1 上加载矩阵 B 的 buffer 数量                                               |
| `L0A_STAGES`       | `2`     | L0 上加载矩阵 A 的 buffer 数量                                               |
| `L0B_STAGES`       | `2`     | L0 上加载矩阵 B 的 buffer 数量                                               |

3、`Gemm::MmadSvd3`

| 模板参数            | 默认值  | 参数说明                                                                                                            |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------- |
| `ArchTag`           | 无      | 架构标签，例如 `Arch::Ascend950`                                                                                    |
| `ENABLE_UNIT_FLAG`  | `false` | 是否开启 UnitFlag；当 `L0C_STAGES > 1`（L0C 多缓冲）时必须为 `false`                                                |
| `L1_SCALE_FACTOR_K` | `16`    | GM→L1 的 MX scale 一次驻留所覆盖的 **L1 K 方向条带个数**；为 `1` 时表示每个 L1 K 条带各搬一次 scale（见类型内注释） |
| `L0C_STAGES`        | `1`     | L0C 缓冲段数；设为 `2` 可开启 L0C 双缓冲（需与 `ENABLE_UNIT_FLAG` 约束一致）                                        |
| `L1A_STAGES`        | `2`     | L1 上加载矩阵 A 的 buffer 数量                                                                                      |
| `L1B_STAGES`        | `2`     | L1 上加载矩阵 B 的 buffer 数量                                                                                      |
| `L0A_STAGES`        | `2`     | L0 上加载矩阵 A 的 buffer 数量                                                                                      |
| `L0B_STAGES`        | `2`     | L0 上加载矩阵 B 的 buffer 数量                                                                                      |

约束：

- `MmadSvd1`的 `L1A_STAGES`需和SmoothQuant的STAGE相等
- `MmadSvd2`和`MmadSvd3`的`L0C_STAGES`、`L1A_STAGES`、`L1B_STAGES`、`L0A_STAGES`、`L0B_STAGES`必须相等，Mmad2和Mmad3共享L1上的buffer，并交替使用。

## 性能优化点

### Tiling

根据SvdQuantMatmul的典型场景，本样例提供了两组Tiling配置，可根据需求自行扩展。

```cpp
enum class SvdQuantTilingTag
{
    Common,
    Small,
};
template <SvdQuantTilingTag TilingTag>
struct TilingTag2Config {};
template <>
struct TilingTag2Config<SvdQuantTilingTag::Common> {
    using L1TileShape1 = Shape<Int<128>, Int<128>, Int<256>>;
    using L0TileShape1 = Shape<Int<128>, Int<128>, Int<128>>;
    using L1TileShape2 = Shape<Int<256>, Int<256>, Int<128>>;
    using L0TileShape2 = Shape<Int<256>, Int<256>, Int<64>>;
    using L1TileShape3 = Shape<Int<256>, Int<256>, Int<512>>;
    using L0TileShape3 = Shape<Int<256>, Int<256>, Int<256>>;
};
template <>
struct TilingTag2Config<SvdQuantTilingTag::Small> {
    using L1TileShape1 = Shape<Int<128>, Int<128>, Int<256>>;
    using L0TileShape1 = Shape<Int<128>, Int<128>, Int<128>>;
    using L1TileShape2 = Shape<Int<128>, Int<256>, Int<128>>;
    using L0TileShape2 = Shape<Int<128>, Int<256>, Int<64>>;
    using L1TileShape3 = Shape<Int<128>, Int<256>, Int<512>>;
    using L0TileShape3 = Shape<Int<128>, Int<256>, Int<256>>;
};
```

选用策略是用 `m1=256, n1=256` 估计任务数，当划分的任务数大于核数的时候用Common配置，否则用Small配置，主要区别是Small减小了Mmad2和Mmad3的m1，可以使用更多核数。

### SmoothQuant+Mmad1 部分的负载均衡优化

这部分的Mmad1 的problemShape为`{m, r, k}`， SmoothQuant的输入X 经过UB计算得到smoothX，随即搬运到L1(没有smooth的场景是先搬运到UB然后立即搬运至L1),
在UB上smoothX继续进行量化计算，在L1上smoothX是Mmad1的左矩阵参与矩阵乘，两边并行执行，可以掩盖Mmad1的搬运计算开销。

![SmoothQuant+Mmad1负载均衡](https://raw.gitcode.com/lavateinn/images/raw/main/svd_quant_matmul/svd_quant_quant_tile.png)

这部分没有采用catlass的BlockSwizzle，而是在kernel层中直接切M进行划分任务(这样做的前提是r较小, `r<=L1TileShape1::M`，否则会有精度问题)。
任务划分的时候先处理完整块，然后对尾块进行平均分到尽可能多的核上。

经过负载均衡之后，在M较小时(部分典型场景M=256,512)可以有效提升这部分的性能。

### Mmad2+Mmad3 部分的负载均衡优化

Mmad2的problemShape为`{m, n, r}`，输入类型为`fp16/bf16`，Mmad3的problemShape为`{m, n, k}`， 输入类型为mxfp4，输出的shape是一样的，可以共用BlockSwizzle进行任务划分。
kernel层伪代码如下

```cpp
for (uint32_t loopIdx = aiCoreIdx; loopIdx < normalBlockNum23 + aiCoreNum; loopIdx += aiCoreNum) {
    // Get tensor A2 B2 C
    // Get tensor A3 MxScale3 B3 MxScale3 C
    mmad2(A2, B2, C, bias); // bias在Mmad2中进行处理
    mmad3(A3, MxScale3, B3, MxScale3, C);
}
```

2个Mmad交替串行执行，结果在L0C上累加，在mmad3中写出GM。

经过负载均衡后同样是先处理完整块，当尾块数量小于核数的一半时，对尾块进行切分，当尾块 m>n 时，切m，当尾块 m<n 时，切n。

![Mmad2+Mmad3负载均衡](https://raw.gitcode.com/lavateinn/images/raw/main/svd_quant_matmul/svd_quant_mmad23_tile.png)
