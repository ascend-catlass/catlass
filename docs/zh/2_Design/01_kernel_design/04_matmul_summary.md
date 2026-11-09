# 矩阵乘模板总结

当前库上`examples`内包含多种矩阵乘的样例模板，其来源是不同的matmul理论模板与工程实践中发现的工程优化点的组合。在充分理解了各个理论模板和工程优化后，开发者可以基于问题场景选择适合的样例模板、甚至进一步自行组合出库上没有的新的样例模板，来达成矩阵乘的高性能优化。

注意，本文档仅总结矩阵乘方案相关的样例，其他涉及[量化](../../../../examples/12_quant_matmul/README.md)、[groupMatmul](../../../../examples/08_grouped_matmul/README.md)、[后处理](../../../../examples/03_matmul_add/README.md)等的矩阵乘不在此处总结。

## 样例模板清单

| 样例                                                                  | 理论模板                                    | 工程优化                                                                                       | dispatchPolicy                 |
| --------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------ |
| [00_basic_matmul](../../../../examples/00_basic_matmul/basic_matmul.cpp) | Common                                      | 流水优化(Multi Buffer)                                                                         | `MmadAtlasA2Pingpong`          |
| [04_padding_matmul](../../../../examples/04_padding_matmul/padding_matmul.cpp) | Common                                      | 流水优化(Multi Buffer)、读取带宽优化(Padding)-PaddingMatrixND                                  | `MmadAtlasA2Pingpong`          |
| [06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp) | Common                                      | 流水优化(Multi Buffer/Preload)、读取带宽优化(PaddingMatrixNZ/ShuffleK)、小M下指令替换（需修改样例启用） | `MmadAtlasA2Preload`           |
| [09_splitk_matmul](../../../../examples/09_splitk_matmul/splitk_matmul.cpp) | 多核切K模板 MultiCoreSplitK                 | 流水优化(Multi Buffer)                                                                         | `MmadAtlasA2Pingpong`          |
| [21_basic_matmul_preload_zN](../../../../examples/21_basic_matmul_preload_zN/basic_matmul_preload_zN.cpp) | Common                                      | 流水优化(Multi Buffer/Preload)、读取带宽优化(ShuffleK)                                         | `MmadAtlasA2Preload`           |
| [22_padding_splitk_matmul](../../../../examples/22_padding_splitk_matmul/padding_splitk_matmul.cpp) | 多核切K模板 MultiCoreSplitK                 | 流水优化(Multi Buffer)、读取带宽优化(Padding)-PaddingMatrixND                                  | `MmadAtlasA2Pingpong`          |
| [25_matmul_full_loadA](../../../../examples/25_matmul_full_loadA/matmul_full_loadA.cpp) | Common                                      | 流水优化(Multi Buffer)、读取带宽优化(L1常驻)                                                   | `MmadAtlasA2FullLoadA`         |
| [31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp) | Common                                      | 流水优化(Multi Buffer)、Scalar开销消减                                                         | `MmadAtlasA2Small`             |
| [34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp) | 单核切K模板 SingleCoreSplitK                | 流水优化(Multi Buffer)、读取带宽优化(PaddingMatrixNZ)、写出带宽优化                            | `MmadAtlasA2SingleCoreSplitk`  |
| [37_streamk_matmul](../../../../examples/37_streamk_matmul/streamk_matmul.cpp) | 尾轮多核切K模板的StreamK方案                | 流水优化(Multi Buffer)、读取带宽优化(ShuffleK)                                                 | `MmadAtlasA2Streamk`           |

<details>
<summary><strong><font size="4">00_basic_matmul</font></strong></summary>

- 理论模板：[`Common模板`](#理论模板清单)
- 工程优化：[`流水优化(Multi Buffer)`](#工程优化清单)
- 关键交付件
  - host：[00_basic_matmul](../../../../examples/00_basic_matmul/basic_matmul.cpp)
  - kernel：[basic_matmul.hpp](../../../../include/catlass/gemm/kernel/basic_matmul.hpp)
  - blockMmad：[block_mmad_pingpong.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`

</details>

<details>
<summary><strong><font size="4">04_padding_matmul</font></strong></summary>

- 理论模板：[`Common模板`](#理论模板清单)
- 工程优化：
- [`流水优化(Multi Buffer)`](#工程优化清单)
- [`读取带宽优化(Padding)- PaddingMatrixND`](#工程优化清单)
- 关键交付件
  - host：[04_padding_matmul](../../../../examples/04_padding_matmul/padding_matmul.cpp)
  - kernel：[padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp)
  - blockMmad：[block_mmad_pingpong.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`

</details>

<details>
<summary><strong><font size="4">06_optimized_matmul</font></strong></summary>

- 理论模板：[`Common模板`](#理论模板清单)
- 工程优化：
- [`流水优化(Multi Buffer)`](#工程优化清单)
- [`流水优化(Preload)`](#工程优化清单)
- [`读取带宽优化(Padding)- PaddingMatrixNZ`](#工程优化清单)
- [`读取带宽优化(ShuffleK)`](#工程优化清单)
  - [`读取带宽优化(小M下指令替换)`](#工程优化清单)（需要修改样例启用）
- 关键交付件
  - host：[06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp)
  - kernel：[optimized_matmul.hpp](../../../../include/catlass/gemm/kernel/optimized_matmul.hpp)
  - Padding前处理组件：[padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp)
  - blockMmad：[block_mmad_preload.hpp](../../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- dispatchPolicy：`MmadAtlasA2Preload`
- ⚠️ 注意：即使没有使用`PaddingMatrixNZ`前处理，依然会产生MIX算子编译和CV1:0启动的开销（比仅AIC启动的开销大）。MIX算子指同时包含AIC与AIV的算子；CV1:0表示启动时AIC与AIV的配比为1:0，即按MIX方式启动但只运行AIC。

</details>

<details>
<summary><strong><font size="4">09_splitk_matmul</font></strong></summary>

- 理论模板：[`多核切K模板 MultiCoreSplitK`](#理论模板清单)
- 工程优化：[`流水优化(Multi Buffer)`](#工程优化清单)
- 关键交付件
  - host：[09_splitk_matmul](../../../../examples/09_splitk_matmul/splitk_matmul.cpp)
  - kernel：[splitk_matmul.hpp](../../../../include/catlass/gemm/kernel/splitk_matmul.hpp)
  - blockMmad：[block_mmad_pingpong.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`

</details>

<details>
<summary><strong><font size="4">21_basic_matmul_preload_zN</font></strong></summary>

（此样例主要承载NZ排布输入的适配方法，也可换成ND排布输入，无MIX算子编译启动的开销）

- 理论模板：[`Common模板`](#理论模板清单)
- 工程优化：
- [`流水优化(Multi Buffer)`](#工程优化清单)
- [`流水优化(Preload)`](#工程优化清单)
- [`读取带宽优化(ShuffleK)`](#工程优化清单)
- 关键交付件
  - host：[21_basic_matmul_preload_zN](../../../../examples/21_basic_matmul_preload_zN/basic_matmul_preload_zN.cpp)
  - kernel：[basic_matmul_preload.hpp](../../../../include/catlass/gemm/kernel/basic_matmul_preload.hpp)
  - blockMmad：[block_mmad_preload.hpp](../../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- dispatchPolicy：`MmadAtlasA2Preload`

</details>

<details>
<summary><strong><font size="4">22_padding_splitk_matmul</font></strong></summary>

- 理论模板：[`多核切K模板 MultiCoreSplitK`](#理论模板清单)
- 工程优化：
- [`流水优化(Multi Buffer)`](#工程优化清单)
- [`读取带宽优化(Padding)- PaddingMatrixND`](#工程优化清单)
- 关键交付件
  - host：[22_padding_splitk_matmul](../../../../examples/22_padding_splitk_matmul/padding_splitk_matmul.cpp)
  - kernel：[padding_splitk_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_splitk_matmul.hpp)
  - Padding前处理组件：[padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp)
  - SplitkReduceAdd后处理组件：[splitk_matmul.hpp](../../../../include/catlass/gemm/kernel/splitk_matmul.hpp)
  - blockMmad：[block_mmad_pingpong.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`

</details>

<details>
<summary><strong><font size="4">25_matmul_full_loadA</font></strong></summary>

（此样例及相关组件仅适配了A矩阵全载实现，需要实现B矩阵全载可参考关键交付件自行开发）

- 理论模板：[`Common模板`](#理论模板清单)
- 工程优化：
- [`流水优化(Multi Buffer)`](#工程优化清单)(全载的A矩阵在L1上不使用多buffer)
- [`读取带宽优化(L1常驻)`](#工程优化清单)
- 关键交付件
  - host：[25_matmul_full_loadA](../../../../examples/25_matmul_full_loadA/matmul_full_loadA.cpp)
  - kernel：[matmul_full_loadA.hpp](../../../../include/catlass/gemm/kernel/matmul_full_loadA.hpp)
  - blockMmad：[block_mmad_pingpong_full_loadA.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong_full_loadA.hpp)
- dispatchPolicy：`MmadAtlasA2FullLoadA`
- BlockScheduler：`GemmIdentityBlockSwizzleL1FullLoad`

</details>

<details>
<summary><strong><font size="4">31_small_matmul</font></strong></summary>

- 理论模板：[`Common模板`](#理论模板清单)
- 工程优化：
- [`流水优化(Multi Buffer)`](#工程优化清单)
  - [`Scalar开销消减`](#工程优化清单)
- 关键交付件
  - host：[31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp)
  - kernel：[small_matmul.hpp](../../../../include/catlass/gemm/kernel/small_matmul.hpp)
  - blockMmad：[block_mmad_small.hpp](../../../../include/catlass/gemm/block/block_mmad_small.hpp)
- dispatchPolicy：`MmadAtlasA2Small`
- BlockScheduler：kernel内实际不使用

</details>

<details>
<summary><strong><font size="4">34_single_core_splitk_matmul</font></strong></summary>

- 理论模板：[`单核切K模板 SingleCoreSplitK`](#理论模板清单)
- 工程优化：
- [`流水优化(Multi Buffer)`](#工程优化清单)
- [`读取带宽优化(Padding)- PaddingMatrixNZ`](#工程优化清单)
  - [`写出带宽优化`](#工程优化清单)
- 关键交付件
  - host：[34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)
  - kernel：[single_core_slicek_matmul.hpp](../../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp)
  - Padding前处理组件和RemovePaddingNDAndCast后处理组件：[padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp)
  - blockMmad：[block_mmad_single_core_splitk.hpp](../../../../include/catlass/gemm/block/block_mmad_single_core_splitk.hpp)
- dispatchPolicy：`MmadAtlasA2SingleCoreSplitk`
- BlockScheduler：`SingleCoreSplitkGemmIdentityBlockSwizzle`

</details>

<details>
<summary><strong><font size="4">37_streamk_matmul</font></strong></summary>

- 理论模板：[`尾轮多核切K`](#理论模板清单)的StreamK方案，详见[设计文档](../../../../examples/102_dynamic_optimized_matmul/docs/zh/StreamkMatmul.md)
- 工程优化：
- [`流水优化(Multi Buffer)`](#工程优化清单)
- [`读取带宽优化(ShuffleK)`](#工程优化清单)
- 关键交付件
  - host：[37_streamk_matmul](../../../../examples/37_streamk_matmul/streamk_matmul.cpp)
  - kernel：[streamk_matmul.hpp](../../../../include/catlass/gemm/kernel/streamk_matmul.hpp)
  - StreamkReduceAdd后处理组件：[streamk_matmul.hpp](../../../../include/catlass/gemm/kernel/streamk_matmul.hpp)
  - blockMmad：[block_mmad_streamk.hpp](../../../../include/catlass/gemm/block/block_mmad_streamk.hpp)
- dispatchPolicy：`MmadAtlasA2Streamk`
- BlockScheduler：`StreamkGemmIdentityBlockSwizzle`

</details>

## 理论模板清单

<details>
<summary><strong><font size="4">Common模板</font></strong></summary>

### Tiling建模

<img src="https://raw.gitcode.com/user-images/assets/7801479/b1cb21ac-af83-4736-8582-4ed7392d766b/1common.png" alt="Common模板Tiling建模图：matA、matB、matC分别按m1、n1、k1分块" width="80%">

如图展示一个常规的fp16的矩阵运算（L0C上按照fp32累加），定义相关参数：

- 问题shape：$M$，$N$，$K$
- 搬入L1Cache时的TileShape：$m_1$，$n_1$，$k_1$
- 搬入L0A/L0B、搬出L0C时的TileShape：$m_0$，$n_0$，$k_0$

采用 $M$、$N$ 方向分核，按照$m_1$、$n_1$切分，产生$\frac{MN}{m_1n_1}$个基本任务块、分配给AIC核完成搬运和计算，每个基本任务块需要搬运$m_1K+Kn_1$的数据、计算得到$m_1n_1$的结果并搬出。由此产生约束：

- $m_1k_1*L1Stage_A + n_1k_1*L1Stage_B <= L1Size / 2Byte$
- $m_0k_0*L0AStage <= L0ASize / 2Byte$
- $n_0k_0*L0BStage <= L0BSize / 2Byte$
- $m_0n_0*L0CStage <= L0CSize / 4Byte$
- $m_0 = m_1$
- $n_0 = n_1$

其中$L1Stage_A$、$L1Stage_B$、$L0AStage$、$L0BStage$、$L0CStage$分别为A/B矩阵在L1、L0A、L0B与L0C上的Buffer份数，单Buffer取1、双Buffer取2；`2Byte`与`4Byte`分别为fp16输入元素和fp32累加结果元素的字节数。

### 读取数据量

每个基本任务块需要搬运$m_1K+Kn_1$的数据，总读取量为：

$2Byte * [m_1K+Kn_1] * \frac{MN}{m_1n_1} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{n_1}]$

### 写出数据量

每个基本任务块计算得到$m_1n_1$的结果并搬出，总写出量为：

$2Byte * MN$

### 计算量

输出矩阵C的每个数据点需要$K$次乘加，总计算量固定为：

$2MNK$

计算耗时多数情况下为刚性时间，仅与参与计算的AIC核数相关，在各理论模板中都一样，后续不再赘述。
</details>

<details>
<summary><strong><font size="4">多核切K模板 MultiCoreSplitK</font></strong></summary>

### Tiling建模

<img src="https://raw.gitcode.com/user-images/assets/7801479/d4b0e2d2-4333-4df4-9af2-44654cc37e54/2multiCoreSplitK.png" alt="多核切K模板Tiling建模图：matA与matB沿K轴切分为多段，各核算得的matC部分结果经ReduceAdd归约" width="80%">

如图展示一个常规的fp16的矩阵运算（L0C上按照fp32累加），$MN$方向共切分12个基本任务块，假设有24个AIC物理核，此时负载不均衡，故引入$K$轴切分成2个$k$、产生24个基本任务块，在AIC上负载均衡。定义相关参数：

- 问题shape：$M$，$N$，$K$
- 搬入L1Cache时的TileShape：$m_1$，$n_1$，$k_1$
- 搬入L0A/L0B、搬出L0C时的TileShape：$m_0$，$n_0$，$k_0$
- <font color="red">相较Common模板</font>，新增$K$方向切分长度$k$

$k$为每个基本任务块在$K$方向上的长度，切分份数为$\lceil K/k \rceil$。为使$\frac{MNK}{m_1n_1k}$为整数，$k$需整除$K$；切分份数不超过$\lceil K/k_1 \rceil$，即$k$与L1TileShape的$k_1$同量级，并受切分份数上限约束：$K \le 1024$时最多2份，$K \le 2048$时最多4份，$K \le 4096$时最多8份，否则最多16份。切分份数的具体选取见[splitk_matmul.hpp](../../../../include/catlass/gemm/kernel/splitk_matmul.hpp)中的`GetSplitkFactor`。

在$m_1$、$n_1$较大时，可能存在负载不均问题，即M,N方向切分的任务块远少于AIC核数，导致读取带宽不高（核数不够），因此可以在K方向上分核。产生$\frac{MNK}{m_1n_1k}$个基本任务块、分配给AIC核完成搬运和计算，每个基本任务块需要搬运$m_1k+kn_1$的数据、计算得到$m_1n_1$的结果并搬出。硬件上约束与Common相同。

### 读取数据量

每个基本任务块需要搬运$m_1k+kn_1$的数据，总读取量与`Common模板`一致：

$2Byte * [m_1k+kn_1] * \frac{MNK}{m_1n_1k} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{n_1}]$

### 写出数据量

每个基本任务块计算得到$m_1n_1$的结果并搬出，需要$\frac{K}{k}$个基本块累加来得到输出矩阵C的$m_1n_1$块的最终输出，总写出量为：

$2Byte * MNK / k$

### 定性分析

相较`Common模板`，搬入数据量不变，写出数据量增加，并产生后处理ReduceAdd的开销（包含MIX算子编译启动的开销），但切分基本块更多、更易负载均衡。

</details>

<details>
<summary><strong><font size="4">单核切K模板 SingleCoreSplitK</font></strong></summary>

### Tiling建模

<img src="https://raw.gitcode.com/user-images/assets/7801479/e16f5a39-2f7b-4a72-9d79-502cc8682e75/3singleCoreSplitK.png" alt="单核切K模板Tiling建模图：K轴分段计算，matC上多段结果经AtomicAdd原子加合并" width="80%">

如图展示一个常规的fp16的矩阵运算（L0C上按照fp32累加），定义相关参数：

- 问题shape：$M$，$N$，$K$
- 搬入L1Cache时的TileShape：$m_1$，$n_1$，$k_1$
- 搬入L0A/L0B、搬出L0C时的TileShape：$m_0$，$n_0$，$k_0$

相比`Common模板`，为了减少读取数据量，进一步增大抽象上的$m_1$、$n_1$，考虑将$m_1k_1$的tile块直接与对应的所有$k_1n_1$的tile块完成计算（等同于将$n_1$放大到$N$），此时输出$m_0n_0$的tile块无法在$L0C$常驻累加，需要及时搬出，通过[`atomicAdd`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/atomic_operations/atomic_operations_overview.md)在`GM`上累加。硬件上约束如下：

- $m_1k_1*L1Stage_A + n_1k_1*L1Stage_B <= L1Size / 2Byte$
- $m_0k_0*L0AStage <= L0ASize / 2Byte$
- $n_0k_0*L0BStage <= L0BSize / 2Byte$
- $m_0n_0*L0CStage <= L0CSize / 4Byte$
- $m_0 <= m_1$
- $n_0 <= n_1$

### 读取数据量

在`Common模板`的读取数据量公式上，将$n_1$放大到$N$；或者从A矩阵分基本任务块来理解，切分$\frac{MK}{m_1k_1}$个基本块，每个基本块完成搬入此块A矩阵tile块以及对应全部的B矩阵tile块，搬入的数据量为$m_1k_1+k_1N$：

$2Byte * [m_1k_1+k_1N] * \frac{MK}{m_1k_1} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{N}]$

### 写出数据量

切分A矩阵为基本任务块，共$\frac{MK}{m_1k_1}$个基本块，每个基本任务块计算得到$m_1N$的结果并搬出，总写出量为：

$2Byte * MNK / k_1$

### 定性分析

相较`Common模板`，搬入数据量减少，写出数据量增加，与AIV无关。

</details>

## 工程优化清单

<details>
<summary><strong><font size="4">流水优化（Multi Buffer）</font></strong></summary>

### 现象分析

如下图构造一个`Common`模板下的简单场景，对于单个AIC处理一个基本任务块C，需要的A/B矩阵Tile块较小，可以直接全部放入L1，且A/B矩阵从L1搬入L0时需要切4次搬入。

<img src="https://raw.gitcode.com/user-images/assets/7801479/a62726da-f3c2-4c37-8a32-711046cfd239/8pingpong0.png" alt="多Buffer优化前Single AIC数据通路图：数据由GM经MTE2搬入L1，再经MTE1搬入L0A/L0B，Cube计算后经FIXPIPE写回GM" width="80%">

各PIPE的指令流水图示例如下：

<img src="https://raw.gitcode.com/user-images/assets/7801479/59f2458f-719d-4711-8ed9-5e51732de2d0/SingleBuffer.png" alt="单Buffer指令流水图：MTE2、MTE1、CUBE、FIXPIPE四个阶段串行执行，搬入与计算无重叠" width="100%">

如果在AIC的L1/L0A/L0B/L0C上，每次载入数据tile块时都尽量填满所有空间，会导致各PIPE的流水串行，整体效率低下。

### 优化方案

使用常规优化手段Multi Buffer，即在L1/L0A/L0B/L0C上启用多buffer，使得流水尽可能并行，提升效率，这一优化策略如下图所示：

<img src="https://raw.gitcode.com/user-images/assets/7801479/40315511-85cc-44ac-be3f-9efb6b5c0194/8pingpong2.png" alt="多Buffer优化方案数据通路图：GM上A1/A2、B1/B2分块交替搬入L1，L0A、L0B按0/1双缓冲供Cube计算" width="80%">

各PIPE的指令流水图示例如下，MTE1上指令的0、1表示pingpong流水：

<img src="https://raw.gitcode.com/user-images/assets/7801479/e5cc2db5-873e-47f0-a425-464957168132/DoubleBuffer.png" alt="双Buffer指令流水图：MTE2、MTE1、CUBE、FIXPIPE按0/1双缓冲交替执行，实现搬运与计算重叠" width="100%">

⚠️ 需要注意的是，与`L1常驻`优化结合时，常驻的A/B矩阵的tile块，不启用多buffer。

### 特性承载代码

由于是常规优化手段，所有blockMmad组件均启用。

</details>

<details>
<summary><strong><font size="4">流水优化（Preload）</font></strong></summary>

### 现象分析

通过仿真流水发现pingpong策略的blockMmad存在问题：

- MTE2流水上存在加载空泡：位于“当前C矩阵基本块计算的最后一个A矩阵（B矩阵）tile块”与“下一个C矩阵基本块计算的第一个A矩阵（B矩阵）tile块”之间。

### 优化方案

针对GM->L1过程，读取当前轮次的$m_1k_1$（$k_1n_1$）时，计算上一轮读取的数据（假设Preload一轮，PRELOAD_STAGES = 1），步骤伪代码如下：

```cpp
for ... {
    // 搬入当前轮次的数据
    copyGM2L1A
    copyGM2L1B
    preload_count++
    if (preload_count == PRELOAD_STAGES) {
        // 计算前PRELOAD_STAGES轮次的数据
        copyL12L0A
        copyL12L0B
        Mmad
    }
}
```

如下图构造一个`Common`模板下的简单场景，对于单个AIC处理两个基本任务块C1和C2，需要的A/B矩阵Tile块放入L1需要切分2次，且A/B矩阵L1Tile块从L1搬入L0时需要切分4次（可以先学习前文[流水优化(Multi Buffer)](#工程优化清单)来增强理解）。以下给出`MmadAtlasA2Pingpong`和`MmadAtlasA2Preload`、`MmadAtlasA2PreloadAsync`的指令对比：

- `MmadAtlasA2Pingpong`中，调用两次blockMmad分别完成C1和C2的计算
- `MmadAtlasA2Preload`中，两次blockMmad也是分别完成C1和C2的计算，但A3/B3的GmToL1搬运提前到了第一次blockMmad中执行
- `MmadAtlasA2PreloadAsync`中，调用两次blockMmad和一次收尾的SynchronizeBlock。A2/B2的L1ToL0搬运、tileMmad以及C1的搬出从第一次blockMmad中推迟到第二次blockMmad中；A4/B4的L1ToL0搬运、tileMmad以及C2的搬出从第二次blockMmad中推迟到SynchronizeBlock中

<img src="https://raw.gitcode.com/user-images/assets/7801479/eca1e06a-7c9a-40d6-934b-6843f9229d7c/9preload0.png" alt="MmadAtlasA2Pingpong、MmadAtlasA2Preload、MmadAtlasA2PreloadAsync三种blockMmad的指令序列对比图" width="100%">

各PIPE的指令流水图示例如下，最终达成了A3、B3块的GmToL1搬运提前，减缓了MTE2上的搬运空泡：

<img src="https://raw.gitcode.com/user-images/assets/7801479/c4302c87-542e-42e1-811d-7bfe6b1ba22f/preload.png" alt="Preload优化指令流水图：MTE2将下一块A3/B3提前preload进L1，与Cube计算、写出并行重叠" width="100%">

### 特性承载代码

- [block_mmad_preload.hpp](../../../../include/catlass/gemm/block/block_mmad_preload.hpp)，对应dispatchPolicy：`MmadAtlasA2Preload`，需要在kernel内手动计算传入下一块预载数据的信息。
- [block_mmad_preload_async.hpp](../../../../include/catlass/gemm/block/block_mmad_preload_async.hpp)，对应dispatchPolicy：`MmadAtlasA2PreloadAsync`，通过异步控制，无需手动计算下一块预载数据信息，并支持mmad计算完成后的`Callback`传入。
- [block_mmad_preload_async_with_callback.hpp](../../../../include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp)，对应dispatchPolicy：`MmadAtlasA2PreloadAsyncWithCallback`，通过异步控制，无需手动计算下一块预载数据信息，并支持blockMmad计算前后的`Callback`传入。

</details>

<details>
<summary><strong><font size="4">读取带宽优化（Padding）</font></strong></summary>

### 现象分析

当数据读取为主流水时，优化读取带宽能带来性能收益，以fp16的A矩阵为例，目前有以下几种低带宽场景：

- **Stride非512Byte对齐导致的低带宽**。搬运参数srcDValue（详见[昇腾文档：DataCopy-随路转换ND2NZ搬运](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_GMToL1_ND2NZ.md)）非512Byte对齐时，单次搬运会被拆分为更短的片段并向上取整，有效带宽下降。CANN官方最佳实践[《GM地址尽量512B对齐》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/gm_address_512b_alignment.md)在Atlas A2系列产品上的实测显示，最差情况下（32B对齐）单核带宽仅为512B对齐的70%，即带宽下降约30%。
- **搬运指令限制导致的低带宽**。ND2NZ的搬运指令，参数srcDValue是uint16类型，最大值65535。当K>65535时，只能通过取ndNum = 1，在m方向循环调用搬运指令，降低了读取带宽。
- 相比ND2ND不转换排布，**ND2NZ随路转换有带宽损失**。

### 优化方案

针对上述情况，可以使用AIV对数据格式进行重排（预处理动作），当重排开销低于带宽损失时，会有性能收益。从复杂度和能应对的场景出发，有下列三种不同的重排方式。

#### PaddingMatrixND

<img src="https://raw.gitcode.com/user-images/assets/7801479/80501346-2ea2-42ba-8cc0-b6f614630606/4paddingND.png" alt="PaddingMatrixND示意图：GM上的ND矩阵写入workSpace时K轴按512Byte对齐，尾部补dummy数据" width="100%">

如图所示，将Stride方向按照512Byte对齐，实现复杂度最低，可以处理Stride非对齐导致的带宽下降。

#### PaddingMatrixBlockND

<img src="https://raw.gitcode.com/user-images/assets/7801479/9e5e0d37-ef6a-41e0-b54a-1f6d079e1179/4paddingBlockND.png" alt="PaddingMatrixBlockND示意图：GM上的矩阵按m1、k1分块写入workSpace并对齐补dummy" width="100%">

如图所示，按$m_1*k_1$作为“block”粒度重排，block内行优先、block间行优先，且$k_1$为512Byte对齐，实现复杂度适中，可以处理Stride非对齐和Stride超过65535导致的带宽下降。

#### PaddingMatrixNZ

<img src="https://raw.gitcode.com/user-images/assets/7801479/11728de5-073d-481b-a030-b262f425161f/4paddingNZ.png" alt="PaddingMatrixNZ示意图：GM上的NZ矩阵按16对齐写入workSpace，尾部补dummy数据" width="100%">

如图所示，重排为NZ格式，实现复杂度最高（在介绍的几种Padding策略内），因为和L1上数据排布一致，搬运带宽也最高，可以处理Stride非对齐、Stride超过65535、ND2NZ随路转换导致的带宽下降。

<font color="red">实际应用中，不同case适合的padding方式不同，暂无全局最优Padding选择。</font>

### 特性承载代码

- [padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp)中包含Padding前处理组件。
- 实际适配可参考[06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp)，通过`PaddingTag`、`PaddingBuilder`组装出A矩阵/B矩阵的Padding前处理。
- `PaddingMatrixBlockND`可参考[14_optimized_matmul_tla](../../../../examples/14_optimized_matmul_tla/optimized_matmul_tla.cpp)与[42_quant_optimized_matmul_tla](../../../../examples/42_quant_optimized_matmul_tla/quant_optimized_matmul_tla.cpp)中的使用方式。

</details>

<details>
<summary><strong><font size="4">读取带宽优化（ShuffleK）</font></strong></summary>

### 现象分析

通常，所有AIC核心都从$K$方向的第一个分块开始搬运计算，会存在多个核心同时读取同一片GM上数据的情况，产生数据读取冲突，导致读取带宽降低。

### 优化方案

<img src="https://raw.gitcode.com/user-images/assets/7801479/3b79fdb8-1154-46fa-bde8-057950d16b86/5shuffleK.png" alt="ShuffleK错位搬运示意图：matB沿K轴行序交错，使经Swizzle<2, 1>映射的matA与matB分块由同一Core计算" width="100%">

以`Common模板`为例，如图：

- $matC$中的$CoreX$表示该基本块分配给第$X$个AIC进行计算
- $Aj$表示A矩阵该$m_1$下沿$K$轴切分的第$j$个L1Tile基本块
- $B_{ij}$表示B矩阵第$i$个$n_1$分块下沿$K$轴切分的第$j$个L1Tile基本块，$i$为该基本块在matC中的$N$方向编号，$j$与A矩阵的$j$一致
- 图中matC基本块分核采用Swizzle<2, 1>，详见[swizzle_explanation](./02_swizzle.md)

如图左原始方案，$Core2$和$Core3$搬运A矩阵时均按照“$A0$->$A1$->$A2$->$A3$”的顺序，产生了数据读取冲突。

ShuffleK方案如图所示，根据$CoreIdx$来偏移起始搬运序号$j$，$Core2$搬运A矩阵时按照“$A2$->$A3$->$A0$->$A1$”的顺序，对应B矩阵按照“$B_{02}$->$B_{03}$->$B_{00}$->$B_{01}$”的顺序；$Core3$搬运A矩阵按照“$A3$->$A0$->$A1$->$A2$”的顺序，对应B矩阵按照“$B_{13}$->$B_{10}$->$B_{11}$->$B_{12}$”的顺序。从时间上错开，避免同地址访问冲突。

### 特性承载代码

- [block_mmad_preload.hpp](../../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- [block_mmad_preload_async.hpp](../../../../include/catlass/gemm/block/block_mmad_preload_async.hpp)
- [block_mmad_preload_async_with_callback.hpp](../../../../include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp)

上述Block层实现代码通过设置起始L1序号为`CoreIdx % kTileCount`来实现错位：

```cpp
// K方向L1Tile块总数，向上取整
kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
// 当前AIC的编号，作为起始搬运序号
startTileIdx = AscendC::GetBlockIdx();
// 按块总数取模，使不同核的起始搬运序号错开
firstTileIdx = startTileIdx % kTileCount;
```

</details>

<details>
<summary><strong><font size="4">读取带宽优化（小M下指令替换）</font></strong></summary>

### 现象分析

<img src="https://raw.gitcode.com/user-images/assets/7801479/e3a90904-677a-4973-b3b5-93e93ea072e2/6smallM.png" alt="小M场景ND2NZ搬运示意图：GM上M较小的ND矩阵按m1*k1分块搬入L1，以NZ分形存放" width="80%">

如图所示，矩阵计算中，当$M$很小时（例如$M$ < 8），采用随路ND2NZ的DataCopy（详见[昇腾文档：DataCopy-随路转换ND2NZ搬运](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_GMToL1_ND2NZ.md)）效率不高。

### 优化方案

通过`for`循环每次搬运一行，并在每一行调用`DataCopy`进行多次搬运（采用普通间隔搬运）。

### 特性承载代码

- [CopyGmToL1IntervalDataCopy](../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)
- 实际适配可参考[06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp)，在`struct TileCopyOpt`中将`CopyGmToL1A`的默认定义替换为`CopyGmToL1IntervalDataCopy`，具体改动如下：

```diff
struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    ...

-   // using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;
+   using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;

-   using CopyGmToL1A = typename Base::CopyGmToL1A;
+   // using CopyGmToL1A = typename Base::CopyGmToL1A;
    ...
};
```

</details>

<details>
<summary><strong><font size="4">读取带宽优化（L1常驻）</font></strong></summary>

### 现象分析

当读取为主流水，且A矩阵（或B矩阵）沿M（或N）方向被多个基本任务块重复读取时，`aic_mte2_time`随方向上的基本块数量线性增长。若L1剩余容量足以容纳完整的A矩阵（$M*K$个元素），同一份矩阵数据在多个基本块间的重复搬运就是可以消除的读取带宽浪费。

### 优化方案

实际开发时，可采用tile块常驻L1的方式，减少tile块数据的重复读取，等效提升了读取带宽，该特性需要结合不同理论模板来考虑实现方法。

### 特性承载代码

- `Common`模板可参考[25_matmul_full_loadA](../../../../examples/25_matmul_full_loadA/matmul_full_loadA.cpp)及相关交付件，该样例通过$M$轴上的单核全载或多核全载来在特定场景下优化性能，并配合专门的swizzle策略来提高L1上全载的A矩阵块的复用频率。
  - kernel：[matmul_full_loadA.hpp](../../../../include/catlass/gemm/kernel/matmul_full_loadA.hpp)
  - blockMmad：[block_mmad_pingpong_full_loadA.hpp](../../../../include/catlass/gemm/block/block_mmad_pingpong_full_loadA.hpp)
  - dispatchPolicy：`MmadAtlasA2FullLoadA`
  - BlockScheduler：`GemmIdentityBlockSwizzleL1FullLoad`
- `单核切K模板`[34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)在理论设计上就考虑了L1Tile块常驻的优化点。
  - kernel：[single_core_slicek_matmul.hpp](../../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp)
  - blockMmad：[block_mmad_single_core_splitk.hpp](../../../../include/catlass/gemm/block/block_mmad_single_core_splitk.hpp)
  - dispatchPolicy：`MmadAtlasA2SingleCoreSplitk`
  - BlockScheduler：`SingleCoreSplitkGemmIdentityBlockSwizzle`

</details>

<details>
<summary><strong><font size="4">Scalar开销消减</font></strong></summary>

### 现象分析

对于小Shape场景，如`Common模板`中：

- $M$、$N$方向分核数小于实际物理核数，每个AIC物理核最多仅处理一个基本任务块
- $k_1$ >= $K$，$K$方向无需切分即可从GM搬入L1

此时kernel总耗时较小，scalar开销对性能影响显著。

### 优化方案

消减冗余的scalar计算。

- kernel内不使用BlockScheduler来将任务块分配给物理核，手动计算每个物理核对应的任务块
- kernel内消除基本块循环（每个AIC仅处理1个任务块）
- kernel内简化offset相关计算
- blockMmad内消除L1A/L1B的$m$、$n$的循环
- blockMmad内简化offset相关计算

### 特性承载代码

- 参考[31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp)，可以和[00_basic_matmul](../../../../examples/00_basic_matmul/basic_matmul.cpp)的相关交付件对比来加深理解
  - kernel：[small_matmul.hpp](../../../../include/catlass/gemm/kernel/small_matmul.hpp)
  - blockMmad：[block_mmad_small.hpp](../../../../include/catlass/gemm/block/block_mmad_small.hpp)

</details>

<details>
<summary><strong><font size="4">写出带宽优化</font></strong></summary>

### 现象分析

当数据写出为主流水时，优化写出带宽能够带来性能收益。

- 当写出时`dstStride`未进行512Byte对齐，写出带宽下降。CANN官方最佳实践[《GM地址尽量512B对齐》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/gm_address_512b_alignment.md)在Atlas A2系列产品上的实测显示，最差情况下（32B对齐）单核带宽仅为512B对齐的70%，即带宽下降约30%
- 写出时用NZ2ND随路格式转换，产生带宽损失

### 优化方案

针对上述情况，可使用AIV对数据格式进行重排，在重排开销低于带宽损失时，会有性能收益。以下提供四种重排方式。
<img src="https://raw.gitcode.com/user-images/assets/7801479/89afcf74-a193-431b-aea3-a8de2abcb4f9/7rmPadding1.png" alt="写出重排方式一：局部workSpace，ND写出到GM时按512Byte对齐（n0尾部补dummy），再按block粒度在UB上重排写回GM" width="80%">

（↑）**方式一**：使用局部workSpace，ND写出到GM时512Byte对齐，随后按block块粒度在UB上重排，再写出到GM上。

<img src="https://raw.gitcode.com/user-images/assets/7801479/12665171-d86f-4208-8c11-3bef2359cfa1/7rmPadding2.png" alt="写出重排方式二：全量workSpace，ND写出到GM时按512Byte对齐（N尾部补dummy），全量写完后在UB上重排再写回GM" width="80%">

（↑）**方式二**：使用全量workSpace，ND写出到GM时512Byte对齐，等全量结果写完后，再启动UB上数据重排，写入到GM上。

<img src="https://raw.gitcode.com/user-images/assets/7801479/13bd80b2-d6ab-4520-839c-cf5e700db0d5/7rmPadding3.png" alt="写出重排方式三：局部workSpace，NZ写出到GM时按512Byte对齐（n0尾部补dummy），按block粒度在UB上重排为ND后写回GM" width="80%">

（↑）**方式三**：使用局部workSpace，NZ写出到GM时512Byte对齐，再按block块粒度在UB上重排，ND写出到GM上。

<img src="https://raw.gitcode.com/user-images/assets/7801479/204d24e9-5dc8-4318-bfe1-7f958598c514/7rmPadding4.png" alt="写出重排方式四：全量workSpace，NZ写出到GM时按512Byte对齐（N尾部补dummy），全量写完后在UB上重排为ND后写回GM" width="80%">

（↑）**方式四**：使用全量workSpace，NZ写出到GM时512Byte对齐，等全量结果写完后，再启动UB上数据重排，ND写出到GM上。

选择依据：方式一、方式三使用局部workSpace，额外GM占用小，适合输出规模较大或GM容量紧张的场景；方式二、方式四使用全量workSpace，需额外申请与输出矩阵等大的GM空间，可在全量结果写完后统一重排。ND写出（方式一、方式二）无需额外的格式转换，NZ写出（方式三、方式四）搬运带宽更高，但需要NZ2ND重排。

### 特性承载代码

- [padding_matmul.hpp](../../../../include/catlass/gemm/kernel/padding_matmul.hpp)中实现了包含**方式二**的`RemovePaddingNDAndCast`后处理组件
- 实际适配可参考[34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)
- 方式一、方式三、方式四当前无对应承载组件，需参考上述流程自行实现

</details>

## 模板应用

参考[102_dynamic_optimized_matmul的select_kernel策略](../../../../examples/102_dynamic_optimized_matmul/include/select_kernel_b16.h)

<details>
<summary><strong><font size="4">模板选择</font></strong></summary>

先尝试基于[00_basic_matmul](../../../../examples/00_basic_matmul/basic_matmul.cpp)进行TileShape调优并获得**性能基线**，可参考[模板库优化指引](../../1_Practice/11_matmul_optimization.md)

依次识别是否满足各模板适合场景，并和性能基线作比较。比较方法为使用[msProf](../../1_Practice/evaluation/performance_tools.md#msprof-single-operator-analysis)分别采集候选模板与基线的Task Duration，在相同shape和数据类型下对比耗时，确认收益后再选定模板：

- [31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp)：
- 计算当前基本任务块taskBlocks

    ```cpp
    taskBlocks = CeilDiv(M, m1) * CeilDiv(N, n1);
    ```

  - 基本任务块小于AIC数：$taskBlocks < aicCoreNum$
  - $K$轴较小，$K <= k_1$
- [09_splitk_matmul](../../../../examples/09_splitk_matmul/splitk_matmul.cpp)或[22_padding_splitk_matmul](../../../../examples/22_padding_splitk_matmul/padding_splitk_matmul.cpp)（带Padding前处理）
  - 选择$m_1$、$n_1$、$k_1$
    - 先设置$m_1 = 128$、$n_1 = 256$、$k_1 = 256$
    - 满足下列场景其一，修改$m_1 = 256$、$n_1 = 128$
      - A矩阵和B矩阵均为ColumnMajor
      - A矩阵为ColumnMajor，B矩阵为RowMajor，且$M > N$
- 计算当前基本任务块taskBlocks

    ```cpp
    taskBlocks = CeilDiv(M, m1) * CeilDiv(N, n1);
    ```

  - 满足下列两种场景之一：
    - 基本任务块少于一半AIC数，且$K$轴够深：$taskBlocks < aicCoreNum / 2$ 且 $K > 5120$
    - 基本任务块不超过2块，且$K$轴较深：$taskBlocks \le 2$ 且 $K > 1024$
  - 阈值说明：切K的收益来自沿$K$方向分摊基本块以用满AIC，代价是每份部分和写出workspace、全核同步和Vector累加，$K$越短，这部分固定开销占比越高。按$k_1 = 256$计，$K > 1024$对应$K$方向至少4个$k_1$基本块，也是[splitk_matmul.hpp](../../../../include/catlass/gemm/kernel/splitk_matmul.hpp)中`GetSplitkFactor`允许切分份数由2份提升到4份的下界；$K > 5120$对应至少20个$k_1$基本块，用于$taskBlocks < aicCoreNum / 2$时每份仍能保留足够多的基本块以摊薄累加开销。上述阈值为模板调优得到的经验值，AIC核数或$k_1$不同时需按比例重新评估，推荐范围见[09_splitk_matmul](../../../../examples/09_splitk_matmul/README.md)
- [06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp)（带Padding前处理）和[21_basic_matmul_preload_zN](../../../../examples/21_basic_matmul_preload_zN/basic_matmul_preload_zN.cpp)（手动改为ND输入）
  - 泛化性更强，适用于剩余场景
  - 不需要使用Padding时，为了节约MIX算子编译启动的开销，建议使用[21_basic_matmul_preload_zN](../../../../examples/21_basic_matmul_preload_zN/basic_matmul_preload_zN.cpp)模板
  - 该样例默认使用NZ排布输入；手动改为ND输入时，将[basic_matmul_preload_zN.cpp](../../../../examples/21_basic_matmul_preload_zN/basic_matmul_preload_zN.cpp)中的`LayoutB`由`layout::zN`改为`layout::RowMajor`（转置场景改为对应的ND排布），并同步修改[basic_matmul_preload.hpp](../../../../include/catlass/gemm/kernel/basic_matmul_preload.hpp)中`ToUnderlyingArguments()`的排布处理

全载特性[25_matmul_full_loadA](../../../../examples/25_matmul_full_loadA/matmul_full_loadA.cpp)适用于$K \le 1024$且$N$较大的场景（如$N \ge 2048$），$N$越大A矩阵复用次数越多；单核切K方案[34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)适用于$M、N \ge 2048$且$K \ge 4000$的大尺寸场景。详见两个样例的模板推荐场景。

</details>

<details>
<summary><strong><font size="4">Padding选择</font></strong></summary>

当Stride非512Byte对齐时可以考虑使用Padding前处理，但需要考虑Padding带来的开销以及MIX算子编译启动的开销（小shape场景使用[31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp)时不推荐额外适配Padding）。

`PaddingMatrixND`、`PaddingMatrixBlockND`和`PaddingMatrixNZ`需结合具体case评估：在相同shape与数据类型下，使用[msProf](../../1_Practice/evaluation/performance_tools.md#msprof-single-operator-analysis)分别采集三种方式（包含Padding前处理开销）的Task Duration，取耗时最低者；泛化上`PaddingMatrixNZ`更具有优势，可作为默认起点。

</details>
