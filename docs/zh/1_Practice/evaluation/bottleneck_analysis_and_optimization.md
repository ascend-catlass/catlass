# 性能瓶颈分析及优化手段

## 写在前面

该文档主要说明在CATLASS算子开发过程中，如何使用性能调优工具获取性能数据、定位性能瓶颈，以及针对不同瓶颈类型选择对应的优化策略。

## 测试环境

本文涉及的实测数据与判据均在以下环境采集，后文不再重复说明：

| 项目         | 配置                                    |
| ------------ | --------------------------------------- |
| 处理器       | Ascend 910B3（20个AI Core，主频1800MHz） |
| CANN toolkit | 9.2.0-beta.2                            |
| 性能数据采集 | `msprof op`                             |

## 性能数据获取工具

CATLASS示例工程适配CANN提供的主流性能调测工具，工具的详细使用步骤见[CATLASS性能调测](../08_evaluation.md)。

### msProf — 单算子性能分析

[msProf](https://www.hiascend.com/document/redirect/CannCommunitymsOpProf)是单算子性能分析工具，对应指令为`msprof op`，支持上板和仿真两种运行模式。使用方式见[msProf&Profiling](./performance_tools.md#msprof-single-operator-analysis)。

**上板性能采集**

直接在NPU上运行算子并采集实际性能数据，是性能调优的第一步。

```bash
# 在仓库根目录编译样例，生成 output/bin/00_basic_matmul
bash scripts/build.sh 00_basic_matmul

# 可执行文件名 | 矩阵m轴 | n轴 | k轴 | Device ID（可选，默认0）
msprof op --application="./00_basic_matmul 256 512 1024 0"
```

常用参数：

| 参数             | 说明                                                           |
| ---------------- | -------------------------------------------------------------- |
| `--application`  | 指定可执行文件及参数（与`--config`二选一）                     |
| `--config`       | 指定算子二进制`.o`文件（与`--application`二选一）              |
| `--kernel-name`  | 指定采集的算子名称，支持模糊匹配                               |
| `--launch-count` | 最大采集算子数量，默认1                                        |
| `--warm-up`      | 预热次数，默认5。小shape场景建议提高到30，以解决芯片未提频问题 |
| `--output`       | 性能数据输出路径，默认当前目录                                 |

采集后生成的关键性能数据文件：

| 文件                        | 内容                                        | 瓶颈分析用途                           |
| --------------------------- | ------------------------------------------- | -------------------------------------- |
| `PipeUtilization.csv`       | 各流水线（Cube/Vector/MTE2/MTE3）耗时及占比 | 定位流水线bound阶段                    |
| `ArithmeticUtilization.csv` | Cube/Vector指令cycle占比                    | 判断计算单元利用效率                   |
| `L2Cache.csv`               | L2 Cache命中率                              | 判断搬运效率是否受Cache影响            |
| `Memory.csv`                | UB、L1和主存储器读写带宽（GB/s）            | 判断带宽利用率                         |
| `MemoryL0.csv`              | L0A、L0B、L0C读写带宽                       | 判断L0数据读写带宽速率                 |
| `MemoryUB.csv`              | Vector/Scalar到UB的读写带宽                 | 判断UB访问效率                         |
| `OpBasicInfo.csv`           | 算子基础信息（Block Dim等）                 | 分析多核利用率、算子执行耗时和频率表现 |
| `ResourceConflictRatio.csv` | UB Bank Group/Bank Conflict占比             | 判断UB访问冲突程度                     |

**性能流水仿真**

当上板Profiling显示存在流水瓶颈后，通过仿真在指令级别进一步定位问题。

```bash
# 编译时启用 simulator 模式
bash scripts/build.sh --simulator 00_basic_matmul

# 按编译输出提示加载仿真器二进制路径，Ascendxxxyy 替换为实际产品目录
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib/libruntime_camodel.so:/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib/libnpu_drv_camodel.so

# 切换到可执行文件所在目录后运行
cd output/bin
# 可执行文件名 | 矩阵m轴 | n轴 | k轴 | Device ID（可选，默认0）
msprof op simulator ./00_basic_matmul 256 512 1024 0
```

仿真生成`simulator/trace.json`（指令流水图，可通过Chrome Trace Viewer或Perfetto查看）和`simulator/visualize_data.bin`（MindStudio Insight可视化数据，含流水图、代码热点图、内存热点图）。

流水图可视化方式：

- Chrome Trace Viewer：在Chrome地址栏输入`chrome://tracing`，将`trace.json`拖入窗口。快捷键：`W`放大、`S`缩小、`A`左移、`D`右移。
- Perfetto：访问[ui.perfetto.dev](https://ui.perfetto.dev/)，导入`trace.json`。
- MindStudio Insight：导入`visualize_data.bin`，以时序图展示指令执行情况，可分析指令详情、执行时间、调用栈及流水间同步关系。

三种工具功能均一致，选择任一可用的即可。

注意事项：

- 若需查看代码热点图，需在`examples/CMakeLists.txt`的开始位置增加以下代码：

    ```cmake
    add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASC>:-Xaicore-start -g -Xaicore-end>")
    ```

- 若仿真结果中大量`Vector`操作被映射为`Scalar`操作导致性能异常（`vector_ratio < 10%`），这是编译优化等级造成的，可在`examples/CMakeLists.txt`的开始位置增加以下代码，用于提升优化等级：

    ```cmake
    add_compile_options($<$<COMPILE_LANGUAGE:ASC>:"-Xaicore-start -O3 -Xaicore-end">)
    ```

- 仿真只能在0卡运行，不能指定NPU卡号。

### Profiling — 整网性能分析

[Profiling](https://www.hiascend.com/document/redirect/CannCommunitymsopprofuserguide)是整网性能分析工具，对应指令为`msprof`。虽然CATLASS主要面向单算子场景，单算子调用示例也可使用Profiling进行分析：

```bash
# 在仓库根目录编译样例；--enable_profiling 已废弃，无需指定，msprof 启动程序时会自动启用 Profiling
bash scripts/build.sh 00_basic_matmul

# 切换到可执行文件所在目录后执行
cd output/bin
# 可执行文件名 | 矩阵m轴 | n轴 | k轴 | Device ID（可选，默认0）
msprof ./00_basic_matmul 256 512 1024 0
```

详细使用教程见[用Profiling进行整网性能分析](./performance_tools.md#profiling-network-analysis)。性能数据各字段含义见[msProf性能数据文件参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/docs/zh/user_guide/profile_data_file_references.md)。

### msTuner_CATLASS — Tiling自动寻优

[msTuner_CATLASS](../../../../tools/tuner/README.md)是面向CATLASS模板库算子的Tiling参数自动寻优工具。它支持自定义搜索空间（L0/L1 TileShape、数据类型、内存排布、Swizzle策略等；Swizzle决定AI Core处理基本块的顺序，见[Swizzle设计](../../2_Design/01_kernel_design/02_swizzle.md)），能够实例化搜索空间内的所有算子并批量完成上板性能测试，输出Top N性能最优配置。

```bash
# 在仓库根目录编译；-DCATLASS_LIBRARY_KERNELS 过滤参与寻优的算子
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=00_basic_matmul mstuner_catlass

# 设置工具依赖库路径后运行寻优
export LD_LIBRARY_PATH=$PWD/output/lib64/:$LD_LIBRARY_PATH
./output/bin/mstuner_catlass --m=256 --n=512 --k=1024 --device=0 --output=results.csv
```

常用参数：

| 参数       | 默认值 | 说明                                 |
| ---------- | ------ | ------------------------------------ |
| `--m`      | 256    | 输入矩阵的m轴大小                    |
| `--n`      | 512    | 输入矩阵的n轴大小                    |
| `--k`      | 1024   | 输入矩阵的k轴大小                    |
| `--device` | 0      | 指定运行的单卡ID                     |
| `--output` | 无     | 性能数据落盘文件路径，不指定时不落盘 |

以上参数均可选，不指定时使用默认值；`--kernels`、`--A/--B/--C`等过滤参数的用法见[msTuner_CATLASS文档](../../../../tools/tuner/README.md)。

输出中逐个case回显`case_id`、`task_duration(us)`、`device_id`、`operation`、`description`、`l0_tile_shape`、`l1_tile_shape`、`swizzle`、`m`、`n`、`k`、`A`、`B`、`C`字段，末尾按`task_duration(us)`汇总Top 10最优配置，汇总列依次为`case_id`、`task_duration(us)`、`device_id`、`operation`、`description`、`m`、`n`、`k`、`A`、`B`、`C`。搜索空间配置支持入门级和高级两种模式，见[msTuner_CATLASS文档](../../../../tools/tuner/README.md)。建议将搜索空间规模控制在5000以内，以避免编译耗时过长。

### 工具选型建议

| 阶段       | 推荐工具                      | 目的                                         |
| ---------- | ----------------------------- | -------------------------------------------- |
| 初测性能   | `msprof op`（上板）           | 获取算子实际耗时和流水占比，判断性能是否达标 |
| 深入定位   | `msprof op simulator`（仿真） | 查看指令级流水图，定位断流原因               |
| 整网场景   | Profiling（`msprof`）         | 在整网上下文中分析算子性能                   |
| Tiling选参 | `mstuner_catlass`（自动寻优）  | 自动搜索最优TileShape / Swizzle组合          |

## 理论性能计算

在分析性能数据前，应先计算理论性能值作为参照。理论值是算子实际性能的理想上界，用于判断优化空间。

### 搬运流水理论耗时

搬运相关流水（MTE1/MTE2/MTE3）的理论耗时计算公式为“搬运数据量（Byte）除以理论带宽”。以Atlas A2训练系列产品为例，其片上内存峰值带宽为1600GB/s，见[Atlas 800T A2 训练服务器 用户指南-技术规格](https://support.huawei.com/enterprise/zh/doc/EDOC1100317202/f3dba488)。`float16`类型的4096*4096矩阵搬运理论耗时如下：

```text
2Byte*4096*4096 / 1.6TB/s ≈ 20.97 μs
```

注意以下三点：

- 不同产品的片上内存峰值带宽不同，计算时需替换为目标型号规格表中的带宽值。
- 多条搬运指令同时存在时共享带宽，例如MTE2与MTE3同时进行GM读写时，总耗时等于两者搬运量之和除以GM带宽。
- 小数据量场景带宽利用率低，实测性能达不到理论带宽，应以实际有效带宽为准。

### 计算流水理论耗时

计算相关流水（Cube/Vector/Scalar）的理论耗时计算公式为“计算数据量（Element）除以理论算力”。以`float16`类型Vector理论峰值算力11.06TFLOPS为例，32768个`float16` Element单指令计算理论耗时如下：

```text
32768 / 11.06TFLOPS ≈ 0.003 μs
```

注意：11.06TFLOPS由Vector单元每拍处理的字节数、Vector核数与主频折算得到，各型号的核数、主频不同，需替换为目标型号的实际规格值，核数可通过`platform_ascendc`的`GetCoreNumAiv`接口查询。

Cube和Vector/Scalar分开计算后再求和，因为三者可并行执行到一定程度。实际分析中以搬运流水与计算流水理论耗时中较大者为主要参照。

## 性能瓶颈分析方法

获取性能数据并计算理论值后，与理论值差异较大或耗时较长的流程即为瓶颈点。以下介绍四种分析方法，建议根据实际情况组合使用。

### 上板Profiling分析流水情况

通过`PipeUtilization.csv`分析各流水线利用率。

关键指标：

| 指标                              | 含义                                              |
| --------------------------------- | ------------------------------------------------- |
| `aic_mac_ratio`                   | Cube流水利用率，越接近100% 说明计算单元利用越充分 |
| `aic_mte2_ratio`                  | MTE2流水利用率                                    |
| `aiv_mte2_time` / `aic_mte2_time` | MTE2实际耗时（μs）                                |
| `aiv_vec_time`                    | Vector指令执行时间（SIMT单指令多线程、SIMD单指令多数据场景均计入） |

分析步骤如下：

1. 根据算子数据量计算理论搬运耗时和理论计算耗时。
2. 计算搬运耗时比 `aic_mte2_time` / 理论搬运耗时，按该比值判定：
   - 不大于1.5：属正常范围。GM地址未512B对齐、搬运指令拆分和ND2NZ随路转换都会带来额外开销；CANN官方最佳实践[《GM地址尽量512B对齐》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/gm_address_512b_alignment.md)在Atlas A2系列产品上的实测显示，最差情况下（32B对齐）带宽仅为512B对齐的70%，对应耗时约为理论值的1.43倍。
   - 大于1.5且不大于2：需检查搬运地址对齐、单次搬运长度（单次16KB以上才能发挥较佳带宽）以及是否启用双缓冲。
   - 大于2：判为数据重复搬运或搬运效率严重偏低。前者指基本块之间无法复用已搬入的数据（如输入数据总量超过L1容量），后者指有效带宽不足理论带宽的一半。
3. 检查`aic_mac_ratio`，按经验阈值判定计算单元是否充分发挥：
   - 不小于70%：Cube流水接近饱和，计算侧不构成瓶颈。
   - 小于70%：计算单元未充分发挥算力，需结合第2步判断是否为MTE2 bound。
   - 小于20%：计算单元基本空转，优先排查数据搬运与流水掩盖。
4. 结合以上对比结果判断瓶颈类型。

| 现象                     | 瓶颈类型      | 优化方向               |
| ------------------------ | ------------- | ---------------------- |
| MTE2耗时与总执行时间持平 | MTE2 bound    | 流水优化 + Tiling优化  |
| Cube/Vector耗时占主导    | Compute bound | 计算流水优化           |
| 两者均不接近理论值       | 流水隐藏不足  | 用仿真流水图进一步分析 |

示例：MatMul算子shape为 (2048, 12288)*(12288, 6144)，bfloat16类型。按Atlas A2训练系列产品的1600GB/s带宽计算，理论搬运约125.8μs；输入数据总大小超过L1空间（Atlas A2系列产品为512KB，见[NPU架构版本2201](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.md)），基本块之间无法复用A、B矩阵。实测`aic_mte2_time`约1215μs、Task Duration约1276μs，搬运耗时比约9.7，`aic_mte2_ratio`约0.96，按上述判据属于数据重复搬运，应结合Tiling优化和仿真流水图进一步分析。

### 上板Profiling分析Tiling情况

通过`OpBasicInfo.csv`中的Block Dim信息分析多核利用情况。若Block Dim未达到硬件可用核数上限（例如某AI处理器有48个Vector核但Block Dim < 48），则存在算力浪费。优化方向为调整Tiling策略使更多核参与计算。若不确定最优TileShape组合，可使用msTuner_CATLASS自动搜索。

### 仿真流水图分析流水情况

当上板Profiling显示存在流水瓶颈，但仅凭`PipeUtilization.csv`无法定位具体原因时，使用仿真流水图在指令级别深入分析。

观察重点：

| 现象                       | 可能原因                   | 排查方向                             |
| -------------------------- | -------------------------- | ------------------------------------ |
| MTE2/MTE3规律性断流        | 数据搬运未与计算充分重叠   | 检查搬运与计算指令的排布和双缓冲策略 |
| Cube与Vector之间存在长间隙 | 数据依赖导致等待           | 检查CV流水同步点和中间数据通路       |
| 某条流水线持续空闲         | 该流水线任务量不足或被阻塞 | 检查TileShape导致的各流水线任务分配  |
| 多个核的流水图形态差异大   | 多核负载不均衡             | 检查分核策略和尾块处理               |

优化方向优先级为流水优化优先于Tiling优化，Tiling优化优先于内存优化。

### 上板Profiling查看头开销

头开销包含核启动、核取指TLB MISS、同地址访问冲突、变量资源初始化等时延。对于延迟为微秒级的推理算子，头开销占比较高，是值得优化的对象。以Atlas A2训练/推理系列产品为例，满核头开销约20~21μs。

操作步骤如下：

1. 构造空Kernel：以目标样例为基础，保留核数（Block Dim）与Kernel Type配置，移除搬运和计算逻辑，使Kernel只完成启动过程。
2. 采集空Kernel性能数据：编译后按[msProf单算子性能分析](./performance_tools.md#msprof-single-operator-analysis)的方式执行`msprof op`，得到性能数据目录。
3. 读取头开销：在`OpBasicInfo.csv`中查看Kernel的执行耗时（Task Duration），空Kernel的执行耗时即为该核配置下的启动开销。
4. 调整核数配置，重复步骤1～3，记录不同核数下的启动开销。
5. 调整Kernel Type，重复步骤1～3。Kernel Type决定算子使用的核类型：AIC为Cube核，AIV为Vector核，MIX为AIC与AIV协同计算；取值通过Kernel入口的`KERNEL_TASK_TYPE_DEFAULT`宏设置，如[`x_attention_device.hpp`](../../../../examples/78_x_attention/x_attention_device.hpp)中的`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`。
6. 在启动开销与计算并行度之间取舍，选择综合耗时最优的核数与Kernel Type组合。

## 优化手段

根据瓶颈分析结果选择对应的优化策略。下表为各优化手段的实测收益，采集环境见[测试环境](#测试环境)，输入均为`float16`，数据为Task Duration中位数（5次，Padding前处理组为2次），收益按`(基线−候选)/基线`计算：

| 优化类别                     | 基线              | 候选                       | `m*n*k`         | 基线耗时(μs) | 候选耗时(μs) | 收益  |
| ---------------------------- | ----------------- | -------------------------- | --------------- | ------------ | ------------ | ----- |
| 流水优化（Preload、ShuffleK） | `00_basic_matmul` | `06_optimized_matmul`      | `4096*4096*4096` | 532.39      | 507.69       | 4.6%  |
| 矩阵全载（A矩阵复用）         | `00_basic_matmul` | `25_matmul_full_loadA`     | `256*4096*1024`  | 21.44       | 18.38        | 14.3% |
| 多核切K                      | `00_basic_matmul` | `09_splitk_matmul`         | `1024*256*8192`  | 70.72       | 50.34        | 28.8% |
| Padding前处理                | `00_basic_matmul` | `04_padding_matmul`        | `1000*1000*1000` | 57.07       | 34.86        | 38.9% |

收益随shape、dtype与Tiling配置变化，上表数值用于判断各手段的量级，复现时需使用相同shape与dtype成对采集。

### 流水优化

适用场景：仿真流水图显示流水线存在规律性断流，或MTE2/Cube/Vector之间存在明显等待间隙。

主要手段：

- 合理编排搬运与计算指令，使数据搬运与计算重叠执行。
- 通过Double Buffer / Triple Buffer机制掩盖搬运延迟。
- 减少流水同步等待，提升各流水线并行度。
- CV融合场景下通过增加workspace stages扩大流水覆盖能力。workspace stages是AIC与AIV之间中转中间结果的缓冲区份数，`2`即双缓冲；增加份数可扩大AIC提前量，但会线性增加workspace容量和同步开销，需权衡取舍，调优方法见[CV融合调优](../others/fused_kernel_optimization.md)。

### Tiling优化

适用场景：Block Dim未用满所有可用核，或单次搬运数据量超过L1缓存导致数据重复搬运。

主要手段：

- 调整L0/L1 TileShape使基本块数量与AI Core数量匹配，减少尾轮造成的长尾耗时。
- 选择合适的Swizzle策略改善A/B访问局部性。
- 使用msTuner_CATLASS自动搜索最优参数组合。

### 内存优化

适用场景：`Memory.csv`/`MemoryL0.csv`显示带宽利用率偏低，或`L2Cache.csv`显示Cache命中率低，或`ResourceConflictRatio.csv`显示UB Bank Conflict占比高。

主要手段：

- 调整数据搬运粒度以匹配硬件带宽特性，提高带宽利用率。
- 优化数据排布以提高L2 Cache命中率。
- 减少UB上同一Bank的读写冲突或Bank Group的读读冲突。
- CV融合场景优先使用片上缓存中转中间结果，避免“UB到GM再到L1”的低效路径。

### 头开销优化

适用场景：推理延迟在微秒级，头开销占比显著。

主要手段：

- 调整核数配置，在计算并行度与核启动开销之间取得平衡。
- 优化Kernel Type选择以减少不必要的资源初始化。

## 工具与文档索引

| 工具                                | 用途                   | CATLASS文档                                                                                                                                                       |
| ----------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| msProf（`msprof op`）               | 单算子上板性能采集     | [msProf&Profiling](./performance_tools.md#msprof-single-operator-analysis)                                                                                             |
| msProf仿真（`msprof op simulator`） | 单算子仿真流水图采集   | [性能流水仿真](./performance_tools.md#performance-pipeline-simulation)                                                                                                               |
| Profiling（`msprof`）               | 整网性能数据采集与分析 | [用Profiling进行整网性能分析](./performance_tools.md#profiling-network-analysis)                                                                                 |
| msTuner_CATLASS                     | Tiling参数自动寻优     | [msTuner_CATLASS README](../../../../tools/tuner/README.md)                                                                                                       |
| MindStudio Insight                  | 性能数据可视化分析     | [MindStudio Insight用户指南](https://www.hiascend.com/document/redirect/MindStudioinsight) |
