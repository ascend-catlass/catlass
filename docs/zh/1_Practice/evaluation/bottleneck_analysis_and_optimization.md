# 性能瓶颈分析及优化手段

## 写在前面

该文档主要说明在CATLASS算子开发过程中，如何使用性能调优工具获取性能数据、定位性能瓶颈，以及针对不同瓶颈类型选择对应的优化策略。

## 性能数据获取工具

CATLASS示例工程适配CANN提供的主流性能调测工具，工具的详细使用步骤见[CATLASS性能调测](../08_evaluation.md)。

### msProf — 单算子性能分析

[msProf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0082.html)是单算子性能分析工具，对应指令为`msprof op`，支持上板和仿真两种运行模式。使用方式见[msProf&Profiling](./performance_tools.md#msprof-single-operator-analysis)。

**上板性能采集**

直接在NPU上运行算子并采集实际性能数据，是性能调优的第一步。

```bash
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

# 加载仿真器环境后执行
cd output/bin
msprof op simulator ./00_basic_matmul 256 512 1024 0
```

仿真生成`simulator/trace.json`（指令流水图，可通过Chrome Trace Viewer或Perfetto查看）和`simulator/visualize_data.bin`（MindStudio Insight可视化数据，含流水图、代码热点图、内存热点图）。

流水图可视化方式：

- Chrome Trace Viewer：在Chrome地址栏输入`chrome://tracing`，将`trace.json`拖入窗口。快捷键：`W`放大、`S`缩小、`A`左移、`D`右移。
- Perfetto：访问[ui.perfetto.dev](https://ui.perfetto.dev/)，导入`trace.json`。
- MindStudio Insight：导入`visualize_data.bin`，以时序图展示指令执行情况，可分析指令详情、执行时间、调用栈及流水间同步关系。

注意事项：

- 若需查看代码热点图，需在`examples/CMakeLists.txt`中增加`add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-Xaicore-start -g -Xaicore-end")`。
- 若仿真结果中大量Vector操作被映射为Scalar操作导致性能异常（`vector_ratio < 10%`），可在`examples/CMakeLists.txt`中增加编译优化选项`-O3`。
- 仿真只能在0卡运行，不能指定NPU卡号。

### Profiling — 整网性能分析

[Profiling](https://www.hiascend.com/document/redirect/CannCommunitymsopprofuserguide)是整网性能分析工具，对应指令为`msprof`。虽然CATLASS主要面向单算子场景，单算子调用示例也可使用Profiling进行分析：

```bash
# 编译时使能 Profiling API
bash scripts/build.sh --enable_profiling 00_basic_matmul

# 使用 msprof 执行
cd output/bin
msprof ./00_basic_matmul 256 512 1024 0
```

详细使用教程见[用Profiling进行整网性能分析](./performance_tools.md#profiling-network-analysis)。性能数据各字段含义参见[msProf性能数据文件参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/Profiling/atlasprofiling_16_0057.html)。

### msTuner_CATLASS — Tiling自动寻优

[msTuner_CATLASS](../../../../tools/tuner/README.md)是面向CATLASS模板库算子的Tiling参数自动寻优工具。它支持自定义搜索空间（L0/L1 TileShape、数据类型、内存排布、Swizzle策略等），能够实例化搜索空间内的所有算子并批量完成上板性能测试，输出Top N性能最优配置。

```bash
# 编译
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=00_basic_matmul mstuner_catlass

# 运行寻优
./output/bin/mstuner_catlass --m=256 --n=512 --k=1024 --device=0 --output=results.csv
```

输出中每行包含`case_id`、`task_duration(us)`、`operation`、`l0_tile_shape`、`l1_tile_shape`、`swizzle`等信息，末尾汇总Top 10最优配置。搜索空间配置支持入门级和高级两种模式，详见[msTuner_CATLASS文档](../../../../tools/tuner/README.md)。建议将搜索空间规模控制在5000以内，以避免编译耗时过长。

### 工具选型建议

| 阶段       | 推荐工具                      | 目的                                         |
| ---------- | ----------------------------- | -------------------------------------------- |
| 初测性能   | `msprof op`（上板）           | 获取算子实际耗时和流水占比，判断性能是否达标 |
| 深入定位   | `msprof op simulator`（仿真） | 查看指令级流水图，定位断流原因               |
| 整网场景   | Profiling（`msprof`）         | 在整网上下文中分析算子性能                   |
| Tiling选参 | msTuner_CATLASS               | 自动搜索最优TileShape / Swizzle组合          |

## 理论性能计算

在分析性能数据前，应先计算理论性能值作为参照。理论值是算子实际性能的理想上界，用于判断优化空间。

### 搬运流水理论耗时

搬运相关流水（MTE1/MTE2/MTE3）的理论耗时计算公式为"搬运数据量（Byte）除以理论带宽"。假设GM峰值带宽约1.8 TB/s，`float16`类型的4096×4096矩阵搬运理论耗时如下：

```text
2Byte × 4096 × 4096 / 1.8TB/s ≈ 18.64 μs
```

注意两点：多条搬运指令同时存在时共享带宽，例如MTE2与MTE3同时进行GM读写时，总耗时等于两者搬运量之和除以GM带宽；小数据量场景带宽利用率低，实测性能达不到理论带宽，应以实际有效带宽为准。

### 计算流水理论耗时

计算相关流水（Cube/Vector/Scalar）的理论耗时计算公式为"计算数据量（Element）除以理论算力"。以`float16`类型Vector理论峰值算力11.06 TOPS为例，32K个`float16` Element单指令计算理论耗时如下：

```text
32K / 11.06TOPS ≈ 0.003 μs
```

Cube和Vector/Scalar分开计算后再求和，因为三者可并行执行到一定程度。实际分析中以两者中较大者为主要参照。

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
| `aiv_vec_time`                    | Vector指令执行时间。SIMT与SIMD场景均计入此项      |

分析步骤：先根据算子数据量计算理论搬运耗时和理论计算耗时；再对比`aic_mte2_time`与理论搬运耗时，若实际值远大于理论值，说明存在数据重复搬运或搬运效率低的问题；接着对比`aic_mac_ratio`与理论水平，若利用率显著偏低，说明计算单元未充分发挥算力；最后判断瓶颈类型。

| 现象                     | 瓶颈类型      | 优化方向               |
| ------------------------ | ------------- | ---------------------- |
| MTE2耗时与总执行时间持平 | MTE2 bound    | 流水优化 + Tiling优化  |
| Cube/Vector耗时占主导    | Compute bound | 计算流水优化           |
| 两者均不接近理论值       | 流水隐藏不足  | 用仿真流水图进一步分析 |

示例：MatMul算子shape为 (2048, 12288) × (12288, 6144)，bfloat16类型。理论搬运约111.8 μs，但实际`aic_mte2_time`远大于该值。原因是输入数据总大小超过L1空间（512 KB），矩阵数据需重复搬运，应结合Tiling优化和仿真流水图进一步分析。

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

头开销包含核启动、核取址TLB MISS、同地址访问冲突、变量资源初始化等时延。对于延迟为微秒级的推理算子，头开销占比较高，是值得优化的对象。以Atlas A2训练/推理系列产品为例，满核头开销约20~21μs。

操作方法：通过上板Profiling的空Kernel TaskDuration数据查看每个核的启动开销耗时，然后通过调整核数和算子Kernel Type找到最优配置。

## 优化手段

根据瓶颈分析结果选择对应的优化策略。

### 流水优化

适用场景：仿真流水图显示流水线存在规律性断流，或MTE2/Cube/Vector之间存在明显等待间隙。

主要手段：合理编排搬运与计算指令的排布，使数据搬运与计算重叠执行；通过Double Buffer / Triple Buffer机制掩盖搬运延迟；减少流水同步等待，提升各流水线并行度；CV融合场景下通过增加workspace stages扩大流水覆盖能力，但需权衡其对workspace容量和同步压力带来的开销。

### Tiling优化

适用场景：Block Dim未用满所有可用核，或单次搬运数据量超过L1缓存导致数据重复搬运。

主要手段：调整L0/L1 TileShape使基本块数量与AI Core数量匹配，减少尾轮长尾；选择合适的Swizzle策略改善A/B访问局部性；使用msTuner_CATLASS自动搜索最优参数组合。

### 内存优化

适用场景：`Memory.csv`/`MemoryL0.csv`显示带宽利用率偏低，或`L2Cache.csv`显示Cache命中率低，或`ResourceConflictRatio.csv`显示UB Bank Conflict占比高。

主要手段：调整数据搬运粒度以匹配硬件带宽特性，提高带宽利用率；优化数据排布以提高L2 Cache命中率；减少UB上同一Bank的读写冲突或Bank Group的读读冲突；CV融合场景优先使用片上缓存中转中间结果，避免"UB到GM再到L1"的低效路径。

### 头尾开销优化

适用场景：推理延迟在微秒级，头开销占比显著。

主要手段：调整核数配置，在计算并行度与核启动开销之间取得平衡；优化Kernel Type选择以减少不必要的资源初始化。

## 工具与文档索引

| 工具                                | 用途                   | CATLASS文档                                                                                                                                                       |
| ----------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| msProf（`msprof op`）               | 单算子上板性能采集     | [msProf&Profiling](./performance_tools.md#msprof-single-operator-analysis)                                                                                             |
| msProf仿真（`msprof op simulator`） | 单算子仿真流水图采集   | [性能流水仿真](./performance_tools.md#performance-pipeline-simulation)                                                                                                               |
| Profiling（`msprof`）               | 整网性能数据采集与分析 | [用Profiling进行整网性能分析](./performance_tools.md#profiling-network-analysis)                                                                                 |
| msTuner_CATLASS                     | Tiling参数自动寻优     | [msTuner_CATLASS README](../../../../tools/tuner/README.md)                                                                                                       |
| MindStudio Insight                  | 性能数据可视化分析     | [MindStudio Insight用户指南](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html) |
