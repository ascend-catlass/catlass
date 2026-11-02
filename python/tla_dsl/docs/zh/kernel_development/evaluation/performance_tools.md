---
nav_order: 20
---

# 对 CATLASS Python 算子进行性能调优

下述文档介绍使用 CATLASS Python工具链的端到端编译、功能测试及性能维测的完整流程。

## 性能调优工具简介

### msProf简介

* [msProf](https://www.hiascend.com/document/redirect/CannCommunitymsOpProf)是单算子性能分析工具，对应的指令为msprof op或msopprof。

* [msProf](https://www.hiascend.com/document/redirect/CannCommunitymsOpProf)工具用于采集和分析运行在昇腾AI处理器上算子的关键性能指标，用户可根据输出的性能数据，快速定位算子的软、硬件性能瓶颈，提升算子性能的分析效率。

### Profiling简介

* [Profiling](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProf/docs/zh/quick_start.md)是整网性能分析工具，对应的指令为msprof。

* [Profiling](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProf/docs/zh/quick_start.md)工具提供了AI任务运行性能数据、昇腾AI处理器系统数据等性能数据的采集和解析能力。

其中，msprof采集通用命令是性能数据采集的基础，用于提供性能数据采集时的基本信息，包括参数说明、AI任务文件、数据存放路径、自定义环境变量等。

## 环境配置

### 基础依赖启用

在进行 Python 开发与测试前，请先行确认基础环境已准备完毕。详细步骤请参考 [快速上手](../../../../../../docs/zh/1_Practice/01_quick_start.md)，完成 CANN 的下载安装与环境变量启用。
Python 的底层编译依赖于 AscendNPU-IR。在执行后续编译操作前，请参阅 [环境准备](../../dsl_development/build_guide/environment.md) 并按该文档指引完成至 [运行测试用例](../../dsl_development/build_guide/testing.md) 相关内容。

## 用msProf进行单算子性能分析

功能验证通过后，可使用 `msprof` 工具抓取算子在 NPU 硬件上的运行状态、算子耗时及内存搬运等性能数据。以basic_mixed为例，演示基于msProf的性能分析过程。

### 上板性能采集

通过上板性能采集，可以直接测定算子在NPU卡上的运行时间，可判断性能是否初步达到预期标准。

### msprof op 使用示例

```bash
msprof op --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py"
```

以下列举一些常用参数，获取完整参数信息请参考[msopprof模式用户指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/optool/docs/zh/user_guide/msopprof_user_guide.md)：

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `-h`, `--help` | 显示帮助信息并退出 | `msprof op -h` |
| `--application` | 指定在Device上执行的应用。指定可执行文件/执行指令。 | `msprof op --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py"`
| `--launch-count` | 可选参数，表示循环执行特定数量的特定kernel进行性能数据采集并计算平均值（需配合 --kernel-name 参数指定目标算子使用。如果不配置该参数，只会采集一组数据）。| `msprof op --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py" --kernel-name=basic_mixed --launch-count=10` |
| `--warm-up` | 可选参数，表示预热次数（解决芯片未提频问题）<br>如果不配置该参数，默认预热次数值为5。 | `msprof op --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py" --warm-up=20` |
| `--replay-mode` | 可选参数，指定算子数据采集的重放模式，默认为 kernel。<br>kernel：核函数级重放，对指定采集范围的单个算子核函数多次重放。<br>application：应用级重放，整个应用进行多次重放。<br>range：范围级重放，指定范围内的多算子整体多次重放（需配合 --mstx=on 使用）。 | `msprof op --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py" --replay-mode=kernel` |

* ⚠ 注意事项
  * 工具默认会读取执行序列中第一个算子的性能数据。在使用官方 example 进行独立调测时，通常可直接获取到目标结果；但若接入其他复杂工程，即使只跑某一个算子的用例，底层也可能伴随着框架（如 PyTorch）其他的初始化或前置算子执行。

    为了避免读取不到目标算子的结果，强烈建议性能分析时通过 --kernel-name 精准指定算子名称。对于使用 Python（如 Catlass / TLA）开发的算子，对应的算子名称即为 Python 代码中被 @tla.kernel 装饰器修饰的函数名（例如，在 basic_mixed.py 脚本中，对应的参数应配置为 --kernel-name=basic_mixed）。
  * 可设置环境变量`ASCEND_RT_VISIBLE_DEVICES`指定上板调测的Device Id号

### 性能数据说明

运行结束后，当前目录下会生成以 `OPPROF_` 开头的性能分析数据目录，其性能数据文件夹结构如下：

```bash
├──dump                       # 原始的性能数据，用户无需关注
├──ArithmeticUtilization.csv  # cube/vector指令cycle占比，建议优化算子逻辑，减少冗余计算指令
├──L2Cache.csv                # L2 Cache命中率，影响MTE2，建议合理规划数据搬运逻辑，增加命中率
├──Memory.csv                 # UB，L1和主存储器读写带宽速率，单位GB/s
├──MemoryL0.csv               # L0A，L0B，和L0C读写带宽速率，单位GB/s
├──MemoryUB.csv               # Vector和Scalar到UB的读写带宽速率，单位GB/s
├──OpBasicInfo.csv            # 算子基础信息
├──PipeUtilization.csv        # pipe类指令耗时和占比，建议优化数据搬运逻辑，提高带宽利用率
└──ResourceConflictRatio.csv  # UB上的 bank group、bank conflict和资源冲突率在所有指令中的占比，建议减少/避免对于同一个bank读写冲突或bank group的读读冲突
```

同时，终端会输出简要的 Performance Summary Report，开发者可借此初步评估硬件利用率与算子瓶颈，例如：

* `MTE2 bandwidth utilization lower than 80% when active.`：提示内存搬运通道利用率偏低。
* `aicore compute usage lower than 20%.`：提示矩阵计算核心算力未充分释放。

获取更多数据解析相关内容，请参考 [用msProf进行单算子性能分析](../../../../../../docs/zh/1_Practice/evaluation/performance_tools.md#msprof-single-operator-analysis)。

### 性能流水仿真

通过仿真，可以获得**流水图**、**指令与代码行映射**、**代码热点图**、**内存热点图**等可视化数据，以便进一步分析优化算子计算瓶颈。

### msprof op simulator使用示例

首先将 CANN 工具包中的仿真器 `lib` 目录加载到 `LD_LIBRARY_PATH` 中。请确保在执行时所处的相对路径正确，或根据实际情况将其替换为用户目录与对应的版本。

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
```

* ⚠ 注意事项:
  * 上述命令中提到的 Ascendxxxyy 为版本占位符，实际可通过执行以下命令获取（*预期输出为具体的 SoC 版本名称*）：

    ```bash
    python3 -c "import acl; print(acl.get_soc_name())"
    ```

  * 例如，当输出显示 Ascendxxxyy 时，上述指令正确写法为：

    ```bash
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
    ```

接下来执行下面这条命令（以下以basic_mixed算子为例）：

```bash
msprof op simulator --soc-version=Ascendxxxyy --core-id=0 --kernel-name=basic_mixed python3 ./examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py
```

* **关键参数解析**：
  * --soc-version：必填。明确指定目标版本（如 Ascendxxxyy，获取方式同上），仿真器将据此加载对应的硬件时钟周期与内存带宽模型。

  * --core-id：必填。指定仿真的目标核心编号（通常单核验证指定为 0 即可）。

  * --kernel-name：强烈建议配置。由于一个 Python 脚本中可能触发大量系统底层的 Kernel 运行，使用该参数可以通过名称前缀匹配（如指定 basic_mixed）精准抓取目标算子。这不仅能过滤掉无关算子的噪音，还能大幅缩短仿真所需的等待时间。

* ⚠ 注意事项
  * **命令结构简化**：在当前版本中，推荐直接将需要执行的脚本及参数追加在 msprof 完整命令的最末尾，无需再使用 --application 参数进行包裹。

     *说明：这是因为目前 msprof 的底层解析行为已全面优化为“原生直传”机制：工具在解析完自身设定的配置选项（options）后，会自动将剩余的全部字符串视作用户的目标程序及其附属参数。这种方式不仅在终端敲击时更符合直觉，也从根本上规避了繁琐的字符转义风险。*

  * **性能优化注意事项**：在编写算子的 Python 验证脚本时，务必将 Kernel 运行前用于对标的真值（Golden）计算，以及运行后的误差（Error）比对逻辑，显式指定在 CPU 侧执行（如使用 PyTorch 时声明 device="cpu" 即可）。

    *说明：这是因为如果在脚本中未严格隔离计算设备，比如直接调用框架原生的 NPU 算子来计算预期结果，会给最终生成的流水图引入海量的干扰噪音，使原本几分钟即可出结果的仿真耗时膨胀，导致工具长时间空转，甚至难以推进到实际需要采集的自定义目标算子上。*

### 仿真数据说明

仿真运行结束后，性能数据目录结构示例如下：

```bash
├──dump                    # 原始的性能数据，用户无需关注
└──simulator               # 算子基础信息
   ├──core0.cubecore0
   ├──...
   ├──core23.cubecore0
   ├──trace.json           # Edge/Chrome Trace Viewer/Perfetto呈现文件
   └──visualize_data.bin   # MindStudio Insight呈现文件
```

获取仿真输出文件夹simulator下的visualize_data.bin，通过[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md)工具加载bin文件查看代码热点图。
关于如何对仿真数据进行深度解析，请参考 [性能流水仿真](../../../../../../docs/zh/1_Practice/evaluation/performance_tools.md#performance-pipeline-simulation)；
关于如何将数据导出并进行图形化深度解析，请参考 [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md)。

## 用Profiling进行整网性能分析

虽然CATLASS只提供单算子的调用示例，但单算子调用示例也可使用[Profiling](https://www.hiascend.com/document/detail/zh/mindstudio/2600/msTT_msIT/msProf/docs/zh/quick_start.md)工具进行性能分析。

### msprof 使用示例

```bash
msprof --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py"
```

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `-h`, `--help` | 显示帮助信息并退出 | `msprof -h` |
| `--application` | 场景相关，通过参数方式传入二进制执行程序或执行脚本及参数。 | `msprof --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py"` |
| `--output=<path>` | 可选参数，收集到的性能数据的存放路径。 | `msprof --output=/home/projects/output /home/projects/main`

更多参数参考[msprof采集命令](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0010.html)

### 性能数据说明

运行结束后，性能数据目录结构示例如下（仅展示性能数据）：

```bash
├── msprof_*.db
├── mindstudio_profiler_output
│    ├── msprof_*.json
│    ├── step_trace_*.json
│    ├── xx_*.csv
...
│    └── README.txt
├── device_{id}
...
│    └── data
├── host
...
      └── data
```

*表示{timestamp}时间戳。

* device_{id}目录主要保存各个Device运行昇腾AI应用的性能原始数据和AI处理器系统原始数据。
* host目录主要保存上层应用接口（msproftx）的昇腾AI应用运行性能原始数据和Host系统原始数据。
关于如何对数据进行深度解析，请参考 [性能数据文件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/docs/zh/user_guide/profile_data_file_references.md)。
