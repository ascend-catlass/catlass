---
nav_order: 20
---

# Tuning the Performance of CATLASS Python Operators

The following document describes the complete workflow of end-to-end compilation, functional testing, and performance measurement using the CATLASS Python toolchain.

## Performance Tuning Tool Overview

### msProf Overview

* [msProf](https://www.hiascend.com/document/redirect/CannCommunitymsOpProf) is a single-operator performance analysis tool whose command is `msprof op` or `msopprof`.

* The [msProf](https://www.hiascend.com/document/redirect/CannCommunitymsOpProf) tool collects and analyzes key performance metrics of operators running on Ascend AI processors. Based on the output performance data, users can quickly locate software and hardware performance bottlenecks of operators and improve the efficiency of operator performance analysis.

### Profiling Overview

* [Profiling](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProf/docs/zh/quick_start.md) is a whole-network performance analysis tool whose command is `msprof`.

* The [Profiling](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProf/docs/zh/quick_start.md) tool provides the ability to collect and parse performance data such as AI task runtime performance data and Ascend AI processor system data.

Among these, the general msprof collection commands are the basis of performance data collection, providing the basic information for performance data collection, including parameter descriptions, AI task files, data storage paths, and custom environment variables.

## Environment Configuration

### Enabling Basic Dependencies

Before developing and testing CATLASS Python, first confirm that the basic environment is ready. For detailed steps, see [Quick Start](../../../../../../docs/en/1_Practice/01_quick_start.md) to complete the CANN download, installation, and environment variable setup.
The low-level compilation of CATLASS Python depends on AscendNPU-IR. Before performing the subsequent compilation operations, see [Environment Preparation](../../dsl_development/build_guide/environment.md) and follow that document through [Running Test Cases](../../dsl_development/build_guide/testing.md).

## Single-Operator Performance Analysis with msProf

After functional verification passes, you can use the `msprof` tool to capture the operator's running state on the NPU hardware, operator duration, memory transfer, and other performance data. Taking basic_mixed as an example, this section demonstrates the msProf-based performance analysis process.

### Onboard Performance Collection

Onboard performance collection directly measures the operator's runtime on the NPU card, which can indicate whether performance initially meets expectations.

### msprof op Usage Example

```bash
msprof op --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py"
```

Some common parameters are listed below; for complete parameter information, see the [msopprof mode user guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/optool/docs/zh/user_guide/msopprof_user_guide.md):

| Parameter | Description | Example |
| --- | --- | --- |
| `-h`, `--help` | Show help information and exit | `msprof op -h` |
| `--application` | Specify the application to execute on the Device. Specify an executable file/execution command. | `msprof op --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py"` |
| `--launch-count` | Optional parameter, meaning to loop a specific kernel a specific number of times to collect performance data and compute an average (must be used together with `--kernel-name` to specify the target operator; if this parameter is not configured, only one set of data is collected). | `msprof op --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py" --kernel-name=basic_mixed --launch-count=10` |
| `--warm-up` | Optional parameter, meaning the number of warm-up runs (to address the issue of the chip not having raised its frequency).<br>If this parameter is not configured, the default warm-up count is 5. | `msprof op --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py" --warm-up=20` |
| `--replay-mode` | Optional parameter, specifying the replay mode for operator data collection; the default is kernel.<br>kernel: kernel-function-level replay, replaying the kernel function of a single operator within the specified collection range multiple times.<br>application: application-level replay, replaying the entire application multiple times.<br>range: range-level replay, replaying multiple operators within a specified range as a whole multiple times (must be used together with --mstx=on). | `msprof op --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py" --replay-mode=kernel` |

* ⚠ Notes
  * By default, the tool reads the performance data of the first operator in the execution sequence. When using an official example for standalone measurement, the target result is usually obtained directly; but when integrating into another complex project, even if only one operator's case is run, the underlying layer may also be accompanied by other initialization or preceding operators from the framework (such as PyTorch).

    To avoid failing to read the target operator's result, it is strongly recommended to specify the operator name precisely with --kernel-name during performance analysis. For operators developed with CATLASS Python (such as Catlass / TLA), the corresponding operator name is the function name decorated with the @tla.kernel decorator in the Python code (for example, in the basic_mixed.py script, the corresponding parameter should be configured as --kernel-name=basic_mixed).
  * You can set the environment variable `ASCEND_RT_VISIBLE_DEVICES` to specify the Device Id for onboard measurement.

### Performance Data Description

After the run finishes, a performance analysis data directory starting with `OPPROF_` is generated in the current directory; its performance data folder structure is as follows:

```bash
├──dump                       # raw performance data, not of concern to users
├──ArithmeticUtilization.csv  # cube/vector instruction cycle share; optimizing the operator logic to reduce redundant computation instructions is recommended
├──L2Cache.csv                # L2 Cache hit rate, which affects MTE2; planning the data transfer logic reasonably to increase the hit rate is recommended
├──Memory.csv                 # UB, L1, and main memory read/write bandwidth rates, in GB/s
├──MemoryL0.csv               # L0A, L0B, and L0C read/write bandwidth rates, in GB/s
├──MemoryUB.csv               # Vector and Scalar to UB read/write bandwidth rates, in GB/s
├──OpBasicInfo.csv            # basic operator information
├──PipeUtilization.csv        # pipe-type instruction duration and share; optimizing the data transfer logic to improve bandwidth utilization is recommended
└──ResourceConflictRatio.csv  # share of bank group, bank conflict, and resource conflict rates on UB among all instructions; reducing/avoiding read/write conflicts on the same bank or read-read conflicts within a bank group is recommended
```

Meanwhile, the terminal outputs a brief Performance Summary Report, from which developers can preliminarily assess hardware utilization and operator bottlenecks, for example:

* `MTE2 bandwidth utilization lower than 80% when active.`: indicates that the memory transfer channel utilization is low.
* `aicore compute usage lower than 20%.`: indicates that the matrix compute core's compute power is not fully used.

For more content on data parsing, see [Single-Operator Performance Analysis with msProf](../../../../../../docs/en/1_Practice/evaluation/performance_tools.md#single-operator-profiling-using-msprof).

### Performance Pipeline Simulation

Through simulation, you can obtain visual data such as the **pipeline chart**, **instruction-to-source-line mapping**, **code hotspot map**, and **memory hotspot map**, so as to further analyze and optimize operator computation bottlenecks.

### msprof op simulator Usage Example

First load the simulator `lib` directory in the CANN toolkit into `LD_LIBRARY_PATH`. Make sure the relative path you are in at execution time is correct, or replace it with your user directory and the corresponding version according to the actual situation.

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
```

* ⚠ Notes:
  * Ascendxxxyy mentioned in the command above is a version placeholder, which can actually be obtained by running the following command (*the expected output is the concrete SoC version name*):

    ```bash
    python3 -c "import acl; print(acl.get_soc_name())"
    ```

  * For example, when the output shows Ascendxxxyy, the correct form of the command above is:

    ```bash
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
    ```

Next, run the following command (taking the basic_mixed operator as an example below):

```bash
msprof op simulator --soc-version=Ascendxxxyy --core-id=0 --kernel-name=basic_mixed python3 ./examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py
```

* **Key parameter analysis**:
  * --soc-version: required. Explicitly specify the target version (such as Ascendxxxyy, obtained as above); the simulator loads the corresponding hardware clock cycles and memory bandwidth model accordingly.

  * --core-id: required. Specify the target core number for simulation (usually specifying 0 is enough for single-core verification).

  * --kernel-name: strongly recommended. Because a Python script may trigger a large number of low-level system Kernel runs, this parameter can precisely capture the target operator through name prefix matching (such as specifying basic_mixed). This not only filters out the noise of irrelevant operators but also greatly shortens the waiting time required for simulation.

* ⚠ Notes
  * **Simplified command structure**: in the current version, it is recommended to append the script to execute and its parameters directly at the very end of the complete msprof command, without wrapping them with the --application parameter.

     *Note: This is because the current low-level parsing behavior of msprof has been fully optimized into a "native pass-through" mechanism: after parsing its own configured options (options), the tool automatically treats all remaining strings as the user's target program and its attached parameters. This is not only more intuitive when typing in the terminal but also fundamentally avoids the risk of cumbersome character escaping.*

  * **Performance optimization notes**: when writing the Python verification script for an operator, be sure to explicitly specify that the Golden truth computation before the Kernel run and the Error comparison logic after the run are executed on the CPU side (for example, declare device="cpu" when using PyTorch).

    *Note: This is because if the compute device is not strictly isolated in the script, for example, if the framework's native NPU operators are called directly to compute the expected result, it introduces a huge amount of interfering noise into the finally generated pipeline chart, inflating the simulation time that originally took only a few minutes, causing the tool to idle for a long time and even making it difficult to advance to the actual custom target operator that needs to be collected.*

### Simulation Data Description

After the simulation run finishes, an example performance data directory structure is as follows:

```bash
├──dump                    # raw performance data, not of concern to users
└──simulator               # basic operator information
   ├──core0.cubecore0
   ├──...
   ├──core23.cubecore0
   ├──trace.json           # file presented by Edge/Chrome Trace Viewer/Perfetto
   └──visualize_data.bin   # file presented by MindStudio Insight
```

Take the visualize_data.bin under the simulation output folder simulator and load the bin file with the [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md) tool to view the code hotspot map.
For how to deeply parse the simulation data, see [Performance Pipeline Simulation](../../../../../../docs/en/1_Practice/evaluation/performance_tools.md#profile-pipeline-simulation);
for how to export the data and perform graphical in-depth analysis, see [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md).

## Whole-Network Performance Analysis with Profiling

Although CATLASS only provides single-operator call examples, the single-operator call examples can also use the [Profiling](https://www.hiascend.com/document/detail/zh/mindstudio/2600/msTT_msIT/msProf/docs/zh/quick_start.md) tool for performance analysis.

### msprof Usage Example

```bash
msprof --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py"
```

| Parameter | Description | Example |
| --- | --- | --- |
| `-h`, `--help` | Show help information and exit | `msprof -h` |
| `--application` | Scenario-related; pass the binary executable program or execution script and its parameters as a parameter. | `msprof --application="python examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py"` |
| `--output=<path>` | Optional parameter, the storage path for the collected performance data. | `msprof --output=/home/projects/output /home/projects/main`

For more parameters, see [msprof collection commands](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0010.html)

### Performance Data Description

After the run finishes, an example performance data directory structure is as follows (only performance data is shown):

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

*denotes the {timestamp} timestamp.

* The device_{id} directory mainly stores the raw performance data of each Device running Ascend AI applications and the raw system data of the AI processors.
* The host directory mainly stores the raw performance data of the Ascend AI application running through the upper-layer application interface (msproftx) and the raw Host system data.
For how to deeply parse the data, see [Performance Data Files](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/docs/zh/user_guide/profile_data_file_references.md).
