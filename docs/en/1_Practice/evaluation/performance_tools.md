# Tuning Performance in a CATLASS Sample Project

CANN provides performance tuning tools for two operator development scenarios: single-operator and whole-network. These tools are [**msProf**](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0082.html) and [**Profiling**](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/Profiling/atlasprofiling_16_0010.html).

## Performance Tuning Tool Overview

### msProf

[msProf](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0082.html) is a single-operator profiler. The corresponding commands are `msprof op` and `msopprof`.

[msProf](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0082.html) collects and analyzes key profile data of operators running on Ascend AI processors. Based on the output profile data, you can quickly locate software and hardware performance bottlenecks in operators, improving the efficiency of operator profiling.

Profile data can currently be collected and automatically parsed based on various running modes (onboard or simulation) and file formats (executable files or operator binary `.o` files).

### Profiling Overview

[Profiling](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/Profiling/atlasprofiling_16_0010.html) is a whole-network profiler. Its corresponding command is `msprof`.

[Profiling](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/Profiling/atlasprofiling_16_0010.html) collects and parses the AI job runtime profile data, system data of Ascend AI processors, and other required data.

The basic `msprof` profiling commands query the basic information about profiling, including parameter description, AI job file, data storage path, and custom environment variables.

## Single-Operator Profiling Using `msProf`

Using `00_basic_matmul` as an example, this section demonstrates the profiling process using [msProf](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0082.html).

### Onboard Profiling

Through onboard profiling, you can directly measure the running time of an operator on an NPU and determine whether the performance meets the expected standard.

#### Example with `msprof op`

1. Compile the operator sample by referring to [Quick Start](../01_quick_start.md).
2. Call msProf in the format of `msprof op *optional_parameters* app [arguments]`.

```bash
msprof op --application="./00_basic_matmul 256 512 1024 0"
```

Common parameters:

| Parameter        | Mandatory        | Description                                                                                               | Value                                       | Note/Associated Parameter                             |
| ---------------- | ---------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------- |
| `--application`  | Yes (one of two) | Specifies the executable file or command.                                                                 | Valid path or command                       | Mutually exclusive with `--config`.                   |
| `--config`       | Yes (one of two) | Specifies the `.o` binary file.                                                                           | Valid path                                  | Mutually exclusive with `--application`.              |
| `--kernel-name`  | No               | Specifies the name of the operator to collect. (Fuzzy match and multi-operator collection are supported.) | Example: `"conv*"` or `"add\|mul"`          | Must be used with `--launch-count`.                   |
| `--launch-count` | No               | Specifies the maximum number of operators to collect.                                                     | Integer ranging from 1 to 100 (default: 1)  | Must be used with `--kernel-name`.                    |
| `--warm-up`      | No               | Specifies the number of warm-up times (for chip frequency increasing).                                    | Integer (default: 5)                        | For small shape scenarios, consider increasing to 30. |
| `--output`       | No               | Specifies the data output path.                                                                           | Valid path (default: the current directory) | Ensure that the path is writable.                     |

For more parameters, see [msProf Overview](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0082.html).

- ⚠ Precautions
  - The tool reads the performance of the first operator by default. When debugging with the example, results can be obtained directly. If integrating with another project where other operators may exist (even if you test only one operator), results may not be obtained if the operator name is not specified with `--kernel-name` during profiling.
  - You can set the environment variable `ASCEND_RT_VISIBLE_DEVICES` to specify the device ID for onboard debugging.

```bash
# Specify that the current process can only use devices with device IDs 0, 1, 2, and 3.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
msprof op ./00_basic_matmul 256 512 1024 0
```

#### Profile Data Description

The following is an example of the profile data folder structure:

```bash
├──dump                       # Raw profile data, which you can ignore.
├──ArithmeticUtilization.csv  # Cube/vector instruction cycle ratio. It is advised to optimize operator logic and reduce redundant computation instructions.
├──L2Cache.csv                # L2 cache hit rate, which affects MTE2. You are advised to properly plan the data transfer logic to increase the hit rate.
├──Memory.csv                 # UB, L1, and main memory read/write bandwidth (GB/s).
├──MemoryL0.csv               # L0A, L0B, and L0C read/write bandwidth (GB/s).
├──MemoryUB.csv               # Vector and scalar read/write bandwidth to/from the UB (GB/s).
├──OpBasicInfo.csv            # Basic operator information.
├──PipeUtilization.csv        # Pipe class instruction execution time and ratio. It is advised to optimize data movement logic to improve bandwidth utilization.
└──ResourceConflictRatio.csv  # Ratio of bank group conflicts, bank conflicts, and resource conflicts on UB across all instructions. It is advised to reduce/avoid read/write conflicts on the same bank or read-read conflicts on the same bank group.
```

### Profile Pipeline Simulation

Through simulation, you can obtain visual data such as **pipeline diagrams**, **instruction-to-code-line mapping**, **code hotspot maps**, and **memory hotspot maps** for further analysis to identify operator computation bottlenecks.

#### Example with msprof op simulator

1. Add the `--simulator` option to the build script to compile the operator in `simulator` mode.

    ```bash
    bash scripts/build.sh --simulator 00_basic_matmul
    ```

    - This option does not actually change the built binary program. The difference is whether it outputs the simulator path prompt in step 2.

2. After the build completes, load the simulator binary path based on the prompt.

    ```bash
    # Execute based on the actual output from step 1.
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
    export LD_PRELOAD=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib/libruntime_camodel.so:/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib/libnpu_drv_camodel.so
    ```

3. Change to the executable build directory `output/bin` and execute the operator sample program using `msprof op simulator`.

    ```bash
    cd output/bin
    # Executable file name | Matrix M-axis | N-axis | K-axis | Device ID (optional)
    msprof op simulator ./00_basic_matmul 256 512 1024 0
    ```

    - ⚠ Precautions
      - To view **code hotspot maps**, add `add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-Xaicore-start -g -Xaicore-end")` to `examples/CMakeLists.txt`.
      - If a large number of obvious `Vector` operations (such as `Add` and `Div`) are mapped to `Scalar` operations in the profiling results, causing significant anomalies in the results (`vector_ratio<10%`, `scalar>90%`), this is due to the compilation optimization level. You can add `add_compile_options($<$<COMPILE_LANGUAGE:ASCEND>:"-Xaicore-start -O3 -Xaicore-end">)` to `examples/CMakeLists.txt`.
      - Simulation can only run on card 0; the NPU ID cannot be specified.

#### Simulation Data Description

```bash
├──dump                    # Raw profile data, which you can ignore.
└──simulator               # Basic operator information.
   ├──core0.cubecore0
   ├──...
   ├──core23.cubecore0
   ├──trace.json           # File for presentation in Edge/Chrome Trace Viewer/Perfetto.
   └──visualize_data.bin   # File for presentation in MindStudio Insight.

```

#### Visualizing Profile Data

- Data visualization relies on the [MindStudio Insight](https://www.hiascend.com/en/developer/software/mindstudio/download?versionId=148), which must be downloaded and installed in advance.

##### Code Hotspot Maps

Obtain the `visualize_data.bin` file from the `simulator` output folder and load the bin file using MindStudio Insight to view the code hotspot map.

![MindStudio Insight Source](https://www.hiascend.com/doc_center/source/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/figure/zh-cn_image_0000002274910673.png)

##### Instruction Pipeline Chart

###### Presentation Using Edge/Chrome Trace Viewer/Perfetto

Choose the appropriate tool based on your browser:

- Edge Trace Viewer（在 Edge 浏览器地址栏输入 `edge://tracing`） (Microsoft Edge)
- Chrome Trace Viewer（在 Chrome 浏览器地址栏输入 `chrome://tracing`） (Google Chrome or Chrome-based browsers)
- [Perfetto](https://ui.perfetto.dev/) (general)

Import the `trace.json` file to view the simulation instruction pipeline chart.

###### Presentation Using MindStudio Insight

Obtain the `visualize_data.bin` file from the `simulator/` output folder. Load the bin file using MindStudio Insight to view the simulation pipeline chart.

![MindStudio Insight Timeline](https://www.hiascend.com/doc_center/source/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/figure/zh-cn_image_0000002274910873.png)

### Viewing More Visual Data with MindStudio Insight

Data collected by msProf can be imported into [MindStudio Insight](https://www.hiascend.com/document/detail/en/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html) for further analysis of operator computation bottlenecks. You can import the `visualize_data.bin` file into the tool to visually analyze operator performance.

![MindStudio Insight Details](https://www.hiascend.com/doc_center/source/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/figure/zh-cn_image_0000002274911037.png)

![MindStudio Insight Cache](https://www.hiascend.com/doc_center/source/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/figure/zh-cn_image_0000002274870637.png)

## Whole-Network Profiling

Although CATLASS only provides single-operator call examples, the single-operator call examples can also be analyzed for performance using [Profiling](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/Profiling/atlasprofiling_16_0010.html).

The following uses `00_basic_matmul` as an example.

### Example with `msProf`

1. Following [Quick Start](../01_quick_start.md), enable the tool's build switch `--enable_profiling` to enable the `Profiling API` for building the operator sample.

    ```bash
    bash scripts/build.sh --enable_profiling 00_basic_matmul
    ```

2. Change to the executable build directory `output/bin` and execute the operator sample program with `msProf`.

    ```bash
    cd output/bin
    # Executable file name | Matrix M-axis | N-axis | K-axis | Device ID (optional)
    msprof ./00_basic_matmul 256 512 1024 0
    ```

For details about the functions of each profile data file, see [msProf Profile Data File Reference](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/Profiling/atlasprofiling_16_0057.html).
