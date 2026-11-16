# Performance Bottleneck Analysis and Tuning Methods

## Before You Start

This document describes how to use performance tuning tools to obtain profile data, locate performance bottlenecks, and select corresponding tuning strategies for different bottleneck types during CATLASS operator development.

## Test Environment

All measured data and criteria in this document are collected in the following environment; later sections do not repeat it:

| Item | Configuration |
| --- | --- |
| Processor | Ascend 910B3 (20 AI Cores, 1.8 GHz) |
| CANN toolkit | 9.2.0-beta.2 |
| Profile data collection | `msprof op` |

## Profile Data Collection Tools

CATLASS sample projects adapt to the mainstream performance tuning tools provided by CANN. For details about how to use these tools, see [CATLASS Performance Profiling and Tuning](../08_evaluation.md).

### msProf — Single-Operator Profiling

[msProf](https://www.hiascend.com/en/document/redirect/CannCommunitymsOpProf) is a single-operator profiling tool. Its command is `msprof op`. It supports both on-board and simulation execution modes. For details, see [msProf Single-Operator Profiling](./performance_tools.md#single-operator-profiling-using-msprof).

#### On-board Profiling

Run the operator directly on the NPU to collect actual profile data. This is the first step in performance tuning.

```bash
# Build the example from the repository root; the executable is installed to output/bin/00_basic_matmul
bash scripts/build.sh 00_basic_matmul

# Change to the directory of the executable and run
cd output/bin
msprof op --application="./00_basic_matmul 256 512 1024 0"
```

Here `./00_basic_matmul` is the executable file name, `256`, `512`, and `1024` are the M-, N-, and K-axis dimensions respectively, and the trailing `0` is the Device ID (optional, defaults to 0).

Common options:

| Option| Description|
| --- | --- |
| `--application` | Specifies the executable file and its arguments (mutually exclusive with `--config`)|
| `--config` | Specifies the operator binary file `.o` (mutually exclusive with `--application`)|
| `--kernel-name` | Name of the operator to collect (fuzzy match supported)|
| `--launch-count` | Maximum number of operators to collect (default: 1)|
| `--warm-up` | Number of warm-up times (default: 5). For small shapes, consider increasing to 30 for chip frequency increasing|
| `--output` | Output path of profile data (default: current directory)|

Key profile data files generated after collection:

| File| Content| Purpose for Bottleneck Analysis|
| --- | --- | --- |
| `PipeUtilization.csv` | Execution time and proportion of each pipeline (Cube/Vector/MTE2/MTE3)| Identify the pipeline-bound stage|
| `ArithmeticUtilization.csv` | Cycle proportion of Cube/Vector instructions| Evaluate compute unit utilization|
| `L2Cache.csv` | L2 cache hit ratio| Check whether movement efficiency is affected by cache|
| `Memory.csv` | Read/write bandwidth of UB, L1, and main memory (GB/s)| Evaluate bandwidth utilization|
| `MemoryL0.csv` | Read/write bandwidth of L0A, L0B, L0C| Evaluate L0 data read/write bandwidth rate|
| `MemoryUB.csv` | Read/Write bandwidth from vector/scalar to UB| Evaluate UB access efficiency|
| `OpBasicInfo.csv` | Basic operator information (such as Block Dim)| Analyze multi-core utilization, operator execution duration, and frequency|
| `ResourceConflictRatio.csv` | Proportion of UB Bank Group/Bank conflicts| Evaluate UB access conflict severity|

#### Profiling Pipeline Simulation

After on-board profiling shows pipeline bottlenecks, use simulation to further locate issues at the instruction level.

```bash
# Enable the simulator mode during build
bash scripts/build.sh --simulator 00_basic_matmul

# Load the simulator binary path as printed by the build output; replace Ascendxxxyy with the actual product directory
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib/libruntime_camodel.so:/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascendxxxyy/lib/libnpu_drv_camodel.so

# Change to the directory of the executable and run
cd output/bin
msprof op simulator ./00_basic_matmul 256 512 1024 0
```

The simulation generates `simulator/trace.json` (instruction trace, viewable via Chrome Trace Viewer or Perfetto) and `simulator/visualize_data.bin` (MindStudio Insight visualization data, including the trace, code hotspot map, and memory hotspot map).

Trace visualization methods:

- Chrome Trace Viewer: Enter `chrome://tracing` in the Chrome address bar and drag the `trace.json` file into the window. Shortcut keys: `W` to zoom in, `S` to zoom out, `A` to move left, and `D` to move right.
- Perfetto: Visit [ui.perfetto.dev](https://ui.perfetto.dev/) and import the `trace.json` file.
- MindStudio Insight: Import the `visualize_data.bin` file to view instruction execution in a sequence diagram. You can analyze the instruction details, execution time, call stack, and synchronization relationship between pipelines.

All three tools provide the same functionality; use whichever is available.

Notes:

- To view code hotspot maps, add the following code at the beginning of `examples/CMakeLists.txt`:

    ```cmake
    add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASC>:-Xaicore-start -g -Xaicore-end>")
    ```

- If a large number of `Vector` operations in the simulation results are mapped to `Scalar` operations, causing abnormal performance (`vector_ratio < 10%`), this is caused by the compilation optimization level. You can add the following code at the beginning of `examples/CMakeLists.txt` to raise the optimization level:

    ```cmake
    add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASC>:-Xaicore-start -O3 -Xaicore-end>")
    ```

- Simulation can only run on card 0; the NPU ID cannot be specified.

### Profiling — Whole-Network Profiling

[Profiling](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0001.html) is a whole-network profiler. Its corresponding command is `msprof`. Although CATLASS primarily targets single-operator scenarios, single-operator call samples can also be profiled using this tool.

```bash
# Build the example from the repository root; --enable_profiling is deprecated and not needed, msprof enables profiling automatically when it launches the program
bash scripts/build.sh 00_basic_matmul

# Change to the directory of the executable and run
cd output/bin
msprof ./00_basic_matmul 256 512 1024 0
```

For details, see [Whole-Network Profiling](./performance_tools.md#whole-network-profiling). For details about the fields in the profile data, see [msProf Profile Data File Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0057.html).

### msTuner_CATLASS — Tiling Auto-Tuning

[msTuner_CATLASS](../../../../tools/tuner/README.md) is a tiling parameter auto-tuning tool for CATLASS operators. It supports custom search spaces (L0/L1 TileShape, data type, memory layout, Swizzle strategy, etc.; Swizzle determines the order in which AI Cores process basic blocks, see [Swizzle Design](../../2_Design/01_kernel_design/02_swizzle.md)). It can instantiate all operators in the search space and run on-board performance tests in batches, outputting the top N configurations with optimal performance.

```bash
# Build from the repository root; -DCATLASS_LIBRARY_KERNELS filters the operators to tune
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=00_basic_matmul mstuner_catlass

# Set the library path required by the tool, then run the tuning
export LD_LIBRARY_PATH=$PWD/output/lib64/:$LD_LIBRARY_PATH
./output/bin/mstuner_catlass --m=256 --n=512 --k=1024 --device=0 --output=results.csv
```

Common options:

| Option     | Default | Description                                                    |
| ---------- | ------- | -------------------------------------------------------------- |
| `--m`      | 256     | Size of the m axis of the input matrix                         |
| `--n`      | 512     | Size of the n axis of the input matrix                         |
| `--k`      | 1024    | Size of the k axis of the input matrix                         |
| `--device` | 0       | ID of the device to run on                                     |
| `--output` | none    | Path of the profile data file; no file is written if omitted   |

All these options are optional and fall back to their defaults when omitted. For filter options such as `--kernels` and `--A/--B/--C`, see the [msTuner_CATLASS](../../../../tools/tuner/README.md).

The output echoes the `case_id`, `task_duration(us)`, `device_id`, `operation`, `description`, `l0_tile_shape`, `l1_tile_shape`, `swizzle`, `m`, `n`, `k`, `A`, `B`, and `C` fields for each case, and ends with a summary of the top 10 optimal configurations sorted by `task_duration(us)`; the summary columns are `case_id`, `task_duration(us)`, `device_id`, `operation`, `description`, `m`, `n`, `k`, `A`, `B`, `C`. The search space configuration supports both entry-level and advanced modes. For details, see [msTuner](../../../../tools/tuner/README.md). It is recommended that the search space be limited to 5000 to avoid excessive build time.

### Tool Selection Recommendations

| Stage| Recommended Tool| Purpose|
| --- | --- | --- |
| Initial performance test| `msprof op` (on-board)| Obtain actual operator execution time and pipeline ratios; determine whether performance meets expectations|
| In-depth locating| `msprof op simulator` (simulation)| View instruction-level trace to locate the pipeline stall cause|
| Whole-network scenario| Profiling (`msprof`)| Profile operator performance in the context of the whole network|
| Tiling parameter selection| `mstuner_catlass` (auto-tuning) | Auto search for optimal combinations of TileShape and Swizzle|

## Theoretical Performance Calculation

Before analyzing profile data, calculate the theoretical performance values as a reference. Theoretical values represent the ideal upper bound of operator performance and are used to gauge the room for tuning.

### Theoretical Execution Time of Movement Pipelines

The theoretical execution time of movement-related pipelines (MTE1/MTE2/MTE3) is calculated as "amount of data moved (bytes) divided by theoretical bandwidth". For Atlas A2 training series products, the on-chip memory peak bandwidth is 1.6 TB/s, as described in [Atlas 800T A2 Training Server User Guide - Technical Specifications](https://support.huawei.com/enterprise/zh/doc/EDOC1100317202/f3dba488). The theoretical movement time for a `float16` matrix of size 4096*4096 is:

```bash
2*4096*4096 / 1.6TB/s ≈ 20.97 μs
```

Note the following three points:

- The on-chip memory peak bandwidth differs between products. Replace it with the bandwidth value in the specification table of the target model.
- When multiple movement instructions execute simultaneously, they share the bandwidth. For example, when MTE2 and MTE3 perform concurrent GM reads and writes, the total execution time is the sum of their data volumes divided by the GM bandwidth.
- For small data volumes, bandwidth utilization is low, and measured performance will not reach the theoretical bandwidth; the actual effective bandwidth should be used instead.

### Theoretical Execution Time of Computation Pipelines

The theoretical execution time of computation-related pipelines (Vector/Scalar) is calculated as "amount of computation (elements) divided by theoretical compute"; the Cube pipeline is calculated as "FLOPs (2*m*n*k) divided by theoretical compute". Using the Vector theoretical peak compute of 11.06 TFLOPS for `float16` as an example, the theoretical execution time for a single instruction operating on 32768 `float16` elements is:

```bash
32768 / 11.06TFLOPS ≈ 3 ns
```

Note: the value 11.06 TFLOPS is derived from the bytes processed per cycle by the Vector Unit, the number of Vector cores, and the core frequency. These parameters differ between models, so replace them with the actual specification values of the target model; the core count can be queried through the `GetCoreNumAiv` interface of `platform_ascendc`.

Cube and Vector/Scalar are calculated separately and then summed, because the three can execute in parallel to a certain extent. In practice, the larger of the theoretical movement-pipeline and computation-pipeline execution times is used as the primary reference.

## Performance Bottleneck Analysis Methods

After obtaining profile data and calculating theoretical values, processes that differ significantly from theoretical values or have high execution times are the bottlenecks. The following four analysis methods are recommended; use them in combination based on the actual situation.

### On-board Profiling: Pipeline Analysis

Analyze the utilization of each pipeline using the `PipeUtilization.csv` file.

Key metrics:

| Metric| Meaning|
| --- | --- |
| `aic_mac_ratio` | Cube pipeline utilization. A value closer to 100% indicates higher compute unit utilization.|
| `aic_mte2_ratio` | MTE2 pipeline utilization|
| `aiv_mte2_time` / `aic_mte2_time` | Actual MTE2 execution time (μs)|
| `aiv_vec_time` | Vector instruction execution time (counted in both SIMT (Single Instruction Multiple Threads) and SIMD (Single Instruction Multiple Data) scenarios)|

The analysis steps are as follows:

1. Calculate the theoretical movement time and theoretical computation time based on the operator's data volume.
2. Calculate the movement time ratio, that is, `aic_mte2_time` / theoretical movement time, and use the ratio for the following judgment:
   - Not greater than 1.5: This is the normal range. Unaligned GM addresses, split movement instructions, and ND2NZ on-the-fly conversion all add overhead. The CANN best practice [Using 512-Byte Alignment for the GM Address](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0014.html), measured on Atlas A2 series products, shows that in the worst case (32-byte alignment) the bandwidth is only 70% of that with 512-byte alignment, equivalent to about 1.43 times the theoretical time.
   - Greater than 1.5 and not greater than 2: Check movement address alignment, the size of each movement (a single movement of 16 KB or more is required to achieve good bandwidth), and whether double buffering is enabled.
   - Greater than 2: This is judged as redundant data movement or severely low movement efficiency. The former means that data already moved in cannot be reused between basic blocks (for example, the total input size exceeds the L1 capacity), and the latter means that the effective bandwidth is less than half of the theoretical bandwidth.
3. Check `aic_mac_ratio` and use empirical thresholds to determine whether the compute units are fully utilized:
   - Not less than 70%: The Cube pipeline is nearly saturated, and the compute side is not the bottleneck.
   - Less than 70%: The compute units are underutilized. Determine whether the operator is MTE2 bound based on step 2.
   - Less than 20%: The compute units are essentially idle. Check data movement and pipeline overlap first.
4. Determine the bottleneck type based on the comparison results above.

| Symptom| Bottleneck Type| Optimization Target|
| --- | --- | --- |
| MTE2 execution time comparable to total execution time| MTE2 bound | Pipeline + Tiling|
| High Cube/Vector execution time| Compute bound | Computation pipeline|
| Neither close to theoretical value| Insufficient pipeline overlap| Further analysis using simulation trace|

Example: A matmul operator with shape (2048, 12288)*(12288, 6144) and bfloat16 type. Based on the 1.6 TB/s bandwidth of Atlas A2 training series products, the theoretical movement time is approximately 125.8 μs. In this scenario, the total input size exceeds the L1 capacity (512 KB on Atlas A2 series products, see [NPU Architecture Version 2201](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.md)), and matrices A and B cannot be reused between basic blocks. The measured `aic_mte2_time` is approximately 1215 μs and the Task Duration is approximately 1276 μs, giving a movement time ratio of approximately 9.7, with an `aic_mte2_ratio` of approximately 0.96. This is judged as redundant data movement according to the criteria above. Further analysis should combine tiling optimization and simulation traces.

### On-board Profiling: Tiling Analysis

Analyze the multi-core utilization based on the Block Dim information in the `OpBasicInfo.csv` file. If Block Dim does not reach the upper limit of available cores on the hardware (for example, an AI processor has 48 Vector cores but Block Dim < 48), there is wasted compute capacity. Prioritize adjusting the tiling policy to involve more cores in computation. If you are not sure about the optimal TileShape combination, use msTuner_CATLASS for automatic search.

### Simulation Trace: Pipeline Analysis

When on-board profiling shows pipeline bottlenecks but `PipeUtilization.csv` alone cannot pinpoint the specific cause, use simulation traces to analyze at the instruction level.

Key observations:

| Symptom| Possible Cause| Troubleshooting|
| --- | --- | --- |
| Regular stalls in MTE2/MTE3| Data movement not fully overlapping with computation| Check the arrangement of movement and computation instructions and the double-buffering policy|
| Long gap between Cube and Vector| Waiting due to data dependencies| Check the CV pipeline synchronization points and intermediate data paths|
| A persistently idle pipeline| Insufficient task volume for that pipeline, or it is blocked| Check the task allocation of each pipeline decided by TileShape|
| Significant trace differences across cores| Multi-core load imbalancing| Check core distribution policy and tail tile processing|

The priority order for optimizations is: pipeline optimization > tiling optimization > memory optimization.

### On-Board Profiling: Launch Overhead Analysis

Launch overhead includes kernel launch, instruction fetch TLB misses, bank conflicts, variable resource initialization, etc. For inference operators with microsecond latency, launch overhead accounts for a significant portion and is worth tuning. Using the Atlas A2 training/inference products as an example, the full-core launch overhead is approximately 20–21 μs.

The operation steps are as follows:

1. Construct an empty kernel: start from the target example, keep the number of cores (Block Dim) and the Kernel Type configuration, and remove the movement and computation logic so that the kernel only goes through the launch process.
2. Collect empty-kernel profile data: after building, run `msprof op` as described in [msProf Single-Operator Profiling](./performance_tools.md#single-operator-profiling-using-msprof) to obtain the profile data directory.
3. Read the launch overhead: check the kernel execution duration (Task Duration) in `OpBasicInfo.csv`. For an empty kernel, this duration is the launch overhead under that core configuration.
4. Adjust the core configuration and repeat steps 1 to 3, recording the launch overhead for each core count.
5. Adjust the Kernel Type and repeat steps 1 to 3. Kernel Type determines the core type used by the operator: AIC is the Cube core, AIV is the Vector core, and MIX means AIC and AIV work together. It is set through the `KERNEL_TASK_TYPE_DEFAULT` macro at the kernel entry, as in `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` in [`x_attention_device.hpp`](../../../../examples/78_x_attention/x_attention_device.hpp).
6. Balance launch overhead against compute parallelism and select the core count and Kernel Type combination with the best overall duration.

## Optimization Methods

Select the corresponding optimization strategy based on the bottleneck analysis results. The following table lists the measured gains of each optimization method; the collection environment is described in [Test Environment](#test-environment). All inputs are `float16`; the data are the median Task Duration (5 runs; 2 runs for the padding preprocessing group), and the gain is calculated as `(baseline − candidate)/baseline`:

| Optimization category | Baseline | Candidate | `m*n*k` | Baseline time (μs) | Candidate time (μs) | Gain |
| --------------------- | -------- | --------- | ------- | ------------------ | ------------------- | ---- |
| Pipeline optimization (Preload, ShuffleK) | `00_basic_matmul` | `06_optimized_matmul` | `4096*4096*4096` | 532.39 | 507.69 | 4.6% |
| Full A load (A matrix reuse) | `00_basic_matmul` | `25_matmul_full_loadA` | `256*4096*1024` | 21.44 | 18.38 | 14.3% |
| Multi-core split-K | `00_basic_matmul` | `09_splitk_matmul` | `1024*256*8192` | 70.72 | 50.34 | 28.8% |
| Padding preprocessing | `00_basic_matmul` | `04_padding_matmul` | `1000*1000*1000` | 57.07 | 34.86 | 38.9% |

The gains vary with shape, dtype, and tiling configuration. The values above indicate the order of magnitude of each method; reproduction requires paired collection with the same shape and dtype.

### Pipeline Optimization

Applicable scenarios: The simulation trace shows regular pipeline stalls, or there are obvious waiting gaps among MTE2, Cube, and Vector.

Main methods:

- Reasonably arrange the order of movement and computation instructions to overlap data movement with computation.
- Use double/triple buffering to hide movement latency.
- Reduce pipeline synchronization waiting and improve pipeline parallelism.
- In CV fusion scenarios, increase workspace stages to expand pipeline overlapping. Workspace stages is the number of buffers used to pass intermediate results between AIC and AIV; `2` means double buffering. More stages give AIC more lookahead but linearly increase workspace capacity and synchronization overhead, so a trade-off is required. For tuning methods, see [CV Fusion Tuning](../others/fused_kernel_optimization.md).

### Tiling Optimization

Applicable scenarios: Block Dim does not saturate all available cores, or the amount of data moved in a single transaction exceeds L1 buffer, causing redundant data movement.

Main methods:

- Adjust L0/L1 TileShape so that the number of tiles matches the number of AI Cores, reducing the tailing caused by the final round.
- Select appropriate Swizzle strategies to improve A/B access locality.
- Use msTuner_CATLASS to automatically search for optimal parameter combinations.

### Memory Optimization

Applicable scenarios: `Memory.csv`/`MemoryL0.csv` shows low bandwidth utilization, or `L2Cache.csv` shows low cache hit rates, or `ResourceConflictRatio.csv` shows a high proportion of UB bank conflicts.

Main methods:

- Adjust data movement granularity to match hardware bandwidth characteristics, improving bandwidth utilization.
- Optimize data layout to increase L2 cache hit rates.
- Reduce read-write conflicts on the same bank or read-read conflicts on the same bank group in UB.
- In CV fusion scenarios, prioritize using on-chip buffers for intermediate results to avoid the inefficient path of UB -> GM -> L1.

### Launch Overhead Optimization

Applicable scenarios: Inference latency is in microseconds, and launch overhead accounts for a significant part.

Main methods:

- Adjust the number of cores to balance compute parallelism and kernel launch overhead.
- Select kernel types that reduce unnecessary resource initialization.

## Tools and Documents

| Tool| Purpose| CATLASS Documentation|
| --- | --- | --- |
| msProf (`msprof op`)| Single-operator on-board profiling| [msProf Single-Operator Profiling](./performance_tools.md#single-operator-profiling-using-msprof)|
| msProf simulation (`msprof op simulator`)| Single-operator simulation trace collection| [Profiling Pipeline Simulation](./performance_tools.md#profile-pipeline-simulation)|
| Profiling (`msProf`)| Network-wide profile data collection and analysis| [Whole-Network Profiling](./performance_tools.md#whole-network-profiling)|
| msTuner_CATLASS | Tiling parameter auto-tuning| [msTuner_CATLASS](../../../../tools/tuner/README.md) |
| MindStudio Insight | Profile data visualization and analysis| [MindStudio Insight User Guide](https://www.hiascend.com/document/detail/en/mindstudio/latest/visualization_tool/MindStudioInsight/docs/en/user_guide/overview.md)|
