# Performance Bottleneck Analysis and Tuning Methods

## Before You Start

This document describes how to use performance tuning tools to obtain profile data, locate performance bottlenecks, and select corresponding tuning strategies for different bottleneck types during CATLASS operator development.

## Profile Data Collection Tools

CATLASS sample projects adapt to the mainstream performance tuning tools provided by CANN. For details about how to use these tools, see [CATLASS Performance Profiling and Tuning](../08_evaluation.md).

### msProf — Single-Operator Profiling

[msProf](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0082.html) is a single-operator profiling tool. Its command is `msprof op`. It supports both on-board and simulation execution modes. For details, see [msProf & Profiling](./performance_tools.md#single-operator-profiling-using-msprof).

**On-board Profiling**

Run the operator directly on the NPU to collect actual profile data. This is the first step in performance tuning.

```bash
msprof op --application="./00_basic_matmul 256 512 1024 0"
```

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

**Profiling Pipeline Simulation**

After on-board profiling shows pipeline bottlenecks, use simulation to further locate issues at the instruction level.

```bash
# Enable the simulator mode during build
bash scripts/build.sh --simulator 00_basic_matmul

# Load the simulator environment and execute
cd output/bin
msprof op simulator ./00_basic_matmul 256 512 1024 0
```

The simulation generates `simulator/trace.json` (instruction trace, viewable via Chrome Trace Viewer or Perfetto) and `simulator/visualize_data.bin` (MindStudio Insight visualization data, including the trace, code hotspot map, and memory hotspot map).

Trace visualization methods:

- Chrome Trace Viewer: Enter `chrome://tracing` in the Chrome address bar and drag the `trace.json` file into the window. Shortcut keys: `W` to zoom in, `S` to zoom out, `A` to move left, and `D` to move right.
- Perfetto: Visit [ui.perfetto.dev](https://ui.perfetto.dev/) and import the `trace.json` file.
- MindStudio Insight: Import the `visualize_data.bin` file to view instruction execution in a sequence diagram. You can analyze the instruction details, execution time, call stack, and synchronization relationship between pipelines.

Notes:

- To view code hotspot maps, add `add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-Xaicore-start -g -Xaicore-end")` to `examples/CMakeLists.txt`.
- If simulation results show that a large number of Vector operations are mapped to Scalar operations, causing abnormal results (`vector_ratio < 10%`), add the build optimization option `-O3` to `examples/CMakeLists.txt`.
- Simulation can only run on card 0; the NPU ID cannot be specified.

### Profiling — Whole-Network Profiling

[Profiling](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/devaids/Profiling/atlasprofiling_16_0010.html) is a whole-network profiler. Its corresponding command is `msprof`. Although CATLASS primarily targets single-operator scenarios, single-operator call samples can also be profiled using this tool.

```bash
# Enable the profiling API during build
bash scripts/build.sh --enable_profiling 00_basic_matmul

# Execute with msProf
cd output/bin
msprof ./00_basic_matmul 256 512 1024 0
```

For details, see [Whole-Network Profiling](./performance_tools.md#whole-network-profiling). For details about the fields in the profile data, see [msProf Profile Data File Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/devaids/Profiling/atlasprofiling_16_0057.html).

### msTuner_CATLASS — Tiling Auto-Tuning

[msTuner_CATLASS](../../../../tools/tuner/README.md) is a tiling parameter auto-tuning tool for CATLASS template library operators. It supports custom search spaces (L0/L1 TileShape, data type, memory layout, Swizzle strategy, etc.). It can instantiate all operators in the search space and run on-board performance tests in batches, outputting the top N configurations with optimal performance.

```bash
# Build
bash scripts/build.sh -DCATLASS_LIBRARY_KERNELS=00_basic_matmul mstuner_catlass

# Tune
./output/bin/mstuner_catlass --m=256 --n=512 --k=1024 --device=0 --output=results.csv
```

Each output line contains information such as `case_id`, `task_duration(us)`, `operation`, `l0_tile_shape`, `l1_tile_shape`, and `swizzle`. The output ends with a summary of the top 10 optimal configurations. The search space configuration supports both entry-level and advanced modes. For details, see [msTuner](../../../../tools/tuner/README.md). It is recommended that the search space be limited to 5000 to avoid excessive build time.

### Tool Selection Recommendations

| Stage| Recommended Tool| Purpose|
| --- | --- | --- |
| Initial performance test| `msprof op` (on-board)| Obtain actual operator execution time and pipeline ratios; determine whether performance meets expectations|
| In-depth locating| `msprof op simulator` (simulation)| View instruction-level trace to locate the pipeline stall cause|
| Whole-network scenario| Profiling (`msprof`)| Profile operator performance in the context of the whole network|
| Tiling parameter selection| msTuner_CATLASS | Auto search for optimal combinations of TileShape and Swizzle|

## Theoretical Performance Calculation

Before analyzing profile data, calculate the theoretical performance values as a reference. Theoretical values represent the ideal upper bound of operator performance and are used to gauge the room for tuning.

### Theoretical Execution Time of Movement Pipelines

The theoretical execution time of movement-related pipelines (MTE1/MTE2/MTE3) is calculated as "amount of data moved (bytes) divided by theoretical bandwidth". Assuming a GM peak bandwidth of approximately 1.8 TB/s, the theoretical movement time for a `float16` matrix of size 4096×4096 is:

```
2 × 4096 × 4096 / 1.8TB/s ≈ 18.64 μs
```

Two notes: When multiple movement instructions execute simultaneously, they share the bandwidth. For example, when MTE2 and MTE3 perform concurrent GM reads and writes, the total execution time is the sum of their data volumes divided by the GM bandwidth. For small data volumes, bandwidth utilization is low, and measured performance will not reach the theoretical bandwidth; the actual effective bandwidth should be used instead.

### Theoretical Execution Time of Computation Pipelines

The theoretical execution time of computation-related pipelines (Cube/Vector/Scalar) is calculated as "amount of computation (elements) divided by theoretical compute". Using the Vector theoretical peak compute of 11.06 TOPS for `float16` as an example, the theoretical execution time for a single instruction operating on 32K `float16` elements is:

```
32K / 11.06TOPS ≈ 0.003 μs
```

Cube and Vector/Scalar are calculated separately and then summed, because the three can execute in parallel to a certain extent. In practice, the larger of the two is used as the primary reference.

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
| `aiv_vec_time` | Vector instruction execution time. This item is counted in both SIMT and SIMD scenarios.|

Analysis steps: First, calculate the theoretical movement and computation times based on the operator's data volume. Then, compare `aic_mte2_time` with the theoretical movement time. If the actual value is significantly larger than the theoretical value, it indicates redundant data movement or low movement efficiency. Next, compare `aic_mac_ratio` with the theoretical level. If utilization is significantly low, it indicates that the compute units are underutilized. Finally, determine the bottleneck type.

| Symptom| Bottleneck Type| Optimization Target|
| --- | --- | --- |
| MTE2 execution time comparable to total execution time| MTE2 bound | Pipeline + Tiling|
| High Cube/Vector execution time| Compute bound | Computation pipeline|
| Neither close to theoretical value| Insufficient pipeline overlap| Further analysis using simulation trace|

Example: A matmul operator with shape (2048, 12288) × (12288, 6144) and bfloat16 type. The theoretical movement time is approximately 111.8 μs, but the `actual aic_mte2_time` is much larger than this value. The reason is that the total input data size exceeds the L1 capacity (512 KB), requiring redundant movement of matrix data. Further analysis should combine tiling optimization and simulation traces.

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

Method: Use the TaskDuration data of an empty kernel in on-board profiling to view the launch overhead of each core, and then find the optimal configuration by adjusting the number of cores and the kernel type.

## Optimization Methods

Select the corresponding optimization strategy based on the bottleneck analysis results.

### Pipeline Optimization

Applicable scenarios: The simulation trace shows regular pipeline stalls, or there are obvious waiting gaps among MTE2, Cube, and Vector.

Methods: Reasonably arrange the order of movement and computation instructions to overlap data movement with computation. Use double/triple buffering to hide movement latency. Reduce pipeline synchronization waiting and improve pipeline parallelism. In CV fusion scenarios, increase workspace stages to expand pipeline overlapping, but weigh this against the increased workspace capacity and synchronization pressure.

### Tiling Optimization

Applicable scenarios: Block Dim does not saturate all available cores, or the amount of data moved in a single transaction exceeds L1 buffer, causing redundant data movement.

Methods: Adjust L0/L1 TileShape so that the number of tiles matches the number of AI Cores, reducing final-round tailing. Select appropriate Swizzle strategies to improve A/B access locality. Use msTuner_CATLASS to automatically search for optimal parameter combinations.

### Memory Optimization

Applicable scenarios: `Memory.csv`/`MemoryL0.csv` shows low bandwidth utilization, or `L2Cache.csv` shows low cache hit rates, or `ResourceConflictRatio.csv` shows a high proportion of UB bank conflicts.

Methods: Adjust data movement granularity to match hardware bandwidth characteristics, improving bandwidth utilization. Optimize data layout to increase L2 cache hit rates. Reduce read-write conflicts on the same bank or read-read conflicts on the same bank group in UB. In CV fusion scenarios, prioritize using on-chip buffers for intermediate results to avoid the inefficient path of UB -> GM -> L1.

### Launch Overhead Optimization

Applicable scenarios: Inference latency is in microseconds, and launch overhead accounts for a significant part.

Methods: Adjust the number of cores to balance compute parallelism and kernel launch overhead. Select kernel types that reduce unnecessary resource initialization.

## Tools and Documents

| Tool| Purpose| CATLASS Documentation|
| --- | --- | --- |
| msProf (`msprof op`)| Single-operator on-board profiling| [msProf & Profiling](./performance_tools.md#single-operator-profiling-using-msprof)|
| msProf simulation (`msprof op simulator`)| Single-operator simulation trace collection| [Profiling Pipeline Simulation](./performance_tools.md#profile-pipeline-simulation)|
| Profiling (`msProf`)| Network-wide profile data collection and analysis| [Whole-Network Profiling](./performance_tools.md#whole-network-profiling)|
| msTuner_CATLASS | Tiling parameter auto-tuning| [msTuner_CATLASS README](../../../../tools/tuner/README.md) |
| MindStudio Insight | Profile data visualization and analysis| [MindStudio Insight User Guide](https://www.hiascend.com/document/detail/en/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html)|
