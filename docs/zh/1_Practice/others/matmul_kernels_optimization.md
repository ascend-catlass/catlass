# Matmul算子性能优化

## 写在前面

该文档的目标是面向Matmul算子的性能调优场景，给出一套从性能数据到样例选择、Tiling调整、流水优化、带宽优化和特殊模板切换的实践路径。

## 调优总流程

Matmul极致性能调优建议按以下顺序推进：

1. 先用基础模板跑通正确性，并记录`M/N/K`、数据类型、A/B/C layout、stride、可用AIC/AIV核数、workspace约束。
2. 使用`msprof op`获取上板性能数据，判断主要瓶颈是Cube利用率不足、MTE2读带宽不足、MTE3/Fixpipe写出不足、Vector后处理不足，还是Scalar头开销过大。
3. 计算任务块数量：

   ```cpp
   mTiles = CeilDiv(M, m1);
   nTiles = CeilDiv(N, n1);
   taskBlocks = mTiles * nTiles;
   ```

   若`taskBlocks`与AIC核数不匹配，优先调`m1/n1`或切换分核模板。

4. 对当前瓶颈做单点优化，避免一次叠加多个特性导致收益来源不可判断。
5. 用仿真流水图检查MTE2、MTE3、Cube、Vector、Scalar是否存在长空泡或互等。
6. 在多个候选模板性能接近时，优先选择启动开销小、workspace小、维护成本低的方案。

## 案例集

### 案例一：基础模板到通用高性能模板

**场景特征**

基础Matmul常见于M/N/K较规整、C矩阵任务块数量充足、没有复杂后处理的场景。此时应先选择[00_basic_matmul](../../../../examples/00_basic_matmul/basic_matmul.cpp)或[43_ascend950_basic_matmul](../../../../examples/43_ascend950_basic_matmul/basic_matmul_tla.cpp)建立基线。

对于`43_ascend950_basic_matmul`，README中已经列出`MmadPingpong`和`MmadPreloadAsyncWithCallback`支持的关键模板参数，包括`l1AStages/l1BStages/l0AStages/l0BStages/l0CStages`、`enableL1Resident`、`enableShuffleK`等。极致性能调优时，应把这些参数视作流水和缓存策略的开关，而不是固定配置。

**调优步骤**

1. **先调任务块数量。**

   对Common模板，基本任务块数量为`CeilDiv(M, m1) * CeilDiv(N, n1)`。当任务块数少于AIC核数时，优先减小`m1`或`n1`；当任务块数远多于AIC核数且每轮负载不均时，优先尝试Swizzle或StreamK。

2. **再调基本块形状。**

   在L1/L0A/L0B/L0C容量约束内，`m1/n1`越大，A/B重复读取次数越少；但`m1/n1`过大又会减少任务块数量，造成负载不均。读取量可用以下公式做一阶估计：

   ```text
   readBytes ~= sizeof(input) * M * N * K * (1 / m1 + 1 / n1)
   ```

   因此，MTE2 bound场景通常希望增大`m1/n1`；Cube利用率不足时通常希望减小`m1/n1`或引入K轴分核。

3. **启用Multi Buffer。**

   Multi Buffer是基础优化，目标是在L1/L0A/L0B/L0C上让搬运和计算重叠。绝大多数blockMmad组件默认承载此能力。调参时需要确认各级buffer stage没有超出对应片上存储空间。

4. **启用Preload。**

   当仿真流水图显示MTE2在K循环或C block切换处有空泡时，使用`MmadAtlasA2Preload`、`MmadPreloadAsyncWithCallback`等策略提前发射下一轮GM到L1的搬运。参考[06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp)、[21_basic_matmul_preload_zN](../../../../examples/21_basic_matmul_preload_zN/basic_matmul_preload_zN.cpp)和[43_ascend950_basic_matmul](../../../../examples/43_ascend950_basic_matmul/README.md)。

5. **按需启用ShuffleK。**

   当多个AIC同时从相同K分块起步，GM同地址访问冲突明显时，启用K方向错峰读取。A2/A3上大K且A/B均非全载时更值得尝试；具体收益依赖shape、layout和硬件代际。

**经验判断**

- 若shape很小，`06_optimized_matmul`这类MIX或通用优化模板可能因AIV启动和Scalar逻辑变慢，此时优先考虑`31_small_matmul`或纯Cube模板。
- 若B已经是zN格式，`21_basic_matmul_preload_zN`通常比带Padding前处理的模板更轻。
- 若启用L1常驻，常驻的A/B tile不应再按普通tile启用多buffer，否则会浪费L1空间。

### 案例二：MTE2读取带宽优化

**场景特征**

MTE2 bound往往来自三类问题：

- A/B矩阵被多个C block重复读取。
- RowMajor/ColumnMajor输入的stride非512Byte对齐，随路格式转换搬运效率低。
- 多核在同一时间访问相同GM地址，产生读取冲突。

**优化一：Padding前处理**

当stride非对齐导致GM到L1搬运带宽明显下降时，可用AIV对A/B做Padding或格式重排，使AIC后续读取更规整。仓内已有三类Padding思路：

| 方式               | 特点                      | 适用倾向                       |
| ------------------ | ------------------------- | ------------------------------ |
| `PADDING_ND`       | 只补齐stride，实现简单    | stride非对齐但转换需求轻       |
| `PADDING_BLOCK_ND` | 按L1 tile粒度重排         | stride很大或tile内访问需规整   |
| `PADDING_NZ`       | 转成贴近L1读取的zN/nZ排布 | 随路转换损失较大或泛化性能优先 |

参考[06_optimized_matmul](../../../../examples/06_optimized_matmul/optimized_matmul.cpp)和[102_dynamic_optimized_matmul/doc/CommonMatmul.md](../../../../examples/102_dynamic_optimized_matmul/docs/zh/CommonMatmul.md)。实际选择时建议用模型估算：

```text
不 Padding 耗时 ~= AIC 非对齐读取量 / 非对齐有效带宽
Padding 耗时 ~= AIV Padding 读写耗时 + AIC 对齐读取耗时 + MIX 启动开销
```

只有当Padding节省的AIC搬运时间大于AIV预处理和启动开销时，Padding才是正收益。小shape或矩阵只读取一次的场景，要谨慎启用Padding。

**优化二：大块读取与基本块选择**

在MTE2 bound场景，增大`m1/n1`可以减少A/B重复读取次数；增大`k1`或使用更高效的layout可以减少K循环中的搬运指令条数。调参时要同时检查：

- L1是否还能容纳A/B多buffer。
- L0A/L0B/L0C是否满足对应tile和stage。
- 任务块数量是否仍能打满AIC。
- 尾块是否引入明显负载不均。

**优化三：特殊小轴搬运**

当M很小或K很小，普通ND2NZ随路转换可能不是最高效的搬运方式。`102_dynamic_optimized_matmul/doc/CommonMatmul.md`中列出了逐行`DataCopy`、K=16连续搬运、M=1时改用等价RowMajor向量读取等特殊路径。此类优化适合写入泛化模板的select逻辑，而不是在单一shape上硬编码。

### 案例三：负载均衡与K轴分核

**MultiCoreSplitK**

当C矩阵较小、`taskBlocks < aicCoreNum`且K足够大时，单纯在M/N方向切块无法打满AIC。此时可使用[09_splitk_matmul](../../../../examples/09_splitk_matmul/splitk_matmul.cpp)或[22_padding_splitk_matmul](../../../../examples/22_padding_splitk_matmul/padding_splitk_matmul.cpp)将K方向也拆给多个AIC。

MultiCoreSplitK的收益来自更多AIC参与计算，代价是：

- 需要workspace保存不同K段的部分和。
- 需要AIV做ReduceAdd。
- C矩阵越大，workspace和ReduceAdd开销越高。
- 若K不够大，切K带来的并行收益可能抵不过累加开销。

因此它更适合“小C、大K”的场景。

**StreamK**

MultiCoreSplitK解决的是任务块太少的问题，但无法完全解决尾轮负载不均。比如`taskBlocks`不能整除AIC核数时，最后一轮会有部分AIC空闲。[37_streamk_matmul](../../../../examples/37_streamk_matmul/streamk_matmul.cpp)通过只切尾轮K方向任务，把尾轮工作更均匀地分给所有AIC。

StreamK的两个关键点：

- **只切尾轮。** 非尾轮本身已满核，不额外切K，避免无意义的workspace和ReduceAdd。
- **尾轮提前计算。** 将尾轮部分和提前到倒数第二轮计算，使AIV ReduceAdd能与后续Cube计算并行，减少Vector拖尾。

判断是否使用StreamK，可先计算：

```cpp
fullRounds = taskBlocks / aicCoreNum;
tailBlocks = taskBlocks % aicCoreNum;
```

若`tailBlocks == 0`，StreamK通常没有必要；若`tailBlocks`较小且K方向可切分，则StreamK更可能带来收益。

**Small模板**

对小shape，性能瓶颈经常不是Cube或MTE2，而是Scalar头开销、调度开销和通用循环逻辑。[31_small_matmul](../../../../examples/31_small_matmul/small_matmul.cpp)样例的思路是减少BlockScheduler、循环和offset计算等运行时逻辑。适用判断可以参考：

- `taskBlocks < aicCoreNum`。
- 每个AIC最多处理一个或很少几个C block。
- `K <= k1`，K方向循环较短。
- 总耗时很小，Profiling中Scalar占比明显。

### 案例四：L1常驻减少重复读取

当某个矩阵tile被多个任务块反复使用时，可考虑L1常驻。可以参考以下两个样例进行优化：

- [25_matmul_full_loadA](../../../../examples/25_matmul_full_loadA/matmul_full_loadA.cpp)：Common模板下的A矩阵全载。
- [34_single_core_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)：单核切K模板中天然考虑L1 tile复用。

L1常驻适合读取重复度高、常驻矩阵能放入L1、且常驻后剩余L1空间仍能支撑另一侧矩阵搬运的场景。常见取舍：

- 常驻能减少GM读取量，但会压缩多buffer空间。
- 常驻矩阵不应再按普通tile做多buffer。
- 需要配合专门的swizzle或任务调度，让常驻数据在相邻任务间有复用机会。

在`43_ascend950_basic_matmul`中，`enableL1Resident`的使用条件已经给出：当`mTiles = 1`且`nTiles > CoreNum`，或`nTiles = 1`且`mTiles > CoreNum`，并且`K < 2 * k1`时，可以考虑开启L1常驻；若同时开启`l0CStages = 2`，需要关注L0C空间和`enableUnitFlag`约束。

### 案例五：MTE3/Fixpipe非对齐写出优化

当M/N值较大、K较小，Cube计算相对较轻时，C写出可能成为主要瓶颈。若C的N轴或stride非对齐，Fixpipe直接写出Global Memory的效率会下降。

[46_ascend950_matmul_fixpipe_opti](../../../../examples/46_ascend950_matmul_fixpipe_opti/46_ascend950_matmul_fixpipe_opti.md)样例展示了Ascend 950上针对非对齐搬出的优化思路：将L0C结果拆分到对应Vector的UB，再由AIV使用更适合非对齐场景的搬运方式写回GM。该方案适用于：

- C矩阵写出非对齐。
- K较小、MN较大，写出占比高。
- Vector侧中转开销能被写出带宽收益覆盖。

若M/N已经对齐，该方案可能没有收益，甚至会因额外路径略有劣化。因此它应作为Fixpipe bound的定向优化，而不是默认替代普通写出。

## 总结

针对Matmul算子，可参考此文档进行优化
