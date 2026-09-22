# Conv算子性能优化

## 写在前面

本文档面向Conv2D/Conv3D算子的性能调优场景，以`examples/33_basic_conv2d`为基线，介绍从任务块均衡、流水深度、Tile Shape选择到边界处理优化的实践路径。

当前`33_basic_conv2d`的默认配置：

| 参数              | 值                                          |
| ----------------- | ------------------------------------------- |
| DispatchPolicy    | `ConvAtlasA2Pingpong<2, 2, 2, 2, 1, false>` |
| FmapL1TileShape   | `<8, 12, 8>`Ho, Wo, Cin1                    |
| FilterL1TileShape | `<96, 8>`Cout, Cin1                         |
| L0TileShape       | `<16, 96, 16>`M, N, K                       |
| BlockScheduler    | `Conv2dIdentityBlockSwizzle<3, 0>`          |

任务空间维度为`{Batch, Ho, Wo, Cout, Cin1}`，基本块数量：

```text
totalLoops = batch * ceilDiv(Ho, 8) * ceilDiv(Wo, 12) * ceilDiv(Cout, 96)
```

## 调优总流程

Conv算子极致性能调优建议按以下顺序推进：

1. 先用基础模板跑通正确性，记录`Batch/Hi/Wi/Cin/Cout/Kh/Kw`、pads/strides/dilations、数据类型、可用AIC核数。
2. 使用`msprof op`获取上板性能数据，判断主要瓶颈是Cube利用率不足、MTE2读带宽不足、MTE3/Fixpipe写出不足，还是Vector后处理不足。
3. 计算任务块数量，判断与AIC核数的匹配程度。
4. 对当前瓶颈做单点优化（任务块 → Tile Shape → Multi Buffer → 边界处理），避免一次叠加多个特性。
5. 用仿真流水图检查MTE2、MTE3、Cube、Vector之间是否存在长空泡或互等。
6. 在多个候选配置性能接近时，优先选择启动开销小、workspace小、维护成本低的方案。

## 案例集

### 案例一：任务块负载均衡

**场景特征**

Conv2D的任务由`Conv2dIdentityBlockSwizzle`在`{Batch, Ho, Wo, Cout}`空间上切分。基本任务块数量由FmapL1TileShape和FilterL1TileShape共同决定：

```cpp
uint32_t hoTiles = ceilDiv(Ho, FmapL1TileShape::Ho);
uint32_t woTiles = ceilDiv(Wo, FmapL1TileShape::Wo);
uint32_t coTiles = ceilDiv(Cout, FilterL1TileShape::Cout);
uint32_t totalLoops = batch * hoTiles * woTiles * coTiles;
```

当`totalLoops`远少于AIC核数时，大量核空闲；当`totalLoops`远多于核数且不能整除时，尾轮会出现负载不均。

**优化方法**

**优先调整 tile shape 使总任务块数接近AIC核数的整数倍。** 以Atlas A2典型48核为例：

```text
期望: totalLoops ≈ N * 48  N >= 2
```

调整方向：

| 现象                           | 调整                                                    | 副作用                           |
| ------------------------------ | ------------------------------------------------------- | -------------------------------- |
| `totalLoops < 48`              | 减小`FmapL1TileShape::Ho/Wo`或`FilterL1TileShape::Cout` | 增加循环次数，可能放大MTE2搬运量 |
| `totalLoops`远大于48且尾轮不均 | 尝试调整`SwizzleOffset`和`SwizzleDirection`             | 不影响搬运量，仅重排任务分配     |

**调整 Swizzle 参数实现尾轮均衡。**`Conv2dIdentityBlockSwizzle<offset, direction>`的两种direction：

- `direction = 0`（默认）：按`Ho → Wo → Cout`顺序遍历。适合`hoTiles * woTiles`较大的场景。
- `direction = 1`：按`Cout → Ho → Wo`顺序遍历。适合`coTiles`较大的场景。

`SwizzleOffset`控制Swizzle块大小，通过改变任务分配顺序让尾轮工作更均匀地分布在核间。

**经验判断**

- `totalLoops`在`[48, 96)`区间时，尾轮负载不均问题最突出。

- 若`totalLoops`刚好是48的整数倍，Swizzle调整通常无收益。
- 小batch场景（batch=1）更容易出现任务块不足，需优先考虑减小tile shape。
- Swizzle调整不改变搬运量，仅重排任务，是低风险的优化起点。

### 案例二：流水深度与Multi-Buffer调优

**场景特征**

`ConvAtlasA2Pingpong`通过`L1A/L1B/L0A/L0B/L0C_STAGES`控制各级存储的Double/Triple Buffer深度。当前默认配置为`L1A=2, L1B=2, L0A=2, L0B=2, L0C=1, UnitFlag=false`。

`L0C_STAGES=1`意味着L0C写出到GM时，Cube必须等待写出完成才能开始下一轮计算。当K循环（Cin1维度）中Cube计算较快而写出较慢时，会在每轮K循环末尾产生空泡。

**优化方法**

**启用L0C流水`L0C_STAGES=2, ENABLE_UNIT_FLAG=true`。**

```cpp
constexpr uint32_t L0C_STAGES = 2;        // 从1改为2
constexpr bool ENABLE_UNIT_FLAG = true;    // 从false改为true
using DispatchPolicy =
    Conv::ConvAtlasA2Pingpong<2, 2, 2, 2, L0C_STAGES, ENABLE_UNIT_FLAG>;
```

原理：L0C输出buffer从1个增加到2个。Cube计算时，当前tile的L0C结果可并行写出到GM，实现Cube计算与L0C写出的重叠。

**调整L1A/L1B Stage数。** 在L1空间允许下，将`L1A_STAGES`或`L1B_STAGES`从2增加到3，Triple Buffer可以让MTE2搬运更充分地与Cube计算重叠：

```cpp
// Triple Buffer配置示例
constexpr uint32_t L1A_STAGES = 3;
constexpr uint32_t L1B_STAGES = 3;
```

L1空间预算的粗略估算：

| Buffer     | 每个stage大小                               | 3 stages 总大小     |
| ---------- | ------------------------------------------- | ------------------- |
| L1A Fmap   | `Ho × Wo × Cin1 × C0 × sizeof(fp16)`        | 3倍单stage          |
| L1B Filter | `Cin1 × Kh × Kw × Cout × C0 × sizeof(fp16)` | 3倍单stage          |
| 合计       | —                                           | 需 < 512KB（L1容量）|

**经验判断**

- 当Profiling显示MTE2占比高且Cube利用率偏低时，优先增大stage数。
- L0C流水`L0C_STAGES=2`收益在K循环较长的场景更明显（Cin1较大）。
- L1 stage从2增加到3时，需确认L1能容纳所有buffer。若超限，可考虑减小Fmap或Filter的tile shape以降低单stage大小。
- `conv_bias`样例使用`L1A=1, L1B=1, L0C=1, UnitFlag=true`，无双缓冲但有unit flag。说明非流水场景不需要L1双缓冲，但L0C unit flag对写出仍有帮助。

### 案例三：Tile Shape选择与调优

**场景特征**

Conv2D的三个Tile Shape分别控制不同存储层级上的分块粒度：

| Tile Shape                      | 控制粒度                     | 影响                  |
| ------------------------------- | ---------------------------- | --------------------- |
| `FmapL1TileShape<Ho, Wo, Cin1>` | Fmap每次加载到L1的tile大小   | Ho×Wo循环次数、L1占用 |
| `FilterL1TileShape<Cout, Cin1>` | Filter每次加载到L1的tile大小 | Cout循环次数、L1占用  |
| `L0TileShape<M, N, K>`          | Cube单次Mmad的计算粒度       | Cube利用率            |

三者通过`K_FMAP_PER_FILTER = FilterL1TileShape::Cin1 / FmapL1TileShape::Cin1`关联。每加载一块Filter tile，需要在K方向循环`K_FMAP_PER_FILTER`次Fmap tile。

**优化方法**

**增大Fmap L1TileShape的Ho/Wo，减少外层循环。** MTE2 bound时，增大Ho/Wo可以降低Fmap数据从GM到L1的重复搬运量：

```text
Fmap重复读取次数 ≈ batch × coTiles × cin1Tiles × (hoTiles × woTiles的减少比例)
```

默认`<8, 12>`的Ho×Wo=96。可尝试增大到`<16, 16>`（Ho×Wo=256），循环次数减少约2.7倍；或`<32, 12>`（Ho×Wo=384）。

**增大FilterL1TileShape的Cout。** 增大Cout可以减少Cout方向的循环次数，降低Filter的重复读取：

```text
Filter重复读取次数 ≈ batch × hoTiles × woTiles × cin1Tiles × (coTiles的减少比例)
```

默认`<96>`在Cout=80时仅1个tile。若Cout较大（如512），可从96逐步增加到128、192，直到L1空间极限。

**增大L0TileShape的M/N/K。** L0TileShape影响Cube单次Mmad的算力利用率。M/N过小会导致Cube未满负荷；过大则可能超过L0A/L0B容量。

| L0Shape         | 单次Mmad计算次数 | L0A大小    | 典型场景             |
| --------------- | ---------------- | ---------- | -------------------- |
| `<16, 96, 16>`  | 24K              | 16×16 fp16 | 当前默认             |
| `<32, 96, 32>`  | 96K              | 32×32 fp16 | Cube bound, 空间充裕 |
| `<16, 128, 16>` | 32K              | 16×16 fp16 | Cube bound, N轴增大  |

**经验判断**

- 三者需联动调整：增大Fmap/Filter tile后L1占用上升，若无法支持多stage，可先保持stage=2，减小tile shape或增大stage tradeoff。
- L0TileShape的M/N应尽量与L1TileShape的对应维度对齐，避免尾块过小。例如`FmapL1TileShape::Ho=16`时，L0的M最好能整除16（如16或32）。
- Cube bound时优先增大L0TileShape；MTE2 bound时优先增大L1TileShape。
- 可使用CATLASS的`msTuner`自动搜索最优组合，搜索空间建议控制在5000以内。

### 案例四：Padding与边界处理优化

**场景特征**

Conv2D的边界处理发生在每个输出tile的计算中。basic_conv2d kernel在每次循环中执行以下操作：

```cpp
// 计算hi有效范围（含padding修正）
int32_t hiStart = hoStart * strideH - padTop;
int32_t hiEnd = hiStart + (actualH - 1) * strideH + (kh - 1) * dilationH;
if (hiStart < 0) { blockPadTop = 0 - hiStart; hiStart = 0; }
if (hiEnd > hi - 1) { blockPadBottom = hiEnd - (hi - 1); hiEnd = hi - 1; }

// 计算wi有效范围（含padding修正）
int32_t wiStart = woStart * strideW - padLeft;
// ...类似边界检查...
```

这些边界判断和裁剪在每个tile上执行，当分批数很大时，Scalar开销不可忽略。

**优化方法**

**对输入数据预Padding。** 在Host端或AIV预处理阶段，对Fmap进行预Padding，使Device端不再需要运行时边界判断。以`padTop=padBottom=padLeft=padRight=P`为例：

```cpp
// Host端预Padding：在Fmap四周补充P行/列0值
size_t paddedHi = hi + 2 * P;
size_t paddedWi = wi + 2 * P;
// 重新计算Layout和偏移，使tile起始位置直接从(0,0)开始
```

预Padding后，Kernel内的边界判断简化为：

```text
hiStart = hoStart * strideH
hiEnd = hiStart + (actualH - 1) * strideH + (kh - 1) * dilationH
(无需边界检查，因为ho/wo范围确保不越界)
```

**对规则shape消除尾块逻辑。** 当`Ho % FmapL1TileShape::Ho == 0`且`Wo % FmapL1TileShape::Wo == 0`时，所有tile都是完整块，`actualBlockShape`恒等于`FmapL1TileShape`。可通过编译期分支跳过`GetActualBlockShape`调用：

```cpp
if constexpr (isFullTile) {
    // 固定使用FmapL1TileShape，无需运行时计算
} else {
    // 运行时计算actualBlockShape
}
```

**合并Padding写入Filter/Fmap。** 如果Filter的`Kh/Kw`固定且padding已知，可以在gen_data阶段将padding直接写入Fmap数据，使Device端Fmap的Layout可以跳过padding偏移计算。

**经验判断**

- 预Padding适合padding值固定、shape不频繁变化的部署场景。
- 若padding值随输入变化（如动态shape），预Padding需要每次重新准备数据，收益可能被Host侧开销抵消。
- 规则shape（Ho/Wo能被tile整除）的收益最大，因为可以完全消除`actualBlockShape`的运行时计算和尾块分支。
- Profiling中Scalar占比高于10%时，边界处理优化值得投入。

## 总结

Conv算子的性能优化可以参考此文档。
