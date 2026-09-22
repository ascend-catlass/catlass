---
name: catlass-tile-level-ut
description: Catlass Gemm/Tile组件单元测试编写技能。当需要为 CopyGmToL1 / CopyL1ToL0A / CopyL1ToL0B / CopyL0CToGm / CopyL0CToUb / CopyUbToGm / CopyGmToUb 及其 TLA 变体 (TileCopyTla / TileCopyTlaExt) 编写或补充 UT 测试用例时使用。
---

# Writing an unittest for CATLASS tile-level utilities

本技能覆盖 CATLASS Gemm/Epilogue Tile 层各条搬运通路的 UT 编写方法。各通路共性方法论写在本文，**大段的组件族谱、断言模式、测试模板、分支覆盖清单等参考材料按通路拆分到 `referrence/` 目录**：

| 通路 (src→dst) | 组件 | Arch | 参考文件 |
|----------------|------|------|---------|
| GM→L1 | CopyGmToL1 / GMMPTD / DynamicOptimized / IntervalDataCopy + TLA(TileCopyTla / TileCopyTlaExt / TileCopySparseTla / TileCopyFAQTla) | AtlasA2, Ascend950 | [referrence/copy-gm-to-l1.md](referrence/copy-gm-to-l1.md) |
| L1→L0A | CopyL1ToL0A + TLA(TileCopyTla / TileCopySparseTla) | AtlasA2 | [referrence/copy-l1-to-l0a.md](referrence/copy-l1-to-l0a.md) |
| L1→L0B | CopyL1ToL0B + TLA | AtlasA2 | [referrence/copy-l1-to-l0b.md](referrence/copy-l1-to-l0b.md) |
| L0C→GM | CopyL0CToGm / CopyL0CToGmTla | Ascend950 | [referrence/copy-l0c-to-gm.md](referrence/copy-l0c-to-gm.md) |
| GM→UB | CopyGm2Ub / CopyGm2UbAligned / CopyPerTokenScale2Ub | AtlasA2, Ascend950 | [referrence/copy-gm-to-ub.md](referrence/copy-gm-to-ub.md) |
| UB→GM | CopyUb2Gm / CopyUb2GmAligned | AtlasA2, Ascend950 | [referrence/copy-ub-to-gm.md](referrence/copy-ub-to-gm.md) |

## About detailed referrence

各Referrence文件的可参考内容如下：
 1. 组件族谱
   - TLA/非TLA组件
   - 断言必验字段
 2. 测试基础设施
   - 关联Stub文件
   - 测试Fixture成员
   - 日志索引约定
 3. 断言模式参考
   - 依照不同通路组件特化

## Workflow

```plain
1. 识别待测组件 → 确定测试组件特征（Arch代际 + TLA/非TLA实现 + 通路 + Layout排布 + Element特化 + 分支路径）
2. 创建测试文件 → 遵循标准模板结构
3. 编写测试用例 → 按分支路径逐一覆盖，每用例验证一组下的 API 调用正确性
4. 运行验证     → cmake 构建 + gtest 测试
```

## About Tile-level utilties

### Top to down

Tile 层搬运组件的公共入口头文件按 `CATLASS_ARCH` 宏分发到架构实现，测试时只需 include 公共头即可（如 `catlass/gemm/tile/copy_gm_to_l1.hpp`）。以 GM→L1 为例：

- `CATLASS_ARCH == 2201` (AtlasA2) → `atlasa2/copy_*.hpp`
- `CATLASS_ARCH == 3510` (Ascend950) → `ascend950/copy_*.hpp`

每条通路的 Struct 族谱（各特化 + 关键分支逻辑）见对应 `referrence/` 文件的「组件族谱」章节。

### What is the kind of this utilty?

编写 UT 前先定位组件的三个特征：

1. **Arch 代际**：`CATLASS_ARCH == 2201` → AtlasA2；`== 3510` → Ascend950。测试文件必须用架构守卫宏包裹（见 4.2）。部分通路仅存在于某一代（如 L0C→GM 仅 Ascend950，L1→L0A/L0B 主要在 AtlasA2）。
2. **TLA / 非TLA**：非 TLA 组件以 `GemmType<Element, Layout>` / `L1Type + L0Type` 为模板参数并接收 `layout` 对象；TLA 变体（`TileCopyTla` / `TileCopyTlaExt` / `TileCopySparseTla` / `TileCopyFAQTla`）使用 TLA tensor 接口。
3. **所在通路 (src→dst)**：即上表的 6 条通路，决定被测头文件、搬运 API（`DataCopy` / `Nd2Nz` / `LoadData` / `LoadDataWithTranspose` / `DataCopyPad` / `Fixpipe`）以及 stub 集合。

确定后可直接查阅对应 `referrence/` 文件里的族谱表以选定待测特化，已有的测试组件位于 `tests/unittest/catlass/gemm/tile/` 目录下。

### Evaluation metrics
<!-- 从Tile组件到单元测试件需包含的测试维度 -->

对每个待测特化，需覆盖以下测试维度的组合：

| 维度 | 说明 |
|------|------|
| **Struct** | 通路下的具体 Struct（基础 / GMMPTD / DynamicOptimized / Aligned / TLA 变体 …） |
| **Arch** | `Arch::AtlasA2` / `Arch::Ascend950`（仅测组件实际支持的代际） |
| **Element** | `float` / `half` / `int8_t` / `int4b_t`(AtlasA2) / `bfloat16_t` / `fp8_e8m0_t`(Ascend950 MX) 等 |
| **Layout (Src→Dst)** | RowMajor↔zN、ColumnMajor↔nZ、zN↔zZ、nZ↔zN、Padding*、VectorLayout、NDC1HWC0 … |
| **分支路径** | 正常 stride / 长 stride(逐行逐列) / 1-row 或少行 / C0 精确对齐 / 转置 / 量化(NO_QUANT/PER_TENSOR/PER_CHANNEL) |

## Begin writing the unittest case

### Key notes for the stub

测试通过 **stub + logger** 捕获底层 AscendC API 调用并对参数做断言，不做真实数据搬运。

公共 stub / 辅助文件：

| 文件 | 作用 |
|------|------|
| `stub/ascendc_test_fixture.h` | `AscendCTest` / `TileCopyTest` / `UBTileCopyTest` fixture，自动 Before/After 每个用例清空日志 |
| `stub/ascendc_logger.h` | `AscendCCallLogger` 单例，捕获 AscendC API 调用（DataCopy / LoadData / Fixpipe / …） |
| `stub/arg.h` | `Arg` 类型擦除参数容器：`.Value<T>()` / `.RawValue()` / `.GetInstAddr()` / `.Type()` |
| `stub/kernel_operator.h` | AscendC API 的 stub，调用时自动记录到 Logger |
| `common/helper.hpp` | `GetEleNumPerC0()` / `setLayout()` / `isContiguous()` / `setShapeImpl()` |
| `common/shape.hpp` | `TestVectorShape` / `TestMatrixShapeWithUnitflag` 等参数化 shape |

通路专用 stub（如 `kernel_struct_mm.h` / `kernel_operator_mm_intf.h` / `kernel_operator_fixpipe_intf.h` / `kernel_struct_fixpipe.h`）见对应 `referrence/` 文件。

**日志参数索引约定**（通用）：

```
argsT[0] = MakeArg<Element>()   →  GetArgsTAt(0).Type() = typeid(Element)
args[0]  = dst tensor           →  目标 Tensor
args[1]  = src tensor           →  源 Tensor
args[2]  = params               →  Nd2NzParams / DataCopyParams / LoadData2DParams / FixpipeParams* / ...
args[3]  = (可选) padParams / scale / 附加参数
```

> **注意 `Value<T>()` 的类型必须与实际 stub 捕获的结构体类型一致**，否则字段错位。`LoadData2DParams` 与 `LoadData2dTransposeParams` 字段偏移不同（见 L1→L0A/L0B 参考文件的字段布局对照表）。

### The test fixture

Fixture 继承基类 `AscendCTest`，在 `setShape<Element[, isTrans]>()` 中按布局是否转置预算好 round / fractal 成员，供各用例复用（矩阵类关键常量如 `BYTE_PER_C0=32` 等定义在 `catlass/catlass.hpp`）。

示例如`/template/fixture_example.cpp`所示，使用`CATLASS_ARCH`宏包裹测试。

用例主体的通用步骤：定义类型 → 实例化被测组件 → 创建 dummy tensor → `setShape` + `setLayout` → （可选）校验 layout 属性以确认进入目标分支 → 执行被测算子 → 从 logger 取日志做断言。各通路的完整测试文件模板可见对应的 `referrence/` 文件。

### Make assertions

断言的核心：验证内容： 
 - API 名称
 - 参数个数
 - 基础API的参数各字段
 - tensor 地址/偏移
 - Element 类型
 
验证过程的基本内容如`template/testsuite_example.cpp`所示（其中`${UnittestName}`和`${TestSuiteName}`为占位符，需根据实际情况替换）。

每条通路/分支的具体字段断言（Nd2Nz、逐行 DataCopyParams、长 stride、Contiguous、LoadData2D(转置/非转置)、Fixpipe 量化等）见对应 `referrence/` 文件的「断言模式」章节。

|API|Params 类型|必验字段|
|---|---|---|
|`DataCopy`|`Nd2NzParams`|`ndNum, nValue, dValue, srcDValue, dstNzC0Stride, dstNzNStride`|
|`DataCopy`|`DataCopyParams`|`blockCount, blockLen, srcStride, dstStride`|
|`Fixpipe`|`FixpipeParamsV220`|`nSize, mSize, srcStride, dstStride, quantPre, reluEn, unitFlag`|
|`LoadData`|`LoadData2DParams`|`startIndex, repeatTimes, srcStride, dstGap, ifTranspose`|
|`LoadDataWithTranspose`|`LoadData2dTransposeParams`|`startIndex, repeatTimes, srcStride, dstGap, dstFracGap`|
|`Mmad`|`MmadParams`|`m, n, k`|

## Build and run

切换至CATLASS根目录下，执行编译和测试动作：

```bash
cmake --build . --target catlass_unittest
./tests/unittest/catlass_unittest_"$CATLASS_ARCH"  --gtest_filter="CopyL0CToGm*"
```

各通路精确的 target 名与可执行名见对应 `referrence/` 文件的「编译与运行」章节。

## Styles, questions and common problems

### Style
<!-- 关于注释、变量，尽可能使用Fixture内的规则 -->

**变量命名规范**：从日志提取结构体指针时变量名必须与结构体类型对应，一律用 `const auto*` 或 `const AscendC::XxxParams*`，禁止无 cv 限定的裸指针，禁止 `p` / `params` 等不规范简称。

| 结构体类型 | 变量名 |
|-----------|--------|
| `AscendC::Nd2NzParams*` | `nd2nzArg` |
| `AscendC::DataCopyParams*` / `DataCopyExtParams*` | `dataCopyArg` / `dataCopyParams` |
| `AscendC::LoadData2DParams*` | `loadDataArg` / `load2DArg` |
| `AscendC::LoadData2dTransposeParams*` | `loadDataArg` |

**优先复用 Fixture 已有变量**：`setShape<Element[, isTrans]>()` 已算好 `_row_round` / `_col_round`（及 `_row_per_fractal` 等），用例内**禁止重复计算同一值**。

```cpp
// isTrans=false (zN 类): _row_round=RoundUp<C0_NUM_PER_FRACTAL>(_row), _col_round=RoundUp<ELE_NUM_PER_C0>(_col)
// isTrans=true  (nZ 类): _row_round=RoundUp<ELE_NUM_PER_C0>(_row),     _col_round=RoundUp<C0_NUM_PER_FRACTAL>(_col)

// ✅ 推荐：复用 setShape 算好的成员
setShape<Element>();          // 或 setShape<Element, true>() 用于 nZ 类
LayoutSrc layoutSrc(_row, _col, C0_NUM_PER_FRACTAL, _row_round / C0_NUM_PER_FRACTAL, ...);
```

> `isTrans` 必须与待测布局匹配：zN 用默认（`false`），nZ 用 `setShape<Element, true>()`，否则 round 基数与布局不符。

### Questions commonly asked

- **架构守卫宏**：每个测试文件必须包裹 `#if !defined(CATLASS_ARCH) || CATLASS_ARCH == 2201`（或 `3510`）… `#endif`，避免在错误 ARCH 配置下被编译。
- **ELE_NUM_PER_C0 计算**：推荐 bit-level 以支持子字节类型（`BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value`）；byte-level 类型可用 `BYTE_PER_C0 / sizeof(Element)`。辅助函数 `GetEleNumPerC0<Element>()` 已封装。
- **setShape 时机**：`setShape<Element>()` 依赖 Element 类型计算 round 值，必须在每个 `TEST_F` 中先调用，不能放进 `SetUp()`（Element 是 per-test 变化的模板参数）。
- **布局连续性验证**：`isContiguous()` 需 include `common/helper.hpp`，重载支持 RowMajor / ColumnMajor / zN / nZ / zZ / nN / VectorLayout。
- **zN 布局创建**：zN 是 Element-dependent 的，须用目标类型 `LayoutDst::template MakeLayout<ElementDst>(m, n)` 创建。

### Problems commonly met

- **Nd2NzParams 单位**：`nValue` / `dValue` / `srcDValue` 为元素个数；`dstNzC0Stride` / `dstNzNStride` / `dstNzMatrixStride` 为 **C0 个数**（需除 `ELE_NUM_PER_C0`，勿填元素数）。
- **ColumnMajor 语义反转**：ColumnMajor→nZ 时 `nValue`=cols、`dValue`=rows，是 RowMajor→zN 的镜像。
- **Params 结构体字段错位**：`LoadData2DParams` 与 `LoadData2dTransposeParams` 偏移不同（`repeatTimes` 类型/位置不同），`Value<T>()` 的 T 必须与 stub 实际捕获类型一致。
- **DataCopyPad 参数个数差异**：CopyGm2Ub 是 4-param（含 `padParams`），CopyUb2Gm 是 3-param（无 `padParams`），断言 `args.size()` 时勿混。
- **量化路径日志序列**（L0C→GM）：NO_QUANT RowMajor 出口是 `SetFixpipeNz2ndFlag` + `DataCopy` 两条日志；PER_TENSOR/PER_CHANNEL 走单条 `Fixpipe`（PER_CHANNEL 为 4-param）。

## Checklists and output

**输出物**：每条通路一个测试文件（`tests/unittest/catlass/.../test_tile_copy_<src>_to_<dst>.cpp`），按对应 `referrence/` 文件的分支覆盖率清单逐项建 `TEST_F`（或`TEST_P`）。

**收尾清单**：

- [ ] 架构守卫宏已包裹整个文件
- [ ] 待测特化在对应 `referrence/` 族谱表中均有覆盖（Struct × Arch × Layout × Element × 分支）
- [ ] 每个用例断言了 API 名称、`args.size()`、params 各字段、tensor 地址/偏移、`Element` 类型
- [ ] 变量命名符合 4.1，复用 fixture 成员，无重复计算
- [ ] `cmake --build` 通过，`gtest` 全部通过
