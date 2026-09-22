# CopyL0CToGm (L0C→GM, Ascend950) UT 参考

被测源码：`include/catlass/gemm/tile/ascend950/copy_l0c_to_gm.hpp` (549行)

---

## 组件族谱

### Non-TLA 特化

| # | LayoutDst | ScaleGranularity | ElementSrc→Dst | 搬运API | Params类型 | Log序列 |
|---|-----------|-----------------|-----------------|---------|-----------|---------|
| 1 | `RowMajor` | `NO_QUANT` | float→float | `DataCopy` | `DataCopyCO12DstParams` | `SetFixpipeNz2ndFlag` → `DataCopy` |
| 2 | `RowMajor` | `PER_TENSOR` | float→half | `Fixpipe`(3-param) | `FixpipeParamsC310<ROW_MAJOR>` | `Fixpipe` |
| 3 | `RowMajor` | `PER_CHANNEL` | float→half | `Fixpipe`(4-param) | `FixpipeParamsC310<ROW_MAJOR>` | `Fixpipe`(dst,src,scale,params) |
| 4 | `zN` | `NO_QUANT` | float→float | `DataCopy` | `DataCopyCO12DstParams` | `DataCopy` (nz2ndEn=false, channelSplit=true) |
| 5 | `zN` | `NO_QUANT` | float→half | `DataCopy` | `DataCopyCO12DstParams` | `DataCopy` (nz2ndEn=false) |

### TLA 变体（CopyL0CToGmTla）

| # | LayoutDst | ScaleGranularity | 搬运API | 特点 |
|---|-----------|-----------------|---------|------|
| 7 | RowMajor | NO_QUANT | `DataCopy` + `SetFixpipeNz2ndFlag` | 支持 `l0Batch`/`dstNdStride` |
| 8 | zN | NO_QUANT | `DataCopy` | channelSplit for float→float |
| 9 | RowMajor | PER_TENSOR | `Fixpipe<CFG_ROW_MAJOR>` | 有 `Params{float scale}` 成员 |
| 10 | RowMajor | PER_CHANNEL | `Fixpipe<CFG_ROW_MAJOR>` + quant tensor | 4-param |

### QuantMode 映射（CopyL0CToDstQuantMode）

| ElementSrc | ElementDst | NO_QUANT | PER_TENSOR | PER_CHANNEL |
|-----------|-----------|----------|------------|-------------|
| float | float | NoQuant | QF322F32_PRE | VQF322F32_PRE |
| float | half | F322F16 | QF322F16_PRE | VQF322F16_PRE |
| float | bfloat16_t | F322BF16 | QF322BF16_PRE | VQF322BF16_PRE |

---

### 断言必验字段

|API|Params 类型|必验字段|
|---|---|---|
|`AscendC::DataCopy`|`DataCopyCO12DstParams`|`nSize, mSize, srcStride, dstStride, quantPre, channelSplit, nz2ndEn, unitFlag`|
|`AscendC::Fixpipe`|`FixpipeParamsC310<ROW_MAJOR>`|`mSize, nSize, srcStride, dstStride, quantPre, deqScalar, unitFlag`|
|`AscendC::Fixpipe<CFG_NZ>`|`FixpipeParamsV220`|`blockCount, blockLen, srcStride, dstStride`|

---

## 测试基础设施

### 关联Stub文件

| 文件 | 作用 |
|------|------|
| `stub/ascendc_test_fixture.h` | `TileCopyTest` fixture |
| `stub/kernel_operator_fixpipe_intf.h` | `Fixpipe` / `SetFixpipeNz2ndFlag` / `DataCopy` stub |
| `stub/kernel_struct_fixpipe.h` | `FixpipeParamsV220` / `FixpipeParamsC310` / `FixpipeParamsArch3510` |
| `stub/kernel_struct_mm.h` | `DataCopyCO12DstParams` |
| `common/helper.hpp` | `setLayout()` / `isContiguous()` |
| `common/shape.hpp` | `TestMatrixShapeWithUnitflag` |

### 测试Fixture成员

`TileCopyTest` 提供矩阵 shape 与量化相关成员，`TestMatrixShapeWithUnitflag` 用于参数化 shape：

```cpp
// 常用成员：
uint32_t _m, _n;          // M / N 方向大小
uint32_t _m_round;        // M 向上取整（srcStride 参照）
uint8_t  _unitFlag;       // 0 或 0x10
uint64_t _scale_uint64;   // PER_TENSOR 反量化标量

// 参数化 shape：
struct TestMatrixShapeWithUnitflag : public TestMatrixShape {
    uint8_t unitFlag;         // 0 或 0x10 (unitFlag enable)
    bool channelSplit;        // 控制 L0C split 输出
};
```

### 日志索引约定

```cpp
// DataCopy 路径 (NO_QUANT):
logs[0] = SetFixpipeNz2ndFlag(ndNum, srcNdStride, dstNdStride)  // RowMajor 出口
logs[1] = DataCopy(dst, src, DataCopyCO12DstParams)              // 实际搬运

// Fixpipe 路径 (PER_TENSOR / PER_CHANNEL):
logs[0] = Fixpipe(dst, src, [scale,] FixpipeParamsC310)          // 单步完成
```

---

## 断言模式

### RowMajor NO_QUANT (#1)

RowMajor 布局下无量化出口，日志包含两步：先 `SetFixpipeNz2ndFlag` 设置 ND→NZ 标志，再 `DataCopy` 执行实际搬运。`nz2ndEn=true` 表示启用 NZ2ND 转换。

```cpp
ASSERT_EQ(logs.size(), 2);
// logs[0]: SetFixpipeNz2ndFlag(1, 1, 1)
ASSERT_EQ(logs[0].name, "SetFixpipeNz2ndFlag");
// logs[1]: DataCopy
const auto* p = logs[1].GetArgsAt(2).Value<AscendC::DataCopyCO12DstParams>();
ASSERT_EQ(p->mSize, _m);
ASSERT_EQ(p->nSize, _n);
ASSERT_EQ(p->srcStride, _m_round);
ASSERT_EQ(p->dstStride, _n);
ASSERT_EQ(p->nz2ndEn, true);
ASSERT_EQ(p->unitFlag, _unitFlag);
ASSERT_EQ(p->quantPre, QuantMode_t::NoQuant);
```

### RowMajor PER_TENSOR

PER_TENSOR 量化出口，使用 `Fixpipe<CFG_ROW_MAJOR>` API 一次性完成反量化+搬运。需额外验证 `deqScalar` 和 `FixpipeConfig` 的 `format` 字段。

```cpp
ASSERT_EQ(logs.size(), 1);
// logs[0]: Fixpipe<CFG_ROW_MAJOR>
const auto* p = logs[0].GetArgsAt(2).Value<AscendC::FixpipeParamsC310<CO2Layout::ROW_MAJOR>>();
ASSERT_EQ(p->mSize, _m);
ASSERT_EQ(p->nSize, _n);
ASSERT_EQ(p->srcStride, _m_round);
ASSERT_EQ(p->dstStride, _n);
ASSERT_EQ(p->quantPre, QuantMode_t::QF322F16_PRE);
ASSERT_EQ(p->deqScalar, _scale_uint64);
ASSERT_EQ(p->unitFlag, _unitFlag);
// FixpipeConfig
auto* cfg = logs[0].GetArgsTAt(2).Value<AscendC::FixpipeConfig>();
ASSERT_EQ(cfg->format, CO2Layout::ROW_MAJOR);
```

### RowMajor PER_CHANNEL (#3)

PER_CHANNEL 量化出口，与 PER_TENSOR 类似但多一个 scale tensor 参数（args.size()==4）。需验证量化为 `VQ*_PRE` 模式。

```cpp
ASSERT_EQ(logs.size(), 1);
// logs[0]: Fixpipe<CFG_ROW_MAJOR>(dst, src, scale, params) — 4 args
ASSERT_EQ(logs[0].args.size(), 4);
const auto* p = logs[0].GetArgsAt(3).Value<AscendC::FixpipeParamsC310<CO2Layout::ROW_MAJOR>>();
ASSERT_EQ(p->mSize, _m);
ASSERT_EQ(p->nSize, _n);
ASSERT_EQ(p->quantPre, QuantMode_t::VQF322F16_PRE);
```

### zN NO_QUANT (#4, #5)

zN 布局下无量化出口，与 RowMajor 不同——没有 `SetFixpipeNz2ndFlag` 前置调用。`nz2ndEn=false`，`float→float` 时 `channelSplit=true`。

```cpp
ASSERT_EQ(logs.size(), 1);
// 无 SetFixpipeNz2ndFlag（与 RowMajor 不同）
// logs[0]: DataCopy(dst, src, DataCopyCO12DstParams)
const auto* p = logs[0].GetArgsAt(2).Value<AscendC::DataCopyCO12DstParams>();
ASSERT_EQ(p->mSize, _m);
ASSERT_EQ(p->nSize, _n);
ASSERT_EQ(p->nz2ndEn, false);
ASSERT_EQ(p->channelSplit, /* true for float→float, false otherwise */);
```

---