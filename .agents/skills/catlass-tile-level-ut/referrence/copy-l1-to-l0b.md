# CopyL1ToL0B (L1→L0B) UT 参考

被测源码：`include/catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp` (lines 1-511)

---

## 组件族谱（非TLA AtlasA2）

### 3-param 特化（L1Type + L0Type）

| # | L1Type | L0Type | Element | Params | API | Trans |
|---|--------|--------|---------|--------|-----|------|
| 1 | `zZ(B1)` | `nZ(B2)` | generic | `LoadData2DParams` | `LoadData` | 是 |
| 2 | `zZ(B1)` | `nZ(B2)` | `float` | `LoadData2dTransposeParams` | `LoadDataWithTranspose` | 是 |
| 3 | `zN(B1)` | `nZ(B2)` | `int8_t` | `LoadData2dTransposeParams` | `LoadDataWithTranspose` | 是 |
| 4 | `nZ(B1)` | `nZ(B2)` | generic | `LoadData2DParams` | `LoadData` | 否 |
| 5 | `zN(B1)` | `zN(B2)` | generic | `LoadData2DParams` | `LoadData` | 否 |
| 6 | `nN(B1)` | `zN(B2)` | generic | `LoadData2DParams` | `LoadData` (单次) | 是 |
| 7 | `nN(B1)` | `zN(B2)` | `float` | `LoadData2dTransposeParams` | `LoadDataWithTranspose` | 是 |
| 8 | `nZ(B1)` | `zN(B2)` | `int8_t` | `LoadData2dTransposeParams` | `LoadDataWithTranspose` | 是 |

### 2-param 特化（L1Type only）

| # | L1Type | Element | Params | API | Trans |
|---|--------|---------|--------|-----|------|
| 9 | `zN(A1)` | `int8_t` | `LoadData2dTransposeParams` | `LoadDataWithTranspose` | 是 |
| 10 | `zN(A1)` | `float` | `LoadData3DParamsV2` | `LoadData<config>` | — |
| 11 | `zN(A1)` | generic | `LoadData2DParams` | `LoadData` | 是 |
| 12 | `zN(A1)` | `int4b_t` | `LoadData2dTransposeParams` | `LoadDataWithTranspose` | 是 |
| 13 | `nZ(A1)` | generic | `LoadData2DParams` | `LoadData` | 否 (分支) |

---

### 断言必验字段

|API|Params 类型|必验字段|
|---|---|---|
|`AscendC::LoadData`|`LoadData2DParams`|`startIndex, repeatTimes, srcStride, dstGap, ifTranspose, addrMode`|
|`AscendC::LoadDataWithTranspose`|`LoadData2dTransposeParams`|`startIndex, repeatTimes, srcStride, dstGap, dstFracGap, addrMode`|
|`AscendC::LoadData<config>`|`LoadData3DParamsV2`|`l1H, l1W, channelSize, kExtension, mExtension, enTranspose, fMatrixCtrl`|

---

## 测试基础设施

### 关联Stub文件

| 文件 | 作用 |
|------|------|
| `stub/ascendc_test_fixture.h` | `AscendCTest` fixture，自动 Before/After 每个用例清空日志 |
| `stub/ascendc_logger.h` | `AscendCCallLogger` 单例，捕获 AscendC API 调用 |
| `stub/kernel_struct_mm.h` | `LoadData2DParams` / `LoadData2dTransposeParams` / `LoadData3DParamsV2` stub 定义 |
| `stub/kernel_operator_mm_intf.h` | `LoadData` / `LoadDataWithTranspose` stub 实现 |
| `common/helper.hpp` | `GetEleNumPerC0()` / `setLayout()` / `isContiguous()` / `setShapeImpl()` |

### 测试Fixture成员

`TileCopyL1ToL0BTest` 继承 `AscendCTest`，`setShape<Element, isTrans>(row, col)` 依据是否转置预算好 round 与 fractal 值供各用例复用：

```cpp
// setShape<Element, isTrans=false>(row, col):
//   _row_round = RoundUp<C0_NUM_PER_FRACTAL>(_row);
//   _col_round = RoundUp<ELE_NUM_PER_C0>(_col);
//   _cols_by_fractal = _col_round / ELE_NUM_PER_C0;   // 分形列数
// setShape<Element, isTrans=true>(row, col):  round 基数互换
uint32_t _row = 128, _col = 64;
uint32_t _row_round = _0, _col_round = _0;
uint32_t _rows_by_fractal = _0, _cols_by_fractal = _0;
```

### 日志索引约定

```cpp
argsT[0] = MakeArg<Element>()    →  GetArgsTAt(0).Type() = typeid(Element)
args[0]  = dstTensor (LocalTensor/L0B)
args[1]  = srcTensor (LocalTensor/L1)
args[2]  = params (LoadData2DParams / LoadData2dTransposeParams / LoadData3DParamsV2)
```

**Params 字段布局对照**（避免字段错位）：

```
LoadData2DParams:               LoadData2dTransposeParams:
 offset 0: startIndex (u16)      offset 0: startIndex (u16)
 offset 2: dstGap (u16)          offset 2: repeatTimes (u8)
 offset 4: srcStride (u16)       offset 4: srcStride (u16)
 offset 6: ifTranspose (bool)    offset 6: dstGap (u16)
 offset 7: repeatTimes (u8)      offset 8: dstFracGap (u16)
 offset 8: sid (u8)              offset 10: addrMode (u8)
 offset 9: addrMode (u8)
```

**规则**：测试端 `Value<T>()` 的 T 必须与实际 stub 捕获的类型一致。

---

## 断言模式

### LoadData2DParams 非转置

zN→zN 或 nZ→nZ 同布局搬运，无转置，使用 `LoadData` API 配合 `LoadData2DParams`。L0B 的 `repeatTimes` 取 `shape(3)`。

```cpp
const auto* p = log.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
ASSERT_EQ(p->startIndex, _0);
ASSERT_EQ(p->repeatTimes, shape(3));
ASSERT_EQ(p->srcStride,  stride(3) / ELE_NUM_PER_FRACTAL);
ASSERT_EQ(p->dstGap,     stride(3) / ELE_NUM_PER_FRACTAL - 1);
ASSERT_EQ(p->ifTranspose, false);
ASSERT_EQ(logs.size(), 1);  // 单次调用
```

### LoadData2DParams 转置

zZ/zN→zN 需要转置，使用 `LoadData` API 但 `ifTranspose=true`。`repeatTimes` 基于目标形状的 `CeilDiv` 计算。

```cpp
const auto* p = log.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
ASSERT_EQ(p->startIndex, _0);
ASSERT_EQ(p->repeatTimes, CeilDiv<ELE_NUM_PER_C0>(orgShape(1)));
ASSERT_EQ(p->srcStride,  1);
ASSERT_EQ(p->dstGap,     0);
ASSERT_EQ(p->ifTranspose, true);
ASSERT_EQ(logs.size(), CeilDiv<C0_NUM_PER_FRACTAL>(dst.orgShape(0)));
```

### LoadData2dTransposeParams

float/int8_t/int4b_t 的转置搬运，使用专有 `AscendC::LoadDataWithTranspose` API。

```cpp
const auto* p = log.GetArgsAt(2).Value<AscendC::LoadData2dTransposeParams>();
ASSERT_EQ(p->startIndex, _0);
ASSERT_EQ(p->repeatTimes, _cols_by_fractal);
ASSERT_EQ(p->srcStride,  _row_round * ELE_NUM_PER_C0 / ELE_NUM_PER_FRACTAL / 2);
ASSERT_EQ(p->dstGap,     _1);
ASSERT_EQ(p->dstFracGap, _0);
```

