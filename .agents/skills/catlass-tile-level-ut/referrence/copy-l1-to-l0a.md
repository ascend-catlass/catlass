# CopyL1ToL0A (L1→L0A) UT 参考

被测源码：`include/catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp` (726行)

---

## 组件族谱

### Non-TLA 特化（3-param: L1Type + L0Type）

| # | L1Type | L0Type | Element | 搬运API | Params类型 | Trans |
|---|--------|--------|---------|---------|-----------|------|
| 1 | `zN(A1)` | `zZ(A2)` | generic | `LoadData` | `LoadData2DParams` | 否 |
| 2 | `nN(A1)` | `zZ(A2)` | generic | `LoadData` | `LoadData2DParams` | 是 |
| 3 | `nN(A1)` | `zZ(A2)` | `float` | `LoadDataWithTranspose` | `LoadData2dTransposeParams` | 是 |
| 4 | `nZ(A1)` | `zZ(A2)` | `int8_t` | `LoadDataWithTranspose` | `LoadData2dTransposeParams` | 是 |
| 5 | `NDC1HWC0(A1)` | `zZ` | generic | `LoadData<config>` | `LoadData3DParamsV2` | — |

### Non-TLA 特化（2-param: L1Type only）

| # | L1Type | Element | 搬运API | Params类型 | Trans |
|---|--------|---------|---------|-----------|------|
| 6 | `zN(A1)` | generic | `LoadData` | `LoadData2DParams` | 否 |
| 7 | `zN(A1)` | `float` | `LoadData<config>` | `LoadData3DParamsV2` | 否 |
| 8 | `nZ(A1)` | generic | `LoadData` | `LoadData2DParams` | 是 |
| 9 | `nZ(A1)` | `int8_t` | `LoadDataWithTranspose` | `LoadData2dTransposeParams` | 是 |
| 10 | `nZ(A1)` | `float` | `LoadData<config>` | `LoadData3DParamsV2` | 是 |

### TLA 变体（TileCopyTla）

| # | Src→Dst | Element | 搬运API | Params类型 |
|---|---------|---------|---------|-----------|
| 11 | `zN→zZ` | generic | `LoadData` | `LoadData2DParams` |
| 12 | `zN→zZ` | `float` | `LoadData<config>` | `LoadData3DParamsV2` |
| 13 | `nZ→zZ` | generic | `LoadData` (trans) | `LoadData2DParams` |
| 14 | `nZ→zZ` | `int8_t` | `LoadDataWithTranspose` | `LoadData2dTransposeParams` |
| 15 | `nZ→zZ` | `float` | `LoadData<config>` | `LoadData3DParamsV2` (trans) |

### Sparse TLA 变体（TileCopySparseTla）

| # | Src→Dst | 搬运API | Params类型 |
|---|---------|---------|-----------|
| 16 | `zN→zZ` | `LoadData` | `LoadData3DParamsV2Pro` |

### 断言必验字段

|API|Params 类型|必验字段|
|---|---|---|
|`AscendC::LoadData`|`LoadData2DParams`|`startIndex, repeatTimes, srcStride, dstGap, ifTranspose, addrMode`|
|`AscendC::LoadDataWithTranspose`|`LoadData2dTransposeParams`|`startIndex, repeatTimes, srcStride, dstGap, dstFracGap, addrMode`|
|`AscendC::LoadData<config>`|`LoadData3DParamsV2`|`l1H, l1W, channelSize, kExtension, mExtension, enTranspose, fMatrixCtrl, padList`|
|`AscendC::LoadData`|`LoadData3DParamsV2Pro`|`channelSize, enTranspose, enSmallK, filterSizeW, filterSizeH, fMatrixCtrl`|

---

## 测试基础设施

### 关联Stub文件

| 文件 | 作用 |
|------|------|
| `stub/ascendc_test_fixture.h` | `AscendCTest` fixture，自动 Before/After 每个用例清空日志 |
| `stub/ascendc_logger.h` | `AscendCCallLogger` 单例，捕获 AscendC API 调用 |
| `stub/kernel_struct_mm.h` | `LoadData2DParams` / `LoadData2dTransposeParams` stub 定义 |
| `stub/kernel_operator_mm_intf.h` | `LoadData` / `LoadDataWithTranspose` stub 实现 |
| `common/helper.hpp` | `GetEleNumPerC0()` / `setLayout()` / `isContiguous()` / `setShapeImpl()`

### 测试Fixture成员

`TileCopyL1ToL0ATest` 继承 `AscendCTest`，`setShape<Element, isTrans>(row, col)` 依据是否转置预算好 round 与 fractal 值供各用例复用：

```cpp
// setShape<Element, isTrans=false>(row, col):
//   _row_round = RoundUp<C0_NUM_PER_FRACTAL>(_row);
//   _col_round = RoundUp<ELE_NUM_PER_C0>(_col);
//   _row_per_fractal = _row_round / C0_NUM_PER_FRACTAL;
//   _col_per_fractal = _col_round / ELE_NUM_PER_C0;
//
// setShape<Element, isTrans=true>(row, col):  _row_round/_col_round 的 round 基数互换
uint32_t _row = 128, _col = 64;
uint32_t _row_round = _0, _col_round = _0;
uint32_t _row_per_fractal = _0, _col_per_fractal = _0;
```

### 日志索引约定

```cpp
argsT[0] = MakeArg<Element>()   →  GetArgsTAt(0).Type() = typeid(Element)
args[0]  = dstTensor (LocalTensor/L0A)
args[1]  = srcTensor (LocalTensor/L1)
args[2]  = params               →  LoadData2DParams / LoadData2dTransposeParams / LoadData3DParamsV2
```

> `LoadData2DParams` 与 `LoadData2dTransposeParams` 字段偏移不同——`Value<T>()` 的 T 必须与实际 stub 捕获的类型一致（见 SKILL.md 4.3）。

---

## 断言模式参考

### LoadData2DParams 无转置

zN→zZ 标准搬运场景，使用 `LoadData` API 通过 `LoadData2DParams` 参数搬运，无转置。每行搬运一次，repeatTimes 等于分形数。

```cpp
const auto* p = log.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
ASSERT_EQ(p->startIndex, _0);
ASSERT_EQ(p->repeatTimes, _col_per_fractal);         // = layoutDst.shape(3)
ASSERT_EQ(p->srcStride,  _row_per_fractal);          // = stride(3) / ELE_NUM_PER_FRACTAL
ASSERT_EQ(p->dstGap,     _0);                        // zN→zZ 分形间无 gap
ASSERT_EQ(p->ifTranspose, _0);
ASSERT_EQ(p->addrMode,   _0);
```

### LoadData2DParams 转置

nN/nZ→zZ 需要转置的搬运场景，仍通过 `LoadData` API 配合 `LoadData2DParams` 参数，但 `ifTranspose` 设为 true。

```cpp
const auto* p = log.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
ASSERT_EQ(p->startIndex, _0);
ASSERT_EQ(p->repeatTimes, _col_per_fractal);
ASSERT_EQ(p->srcStride,  _row_per_fractal);
ASSERT_EQ(p->dstGap,     _0);
ASSERT_EQ(p->ifTranspose, _1);
ASSERT_EQ(p->addrMode,   _0);
```

### LoadData2dTransposeParams

float/int8_t 类型的转置搬运，使用专用的 `LoadDataWithTranspose` API 和 `LoadData2dTransposeParams` 参数结构。`srcStride` 固定为 1。

```cpp
const auto* p = log.GetArgsAt(2).Value<AscendC::LoadData2dTransposeParams>();
ASSERT_EQ(p->startIndex, _0);
ASSERT_EQ(p->repeatTimes, _col_per_fractal);
ASSERT_EQ(p->srcStride,  1);                           // transpose 场景下固定为 1
ASSERT_EQ(p->dstGap,     0);
ASSERT_EQ(p->dstFracGap, 0);
```

### 循环地址验证

搬运循环次数等于目标行数（`_row_per_fractal`）。每次迭代验证 src/dst 的 offset 递增是否正确，以及 params 字段不变。

```cpp
ASSERT_EQ(logs.size(), _row_per_fractal);  // 搬运循环次数
for (int i = 0; i < _row_per_fractal; i++) {
    ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), i * BYTE_PER_FRACTAL * _col_per_fractal);
    ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), i * BYTE_PER_FRACTAL);
}
```
