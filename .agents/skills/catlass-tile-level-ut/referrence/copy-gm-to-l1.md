# CopyGmToL1 (GM→L1) UT 参考

被测源码：
- `include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp` (2227行)
- `include/catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp` (986行)

顶层调度：`include/catlass/gemm/tile/copy_gm_to_l1.hpp` 按 `CATLASS_ARCH` 宏分发：
- `CATLASS_ARCH == 2201` (AtlasA2) → `atlasa2/copy_gm_to_l1.hpp`
- `CATLASS_ARCH == 3510` (Ascend950) → `ascend950/copy_gm_to_l1.hpp`

---

## 组件族谱

### Struct 族系 (AtlasA2 / Ascend950 共性)

| Struct | 用途 | 关键分支逻辑 |
|--------|------|-------------|
| **CopyGmToL1** | GTensor→LTensor 基础搬运，GM→L1 | 短stride Nd2Nz / 长stride逐行 / int4b_t 修正 / 多nd批量 / zN→zN(nZ→nZ)直传 / Padding→分形 / Vector→分形 / NDC1HWC0 / GEMM多layout组合 |
| **CopyGmToL1GMMPTD** | GMMP TD 场景专用 (支持 1-row 优化) | shape(0)==1 → DataCopyParams / stride<LIMIT → Nd2Nz / stride>=LIMIT → 逐行 |
| **CopyGmToL1DynamicOptimized** | 动态选择最优搬运策略 | rows<=16 → 逐行DataCopyParams / stride<LIMIT + !exactMatch → Nd2Nz / stride<LIMIT + exactMatch → contiguous / stride>=LIMIT → 逐行Nd2Nz |
| **CopyGmToL1IntervalDataCopy** | half 短宽矩阵专用 (逐行 DataCopyParams) | 仅 half + RowMajor/PaddingRowMajor/ColumnMajor/PaddingColumnMajor |

### TLA 变体 (AtlasA2)

| Struct | 用途 |
|--------|------|
| **TileCopyTla** | TLA tensor 接口版本：GM RowMajor→L1 zN, GM ColumnMajor→L1 nZ, GM zN→L1 zN, GM nZ→L1 nZ |
| **TileCopyTlaExt** | 带 actualShape 的 TLA 扩展：PaddingRowMajor→zN, PaddingColumnMajor→nZ, ColumnMajor→nZ, zN→zN, nZ→nZ |
| **TileCopySparseTla** | 稀疏拷贝：RowMajor/ColumnMajor→zN, ColumnMajor→nZ, 稀疏zN→zN, 稀疏nZ→nZ |
| **TileCopyFAQTla** | FlashAttention LoadQ: 多矩阵 ND→NZ |

### TLA 变体 (Ascend950)

| Struct | 用途 |
|--------|------|
| **TileCopyTla** | RowMajor→zN, zN→zN, ColumnMajor→nZ, nZ→nZ, Vector→Vector, MX fp8_e8m0 Scale |

### 测试维度矩阵

| 维度 | 取值范围 |
|------|---------|
| **Arch** | `Arch::AtlasA2`, `Arch::Ascend950` |
| **Element** | `float`, `half`, `int4b_t`(仅A2), `fp8_e8m0_t`(A5 MX), `float4_e2m1x2_t`/`float4_e1m2x2_t`(A5) |
| **LayoutSrc→LayoutDst** | RowMajor→zN, ColumnMajor→nZ, zN→zN, nZ→nZ, PaddingRowMajor→zN, PaddingColumnMajor→nZ, VectorLayout→zN, RowMajor→RowMajor, RowMajor→zZ, ColumnMajor→nN, NDC1HWC0, KDC1KHKWN1N0C0 |
| **分支路径** | 正常stride(Nd2Nz), 长stride(逐行/逐col), 1-row/少rows(DataCopyParams), 精确C0对齐(contiguous), int4b_t修正 |

### 断言必验字段

|API|Params 类型|必验字段|
|---|---|---|
|`AscendC::DataCopy`|`Nd2NzParams`|`ndNum, nValue, dValue, srcDValue, dstNzC0Stride, dstNzNStride`|
|`AscendC::DataCopy`|`DataCopyParams`|`blockCount, blockLen, srcStride, dstStride`|

---

## 测试基础设施

### 关联Stub文件

| 文件 | 作用 |
|------|------|
| `stub/ascendc_test_fixture.h` | `AscendCTest` fixture，自动 Before/After 每个用例清空日志 |
| `stub/ascendc_logger.h` | `AscendCCallLogger` 单例，捕获 AscendC API 调用（DataCopy 等） |
| `stub/arg.h` | `Arg` 类型擦除参数容器：`.Value<T>()` / `.RawValue()` / `.GetInstAddr()` / `.Type()` |
| `stub/kernel_operator.h` | AscendC API 的 stub，调用时自动记录到 Logger |
| `stub/kernel_struct_mm.h` | `Nd2NzParams` / `DataCopyParams` stub 定义 |
| `common/helper.hpp` | `GetEleNumPerC0()` / `setLayout()` / `isContiguous()` |

### 测试Fixture成员

`TileCopyGmToL1Test` 继承 `AscendCTest`，`setShape<Element>(row, col)` 依据 Element 预算好 round 值供各用例复用：

```cpp
// setShape<Element>(row, col):
//   _row_round = RoundUp<C0_NUM_PER_FRACTAL>(_row);
//   _col_round = RoundUp<ELE_NUM_PER_C0>(_col);
uint32_t _row = 128, _col = 256;   // shape
uint32_t _row_round = _0;          // 行按 C0_NUM_PER_FRACTAL 向上取整
uint32_t _col_round = _0;          // 列按 ELE_NUM_PER_C0 向上取整
```

关键常量（`include/catlass/catlass.hpp`）：

```cpp
constexpr uint32_t BYTE_PER_C0        = 32;   // 32 bytes per C0
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;   // 16 C0 per fractal
constexpr uint32_t BYTE_PER_FRACTAL   = 512;  // 32*16 = 512 bytes
constexpr uint32_t STRIDE_LIMIT       = 65536;
```

### 日志索引约定

```cpp
argsT[0] = MakeArg<Element>()   →  GetArgsTAt(0).Type() = typeid(Element)
args[0]  = dstTensor (LocalTensor/L1)
args[1]  = srcTensor (GlobalTensor/GM)
args[2]  = params               →  Nd2NzParams / DataCopyParams
```

> Nd2NzParams 单位注意：`nValue`/`dValue`/`srcDValue` 为元素个数；`dstNzC0Stride`/`dstNzNStride`/`dstNzMatrixStride` 为 C0 个数（需除 `ELE_NUM_PER_C0`）。

---

## 断言模式参考

### Nd2Nz 分支 (DataCopy 调用)

当被测组件走 Nd2Nz 路径 (stride < STRIDE_LIMIT 且非精确C0对齐) 时，调用**随路转换ND2NZ搬运**。

```cpp
ASSERT_EQ(logs.size(), 1);   // 单次 DataCopy

AscendCCallLog logTileCopy = logs[0];
ASSERT_EQ(logTileCopy.name, "DataCopy");
ASSERT_EQ(logTileCopy.args.size(), 3);

// 验证 tensor 地址
ASSERT_EQ(logTileCopy.GetArgsAt(1).RawValue(), &gmTensor);  // src
ASSERT_EQ(logTileCopy.GetArgsAt(0).RawValue(), &l1Tensor);  // dst

// 验证 Nd2NzParams
const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
ASSERT_EQ(nd2nzArg->ndNum, 1);
ASSERT_EQ(nd2nzArg->nValue, _row);
ASSERT_EQ(nd2nzArg->dValue, _col);
ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
ASSERT_EQ(nd2nzArg->srcDValue, _col);                    // = layoutSrc.stride(0)
ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);          // = layoutDst.stride(3) / ELE_NUM_PER_C0
ASSERT_EQ(nd2nzArg->dstNzNStride, _1);                   // = layoutDst.stride(0) / ELE_NUM_PER_C0
ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
```

### DataCopyParams 分支 (逐行 interval-based)

当走逐行 DataCopyParams 路径 (1-row / rows<=16 等) 时，搬运逻辑是**多轮循环的基础数据搬运**：

```cpp
ASSERT_EQ(logs.size(), _row);   // 每行一次 DataCopy

for (int i = 0; i < _row; i++) {
    AscendCCallLog logTileCopy = logs[i];
    ASSERT_EQ(logTileCopy.name, "DataCopy");
    ASSERT_EQ(logTileCopy.args.size(), 3);

    // 验证每行的 src/dst offset
    ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(0) * sizeof(Element));
    ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * layoutDst.shape(2) * sizeof(Element));

    // 验证 DataCopyParams
    const AscendC::DataCopyParams* dataCopyArg =
        logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(dataCopyArg->blockCount, CeilDiv(_col, ELE_NUM_PER_C0));
    ASSERT_EQ(dataCopyArg->blockLen, _1);
    ASSERT_EQ(dataCopyArg->srcStride, _0);
    ASSERT_EQ(dataCopyArg->dstStride, C0_NUM_PER_FRACTAL - _1);
}
```

### 长 stride 逐行 Nd2Nz 分支

当 stride(0) >= STRIDE_LIMIT (65536) 时，搬运逻辑是**多轮循环的ND2NZ随路搬运**：

```cpp
ASSERT_EQ(logs.size(), _row);   // 每行一次 Nd2Nz DataCopy

for (int i = 0; i < _row; i++) {
    AscendCCallLog logTileCopy = logs[i];
    ASSERT_EQ(logTileCopy.name, "DataCopy");

    // 验证 src offset 使用长stride
    ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * _very_long_stride * sizeof(Element));
    ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * BYTE_PER_C0);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _1);       // 每行单独搬运
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, _0);    // 长stride时设为0
    ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
}
```

### Contiguous DataCopy 分支 (C0精确对齐)

当 shape(1)==ELE_NUM_PER_C0 && stride(0)==ELE_NUM_PER_C0 时，使用**基础数据搬运**：

```cpp
ASSERT_EQ(logs.size(), 1);
ASSERT_EQ(logs[0].name, "DataCopy");
ASSERT_EQ(logs[0].args.size(), 3);

// 第三个参数是元素个数 (而非 Nd2NzParams 指针)
// 通过检查 GetArgsAt(2) 的类型来判断
```

### 多参数重载 (Explicit Multi-Param)

CopyGmToL1 和 CopyGmToL1GMMPTD 提供了带 ndNum/srcNdMatrixStride/dstNzNStride/dstNzMatrixStride/dstNzC0Stride 的多参数重载：

```cpp
copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc,
           ndNum, srcNdMatrixStride, dstNzNStride, dstNzMatrixStride, dstNzC0Stride);

// 验证
const AscendC::Nd2NzParams* nd2nzArg = logs[0].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
ASSERT_EQ(nd2nzArg->ndNum, ndNum);
ASSERT_EQ(nd2nzArg->srcNdMatrixStride, srcNdMatrixStride);
ASSERT_EQ(nd2nzArg->dstNzNStride, dstNzNStride);
ASSERT_EQ(nd2nzArg->dstNzMatrixStride, dstNzMatrixStride);
ASSERT_EQ(nd2nzArg->dstNzC0Stride, dstNzC0Stride);
```

### 关于 ColumnMajor→nZ 分支

与 RowMajor→zN 镜像，区别在于：
- `nValue` = shape(1) (cols), `dValue` = shape(0) (rows)
- `srcDValue` = stride(1)
- `dstNzC0Stride` = stride(1) / ELE_NUM_PER_C0
- `dstNzNStride` = stride(2) / ELE_NUM_PER_C0

```cpp
ASSERT_EQ(nd2nzArg->nValue, _col);
ASSERT_EQ(nd2nzArg->dValue, _row);
ASSERT_EQ(nd2nzArg->srcDValue, layoutSrc.stride(1));
ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(2) / ELE_NUM_PER_C0);
```

---

## 测试场景清单

### CopyGmToL1<AtlasA2, RowMajor>

| # | 测试场景 | 关键 shape/stride | 预期 API |
|---|---------|------------------|---------|
| 1 | 正常 Nd2Nz | row=128, col=256, contiguous | 1× DataCopy + Nd2NzParams |
| 2 | 长stride逐行 | stride(0) >= 65536 | row× DataCopy + Nd2NzParams(nValue=1) |
| 3 | int4b_t 修正 | Element=int4b_t | dValue/srcDValue = CeilDiv(orig, 2) |
| 4 | 多参数重载 | 传入 5个额外参数 | Nd2NzParams 直接使用传入值 |
| 5 | 多参数重载-长stride | srcNdMatrixStride >= 65536 | ndNum× 逐行 |

### CopyGmToL1<AtlasA2, ColumnMajor>

| # | 测试场景 | 预期 |
|---|---------|------|
| 6 | 正常 Nd2Nz | 1× DataCopy + Nd2NzParams (dValue=rows, nValue=cols) |
| 7 | 长stride逐列 | stride(1)>=65536 → col× DataCopy |

### CopyGmToL1GMMPTD<AtlasA2, RowMajor>

| # | 测试场景 | 预期 |
|---|---------|------|
| 8 | 1-row DataCopyParams | shape(0)==1 → 1× DataCopyParams |
| 9 | 正常多行 Nd2Nz | shape(0)>1, stride<LIMIT → Nd2Nz |
| 10 | 长stride | stride(0)>=LIMIT → 逐行 |
| 11 | 多参数重载 | 传入额外参数 |

### CopyGmToL1DynamicOptimized<AtlasA2, RowMajor>

| # | 测试场景 | 预期 |
|---|---------|------|
| 12 | rows<=16 | 逐行 DataCopyParams |
| 13 | rows>16, 非C0对齐 | Nd2Nz |
| 14 | rows>16, C0精确对齐 | 1× Contiguous DataCopy |

### CopyGmToL1IntervalDataCopy (half only)

| # | 测试场景 | 预期 |
|---|---------|------|
| 15 | half RowMajor→zN | 逐行 DataCopyParams |
| 16 | half PaddingRowMajor→zN | 逐行 DataCopyParams (use orgShape) |
| 17 | half ColumnMajor→nZ | 逐列 DataCopyParams |
| 18 | half PaddingColumnMajor→nZ | 逐列 DataCopyParams (use orgShape) |

### 其他 Layout 特化 (AtlasA2)

| # | 测试场景 | Struct |
|---|---------|--------|
| 19 | zN→zN | CopyGmToL1 (DataCopyParams, short/long stride) |
| 20 | nZ→nZ | CopyGmToL1 (DataCopyParams, short/long stride) |
| 21 | PaddingRowMajor→zN | CopyGmToL1 (Nd2Nz with orgShape) |
| 22 | PaddingColumnMajor→nZ | CopyGmToL1 (Nd2Nz with orgShape) |
| 23 | VectorLayout→zN (A1) | CopyGmToL1 (Nd2Nz, 1-row) |
| 24 | RowMajor→RowMajor (A1) | CopyGmToL1 (DataCopy/DataCopyParams) |

### 3-Param GmType 特化 (AtlasA2)

| # | 测试场景 |
|---|---------|
| 25 | RowMajor→zN(A1) |
| 26 | RowMajor→zZ(B1) |
| 27 | ColumnMajor→nN(A1) |
| 28 | ColumnMajor→nZ(B1) |
| 29 | ColumnMajor→nZ(A1) |
| 30 | RowMajor→zN(B1) |

### TLA 变体 (AtlasA2)

| # | 测试场景 |
|---|---------|
| 31 | TileCopyTla RowMajor→zN |
| 32 | TileCopyTla ColumnMajor→nZ |
| 33 | TileCopyTla zN→zN |
| 34 | TileCopyTla nZ→nZ |
| 35 | TileCopyTlaExt PaddingRowMajor→zN |
| 36 | TileCopyTlaExt PaddingColumnMajor→nZ |
| 37 | TileCopyTlaExt ColumnMajor→nZ |
| 38 | TileCopyTlaExt zN→zN |
| 39 | TileCopyTlaExt nZ→nZ |
| 40 | TileCopySparseTla RowMajor/ColMajor→zN |
| 41 | TileCopySparseTla ColMajor→nZ |
| 42 | TileCopySparseTla zN→zN |
| 43 | TileCopySparseTla nZ→nZ |
| 44 | TileCopyFAQTla multi-matrix ND→NZ |

### Ascend950 特化

| # | 测试场景 |
|---|---------|
| 45 | TileCopyTla RowMajor→zN |
| 46 | TileCopyTla zN→zN |
| 47 | TileCopyTla ColumnMajor→nZ |
| 48 | TileCopyTla nZ→nZ |
| 49 | TileCopyTla Vector→Vector |
| 50 | TileCopyTla MX fp8_e8m0 RowMajor→zZ |
| 51 | TileCopyTla MX fp8_e8m0 ColumnMajor→zZ |
| 52 | TileCopyTla MX fp8_e8m0 RowMajor→nN |
| 53 | TileCopyTla MX fp8_e8m0 ColumnMajor→nN |
| 54 | CopyGmToL1 (No-TLA) RowMajor→zN |
| 55 | CopyGmToL1 (No-TLA) ColumnMajor→nZ |
| 56 | CopyGmToL1 (No-TLA) zN→zN |
| 57 | CopyGmToL1 (No-TLA) nZ→nZ |
| 58 | CopyGmToL1GMMPTD RowMajor→zN |
| 59 | CopyGmToL1DynamicOptimized RowMajor→zN |
| 60 | CopyGmToL1DynamicOptimized ColumnMajor→nZ |

---

