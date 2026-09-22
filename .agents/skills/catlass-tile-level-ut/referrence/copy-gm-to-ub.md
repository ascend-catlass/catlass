# CopyGm2Ub (GM→UB) UT 参考

被测源码：`include/catlass/epilogue/tile/copy_gm_to_ub.hpp` (246行)

---

## 组件族谱

### CopyGm2Ub Non-TLA 特化

| # | ArchTag | GmType | Element | 搬运API | Params类型 | padParams |
|---|---------|--------|---------|---------|-----------|-----------|
| 1 | AtlasA2 | RowMajor | float | DataCopyPad(4-param) | DataCopyExtParams | DataCopyPadExtParams |
| 2 | AtlasA2 | VectorLayout | float | DataCopyPad(4-param) | DataCopyExtParams | DataCopyPadExtParams |
| 3 | Ascend950 | RowMajor | float | DataCopyPad(4-param) | DataCopyExtParams | DataCopyPadExtParams |
| 4 | Ascend950 | VectorLayout | float | DataCopyPad(4-param) | DataCopyExtParams | DataCopyPadExtParams |

### CopyGm2UbAligned 特化

| # | ArchTag | GmType | 搬运API | 说明 |
|---|---------|--------|---------|------|
| 5 | AtlasA2 | RowMajor | DataCopy(3-param) | 连续时一次性拷贝，非连续时分块 DataCopyParams |
| 6 | Ascend950 | — | — | 不存在此特化 |

### CopyPerTokenScale2Ub 特化

| # | ArchTag | GmType | 说明 |
|---|---------|--------|------|
| 7 | Arch-tag-agnostic | ColumnMajor | 从 (m,1) ColumnMajor 拷贝到 (m,n) RowMajor 的第一列 |

---

### 断言必验字段

|API|Params 类型|必验字段|
|---|---|---|
|`AscendC::DataCopyPad`(4-param)|`DataCopyExtParams`|`blockCount, blockLen, srcStride, dstStride`|
|`AscendC::DataCopyPad`(4-param)|`DataCopyPadExtParams<Element>`|`isPad`|
|`AscendC::DataCopy`|`DataCopyParams`|`blockCount, blockLen, srcGap, dstGap`|
|`AscendC::DataCopy`|`uint32_t` (count)|验证 `*count == _totalLen`|

---

## 测试基础设施

### 关联Stub文件

| 文件 | 作用 |
|------|------|
| `stub/ascendc_test_fixture.h` | `AscendCTest` / `UBTileCopyTest` fixture |
| `stub/ascendc_logger.h` | `AscendCCallLogger` 单例，捕获 API 调用 |
| `common/helper.hpp` | `setLayout()` / `isContiguous()` |
| `common/shape.hpp` | `TestVectorShape` / `TestVectorShapeWithStride` |

### 测试Fixture成员

`UBTileCopyTest` 以 block 粒度描述 UB 数据，成员在 `setShape(blkLen, blkCnt)` 中算好：

```cpp
_blkLen   = blkLen;            // 每个 block 的元素数
_blkCnt   = blkCnt;            // block 行数
_totalLen = blkLen * blkCnt;  // 总元素数
```

### 日志索引约定

```cpp
// CopyGm2Ub (4-param DataCopyPad):
argsT[0] = MakeArg<Element>()   →  GetArgsTAt(0).Type() = typeid(Element)
args[0]  = dstTensor (LocalTensor/UB)
args[1]  = srcTensor (GlobalTensor/GM)
args[2]  = dataCopyParams       →  DataCopyExtParams
args[3]  = padParams            →  DataCopyPadExtParams<Element>

// CopyGm2UbAligned (3-param DataCopy):
args[0]  = dstTensor
args[1]  = srcTensor
args[2]  = count(uint32_t)      →  一次拷贝 total elements
```


---

## 断言模式

### CopyGm2Ub RowMajor (#1, #3)

GM→UB 标准搬运场景，使用 `DataCopyPad` API（4 参数，含 padParams）。连续布局下 `srcStride` 和 `dstStride` 均为 0，`isPad=false`。

```cpp
const auto* dataCopyParams = log.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
const auto* padParams = log.GetArgsAt(3).Value<AscendC::DataCopyPadExtParams<Element>>();
ASSERT_EQ(dataCopyParams->blockCount, _blkCnt);
ASSERT_EQ(dataCopyParams->blockLen, _blkLen * sizeof(Element));
ASSERT_EQ(dataCopyParams->srcStride, _0);  // 连续布局
ASSERT_EQ(dataCopyParams->dstStride, _0);  // 连续布局
ASSERT_EQ(padParams->isPad, false);
```

### CopyGm2Ub VectorLayout (#2, #4)

VectorLayout 场景将整个 GM buffer 视为单个连续 block，`blockCount=1`，`blockLen` 为总元素数的字节大小。

```cpp
const auto* dataCopyParams = log.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
ASSERT_EQ(dataCopyParams->blockCount, _1);
ASSERT_EQ(dataCopyParams->blockLen, _totalLen * sizeof(Element));
ASSERT_EQ(dataCopyParams->srcStride, _0);
ASSERT_EQ(dataCopyParams->dstStride, _0);
```

### CopyGm2UbAligned 连续路径 (#5)

连续布局下走 `DataCopy` API，第三个参数直接传总元素个数 count。

```cpp
ASSERT_EQ(log.name, "DataCopy");
const uint32_t* count = log.GetArgsAt(2).Value<uint32_t>();
ASSERT_EQ(*count, _totalLen);
```

### CopyGm2UbAligned 非连续路径

非连续布局时走逐块 `DataCopy` API，使用 `DataCopyParams` 参数。每块的 `blockLen`、`srcGap`、`dstGap` 根据实际 stride 和 block 长度计算。

```cpp
const auto* params = log.GetArgsAt(2).Value<AscendC::DataCopyParams>();
ASSERT_EQ(params->blockCount, currentBlockCount);
ASSERT_EQ(params->blockLen, _blkLen / ELE_NUM_PER_BLK);
ASSERT_EQ(params->srcGap, (_srcStride - _blkLen) / ELE_NUM_PER_BLK);
ASSERT_EQ(params->dstGap, (_dstStride - _blkLen) / ELE_NUM_PER_BLK);
```

---
