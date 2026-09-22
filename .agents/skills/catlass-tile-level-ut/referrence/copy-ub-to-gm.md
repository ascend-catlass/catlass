# CopyUb2Gm (UB→GM) UT 参考

被测源码：`include/catlass/epilogue/tile/copy_ub_to_gm.hpp` (173行)

---

## 组件族谱

### CopyUb2Gm Non-TLA 特化

| # | ArchTag | GmType | Element | 搬运API | Params类型 | 说明 |
|---|---------|--------|---------|---------|-----------|------|
| 1 | AtlasA2 | RowMajor | float | DataCopyPad(3-param) | DataCopyExtParams | 无 padParams |
| 2 | AtlasA2 | VectorLayout | float | DataCopyPad(3-param) | DataCopyExtParams | 无 padParams |
| 3 | Ascend950 | RowMajor | float | DataCopyPad(3-param) | DataCopyExtParams | 无 padParams；srcStride 使用 ELE_NUM_PER_C0 |
| 4 | Ascend950 | VectorLayout | — | — | — | 不存在此特化 |

### CopyUb2GmAligned 特化

| # | ArchTag | GmType | 搬运API | 说明 |
|---|---------|--------|---------|------|
| 5 | AtlasA2 | RowMajor | DataCopy(3-param) | 连续时一次性拷贝，非连续时分块 DataCopyParams |
| 6 | Ascend950 | — | — | 不存在此特化 |

---

### 断言必验字段

|API|Params 类型|必验字段|
|---|---|---|
|`AscendC::DataCopyPad` |`DataCopyExtParams`|`blockCount, blockLen, srcStride, dstStride`|
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
// CopyUb2Gm (3-param DataCopyPad, 无 padParams):
argsT[0] = MakeArg<Element>()   →  GetArgsTAt(0).Type() = typeid(Element)
args[0]  = dstTensor (GlobalTensor/GM)   →  从 UB 搬出到 GM
args[1]  = srcTensor (LocalTensor/UB)    →  源数据在 UB
args[2]  = dataCopyParams                →  DataCopyExtParams

// CopyUb2GmAligned (3-param DataCopy):
args[0]  = dstTensor
args[1]  = srcTensor
args[2]  = count(uint32_t)
```

**与 CopyGm2Ub 的关键差异**：
- CopyUb2Gm 的 `DataCopyPad` 只有 3 个参数，无 `padParams`
- Ascend950 RowMajor 的 `srcStride` 使用 `ELE_NUM_PER_C0` 分母（与 AtlasA2 一致）

---

## 断言模式

### CopyUb2Gm RowMajor (#1, #3)

UB→GM 标准搬运场景，使用 `DataCopyPad` API（3 参数，无 padParams）。连续布局下 `srcStride` 和 `dstStride` 均为 0。

```cpp
ASSERT_EQ(log.name, "DataCopyPad");
ASSERT_EQ(log.args.size(), 3);  // 无 padParams

const auto* dataCopyParams = log.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
ASSERT_EQ(dataCopyParams->blockCount, _blkCnt);
ASSERT_EQ(dataCopyParams->blockLen, _blkLen * sizeof(Element));
ASSERT_EQ(dataCopyParams->srcStride, _0);  // 连续布局
ASSERT_EQ(dataCopyParams->dstStride, _0);  // 连续布局
```

### CopyUb2Gm VectorLayout (#2)

VectorLayout 场景将整个 UB buffer 视为单个连续 block，`blockCount=1`，`blockLen` 为总元素数的字节大小。

```cpp
const auto* dataCopyParams = log.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
ASSERT_EQ(dataCopyParams->blockCount, _1);
ASSERT_EQ(dataCopyParams->blockLen, _totalLen * sizeof(Element));
ASSERT_EQ(dataCopyParams->srcStride, _0);
ASSERT_EQ(dataCopyParams->dstStride, _0);
```

### CopyUb2GmAligned 连续路径

CopyUb2GmAligned 在连续布局下走 `DataCopy` API，第三个参数直接传总元素个数 count。

```cpp
ASSERT_EQ(log.name, "DataCopy");
const uint32_t* count = log.GetArgsAt(2).Value<uint32_t>();
ASSERT_EQ(*count, _totalLen);
```

