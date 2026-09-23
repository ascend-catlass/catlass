# Catlass Unittest

`tests/unittest/catlass/gemm/tile` 目录下包含对当前Catlass GEMM类算子Tile组件的全量打桩单元测试，验证范围包括：

1. 组件实例化正确性：模板特化能否被正确选择；
2. API 调用正确性：调用的`AscendC`基础API名称, 形参数量是否符合预期；
3. 参数计算正确性：`AscendC`基础API所使用的结构体（如`AscendC::MmadParams`）的各字段值与逻辑推导是否一致；
4. 分支覆盖：对`include/catlass/gemm/tile`下的全量Tile层组件均有用例覆盖。

## 目录结构

```plain
tests/unittest/catlass/gemm/tile
├── ascend950                              # ascend950架构下Tile层组件UT
├── atlasa2                                # atlasa2架构下Tile层组件UT
├── common                                 # 辅助组件
├── README.md
├── test_copy_l1_to_fp.cpp                 # L1->FP数据通路测试
├── test_tile_copy_l1_to_bt.cpp            # L1->BT数据通路测试
├── test_tile_copy_gm_to_l1.cpp            # GM->L1数据通路测试(路由至ascend950/和atlasa2/下)
├── test_tile_copy_gm_to_l1_tla.cpp        # GM->L1数据通路测试(TLA组件)
├── test_tile_copy_gm_to_ub.cpp            # GM->UB数据通路测试(路由至ascend950/和atlasa2/下)
├── test_tile_copy_l0c_to_gm.cpp          # L0C->GM数据通路测试(路由至ascend950/和atlasa2/下)
├── test_tile_copy_l0c_to_ub.cpp          # L0C->UB数据通路测试(路由至ascend950/和atlasa2/下)
├── test_tile_copy_l1_to_l0a.cpp        # L1->L0A数据通路测试(路由至ascend950/和atlasa2/下)
├── test_tile_copy_l1_to_l0b.cpp        # L1->L0B数据通路测试(路由至ascend950/和atlasa2/下)
├── test_tile_copy_ub_to_gm.cpp          # UB->GM数据通路测试(路由至ascend950/和atlasa2/下)
└── test_tile_mmad.cpp                    # cube计算单元测试
```

## 测试框架

本套测试采用**打桩（stub）单元测试**方式，**不依赖真实 NPU 硬件**即可在主机侧（Host）编译运行：将被测组件实际调用的`AscendC`基础API（如 `DataCopy` / `LoadData` / `Fixpipe` / `Mmad` 等）替换为打桩实现，每次调用会被 `AscendCCallLogger` 单例记录下来，测试用例据此断言被测组件调用的逻辑正确性。

整体分为三层，自上而下依次为：

- 测试用例层（`test_tile_*.cpp`，基于已有的Fixture，构造`layout/tensor`、调用被测组件、捕获日志并断言）
- 打桩层（`tests/unittest/stub/*.h`，拦截并记录 `AscendC`基础API调用
- 被测源码层（`include/catlass/gemm/tile/` 等）。每个测试单元的典型流程如下图所示：

```plain
┌─────────────────────────────────────────────┐
│  TEST CASES  (test_tile_*.cpp)              │
│  GTest TEST_F macros                        │
│  1. setup layout & tensor                   │
│  2. invoke component under test             │
│  3. capture logs via AscendCCallLogger      │
│  4. ASSERT on name / args / params fields   │
├─────────────────────────────────────────────┤
│  STUB LAYER  (tests/unittest/stub/*.h)      │
│  AscendC API stubs → log each call          │
│  DataCopy / LoadData / Fixpipe / Mmad ...   │
├─────────────────────────────────────────────┤
│  SOURCE CODE  (include/catlass/gemm/tile/)  │
│  CopyGmToL1 / CopyL1ToL0A / CopyL0CToGm ... │
└─────────────────────────────────────────────┘
```

## 构建与测试

### 环境要求

以下是UT测试组件的构建要求：

- `gcc` >= 7.5, <= 12.0
- `cmake` >= 3.16
- `lcov` >= 1.16 (可选)
- `googletest` >= 1.14.0 (可自动拉取)

### 编译与测试

在主目录下执行下述编译：

```bash
bash scripts/build.sh --tests catlass_unittest

# 启用覆盖率检查
bash scripts/build.sh --tests catlass_unittest -DENABLE_COVERAGE=ON
```

默认在`build/tests/unittest/`下生成可执行文件，并以代际编号区分，分别执行如下：

```bash
# AtlasA2 代际下Tile组件单元测试
./build/tests/unittest/catlass_unittest_2201

# Ascend950 代际下Tile组件单元测试
./build/tests/unittest/catlass_unittest_3510
```

执行上述测试动作后可进一步生成覆盖率报告（默认在`build/coverage/`目录下生成），执行：

```bash
cd build && make coverage_collect
```

上述过程也可以直接运行测试脚本`tests/run_unittest.sh`完成，预期测试例通过且全覆盖。

```plain
Overall coverage rate:
  lines......: 100.0% (2516 of 2516 lines)
  functions..: 99.3% (1479 of 1489 functions)
Reading tracefile build/coverage/coverage.info
                                               |Lines      |Functions|Branches
Filename                                       |Rate    Num|Rate  Num|Rate   Num
================================================================================
[/xxx/catlass/include/]
catlass/coord.hpp                              | 100%    17| 100%  24|    -    0
catlass/detail/alignment.hpp                   | 100%    10| 100%  25|    -    0
catlass/epilogue/tile/copy_gm_to_ub.hpp        | 100%    68| 100%   6|    -    0
catlass/epilogue/tile/copy_gm_to_ub_tla.hpp    | 100%    19| 100%   2|    -    0
catlass/epilogue/tile/copy_ub_to_gm.hpp        | 100%    45| 100%   5|    -    0
catlass/epilogue/tile/copy_ub_to_gm_tla.hpp    | 100%    10| 100%   1|    -    0
catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp  | 100%   417| 100%  46|    -    0
catlass/gemm/tile/ascend950/copy_l0c_to_gm.hpp | 100%    99| 100%  10|    -    0
catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp | 100%    44| 100%   3|    -    0
catlass/gemm/tile/ascend950/copy_l1_to_bt.hpp  | 100%    22| 100%   8|    -    0
catlass/gemm/tile/ascend950/copy_l1_to_l0a.hpp | 100%    98| 100%  16|    -    0
catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp | 100%   162| 100%  20|    -    0
catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp    | 100%   611| 100%  68|    -    0
catlass/gemm/tile/atlasa2/copy_gm_to_ub.hpp    | 100%    12| 100%   2|    -    0
catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp   | 100%    69| 100%   9|    -    0
catlass/gemm/tile/atlasa2/copy_l1_to_bt.hpp    | 100%     9| 100%   4|    -    0
catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp   | 100%   171| 100%  22|    -    0
catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp   | 100%   219| 100%  26|    -    0
catlass/gemm/tile/atlasa2/copy_ub_to_gm.hpp    | 100%    11| 100%   2|    -    0
catlass/gemm/tile/copy_l1_to_fp.hpp            | 100%     9| 100%   8|    -    0
catlass/gemm/tile/tile_mmad.hpp                | 100%    91| 100%  20|    -    0
catlass/layout/matrix.hpp                      | 100%   139| 100%  63|    -    0
catlass/layout/vector.hpp                      | 100%     7| 100%   4|    -    0
catlass/numeric_size.hpp                       | 100%     2| 100%   1|    -    0
tla/int_tuple.hpp                              | 100%    16| 100%  39|    -    0
tla/layout.hpp                                 | 100%    85|99.6% 242|    -    0
tla/numeric/integral_constant.hpp              | 100%     4| 100%  17|    -    0
tla/numeric/math.hpp                           | 100%     4| 100%   3|    -    0
tla/tensor.hpp                                 | 100%    24|99.2% 241|    -    0
tla/tuple.hpp                                  | 100%    22|98.7% 552|    -    0
================================================================================
                                         Total:| 100%  2516|99.3%  1k|    -    0
Built target coverage_collect
```
