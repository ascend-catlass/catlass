# 排序与 TopK 算子

本目录实现 **Sort** 与 **TopK** 两个算子，只用 TLA 已有的向量原语组合。

## 算法说明

### Sort（`sort.py`）

给定一维 key 序列，输出按 key 升序（或降序）排列的值序列，以及每个输出元素在
原输入中的位置（置换索引）。排序稳定。

| 张量 | 方向 | 说明 |
|---|---|---|
| `mem_keys` | 输入 | GM 上的 key（以 i32 位模式承载） |
| `mem_val` | 输出 | 排序后的值 |
| `mem_idx` | 输出 | 置换：`mem_keys_original[mem_idx[i]] == mem_val[i]` |

LSD 基数排序，一个 2-bit pass 处理 2 个二进制位，一个字节轮跑 4 个 pass
（shift 0 / 2 / 4 / 6）。每个 pass 三步：

| 步骤 | 用到的原语 | 说明 |
| --- | --- | --- |
| 桶起始偏移 | `histograms` | FREQUENCY 与 ACCUMULATE 相减得到每桶基址。FREQUENCY 是「恰好等于 n 的个数」，ACCUMULATE 是「≤ n 的个数」，相减即「< n 的个数」，也就是桶 n 的起始偏移 |
| 桶内压缩 | `cmp` + `squeeze` | `cmp` 出 4 个桶掩码，`squeeze` 把索引按桶压缩 |
| 落桶 | `scatter` | 把每个桶的元素落到第 1 步给出的桶基址 |

带符号与降序通过翻转映射成无符号可比值，排完翻回：有符号整数 XOR 符号位
（`0x80`）；降序对符号位掩码做 XOR 或整位取反。

### TopK（`topk.py`）

取一维 key 序列中最大（或最小）的 k 个，输出其值与在原输入中的位置。

复用 `sort.py` 的 `_pass` 排序，再用 `lane < K` 的掩码 `squeeze` 压出前 k 个，
值与索引同时压缩。取最大需降序，故排序前整位取反、排完再取反。

## 代码组织

```plain
./sort_topk
├── sort.py
├── topk.py
└── README.md
```

| 文件 | 概述 |
| --- | --- |
| [**`sort.py`**](sort.py) | Sort 算子实现与可运行入口。`_pass` 是一个 2-bit pass，`sort_kernel` 按字节轮调用 4 次。 |
| [**`topk.py`**](topk.py) | TopK 算子实现与可运行入口。复用 `sort.py` 的 `_pass`，再压缩前 k 个。 |

## 约束说明

| 维度 | Sort | TopK |
| --- | --- | --- |
| 产品 / arch | Ascend 950PR，`--npu-arch 3510`（`dav-c310`） | 同左 |
| dtype | `u8`、`i8` | `u8` |
| 方向 | 升序 / 降序 | 取最大 / 取最小 |
| 长度 | 一个向量寄存器（256 个 key） | 同左 |
| k | — | `1 <= k <= 256`（host 侧校验） |

### 实现约束（实测）

**Sort 的每个 pass 必须写到一块独立的目标缓冲。** 曾尝试用 4 块轮转复用
（`t0→t1→t2→t3→t0`）以节省 UB，实测结果错误 —— 最后一个 pass 的目标会覆盖
第一个 pass 的源。5 块独立缓冲实测正确。

**TopK 的 k 常量不能用 `Int8` 字面量构造。** `tla.full(k, dtype=Int8)` 在
`k > 127` 时越界（i8 上限 127）。本样例按 k 的 u8 位模式构造有符号常量
（`k > 127` 时取 `k - 256`），`k == 256` 时 `lane < 256` 恒真，改用全 1 掩码。

## 运行方式

在 CATLASS 仓库根目录执行：

```bash
# Sort
python examples/python/tla_dsl/dsl_tutorials/advanced_impl/sort_topk/sort.py --device 0
python examples/python/tla_dsl/dsl_tutorials/advanced_impl/sort_topk/sort.py --device 0 --dtype i8 --desc 1

# TopK
python examples/python/tla_dsl/dsl_tutorials/advanced_impl/sort_topk/topk.py --device 0 --k 8
python examples/python/tla_dsl/dsl_tutorials/advanced_impl/sort_topk/topk.py --device 0 --k 8 --smallest 1
```

传入 `--help` 可查看全部参数。

输出示例：

```text
[u8] sort desc=False: values=OK indices=OK
topk k=8 largest=True: values=OK indices=OK
```

## 测试覆盖

本目录已接入端到端 battery（`tests/dsl_battery/test_battery.py`），共 11 条用例：

| 用例 | 覆盖点 |
| --- | --- |
| `sort-u8-asc` / `sort-u8-desc` | u8 升序 / 降序 |
| `sort-i8-asc` / `sort-i8-desc` | i8 升序 / 降序 |
| `topk-k1` / `topk-k8` | 常规 k |
| `topk-k127` / `topk-k128` | 跨越 Int8 常量边界 |
| `topk-k200` / `topk-k256` | k 的高位段与上界 |
| `topk-k8-smallest` | 取最小 |

运行：

```bash
TLA_DSL_RUN_BATTERY=1 python -m pytest tests/dsl_battery/test_battery.py \
    -k "sort-u8 or sort-i8 or topk-" -q
```

## 验证结果

设备端自验，比对基准为 `torch.sort`，同时校验值与置换索引两个维度。

| dtype | 升序 | 降序 |
| --- | --- | --- |
| `u8` | values=OK indices=OK | values=OK indices=OK |
| `i8` | values=OK indices=OK | values=OK indices=OK |

TopK 取最大，k = 1 / 8 / 64 / 127 / 128 / 200 / 255 / 256 全部 values=OK indices=OK。
