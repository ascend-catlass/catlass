# `tla.print` 打印示例

[English](README_en.md)

## 功能说明

本目录展示 `tla.cube()`／`tla.vector()` 和 SIMD／SIMT `tla.vec.func` 中的打印，并检查输出及计算结果。

| 示例 | 展示内容 |
| --- | --- |
| [`debug_print.py`](debug_print.py)、[`debug_print_format.py`](debug_print_format.py) | 外层标量、标量表达式和格式化字符串 |
| [`vf_print.py`](vf_print.py) | SIMD 标量、寄存器和 UB 数据 |
| [`simt_print.py`](simt_print.py) | SIMT 线程标量、GM 和 UB 数据 |
| [`vf_print_mixed.py`](vf_print_mixed.py) | Cube、Vector、SIMD 和 SIMT 的组合调用 |

类型、内存位置、`length`、同步和日志容量见[打印指南](../../../../../python/tla_dsl/docs/zh/kernel_development/evaluation/print.md)。`tla.print` 不支持指针格式化。

## 运行示例

以下命令在仓库根目录执行。`--device` 是当前程序可见设备的逻辑编号。含 `tla.vec.func` 的示例不传 `--run` 时只编译，传入 `--run --device` 时编译并运行。

### 标量表达式

`debug_print.py --expression` 在 `tla.vector()` 内打印 `lhs + rhs`，`--value` 和 `--rhs` 分别指定两个操作数：

```bash
python examples/python/tla_dsl/dsl_tutorials/debug_print/debug_print.py --expression --dtype f32 --value 1.25 --rhs 0.75 --run --device 0
```

### SIMD 打印

```bash
python examples/python/tla_dsl/dsl_tutorials/debug_print/vf_print.py --run --device 0 --calls 1
```

`--calls 1` 只启动一次。示例打印寄存器结果的前 4 个元素 `[-7, -6.5, -6, -5.5]` 和输入／输出 UB 前缀，检查通过后输出 `output_ok=True`。

### SIMT 打印

`simt_print.py` 展示选择打印线程、共享 UB 的全线程屏障，以及更新输入后重复调用同一编译对象。

```bash
# 只编译。
python examples/python/tla_dsl/dsl_tutorials/debug_print/simt_print.py

# 编译并运行。
python examples/python/tla_dsl/dsl_tutorials/debug_print/simt_print.py --run --device 0
```

### 混合打印

`vf_print_mixed.py` 使用 C1V2 核配比和两个逻辑 block。每个 block 有两个 Vector 子块，各处理 64 个元素；输出中的 `subblock=0` 和 `subblock=1` 区分两个子块。示例检查打印值与 block／subblock 标识，核之间不保证输出顺序。

```bash
python examples/python/tla_dsl/dsl_tutorials/debug_print/vf_print_mixed.py --run --device 0 --block-num 2
```
