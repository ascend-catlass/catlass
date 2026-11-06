---
nav_order: 30
---

# 在 CATLASS Python 中使用打印调试接口

CATLASS Python 提供 `tla.print` 接口，用于在 kernel 内部进行调试打印。`tla.print` 有以下用法：

| 用法        | 形态                                              | 说明                                 |
| ----------- | ------------------------------------------------- | ------------------------------------ |
| 标量打印    | `tla.print(value)` / `tla.print(fmt, *args)`      | 打印单个标量、纯字符串或格式化字符串 |
| Tensor 打印 | `tla.print(tensor)` / `tla.print(tensor, length)` | 从 Tensor 视图起始地址打印连续物理前缀 |
| SIMD 寄存器打印 | `tla.print(values)` / `tla.print(values, length)` | 打印 `VectorSSA` 中的元素 |

标量、Tensor 和寄存器均支持 `f16`、`f32`、`i8`、`i16`、`i32`、`u8`、`u16`、`u32`。

`tla.print` 只接受位置参数，`length` 须为整数，单位为元素。打印 Tensor 时，只有静态形状且全部元素不超过单次上限，才可以省略 `length`；其他情况须显式指定长度。

下文“外层”指 `tla.cube()` 或 `tla.vector()` 内、`tla.vec.func` 之外的代码。

## 使用示例

下面以 `examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py` 中的 `basic_mmad_kernel` 为例。

### 标量打印

标量打印适合查看运行时整数、浮点数、block 调度和执行流。

```diff
# examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py
    with tla.cube():
+       tla.print(123)  # 打印常量
+       tla.print("Hello world")  # 打印字符串
        tla.set_flag(l1a0_available)
        ...

        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n
+           tla.print(
+               "tile={} row={} col={}", block_linear, block_row, block_col
+           )  # 打印格式化字符串
            ...
            for k_l1 in k_l1_range:
                ...
                for k_l0 in k_l0_range:
                    ...
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
+                   if k_l1 == 0 and k_l0 == 0:
+                       tla.print("first mmad done")
```

上面的代码展示了三种标量打印：

1. `tla.print(123)`：打印单个标量。
2. `tla.print("Hello world")`：打印纯字符串。
3. `tla.print("tile={} ...", ...)`：格式化打印 `block_linear`、`block_row` 和 `block_col` 等运行时值。

纯字符串和格式串须为编译期 ASCII 字符串，格式串最多包含 8 个 `{}` 标量字段；`{{`、`}}` 输出花括号。
SIMD 格式化打印的完整记录最多为 2024 字节，包含记录头、参数槽、字符串结束符和对齐开销，超限在编译时报错。

### Tensor 打印

Tensor 打印适合查看 GM 或 UB 中的数据。

```diff
# examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py
    with tla.cube():
+       tla.print(gm_a, 256)
        tla.set_flag(l1a0_available)
        ...
```

`tla.print(gm_a, 256)` 打印 GM 输入 A 从有效地址开始的连续 256 个元素。使用 Basic MMAD 的默认参数时，`gm_a` 的 dtype 为 `f16`，shape 为 `[256, 1024]`。

### Tensor 与寄存器打印支持范围

| 调用位置 | 可打印的数据 | 单次打印范围 |
| --- | --- | --- |
| 外层 `tla.cube()` | GM Tensor | 一维／二维 Tensor，最多 262112 个元素 |
| 外层 `tla.vector()` | GM、UB Tensor | 一维／二维 Tensor，最多 262112 个元素 |
| SIMD `tla.vec.func` | 寄存器 | 最多一个 256 字节寄存器；FP32 最多 64 个元素，FP16 最多 128 个元素 |
| SIMD `tla.vec.func` | UB Tensor | 一维／二维 Tensor，数据最多 1952 字节；FP32 最多 488 个元素，FP16 最多 976 个元素 |
| SIMT `tla.vec.func` | GM、UB Tensor | 静态形状的一维／二维 Tensor，最多 262112 个元素 |

上表中的限制来源不同。当前 Ascend 950 目标的向量寄存器容量为 256 字节。CANN 9.2 的 `asc_debug_types.h` 为 VF 调试预留 2048 字节 UB，扣除 24 字节控制头后，一条完整记录最多占 2024 字节；Tensor 记录再扣除 72 字节记录头，剩余 1952 字节用于数据。

GM／UB 的 262112 个元素沿用外层打印的统一安全上限：从 1 MiB FIFO 中扣除 48 字节形状记录头和 72 字节 Tensor 记录头，再将数据区向下按 32 字节对齐，按 FP32 元素计算。此元素上限也适用于其他支持的 dtype 和 SIMT。

一维／二维支持范围以及 SIMT shape 的 32 位无符号整数编码，属于当前前端和日志协议的软件约定。

- 编译期已知的 Tensor 各维度须大于 0。
- Tensor 省略 `length` 时打印全部元素，要求形状静态且元素总数不超过表中的单次上限。
- 编译期整数 `length` 须大于 0，且不超过 Tensor 元素数和单次上限，否则报错。运行时 `length` 小于等于 0 时不打印。
- 在 SIMD／SIMT 内，正的运行时长度取 `length`、Tensor 元素数和单次上限中的最小值。在外层，运行时长度超过 Tensor 元素数或单次上限时，不产生 Tensor 记录。
- GM／UB Tensor 打印从视图起始地址开始的连续物理前缀，不按 stride 选取元素，也不重排布局。例如，底层数据为 `[0, 1, 2, 3, 4, 5, 6, 7]`，shape 为 4、stride 为 2 的视图逻辑元素是 `[0, 2, 4, 6]`，而 `tla.print(view, 4)` 打印 `[0, 1, 2, 3]`。调用方保证这段连续内存可读。SIMT 中通过 `view[index]` 取数后打印标量，则按视图的 stride 寻址。
- 外层 UB 打印的视图起始地址须按 32 字节对齐，未对齐时不产生 Tensor 记录。
- SIMT Tensor 必须为静态形状，每一维在 `1..4294967295` 内，超限在编译时报错。
- 寄存器省略 `length` 时，使用编译期已知的有效元素数；无法在编译期确定时，使用物理容量。显式静态长度须在 `1..物理容量` 内，运行时长度限制到物理容量；小于等于 0 时不打印。未写入的寄存器元素没有确定的打印值。

### 同步与缓冲区复用

在 SIMD／SIMT `vec.func` 内，库负责打印自身的取数、日志写入和发布，使用必要的访问屏障。成功写入的日志保留本次取数的值，随后覆盖源 UB 不会改变它。打印前，用户确保搬运或其他线程、核的写入已完成；取数期间，其他执行单元不能同时覆盖这些数据。

在同一 SIMD `vec.func` 内，打印通过内部屏障安排自身取数与前后 Vector store 的先后顺序，无需为打印额外调用 `tla.local_mem_bar`。下面的代码位于 `tla.vector()` 内，两个 UB Tensor 已分配，输入数据已就绪：

```python
with tla.vec.func(mode="simd"):
    values = input_ub.load() * 2.0
    tla.print(values, 4)
    output_ub.store(values)
    tla.print(output_ub, 4)
    output_ub.store(values + 1.0)
```

UB 日志保留 `values` 的前 4 个元素；最后一次 store 不改变已保存的记录。

在 SIMT 中，`tla.print` 只由当前线程执行，可用 `if thread == 0:` 等条件选择打印线程。库用原子操作和访问屏障保护日志写入，不隐式执行全线程屏障；只打印线程自己的标量时，无需为打印增加全线程屏障。打印其他线程写入的共享 UB 前，整个线程块先执行屏障。其他线程随后要覆盖同一 UB 时，打印后再次执行全线程屏障，等待取数完成：

```python
with tla.vec.func(mode="simt", thread_block_dim=64):
    thread, _, _ = tla.arch.thread_idx()
    shared_ub[thread] = source_gm[thread] * 2.0
    tla.arch.sync_threads()
    if thread == 0:
        tla.print(shared_ub, 8)
    tla.arch.sync_threads()
    shared_ub[thread] = shared_ub[thread] + 1.0
    output_gm[thread] = shared_ub[thread]
```

示例位于 `tla.vector()` 内，GM 和 UB Tensor 均有 64 个 FP32 元素，UB 在 `vec.func` 外分配。两个屏障都放在 `if thread == 0:` 外，由线程块的全部线程执行；若放在分支内，其他线程不到达屏障，可能死锁。后续没有其他线程覆盖这些数据时，无需因打印增加第二个屏障。

搬运、跨流水、跨核及跨流的数据依赖继续由用户按算法表达；打印的内部屏障仅保护自身访问。
打印会改变执行时序，算法在移除打印后也须保持同步正确。

## 编译运行

同一 kernel 中可以组合标量、GM／UB Tensor 和 VF 打印，无需额外编译选项：

```bash
cd examples/python/tla_dsl/tla/kernel/matmul/basic_mmad

# 单 block 启动方便观察输出
python basic_matmul.py --device 0 --block-num 1
```

## 输出示例

### 标量打印输出

标量、纯字符串和格式化打印均以 `TLA printf:` 为前缀。下面是外层打印的输出格式：

```text
TLA printf: core=0 block=0 x=123
TLA printf: core=0 block=0 Hello world
TLA printf: core=0 block=0 tile=0 row=0 col=0
TLA printf: core=0 block=0 first mmad done
...
passed=True cache_key=<cache-key>
kernel.o=<cache-dir>/kernel.o
```

SIMT 标量输出包含 block 和 thread 坐标，例如 `simt_print.py` 中线程 0 的输出：

```text
TLA printf: block=(0, 0, 0) thread=(0, 0, 0) tid=0 x=0.000000
```

### Tensor 打印输出

外层和 SIMD Tensor 输出以 `DumpTensor:` 为前缀。下面展示 `tla.print(gm_a, 256)` 的格式，省略部分数值：

```text
DumpTensor: call=0, block=0, data_type=float16, position=GM, shape=[256,1024] dump_size=256 [-1.66015625, -2.26953125, 3.0703125, 0.9375, ...]
```

`shape` 是传入 Tensor 视图的形状，`dump_size` 是本次打印的元素数。打印前缀时，两者的元素数可以不同。SIMD 寄存器的位置显示为 `REG`；MIX kernel 的 Vector 记录还包含 `subblock`，区分同一逻辑 block 的 Vector 子块。

SIMT Tensor 输出同样使用 `DumpTensor:`，并显示 block／thread 坐标。例如，`simt_print.py` 中共享 UB 的前 8 个元素为：

```text
DumpTensor: block=(0, 0, 0) thread=(0, 0, 0) dtype=float32 shape=(64,) [0, 2, 4, 6, 8, 10, 12, 14]
```

实际数值和产物路径由输入及运行环境决定。不同线程、核和打印通道之间不保证输出顺序。

## 注意事项

- 打印用于调试，会增加日志缓冲区、数据传输和同步开销。主机读取输出前会同步本次启动使用的设备流。正式运行和性能测量前移除打印代码；运行时条件跳过打印仍可能保留这些开销。
- SIMD 打印要求 CANN 提供 VF 调试支持，并保留默认预留 UB 配置。
- Tensor 打印支持 GM、UB，以及 C310 上外层 `tla.cube()` 的 L1/L0C Tensor。L1 仅支持
  32 字节对齐的 zN/nZ 稠密前缀，C0 维度须占 32 字节，最多打印 8 个元素；
  L0C 仅支持 32 字节对齐的 L0Clayout 16x16 完整分形（256 个元素）。两者
  支持 `f16`、`f32`、`i8`、`i16`、`i32`、`u8`、`u16`、`u32`。该本地
  buffer 路径依赖 CANN 的私有 DebugTunnel ABI，当前已在 CANN 9.2.0 验证。
  升级 CANN 时须核对该 ABI 并验证 L1/L0C 打印。L1 还要求驱动提供
  `halGetMaxResMapType`、`halResMap` 和 `halResUnmap`。同一 kernel 不能同时打印 L1 和 L0C；L0A/L0B 暂不支持。
- 异常或用户中断后，若无法确认设备已结束访问打印缓冲区，运行时会保留缓冲区，避免提前释放。此后在新进程中继续调试，避免重复启动累计占用显存；确认完成后会正常释放。
- 日志按完整记录写入。剩余空间放不下整条记录时，保留已完成记录，kernel 继续计算，具体规则如下。

| 打印位置 | 日志容量 | 空间不足时 |
| --- | --- | --- |
| 外层和 SIMD | 共用数据区，固定分为 108 个 1 MiB 分区，合计 108 MiB，控制信息另计 | 该分区停止接收后续记录；输出 `output truncated` 和对应核标识，其他分区继续写入 |
| SIMT | 每次启动独立分配 1 MiB 数据区，所有 block／thread 共用，另有 64 字节控制头 | 丢弃放不下的本次调用及后续调用；输出 `dropped N print call(s)`，汇总丢弃次数 |

容量包含记录头和对齐开销。一次 Tensor 打印作为一条记录，不会只保存其中一部分元素。只有外层 Tensor 打印的 kernel，如果编译时可确定的累计输出超过容量，会在启动前报错。

减少打印次数或 `length` 可以减少日志量。示例中的 `--block-num 1` 可以减少同时打印的核数；它不减少外层／SIMD 的固定日志分配，也不保证减少单个 tile 内的打印行数。

完整 VF 调用方式见 [打印示例](../../../../../../examples/python/tla_dsl/dsl_tutorials/debug_print/README.md)。CANN 调测接口参考：

- [Ascend C DumpTensor](https://www.hiascend.com/document/redirect/CannCommunityascendcapidumptensor)
- [Ascend C printf](https://www.hiascend.com/document/redirect/CannCommunityascendcapiprintf)
