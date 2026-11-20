# DSL 教学示例

本目录按主题收录 **CATLASS Python** 的教学示例：每个子目录聚焦一种指令类别或一条语言特性，可独立运行。运行命令均在 CATLASS 仓库根目录执行，路径形如 `examples/python/tla_dsl/dsl_tutorials/<子目录>/<脚本>.py`。

## 指令类别

| 目录 | 概述 |
|------|------|
| [`vector_simd_ops/`](vector_simd_ops/README.md) | SIMD 向量指令全集：二元 / 单目 / 比较、mask、各类 load-store、归约、寄存器控制流等，附带执行流程讲解。 |
| [`vector_simt_ops/`](vector_simt_ops/) | SIMT 逐线程编程，见下表。 |

### SIMT（vector_simt_ops）

| 样例 | 概述 |
|------|------|
| [`basic_vadd_simt/`](vector_simt_ops/basic_vadd_simt/basic_vadd_simt.py) | 最小 SIMT 示例：一个线程处理一个元素，直接从 GM 读写。 |
| [`multi_ops_simt/`](vector_simt_ops/multi_ops_simt/README.md) | SIMT 区内可用的标量运算，覆盖全部元素类型。 |
| [`multiple_blocks_simt/`](vector_simt_ops/multiple_blocks_simt/README.md) | 多 Block 协作的 SIMT 向量加，使用 3D 线程坐标线性化切分任务。 |
| [`rank2_view/`](vector_simt_ops/rank2_view/rank2_view.py) | 在 SIMT 区内读取 rank-2 UB tensor。 |
| [`reduction_simt/`](vector_simt_ops/reduction_simt/reduction_simt.py) | 通过线程间共享的 UB buffer 完成 SIMT block 归约。 |
| [`simt_mixed/`](vector_simt_ops/simt_mixed/simt_mixed.py) | 单个 kernel 中串联 Cube、SIMD 与 SIMT 工作，全部通过 UB 交接。 |
| [`ub_offset_view/`](vector_simt_ops/ub_offset_view/ub_offset_view.py) | 同一块 UB 分配的两个视图（其一带偏移），在 SIMT 区内读取。 |

## 语言特性

| 目录 | 概述 |
|------|------|
| [`structured_arguments/`](structured_arguments/README.md) | Python 结构化参数如何进入设备端：静态 Tensor 接口、标量子类参数与混合 memref 参数。 |
| [`mixed_memref_arguments/`](mixed_memref_arguments/README.md) | Dynamic-GM Tensor、普通静态 Tensor 与标量混合作 kernel 参数。 |
| [`dataclass_arg/`](dataclass_arg/dataclass_arg.py) | 以标准库 `@dataclass` 实例作为 kernel 参数。 |
| [`constexpr_callable/`](constexpr_callable/constexpr_callable_epilogue.py) | `tla.Constexpr` Callable 形式的 epilogue 在 staging 阶段被内联。 |
| [`jit_callable/`](jit_callable/jit_callable_epilogue.py) | `@tla.jit` 辅助函数被当作 `tla.Constexpr` Callable 参与 staging。 |
| [`lazy_conditions/`](lazy_conditions/lazy_conditions.py) | DSL 控制流中 `and` / `or` 惰性求值的运行时探测。 |
| [`kernel_in_class/`](kernel_in_class/README.md) | 以类实例组织 kernel，编译期输入由实例携带、不占启动 ABI。 |
| [`tensor_index/`](tensor_index/scalar_index_control_flow.py) | GM tensor 标量索引（`tensor[i]` / `tensor[r, c]`）以及索引参与控制流。 |

## 标量与同步

| 目录 | 概述 |
|------|------|
| [`scalar_ops/`](scalar_ops/scalar_ops.py) | 在每个核自身的标量单元上执行的运算。 |
| [`scalar_arg_alignment/`](scalar_arg_alignment/scalar_arg_alignment.py) | tensor / scalar / tensor 形式的启动参数与对齐约束。 |
| [`cross_flag_two_way/`](cross_flag_two_way/README.md) | 一个 AIC 与两个 AIV 子块之间的双向 mode-4 cross-flag 握手。 |

## 调试与输出

| 目录 | 概述 |
|------|------|
| [`debug_print/`](debug_print/README.md) | `tla.print` 的标量打印，含格式化与混合核场景。 |
| [`print_tensor/`](print_tensor/README.md) | 以 `tla.print` 打印 tensor 在不同存储层级（GM / UB / L1 / L0C）的内容。 |

## 进阶算法算子

| 目录 | 概述 |
|------|------|
| [`advanced_impl/`](advanced_impl/README.md) | 进阶算法算子：完整算子的实现，含多阶段数据通路、缓冲分配与数值域映射。当前收录基数排序与 TopK。 |

## 复用已有实现

| 目录 | 概述 |
|------|------|
| [`extern_op/`](extern_op/README.md) | 用 `@tla.extern` 复用已有 Ascend C 算子，含单核、双核、多算子与自定义头文件场景。 |
