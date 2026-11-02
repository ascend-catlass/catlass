<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/index.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# CATLASS Python 语言参考

先通过 [API 速查](../quick_reference.md) 按任务与执行上下文选择接口。
本文档是 **CATLASS Python 语言参考**（通常以 `import catlass.tla as tla` 导入），
覆盖值与数据类型、Layout 与 Tensor、数据搬运、计算、同步、资源与调试打印。
编译、启动与 Host tensor 绑定见 [程序生命周期参考](../runtime/index.md)；
DLPack 接入教程见 [Host Tensor 接入](../../kernel_development/core_concepts/tensor_binding.md)。

接口说明与调用示例来自各 op 源码 docstring。除非条目明确说明可在 Host 使用，
所有接口均须在 `@tla.kernel` 装饰的 kernel 函数体内调用。例如
`tla.arch.get_capacity_in_bytes` 也可在 Host 调用。

向量单元有两种执行模式，下文分章说明。**SIMD 向量计算**是默认路径：整寄存器覆盖
已搬入 UB 的数据，适合规则稠密计算。**SIMT 向量计算**是一线程一元素、按索引寻址，
适合不规则或难以靠 tiling 摊平的工作。部分标量接口随所在模式行为不同——元素访问与
标量算术 / 比较 / 选择 / 转换在 SIMT 区域内均生成逐线程 op，各条目约束说明中会写明。

## 分类页面

| 章节 | 页面 |
| --- | --- |
| 1 | [Layout 与 Tensor](layout_tensor.md) |
| 2 | [值与数据类型](values_types.md) |
| 3 | [数据搬运](data_movement.md) |
| 4 | [矩阵运算](matrix.md) |
| 5 | [SIMD 向量计算](simd/index.md) |
| 5.1 | [Mask 计算](simd/mask.md) |
| 5.2 | [基础算术](simd/arithmetic.md) |
| 5.3 | [逻辑计算](simd/logical.md) |
| 5.4 | [比较与选择](simd/compare_select.md) |
| 5.5 | [数据填充](simd/fill.md) |
| 5.6 | [离散与聚合](simd/discrete_aggregate.md) |
| 5.7 | [数据重排](simd/rearrange.md) |
| 5.8 | [数据压缩](simd/compress.md) |
| 5.9 | [类型转换](simd/type_conversion.md) |
| 5.10 | [直方图计算](simd/histogram.md) |
| 6 | [SIMT 向量计算](simt.md) |
| 7 | [同步控制](sync.md) |
| 8 | [系统变量访问](system.md) |
| 9 | [资源管理](resources.md) |
| 10 | [调试与打印](debug.md) |
| 11 | [作用域与控制流](scopes.md) |
| 12 | [实验性接口](experimental.md) |

---
