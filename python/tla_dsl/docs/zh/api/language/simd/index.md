---
nav_order: 5
---

<!--
Manually maintained Chinese Language reference.
Generated English source of truth: docs/en/api/language/simd/index.md
(from python/tla_dsl/tools/generate_kernel_api_reference.py + English docstrings).
Translate/update this file by hand when the English reference changes.
Do not regenerate this file from a glossary.
-->

# SIMD 向量计算

[语言参考](../index.md) | [API 速查](../../quick_reference.md)

默认向量执行模式，以 `tla.vec.func(mode="simd")` 进入。主体在整寄存器上运算：数据
先搬到 UB、按 tile 视图、`load` 成 `VectorSSA`，各 op 在可选 lane mask 下覆盖整寄存器。
适合规则、稠密计算；下列接口均属此路径。

部分接口在 `tla.vec.func(mode="simt")` 内还有逐线程形态，各条目约束说明中会写明；
该模式见 [SIMT 向量计算](../simt.md#simt-向量计算)。

## 子页面

| 章节 | 页面 |
| --- | --- |
| 5.1 | [Mask 计算](mask.md) |
| 5.2 | [基础算术](arithmetic.md) |
| 5.3 | [逻辑计算](logical.md) |
| 5.4 | [比较与选择](compare_select.md) |
| 5.5 | [数据填充](fill.md) |
| 5.6 | [离散与聚合](discrete_aggregate.md) |
| 5.7 | [数据重排](rearrange.md) |
| 5.8 | [数据压缩](compress.md) |
| 5.9 | [类型转换](type_conversion.md) |
| 5.10 | [直方图计算](histogram.md) |
