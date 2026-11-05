# CATLASS 样例设计文档

本文档汇总当前一些样例的设计思路和代码拆解，读者可按照个人兴趣查阅具体内容。

- [102_泛化Matmul工程](../../../../examples/102_dynamic_optimized_matmul/README.md) - 根据shape动态确定Tiling参数，并尝试选择最好的模板进行计算，尽力获取最优性能。
  - [CommonMatmul模板](../../../../examples/102_dynamic_optimized_matmul/docs/zh/CommonMatmul.md)
  - [MultiCoreSplitkMatmul多核切K模板](../../../../examples/102_dynamic_optimized_matmul/docs/zh/MultiCoreSplitkMatmul.md)
  - [StreamkMatmul模板](../../../../examples/102_dynamic_optimized_matmul/docs/zh/StreamkMatmul.md)
- [10_grouped_matmul_slice_m_per_token_dequant](../../../../examples/10_grouped_matmul_slice_m_per_token_dequant/10_grouped_matmul_slice_m_per_token_dequant.md) - 拆解模板库下的样例10，包含原型设计、样例实现、example组装、kernel实现方案。对希望了解"groupMatmul+后处理"类型的算子实现有指导价值。
- [19_mla](../../../../examples/19_mla/mla.md) - 拆解模板库下的样例19、亲和昇腾Atlas A2硬件的Flash-MLA算子的实现。
- [34_single_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/34_single_splitk_matmul.md) - 拆解模板库下的样例34单核切K矩阵乘样例，讲解算法实现和评估收益区间。
- [44_quant_matmul_full_loadA_tla](../../../../examples/44_quant_matmul_full_loadA_tla/44_quant_matmul_full_loadA_tla.md) - 拆解模板库下的样例44、quant量化下的A矩阵全载matmul样例的实现。
- [49_ascend950_flash_attention_infer](../../../../examples/49_ascend950_flash_attention_infer/flash_attention_infer.md) - 拆解模板库下的样例49，亲和昇腾950硬件的FlashAttention推理算子的实现。
- [52_quant_multi_core_splitk_matmul_tla](../../../../examples/52_quant_multi_core_splitk_matmul_tla/52_quant_multi_core_splitk_matmul_tla.md) - 拆解模板库下的样例52、quant量化下的多核切K样例的实现。
- [65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant](../../../../examples/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant.md) - 拆解模板库下的样例65、亲和昇腾950硬件的FP8分组Matmul+Swiglu+Quant样例的实现。
- [71_ascend950_fp8_mx_grouped_matmul_finalize_routing](../../../../examples/71_ascend950_fp8_mx_grouped_matmul_finalize_routing/71_ascend950_fp8_mx_grouped_matmul_finalize_routing.md) - 拆解模板库下的样例71、亲和昇腾950硬件的FP8分组Matmul+FinalizeRouting样例的实现。
- [75_symm](../../../../examples/75_symm/75_symm.md) - 拆解模板库下的样例75、symm对称矩阵乘样例的实现。
- [76_trmm](../../../../examples/76_trmm/76_trmm.md) - 拆解模板库下的样例76、trmm矩阵乘样例的实现。
- [77_planar_complex_matmul](../../../../examples/77_planar_complex_matmul/77_planar_complex_matmul.md) - 拆解模板库下的样例77、平面复数矩阵乘样例的实现。
