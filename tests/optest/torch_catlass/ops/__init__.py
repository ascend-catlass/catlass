# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from .a2_fp8_e4m3_matmul import a2_fp8_e4m3_matmul  # example 29
from .ascend950_basic_conv2d_tla import ascend950_basic_conv2d_tla  # example 56
from .ascend950_basic_matmul import ascend950_basic_matmul  # example 43
from .ascend950_basic_matmul_gemv import ascend950_basic_matmul_gemv  # example 50
from .ascend950_batched_matmul import ascend950_batched_matmul  # example 67
from .ascend950_flash_attention_chunk_prefill import (
    ascend950_flash_attention_chunk_prefill,  # example 70
)
from .ascend950_rain_fusion_attention import (
    ascend950_rain_fusion_attention,  # example 76
)
from .ascend950_flash_attention_infer import ascend950_flash_attention_infer  # example 49
from .ascend950_hstu_infer import ascend950_hstu_infer  # example 83
from .ascend950_fp8_mx_flash_attention_infer import (
    ascend950_fp8_mx_flash_attention_infer,  # example 65
)
from .ascend950_fp8_mx_grouped_matmul_finalize_routing import (  # example 71
    ascend950_fp8_mx_grouped_matmul_finalize_routing,
)
from .ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter import (  # example 71 no_deter
    ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter,
)
from .ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant import (
    ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant,  # example 65
)
from .ascend950_grouped_matmul_slice_m import ascend950_grouped_matmul_slice_m  # example 60
from .ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant import (
    ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant,  # example 48
)
from .ascend950_grouped_matmul_slice_m_per_token_dequant import (
    ascend950_grouped_matmul_slice_m_per_token_dequant,  # example 47
)
from .ascend950_matmul_evg import EvgPostprocessMode, ascend950_matmul_evg  # example 64
from .ascend950_matmul_fixpipe_opti import ascend950_matmul_fixpipe_opti  # example 46
from .ascend950_matmul_full_dequant import ascend950_matmul_full_dequant  # example 57
from .ascend950_matmul_full_loadA import ascend950_matmul_full_loadA  # example 73
from .ascend950_dual_matmul_silu_mul import ascend950_dual_matmul_silu_mul  # example 78
from .ascend950_mx_grouped_matmul_slice_m import ascend950_mx_grouped_matmul_slice_m  # example 55
from .ascend950_mx_matmul import (  # example 53, 54, 58, 59, 63
    ascend950_a8w4_mx_matmul,
    ascend950_dual_level_quant_mx_batch_matmul,
    ascend950_fp4_mx_matmul_aswt,
    ascend950_fp8_mx_batch_matmul,
    ascend950_fp8_mx_matmul_aswt,
)
from .ascend950_a8w4_grouped_mx_matmul import ascend950_a8w4_grouped_mx_matmul  # example 74
from .svd_quant_matmul import ascend950_svd_quant_matmul  # example 61
from .broadcast_matmul_perblock_quant import broadcast_matmul_perblock_quant  # example 62
from .ascend950_matmul_evg import EvgPostprocessMode, ascend950_matmul_evg  # example 64
from .ascend950_matmul_full_loadA import ascend950_matmul_full_loadA  # example 73
from .matmul_gather_scatter import ascend950_matmul_gather_scatter  # example 79
from .ascend950_batched_matmul import ascend950_batched_matmul  # example 67
from .ascend950_basic_matmul_gemv import ascend950_basic_matmul_gemv  # example 50
from .ascend950_quant_matmul_per_group_per_block_tla import ascend950_quant_matmul_per_group_per_block_tla  # example 51
from .ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant import (
    ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant,  # example 65
)
from .ascend950_streamk_matmul import ascend950_streamk_matmul  # example 66
from .basic_conv2d import basic_conv2d  # example 33
from .basic_matmul import basic_matmul  # example 00
from .basic_matmul_preload_zN import basic_matmul_preload_zN  # example 21
from .basic_matmul_tla import basic_matmul_tla  # example 13
from .batched_matmul import batched_matmul  # example 01
from .broadcast_matmul_perblock_quant import broadcast_matmul_perblock_quant  # example 62
from .conv_bias import conv_bias  # example 24
from .flash_attention_infer import flash_attention_infer  # example 23
from .flash_attention_infer_tla import flash_attention_infer_tla  # example 40
from .fag_tla import fag_tla  # example 87
from .gemm import gemm  # example 15
from .gemv_aic import gemv_aic  # example 18
from .gemv_aiv import gemv_aiv  # example 17
from .group_gemm import group_gemm  # example 16
from .grouped_matmul import grouped_matmul  # example 08
from .grouped_matmul_slice_k import grouped_matmul_slice_k  # example 05
from .grouped_matmul_slice_m import grouped_matmul_slice_m  # example 02
from .grouped_matmul_slice_m_per_token_dequant import (
    grouped_matmul_slice_k_per_token_dequant,  # example 11
    grouped_matmul_slice_m_per_token_dequant,  # example 07
    grouped_matmul_slice_m_per_token_dequant_multistage,  # example 10
)
from .matmul_add import matmul_add  # example 03
from .matmul_bias import matmul_bias  # example 20
from .matmul_full_loadA import matmul_full_loadA  # example 25
from .matmul_gelu import matmul_gelu  # example 27
from .matmul_relu import matmul_relu  # example 26
from .matmul_silu import matmul_silu  # example 28
from .mla import mla  # example 19
from .multi_core_splitk_matmul import ascend950_multi_core_splitk_matmul  # example 68
from .optimized_matmul import optimized_matmul  # example 06
from .optimized_matmul_tla import optimized_matmul_tla  # example 14
from .padding_matmul import padding_matmul  # example 04
from .padding_splitk_matmul import padding_splitk_matmul  # example 22
from .quant_matmul import quant_matmul  # example 12
from .quant_matmul_full_loadA_tla import quant_matmul_full_loadA_tla  # example 44
from .quant_multi_core_splitk_matmul_tla import quant_multi_core_splitk_matmul_tla  # example 52
from .quant_optimized_matmul_tla import quant_optimized_matmul_tla  # example 42
from .single_core_splitk_matmul import single_core_splitk_matmul  # example 34
from .small_matmul import small_matmul  # example 31
from .sparse_matmul_tla import sparse_matmul_tla  # example 41
from .splitk_matmul import splitk_matmul  # example 09
from .streamk_matmul import streamk_matmul  # example 37
from .strided_batched_matmul_tla import strided_batched_matmul_tla  # example 45
from .svd_quant_matmul import ascend950_svd_quant_matmul  # example 61
from .symm import symm  # example 75
from .trmm import trmm  # example 76
from .planar_complex_matmul import planar_complex_matmul  # example 77
from .tail_multi_core_splitk_matmul import ascend950_tail_multi_core_splitk_matmul  # example 69
from .w4a4_matmul_per_token_per_channel_dequant import (
    w4a4_matmul_per_token_per_channel_dequant,  # example 38
)
from .w4a8_matmul import w4a8_matmul  # example 32
from .w8a16_matmul import w8a16_matmul  # example 30
from .grouped_matmul_slice_m_gelu import grouped_matmul_slice_m_gelu  # example 80
from .matrix_inverse import matrix_inverse  # example 78
from .x_attention import x_attention  # example 78
from .ascend950_basic_syrk import ascend950_basic_syrk  # example 82
from .ascend950_syrk import ascend950_syrk  # 84

from .ascend950_quant_matmul import (  # example 79, 81
    ascend950_fp4_mx_quant_matmul,
    ascend950_fp8_e4m3_quant_matmul,
)
__all__ = [
    "basic_matmul",  # example 00
    "batched_matmul",  # example 01
    "grouped_matmul_slice_m",  # example 02
    "matmul_add",  # example 03
    "padding_matmul",  # example 04
    "grouped_matmul_slice_k",  # example 05
    "optimized_matmul",  # example 06
    "grouped_matmul_slice_m_per_token_dequant",  # example 07
    "grouped_matmul",  # example 08
    "splitk_matmul",  # example 09
    "grouped_matmul_slice_m_per_token_dequant_multistage",  # example 10
    "grouped_matmul_slice_k_per_token_dequant",  # example 11
    "quant_matmul",  # example 12
    "basic_matmul_tla",  # example 13
    "optimized_matmul_tla",  # example 14
    "gemm",  # example 15
    "group_gemm",  # example 16
    "gemv_aiv",  # example 17
    "gemv_aic",  # example 18
    "mla",  # example 19
    "matmul_bias",  # example 20
    "basic_matmul_preload_zN",  # example 21
    "padding_splitk_matmul",  # example 22
    "flash_attention_infer",  # example 23
    "conv_bias",  # example 24
    "flash_attention_infer_tla",  # example 40
    "fag_tla",  # example 87
    "ascend950_flash_attention_infer",  # example 49
    "ascend950_hstu_infer",  # example 83
    "ascend950_fp8_mx_flash_attention_infer",  # example 65
    "matmul_full_loadA",  # example 25
    "matmul_relu",  # example 26
    "matmul_gelu",  # example 27
    "matmul_silu",  # example 28
    "a2_fp8_e4m3_matmul",  # example 29
    "w8a16_matmul",  # example 30
    "small_matmul",  # example 31
    "w4a8_matmul",  # example 32
    "basic_conv2d",  # example 33
    "single_core_splitk_matmul",  # example 34
    "streamk_matmul",  # example 37
    "w4a4_matmul_per_token_per_channel_dequant",  # example 38
    "sparse_matmul_tla",  # example 41
    "quant_optimized_matmul_tla",  # example 42
    "ascend950_basic_matmul",  # example 43
    "quant_matmul_full_loadA_tla",  # example 44
    "strided_batched_matmul_tla",  # example 45
    "ascend950_matmul_fixpipe_opti",  # example 46
    "ascend950_grouped_matmul_slice_m_per_token_dequant",  # example 47
    "ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant",  # example 48
    "ascend950_basic_matmul_gemv",  # example 50
    "ascend950_quant_matmul_per_group_per_block_tla",  # example 51
    "ascend950_matmul_full_dequant",  # example 57
    "quant_multi_core_splitk_matmul_tla",  # example 52
    "ascend950_fp8_mx_matmul_aswt",  # example 53
    "ascend950_fp4_mx_matmul_aswt",  # example 54
    "ascend950_mx_grouped_matmul_slice_m",  # example 55
    "ascend950_fp8_mx_batch_matmul",      # example 58
    "ascend950_a8w4_mx_matmul",           # example 59
    "ascend950_a8w4_grouped_mx_matmul",   # example 74
    "ascend950_grouped_matmul_slice_m",   # example 60
    "broadcast_matmul_perblock_quant",    # example 62
    "ascend950_dual_level_quant_mx_batch_matmul", # example 63
    "ascend950_svd_quant_matmul",         # example 61
    "ascend950_matmul_evg",               # example 64
    "EvgPostprocessMode",                 # example 64
    "ascend950_batched_matmul",           # example 67
    "ascend950_matmul_fixpipe_opti",      # example 46
    "ascend950_basic_matmul_gemv",        # example 50
    "ascend950_quant_matmul_per_group_per_block_tla",  # example 51
    "ascend950_matmul_full_dequant",  # example 57
    "ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant",  # example 65
    "ascend950_streamk_matmul",  # example 66
    "ascend950_batched_matmul",  # example 67
    "ascend950_multi_core_splitk_matmul",  # example 68
    "ascend950_tail_multi_core_splitk_matmul",  # example 69
    "ascend950_flash_attention_chunk_prefill",  # example 70
    "ascend950_rain_fusion_attention",  # example 76
    "ascend950_fp8_mx_grouped_matmul_finalize_routing",  # example 71
    "ascend950_fp4_mx_quant_matmul",  # experimental/matmul
    "ascend950_fp8_e4m3_quant_matmul",  # experimental/matmul
    "ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter",  # example 71 no_deter
    "ascend950_basic_conv2d_tla",  # example 56
    "ascend950_matmul_full_loadA",  # example 73
    "ascend950_dual_matmul_silu_mul",  # example 78
    "symm",  # example 75
    "trmm",  # example 76
    "planar_complex_matmul",  # example 77
    "ascend950_matmul_gather_scatter",  # example 79
    "grouped_matmul_slice_m_gelu",  # example 80
    "matrix_inverse",  # example 78
    "x_attention",  # example 78
    "ascend950_basic_syrk",  # example 82
    "ascend950_syrk",  # 84
]
