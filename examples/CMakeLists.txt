# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CATLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_custom_target(catlass_examples)

add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${CATLASS_ARCH}>)
add_compile_definitions(CATLASS_ARCH=${CATLASS_ARCH})

# arch specific compile options
if(CATLASS_ARCH STREQUAL "2201")
    add_compile_options(
        "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-mllvm -cce-aicore-record-overflow=true>"
        "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-mllvm -cce-aicore-dcci-insert-for-scalar=false>"
    )
endif()

# common compile options
add_compile_options(
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-ignored-attributes>
    # these options may be removed after CANN 8.5.0
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-value>
)

# these options may be removed after CANN 8.5.0
add_link_options($<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-command-line-argument>)

# compile options for debug features
if(DEFINED ENABLE_SIMULATOR AND ENABLE_SIMULATOR)
    if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
        message(FATAL_ERROR "SOC_VERSION is not defined.")
        return()
    endif()

    message("To run examples in simulator mode, execute the following commands in your terminal:")
    message("export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:\$LD_LIBRARY_PATH")
    message("export LD_PRELOAD=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib/libruntime_camodel.so:${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib/libnpu_drv_camodel.so")
endif()

# compile options for msSanitizer
if(DEFINED ENABLE_MSSANITIZER AND ENABLE_MSSANITIZER)
    add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASC>:-g --cce-enable-sanitizer>")
    add_link_options("SHELL:$<$<COMPILE_LANGUAGE:ASC>:--cce-enable-sanitizer --npu-arch=dav-${CATLASS_ARCH}>")
endif()

macro(catlass_example_add_executable NAME OPTYPE)
    set(EXAMPLE_DESTINATION bin)

    add_executable(${NAME} ${ARGN})

    target_compile_definitions(${NAME} PRIVATE CATLASS_EXAMPLE_NAME=${NAME})
    target_include_directories(${NAME} PRIVATE ${ASCEND_HOME_PATH}/include/aclnn)

    add_dependencies(catlass_examples ${NAME})
    install(TARGETS ${NAME} DESTINATION ${EXAMPLE_DESTINATION} COMPONENT ${NAME})
    install(TARGETS ${NAME} DESTINATION ${EXAMPLE_DESTINATION} COMPONENT catlass_examples)
endmacro()

include_directories(
    ${CATLASS_EXAMPLES_COMMON_SOURCE_DIR}
    ${CATLASS_INCLUDE_DIR}
)

link_directories(${ASCEND_HOME_PATH}/lib64)
link_libraries(
    # AscendC necessary libraries
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl
    # other necessary libraries
    tiling_api nnopbase platform
    # system libraries
    dl
)
# CANN 9.0.0.beta2 incompatible change
if(EXISTS ${ASCEND_HOME_PATH}/lib64/libascendalog.so)
    link_libraries(ascendalog)
endif()
if(EXISTS ${ASCEND_HOME_PATH}/lib64/libunified_dlog.so)
    link_libraries(unified_dlog)
endif()

if(NOT DEFINED ENABLE_SIMULATOR OR NOT ENABLE_SIMULATOR)
    link_directories(${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib)
    link_libraries(ascend_hal)
endif()

# regular examples
set(EXAMPLE_ATLASA2
    00_basic_matmul
    01_batched_matmul
    02_grouped_matmul_slice_m
    03_matmul_add
    04_padding_matmul
    05_grouped_matmul_slice_k
    06_optimized_matmul
    07_grouped_matmul_slice_m_per_token_dequant_moe
    08_grouped_matmul
    09_splitk_matmul
    10_grouped_matmul_slice_m_per_token_dequant
    11_grouped_matmul_slice_k_per_token_dequant
    12_quant_matmul
    13_basic_matmul_tla
    14_optimized_matmul_tla
    15_gemm
    16_group_gemm
    17_gemv_aiv
    18_gemv_aic
    19_mla
    20_matmul_bias
    21_basic_matmul_preload_zN
    22_padding_splitk_matmul
    23_flash_attention_infer
    24_conv_bias
    25_matmul_full_loadA
    26_matmul_relu
    27_matmul_gelu
    28_matmul_silu
    29_a2_fp8_e4m3_matmul
    30_w8a16_matmul
    31_small_matmul
    32_w4a8_matmul
    33_basic_conv2d
    34_single_core_splitk_matmul
    37_streamk_matmul
    38_w4a4_matmul_per_token_per_channel_dequant
    40_flash_attention_infer_tla
    41_sparse_matmul_tla
    42_quant_optimized_matmul_tla
    44_quant_matmul_full_loadA_tla
    45_strided_batched_matmul_tla
    52_quant_multi_core_splitk_matmul_tla
    75_symm
    76_trmm
    77_planar_complex_matmul
    78_x_attention
    102_dynamic_optimized_matmul
    103_dynamic_optimized_quant_matmul_per_token_basic
)

set(EXAMPLE_ASCEND950
    43_ascend950_basic_matmul
    46_ascend950_matmul_fixpipe_opti
    47_ascend950_grouped_matmul_slice_m_per_token_dequant
    48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant
    49_ascend950_flash_attention_infer
    50_ascend950_basic_matmul_gemv
    51_ascend950_quant_matmul_per_group_per_block_tla
    53_ascend950_fp8_mx_matmul
    54_ascend950_fp4_mx_matmul
    55_ascend950_mx_grouped_matmul_slice_m
    56_ascend950_basic_conv2d_tla
    57_ascend950_matmul_full_dequant
    58_ascend950_mx_batch_matmul
    59_ascend950_a8w4_mx_matmul
    60_ascend950_grouped_matmul_slice_m
    61_ascend950_svd_quant_matmul
    62_ascend950_broadcast_matmul_perblock_quant
    63_ascend950_dual_level_quant_mx_batch_matmul
    64_ascend950_matmul_evg
    65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant
    66_ascend950_streamk_matmul
    67_ascend950_batched_matmul
    68_ascend950_multi_core_splitk_matmul
    69_ascend950_tail_multi_core_splitk_matmul
    70_ascend950_flash_attention_chunk_prefill
    71_ascend950_fp8_mx_grouped_matmul_finalize_routing
    73_ascend950_matmul_full_loadA
    74_ascend950_weight_quant_a8w4_grouped_mx_matmul
    80_ascend950_grouped_matmul_slice_m_gelu
    82_ascend950_basic_syrk
    83_ascend950_hstu_infer
    84_ascend950_syrk
    88_ascend950_bmm_broadcast
)

if(CATLASS_ARCH STREQUAL "2201")
    set(EXAMPLE_LIST ${EXAMPLE_ATLASA2})
elseif(CATLASS_ARCH STREQUAL "3510")
    set(EXAMPLE_LIST ${EXAMPLE_ASCEND950})
endif()

foreach(EXAMPLE ${EXAMPLE_LIST})
    add_subdirectory(${EXAMPLE})
endforeach()

# advanded examples
if(CATLASS_ARCH STREQUAL "2201")
    add_subdirectory(shared_lib)
    add_subdirectory(python_extension)
    if(DEFINED SOC_VERSION AND NOT SOC_VERSION STREQUAL "")
        add_subdirectory(advanced/basic_matmul_aclnn)
    else()
        message(WARNING "SOC_VERSION is not defined. basic_matmul_aclnn will not be compiled.")
    endif()
endif()
