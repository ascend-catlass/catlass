# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CATLASS_ALL_ARCHS "")
include_directories(${CATLASS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/jit_verify_template.cmake)

set(_JIT_COMPILER_TARGET catlass_kernel_jit_compiler)
add_library(${_JIT_COMPILER_TARGET} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/jit_compiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/jit_macro_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/jit_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/jit_logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit/workspace_alloc.cpp)
target_include_directories(${_JIT_COMPILER_TARGET} PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${CATLASS_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}
)
target_link_directories(${_JIT_COMPILER_TARGET} PRIVATE
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(${_JIT_COMPILER_TARGET} PRIVATE
    catlass_kernel_utils
    ascendc_runtime
    runtime
    profapi
    mmpa
    ascend_dump
    c_sec
    error_manager
    ascendcl
    tiling_api
    nnopbase
    platform
    dl
)
install(TARGETS ${_JIT_COMPILER_TARGET}
    DESTINATION lib/jit)

if(DISABLE_ASCEND950)
    target_compile_definitions(${_JIT_COMPILER_TARGET} PRIVATE ENABLE_ASCEND950=0)
endif()

set(_JIT_KERNEL_ENTRIES "")
set(_JIT_KERNEL_TEMPLATES "")


macro(add_kernel)
    set(_K_OPTIONS "SKIP_MS_BUILD")
    set(_K_ONE_VALUE "NAME;KERNEL_TYPE")
    set(_K_MULTI_VALUE "NPU_ARCH_LIST;TEMPLATE")
    cmake_parse_arguments(_K "${_K_OPTIONS}" "${_K_ONE_VALUE}" "${_K_MULTI_VALUE}" ${ARGN})

    # 参数校验
    if(NOT DEFINED _K_NAME)
        message(FATAL_ERROR "add_kernel: NAME is required")
    endif()
    if(NOT DEFINED _K_NPU_ARCH_LIST)
        message(FATAL_ERROR "add_kernel: NPU_ARCH_LIST is required")
    endif()
    if(NOT DEFINED _K_KERNEL_TYPE)
        message(FATAL_ERROR "add_kernel: KERNEL_TYPE is required (jit or prebuilt)")
    endif()

    if("${_K_KERNEL_TYPE}" STREQUAL "jit")
        # -------------------------------------------------------
        # JIT 模式 — 收集源文件到统一库
        # -------------------------------------------------------
        set(_KERNEL_ENTRY "${_K_UNPARSED_ARGUMENTS}")

        # 收集 entry 源文件到列表，构建时统一编译
        list(APPEND _JIT_KERNEL_ENTRIES "${_KERNEL_ENTRY}")
        set(_JIT_KERNEL_ENTRIES ${_JIT_KERNEL_ENTRIES} PARENT_SCOPE)

        if(_K_TEMPLATE)
            # 收集所有 template 源文件，用于运行时分发
            foreach(_TEMPLATE_FILE ${_K_TEMPLATE})
                list(APPEND _JIT_KERNEL_TEMPLATES "${_TEMPLATE_FILE}")
            endforeach()
            set(_JIT_KERNEL_TEMPLATES ${_JIT_KERNEL_TEMPLATES} PARENT_SCOPE)

            set(_JIT_INCLUDE_DIRS
                "${CATLASS_INCLUDE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/.."
                "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_BINARY_DIR}"
            )

            # 仅对 CATLASS_ARCH_LIST 中启用的架构做模板验证
            set(_K_LEGAL_ARCH "")
            foreach(_ARCH ${_K_NPU_ARCH_LIST})
                list(FIND CATLASS_ARCH_LIST "${_ARCH}" _ARCH_INDEX)
                if(NOT _ARCH_INDEX EQUAL -1)
                    list(APPEND _K_LEGAL_ARCH "${_ARCH}")
                endif()
            endforeach()

            if(_K_LEGAL_ARCH)
                # 为每个 template 生成独立的 verify 目标（target 名按文件 stem 区分）
                foreach(_TEMPLATE_FILE ${_K_TEMPLATE})
                    get_filename_component(_TEMPLATE_STEM ${_TEMPLATE_FILE} NAME_WE)
                    jit_verify_template(
                        NAME "${_K_NAME}_${_TEMPLATE_STEM}"
                        TEMPLATE "${_TEMPLATE_FILE}"
                        NPU_ARCH_LIST ${_K_LEGAL_ARCH}
                        INCLUDE_DIRS ${_JIT_INCLUDE_DIRS}
                    )
                endforeach()
            else()
                message(STATUS "add_kernel: skip JIT template verification for '${_K_NAME}' "
                    "(declared archs: ${_K_NPU_ARCH_LIST}; CATLASS_ARCH_LIST: ${CATLASS_ARCH_LIST})")
            endif()
        endif()

    elseif("${_K_KERNEL_TYPE}" STREQUAL "prebuilt")
        # prebuilt kernel 默认为每个 arch 编译两份：普通版本和 _ms (sanitizer) 版本
        foreach(_ARCH ${_K_NPU_ARCH_LIST})
            list(FIND CATLASS_ARCH_LIST "${_ARCH}" _ARCH_INDEX)
            if(_ARCH_INDEX EQUAL -1)
                continue()
            endif()

            list(APPEND CATLASS_ALL_ARCHS ${_ARCH})

            set(_INSTALL_TARGETS "")
            foreach(_MS_SUFFIX IN ITEMS "" "_ms")
                if(_MS_SUFFIX STREQUAL "_ms" AND _K_SKIP_MS_BUILD)
                    continue()
                endif()
                set(_LIB_NAME "catlass_kernel_${_ARCH}_${_K_NAME}${_MS_SUFFIX}")
                list(APPEND _INSTALL_TARGETS ${_LIB_NAME})
                add_library(${_LIB_NAME} SHARED ${_K_UNPARSED_ARGUMENTS})

                target_compile_definitions(${_LIB_NAME} PRIVATE
                    CATLASS_ARCH=${_ARCH}
                    KERNEL_NAME=${_K_NAME}_entry)

                # 通用 ASC 编译选项
                target_compile_options(${_LIB_NAME} PRIVATE
                    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${_ARCH}>
                    $<$<COMPILE_LANGUAGE:ASC>:-Wno-macro-redefined>
                    $<$<COMPILE_LANGUAGE:ASC>:-Wno-ignored-attributes>
                    $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-value>
                )

                # _ms 版本附加 sanitizer 标志 (ARCH 3510不需要静态插桩)
                if(_MS_SUFFIX STREQUAL "_ms" AND NOT _ARCH EQUAL 3510)
                    target_compile_options(${_LIB_NAME} PRIVATE
                        "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-g --cce-enable-sanitizer>"
                    )
                    target_link_options(${_LIB_NAME} PRIVATE
                        "SHELL:$<$<COMPILE_LANGUAGE:ASC>:--cce-enable-sanitizer --npu-arch=dav-${_ARCH}>"
                    )
                    message(STATUS "  [ms] ${_LIB_NAME} will be built with msSanitizer")
                endif()

                target_link_options(${_LIB_NAME} PRIVATE
                    $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-command-line-argument>
                )

                target_include_directories(${_LIB_NAME} PRIVATE
                    ${ASCEND_HOME_PATH}/include
                    ${CATLASS_DIR}/include
                    ${CMAKE_BINARY_DIR}
                )

                target_link_directories(${_LIB_NAME} PRIVATE
                    ${ASCEND_HOME_PATH}/lib64
                )

                target_link_libraries(${_LIB_NAME} PRIVATE
                    ascendc_runtime
                    runtime
                    profapi
                    mmpa
                    ascend_dump
                    c_sec
                    error_manager
                    ascendcl
                    tiling_api
                    nnopbase
                    platform
                    dl
                )

                if(EXISTS ${ASCEND_HOME_PATH}/lib64/libascendalog.so)
                    target_link_libraries(${_LIB_NAME} PRIVATE ascendalog)
                endif()
                if(EXISTS ${ASCEND_HOME_PATH}/lib64/libunified_dlog.so)
                    target_link_libraries(${_LIB_NAME} PRIVATE unified_dlog)
                endif()
            endforeach()

            install(TARGETS ${_INSTALL_TARGETS}
                DESTINATION lib/${_ARCH}
            )
            unset(_INSTALL_TARGETS)
        endforeach()

    else()
        message(FATAL_ERROR
            "add_kernel: unknown KERNEL_TYPE '${_K_KERNEL_TYPE}'. "
            "Expected 'jit' or 'prebuilt'.")
    endif()
endmacro()

add_subdirectory(00_basic_matmul)
add_subdirectory(01_batched_matmul)
add_subdirectory(02_grouped_matmul_slice_m)
add_subdirectory(03_matmul_add)
add_subdirectory(04_padding_matmul)
add_subdirectory(05_grouped_matmul_slice_k)
add_subdirectory(06_optimized_matmul)
add_subdirectory(07_grouped_matmul_slice_m_per_token_dequant_moe)
add_subdirectory(08_grouped_matmul)
add_subdirectory(09_splitk_matmul)
add_subdirectory(10_grouped_matmul_slice_m_per_token_dequant)
add_subdirectory(11_grouped_matmul_slice_k_per_token_dequant)
add_subdirectory(12_quant_matmul)
add_subdirectory(13_basic_matmul_tla)
add_subdirectory(14_optimized_matmul_tla)
add_subdirectory(15_gemm)
add_subdirectory(16_group_gemm)
add_subdirectory(17_gemv_aiv)
add_subdirectory(18_gemv_aic)
add_subdirectory(19_mla)
add_subdirectory(20_matmul_bias)
add_subdirectory(21_basic_matmul_preload_zN)
add_subdirectory(22_padding_splitk_matmul)
add_subdirectory(23_flash_attention_infer)
add_subdirectory(33_basic_conv2d)
add_subdirectory(24_conv_bias)
add_subdirectory(40_flash_attention_infer_tla)
add_subdirectory(49_ascend950_flash_attention_infer)
add_subdirectory(25_matmul_full_loadA)
add_subdirectory(26_matmul_relu)
add_subdirectory(27_matmul_gelu)
add_subdirectory(28_matmul_silu)
add_subdirectory(31_small_matmul)
add_subdirectory(34_single_core_splitk_matmul)
add_subdirectory(37_streamk_matmul)
add_subdirectory(38_w4a4_matmul_per_token_per_channel_dequant)
add_subdirectory(42_quant_optimized_matmul_tla)
add_subdirectory(43_ascend950_basic_matmul)
add_subdirectory(44_quant_matmul_full_loadA_tla)
add_subdirectory(46_ascend950_matmul_fixpipe_opti)
add_subdirectory(29_a2_fp8_e4m3_matmul)
add_subdirectory(30_w8a16_matmul)
add_subdirectory(32_w4a8_matmul)
add_subdirectory(41_sparse_matmul_tla)
add_subdirectory(45_strided_batched_matmul_tla)
add_subdirectory(67_ascend950_batched_matmul)
add_subdirectory(50_ascend950_basic_matmul_gemv)
add_subdirectory(51_ascend950_quant_matmul_per_group_per_block_tla)
add_subdirectory(52_quant_multi_core_splitk_matmul_tla)
add_subdirectory(47_ascend950_grouped_matmul_slice_m_per_token_dequant)
add_subdirectory(48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant)
add_subdirectory(53_ascend950_fp8_mx_matmul)
add_subdirectory(54_ascend950_fp4_mx_matmul)
add_subdirectory(55_ascend950_mx_grouped_matmul_slice_m)
add_subdirectory(57_ascend950_matmul_full_dequant)
add_subdirectory(58_ascend950_fp8_mx_batch_matmul)
add_subdirectory(59_ascend950_a8w4_mx_matmul)
add_subdirectory(60_ascend950_grouped_matmul_slice_m)
add_subdirectory(61_ascend950_svd_quant_matmul)
add_subdirectory(62_ascend950_broadcast_matmul_perblock_quant)
add_subdirectory(63_ascend950_dual_level_quant_mx_batch_matmul)
add_subdirectory(64_ascend950_matmul_evg)
add_subdirectory(65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant)
add_subdirectory(66_ascend950_streamk_matmul)
add_subdirectory(68_ascend950_multi_core_splitk_matmul)
add_subdirectory(69_ascend950_tail_multi_core_splitk_matmul)
add_subdirectory(70_ascend950_flash_attention_chunk_prefill)
add_subdirectory(71_ascend950_fp8_mx_grouped_matmul_finalize_routing)
add_subdirectory(73_ascend950_matmul_full_loadA)
add_subdirectory(78_ascend950_dual_matmul_silu_mul)
add_subdirectory(85_ascend950_fp4_mx_quant_matmul)
add_subdirectory(86_ascend950_fp8_e4m3_quant_matmul)
add_subdirectory(74_ascend950_weight_quant_a8w4_grouped_mx_matmul)
add_subdirectory(56_ascend950_basic_conv2d_tla)
add_subdirectory(72_ascend950_fp8_mx_flash_attention_infer)
add_subdirectory(75_symm)
add_subdirectory(76_trmm)
add_subdirectory(77_planar_complex_matmul)
add_subdirectory(78_x_attention)
add_subdirectory(78_matrix_inverse)
add_subdirectory(79_ascend950_matmul_gather_scatter)
add_subdirectory(80_grouped_matmul_slice_m_gelu)
add_subdirectory(81_ascend950_rain_fusion_attention)
add_subdirectory(82_ascend950_basic_syrk)
add_subdirectory(83_ascend950_hstu_infer)
add_subdirectory(84_ascend950_syrk)

if(_JIT_KERNEL_ENTRIES)
    set(_JIT_TARGET catlass_kernel_jit)
    add_library(${_JIT_TARGET} SHARED ${_JIT_KERNEL_ENTRIES})

    target_include_directories(${_JIT_TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/include
        ${CATLASS_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        ${CMAKE_BINARY_DIR}
    )
    target_link_directories(${_JIT_TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/lib64
    )
    target_link_libraries(${_JIT_TARGET} PRIVATE
        ${_JIT_COMPILER_TARGET}
        catlass_kernel_utils
        dl
    )

    get_property(_JIT_VERIFY_TARGETS GLOBAL PROPERTY _JIT_VERIFY_TARGETS)
    foreach(_TGT ${_JIT_VERIFY_TARGETS})
        add_dependencies(${_JIT_TARGET} ${_TGT})
    endforeach()

    install(TARGETS ${_JIT_TARGET}
        DESTINATION lib/jit)
endif()

# 安装 JIT 模板
foreach(_TEMPLATE ${_JIT_KERNEL_TEMPLATES})
    install(FILES ${_TEMPLATE}
        DESTINATION jit/templates)
endforeach()
