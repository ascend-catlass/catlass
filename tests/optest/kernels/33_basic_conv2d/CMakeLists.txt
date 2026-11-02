# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_BASIC_CONV2D_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/basic_conv2d.cpp
)
set_source_files_properties(${_BASIC_CONV2D_SOURCES} PROPERTIES LANGUAGE ASC)

add_kernel(NAME basic_conv2d
    NPU_ARCH_LIST 2201
    KERNEL_TYPE prebuilt
    ${_BASIC_CONV2D_SOURCES}
)

foreach(_SRC ${_BASIC_CONV2D_SOURCES})
    get_filename_component(_SRC_NAME ${_SRC} NAME)
    foreach(_ARCH 2201)
        foreach(_MS_SUFFIX "" "_ms")
            set(_LIB_NAME "catlass_kernel_${_ARCH}_basic_conv2d${_MS_SUFFIX}")
            if(TARGET ${_LIB_NAME})
                target_compile_definitions(${_LIB_NAME} PRIVATE TILING_KEY_VAR)
            endif()
        endforeach()
    endforeach()
endforeach()

set(_CONV2D_BIAS_SILU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/conv2d_bias_silu.cpp
)
set_source_files_properties(${_CONV2D_BIAS_SILU_SOURCES} PROPERTIES LANGUAGE ASC)

add_kernel(NAME conv2d_bias_silu
    NPU_ARCH_LIST 2201
    KERNEL_TYPE prebuilt
    ${_CONV2D_BIAS_SILU_SOURCES}
)

foreach(_SRC ${_CONV2D_BIAS_SILU_SOURCES})
    get_filename_component(_SRC_NAME ${_SRC} NAME)
    foreach(_ARCH 2201)
        foreach(_MS_SUFFIX "" "_ms")
            set(_LIB_NAME "catlass_kernel_${_ARCH}_conv2d_bias_silu${_MS_SUFFIX}")
            if(TARGET ${_LIB_NAME})
                target_compile_definitions(${_LIB_NAME} PRIVATE TILING_KEY_VAR)
            endif()
        endforeach()
    endforeach()
endforeach()
