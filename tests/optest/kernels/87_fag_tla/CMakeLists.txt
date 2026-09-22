# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# fag_tla is a single translation unit: fag_tla.cpp includes the FAG kernel
# (fag_tla_kernel.cpp) followed by the host tiling (fag_tla_tiling.cpp). The
# kernel must come first so the catlass epilogue headers pull in kernel_operator.h
# and thereby define SoftMaxTiling before the tiling code references it.
set(_FAG_TLA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/fag_tla.cpp
)
set_source_files_properties(${_FAG_TLA_SOURCES} PROPERTIES LANGUAGE ASC)

add_kernel(NAME fag_tla
    NPU_ARCH_LIST 2201
    KERNEL_TYPE prebuilt
    ${_FAG_TLA_SOURCES}
)

foreach(_SRC ${_FAG_TLA_SOURCES})
    get_filename_component(_SRC_NAME ${_SRC} NAME)
    foreach(_ARCH 2201)
        foreach(_MS_SUFFIX "" "_ms")
            set(_LIB_NAME "catlass_kernel_${_ARCH}_fag_tla${_MS_SUFFIX}")
            if(TARGET ${_LIB_NAME})
                target_compile_definitions(${_LIB_NAME} PRIVATE TILING_KEY_VAR)
                target_include_directories(${_LIB_NAME} PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    ${CATLASS_DIR}/experimental/attention/fag_tla
                )
            endif()
        endforeach()
    endforeach()
endforeach()