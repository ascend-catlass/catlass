# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

# ---------------------------------------------------------------------------
# jit_verify_template.cmake
#
# Build-time verification that JIT templates compile AND link with bisheng for
# each target NPU architecture. Creates SHARED libraries per template+arch so
# the build tool (Ninja/Make) naturally parallelizes all checks via -j.
#
# The verification intentionally mirrors the runtime JIT invocation, 
# `KERNEL_NAME` and `KERNEL_TYPE` are required to be set.
#
# Usage:
#   include(jit_verify_template)
#   jit_verify_template(
#       NAME          basic_matmul
#       TEMPLATE      path/to/basic_matmul_impl.cpp
#       NPU_ARCH_LIST 2201 3510
#       KERNEL_TYPE   __cube__
#       KERNEL_NAME   basic_matmul
#       INCLUDE_DIRS  dir1 dir2 ...)
# ---------------------------------------------------------------------------

function(jit_verify_template)
    cmake_parse_arguments(_TV "" "NAME;KERNEL_TYPE;KERNEL_NAME" "NPU_ARCH_LIST;INCLUDE_DIRS;TEMPLATE" ${ARGN})

    if(NOT _TV_NAME OR NOT _TV_TEMPLATE OR NOT _TV_NPU_ARCH_LIST)
        message(FATAL_ERROR "jit_verify_template: NAME, TEMPLATE, NPU_ARCH_LIST required")
    endif()

    # Keep the macro environment identical to the runtime JIT compilation.
    if(NOT _TV_KERNEL_TYPE)
        set(_TV_KERNEL_TYPE "__cube__")
    endif()
    if(NOT _TV_KERNEL_NAME)
        set(_TV_KERNEL_NAME "${_TV_NAME}")
    endif()

    # 将所有 template 路径解析为绝对路径，并校验存在
    set(_TV_TEMPLATE_ABS "")
    foreach(_TPL ${_TV_TEMPLATE})
        if(NOT IS_ABSOLUTE "${_TPL}")
            set(_TPL "${CMAKE_CURRENT_SOURCE_DIR}/${_TPL}")
        endif()
        if(NOT EXISTS "${_TPL}")
            message(FATAL_ERROR "jit_verify_template: template not found at ${_TPL}")
        endif()
        list(APPEND _TV_TEMPLATE_ABS "${_TPL}")
    endforeach()

    foreach(_ARCH ${_TV_NPU_ARCH_LIST})
        set(_TGT jit_verify_${_TV_NAME}_${_ARCH})
        # As SHARED (not OBJECT) so that link-time failures are detected too.
        add_library(${_TGT} SHARED EXCLUDE_FROM_ALL ${_TV_TEMPLATE_ABS})
        set_source_files_properties(${_TV_TEMPLATE_ABS} PROPERTIES LANGUAGE ASC)
        target_compile_options(${_TGT} PRIVATE
            --npu-arch=dav-${_ARCH}
            -Wno-macro-redefined
        )
        target_compile_definitions(${_TGT} PRIVATE
            CATLASS_ARCH=${_ARCH}
            "KERNEL_TYPE=${_TV_KERNEL_TYPE}"
            "KERNEL_NAME=${_TV_KERNEL_NAME}"
            "CATLASS_JIT_KERNEL_NAME=${_TV_KERNEL_NAME}_arch${_ARCH}"
        )
        target_include_directories(${_TGT} PRIVATE ${_TV_INCLUDE_DIRS})
        set_property(GLOBAL APPEND PROPERTY _JIT_VERIFY_TARGETS ${_TGT})
    endforeach()
endfunction()
