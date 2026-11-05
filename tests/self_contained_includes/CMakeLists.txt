# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(test_self_contained_includes.cpp PROPERTIES LANGUAGE ASC)
add_custom_target(test_self_contained_includes)

include_directories(${CATLASS_INCLUDE_DIR})
file(GLOB_RECURSE CATLASS_HEADERS ${CATLASS_INCLUDE_DIR}/**/*.hpp)
list(FILTER CATLASS_HEADERS EXCLUDE REGEX "block")
set(ARCH_CONFIGS
    "2201\;atlasa2"
    "3510\;ascend950"
)

foreach(ARCH_CONFIG ${ARCH_CONFIGS})
    string(REPLACE ";" " " ARCH_CONFIG_STR ${ARCH_CONFIG})

    list(GET ARCH_CONFIG 0 NPU_ARCH)
    list(GET ARCH_CONFIG 1 CATLASS_ARCH_TAG)

    if(CATLASS_ARCH STREQUAL ${NPU_ARCH})
        add_compile_options("--npu-arch=dav-${NPU_ARCH}")
        add_compile_definitions(CATLASS_ARCH=${NPU_ARCH})
    else()
        list(FILTER CATLASS_HEADERS EXCLUDE REGEX ${CATLASS_ARCH_TAG})
    endif()
endforeach()

# EVG (epilogue/fusion) is Ascend950-only; skip on Atlas A2 and Atlas A3 UT.
if(NOT CATLASS_ARCH STREQUAL "3510")
    list(FILTER CATLASS_HEADERS EXCLUDE REGEX "epilogue/fusion")
endif()

# non-tla tile components of ascend950 is not ready, gemv and conv do not work.
if(CATLASS_ARCH STREQUAL "3510")
    list(FILTER CATLASS_HEADERS EXCLUDE REGEX "gemv")
    list(FILTER CATLASS_HEADERS EXCLUDE REGEX "conv")
endif()

foreach(CATLASS_HEADER ${CATLASS_HEADERS})
    file(RELATIVE_PATH REL_CATLASS_HEADER ${CATLASS_INCLUDE_DIR} ${CATLASS_HEADER})
    string(REPLACE "/" "_" TEST_TARGET ${REL_CATLASS_HEADER})
    string(REPLACE "." "_" TEST_TARGET ${TEST_TARGET})
    add_library(${TEST_TARGET} OBJECT test_self_contained_includes.cpp)
    target_compile_definitions(${TEST_TARGET} PRIVATE CATLASS_TEST_HEADER=${REL_CATLASS_HEADER})
    add_dependencies(test_self_contained_includes ${TEST_TARGET})

    # install(TARGETS ${TEST_TARGET} DESTINATION obj COMPONENT catlass_test_self_contained_includes)
endforeach()
