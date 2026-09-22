# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../utils)

add_library(catlass_torch SHARED)

target_include_directories(catlass_torch PRIVATE
    ${COMMON_INCLUDE_DIR}
    ${UTILS_DIR}
    ${ASCEND_HOME_PATH}/include
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include
    ${CATLASS_DIR}/include
)

target_link_directories(catlass_torch PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${TORCH_NPU_DIR}/lib
)

target_link_libraries(catlass_torch PRIVATE
    catlass_torch_utils
    catlass_kernel_utils
    ${TORCH_LIBRARIES}
    torch_npu
    tiling_api
    dl
)

if(DISABLE_ASCEND950)
    target_compile_definitions(catlass_torch PRIVATE ENABLE_ASCEND950=0)
else()
    target_compile_definitions(catlass_torch PRIVATE ENABLE_ASCEND950=1)
endif()

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
if(SRC_FILES)
    target_sources(catlass_torch PRIVATE ${SRC_FILES})
endif()

install(TARGETS catlass_torch DESTINATION lib)
