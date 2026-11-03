# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(Catlass LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

include(cmake/funcs.cmake)
ensure_bisheng_stdlib_source()

if((NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
        OR (NOT DEFINED CANN_VERSION OR CANN_VERSION STREQUAL ""))
    get_cann_version_and_soc_name()
endif()

if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
    message(WARNING "Failed to detect SOC_VERSION. Simulator or aclnn examples may not run correctly.\nYou can set SOC_VERSION manually by -DSOC_VERSION=Ascendxxxyy.")
endif()

if(NOT DEFINED CANN_VERSION OR CANN_VERSION STREQUAL "")
    message(WARNING "Failed to detect CANN_VERSION. Version compatibility checks will be skipped.")
endif()

set(CATLASS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(CATLASS_3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty)

if(NOT DEFINED CMAKE_BUILD_TYPE OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT DEFINED CATLASS_ARCH OR NOT CATLASS_ARCH)
    message(WARNING "CATLASS_ARCH is not defined, use default value \"2201\"")
    set(CATLASS_ARCH 2201)
endif()

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# Compile options for secure compile
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_SKIP_RPATH TRUE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fstack-protector-strong>)
    add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASC>:-Xhost-start -fstack-protector-strong -Xhost-end>")
    add_link_options(-Wl,-z,relro,-z,now)
    add_link_options(-Wl,-z,noexecstack)
    add_link_options(-s)
endif()

add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(3rdparty)

if(DEFINED BUILD_TESTS AND BUILD_TESTS)
    add_subdirectory(tests)
endif()
