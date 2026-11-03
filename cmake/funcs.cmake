# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(get_cann_version_and_soc_name)
    set(GET_CANN_VERSION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/get_cann_version_and_soc_name.cpp)
    set(GET_CANN_VERSION_BIN ${CMAKE_BINARY_DIR}/get_cann_version_and_soc_name)

    execute_process(
        COMMAND g++ ${GET_CANN_VERSION_SRC} -o ${GET_CANN_VERSION_BIN} -std=c++17
            -I${ASCEND_HOME_PATH}/include -L${ASCEND_HOME_PATH}/lib64 -lascendcl -lc_sec
        RESULT_VARIABLE COMPILE_RESULT
        OUTPUT_VARIABLE COMPILE_OUTPUT
        ERROR_VARIABLE COMPILE_ERROR
    )

    if(NOT COMPILE_RESULT EQUAL 0)
        message(WARNING "Failed to compile get_cann_version_and_soc_name: ${COMPILE_ERROR}")
        return()
    endif()

    execute_process(
        COMMAND ${GET_CANN_VERSION_BIN}
        RESULT_VARIABLE RUN_RESULT
        OUTPUT_VARIABLE RUN_OUTPUT
        ERROR_VARIABLE RUN_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT RUN_RESULT EQUAL 0)
        message(WARNING "Failed to run get_cann_version_and_soc_name: ${RUN_ERROR}")
        return()
    endif()

    string(REGEX MATCH "^[^ ]+" CANN_VERSION_TMP "${RUN_OUTPUT}")

    # 只在调用方未预先设定 CANN_VERSION 时才覆盖
    if(NOT DEFINED CANN_VERSION OR CANN_VERSION STREQUAL "")
        set(CANN_VERSION ${CANN_VERSION_TMP} PARENT_SCOPE)
        message("Detected CANN_VERSION: ${CANN_VERSION_TMP}")
    else()
        message("Using preset CANN_VERSION: ${CANN_VERSION}")
    endif()

    string(LENGTH "${RUN_OUTPUT}" OUTPUT_LEN)
    string(FIND "${RUN_OUTPUT}" " " SPACE_POS)
    if(SPACE_POS GREATER_EQUAL 0)
        math(EXPR SOC_NAME_START "${SPACE_POS} + 1")
        string(SUBSTRING "${RUN_OUTPUT}" ${SOC_NAME_START} -1 SOC_VERSION_TMP)

        # 只在调用方未预先设定 SOC_VERSION 时才覆盖
        if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
            set(SOC_VERSION ${SOC_VERSION_TMP} PARENT_SCOPE)
            message("Detected SOC_VERSION: ${SOC_VERSION_TMP}")
        else()
            message("Using preset SOC_VERSION: ${SOC_VERSION}")
        endif()
    endif()
endfunction()

function(ensure_bisheng_stdlib_source)
    # Probe 1: can bisheng parse C++ standard library headers?
    execute_process(
        COMMAND printf "%s" "#include <cstdlib>\n"
        COMMAND "${CMAKE_ASC_COMPILER}" -x c++ - -fsyntax-only
        RESULT_VARIABLE STDLIB_HEADER_RESULT
        OUTPUT_QUIET
        ERROR_VARIABLE STDLIB_HEADER_ERROR
    )

    # Probe 2: can bisheng link against the C++ standard library?
    # -fsyntax-only does not link, so a link failure such as
    # "unable to find library -lstdc++" is invisible to the header probe.
    execute_process(
        COMMAND printf "%s" "int main() { return 0; }\n"
        COMMAND "${CMAKE_ASC_COMPILER}" -x c++ - -o "${CMAKE_CURRENT_BINARY_DIR}/stdlib_probe" -lstdc++
        RESULT_VARIABLE STDLIB_LINK_RESULT
        OUTPUT_QUIET
        ERROR_VARIABLE STDLIB_LINK_ERROR
    )
    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/stdlib_probe")

    if(STDLIB_HEADER_RESULT STREQUAL "0" AND STDLIB_LINK_RESULT STREQUAL "0")
        return()
    endif()

    # C++ standard library headers only; exclude GCC builtin headers.
    set(STDLIB_DIRS "")
    foreach(DIR IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
        if(EXISTS "${DIR}/cstdint"
           OR EXISTS "${DIR}/bits/c++config.h"
           OR EXISTS "${DIR}/backward_warning.h")
            list(APPEND STDLIB_DIRS "${DIR}")
        endif()
    endforeach()

    if(NOT STDLIB_DIRS OR NOT CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES)
        message(FATAL_ERROR
            "Cannot locate host C++ standard library paths.\n"
            "Header probe: ${STDLIB_HEADER_RESULT}\n${STDLIB_HEADER_ERROR}\n"
            "Link probe: ${STDLIB_LINK_RESULT}\n${STDLIB_LINK_ERROR}")
    endif()

    list(JOIN STDLIB_DIRS ":" INCLUDE_PATH)
    list(JOIN CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES ":" LIBRARY_PATH)

    if(NOT "$ENV{CPLUS_INCLUDE_PATH}" STREQUAL "")
        string(APPEND INCLUDE_PATH ":$ENV{CPLUS_INCLUDE_PATH}")
    endif()
    if(NOT "$ENV{LIBRARY_PATH}" STREQUAL "")
        string(APPEND LIBRARY_PATH ":$ENV{LIBRARY_PATH}")
    endif()

    string(CONCAT PREFIX
        "\"${CMAKE_COMMAND}\" -E env "
        "\"CPLUS_INCLUDE_PATH=${INCLUDE_PATH}\" "
        "\"LIBRARY_PATH=${LIBRARY_PATH}\" "
    )

    # ASC rule variables: executables use LINK_EXECUTABLE, shared libraries use
    # CREATE_SHARED_LIBRARY, device modules use CREATE_SHARED_MODULE.
    foreach(VAR CMAKE_ASC_COMPILE_OBJECT
                CMAKE_ASC_LINK_EXECUTABLE
                CMAKE_ASC_CREATE_SHARED_LIBRARY
                CMAKE_ASC_CREATE_SHARED_MODULE)
        if(DEFINED ${VAR} AND NOT "${${VAR}}" STREQUAL "")
            list(TRANSFORM ${VAR} PREPEND "${PREFIX}")
            set(${VAR} "${${VAR}}" PARENT_SCOPE)
        endif()
    endforeach()
endfunction()
