cmake_minimum_required(VERSION 3.20)
# Use the host sysroot by default (conda glibc 2.28 lacks LLVM symbols); keep explicit overrides.
if(NOT DEFINED CMAKE_SYSROOT OR "${CMAKE_SYSROOT}" STREQUAL "")
  set(CMAKE_SYSROOT "/")
endif()
project(tla LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-fstack-protector-strong)
  add_link_options(-Wl,-z,relro,-z,now)
  add_link_options(-Wl,-z,noexecstack)
  add_link_options(-s)
endif()

option(ENABLE_CPU_TRACE_INTRINSIC "If enable debug, use cpu compile." OFF)
message("-- ENABLE_CPU_TRACE_INTRINSIC Value: ${ENABLE_CPU_TRACE_INTRINSIC}")

if(DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
else()
  message(FATAL_ERROR "ASCEND_HOME_PATH is not defined.")
endif()
message("-- ASCEND_HOME_PATH Value: ${ASCEND_HOME_PATH}")

# Catlass public headers for bisheng template bitcode (see mlir/bc).
if(DEFINED CATLASS_INCLUDE_DIR AND NOT "${CATLASS_INCLUDE_DIR}" STREQUAL "")
  set(CATLASS_DSL_CATLASS_INCLUDE_DIR "${CATLASS_INCLUDE_DIR}")
else()
  get_filename_component(CATLASS_DSL_CATLASS_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include" ABSOLUTE)
endif()
message("-- CATLASS_DSL_CATLASS_INCLUDE_DIR Value: ${CATLASS_DSL_CATLASS_INCLUDE_DIR}")

set(_catlass_dsl_ascendnpu_ir_source_dir
  "$ENV{CATLASS_DSL_ASCENDNPU_IR_ROOT}")
set(_catlass_dsl_ascendnpu_ir_install_dir
  "$ENV{CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR}")
if(NOT _catlass_dsl_ascendnpu_ir_source_dir
   OR NOT _catlass_dsl_ascendnpu_ir_install_dir)
  message(FATAL_ERROR
    "AscendNPU-IR source and install paths were not prepared by build.sh")
endif()
get_filename_component(_catlass_dsl_ascendnpu_ir_source_dir
  "${_catlass_dsl_ascendnpu_ir_source_dir}" ABSOLUTE)
get_filename_component(_catlass_dsl_ascendnpu_ir_install_dir
  "${_catlass_dsl_ascendnpu_ir_install_dir}" ABSOLUTE)

message("-- AscendNPU-IR source: ${_catlass_dsl_ascendnpu_ir_source_dir}")
message("-- AscendNPU-IR install: ${_catlass_dsl_ascendnpu_ir_install_dir}")

set(CATLASS_DSL_HIVM_INCLUDE_DIR
    ${_catlass_dsl_ascendnpu_ir_install_dir}/include)
set(CATLASS_DSL_HIVM_LIB_DIR
    ${_catlass_dsl_ascendnpu_ir_install_dir}/lib)
set(CATLASS_DSL_HIVM_HEADER
    ${CATLASS_DSL_HIVM_INCLUDE_DIR}/bishengir/Dialect/HIVM/IR/HIVM.h)
set(CATLASS_DSL_HIVM_ENUMS_INC
    ${CATLASS_DSL_HIVM_INCLUDE_DIR}/bishengir/Interfaces/BiShengIREnums.h.inc)
set(CATLASS_DSL_HIVM_REQUIRED_LIBS
    BiShengIRHIVMDialect
    BiShengIRHIVMTransforms
    BiShengIRHIVMToStandard
    MLIRAVEDialect
    MLIRHIVMAVETransforms
    MLIRAVEUtils
    MLIRHIVMAVEToAVEIntrin
    MLIRHIVMAVEToStandard
    MLIRHIVMRegbaseIntrinsDialect
    MLIRHIVMRegbaseIntrinsUtils
    BiShengIRHFusionToHIVM
    MLIRTensorUtils
    BiShengIRAnnotationDialect
    BiShengIRDialectUtils
    BiShengIRTensorDialect
    BiShengIRHFusionDialect
    BiShengIRHFusionUtils
    BiShengIRMathExtDialect
    BiShengIRSymbolDialect
    BiShengIRMemRefDialect
    BiShengIRMemRefExtDialect
    BiShengIRHACCUtils
    BiShengIRHACCTransforms
    BiShengIRHACCDialect
    BiShengIRScopeDialect)
if(NOT EXISTS ${CATLASS_DSL_HIVM_HEADER} OR NOT EXISTS ${CATLASS_DSL_HIVM_ENUMS_INC})
  message(FATAL_ERROR
          "AscendNPU-IR headers are missing from "
          "${_catlass_dsl_ascendnpu_ir_install_dir}/include")
endif()
foreach(tla_dsl_hivm_lib IN LISTS CATLASS_DSL_HIVM_REQUIRED_LIBS)
  set(tla_dsl_hivm_archive
      "${CATLASS_DSL_HIVM_LIB_DIR}/lib${tla_dsl_hivm_lib}.a")
  if(NOT EXISTS "${tla_dsl_hivm_archive}")
    message(FATAL_ERROR
            "AscendNPU-IR static library is missing: ${tla_dsl_hivm_archive}")
  endif()
  list(APPEND CATLASS_DSL_HIVM_REQUIRED_LIB_FILES
       "${tla_dsl_hivm_archive}")
endforeach()

set(MLIR_DIR ${_catlass_dsl_ascendnpu_ir_install_dir}/lib/cmake/mlir)
set(LLVM_DIR ${_catlass_dsl_ascendnpu_ir_install_dir}/lib/cmake/llvm)
set(MLIR_TBLGEN_INCLUDE_DIR ${_catlass_dsl_ascendnpu_ir_install_dir}/include)

# Install bc stubs + dependent headers for wheel packaging (see bc_compile.py).
add_subdirectory(bc)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Threads REQUIRED)
# python/tla_dsl: sibling of catlass/ (Python pkg) and csrc/ (contains tests/, pyproject, .artifacts, tools).
set(TLA_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _catlass_pybind11_cmake_dir
    RESULT_VARIABLE _catlass_pybind11_result
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_catlass_pybind11_result EQUAL 0 AND _catlass_pybind11_cmake_dir)
    set(pybind11_DIR "${_catlass_pybind11_cmake_dir}")
    find_package(pybind11 CONFIG QUIET)
  endif()
endif()
if(NOT pybind11_FOUND)
  message(FATAL_ERROR "pybind11 must be installed in the active Python environment")
endif()
set(TLA_TD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/Dialect/Tla/IR)
# Tla.td is the entry point mlir-tblgen is run on; it only includes the files
# below. Keep this list in sync with the include lines in Tla.td, or an edit to
# a fragment will not trigger a regeneration.
set(TLA_TD_FILE ${TLA_TD_DIR}/Tla.td)
set(TLA_TD_SOURCES
  ${TLA_TD_FILE}
  ${TLA_TD_DIR}/TlaBase.td
  ${TLA_TD_DIR}/TlaEnums.td
  ${TLA_TD_DIR}/TlaTypes.td
  ${TLA_TD_DIR}/TlaRegionOps.td
  ${TLA_TD_DIR}/TlaArchOps.td
  ${TLA_TD_DIR}/TlaScalarOps.td
  ${TLA_TD_DIR}/TlaStructureOps.td
  ${TLA_TD_DIR}/TlaTensorOps.td
  ${TLA_TD_DIR}/TlaMemoryOps.td
  ${TLA_TD_DIR}/TlaSyncOps.td
  ${TLA_TD_DIR}/TlaCubeOps.td
  ${TLA_TD_DIR}/TlaSimdOps.td
  ${TLA_TD_DIR}/TlaSimtOps.td
)
# Dialect TableGen targets (Tla dialect).
find_package(MLIR REQUIRED CONFIG)
if(DEFINED LLVM_CMAKE_DIR)
  include("${LLVM_CMAKE_DIR}/AddLLVM.cmake")
  include("${LLVM_CMAKE_DIR}/TableGen.cmake")
endif()
if(DEFINED MLIR_CMAKE_DIR)
  include("${MLIR_CMAKE_DIR}/AddMLIR.cmake")
endif()
# Some MLIR installations do not expose MLIR_TBLGEN_INCLUDE_DIR; mlir-tblgen
# still needs the installed TableGen include tree (fall back to MLIR_INCLUDE_DIRS).
if(NOT DEFINED MLIR_TBLGEN_INCLUDE_DIR OR "${MLIR_TBLGEN_INCLUDE_DIR}" STREQUAL "")
  if(DEFINED MLIR_INCLUDE_DIRS AND NOT "${MLIR_INCLUDE_DIRS}" STREQUAL "")
    set(MLIR_TBLGEN_INCLUDE_DIR "${MLIR_INCLUDE_DIRS}")
  elseif(DEFINED LLVM_INCLUDE_DIRS AND NOT "${LLVM_INCLUDE_DIRS}" STREQUAL "")
    set(MLIR_TBLGEN_INCLUDE_DIR "${LLVM_INCLUDE_DIRS}")
  endif()
endif()
if(NOT DEFINED MLIR_TBLGEN_INCLUDE_DIR OR "${MLIR_TBLGEN_INCLUDE_DIR}" STREQUAL "")
  message(FATAL_ERROR
          "MLIR_TBLGEN_INCLUDE_DIR is not set and could not be inferred. "
          "Pass -DMLIR_TBLGEN_INCLUDE_DIR=<dir containing mlir/IR/OpBase.td>.")
endif()
message(STATUS "MLIR_TBLGEN_INCLUDE_DIR=${MLIR_TBLGEN_INCLUDE_DIR}")
find_file(CATLASS_DSL_MLIR_PYBIND_ADAPTORS
  NAMES mlir/Bindings/Python/PybindAdaptors.h
  PATHS ${MLIR_INCLUDE_DIRS}
  NO_DEFAULT_PATH
)
if(NOT CATLASS_DSL_MLIR_PYBIND_ADAPTORS)
  message(FATAL_ERROR
          "MLIR Python PybindAdaptors.h is required for the typed Python bridge.")
endif()
set(CATLASS_DSL_MLIR_PYTHON_CAPI_LIBRARY "$<TARGET_FILE:CatlassAggregateCAPI>")
set(CATLASS_DSL_MLIR_PYTHON_CAPI_DIR "$<TARGET_FILE_DIR:CatlassAggregateCAPI>")

if(NOT TARGET LLVM)
  find_package(LLVM REQUIRED CONFIG HINTS "${MLIR_CMAKE_DIR}/../llvm")
endif()

set(METAPASSES_TBLGEN_EXE "${MLIR_TABLEGEN_EXE}")
if(NOT METAPASSES_TBLGEN_EXE)
  set(METAPASSES_TBLGEN_EXE "${MLIR_TBLGEN}")
endif()
set(METAPASSES_TBLGEN_INCLUDES
  ${MLIR_TBLGEN_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set(METAPASSES_TBLGEN_INCLUDE_FLAGS)
foreach(_inc ${METAPASSES_TBLGEN_INCLUDES})
  if (_inc)
    list(APPEND METAPASSES_TBLGEN_INCLUDE_FLAGS -I ${_inc})
  endif()
endforeach()
list(APPEND METAPASSES_TBLGEN_INCLUDE_FLAGS -D BSPUB_DAVINCI_BISHENGIR_A5)
set(METAPASSES_TBLGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tblgen)

# tla dialect TableGen outputs (dialect/attrs/types/ops).
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.h.inc
  COMMAND ${CMAKE_COMMAND} -E make_directory ${METAPASSES_TBLGEN_OUTPUT_DIR}/tla
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-dialect-defs ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.cpp.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-dialect-decls ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.h.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-attrdef-defs ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.cpp.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-attrdef-decls ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.h.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-enum-defs ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.cpp.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-enum-decls ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.h.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-typedef-defs ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.cpp.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-typedef-decls ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.h.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-op-defs ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.cpp.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-op-decls ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.h.inc
  COMMAND_EXPAND_LISTS
  DEPENDS ${TLA_TD_SOURCES}
  COMMENT "Generate TableGen files for tla"
)
add_custom_target(TlaTblgen DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.h.inc
)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=catlass._mlir.")
add_compile_definitions(BSPUB_DAVINCI_BISHENGIR_A5)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${CATLASS_DSL_HIVM_INCLUDE_DIR})
link_directories(${CATLASS_DSL_HIVM_LIB_DIR})
set(CATLASS_DSL_PYTHON_PACKAGE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/python/catlass)
file(MAKE_DIRECTORY ${CATLASS_DSL_PYTHON_PACKAGE_OUTPUT_DIR})

add_subdirectory(lib)
add_subdirectory(tools/tla-compile)

add_subdirectory(python)

# Convenience target: build all compiler tools.
add_custom_target(tla-compiler DEPENDS TlaCompile _tla_type_bridge_native)

# Lit regression tests for compiler tools.
set(CATLASS_DSL_TLA_COMPILE_TOOL
    "${CMAKE_CURRENT_BINARY_DIR}/tools/tla-compile/TlaCompile")
if (EXISTS "${TLA_REPO_ROOT}/tests/lit/CMakeLists.txt")
  add_subdirectory(${TLA_REPO_ROOT}/tests/lit ${CMAKE_CURRENT_BINARY_DIR}/tests/lit)
endif()
