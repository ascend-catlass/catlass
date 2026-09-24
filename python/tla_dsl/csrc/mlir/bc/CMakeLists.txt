# Install bc stubs and their catlass header dependencies into the cmake
# install prefix (CMAKE_INSTALL_PREFIX).  Together with the .so and _mlir
# install rules, this lets setup.py copy everything from a single prefix
# to the wheel (build → install → wheel pattern).
#
# The actual cpp → bc compilation happens at runtime (see bc_compile.py).
#
# Wheel layout:
#   catlass/csrc/mlir/bc/Cube/*.cpp   ← stub sources
#   catlass/csrc/mlir/bc/Vector/*.cpp
#   catlass/csrc/mlir/bc/*.h          ← bc helper headers
#   catlass/include/catlass/…         ← catlass headers (tile-level only)
#   catlass/include/tla/…             ← tla headers

# --- 1. bc stubs (.cpp + .h) ------------------------------------------------

install(
  DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/Cube"
    "${CMAKE_CURRENT_SOURCE_DIR}/Vector"
    "${CMAKE_CURRENT_SOURCE_DIR}/Scalar"
  DESTINATION catlass/csrc/mlir/bc
  COMPONENT catlass_wheel
  FILES_MATCHING
    PATTERN "*.cpp"
    PATTERN "*.h"
)

# bc-root helper headers (common.h, dma_common.h, print_tensor_workspace.h)
file(GLOB _bc_root_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
foreach(_hdr IN LISTS _bc_root_hdrs)
  install(
    FILES "${_hdr}"
    DESTINATION catlass/csrc/mlir/bc
    COMPONENT catlass_wheel
  )
endforeach()

# --- 2. catlass + tla headers required by bc stubs --------------------------
#
# Glob all catlass/*.hpp and tla/*.hpp recursively, exclude block / device /
# kernel directories (upper-level operator components not needed by bc stubs).

file(GLOB_RECURSE _bc_needed_hdrs
  "${CATLASS_DSL_CATLASS_INCLUDE_DIR}/catlass/*.hpp"
  "${CATLASS_DSL_CATLASS_INCLUDE_DIR}/tla/*.hpp"
)
list(FILTER _bc_needed_hdrs EXCLUDE REGEX "/(block|device|kernel)/")

foreach(_hdr IN LISTS _bc_needed_hdrs)
  file(RELATIVE_PATH _rel "${CATLASS_DSL_CATLASS_INCLUDE_DIR}" "${_hdr}")
  get_filename_component(_rel_dir "${_rel}" DIRECTORY)
  install(
    FILES "${_hdr}"
    DESTINATION "catlass/include/${_rel_dir}"
    COMPONENT catlass_wheel
  )
endforeach()
