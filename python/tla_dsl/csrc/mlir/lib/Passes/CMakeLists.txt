add_library(TlaPasses
  PassRegistry.cpp
  TlaLowerDebugPrintPass.cpp
  TlaLowerScalarOpsPass.cpp
  TlaLowerFuncPass.cpp
  TlaInsertAutoMutexPass.cpp
  TlaLowerScalarAccessPass.cpp
  TlaSplitMixedFuncPass.cpp
  TlaLowerTensorDescPass.cpp
  TlaTensorDescDerivation.cpp
  TlaTensorDescriptor.cpp
  TlaLowerBlockIdxPass.cpp
  TlaScratchAllocation.cpp
  TlaLowerPtrPass.cpp
  TlaLowerExternCallPass.cpp
  TlaLowerMutexToStdPass.cpp
  TlaFinalizeMemrefPass.cpp
  TlaCubeRegionPass.cpp
  TlaTensorToMemref.cpp
  TlaLowerFlagBarrierToHivmPass.cpp
  TlaPrologueEpiloguePass.cpp
  TlaLowerAVEToRegbaseIntrinsPass.cpp
  TlaVectorRegionPass.cpp
  ../Dialect/Tla/IR/TlaDialect.cpp
  ../Dialect/Tla/IR/TlaAttrs.cpp
  ../Dialect/Tla/IR/TlaOps.cpp
)
add_dependencies(TlaPasses TlaTblgen)
set_target_properties(TlaPasses PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(TlaPasses PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/tblgen
  ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(TlaPasses PUBLIC -Wl,--start-group ${CATLASS_DSL_HIVM_REQUIRED_LIBS} -Wl,--end-group)
target_link_libraries(TlaPasses
  PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRDialect
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRArithTransforms
    MLIRControlFlowDialect
    MLIRAffineToStandard
    MLIRAffineTransforms
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRFuncTransforms
    MLIRDestinationStyleOpInterface
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRMeshDialect
    MLIRSideEffectInterfaces
    MLIRTensorDialect
    MLIRViewLikeInterface
    MLIRMemRefUtils
    MLIRDLTIDialect
    MLIRLLVMCommonConversion
    MLIRConvertToLLVMInterface
    MLIRLLVMDialect
    MLIRSCFToControlFlow
)

add_executable(tla_print_tensor_lowering_type_mapping_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/PrintTensorLoweringTypeMappingTest.cpp
)
target_include_directories(tla_print_tensor_lowering_type_mapping_test PRIVATE
  ${PROJECT_SOURCE_DIR}/lib
)
target_link_libraries(tla_print_tensor_lowering_type_mapping_test PRIVATE
  TlaPasses
)

add_executable(tla_debug_print_lowering_type_mapping_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/DebugPrintLoweringTypeMappingTest.cpp
)
target_include_directories(tla_debug_print_lowering_type_mapping_test PRIVATE
  ${PROJECT_SOURCE_DIR}/lib
)
target_link_libraries(tla_debug_print_lowering_type_mapping_test PRIVATE
  TlaPasses
)
