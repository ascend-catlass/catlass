#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace tla {

std::unique_ptr<mlir::Pass> createTlaLowerDebugPrintPass();
std::unique_ptr<mlir::Pass> createTlaLowerPtrPass();
std::unique_ptr<mlir::Pass> createTlaLowerExternCallPass();
std::unique_ptr<mlir::Pass> createTlaCubeRegionPass();
std::unique_ptr<mlir::Pass> createTlaFinalizeMemrefPass();
std::unique_ptr<mlir::Pass> createTlaLowerFlagBarrierToHivmPass();
std::unique_ptr<mlir::Pass> createTlaLowerFuncPass();
std::unique_ptr<mlir::Pass> createTlaInsertAutoMutexPass();
std::unique_ptr<mlir::Pass> createTlaLowerScalarAccessPass();
std::unique_ptr<mlir::Pass> createTlaLowerScalarOpsPass();
std::unique_ptr<mlir::Pass> createTlaSplitMixedFuncPass();
std::unique_ptr<mlir::Pass> createTlaLowerTensorDescPass();
std::unique_ptr<mlir::Pass> createTlaLowerBlockIdxPass();
std::unique_ptr<mlir::Pass> createTlaVectorRegionPass();
std::unique_ptr<mlir::Pass> createTlaPrologueEpiloguePass();
std::unique_ptr<mlir::Pass> createTlaLowerAVEToRegbaseIntrinsPass();

void registerTlaPasses();
void buildTlaPipeline(mlir::OpPassManager& pm);

} // namespace tla
