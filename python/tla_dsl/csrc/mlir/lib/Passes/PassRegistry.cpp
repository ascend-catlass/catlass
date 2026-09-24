#include "PassesCommon.h"
#include "PassesInternal.h"

#include "bishengir/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.h"
#include "bishengir/Conversion/HIVMAVEToStandard/HIVMAVEToStandard.h"
#include "bishengir/Conversion/HIVMToStandard/HIVMToStandard.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace tla {

void registerTlaPasses()
{
    // Register the external pass arguments used by TlaCompile's
    // --mlir-print-ir-before/after filters without pulling in every upstream pass.
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> { return hivmave::createCombineAVEOPsPass(); });
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> { return mlir::createConvertHIVMAVEToAVEIntrinPass(); });
    registerTlaLowerDebugPrintPass();
    registerTlaLowerFuncPass();
    registerTlaInsertAutoMutexPass();
    registerTlaLowerScalarAccessPass();
    registerTlaSplitMixedFuncPass();
    registerTlaLowerTensorDescPass();
    registerTlaLowerBlockIdxPass();
    registerTlaVectorRegionPass();
    registerTlaLowerFlagBarrierToHivmPass();
    registerTlaLowerPtrPass();
    registerTlaLowerExternCallPass();
    registerTlaLowerMutexToStdPass();
    registerTlaCubeRegionPass();
    registerTlaFinalizeMemrefPass();
    registerTlaPrologueEpiloguePass();
    registerTlaLowerAVEToRegbaseIntrinsPass();
    registerTlaLowerScalarOpsPass();
}

void buildTlaPipeline(OpPassManager& pm)
{
    // Lower Tla function containers to func.func before HIVM lowering so
    // downstream passes and tools inspect func.func. This single pass classifies
    // each device function's AIC/AIV/MIX core type from its tla.cube/tla.vector
    // regions, stamps the HACC/HIVM entry metadata, lowers tla.func to func.func,
    // and tags the module core type -- one structure-derived classification the
    // HACC machinery and the mixed-func split both consume.
    pm.addPass(createTlaLowerFuncPass());
    pm.addPass(createTlaInsertAutoMutexPass());
    pm.addPass(createTlaLowerExternCallPass());
    pm.addPass(createTlaLowerPtrPass());
    pm.addPass(createTlaSplitMixedFuncPass());
    // Materialize tensor-view producer chains as tla.tensor_desc before region
    // passes consume them. Runs after tla-lower-ptr so descriptor bases use the
    // bridged memref or !tla.ptr produced by the standard pipeline.
    pm.addPass(createTlaLowerTensorDescPass());
    // Lower all GM scalar accesses from the materialized descriptor form.
    pm.addPass(createTlaLowerScalarAccessPass());
    pm.addPass(createTlaVectorRegionPass());
    pm.addPass(createTlaCubeRegionPass());
    pm.addPass(createTlaFinalizeMemrefPass());
    pm.addPass(createTlaLowerDebugPrintPass());
    pm.addPass(createTlaLowerScalarOpsPass());
    pm.addPass(createTlaLowerBlockIdxPass());
    pm.addPass(createTlaLowerFlagBarrierToHivmPass());
    pm.addPass(createTlaLowerMutexToStdPass());
    pm.addPass(createTlaPrologueEpiloguePass());
    pm.addPass(createCSEPass());
    // Fuse AVE instruction sequences after every TLA/vector/arith producer has
    // been lowered. In particular, vsub followed by vexp becomes vexpdif on
    // Ascend 950 targets before AVE intrinsic conversion consumes the ops.
    pm.nest<func::FuncOp>().addPass(hivmave::createCombineAVEOPsPass());
    // Decompose HIVM ops which are not natively supported by hivmc.
    // For example, ``SetAtomicOp`` is decomposed into a sequence of
    // ``set_ctrl`` operations.
    pm.nest<func::FuncOp>().addPass(hivm::createHIVMDecomposeOpPass());
    mlir::ConvertHIVMToStandardOptions hivmToStdOptions;
    pm.addPass(mlir::createConvertHIVMToStandardPass(hivmToStdOptions));
    pm.addPass(mlir::createConvertHIVMAVEToStandardPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    // Prefer NPU-IR's built-in ForOpIVNarrowing (enable-i16-indvar) over a
    // duplicated TLA pass. Frontend ``tla.range`` emits i32 IVs; ccec VLOOPv2
    // recognition benefits from i16 when bounds fit. Default TableGen option is
    // false, so turn it on here when constructing the pass.
    {
        std::unique_ptr<mlir::Pass> hivmaveToIntrin = mlir::createConvertHIVMAVEToAVEIntrinPass();
        if (failed(hivmaveToIntrin->initializeOptions("enable-i16-indvar=true", [](const llvm::Twine& msg) {
                llvm::errs() << "convert-hivmave-to-ave-intrin options: " << msg << "\n";
                return mlir::failure();
            }))) {
            llvm::report_fatal_error("failed to set enable-i16-indvar=true on convert-hivmave-to-ave-intrin");
        }
        pm.addPass(std::move(hivmaveToIntrin));
    }
    pm.addPass(createTlaLowerAVEToRegbaseIntrinsPass());
    pm.addPass(createConvertSCFToCFPass());
}

} // namespace tla
