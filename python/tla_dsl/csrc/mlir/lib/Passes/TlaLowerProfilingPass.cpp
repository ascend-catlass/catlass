#include "PassesCommon.h"

namespace tla {
namespace {

constexpr int kProfilingCtrlBit = 0;

class TlaLowerProfilingPass : public PassWrapper<TlaLowerProfilingPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerProfilingPass)

    StringRef getArgument() const override
    {
        return "tla-lower-profiling";
    }
    StringRef getDescription() const override
    {
        return "Lower profiling markers in place to native A5 controls.";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<hivm::HIVMDialect>();
    }

    void runOnOperation() override
    {
        SmallVector<Operation*> markers;
        getOperation().walk([&](Operation* op) {
            if (isa<::tla::ProfStartOp, ::tla::ProfEndOp>(op))
                markers.push_back(op);
        });
        auto pipeAll = hivm::PipeAttr::get(&getContext(), hivm::PIPE::PIPE_ALL);
        for (Operation* op : markers) {
            OpBuilder builder(op);
            // CANN 9.3 __clang_cce_aicore_functions.h metrics_prof_start/stop:
            // barrier(ALL); set_ctrl(sbitset{1,0}(get_ctrl(), 0)); barrier(ALL).
            // These are the native fences, not additional synchronization.
            builder.create<hivm::PipeBarrierOp>(op->getLoc(), pipeAll);
            builder.create<hivm::SetCtrlOp>(op->getLoc(), isa<::tla::ProfStartOp>(op), kProfilingCtrlBit);
            builder.create<hivm::PipeBarrierOp>(op->getLoc(), pipeAll);
            op->erase();
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerProfilingPass()
{
    return std::make_unique<TlaLowerProfilingPass>();
}
void registerTlaLowerProfilingPass()
{
    PassRegistration<TlaLowerProfilingPass>();
}

} // namespace tla
