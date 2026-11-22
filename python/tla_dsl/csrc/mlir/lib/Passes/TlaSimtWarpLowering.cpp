#include "TlaSimtWarpLowering.h"

#include "Dialect/Tla/IR/TlaOps.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace tla {
namespace {
Value buildLaneIdx(OpBuilder& b, Location loc, int64_t bx, int64_t by, int64_t bz)
{
    Type i32 = b.getI32Type();
    Value lane = b.create<hivm_regbaseintrins::ThreadIdXOp>(loc, i32);
    Value mask = b.create<arith::ConstantIntOp>(loc, 31, 32);
    // Warps partition the x-fastest linear thread index into groups of 32.
    if (bx % 32 == 0)
        return b.createOrFold<arith::AndIOp>(loc, lane, mask);
    if (by > 1) {
        Value ty = b.create<hivm_regbaseintrins::ThreadIdYOp>(loc, i32);
        Value scale = b.create<arith::ConstantIntOp>(loc, bx, 32);
        lane = b.createOrFold<arith::AddIOp>(loc, lane, b.createOrFold<arith::MulIOp>(loc, ty, scale));
    }
    if (bz > 1) {
        Value tz = b.create<hivm_regbaseintrins::ThreadIdZOp>(loc, i32);
        Value scale = b.create<arith::ConstantIntOp>(loc, bx * by, 32);
        lane = b.createOrFold<arith::AddIOp>(loc, lane, b.createOrFold<arith::MulIOp>(loc, tz, scale));
    }
    return b.createOrFold<arith::AndIOp>(loc, lane, mask);
}

Value buildLanemask(OpBuilder& b, Location loc, StringRef mode, Value lane)
{
    Value one = b.create<arith::ConstantIntOp>(loc, 1, 32);
    Value bit = b.create<arith::ShLIOp>(loc, one, lane);
    if (mode == "eq")
        return bit;
    Value lt = b.create<arith::SubIOp>(loc, bit, one);
    if (mode == "lt")
        return lt;
    Value all = b.create<arith::ConstantIntOp>(loc, -1, 32);
    if (mode == "ge")
        return b.createOrFold<arith::XOrIOp>(loc, lt, all);
    // Form the inclusive mask without shifting by lane + 1 (32 at lane 31).
    Value le = b.createOrFold<arith::OrIOp>(loc, bit, lt);
    if (mode == "le")
        return le;
    return b.createOrFold<arith::XOrIOp>(loc, le, all);
}

template <typename AddOp, typename MaxOp, typename MinOp>
Value buildNativeReduce(OpBuilder& b, Location loc, StringRef kind, Value value)
{
    if (kind == "add")
        return b.create<AddOp>(loc, value.getType(), value);
    if (kind == "max")
        return b.create<MaxOp>(loc, value.getType(), value);
    return b.create<MinOp>(loc, value.getType(), value);
}

Value buildWarpReduce(OpBuilder& b, Location loc, StringRef kind, bool isUnsigned, Value value)
{
    using namespace hivm_regbaseintrins;
    Type type = value.getType();
    if (type.isInteger(8) || type.isInteger(16)) {
        Value wide;
        if (isUnsigned)
            wide = b.create<arith::ExtUIOp>(loc, b.getI32Type(), value);
        else
            wide = b.create<arith::ExtSIOp>(loc, b.getI32Type(), value);
        Value reduced = buildWarpReduce(b, loc, kind, isUnsigned, wide);
        return b.create<arith::TruncIOp>(loc, type, reduced);
    }
    if (type.isF16())
        return buildNativeReduce<ReduceAddOpF16, ReduceMaxOpF16, ReduceMinOpF16>(b, loc, kind, value);
    if (type.isF32())
        return buildNativeReduce<ReduceAddOpF32, ReduceMaxOpF32, ReduceMinOpF32>(b, loc, kind, value);
    if (isUnsigned)
        return buildNativeReduce<ReduceAddOpU32, ReduceMaxOpU32, ReduceMinOpU32>(b, loc, kind, value);
    return buildNativeReduce<ReduceAddOpS32, ReduceMaxOpS32, ReduceMinOpS32>(b, loc, kind, value);
}
class LowerArchActivemaskPattern : public OpRewritePattern<ArchActivemaskOp> {
public:
    LowerArchActivemaskPattern(MLIRContext* context, int64_t bx, int64_t by, int64_t bz)
        : OpRewritePattern<ArchActivemaskOp>(context), bx(bx), by(by), bz(bz)
    {}

    LogicalResult matchAndRewrite(ArchActivemaskOp op, PatternRewriter& rewriter) const override
    {
        if (op->use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        Location loc = op.getLoc();
        Value lane = buildLaneIdx(rewriter, loc, bx, by, bz);
        Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
        Value bit = rewriter.create<arith::ShLIOp>(loc, one, lane);
        // Distinct lane bits make unsigned addition equivalent to bitwise OR.
        // Emit at the query site so only the current active lanes participate.
        Value mask = rewriter.create<hivm_regbaseintrins::ReduceAddOpU32>(loc, rewriter.getI32Type(), bit);
        rewriter.replaceOp(op, mask);
        return success();
    }

private:
    int64_t bx, by, bz;
};
} // namespace

LogicalResult lowerSimtWarpOpsIn(func::FuncOp vf, int64_t bx, int64_t by, int64_t bz)
{
    vf.walk([&](ArchLaneIdxOp op) {
        if (!op.getResult().use_empty()) {
            OpBuilder b(op);
            op.getResult().replaceAllUsesWith(buildLaneIdx(b, op.getLoc(), bx, by, bz));
        }
        op.erase();
    });
    vf.walk([&](ArchLanemaskOp op) {
        if (!op.getResult().use_empty()) {
            OpBuilder b(op);
            Value lane = buildLaneIdx(b, op.getLoc(), bx, by, bz);
            op.getResult().replaceAllUsesWith(buildLanemask(b, op.getLoc(), op.getMode(), lane));
        }
        op.erase();
    });
    vf.walk([&](ArchWarpReduceOp op) {
        OpBuilder b(op);
        op.getResult().replaceAllUsesWith(
            buildWarpReduce(b, op.getLoc(), op.getKind(), op.getIsUnsigned(), op.getValue()));
        op.erase();
    });
    RewritePatternSet patterns(vf.getContext());
    patterns.add<LowerArchActivemaskPattern>(vf.getContext(), bx, by, bz);
    // The outlined function is outside the original function's greedy driver.
    // Apply only the lowering here, leaving folding and control-flow cleanup to
    // the surrounding pipeline.
    GreedyRewriteConfig config;
    config.fold = false;
    config.cseConstants = false;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
    return applyPatternsGreedily(vf, std::move(patterns), config);
}
} // namespace tla
