#include "TlaSimtWarpLowering.h"

#include "Dialect/Tla/IR/TlaOps.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

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
    return success();
}
} // namespace tla
