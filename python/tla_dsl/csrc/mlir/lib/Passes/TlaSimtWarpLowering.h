#pragma once

#include <cstdint>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace tla {
mlir::LogicalResult lowerSimtWarpOpsIn(mlir::func::FuncOp vf, int64_t bx, int64_t by, int64_t bz);
} // namespace tla
