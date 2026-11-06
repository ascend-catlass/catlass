#pragma once

#include "Dialect/Tla/IR/TlaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace tla {
mlir::LogicalResult lowerSimtPrints(mlir::func::FuncOp function, mlir::Value workspace);
mlir::BlockArgument getOrAppendSimtPrintWorkspaceArg(mlir::func::FuncOp function);
// Reserve formats before the UB scratch global and dynamic base are finalized.
mlir::FailureOr<uint64_t> reserveVfPrintFormats(mlir::ModuleOp module, uint64_t staticBytes);
mlir::BlockArgument getOrAppendDebugPrintWorkspaceArg(mlir::func::FuncOp function);
mlir::LogicalResult lowerVfScalarPrint(mlir::OpBuilder& builder, DebugPrintOp op, mlir::ValueRange values);
void emitVfDebugBoundary(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::ModuleOp module, mlir::Value workspace, bool begin);
} // namespace tla
