#pragma once

#include "Dialect/Tla/IR/TlaAttrs.h"
#include "Dialect/Tla/IR/TlaDialect.h"
#include "Dialect/Tla/IR/TlaTypes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "tla/Ops.h.inc"

namespace tla {
// Resolve the storage behind tensor_ptr without applying a view's coordinates.
mlir::FailureOr<mlir::Value> resolveTensorBacking(mlir::Value tensor);
} // namespace tla
