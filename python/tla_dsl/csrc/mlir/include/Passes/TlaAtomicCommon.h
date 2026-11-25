//===- TlaAtomicCommon.h - shared atomic intrinsic naming -----*- C++ -*-===//
//
// Helpers shared by the SIMD atomic lowering. The atomic intrinsics are named
// ``llvm.hivm.atom.<ADD|EXCH>.<G|S>.<token>``; the token depends on the
// element type and, for EXCH, on the signedness the instruction accepts.
//
//===----------------------------------------------------------------------===//

#ifndef TLA_PASSES_TLAATOMICCOMMON_H
#define TLA_PASSES_TLAATOMICCOMMON_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <optional>
#include <string>

namespace tla {

enum class AtomicOpKind
{
    Add,
    Exch,
};

// Element type token of the intrinsics. ADD maps both signednesses to the
// signed instruction (the bit pattern is identical); EXCH only has the
// unsigned forms. Returns std::nullopt for an unsupported element type.
inline std::optional<llvm::StringRef> atomicTypeToken(mlir::Type elementType, AtomicOpKind kind)
{
    auto intType = llvm::dyn_cast<mlir::IntegerType>(elementType);
    if (kind == AtomicOpKind::Exch) {
        if (!intType || !intType.isUnsigned())
            return std::nullopt;
        if (intType.getWidth() == 32)
            return llvm::StringRef("u32");
        if (intType.getWidth() == 64)
            return llvm::StringRef("u64");
        return std::nullopt;
    }
    if (intType) {
        if (intType.getWidth() == 32)
            return llvm::StringRef("s32");
        if (intType.getWidth() == 64)
            return llvm::StringRef("s64");
        return std::nullopt;
    }
    if (elementType.isF32())
        return llvm::StringRef("fp32");
    return std::nullopt;
}

// Full intrinsic name. Returns an empty string for an unsupported element
// type; callers report the type in their own diagnostic.
inline std::string atomicIntrinsicName(llvm::StringRef op, bool gm, mlir::Type elementType, AtomicOpKind kind)
{
    std::optional<llvm::StringRef> token = atomicTypeToken(elementType, kind);
    if (!token)
        return {};
    return (llvm::Twine("llvm.hivm.atom.") + op + "." + (gm ? "G" : "S") + "." + *token).str();
}

} // namespace tla

#endif // TLA_PASSES_TLAATOMICCOMMON_H
