#pragma once

// Producer-side descriptor derivation owned exclusively by
// `tla-lower-tensor-desc`.

#include "Passes/TlaTensorDescriptor.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"

#include <cstdint>
#include <limits>

namespace tla {

class IndexConstantCache {
public:
    mlir::Value get(mlir::Operation* anchor, int64_t value, unsigned bits = 0);

private:
    struct Key {
        int64_t value;
        unsigned bits;

        bool operator==(const Key& other) const
        {
            return value == other.value && bits == other.bits;
        }
    };

    struct KeyInfo {
        static inline Key getEmptyKey()
        {
            return {std::numeric_limits<int64_t>::min(), 0};
        }
        static inline Key getTombstoneKey()
        {
            return {std::numeric_limits<int64_t>::min() + 1, 0};
        }
        static unsigned getHashValue(const Key& key)
        {
            return llvm::hash_combine(key.value, key.bits);
        }
        static bool isEqual(const Key& lhs, const Key& rhs)
        {
            return lhs == rhs;
        }
    };

    llvm::DenseMap<mlir::Block*, llvm::DenseMap<Key, mlir::Value, KeyInfo>> byScope;
};

/// Stateful producer-chain analysis shared by `tla-lower-tensor-desc`, which
/// materializes the descriptors, and `tla-lower-ptr`, which needs a view's
/// coordinate/stride before those descriptors exist.
class TensorDescriptorDerivation {
public:
    mlir::LogicalResult derive(mlir::func::FuncOp funcOp);

    llvm::DenseMap<mlir::Value, TensorDescriptor> descriptorByValue;

private:
    IndexConstantCache constants;
};

} // namespace tla
