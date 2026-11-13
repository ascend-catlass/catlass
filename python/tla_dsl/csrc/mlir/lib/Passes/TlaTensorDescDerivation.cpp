#include "Passes/TlaTensorDescDerivation.h"

#include "PassesCommon.h"

#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <array>

namespace tla {

using ConstantFactory = llvm::function_ref<mlir::Value(mlir::Operation* anchor, int64_t value, unsigned bits)>;

mlir::Value IndexConstantCache::get(mlir::Operation* anchor, int64_t value, unsigned bits)
{
    Key key{value, bits};
    Block* scopeBlock = nullptr;
    if (auto tlaFunc = anchor->getParentOfType<::tla::FuncOp>()) {
        scopeBlock = &tlaFunc.getBody().front();
    } else if (auto func = anchor->getParentOfType<mlir::func::FuncOp>()) {
        scopeBlock = &func.getBody().front();
    } else if (auto module = anchor->getParentOfType<ModuleOp>()) {
        scopeBlock = &module.getBodyRegion().front();
    } else {
        scopeBlock = anchor->getBlock();
    }
    auto& cache = byScope[scopeBlock];
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    OpBuilder builder(scopeBlock, scopeBlock->begin());
    Value constant;
    if (bits == 0) {
        constant = builder.create<arith::ConstantIndexOp>(anchor->getLoc(), value);
    } else {
        auto intType = builder.getIntegerType(bits);
        auto intAttr = builder.getIntegerAttr(intType, value);
        constant = builder.create<arith::ConstantOp>(anchor->getLoc(), intType, intAttr);
    }
    cache[key] = constant;
    return constant;
}

static bool isConstantOne(mlir::Value value)
{
    auto constant = mlir::getConstantIntValue(value);
    return constant && *constant == 1;
}

// Look through operations that leave a value unchanged, so that two
// expressions for the same quantity compare equal. A tile offset typically
// arrives as index_cast(muli(%iv, 1)) and the loop bound as index_cast(%ub);
// after unwrapping, they are %iv and %ub.
//
// Only multiplication by 1 and casts to index are looked through. A cast is
// kept when its input is wider than index, because it may truncate the value.
// The index width comes from the enclosing data layout.
static mlir::Value unwrapValuePreservingOps(mlir::Value value)
{
    while (value) {
        if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
            if (!mlir::isa<mlir::IndexType>(cast.getType()) || mlir::isa<mlir::IndexType>(cast.getIn().getType()))
                break;
            auto indexBitwidth = mlir::DataLayout::closest(cast).getTypeSizeInBits(cast.getType()).getFixedValue();
            if (cast.getIn().getType().getIntOrFloatBitWidth() > indexBitwidth)
                break;
            value = cast.getIn();
            continue;
        }
        if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
            if (isConstantOne(mul.getRhs())) {
                value = mul.getLhs();
                continue;
            }
            if (isConstantOne(mul.getLhs())) {
                value = mul.getRhs();
                continue;
            }
        }
        break;
    }
    return value;
}

// Return the scf.for enclosing `op` whose induction variable is `inductionVar`,
// or null if there is none.
static mlir::scf::ForOp findEnclosingLoopCountingWith(mlir::Operation* op, mlir::BlockArgument inductionVar)
{
    for (mlir::Operation* parent = op->getParentOp(); parent; parent = parent->getParentOp()) {
        auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(parent);
        if (forOp && forOp.getInductionVar() == inductionVar)
            return forOp;
    }
    return {};
}

// With a positive constant step, `inductionVar < upperBound` holds on every
// iteration.
static bool countsUpwards(mlir::scf::ForOp forOp)
{
    auto step = mlir::getConstantIntValue(forOp.getStep());
    return step && *step > 0;
}

// Return true if `extent - offset >= 1` is guaranteed at `op`. This holds when
// `offset` is the induction variable of an enclosing loop that counts up to
// `extent`. In every other case, return false.
static bool subtractingOffsetLeavesAtLeastOneElement(mlir::Operation* op, mlir::Value extent, mlir::Value offset)
{
    // Step 1: the offset must be a loop induction variable.
    mlir::Value offsetRoot = unwrapValuePreservingOps(offset);
    if (!offsetRoot)
        return false;
    auto inductionVar = mlir::dyn_cast<mlir::BlockArgument>(offsetRoot);
    if (!inductionVar)
        return false;
    // Step 2: that loop must enclose `op` and count upwards, so that
    // offset < upperBound inside it.
    mlir::scf::ForOp forOp = findEnclosingLoopCountingWith(op, inductionVar);
    if (!forOp || !countsUpwards(forOp))
        return false;
    // Step 3: the upper bound must be the extent itself, so that
    // extent - offset >= 1.
    mlir::Value extentRoot = unwrapValuePreservingOps(extent);
    return extentRoot && unwrapValuePreservingOps(forOp.getUpperBound()) == extentRoot;
}

// Return the extent of one tile dimension after cropping it to the parent
// tensor: `min(requested, parentExtent - offset)`.
//
// The minimum is skipped when the view is one element wide and an enclosing
// loop proves that this element lies inside the parent. Skipping it saves
// instructions in two places:
//   - The scalar unit has no signed minimum, so a run-time `arith.minsi` is
//     lowered to a chain of selects.
//   - The extent stays a constant, so getTlaTensorValidLaneCount can fold the
//     lane-count multiply.
static mlir::Value buildCroppedExtent(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Operation* op, mlir::Value requested, mlir::Value parentExtent,
    mlir::Value offset, mlir::Value remaining)
{
    if (isConstantOne(requested) && subtractingOffsetLeavesAtLeastOneElement(op, parentExtent, offset))
        return requested;
    return builder.createOrFold<arith::MinSIOp>(loc, requested, remaining);
}

static mlir::FailureOr<TensorDescriptor> buildTileViewResultDescriptorFromParent(
    mlir::Operation* op, mlir::Value base, mlir::MemRefType bridgedBaseType, const TensorTypeInfo& info,
    const TensorDescriptor& parent, mlir::Value row, mlir::Value col, mlir::Value sh0, mlir::Value sh1,
    ConstantFactory getConstant)
{
    Location loc = op->getLoc();
    OpBuilder b(op);

    if (!isNZFamilyLayout(info.layoutTag) && !isLinearLayout(info.layoutTag)) {
        op->emitError() << "tile_view: unsupported layout tag for descriptor lowering";
        return failure();
    }

    // createOrFold so static absolute coord / origin arithmetic folds to a constant
    // in place (at the tile_view) instead of leaving an arith op for downstream
    // passes to clone -- tla-lower-tensor-desc is the sole descriptor producer,
    // and the vector helper consumes these operands directly.
    Value abs0 = b.createOrFold<arith::AddIOp>(loc, parent.coord[0], row);
    Value abs1 = b.createOrFold<arith::AddIOp>(loc, parent.coord[1], col);
    Value rest0 = b.createOrFold<arith::SubIOp>(loc, parent.originShape[0], row);
    Value rest1 = b.createOrFold<arith::SubIOp>(loc, parent.originShape[1], col);
    Value origin0 = buildCroppedExtent(b, loc, op, sh0, parent.originShape[0], row, rest0);
    Value origin1 = buildCroppedExtent(b, loc, op, sh1, parent.originShape[1], col, rest1);

    Value one = getConstant(op, 1, 0);
    std::array<Value, 4> shape;
    std::array<Value, 4> stride;

    if (isLinearLayout(info.layoutTag)) {
        auto materializeRowMajorStride = [&](int64_t dim, Value parentStride) -> FailureOr<Value> {
            if (dim == ShapedType::kDynamic) {
                if (!parentStride || !parentStride.getType().isIndex()) {
                    op->emitError() << "tile_view: dynamic stride requires parent tile descriptor "
                                       "stride (index SSA)";
                    return failure();
                }
                return parentStride;
            }
            return getConstant(op, dim, 0);
        };
        FailureOr<Value> st0 = materializeRowMajorStride(info.strideDims[0], parent.stride[0]);
        FailureOr<Value> st1 = materializeRowMajorStride(info.strideDims[1], parent.stride[1]);
        if (failed(st0) || failed(st1))
            return failure();
        shape = {
            info.shapeDims[0] == ShapedType::kDynamic ? sh0 : getConstant(op, info.shapeDims[0], 0),
            info.shapeDims[1] == ShapedType::kDynamic ? sh1 : getConstant(op, info.shapeDims[1], 0), one, one};
        stride = {*st0, *st1, one, one};
    } else {
        auto ceilDivIndexByPositiveConst = [&](Value numerator, int64_t divisor) -> FailureOr<Value> {
            if (divisor <= 0) {
                op->emitError() << "tile_view: NZFamily layout shape dynamic leaf requires positive divisor, got "
                                << divisor;
                return failure();
            }
            Value divisorV = getConstant(op, divisor, 0);
            Value adjusted =
                b.createOrFold<arith::AddIOp>(loc, numerator, b.createOrFold<arith::SubIOp>(loc, divisorV, one));
            return b.createOrFold<arith::DivSIOp>(loc, adjusted, divisorV);
        };
        auto materializeNZFamilyShapeLeaf = [&](size_t idx) -> FailureOr<Value> {
            int64_t leaf = info.shapeDims[idx];
            if (leaf != ShapedType::kDynamic)
                return getConstant(op, leaf, 0);
            if (info.shapeDims.size() < 4) {
                op->emitError() << "tile_view: NZFamily layout shape must have 4 leaves";
                return failure();
            }
            // zNUnAlign: shape[0] = tile_M = sh0. The M axis is not fractal-blocked, so leaf[0]
            // is the runtime tile row count (zN's leaf[0] is the static C0_NUM_PER_FRACTAL).
            if (info.layoutTag == LayoutTag::zNUnAlign && idx == 0) {
                return sh0;
            }
            if (idx == 1) {
                if (info.shapeDims[0] == ShapedType::kDynamic) {
                    op->emitError()
                        << "tile_view: dynamic NZFamily layout shape leaf index 1 requires static leaf index 0";
                    return failure();
                }
                return ceilDivIndexByPositiveConst(sh0, info.shapeDims[0]);
            }
            if (idx == 3) {
                if (info.shapeDims[2] == ShapedType::kDynamic) {
                    op->emitError()
                        << "tile_view: dynamic NZFamily layout shape leaf index 3 requires static leaf index 2";
                    return failure();
                }
                return ceilDivIndexByPositiveConst(sh1, info.shapeDims[2]);
            }
            op->emitError() << "tile_view: dynamic NZFamily layout shape leaf at index " << idx
                            << " is unsupported; only indices 1 and 3 may be dynamic";
            return failure();
        };
        auto materializeNZFamilyStrideLeaf = [&](size_t idx) -> FailureOr<Value> {
            int64_t leaf = info.strideDims[idx];
            if (leaf != ShapedType::kDynamic)
                return getConstant(op, leaf, 0);
            if (parent.stride[idx] && parent.stride[idx].getType().isIndex())
                return parent.stride[idx];
            op->emitError() << "tile_view: dynamic NZFamily layout stride leaf index " << idx
                            << " requires parent stride SSA";
            return failure();
        };
        for (size_t i = 0; i < shape.size(); ++i) {
            FailureOr<Value> leaf = materializeNZFamilyShapeLeaf(i);
            if (failed(leaf))
                return failure();
            shape[i] = *leaf;
        }
        for (size_t i = 0; i < stride.size(); ++i) {
            FailureOr<Value> leaf = materializeNZFamilyStrideLeaf(i);
            if (failed(leaf))
                return failure();
            stride[i] = *leaf;
        }
    }

    TensorDescriptor desc;
    desc.base = base;
    desc.bridgedBaseMemrefType = bridgedBaseType;
    desc.shape = shape;
    desc.stride = stride;
    desc.originShape = {origin0, origin1};
    desc.coord = {abs0, abs1};
    desc.layoutTag = info.layoutTag;
    desc.addrspace = info.addressSpace;
    desc.elementType = info.elementType;
    return desc;
}

// The storage type a logical element type is buffered as, plus the element count
// in that storage type.
//
// Sub-byte elements are the only case where these differ: a packed fp4 tile is
// logically i4, but buffering it as an i4 memref would silently redenominate the
// memref's offset and strides in half-bytes, while the bc layer advances a
// byte-typed pointer by them. So the *storage* stays i8 with half as many
// elements, and everything byte-denominated downstream keeps meaning bytes. The
// logical i4 survives on the descriptor's elementType, which is where the C0
// geometry reads its width from.
static std::pair<mlir::Type, int64_t> storageTypeAndCount(mlir::Type elementType, int64_t elementCount)
{
    int64_t bits = getBitSizeOfFixedWidthScalarType(elementType);
    mlir::Type storage = mlir::IntegerType::get(elementType.getContext(), 8);
    // The dialect's own cube element formats (f4e2m1 / f4e1m2 / f8e8m0) exist
    // only to carry the format; they have no LLVM lowering, so a memref must
    // never be built over one. A byte-wide one (e8m0) is simply i8; a sub-byte
    // one additionally packs, below.
    if (isTlaCustomElementType(elementType) && bits % 8 == 0)
        return {storage, elementCount};
    // An unknown type answers 0 bits, which this catches too: a memref over it
    // keeps whatever element type it already had.
    if (bits % 8 == 0)
        return {elementType, elementCount};
    const int64_t perByte = 8 / bits;
    if (elementCount == mlir::ShapedType::kDynamic)
        return {storage, elementCount};
    return {storage, (elementCount + perByte - 1) / perByte};
}

mlir::LogicalResult TensorDescriptorDerivation::derive(mlir::func::FuncOp funcOp)
{
    bool derivationFailed = false;
    auto& tensorDescriptorByValue = descriptorByValue;
    tensorDescriptorByValue.clear();
    auto isTensorScfCarrier = [](Value value) {
        if (Operation* def = value.getDefiningOp())
            return isa<scf::IfOp, scf::ForOp, scf::WhileOp>(def);

        auto blockArg = dyn_cast<BlockArgument>(value);
        if (!blockArg)
            return false;
        Operation* parent = blockArg.getOwner()->getParentOp();
        if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent))
            return blockArg.getOwner() == forOp.getBody() && blockArg.getArgNumber() > 0;
        if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(parent))
            return blockArg.getOwner() == whileOp.getBeforeBody() || blockArg.getOwner() == whileOp.getAfterBody();
        return false;
    };

    llvm::DenseSet<Value> scfDependentTensorValues;
    auto deferScfDependentProducer = [&](Value result, Value source) {
        if (!isTensorScfCarrier(source) && !scfDependentTensorValues.contains(source))
            return false;
        scfDependentTensorValues.insert(result);
        return true;
    };
    auto getOrCreateConstant = [&](Operation* anchor, int64_t value, unsigned bits) -> Value {
        return constants.get(anchor, value, bits);
    };

    auto materializeIndexPair = [&](Operation* op, Value aggregateValue,
                                    StringRef kind) -> FailureOr<std::array<Value, 2>> {
        auto emitError = [&](Twine message) -> FailureOr<std::array<Value, 2>> {
            op->emitError() << message;
            return failure();
        };

        SmallVector<int64_t, 2> leaves;
        if (kind == "shape") {
            auto shapeTy = dyn_cast<::tla::ShapeType>(aggregateValue.getType());
            if (!shapeTy || failed(::tla::getTlaIndexTreeLeaves(shapeTy.getTree(), leaves))) {
                return emitError("expected flat rank-2 tla.shape operand");
            }
            if (shapeTy.getTree().size() == 1 && leaves.size() == 1) {
                leaves = {1, leaves[0]};
            } else if (shapeTy.getTree().size() != 2) {
                return emitError("expected flat rank-2 tla.shape operand");
            }
        } else {
            auto coordTy = dyn_cast<::tla::CoordType>(aggregateValue.getType());
            if (!coordTy || failed(::tla::getTlaIndexTreeLeaves(coordTy.getTree(), leaves))) {
                return emitError("expected flat rank-2 tla.coord operand");
            }
            if (coordTy.getTree().size() == 1 && leaves.size() == 1) {
                leaves = {0, leaves[0]};
            } else if (coordTy.getTree().size() != 2) {
                return emitError("expected flat rank-2 tla.coord operand");
            }
        }
        if (leaves.size() != 2) {
            return emitError(Twine("tla.") + kind + " descriptor requires exactly 2 elements");
        }

        SmallVector<Value, 2> dynamicValues;
        if (llvm::any_of(leaves, [](int64_t leaf) { return leaf == ShapedType::kDynamic; })) {
            if (kind == "shape") {
                auto makeShape = aggregateValue.getDefiningOp<::tla::MakeShapeOp>();
                if (!makeShape) {
                    return emitError("dynamic tla.shape operands must come from tla.make_shape");
                }
                dynamicValues.append(makeShape.getDynElems().begin(), makeShape.getDynElems().end());
            } else {
                auto makeCoord = aggregateValue.getDefiningOp<::tla::MakeCoordOp>();
                if (!makeCoord) {
                    return emitError("dynamic tla.coord operands must come from tla.make_coord");
                }
                dynamicValues.append(makeCoord.getDynElems().begin(), makeCoord.getDynElems().end());
            }
        }

        std::array<Value, 2> result{};
        size_t dynamicIndex = 0;
        for (auto [index, leaf] : llvm::enumerate(leaves)) {
            if (leaf == ShapedType::kDynamic) {
                if (dynamicIndex >= dynamicValues.size()) {
                    return emitError(Twine("tla.") + kind + " type/operand dynamic element count mismatch");
                }
                Value dynamicValue = dynamicValues[dynamicIndex++];
                if (!dynamicValue.getType().isIndex()) {
                    return emitError(Twine("tla.") + kind + " dynamic operands must be index type");
                }
                result[index] = dynamicValue;
                continue;
            }

            result[index] = getOrCreateConstant(op, leaf, 0);
        }

        if (dynamicIndex != dynamicValues.size()) {
            return emitError(Twine("tla.") + kind + " type/operand dynamic element count mismatch");
        }
        return result;
    };

    auto unpackTileOffsetsAndShape = [&](Operation* op) -> FailureOr<std::array<Value, 4>> {
        Value row;
        Value col;
        Value shape0;
        Value shape1;
        if (op->getNumOperands() == 5) {
            row = op->getOperand(1);
            col = op->getOperand(2);
            shape0 = op->getOperand(3);
            shape1 = op->getOperand(4);
        } else {
            auto shapePair = materializeIndexPair(op, op->getOperand(1), "shape");
            if (failed(shapePair))
                return failure();
            shape0 = (*shapePair)[0];
            shape1 = (*shapePair)[1];
            // Prefer the result ``!tla.tensor`` coord segment over ``!tla.coord`` printing:
            // the tensor metadata (e.g. ``0,?`` for outer B along N) stays correct even when
            // the packed coord type string loses ``?`` and would mis-bind dynamic operands.
            if (auto tileOp = dyn_cast<::tla::TileViewOp>(op)) {
                if (auto resTlaTy = dyn_cast<::tla::TlaTensorType>(tileOp.getResult().getType())) {
                    SmallVector<int64_t, 4> coordLeaves;
                    if (succeeded(::tla::getTlaIndexTreeLeaves(resTlaTy.getCoord().getTree(), coordLeaves)) &&
                        coordLeaves.size() == 2) {
                        auto mc = tileOp.getCoord().getDefiningOp<::tla::MakeCoordOp>();
                        if (mc) {
                            unsigned dynInTensor = 0;
                            for (int64_t l : coordLeaves)
                                if (l == ShapedType::kDynamic)
                                    ++dynInTensor;
                            if (dynInTensor == mc.getNumOperands()) {
                                size_t di = 0;
                                row = coordLeaves[0] == ShapedType::kDynamic ?
                                          mc.getDynElems()[di++] :
                                          getOrCreateConstant(op, coordLeaves[0], 0);
                                col = coordLeaves[1] == ShapedType::kDynamic ?
                                          mc.getDynElems()[di++] :
                                          getOrCreateConstant(op, coordLeaves[1], 0);
                                return std::array<Value, 4>{row, col, shape0, shape1};
                            }
                        }
                    }
                }
            }
            auto coordPair = materializeIndexPair(op, op->getOperand(2), "coord");
            if (failed(coordPair))
                return failure();
            row = (*coordPair)[0];
            col = (*coordPair)[1];
        }
        return std::array<Value, 4>{row, col, shape0, shape1};
    };

    // Build ``tile_view`` result descriptors for ``!tla.tensor`` sources from operand SSA
    // (shape/coord packs or explicit index operands). Linear layouts
    // (RowMajor/ColumnMajor) may use dynamic shape in the type (``?``) filled from ``sh0/sh1``
    // operands, and dynamic stride (``?``) taken from the parent tile descriptor stride SSA.
    // NZFamily layouts may also carry dynamic leaves when they can be derived from explicit
    // shape operands or inherited from parent descriptors. Absolute coord and cropped origin
    // follow TLA
    // ``TileViewImpl``: ``coord = parent.coord + tileCoord`` and
    // ``origin_i = min(tileShape_i, parent.origin_i - tileCoord_i)``.
    // The producer-local descriptor builder takes the constant factory
    // explicitly.
    auto buildTileViewResultDescriptorFromParent =
        [&](Operation* op, Value base, MemRefType bridgedBaseType, const TensorTypeInfo& info,
            const TensorDescriptor& parent, Value row, Value col, Value sh0, Value sh1) -> FailureOr<TensorDescriptor> {
        return ::tla::buildTileViewResultDescriptorFromParent(
            op, base, bridgedBaseType, info, parent, row, col, sh0, sh1, getOrCreateConstant);
    };

    // Derive descriptors for tile-producing ops in SSA order: mixed
    // tla.tile_view/tla.make_tensor_like chains rely on producer descriptors
    // being available when their users are visited.
    funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
        // This method is producer-side and is called repeatedly while
        // tla-lower-tensor-desc rewrites structural SCF carriers. Descriptors
        // reconstructed at SCF region entries seed producer chains in that region.
        if (auto descOp = llvm::dyn_cast<::tla::TensorDescOp>(op)) {
            auto desc = descriptorFromTensorDescOp(descOp);
            if (failed(desc)) {
                derivationFailed = true;
                return;
            }
            tensorDescriptorByValue[descOp.getResult()] = *desc;
            return;
        }
        if (auto tileOp = llvm::dyn_cast<::tla::TileViewOp>(op)) {
            if ((op->getNumOperands() != 5 && op->getNumOperands() != 3) || op->getNumResults() != 1) {
                op->emitError() << "expected tla.tile_view to have exactly 3 or 5 operands and 1 result";
                derivationFailed = true;
                return;
            }

            auto rowColShape = unpackTileOffsetsAndShape(op);
            if (failed(rowColShape)) {
                derivationFailed = true;
                return;
            }
            Value row = (*rowColShape)[0];
            Value col = (*rowColShape)[1];
            Value shape0 = (*rowColShape)[2];
            Value shape1 = (*rowColShape)[3];
            if (!row.getType().isIndex() || !col.getType().isIndex() || !shape0.getType().isIndex() ||
                !shape1.getType().isIndex()) {
                op->emitError() << "tla.tile_view row/col/shape operands must be index type";
                derivationFailed = true;
                return;
            }

            auto resultInfo = decodeTensorTypeInfo(tileOp.getResult().getType());
            if (failed(resultInfo)) {
                op->emitError() << "tla.tile_view currently requires a structured tla.tensor result "
                                   "type";
                derivationFailed = true;
                return;
            }
            if (resultInfo->rank != 2) {
                op->emitError() << "tla.tile_view descriptor supports only normalized rank-2 tiles";
                derivationFailed = true;
                return;
            }

            Value source = tileOp.getSource();
            auto parentIt = tensorDescriptorByValue.find(source);
            if (parentIt == tensorDescriptorByValue.end()) {
                if (deferScfDependentProducer(tileOp.getResult(), source))
                    return;
                op->emitError() << "missing descriptor for tla.tile_view source tile; expected "
                                   "a materialized tensor producer or structural SCF carrier";
                derivationFailed = true;
                return;
            }
            const TensorDescriptor& parent = parentIt->second;
            if (!validateTensorDescriptor(
                    op, parent, "malformed parent tensor descriptor for tla.tile_view source tile")) {
                derivationFailed = true;
                return;
            }
            if (resultInfo->addressSpace != parent.addrspace || resultInfo->elementType != parent.elementType) {
                op->emitError() << "tla.tile_view result tile metadata must match parent descriptor "
                                   "(element type/addrspace) when source is a tile";
                derivationFailed = true;
                return;
            }

            if (auto layoutTagAttr = op->getAttrOfType<::tla::LayoutTagAttr>("layoutTag")) {
                if (layoutTagAttr.getValue() != resultInfo->layoutTag) {
                    op->emitError() << "tla.tile_view layoutTag must match result tensor layout_tag";
                    derivationFailed = true;
                    return;
                }
            }

            auto bridgedParent = dyn_cast<MemRefType>(parent.bridgedBaseMemrefType);
            if (!bridgedParent) {
                op->emitError() << "tla.tile_view parent descriptor missing bridged memref type";
                derivationFailed = true;
                return;
            }
            FailureOr<TensorDescriptor> desc = buildTileViewResultDescriptorFromParent(
                op, parent.base, bridgedParent, *resultInfo, parent, row, col, shape0, shape1);
            if (failed(desc)) {
                derivationFailed = true;
                return;
            }
            tensorDescriptorByValue[tileOp.getResult()] = *desc;
            return;
        }

        if (llvm::isa<::tla::MakeTensorLikeOp>(op)) {
            if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
                op->emitError() << "expected tla.make_tensor_like to have exactly 2 operands and 1 result";
                derivationFailed = true;
                return;
            }

            Value ptrValue = op->getOperand(0);
            if (!llvm::isa<::tla::PtrType>(ptrValue.getType())) {
                op->emitError() << "tla.make_tensor_like pointer operand must be !tla.ptr";
                derivationFailed = true;
                return;
            }

            Value likeTile = op->getOperand(1);
            auto parentIt = tensorDescriptorByValue.find(likeTile);
            if (parentIt == tensorDescriptorByValue.end()) {
                if (deferScfDependentProducer(op->getResult(0), likeTile))
                    return;
                op->emitError() << "missing descriptor for tla.make_tensor_like reference tile; "
                                   "expected a materialized tensor producer or structural SCF carrier";
                derivationFailed = true;
                return;
            }
            const TensorDescriptor& parent = parentIt->second;
            if (!validateTensorDescriptor(
                    op, parent, "malformed parent tensor descriptor for tla.make_tensor_like reference tile")) {
                derivationFailed = true;
                return;
            }

            auto childInfo = decodeTensorTypeInfo(op->getResult(0).getType());
            if (failed(childInfo)) {
                op->emitError() << "tla.make_tensor_like currently requires a structured tla.tensor "
                                   "result type";
                derivationFailed = true;
                return;
            }
            int64_t flatElemCount = ShapedType::kDynamic;
            if (auto n = getStaticAllocationElementCount(ptrValue); succeeded(n) && *n > 0) {
                flatElemCount = *n;
            } else if (
                childInfo->originShapeDims.size() >= 2 && childInfo->originShapeDims[0] != ShapedType::kDynamic &&
                childInfo->originShapeDims[1] != ShapedType::kDynamic) {
                int64_t dim0 = childInfo->originShapeDims[0];
                int64_t dim1 = childInfo->originShapeDims[1];
                if (dim0 > 0 && dim1 > 0)
                    flatElemCount = dim0 * dim1;
            }
            auto [storageElemType, storageElemCount] = storageTypeAndCount(childInfo->elementType, flatElemCount);
            auto bridgedBaseType =
                buildHivmMemrefType(op->getContext(), {storageElemCount}, storageElemType, childInfo->tlaAddressSpace);
            if (failed(bridgedBaseType)) {
                op->emitError() << "tla.make_tensor_like buffer memref must be bridgeable to builtin memref type";
                derivationFailed = true;
                return;
            }

            OpBuilder builder(op);
            auto layoutTagAttr = op->getAttrOfType<::tla::LayoutTagAttr>("layoutTag");
            if (!layoutTagAttr) {
                op->emitError() << "tla.make_tensor_like requires a layoutTag attribute";
                derivationFailed = true;
                return;
            }
            if (layoutTagAttr.getValue() != childInfo->layoutTag) {
                op->emitError() << "tla.make_tensor_like layoutTag must match result tensor layout_tag";
                derivationFailed = true;
                return;
            }
            Value typedBuffer = ptrValue;
            auto materializeLeafFromTypeOrParent = [&](int64_t leaf, Value parentValue,
                                                       StringRef fieldName) -> FailureOr<Value> {
                if (leaf == ShapedType::kDynamic) {
                    if (parentValue && parentValue.getType().isIndex())
                        return parentValue;
                    op->emitError() << "dynamic tensor metadata leaf in " << fieldName
                                    << " is not supported for tla.make_tensor_like without parent SSA";
                    return failure();
                }
                return getOrCreateConstant(op, leaf, 0);
            };
            auto ceilDivIndexByPositiveConst = [&](Value numerator, int64_t divisor) -> FailureOr<Value> {
                if (divisor <= 0) {
                    op->emitError() << "NZFamily layout shape dynamic leaf requires positive divisor, got " << divisor;
                    return failure();
                }
                Value divisorV = getOrCreateConstant(op, divisor, 0);
                Value one = getOrCreateConstant(op, 1, 0);
                Value adjusted = builder.create<arith::AddIOp>(
                    op->getLoc(), numerator, builder.create<arith::SubIOp>(op->getLoc(), divisorV, one));
                return builder.create<arith::DivSIOp>(op->getLoc(), adjusted, divisorV).getResult();
            };
            auto materializeNZFamilyShapeDynamicLeafFromOrigin = [&](ArrayRef<int64_t> leaves,
                                                                     size_t idx) -> FailureOr<Value> {
                // NZFamily layout shape trees flatten as (m0,m1),(n0,n1).
                // For zN/nZ/L0C dynamic logical extents live in m1 / n1:
                //   m1 <- ceil_div(origin0, m0), n1 <- ceil_div(origin1, n0).
                // m0 / n0 are layout constants (tile fractal factors), not runtime-varying.
                if (!isNZFamilyLayout(childInfo->layoutTag)) {
                    op->emitError() << "dynamic NZFamily layout shape leaf at index " << idx
                                    << " has no SSA derivation rule for layout "
                                    << stringifyLayoutTag(childInfo->layoutTag);
                    return failure();
                }
                if (leaves.size() < 4) {
                    op->emitError() << "NZFamily layout shape must have 4 leaves for layout "
                                    << stringifyLayoutTag(childInfo->layoutTag);
                    return failure();
                }
                // zNUnAlign: shape[0] = rows = origin M. The M axis is not fractal-blocked,
                // so leaf[0] is the runtime row count (not a compile-time divisor like zN's
                // C0_NUM_PER_FRACTAL) and is derived directly from the logical origin M.
                if (childInfo->layoutTag == LayoutTag::zNUnAlign && idx == 0) {
                    return parent.originShape[0];
                }
                if (idx == 1) {
                    if (leaves[0] == ShapedType::kDynamic) {
                        op->emitError()
                            << "dynamic NZFamily layout shape leaf index 1 requires static divisor leaf index 0";
                        return failure();
                    }
                    return ceilDivIndexByPositiveConst(parent.originShape[0], leaves[0]);
                }
                if (idx == 3) {
                    if (leaves[2] == ShapedType::kDynamic) {
                        op->emitError()
                            << "dynamic NZFamily layout shape leaf index 3 requires static divisor leaf index 2";
                        return failure();
                    }
                    return ceilDivIndexByPositiveConst(parent.originShape[1], leaves[2]);
                }
                op->emitError() << "dynamic NZFamily layout shape leaf at index " << idx
                                << " is unsupported; only indices 1 and 3 may be dynamic";
                return failure();
            };
            auto materializeNZFamilyLeafFromTypeOrParent = [&](ArrayRef<int64_t> leaves, ArrayRef<Value> parentLeaves,
                                                               size_t idx, StringRef fieldName) -> FailureOr<Value> {
                int64_t leaf = leaves[idx];
                if (leaf == ShapedType::kDynamic) {
                    if (fieldName == "NZFamily layout shape") {
                        FailureOr<Value> derived = materializeNZFamilyShapeDynamicLeafFromOrigin(leaves, idx);
                        if (succeeded(derived))
                            return *derived;
                    }
                    if (idx < parentLeaves.size() && parentLeaves[idx] && parentLeaves[idx].getType().isIndex())
                        return parentLeaves[idx];
                    op->emitError() << "dynamic tensor metadata leaf in " << fieldName
                                    << " is not supported for tla.make_tensor_like without parent SSA";
                    return failure();
                }
                return getOrCreateConstant(op, leaf, 0);
            };
            auto deriveNZFamilyStrideLeafFromShape = [&](ArrayRef<Value> shapeLeaves, size_t idx) -> FailureOr<Value> {
                auto mulShapeLeaves = [&](size_t a, size_t b, size_t c) -> FailureOr<Value> {
                    if (shapeLeaves.size() <= std::max({a, b, c})) {
                        op->emitError() << "dynamic NZFamily layout stride derivation requires shape leaves " << a
                                        << ", " << b << ", " << c;
                        return failure();
                    }
                    Value ab = builder.create<arith::MulIOp>(op->getLoc(), shapeLeaves[a], shapeLeaves[b]);
                    return builder.create<arith::MulIOp>(op->getLoc(), ab, shapeLeaves[c]).getResult();
                };
                // Layout-coupled NZFamily stride derivation from the remapped fractal shape leaves.
                // zN/L0C: stride[3] = ceil_div_rows * c0 * ele_num_per_c0 = shape[1]*shape[0]*shape[2]
                if ((childInfo->layoutTag == LayoutTag::zN || childInfo->layoutTag == LayoutTag::L0Clayout) &&
                    idx == 3) {
                    return mulShapeLeaves(/*a=*/1, /*b=*/0, /*c=*/2);
                }
                // nZ/zZMxScale: stride[1] = ceil_div_cols * c0 * ele_num_per_c0 = shape[3]*shape[2]*shape[0]
                // zZMxScale nests exactly like zZ (only its constants differ: the MX scale C0 is
                // fixed at 2 by the e8m0 format), so the same product derives its dynamic leaf.
                if ((childInfo->layoutTag == LayoutTag::nZ || childInfo->layoutTag == LayoutTag::zZMxScale) &&
                    idx == 1) {
                    return mulShapeLeaves(/*a=*/3, /*b=*/2, /*c=*/0);
                }
                // nNMxScale is the transposed MX scale nesting, so its dynamic leaf is stride[3]
                // = rows_round_up * c0 = shape[1]*shape[0]*shape[2], matching the zN rule.
                if (childInfo->layoutTag == LayoutTag::nNMxScale && idx == 3) {
                    return mulShapeLeaves(/*a=*/1, /*b=*/0, /*c=*/2);
                }
                // zNUnAlign: stride[1] = stride[3] = rows * ele_num_per_c0 = shape[1]*shape[0]*shape[2]
                // (shape[1] == 1, so the product is rows*ele_num_per_c0). Both leaves are runtime-
                // varying because rows is dynamic (M axis is not fractal-blocked).
                if (childInfo->layoutTag == LayoutTag::zNUnAlign && (idx == 1 || idx == 3)) {
                    return mulShapeLeaves(/*a=*/1, /*b=*/0, /*c=*/2);
                }
                op->emitError() << "dynamic NZFamily layout stride leaf at index " << idx
                                << " has no SSA derivation rule for layout "
                                << stringifyLayoutTag(childInfo->layoutTag);
                return failure();
            };

            FailureOr<Value> coord0 =
                materializeLeafFromTypeOrParent(childInfo->coordDims[0], parent.coord[0], "coord");
            FailureOr<Value> coord1 =
                materializeLeafFromTypeOrParent(childInfo->coordDims[1], parent.coord[1], "coord");
            FailureOr<Value> origin0 =
                materializeLeafFromTypeOrParent(childInfo->originShapeDims[0], parent.originShape[0], "origin_shape");
            FailureOr<Value> origin1 =
                materializeLeafFromTypeOrParent(childInfo->originShapeDims[1], parent.originShape[1], "origin_shape");
            if (failed(coord0) || failed(coord1) || failed(origin0) || failed(origin1)) {
                derivationFailed = true;
                return;
            }

            Value one = getOrCreateConstant(op, 1, 0);
            std::array<Value, 4> shape;
            std::array<Value, 4> stride;

            if (isLinearLayout(childInfo->layoutTag)) {
                Value parentShape0 = isLinearLayout(parent.layoutTag) ? parent.shape[0] : parent.originShape[0];
                Value parentShape1 = isLinearLayout(parent.layoutTag) ? parent.shape[1] : parent.originShape[1];
                FailureOr<Value> shape0Or =
                    materializeLeafFromTypeOrParent(childInfo->shapeDims[0], parentShape0, "shape");
                FailureOr<Value> shape1Or =
                    materializeLeafFromTypeOrParent(childInfo->shapeDims[1], parentShape1, "shape");
                if (failed(shape0Or) || failed(shape1Or)) {
                    derivationFailed = true;
                    return;
                }
                shape[0] = *shape0Or;
                shape[1] = *shape1Or;
                shape[2] = one;
                shape[3] = one;
                // Counted in bits, not bytes: a sub-byte element still tiles the
                // 32-byte stride quantum, it just takes more of them -- fp4 puts
                // 64 elements in it where int8 puts 32.
                // getByteSizeOfFixedWidthScalarType answers 0 for a sub-byte type
                // and would reject it here.
                constexpr int64_t linearStrideAlignmentBits = 32 * 8;
                int64_t elementBits = getBitSizeOfFixedWidthScalarType(childInfo->elementType);
                // i1 is addressed as a byte, not as a bit.
                if (childInfo->elementType.isInteger(1))
                    elementBits = 8;
                if (elementBits <= 0 || linearStrideAlignmentBits % elementBits != 0) {
                    op->emitError() << "tla.make_tensor_like cannot derive a 32-byte-aligned stride for element type "
                                    << childInfo->elementType;
                    derivationFailed = true;
                    return;
                }
                int64_t alignmentElements = linearStrideAlignmentBits / elementBits;
                auto alignLinearExtent = [&](Value extent) -> Value {
                    Value multiple = getOrCreateConstant(op, alignmentElements, 0);
                    Value one = getOrCreateConstant(op, 1, 0);
                    Value adjusted = builder.createOrFold<arith::AddIOp>(
                        op->getLoc(), extent, builder.createOrFold<arith::SubIOp>(op->getLoc(), multiple, one));
                    Value quotient = builder.createOrFold<arith::DivSIOp>(op->getLoc(), adjusted, multiple);
                    return builder.createOrFold<arith::MulIOp>(op->getLoc(), quotient, multiple);
                };

                // Derive dynamic linear strides from the child shape and alignment policy.
                auto materializeLinearStride = [&](int64_t leaf, size_t idx) -> FailureOr<Value> {
                    if (leaf != ShapedType::kDynamic)
                        return getOrCreateConstant(op, leaf, 0);
                    // Vector is stored as the normalized (1, N) row-major shape in
                    // the descriptor, so it takes the same stride rule.
                    if (childInfo->layoutTag == LayoutTag::RowMajor || isVectorLayout(childInfo->layoutTag))
                        return idx == 0 ? alignLinearExtent(shape[1]) : one;
                    if (childInfo->layoutTag == LayoutTag::ColumnMajor)
                        return idx == 0 ? one : alignLinearExtent(shape[0]);
                    op->emitError() << "unsupported linear layout for dynamic stride derivation";
                    return failure();
                };
                FailureOr<Value> stride0Or = materializeLinearStride(childInfo->strideDims[0], 0);
                FailureOr<Value> stride1Or = materializeLinearStride(childInfo->strideDims[1], 1);
                if (failed(stride0Or) || failed(stride1Or)) {
                    derivationFailed = true;
                    return;
                }
                stride = {*stride0Or, *stride1Or, one, one};
            } else {
                if (!isNZFamilyLayout(childInfo->layoutTag)) {
                    op->emitError() << "unsupported tla.make_tensor_like layout for descriptor";
                    derivationFailed = true;
                    return;
                }
                for (size_t i = 0; i < shape.size(); ++i) {
                    FailureOr<Value> leaf = materializeNZFamilyLeafFromTypeOrParent(
                        childInfo->shapeDims, parent.shape, i, "NZFamily layout shape");
                    if (failed(leaf)) {
                        derivationFailed = true;
                        return;
                    }
                    shape[i] = *leaf;
                }
                for (size_t i = 0; i < stride.size(); ++i) {
                    FailureOr<Value> leaf;
                    bool dynamicStrideLeaf = childInfo->strideDims[i] == ShapedType::kDynamic;
                    // A `make_tensor_like` target is a *newly addressed* buffer whose
                    // extent is the reference tile's, so a NZFamily pitch has to come
                    // from this child's own shape. The reference's stride describes the
                    // buffer the reference came from: for a GM reference that is the
                    // whole matrix, not this tile, so inheriting it makes an L1 or L0
                    // tile stride by the far matrix's row pitch -- past the end of a
                    // buffer sized to the tile. This is why the old `layoutChanged`
                    // guard was not enough: zN -> zN and nZ -> nZ keep the same tag while
                    // still moving to a differently sized buffer.
                    if (dynamicStrideLeaf) {
                        leaf = deriveNZFamilyStrideLeafFromShape(shape, i);
                    } else {
                        leaf = materializeNZFamilyLeafFromTypeOrParent(
                            childInfo->strideDims, parent.stride, i, "NZFamily layout stride");
                    }
                    if (failed(leaf)) {
                        derivationFailed = true;
                        return;
                    }
                    stride[i] = *leaf;
                }
            }

            TensorDescriptor desc;
            desc.base = typedBuffer;
            desc.bridgedBaseMemrefType = *bridgedBaseType;
            desc.shape = shape;
            desc.stride = stride;
            desc.originShape = {*origin0, *origin1};
            desc.coord = {*coord0, *coord1};
            desc.layoutTag = childInfo->layoutTag;
            desc.addrspace = childInfo->addressSpace;
            desc.elementType = childInfo->elementType;
            tensorDescriptorByValue[op->getResult(0)] = std::move(desc);
            return;
        }

        if (llvm::isa<::tla::MakeTensorOp>(op)) {
            if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
                op->emitError() << "expected tla.make_tensor to have exactly 3 operands and 1 result";
                derivationFailed = true;
                return;
            }

            Value ptrValue = op->getOperand(0);
            if (!llvm::isa<::tla::PtrType>(ptrValue.getType())) {
                op->emitError() << "tla.make_tensor pointer operand must be !tla.ptr";
                derivationFailed = true;
                return;
            }
            Value layoutValue = op->getOperand(1);
            Value coordValue = op->getOperand(2);
            auto makeLayout = layoutValue.getDefiningOp<::tla::MakeLayoutOp>();
            if (!makeLayout) {
                op->emitError() << "tla.make_tensor layout operand must come from tla.make_layout";
                derivationFailed = true;
                return;
            }

            auto childInfo = decodeTensorTypeInfo(op->getResult(0).getType());
            if (failed(childInfo)) {
                op->emitError() << "tla.make_tensor currently requires a structured tla.tensor "
                                   "result type";
                derivationFailed = true;
                return;
            }
            if (!isLinearLayout(childInfo->layoutTag) && !isNZFamilyLayout(childInfo->layoutTag)) {
                op->emitError() << "tla.make_tensor has an unsupported layout for descriptor lowering";
                derivationFailed = true;
                return;
            }

            // Buffer element count for the synthetic !tla.memref type: prefer a static 1D
            // length from an HIVM pointer-cast bridge (allocator-backed ptr), else multiply
            // the first two origin_shape dims (e.g. inttoptr-backed ptr with static layout).
            int64_t flatElemCount = ShapedType::kDynamic;
            if (auto n = getStaticAllocationElementCount(ptrValue); succeeded(n) && *n > 0) {
                flatElemCount = *n;
            } else if (
                childInfo->originShapeDims.size() >= 2 && childInfo->originShapeDims[0] != ShapedType::kDynamic &&
                childInfo->originShapeDims[1] != ShapedType::kDynamic) {
                int64_t dim0 = childInfo->originShapeDims[0];
                int64_t dim1 = childInfo->originShapeDims[1];
                if (dim0 > 0 && dim1 > 0)
                    flatElemCount = dim0 * dim1;
            }
            auto [storageElemType, storageElemCount] = storageTypeAndCount(childInfo->elementType, flatElemCount);
            auto bridgedBaseType =
                buildHivmMemrefType(op->getContext(), {storageElemCount}, storageElemType, childInfo->tlaAddressSpace);
            if (failed(bridgedBaseType)) {
                op->emitError() << "tla.make_tensor buffer memref must be bridgeable to builtin memref type";
                derivationFailed = true;
                return;
            }

            Value typedBuffer = ptrValue;

            // Materialize index-tree leaves from the operand defining ops. Static leaves
            // become constants; dynamic leaves are pulled from tla.make_shape/make_stride/
            // make_coord dyn-elems in leaf order. ``childInfo`` already promotes rank-1
            // linear to rank-2, so the leading synthetic ``1``/``0`` leaves are static
            // here. The promoted leading stride may be derived (extent * elemStride) and
            // is not present on make_stride; that case is handled after shape leaves are
            // available below.
            auto materializeLeaves = [&](Value packedValue, ArrayRef<int64_t> leaves,
                                         StringRef kind) -> FailureOr<SmallVector<Value, 4>> {
                SmallVector<Value, 4> result;
                unsigned dynLeafCount = 0;
                for (int64_t leaf : leaves)
                    if (leaf == ShapedType::kDynamic)
                        ++dynLeafCount;
                SmallVector<Value, 4> dynElems;
                if (dynLeafCount > 0) {
                    if (kind == "shape") {
                        if (auto ms = packedValue.getDefiningOp<::tla::MakeShapeOp>())
                            dynElems.append(ms.getDynElems().begin(), ms.getDynElems().end());
                    } else if (kind == "stride") {
                        if (auto mst = packedValue.getDefiningOp<::tla::MakeStrideOp>())
                            dynElems.append(mst.getDynElems().begin(), mst.getDynElems().end());
                    } else {
                        if (auto mc = packedValue.getDefiningOp<::tla::MakeCoordOp>())
                            dynElems.append(mc.getDynElems().begin(), mc.getDynElems().end());
                    }
                    if (dynElems.size() < dynLeafCount) {
                        op->emitError()
                            << "tla.make_tensor " << kind
                            << " has a derived dynamic leaf that is not directly operand-backed "
                               "(e.g. rank-1 stride with dynamic extent); pass explicit leaves via tla.make_"
                            << kind;
                        return failure();
                    }
                }
                size_t di = 0;
                for (int64_t leaf : leaves) {
                    if (leaf == ShapedType::kDynamic) {
                        Value dv = dynElems[di++];
                        if (!dv.getType().isIndex()) {
                            op->emitError() << "tla.make_tensor " << kind << " dynamic operands must be index type";
                            return failure();
                        }
                        result.push_back(dv);
                    } else {
                        result.push_back(getOrCreateConstant(op, leaf, 0));
                    }
                }
                return result;
            };

            auto shapeLeaves = materializeLeaves(makeLayout.getShape(), childInfo->shapeDims, "shape");
            if (failed(shapeLeaves)) {
                derivationFailed = true;
                return;
            }

            // Rank-1 linear layout is normalized to rank-2 as shape=(1,E),
            // stride=(E*S,S). When E is dynamic, stride0 is derived rather than
            // represented by a make_stride operand. Recover it from the
            // materialized extent and element stride.
            FailureOr<SmallVector<Value, 4>> strideLeaves;
            {
                unsigned strideDynLeafCount = 0;
                for (int64_t leaf : childInfo->strideDims)
                    if (leaf == ShapedType::kDynamic)
                        ++strideDynLeafCount;
                SmallVector<Value, 4> strideDynElems;
                if (auto makeStride = makeLayout.getStride().getDefiningOp<::tla::MakeStrideOp>())
                    strideDynElems.append(makeStride.getDynElems().begin(), makeStride.getDynElems().end());

                bool needsDerivedLeadingStride = isLinearLayout(childInfo->layoutTag) && strideDynLeafCount > 0 &&
                                                 strideDynElems.size() < strideDynLeafCount &&
                                                 childInfo->shapeDims.size() == 2 &&
                                                 childInfo->strideDims.size() == 2 && childInfo->shapeDims[0] == 1 &&
                                                 childInfo->strideDims[0] == ShapedType::kDynamic;
                if (needsDerivedLeadingStride) {
                    unsigned trailingDynNeeded = childInfo->strideDims[1] == ShapedType::kDynamic ? 1u : 0u;
                    if (strideDynElems.size() < trailingDynNeeded) {
                        op->emitError() << "tla.make_tensor stride has a derived dynamic leaf that is not "
                                           "directly operand-backed (e.g. rank-1 stride with dynamic extent); "
                                           "pass explicit leaves via tla.make_stride";
                        derivationFailed = true;
                        return;
                    }
                    Value elemStride = childInfo->strideDims[1] == ShapedType::kDynamic ?
                                           strideDynElems[0] :
                                           getOrCreateConstant(op, childInfo->strideDims[1], 0);
                    if (!elemStride.getType().isIndex()) {
                        op->emitError() << "tla.make_tensor stride dynamic operands must be index type";
                        derivationFailed = true;
                        return;
                    }
                    OpBuilder builder(op);
                    Value leading = builder.create<arith::MulIOp>(op->getLoc(), (*shapeLeaves)[1], elemStride);
                    strideLeaves = SmallVector<Value, 4>{leading, elemStride};
                } else {
                    strideLeaves = materializeLeaves(makeLayout.getStride(), childInfo->strideDims, "stride");
                }
            }

            auto coordLeaves = materializeLeaves(coordValue, childInfo->coordDims, "coord");
            if (failed(strideLeaves) || failed(coordLeaves)) {
                derivationFailed = true;
                return;
            }
            Value coord0 = (*coordLeaves)[0];
            Value coord1 = (*coordLeaves)[1];

            Value origin0;
            Value origin1;
            if (Value originOperand = makeLayout.getOriginShape()) {
                auto originLeaves = materializeLeaves(originOperand, childInfo->originShapeDims, "shape");
                if (failed(originLeaves)) {
                    derivationFailed = true;
                    return;
                }
                origin0 = (*originLeaves)[0];
                origin1 = (*originLeaves)[1];
            } else if (isNZFamilyLayout(childInfo->layoutTag)) {
                // NZFamily shapes flatten as (m0,m1),(n0,n1). Without an explicit
                // logical origin, use the padded logical dimensions represented
                // by the physical blocking.
                OpBuilder builder(op);
                origin0 = builder.create<arith::MulIOp>(op->getLoc(), (*shapeLeaves)[0], (*shapeLeaves)[1]);
                origin1 = builder.create<arith::MulIOp>(op->getLoc(), (*shapeLeaves)[2], (*shapeLeaves)[3]);
            } else {
                origin0 = (*shapeLeaves)[0];
                origin1 = (*shapeLeaves)[1];
            }

            Value one = getOrCreateConstant(op, 1, 0);
            std::array<Value, 4> shape;
            std::array<Value, 4> stride;
            if (isNZFamilyLayout(childInfo->layoutTag)) {
                shape = {(*shapeLeaves)[0], (*shapeLeaves)[1], (*shapeLeaves)[2], (*shapeLeaves)[3]};
                stride = {(*strideLeaves)[0], (*strideLeaves)[1], (*strideLeaves)[2], (*strideLeaves)[3]};
            } else {
                shape = {(*shapeLeaves)[0], (*shapeLeaves)[1], one, one};
                stride = {(*strideLeaves)[0], (*strideLeaves)[1], one, one};
            }
            TensorDescriptor desc;
            desc.base = typedBuffer;
            desc.bridgedBaseMemrefType = *bridgedBaseType;
            desc.shape = shape;
            desc.stride = stride;
            desc.originShape = {origin0, origin1};
            desc.coord = {coord0, coord1};
            desc.layoutTag = childInfo->layoutTag;
            desc.addrspace = childInfo->addressSpace;
            desc.elementType = childInfo->elementType;
            tensorDescriptorByValue[op->getResult(0)] = std::move(desc);
            return;
        }
    });
    return derivationFailed ? failure() : success();
}

} // namespace tla
