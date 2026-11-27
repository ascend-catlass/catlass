#include "Dialect/Tla/IR/TlaAttrs.h"
#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Dominance.h"

// tla-cube-region: lowers the cube (AIC) compute ops (tla.copy / tla.mmad) via
// the shared !tla.tensor->memref lowering (TlaTensorMemrefLowering), then flattens
// the tla.cube region. Runs after the two tla.vector lowering passes, before
// tla-finalize-memref.
//
// Flow: collect materialized descriptors, then lower tla.copy (descriptor +
// payload driven) and tla.mmad, both reconstructing tile memrefs directly from
// tla.tensor_desc. Descriptors are available for every copy, so copy lowering is
// a single descriptor-driven path. Finalize DCEs dead scaffolding and unrealized
// casts.

namespace tla {
namespace {

// tla.mmad_mx is only meaningful when its L0 operands were produced by an MX
// load. The hardware's mad_mx takes no scale operand: the e8m0 block is attached
// by the L1->L0 copy and read from a side buffer addressed off the L0 tile. That
// makes MX-ness a property of how the tile was *written*, which the element type
// and layout tag cannot express -- an ordinary fp8 tla.copy leaves an operand
// that looks identical to an MX-loaded one.
//
// The frontend has both ops and the user picks; nothing here infers one from the
// other.

// The bc symbol suffix is always the C++ type the wrapper instantiates. The DSL
// spells the packed fp4 formats !tla.f4e2m1 / !tla.f4e1m2; the C++ types are
// float4_e2m1x2_t / float4_e1m2x2_t. Kept here so every caller that builds an
// fp4 callee name uses the same mapping.
//
// The encoding is the element type -- there is no separate attribute to consult
// and therefore nothing that can disagree with it. A tile is !tla.f4e2m1 or
// !tla.f4e1m2 from the moment it is allocated, so the load and the matmul read
// the same answer by construction; ::tla::isPackedFp4Type is the shared test.
static StringRef fp4CppType(Type elementType)
{
    return ::llvm::isa<::tla::Float4E1M2Type>(elementType) ? "float4_e1m2x2_t" : "float4_e2m1x2_t";
}

static void setMmadComputeOrder(OpBuilder& builder, Location loc, ComputeOrder order)
{
    builder.create<hivm::SetCtrlOp>(loc, order == ComputeOrder::N_FIRST, 51);
}

static void setMmadHF32Mode(OpBuilder& builder, Location loc, HF32Mode mode)
{
    builder.create<hivm::SetCtrlOp>(loc, mode != HF32Mode::HF32_DISABLE, 46);
    builder.create<hivm::SetCtrlOp>(loc, mode == HF32Mode::HF32_NEAREST_ZERO, 47);
}

template <typename AttrTy>
static AttrTy getUniformMmadAttr(ArrayRef<Operation*> ops, StringRef name)
{
    AttrTy uniform;
    for (Operation* op : ops) {
        AttrTy attr = op->getAttrOfType<AttrTy>(name);
        if (!attr)
            continue;
        if (uniform && uniform != attr)
            return {};
        uniform = attr;
    }
    return uniform;
}

// Shared by tla.mmad and tla.mmad_mx: the two ops differ only in whether the
// L0 operands carry an attached e8m0 scale block, and that is a property of the
// route, not of a type -- the callee is `mmad_*` or `mmadmx_*` accordingly, and
// mmad.cpp maps those prefixes to GemmMmad and GemmMmadMx. The operand list,
// shape contract and descriptor handling are identical.
template <typename MmadOpTy>
struct LowerTlaMmadPatternImpl : public OpRewritePattern<MmadOpTy> {
    static constexpr bool kIsMx = std::is_same_v<MmadOpTy, ::tla::MmadMxOp>;

    LowerTlaMmadPatternImpl(
        MLIRContext* ctx, DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue,
        SmallVectorImpl<Operation*>& toErase, DenseMap<Value, Value>& loweredMemrefByValue, bool hasUniformOrder,
        bool hasUniformHF32, bool hasExplicitCtrl)
        : OpRewritePattern<MmadOpTy>(ctx),
          tensorDescriptorByValue(tensorDescriptorByValue),
          toErase(toErase),
          loweredMemrefByValue(loweredMemrefByValue),
          hasUniformOrder(hasUniformOrder),
          hasUniformHF32(hasUniformHF32),
          hasExplicitCtrl(hasExplicitCtrl)
    {}

    LogicalResult matchAndRewrite(MmadOpTy op, PatternRewriter& rewriter) const override
    {
        if (op->getNumOperands() < 3)
            return success();

        Value acc = op->getOperand(0);
        Value lhs = op->getOperand(1);
        Value rhs = op->getOperand(2);
        Type accType = acc.getType();
        Type lhsType = lhs.getType();
        Type rhsType = rhs.getType();

        Value initC = op->getOperand(3);
        Value unitFlag = op->getOperand(4);

        auto i1Type = rewriter.getI1Type();
        auto i64Type = rewriter.getI64Type();
        auto i8Type = rewriter.getI8Type();

        Value initCVal = initC;
        Value unitFlagVal = rewriter.create<arith::TruncIOp>(op.getLoc(), i8Type, unitFlag);

        auto lhsInfo = ::tla::decodeTensorTypeInfo(lhsType);
        auto rhsInfo = ::tla::decodeTensorTypeInfo(rhsType);
        auto accInfo = ::tla::decodeTensorTypeInfo(accType);
        if (failed(lhsInfo) || failed(rhsInfo) || failed(accInfo)) {
            op.emitError() << "tla.mmad currently requires structured tla.tensor operand types";
            return failure();
        }
        if (lhsInfo->rank != 2 || rhsInfo->rank != 2 || accInfo->rank != 2 || lhsInfo->hasBatch || rhsInfo->hasBatch ||
            accInfo->hasBatch) {
            op.emitError() << "tla.mmad currently supports rank-2 tiles only";
            return failure();
        }

        // A 1-D VectorLayout lhs selects the gemv form, so the layout tag alone
        // carries the mode; the bc wrapper's disableGemv is derived from it here.
        // mmad_mx has no gemv form.
        const bool lhsIsVectorLayout = lhsInfo->layoutTag == LayoutTag::VectorLayout;
        const bool isGemv = !kIsMx && lhsIsVectorLayout;
        if (lhsInfo->addressSpace != "l0a" || rhsInfo->addressSpace != "l0b" || accInfo->addressSpace != "l0c") {
            op.emitError() << "unsupported tla.mmad tile addrspaces; expected acc in l0c, lhs in l0a, rhs in l0b";
            return failure();
        }
        bool supportedF16Route =
            lhsInfo->elementType.isF16() && rhsInfo->elementType.isF16() && accInfo->elementType.isF32();
        bool supportedBf16Route =
            lhsInfo->elementType.isBF16() && rhsInfo->elementType.isBF16() && accInfo->elementType.isF32();
        bool supportedF32Route =
            lhsInfo->elementType.isF32() && rhsInfo->elementType.isF32() && accInfo->elementType.isF32();
        // Integer route: unlike the float routes, the L0C accumulator is i32.
        bool supportedI8Route = lhsInfo->elementType.isSignlessInteger(8) &&
                                rhsInfo->elementType.isSignlessInteger(8) && accInfo->elementType.isSignlessInteger(32);
        // HiFloat8 has byte-sized storage like fp8, but its Cube decode mode is
        // selected by a dedicated BC route rather than by the fp8 route.
        bool supportedHiF8Route = !kIsMx && ::llvm::isa<::tla::HiFloat8Type>(lhsInfo->elementType) &&
                                  ::llvm::isa<::tla::HiFloat8Type>(rhsInfo->elementType) &&
                                  accInfo->elementType.isF32();
        // fp8 routes support both MLIR f8E4M3FN and f8E5M2 types. The two formats
        // mix freely on the cube, so all four operand pairings are legal; each
        // accumulates into fp32.
        constexpr bool isMx = kIsMx;
        auto isFp8 = [](Type elem) { return elem.isFloat8E4M3FN() || elem.isFloat8E5M2(); };
        bool bothFp8 = isFp8(lhsInfo->elementType) && isFp8(rhsInfo->elementType) && accInfo->elementType.isF32();
        // Plain fp8 (tla.mmad): the L0 tiles carry no scale block, so this lands on
        // the same `mad` intrinsic as the float routes.
        bool supportedFp8Route = !isMx && bothFp8;
        // MX fp8 (tla.mmad_mx): the L0 tiles were loaded with their scale block
        // attached (tla.copy with a scale operand). The operands stay real fp8;
        // the `mmadmx_` callee prefix is what routes to MmadMx.
        bool isMxFp8 = isMx && bothFp8;
        // MX fp4 tiles state both width and encoding in the element type
        // (!tla.f4e2m1 / !tla.f4e1m2). They use the ordinary zN / nZ layouts like
        // every other operand, and the two sides need not share an encoding -- the
        // cube has a mad_mx for every pairing.
        bool isMxFp4 = isMx && ::tla::isPackedFp4Type(lhsInfo->elementType) &&
                       ::tla::isPackedFp4Type(rhsInfo->elementType) && accInfo->elementType.isF32();
        if (isMx && !isMxFp8 && !isMxFp4) {
            op.emitError() << "tla.mmad_mx requires f8E4M3FN/f8E5M2 operands or fp4-tagged "
                              "tiles, and an f32 accumulator";
            return failure();
        }
        if (!isMx && !supportedF16Route && !supportedBf16Route && !supportedF32Route && !supportedI8Route &&
            !supportedHiF8Route && !supportedFp8Route) {
            // A packed fp4 tile reaching a plain tla.mmad means it was loaded
            // without a scale: there is no non-microscaling fp4 route on the cube,
            // so say that rather than listing the element types it is not.
            if (::tla::isPackedFp4Type(lhsInfo->elementType) || ::tla::isPackedFp4Type(rhsInfo->elementType)) {
                op.emitError() << "packed fp4 tla.mmad operands must be loaded by tla.copy(..., scale=...); "
                                  "the cube has no non-microscaling fp4 route";
                return failure();
            }
            op.emitError() << "unsupported tla.mmad element types; expected f16,f16 -> f32, bf16,bf16 "
                              "-> f32, f32,f32 -> f32, HiFloat8,HiFloat8 -> f32, any f8E4M3FN/f8E5M2 "
                              "pair -> f32 (fp32 L0C accumulator), or i8,i8 -> i32 (i32 L0C accumulator)";
            return failure();
        }

        auto maybeStaticShapeCheck = [&](int64_t lhsM, int64_t lhsK, int64_t rhsK, int64_t rhsN, int64_t accM,
                                         int64_t accN) -> LogicalResult {
            if (lhsM == ShapedType::kDynamic || lhsK == ShapedType::kDynamic || rhsK == ShapedType::kDynamic ||
                rhsN == ShapedType::kDynamic || accM == ShapedType::kDynamic || accN == ShapedType::kDynamic) {
                return success();
            }
            if (lhsK != rhsK || lhsM != accM || rhsN != accN) {
                op.emitError() << "unsupported tla.mmad tile shape contract; expected lhs(MxK), "
                                  "rhs(KxN), acc(MxN)";
                return failure();
            }
            return success();
        };
        if (failed(maybeStaticShapeCheck(
                lhsInfo->originShapeDims[0], lhsInfo->originShapeDims[1], rhsInfo->originShapeDims[0],
                rhsInfo->originShapeDims[1], accInfo->originShapeDims[0], accInfo->originShapeDims[1])))
            return failure();
        // lhs is zN in the matrix form and VectorLayout in the gemv form; acc and
        // rhs keep their layouts in both.
        bool lhsLayoutOk = lhsIsVectorLayout || lhsInfo->layoutTag == LayoutTag::zN;
        bool rhsLayoutOk = rhsInfo->layoutTag == LayoutTag::nZ;
        if (accInfo->layoutTag != LayoutTag::L0Clayout || !lhsLayoutOk || !rhsLayoutOk) {
            op.emitError() << "unsupported tla.mmad operand layout; expected acc L0Clayout, rhs nZ, and lhs "
                              "either zN (matrix form) or VectorLayout (gemv form)";
            return failure();
        }

        // Optional bias: must be a VectorLayout biasbuf tile whose N matches acc's.
        // A VectorLayout tile is rank-1 by definition, so it is a flat run of N
        // elements -- decodeTensorTypeInfo has already normalized it to the
        // descriptor's (1, N) slot form, which is why N is read from slot 1.
        // Both ops carry one, and the bc wrappers have a bias twin per pairing:
        // the MX instantiation reaches mad_mx with the bias table exactly as the
        // plain one reaches mad.
        Value biasValue = op.getBias();
        if (biasValue) {
            auto biasInfo = ::tla::decodeTensorTypeInfo(biasValue.getType());
            if (failed(biasInfo)) {
                op.emitError() << "tla.mmad bias requires a structured tla.tensor operand type";
                return failure();
            }
            if (biasInfo->addressSpace != "biasbuf") {
                op.emitError() << "unsupported tla.mmad bias addrspace; expected biasbuf, got "
                               << biasInfo->addressSpace;
                return failure();
            }
            if (biasInfo->elementType != accInfo->elementType) {
                op.emitError() << "unsupported tla.mmad bias element type; expected same as acc elementType, got "
                               << biasInfo->elementType;
                return failure();
            }
            if (biasInfo->layoutTag != LayoutTag::VectorLayout) {
                op.emitError() << "unsupported tla.mmad bias layout; expected a 1-D VectorLayout tile, got "
                               << stringifyLayoutTag(biasInfo->layoutTag);
                return failure();
            }
            int64_t accN = accInfo->originShapeDims[1];
            int64_t biasN = biasInfo->originShapeDims[1];
            if (accN != ShapedType::kDynamic && biasN != ShapedType::kDynamic && biasN != accN) {
                op.emitError() << "unsupported tla.mmad bias shape; expected N matching acc (" << accN << ")";
                return failure();
            }
        }

        // Materialize each tile operand's memref directly from its tla.tensor_desc
        // descriptor (shared with the tla.copy path).
        // Every operand here is handed straight to a wrapper that reads the
        // memref's start address, so a rank-1 tile's coordinate has to be baked
        // into the memref. The copy routes do the opposite -- they ship the
        // coordinate in the payload -- and must not be given the folded form.
        auto materializeTensorOperand = [&](Value tensor) -> FailureOr<Value> {
            auto it = tensorDescriptorByValue.find(tensor);
            if (it == tensorDescriptorByValue.end()) {
                op.emitError() << "missing descriptor for tla.mmad tile operand";
                return failure();
            }
            return ::tla::materializeTileMemrefFromDescriptor(
                rewriter, op.getLoc(), it->second, op.getOperation(), loweredMemrefByValue,
                /*foldRankOneCoord=*/true);
        };

        FailureOr<Value> lhsMemref = materializeTensorOperand(lhs);
        FailureOr<Value> rhsMemref = materializeTensorOperand(rhs);
        FailureOr<Value> accMemref = materializeTensorOperand(acc);
        if (failed(lhsMemref) || failed(rhsMemref) || failed(accMemref)) {
            op.emitError() << "failed to bridge tla.mmad operands to memref values";
            return failure();
        }
        FailureOr<Value> biasMemref;
        if (biasValue) {
            biasMemref = materializeTensorOperand(biasValue);
            if (failed(biasMemref)) {
                op.emitError() << "failed to bridge tla.mmad bias operand to memref";
                return failure();
            }
        }

        // Match the tla.copy runtime ABI: pass dynamic strided memrefs to the C stub
        // (same as buildRuntimeMemref in LowerTlaCopyPattern).
        auto toRuntimeMemref = [&](Value v) -> FailureOr<Value> {
            auto baseType = dyn_cast<MemRefType>(v.getType());
            if (!baseType) {
                op.emitError() << "tla.mmad memref operand must have memref type";
                return failure();
            }
            MemRefType runtimeType = ::tla::getDynamicStridedMemrefType(baseType);
            return ::tla::castMemrefToType(rewriter, op.getLoc(), v, runtimeType);
        };
        FailureOr<Value> lhsRuntime = toRuntimeMemref(*lhsMemref);
        FailureOr<Value> rhsRuntime = toRuntimeMemref(*rhsMemref);
        FailureOr<Value> accRuntime = toRuntimeMemref(*accMemref);
        if (failed(lhsRuntime) || failed(rhsRuntime) || failed(accRuntime))
            return failure();

        FailureOr<Value> biasRuntime;
        if (biasValue) {
            biasRuntime = toRuntimeMemref(*biasMemref);
            if (failed(biasRuntime))
                return failure();
        }

        auto materializeIndexDim = [&](Value tensor, int64_t staticOriginDim, StringRef fieldName,
                                       bool takeSecondDim) -> FailureOr<Value> {
            auto it = tensorDescriptorByValue.find(tensor);
            if (it != tensorDescriptorByValue.end()) {
                Value dim = it->second.originShape[takeSecondDim ? 1 : 0];
                if (dim && dim.getType().isIndex())
                    return dim;
            }
            if (staticOriginDim == ShapedType::kDynamic) {
                op.emitError() << "tla.mmad requires " << fieldName
                               << " from tensor descriptor SSA when type origin_shape is dynamic";
                return failure();
            }
            return rewriter.create<arith::ConstantIndexOp>(op.getLoc(), staticOriginDim).getResult();
        };
        FailureOr<Value> mIndex = materializeIndexDim(lhs, lhsInfo->originShapeDims[0], "M", false);
        FailureOr<Value> kIndex = materializeIndexDim(lhs, lhsInfo->originShapeDims[1], "K", true);
        FailureOr<Value> nIndex = materializeIndexDim(rhs, rhsInfo->originShapeDims[1], "N", true);
        if (failed(mIndex) || failed(kIndex) || failed(nIndex))
            return failure();

        auto castIndexToI64 = [&](Value v) -> Value {
            return rewriter.create<arith::IndexCastOp>(op.getLoc(), i64Type, v).getResult();
        };
        Value mI64 = castIndexToI64(*mIndex);
        Value kI64 = castIndexToI64(*kIndex);
        Value nI64 = castIndexToI64(*nIndex);

        SmallVector<Type, 9> operandTypes = {(*lhsRuntime).getType(),
                                             (*rhsRuntime).getType(),
                                             (*accRuntime).getType(),
                                             i64Type,
                                             i64Type,
                                             i64Type,
                                             i1Type,
                                             i8Type};
        if constexpr (!kIsMx)
            operandTypes.push_back(i1Type);
        if (biasValue)
            operandTypes.push_back((*biasRuntime).getType());
        // FP8 needs a per-operand tag because the two formats can be mixed.
        //
        // The tag is the C++ element type the wrapper instantiates -- the CANN
        // builtin fp8_e4m3fn_t / fp8_e5m2_t -- which is a valid identifier and so
        // can be pasted straight into the symbol. fp4 uses the same rule via its
        // own element type (!tla.f4e2m1 / !tla.f4e1m2).
        //
        // The microscaled fp8 routes spell their operands the same way; only the
        // `mmadmx_` prefix distinguishes them, and it is what makes mmad.cpp call
        // GemmMmadMx instead of GemmMmad.
        //
        // Only the tag is built here. Both families register their wrappers as a
        // pair -- `mmad_` / `mmad_bias_` on the plain routes, `mmadmx_` /
        // `mmadmx_bias_` on the MX ones -- so the prefix and the trailing `_float`
        // follow from the route and the tag is the same string either way.
        auto fp8Tag = [](Type elem) -> StringRef { return elem.isFloat8E4M3FN() ? "fp8_e4m3fn_t" : "fp8_e5m2_t"; };
        const bool isMxRoute = isMxFp8 || isMxFp4;
        std::string fp8OperandTag;
        if (isMxFp4) {
            // Per operand, not once: the two sides carry their own encodings and
            // the bc registers every pairing.
            fp8OperandTag = (fp4CppType(lhsInfo->elementType) + "_" + fp4CppType(rhsInfo->elementType)).str();
        } else if (isMxFp8 || supportedFp8Route) {
            fp8OperandTag = (fp8Tag(lhsInfo->elementType) + "_" + fp8Tag(rhsInfo->elementType)).str();
        }
        std::string calleeName;
        if (biasValue) {
            calleeName = supportedF16Route  ? "mmad_bias_half_half_float" :
                         supportedBf16Route ? "mmad_bias_bf16_bf16_float" :
                         supportedI8Route   ? "mmad_bias_int8_int8_int32" :
                         supportedHiF8Route ? "mmad_bias_hifloat8_t_hifloat8_t_float" :
                         isMxRoute          ? "mmadmx_bias_" + fp8OperandTag + "_float" :
                         supportedFp8Route  ? "mmad_bias_" + fp8OperandTag + "_float" :
                                              "mmad_bias_float_float_float";
        } else {
            calleeName = supportedF16Route  ? "mmad_half_half_float" :
                         supportedBf16Route ? "mmad_bf16_bf16_float" :
                         supportedI8Route   ? "mmad_int8_int8_int32" :
                         supportedHiF8Route ? "mmad_hifloat8_t_hifloat8_t_float" :
                         isMxRoute          ? "mmadmx_" + fp8OperandTag + "_float" :
                         supportedFp8Route  ? "mmad_" + fp8OperandTag + "_float" :
                                              "mmad_float_float_float";
        }
        auto callee = ::tla::getOrCreateRuntimeCall(op->template getParentOfType<ModuleOp>(), calleeName, operandTypes);
        SmallVector<Value, 9> operands = {*lhsRuntime, *rhsRuntime, *accRuntime, mI64,
                                          nI64,        kI64,        initCVal,    unitFlagVal};
        if constexpr (!kIsMx)
            operands.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getBoolAttr(isGemv)));
        if (biasValue)
            operands.push_back(*biasRuntime);
        if (!hasUniformOrder)
            setMmadComputeOrder(rewriter, op.getLoc(), op.getComputeOrder().getValue());
        if constexpr (!kIsMx) {
            if (!hasUniformHF32)
                setMmadHF32Mode(rewriter, op.getLoc(), op.getHf32Mode().getValue());
            // An explicit CTRL patch may select HF8 interpretation for these inputs.
            if (hasExplicitCtrl && lhsInfo->elementType.isFloat8E4M3FN() && rhsInfo->elementType.isFloat8E4M3FN())
                rewriter.create<hivm::SetCtrlOp>(op.getLoc(), /*enable=*/false, 45);
        }
        rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
        toErase.push_back(op.getOperation());
        return success();
    }

private:
    DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue;
    SmallVectorImpl<Operation*>& toErase;
    DenseMap<Value, Value>& loweredMemrefByValue;
    bool hasUniformOrder;
    bool hasUniformHF32;
    bool hasExplicitCtrl;
};

using LowerTlaMmadPattern = LowerTlaMmadPatternImpl<::tla::MmadOp>;
using LowerTlaMmadMxPattern = LowerTlaMmadPatternImpl<::tla::MmadMxOp>;

// tla.copy_mx: L1 -> L0A/L0B with the operand's e8m0 scale block attached. The
// scale is consumed by this load; the L0 tile keeps the real fp8 / fp4 element
// type, and the later tla.mmad_mx picks MmadMx by its route.
struct LowerTlaCopyMxPattern : public OpRewritePattern<::tla::CopyMxOp> {
    LowerTlaCopyMxPattern(
        MLIRContext* ctx, DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue,
        SmallVectorImpl<Operation*>& toErase, DenseMap<Value, Value>& loweredMemrefByValue)
        : OpRewritePattern<::tla::CopyMxOp>(ctx),
          tensorDescriptorByValue(tensorDescriptorByValue),
          toErase(toErase),
          loweredMemrefByValue(loweredMemrefByValue)
    {}

    LogicalResult matchAndRewrite(::tla::CopyMxOp op, PatternRewriter& rewriter) const override
    {
        Value dstTile = op.getDst();
        Value srcTile = op.getSrc();
        Value scaleTile = op.getScale();

        auto lookup = [&](Value v, StringRef what) -> const TensorDescriptor* {
            auto it = tensorDescriptorByValue.find(v);
            if (it == tensorDescriptorByValue.end()) {
                op.emitError() << "missing descriptor for tla.copy_mx " << what << " tile";
                return nullptr;
            }
            return &it->second;
        };
        const TensorDescriptor* dstDesc = lookup(dstTile, "dst");
        const TensorDescriptor* srcDesc = lookup(srcTile, "src");
        const TensorDescriptor* scaleDesc = lookup(scaleTile, "scale");
        if (!dstDesc || !srcDesc || !scaleDesc)
            return failure();

        // The bc layer applies the tile coordinate from the payload itself, so
        // this passes the descriptor's base memref, exactly like
        // LowerTlaCopyPattern: folding the coordinate into a tile subview here
        // as well would apply the offset twice.
        auto buildRuntimeMemref = [&](const TensorDescriptor& desc) -> FailureOr<Value> {
            FailureOr<Value> baseMemref = ::tla::getOrMaterializeDescriptorBaseMemref(
                rewriter, op.getLoc(), desc, op.getOperation(), loweredMemrefByValue);
            if (failed(baseMemref))
                return failure();
            auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
            if (!baseType)
                return failure();
            return ::tla::castMemrefToType(
                rewriter, op.getLoc(), *baseMemref, ::tla::getDynamicStridedMemrefType(baseType));
        };

        FailureOr<Value> srcMemref = buildRuntimeMemref(*srcDesc);
        FailureOr<Value> dstMemref = buildRuntimeMemref(*dstDesc);
        FailureOr<Value> scaleMemref = buildRuntimeMemref(*scaleDesc);
        if (failed(srcMemref) || failed(dstMemref) || failed(scaleMemref))
            return failure();

        bool toL0A = dstDesc->addrspace == "l0a";
        // The element type states both the 4-bit width and the encoding, so the
        // callee suffix comes straight off the tile.
        bool isFp4 = ::tla::isPackedFp4Type(srcDesc->elementType);
        std::string calleeName;
        if (isFp4) {
            calleeName = (Twine("copy_mx_l1_") + stringifyLayoutTag(srcDesc->layoutTag) +
                          (toL0A ? "_to_l0a_zN_" : "_to_l0b_nZ_") + fp4CppType(srcDesc->elementType))
                             .str();
        } else {
            // The suffix is the element type the wrapper instantiates, L1 and L0
            // alike: the scale block does not change it.
            StringRef fmt = srcDesc->elementType.isFloat8E4M3FN() ? "fp8_e4m3fn_t" :
                            srcDesc->elementType.isFloat8E5M2()   ? "fp8_e5m2_t" :
                                                                    "";
            if (fmt.empty()) {
                op.emitError() << "unsupported tla.copy_mx element type " << srcDesc->elementType
                               << "; expected f8E4M3FN, f8E5M2, f4e2m1, or f4e1m2";
                return failure();
            }
            // Either source layout, on either side: the L1 tile keeps whatever
            // orientation its GM operand had, and the transposing pairing is a
            // real Catlass specialization.
            calleeName = (Twine("copy_mx_l1_") + stringifyLayoutTag(srcDesc->layoutTag) +
                          (toL0A ? "_to_l0a_zN_" : "_to_l0b_nZ_") + fmt)
                             .str();
        }

        SmallVector<Value, 40> payload;
        auto appendDesc = [&](const TensorDescriptor& desc) {
            SmallVector<Value, 12> fields = ::tla::buildCopyPayloadForDescriptor(rewriter, op.getLoc(), desc);
            payload.append(fields.begin(), fields.end());
        };
        appendDesc(*srcDesc);
        appendDesc(*dstDesc);
        appendDesc(*scaleDesc);

        SmallVector<Type, 40> operandTypes = {(*srcMemref).getType(), (*dstMemref).getType(), (*scaleMemref).getType()};
        for (Value v : payload)
            operandTypes.push_back(v.getType());
        SmallVector<Value, 40> operands = {*srcMemref, *dstMemref, *scaleMemref};
        operands.append(payload.begin(), payload.end());

        auto callee = ::tla::getOrCreateRuntimeCall(op->getParentOfType<ModuleOp>(), calleeName, operandTypes);
        rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
        toErase.push_back(op.getOperation());
        return success();
    }

private:
    DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue;
    SmallVectorImpl<Operation*>& toErase;
    DenseMap<Value, Value>& loweredMemrefByValue;
};

// tla.img2col: L1 NC1HWC0 feature map -> L0A or L0B, expanded by the img2col unit. The
// geometry rides along as scalars, so one compiled kernel serves any
// pad/stride/dilation; the extents and start points come off the destination
// tile's descriptor inside the callee.
struct LowerTlaImg2colPattern : public OpRewritePattern<::tla::Img2colOp> {
    LowerTlaImg2colPattern(
        MLIRContext* ctx, DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue,
        SmallVectorImpl<Operation*>& toErase, DenseMap<Value, Value>& loweredMemrefByValue)
        : OpRewritePattern<::tla::Img2colOp>(ctx),
          tensorDescriptorByValue(tensorDescriptorByValue),
          toErase(toErase),
          loweredMemrefByValue(loweredMemrefByValue)
    {}

    LogicalResult matchAndRewrite(::tla::Img2colOp op, PatternRewriter& rewriter) const override
    {
        auto lookup = [&](Value v, StringRef what) -> const TensorDescriptor* {
            auto it = tensorDescriptorByValue.find(v);
            if (it == tensorDescriptorByValue.end()) {
                op.emitError() << "missing descriptor for tla.img2col " << what << " tile";
                return nullptr;
            }
            return &it->second;
        };
        const TensorDescriptor* dstDesc = lookup(op.getDst(), "dst");
        const TensorDescriptor* srcDesc = lookup(op.getSrc(), "src");
        if (!dstDesc || !srcDesc)
            return failure();

        auto buildRuntimeMemref = [&](const TensorDescriptor& desc) -> FailureOr<Value> {
            FailureOr<Value> base = ::tla::materializeTileMemrefFromDescriptor(
                rewriter, op.getLoc(), desc, op.getOperation(), loweredMemrefByValue);
            if (failed(base))
                return failure();
            auto baseType = dyn_cast<MemRefType>((*base).getType());
            if (!baseType)
                return failure();
            return ::tla::castMemrefToType(rewriter, op.getLoc(), *base, ::tla::getDynamicStridedMemrefType(baseType));
        };
        FailureOr<Value> srcMemref = buildRuntimeMemref(*srcDesc);
        FailureOr<Value> dstMemref = buildRuntimeMemref(*dstDesc);
        if (failed(srcMemref) || failed(dstMemref))
            return failure();

        StringRef typeName = ::tla::getImg2colRuntimeElemSuffix(srcDesc->elementType);
        if (typeName.empty()) {
            op.emitError() << "unsupported tla.img2col element type " << srcDesc->elementType;
            return failure();
        }
        StringRef route = dstDesc->addrspace == "l0b" ? "l0b_nZ_" : "l0a_zN_";
        std::string calleeName = (Twine("img2col_l1_zNUnAlign_to_") + route + typeName).str();

        SmallVector<Value, 40> payload;
        auto appendDesc = [&](const TensorDescriptor& desc) {
            SmallVector<Value, 12> fields = ::tla::buildCopyPayloadForDescriptor(rewriter, op.getLoc(), desc);
            payload.append(fields.begin(), fields.end());
        };
        appendDesc(*srcDesc);
        appendDesc(*dstDesc);

        // The callee takes the geometry as i32; the op carries index so the
        // frontend can pass either a literal or a runtime value.
        auto toI32 = [&](Value v) -> Value {
            return rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getI32Type(), v);
        };
        for (Value v :
             {op.getFmapH(), op.getFmapW(), op.getFilterH(), op.getFilterW(), op.getStrideH(), op.getStrideW(),
              op.getDilationH(), op.getDilationW(), op.getPadLeft(), op.getPadRight(), op.getPadTop(),
              op.getPadBottom()})
            payload.push_back(toI32(v));
        auto pushI32Attr = [&](int32_t value) {
            payload.push_back(rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(value)));
        };
        pushI32Attr(static_cast<int32_t>(op.getPadValue()));
        pushI32Attr(static_cast<int32_t>(op.getRepeatStride()));
        pushI32Attr(static_cast<int32_t>(op.getRepeatTime()));
        pushI32Attr(static_cast<int32_t>(op.getRepeatMode()));

        SmallVector<Type, 42> operandTypes = {(*srcMemref).getType(), (*dstMemref).getType()};
        for (Value v : payload)
            operandTypes.push_back(v.getType());
        SmallVector<Value, 42> operands = {*srcMemref, *dstMemref};
        operands.append(payload.begin(), payload.end());

        auto callee = ::tla::getOrCreateRuntimeCall(op->getParentOfType<ModuleOp>(), calleeName, operandTypes);
        rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
        toErase.push_back(op.getOperation());
        return success();
    }

private:
    DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue;
    SmallVectorImpl<Operation*>& toErase;
    DenseMap<Value, Value>& loweredMemrefByValue;
};

struct LowerTlaCopyPattern : public OpRewritePattern<::tla::CopyOp> {
    LowerTlaCopyPattern(
        MLIRContext* ctx, DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue,
        SmallVectorImpl<Operation*>& toErase, DenseMap<Value, Value>& loweredMemrefByValue)
        : OpRewritePattern<::tla::CopyOp>(ctx),
          tensorDescriptorByValue(tensorDescriptorByValue),
          toErase(toErase),
          loweredMemrefByValue(loweredMemrefByValue)
    {}

    struct L0C2DstInfo {
        uint8_t unitFlag = 0;
        bool relu_enable = false;
        QuantMode quantMode = QuantMode::NO_QUANT;
        L0C2UBMode l0c2UbMode = L0C2UBMode::NO_SPLIT_VEC_0;
        uint8_t subBlockId = 0;
        Value quantScale;
        Value quantOffset;
        TensorDescriptor quantDesc;
    };

    /// Decode the tla.CopyL0C2DstParams operand of an L0C-source copy.
    ///
    /// Fills the fixpipe knobs the runtime call takes (unit flag, sub-block id)
    /// and appends the fragment that carries quant mode / relu to the runtime
    /// symbol name, in the order those names expect. The element types follow
    /// that fragment: the L0C routes carry *both* operands, because the
    /// destination alone does not identify the wrapper template (f32 -> f16
    /// and i32 -> f16 both exist).
    ///   l0c -> gm/l1 : {quantMode}_{relu}_{srcDtype}_{dstDtype}
    ///   l0c -> ub    : {splitMode}_{quantMode}_{relu}_{srcDtype}_{dstDtype}
    LogicalResult parseL0C2DstParams(
        ::tla::CopyOp op, StringRef dstAddrspace, const TensorDescriptor& srcDesc, const TensorDescriptor& dstDesc,
        L0C2DstInfo& info, std::string& extraDesc) const
    {
        auto params = op->getOperand(2);
        auto l0c2DstParamsOp = dyn_cast<::tla::CopyL0C2DstParamsOp>(params.getDefiningOp());
        if (!l0c2DstParamsOp) {
            op.emitError() << "expected tla.CopyL0C2DstParams as third operand";
            return failure();
        }
        info.unitFlag = static_cast<uint8_t>(l0c2DstParamsOp.getUnitFlag());
        info.relu_enable = l0c2DstParamsOp.getReluEnable();
        info.quantMode = l0c2DstParamsOp.getQuantMode().getQuantMode();
        if (dstAddrspace == "ub") {
            info.l0c2UbMode = l0c2DstParamsOp.getL0c2ubMode().getL0c2ubMode();
            if (info.l0c2UbMode == L0C2UBMode::NO_SPLIT_VEC_0) {
                extraDesc += "nosplit_";
            } else if (info.l0c2UbMode == L0C2UBMode::NO_SPLIT_VEC_1) {
                info.subBlockId = 1;
                extraDesc += "nosplit_";
            } else if (info.l0c2UbMode == L0C2UBMode::SPLIT_M) {
                extraDesc += "splitm_";
            } else if (info.l0c2UbMode == L0C2UBMode::SPLIT_N) {
                extraDesc += "splitn_";
            }
            if ((info.l0c2UbMode == L0C2UBMode::SPLIT_M || info.l0c2UbMode == L0C2UBMode::SPLIT_N) &&
                (srcDesc.elementType != dstDesc.elementType)) {
                op->emitError("When copy l0c to ub with split mode, src and dst type must be same");
                return failure();
            }
        }
        if (info.quantMode == QuantMode::PER_TENSOR) {
            extraDesc += "pertensor_";
            info.quantScale = l0c2DstParamsOp.getQuantScale();
            if (!info.quantScale) {
                op->emitError("PER_TENSOR copy requires a non-null quant_scale operand");
                return failure();
            }
            // quant_offset is optional: it only acts on the 8-bit destinations
            // and defaults to 0 at the runtime-call site below.
            info.quantOffset = l0c2DstParamsOp.getQuantOffset();
        } else if (info.quantMode == QuantMode::PER_CHANNEL) {
            extraDesc += "perchannel_";
            Value quantTensor = l0c2DstParamsOp.getQuantTensor();
            if (!quantTensor || !::llvm::isa<::tla::TlaTensorType>(quantTensor.getType())) {
                op->emitError("PER_CHANNEL copy requires a tensor quant_tensor operand");
                return failure();
            }
            auto quantIt = tensorDescriptorByValue.find(quantTensor);
            if (quantIt == tensorDescriptorByValue.end()) {
                op->emitError("missing descriptor for per-channel quant tensor");
                return failure();
            }
            const TensorDescriptor& quantDesc = quantIt->second;
            if (!validateTensorDescriptor(op, quantDesc, "malformed descriptor for per-channel quant tensor"))
                return failure();
            info.quantDesc = quantDesc;
        }
        if (info.relu_enable) {
            extraDesc += "relu_";
        }
        return success();
    }

    LogicalResult matchAndRewrite(::tla::CopyOp op, PatternRewriter& rewriter) const override
    {
        if ((op->getNumOperands() != 2 && op->getNumOperands() != 3) || op->getNumResults() != 0) {
            op.emitError() << "expected tla.copy to have 2 or 3 operands and 0 results";
            return failure();
        }

        Value dstTile = op->getOperand(0);
        Value srcTile = op->getOperand(1);
        auto dstIt = tensorDescriptorByValue.find(dstTile);
        auto srcIt = tensorDescriptorByValue.find(srcTile);
        if (dstIt == tensorDescriptorByValue.end()) {
            op.emitError() << "missing descriptor for tla.copy dst tile; expected a tla.tensor_desc "
                              "operand materialized by tla-lower-tensor-desc";
            return failure();
        }
        if (srcIt == tensorDescriptorByValue.end()) {
            op.emitError() << "missing descriptor for tla.copy src tile; expected a tla.tensor_desc "
                              "operand materialized by tla-lower-tensor-desc";
            return failure();
        }

        const TensorDescriptor& dstDesc = dstIt->second;
        const TensorDescriptor& srcDesc = srcIt->second;
        if (!::tla::validateTensorDescriptor(op, dstDesc, "malformed descriptor for tla.copy dst tile operand")) {
            return failure();
        }
        if (!::tla::validateTensorDescriptor(op, srcDesc, "malformed descriptor for tla.copy src tile operand")) {
            return failure();
        }
        StringRef srcAddrspace = srcDesc.addrspace;
        StringRef dstAddrspace = dstDesc.addrspace;
        std::string src2Dst = std::string(srcDesc.addrspace) + "2" + std::string(dstAddrspace);
        if (srcAddrspace == "l0c") {
            if (op->getNumOperands() != 3) {
                op.emitError() << "expected tla.copy " << src2Dst << " has 3 operands";
                return failure();
            }
        } else if (op->getNumOperands() == 3) {
            // GM→L1 VectorLayout may carry CopyPadParams; other cube routes stay 2-op.
            Value params = op.getParams();
            const bool gmToL1Pad = srcAddrspace == "gm" && dstAddrspace == "l1" && params &&
                                   mlir::isa<::tla::CopyPadParamsType>(params.getType());
            if (!gmToL1Pad) {
                op.emitError() << "expected tla.copy " << src2Dst << " has 2 operands";
                return failure();
            }
            if (srcDesc.layoutTag != LayoutTag::VectorLayout || dstDesc.layoutTag != LayoutTag::VectorLayout) {
                op.emitError() << "GM→L1 CopyPadParams require VectorLayout→VectorLayout";
                return failure();
            }
        } else if (op->getNumOperands() != 2) {
            op.emitError() << "expected tla.copy " << src2Dst << " has 2 operands";
            return failure();
        }
        auto buildRuntimeMemref = [&](const TensorDescriptor& desc) -> FailureOr<Value> {
            FailureOr<Value> baseMemref = ::tla::getOrMaterializeDescriptorBaseMemref(
                rewriter, op.getLoc(), desc, op.getOperation(), loweredMemrefByValue);
            if (failed(baseMemref))
                return failure();
            auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
            if (!baseType)
                return failure();
            MemRefType runtimeType = ::tla::getDynamicStridedMemrefType(baseType);
            return ::tla::castMemrefToType(rewriter, op.getLoc(), *baseMemref, runtimeType);
        };

        std::string extraDesc = "";
        L0C2DstInfo l0c2DstInfo;
        if (srcAddrspace == "l0c") {
            if (failed(parseL0C2DstParams(op, dstAddrspace, srcDesc, dstDesc, l0c2DstInfo, extraDesc)))
                return failure();
        }

        std::string calleeName = ::tla::getCopyRouteCallee(
            op.getContext(), srcAddrspace, dstAddrspace, srcDesc.layoutTag, dstDesc.layoutTag, srcDesc.elementType,
            dstDesc.elementType, extraDesc);
        const bool batched = !srcDesc.batch.empty() || !dstDesc.batch.empty();
        if (batched) {
            if (srcDesc.batch.size() != 4 || dstDesc.batch.size() != 4 || srcAddrspace != "gm" || dstAddrspace != "l1")
                return op.emitError("batch-matrix copy requires two GM/L1 batch descriptors");
            if (srcDesc.elementType != dstDesc.elementType ||
                !::tla::isSupportedBatchCopyLayout(srcDesc.layoutTag, dstDesc.layoutTag, srcDesc.elementType) ||
                calleeName.empty())
                return op.emitError("unsupported batch-matrix copy route");
            calleeName += "_batch";
        }
        if (!calleeName.empty()) {
            FailureOr<Value> dstRuntimeMemref = buildRuntimeMemref(dstDesc);
            FailureOr<Value> srcRuntimeMemref = buildRuntimeMemref(srcDesc);
            if (failed(dstRuntimeMemref) || failed(srcRuntimeMemref))
                return failure();
            SmallVector<Value, 24> payload = ::tla::buildCopyPayloadForRoute(rewriter, op.getLoc(), srcDesc, dstDesc);
            SmallVector<Type, 22> operandTypes = {(*srcRuntimeMemref).getType(), (*dstRuntimeMemref).getType()};
            operandTypes.reserve(2 + payload.size() + 4);
            for (Value payloadValue : payload)
                operandTypes.push_back(payloadValue.getType());
            SmallVector<Value, 22> operands = {*srcRuntimeMemref, *dstRuntimeMemref};
            operands.append(payload.begin(), payload.end());
            if (batched) {
                // Batch shape and both matrix steps come from the tensor layouts.
                for (Value v :
                     {srcDesc.batch[2], srcDesc.batch[1], dstDesc.batch[1], srcDesc.batch[3], dstDesc.batch[3]}) {
                    Value i64 = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getI64Type(), v);
                    operandTypes.push_back(rewriter.getI64Type());
                    operands.push_back(i64);
                }
            }
            if (srcAddrspace == "l0c") {
                auto i8Type = rewriter.getI8Type();
                auto unitFlagVal = rewriter.create<arith::ConstantIntOp>(op.getLoc(), l0c2DstInfo.unitFlag, 8);
                operandTypes.push_back(i8Type);
                operands.push_back(unitFlagVal);
                if (dstAddrspace == "ub") {
                    auto subBlockIdVal = rewriter.create<arith::ConstantIntOp>(op.getLoc(), l0c2DstInfo.subBlockId, 8);
                    operandTypes.push_back(i8Type);
                    operands.push_back(subBlockIdVal);
                }
                if (l0c2DstInfo.quantMode == QuantMode::PER_TENSOR) {
                    operandTypes.push_back(rewriter.getF32Type());
                    operands.push_back(l0c2DstInfo.quantScale);
                    // quant_offset is optional; the runtime ABI always takes an
                    // i32 offset, so default it to 0 when absent.
                    Value quantOffset = l0c2DstInfo.quantOffset;
                    if (!quantOffset)
                        quantOffset = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
                    operandTypes.push_back(rewriter.getI32Type());
                    operands.push_back(quantOffset);
                } else if (l0c2DstInfo.quantMode == QuantMode::PER_CHANNEL) {
                    FailureOr<Value> quantRuntimeMemref = buildRuntimeMemref(l0c2DstInfo.quantDesc);
                    if (failed(quantRuntimeMemref))
                        return failure();
                    operandTypes.push_back((*quantRuntimeMemref).getType());
                    operands.push_back(*quantRuntimeMemref);
                    SmallVector<Value, 12> quantPayload =
                        ::tla::buildCopyPayloadForDescriptor(rewriter, op.getLoc(), l0c2DstInfo.quantDesc);
                    for (Value quantValue : quantPayload) {
                        operandTypes.push_back(quantValue.getType());
                        operands.push_back(quantValue);
                    }
                }
            }
            // Trailing gmCachePolicy for GM-endpoint stubs (after unitFlag/quant).
            ::tla::appendGmCachePolicyOperand(
                rewriter, op.getLoc(), op, srcAddrspace, dstAddrspace, operandTypes, operands);
            // VL GM→L1 ciface always takes pad trailing args (zeros when no CopyPadParams).
            ::tla::appendGmToL1VectorPadOperands(
                rewriter, op.getLoc(), op, srcAddrspace, dstAddrspace, srcDesc.layoutTag, dstDesc.layoutTag,
                operandTypes, operands);
            auto callee = ::tla::getOrCreateRuntimeCall(op->getParentOfType<ModuleOp>(), calleeName, operandTypes);

            emitCopyRuntimeCall(rewriter, op, callee, operands, *dstRuntimeMemref);
            toErase.push_back(op.getOperation());
            return success();
        }

        // GM<->UB row-major copies are owned by LowerCopyPattern in
        // TlaVectorRegionCommon.cpp
        // (UB is vector-core memory); the cube pass only lowers the cube-side routes
        // above (L1 / L0A / L0B / L0C).

        op.emitError() << "tla.copy descriptor/layout combination is unsupported: " << srcAddrspace << "("
                       << ::stringifyLayoutTag(srcDesc.layoutTag) << ") -> " << dstAddrspace << "("
                       << ::stringifyLayoutTag(dstDesc.layoutTag) << ")";
        return failure();
    }

private:
    DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue;
    SmallVectorImpl<Operation*>& toErase;
    DenseMap<Value, Value>& loweredMemrefByValue;
};

// Materialize a flat rank-2 `!tla.shape` / `!tla.coord` aggregate operand into
// two index SSA values: static leaves become constants, dynamic leaves come from
// the defining make op's dyn operands. Mirrors the descriptor derivation's
// index-pair decoding, but emits through the pattern's rewriter.
static FailureOr<std::array<Value, 2>> materializeFillIndexPair(
    Operation* op, PatternRewriter& rewriter, Value aggregate, StringRef kind)
{
    ArrayRef<int64_t> tree;
    Operation* makeOp = nullptr;
    if (kind == "shape") {
        auto shapeTy = dyn_cast<::tla::ShapeType>(aggregate.getType());
        if (!shapeTy) {
            op->emitError("expected a tla.shape operand for tla.fill");
            return failure();
        }
        tree = shapeTy.getTree();
        makeOp = aggregate.getDefiningOp<::tla::MakeShapeOp>();
    } else {
        auto coordTy = dyn_cast<::tla::CoordType>(aggregate.getType());
        if (!coordTy) {
            op->emitError("expected a tla.coord operand for tla.fill");
            return failure();
        }
        tree = coordTy.getTree();
        makeOp = aggregate.getDefiningOp<::tla::MakeCoordOp>();
    }

    SmallVector<int64_t, 4> leaves;
    if (failed(::tla::getTlaIndexTreeLeaves(tree, leaves)) || tree.size() != 2 || leaves.size() != 2) {
        op->emitError() << "tla.fill " << kind << " operand must be a flat rank-2 aggregate";
        return failure();
    }
    if (llvm::any_of(leaves, [](int64_t leaf) { return leaf == ShapedType::kDynamic; }) && !makeOp) {
        op->emitError() << "dynamic tla.fill " << kind << " operands must come from a tla.make " << kind << " op";
        return failure();
    }

    size_t dynamicIndex = 0;
    std::array<Value, 2> result{};
    SmallVector<Value, 2> dynamicValues;
    if (auto makeShape = llvm::dyn_cast_or_null<::tla::MakeShapeOp>(makeOp)) {
        dynamicValues.append(makeShape.getDynElems().begin(), makeShape.getDynElems().end());
    } else if (auto makeCoord = llvm::dyn_cast_or_null<::tla::MakeCoordOp>(makeOp)) {
        dynamicValues.append(makeCoord.getDynElems().begin(), makeCoord.getDynElems().end());
    }
    for (auto [index, leaf] : llvm::enumerate(leaves)) {
        if (leaf == ShapedType::kDynamic) {
            if (dynamicIndex >= dynamicValues.size()) {
                op->emitError() << "tla.fill " << kind << " dynamic element count mismatch";
                return failure();
            }
            Value dynamicValue = dynamicValues[dynamicIndex++];
            if (!dynamicValue.getType().isIndex()) {
                op->emitError() << "tla.fill " << kind << " dynamic operands must be index type";
                return failure();
            }
            result[index] = dynamicValue;
            continue;
        }
        result[index] = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), leaf);
    }
    return result;
}

// tla.fill lowers to an inlinable L1 fill runtime template on the AIC core. The
// payload carries the destination's 12-field descriptor, the flat rank-2 fill
// region (the tile coord and origin shape), and the i32 fill bit pattern.
struct LowerTlaFillPattern : public OpRewritePattern<::tla::FillOp> {
    LowerTlaFillPattern(
        MLIRContext* ctx, DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue,
        SmallVectorImpl<Operation*>& toErase, DenseMap<Value, Value>& loweredMemrefByValue)
        : OpRewritePattern<::tla::FillOp>(ctx),
          tensorDescriptorByValue(tensorDescriptorByValue),
          toErase(toErase),
          loweredMemrefByValue(loweredMemrefByValue)
    {}

    LogicalResult matchAndRewrite(::tla::FillOp op, PatternRewriter& rewriter) const override
    {
        Value dstTile = op.getDst();
        auto dstIt = tensorDescriptorByValue.find(dstTile);
        if (dstIt == tensorDescriptorByValue.end()) {
            op.emitError() << "missing descriptor for tla.fill dst tile; expected a tla.tensor_desc "
                              "operand materialized by tla-lower-tensor-desc";
            return failure();
        }
        const TensorDescriptor& dstDesc = dstIt->second;
        if (!::tla::validateTensorDescriptor(op, dstDesc, "malformed descriptor for tla.fill dst tile operand")) {
            return failure();
        }

        std::string calleeName = ::tla::getFillCallee(dstDesc.layoutTag, dstDesc.elementType);
        if (calleeName.empty()) {
            op.emitError() << "tla.fill layout/element-type combination is unsupported: " << dstDesc.addrspace << "("
                           << ::stringifyLayoutTag(dstDesc.layoutTag) << ") " << dstDesc.elementType;
            return failure();
        }

        // Base memref, materialized the same way the copy routes do it.
        FailureOr<Value> baseMemref = ::tla::getOrMaterializeDescriptorBaseMemref(
            rewriter, op.getLoc(), dstDesc, op.getOperation(), loweredMemrefByValue);
        if (failed(baseMemref))
            return failure();
        auto baseType = dyn_cast<MemRefType>((*baseMemref).getType());
        if (!baseType)
            return failure();
        FailureOr<Value> runtimeMemref =
            ::tla::castMemrefToType(rewriter, op.getLoc(), *baseMemref, ::tla::getDynamicStridedMemrefType(baseType));
        if (failed(runtimeMemref))
            return failure();

        auto coordPair = materializeFillIndexPair(op, rewriter, op.getCoord(), "coord");
        if (failed(coordPair))
            return failure();
        auto shapePair = materializeFillIndexPair(op, rewriter, op.getShape(), "shape");
        if (failed(shapePair))
            return failure();

        SmallVector<Value, 18> payload = ::tla::buildCopyPayloadForDescriptor(rewriter, op.getLoc(), dstDesc);
        for (Value v : *coordPair)
            payload.push_back(::tla::castValueToI64(rewriter, op.getLoc(), v));
        for (Value v : *shapePair)
            payload.push_back(::tla::castValueToI64(rewriter, op.getLoc(), v));
        payload.push_back(op.getValue());

        SmallVector<Type, 18> operandTypes = {(*runtimeMemref).getType()};
        for (Value payloadValue : payload)
            operandTypes.push_back(payloadValue.getType());
        SmallVector<Value, 18> operands = {*runtimeMemref};
        operands.append(payload.begin(), payload.end());

        auto callee = ::tla::getOrCreateRuntimeCall(op->getParentOfType<ModuleOp>(), calleeName, operandTypes);
        rewriter.create<func::CallOp>(op.getLoc(), callee, operands);
        toErase.push_back(op.getOperation());
        return success();
    }

private:
    DenseMap<Value, TensorDescriptor>& tensorDescriptorByValue;
    SmallVectorImpl<Operation*>& toErase;
    DenseMap<Value, Value>& loweredMemrefByValue;
};

// Flatten a tla.cube region by splicing its body into the parent block.
struct LowerTlaCubePattern : public OpRewritePattern<::tla::CubeOp> {
    using OpRewritePattern<::tla::CubeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(::tla::CubeOp op, PatternRewriter& rewriter) const override
    {
        if (op->getNumRegions() == 0 || op->getRegion(0).empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        Block& body = op->getRegion(0).front();
        Block* parentBlock = op->getBlock();
        parentBlock->getOperations().splice(op->getIterator(), body.getOperations(), body.begin(), body.end());
        rewriter.eraseOp(op);
        return success();
    }
};

class TlaCubeRegionPass : public PassWrapper<TlaCubeRegionPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaCubeRegionPass)

    StringRef getArgument() const override
    {
        return "tla-cube-region";
    }
    StringRef getName() const override
    {
        return "TlaCubeRegionPass";
    }
    StringRef getDescription() const override
    {
        return "Lower tla.cube compute ops (tla.copy / tla.mmad) and flatten the cube region.";
    }
    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect, scf::SCFDialect, hivm::HIVMDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();

        // Drive one function at a time so the cube lowering only touches the cube
        // (AIC) function's ops, not a sibling vector (AIV) function in a mixed kernel.
        // Mirrors the tla.vector lowering passes, which iterate the module's functions and filter
        // by core type. Snapshot the functions up front: lowering appends runtime-call
        // declarations to the module, which must not be fed back through the loop.
        SmallVector<func::FuncOp, 4> funcOps(module.getOps<func::FuncOp>());
        for (func::FuncOp funcOp : funcOps) {
            if (funcOp.isDeclaration())
                continue;
            // Skip the generated vector_region helpers: they hold lowered AVE ops and
            // no cube work.
            if (funcOp->hasAttr(kHivmVectorFunctionAttrName))
                continue;
            // Only AIC (and not-yet-split MIX) functions hold cube work. Pure vector
            // (AIV) functions have no tla.cube / tla.copy / tla.mmad to lower.
            std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(funcOp.getOperation());
            if (coreType != hivm::TFuncCoreType::AIC && coreType != hivm::TFuncCoreType::MIX)
                continue;
            if (failed(runOnCubeFunction(funcOp))) {
                signalPassFailure();
                return;
            }
        }
    }

    // Lower all cube compute ops within a single cube (AIC/MIX) function. The
    // lowering state (descriptors, memref cache, staged erases) is fresh per
    // function, matching the tla.vector passes' per-function handoff. Only `root` (the
    // function) is threaded; the ModuleOp needed for runtime-symbol insertion is
    // derived on demand from the op being rewritten (getParentOfType<ModuleOp>()).
    LogicalResult runOnCubeFunction(func::FuncOp funcOp)
    {
        Operation* root = funcOp.getOperation();
        const bool hasExplicitCtrl = funcOp
                                         .walk([](Operation* op) {
                                             if (isa<ArchGetCtrlOp, ArchSetCtrlOp>(op))
                                                 return WalkResult::interrupt();
                                             return WalkResult::advance();
                                         })
                                         .wasInterrupted();
        // Scalar CTRL reads can observe the order of updates. Opaque calls can
        // also clobber CTRL, so set modes at each MMAD when either is present.
        const bool hasOpaqueCalls = funcOp
                                        .walk([](Operation* op) {
                                            if (isa<func::CallOp, CallExternOp>(op))
                                                return WalkResult::interrupt();
                                            return WalkResult::advance();
                                        })
                                        .wasInterrupted();
        SmallVector<Operation*, 8> toErase;
        ::tla::TlaTensorMemrefLowering lowering;
        auto& tensorDescriptorByValue = lowering.descriptorByValue;
        // Set on a lowering failure that does not abort the remaining work; reported
        // once at the end.
        bool passFailed = false;

        // Read the descriptors materialized by tla-lower-tensor-desc. Cube lowering
        // must not reconstruct metadata from raw tensor producer chains.
        if (failed(::tla::collectMaterializedTensorDescriptors(funcOp, tensorDescriptorByValue)))
            return failure();

        // tla.tensor_ptr / tla.ptr_add were already folded into the inttoptr byte
        // address by tla-lower-ptr (run before tla-lower-tensor-desc), so each
        // tensor_desc.base here is the raw inttoptr boundary and the copy / subview
        // materialization resolves it straight to a memref.

        // Descriptor-driven tla.copy lowering (supported v1 routes -> runtime calls;
        // unsupported combinations stay as tla.copy and fail legalization later).
        LowerTlaCopyPattern lowerCopy(&getContext(), tensorDescriptorByValue, toErase, lowering.loweredMemrefByValue);
        SmallVector<::tla::CopyOp, 16> copyOps;
        root->walk([&](::tla::CopyOp op) { copyOps.push_back(op); });
        bool copyLoweringFailed = false;
        for (::tla::CopyOp op : copyOps) {
            if (!op || !op->getBlock())
                continue;
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            if (failed(lowerCopy.matchAndRewrite(op, rewriter)))
                copyLoweringFailed = true;
        }
        if (copyLoweringFailed)
            passFailed = true;

        // tla.copy_mx runs on the same descriptors, before the cube region flattens.
        LowerTlaCopyMxPattern lowerCopyMx(
            &getContext(), tensorDescriptorByValue, toErase, lowering.loweredMemrefByValue);
        SmallVector<::tla::CopyMxOp, 8> copyMxOps;
        root->walk([&](::tla::CopyMxOp op) { copyMxOps.push_back(op); });
        for (::tla::CopyMxOp op : copyMxOps) {
            if (!op || !op->getBlock())
                continue;
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            if (failed(lowerCopyMx.matchAndRewrite(op, rewriter)))
                passFailed = true;
        }

        // tla.img2col runs on the same descriptors, for the same reason copy_mx does.
        LowerTlaImg2colPattern lowerImg2col(
            &getContext(), tensorDescriptorByValue, toErase, lowering.loweredMemrefByValue);
        SmallVector<::tla::Img2colOp, 8> img2colOps;
        root->walk([&](::tla::Img2colOp op) { img2colOps.push_back(op); });
        for (::tla::Img2colOp op : img2colOps) {
            if (!op || !op->getBlock())
                continue;
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            if (failed(lowerImg2col.matchAndRewrite(op, rewriter)))
                passFailed = true;
        }

        // tla.fill: initialize an L1 region with a constant value (zN/nZ destinations),
        // lowered like the copy routes so the region lands in the same MTE2 window.
        LowerTlaFillPattern lowerFill(&getContext(), tensorDescriptorByValue, toErase, lowering.loweredMemrefByValue);
        SmallVector<::tla::FillOp, 8> fillOps;
        root->walk([&](::tla::FillOp op) { fillOps.push_back(op); });
        for (::tla::FillOp op : fillOps) {
            if (!op || !op->getBlock())
                continue;
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            if (failed(lowerFill.matchAndRewrite(op, rewriter)))
                passFailed = true;
        }

        LowerTlaCubePattern lowerCube(&getContext());
        SmallVector<::tla::CubeOp, 4> cubeOps;
        root->walk<WalkOrder::PostOrder>([&](::tla::CubeOp op) { cubeOps.push_back(op); });
        for (::tla::CubeOp op : cubeOps) {
            if (!op || !op->getBlock())
                continue;
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            if (failed(lowerCube.matchAndRewrite(op, rewriter)))
                return failure();
        }

        SmallVector<Operation*, 16> mmadOps;
        root->walk([&](Operation* op) {
            if (llvm::isa<::tla::MmadOp, ::tla::MmadMxOp>(op))
                mmadOps.push_back(op);
        });
        ComputeOrderAttr uniformOrder;
        HF32ModeAttr uniformHF32;
        if (!hasExplicitCtrl && !hasOpaqueCalls && !mmadOps.empty()) {
            uniformOrder = getUniformMmadAttr<ComputeOrderAttr>(mmadOps, "compute_order");
            uniformHF32 = getUniformMmadAttr<HF32ModeAttr>(mmadOps, "hf32_mode");
            OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
            if (uniformOrder)
                setMmadComputeOrder(builder, funcOp.getLoc(), uniformOrder.getValue());
            if (uniformHF32)
                setMmadHF32Mode(builder, funcOp.getLoc(), uniformHF32.getValue());
        }

        LowerTlaMmadPattern lowerMmad(
            &getContext(), tensorDescriptorByValue, toErase, lowering.loweredMemrefByValue,
            static_cast<bool>(uniformOrder), static_cast<bool>(uniformHF32), hasExplicitCtrl);
        LowerTlaMmadMxPattern lowerMmadMx(
            &getContext(), tensorDescriptorByValue, toErase, lowering.loweredMemrefByValue,
            static_cast<bool>(uniformOrder), /*hasUniformHF32=*/false, hasExplicitCtrl);
        for (Operation* op : mmadOps) {
            if (!op->getBlock())
                continue;
            PatternRewriter rewriter(op->getContext());
            rewriter.setInsertionPoint(op);
            if (auto mmadOp = llvm::dyn_cast<::tla::MmadOp>(op)) {
                if (failed(lowerMmad.matchAndRewrite(mmadOp, rewriter))) {
                    return failure();
                }
            } else if (auto mmadMxOp = llvm::dyn_cast<::tla::MmadMxOp>(op)) {
                if (failed(lowerMmadMx.matchAndRewrite(mmadMxOp, rewriter))) {
                    return failure();
                }
            }
        }

        // The tla.tensor_desc ops are dead (tla.copy / tla.mmad were lowered off their
        // descriptors, not their values), but they still hold their `base` scaffolding
        // (the inttoptr boundary) which the base-memref materialization staged for
        // erasure. Stage the dead tensor_descs so the flush erases them first; that
        // scaffolding cannot be erased while a live tensor_desc still references it.
        ::tla::stageDeadTensorDescriptors(root, toErase);

        // Flush staged erases: the lowered tla.copy / tla.mmad ops (which
        // tla-finalize-memref marks illegal, so they must be erased here), the dead
        // tla.tensor_desc ops, and the ptr bridges / scaffolding consumed while
        // materializing their tile memrefs.
        DenseSet<Operation*> pendingErase;
        for (Operation* op : toErase)
            if (op && op->getBlock())
                pendingErase.insert(op);
        bool progress = true;
        while (progress && !pendingErase.empty()) {
            progress = false;
            for (Operation* op : toErase) {
                if (!op || !pendingErase.contains(op) || !op->getBlock())
                    continue;
                bool hasLiveResultUses = false;
                for (Value result : op->getResults())
                    if (!result.use_empty()) {
                        hasLiveResultUses = true;
                        break;
                    }
                if (hasLiveResultUses)
                    continue;
                pendingErase.erase(op);
                op->erase();
                progress = true;
            }
        }
        if (!pendingErase.empty()) {
            for (Operation* op : pendingErase)
                op->emitError() << "staged erase failed for '" << op->getName().getStringRef()
                                << "' in tla-cube-region: operation still has live result users";
            return failure();
        }
        return passFailed ? failure() : success();
    }
};

} // namespace

std::unique_ptr<Pass> createTlaCubeRegionPass()
{
    return std::make_unique<TlaCubeRegionPass>();
}

void registerTlaCubeRegionPass()
{
    PassRegistration<TlaCubeRegionPass>();
}

} // namespace tla
