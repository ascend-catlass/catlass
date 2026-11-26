#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaDebugPrint.h"
#include "Passes/TlaTensorToMemref.h"
#include "Passes/TlaVectorRegionCommon.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

namespace tla {
using namespace ::tla::vecregion;
namespace {

/// Align PostMode UPDATE: emit HIVM ``.post`` load/store (UnAlign is not this path).
static Type getUbPtrType(MLIRContext* ctx)
{
    return LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/6);
}

static Value intToUbPtr(OpBuilder& b, Location loc, Value addrI64)
{
    Value asI64 = castValueToI64(b, loc, addrI64);
    return b.create<LLVM::IntToPtrOp>(loc, getUbPtrType(b.getContext()), asI64);
}

/// AscendC ``vlds/vsts`` POST stride is elements; HIVM ``.post`` offset is bytes.
static Value strideElemsToByteOffI32(OpBuilder& b, Location loc, Value strideElems, Type elemType)
{
    int64_t elemBytes = elementStorageBytes(elemType);
    Value s = ::tla::castToI32(b, loc, strideElems);
    if (elemBytes == 1)
        return s;
    Value scale = b.create<arith::ConstantIntOp>(loc, elemBytes, /*width=*/32);
    return b.create<arith::MulIOp>(loc, s, scale);
}

/// Align b64 UPDATE follows AscendC RegTraitNumOne: reinterpret as
/// ``uint32_t``, ``postUpdateStride * 2`` in u32 elems (= same byte off as
/// ``stride * 8``), DistVST ``NORM_B32``. ``TypeGet<double>`` → ``vector_u64``,
/// same ActualT==8 path as i64/u64.
static bool isAlignB64Elem(Type elemType)
{
    return elemType.isInteger(64) || elemType.isF64();
}

/// Align UPDATE has no ``vldsx1.post.v256f8*`` / f4 / e8m0 / Bool post; these
/// move as s8 bits (AscendC ``SupportBytes<1>()`` / ``TypeGet<bool>`` →
/// ``vector_s8`` → ``uint8_t``/``int8_t`` vlds/vsts).
static bool isAlignPostByteAliasElem(Type elemType)
{
    return isBoolElementType(elemType) ||
           isa<Float8E4M3FNType, Float8E5M2Type, ::tla::HiFloat8Type, ::tla::Float8E8M0Type, ::tla::Float4E2M1x2Type,
               ::tla::Float4E1M2x2Type, ::tla::Int4x2Type>(elemType);
}

/// True when the integer element should take the unsigned HIVM load/store
/// variant; signed and signless share the signed variant.
static bool isUnsignedHivmElem(Type elemType)
{
    auto intTy = dyn_cast<IntegerType>(elemType);
    return intTy && intTy.isUnsigned();
}

/// Cast ``newPtr`` to the live UB pointer type without UnrealizedConversionCast
/// (those survive into the VF and tank VLOOP / post-update recognition).
static Value castToLiveUbPtr(OpBuilder& b, Location loc, Value newPtr, Type ubPtrTy)
{
    if (newPtr.getType() == ubPtrTy)
        return newPtr;
    auto fromTy = dyn_cast<LLVM::LLVMPointerType>(newPtr.getType());
    auto toTy = dyn_cast<LLVM::LLVMPointerType>(ubPtrTy);
    if (fromTy && toTy && fromTy.getAddressSpace() != toTy.getAddressSpace())
        return b.create<LLVM::AddrSpaceCastOp>(loc, ubPtrTy, newPtr);
    return b.create<LLVM::BitcastOp>(loc, ubPtrTy, newPtr);
}

/// Emit ``llvm.hivm.vldsx1.post.*`` and return ``{vec, newPtr}``.
static FailureOr<std::pair<Value, Value>> emitVldsx1Post(
    OpBuilder& b, Location loc, Value ptr, Value strideElems, Type elemType, Type vecTy)
{
    Value byteOff = strideElemsToByteOffI32(b, loc, strideElems, elemType);
    Value dist = b.create<arith::ConstantIntOp>(loc, /*value=*/0, /*width=*/32);
    Value postEn = b.create<arith::ConstantIntOp>(loc, /*value=*/1, /*width=*/32);
    Type ubPtrTy = getUbPtrType(b.getContext());
    Type structTy = LLVM::LLVMStructType::getLiteral(b.getContext(), {vecTy, ubPtrTy});
    Operation* op = nullptr;
    Value vec;
    Value newPtr;
    if (isAlignPostByteAliasElem(elemType)) {
        // No Align ``.post.v256f8*``; HIVM only exposes s8/u8 post for b8.
        Type i8VecTy = VectorType::get({256}, IntegerType::get(b.getContext(), 8));
        Type s8StructTy = LLVM::LLVMStructType::getLiteral(b.getContext(), {i8VecTy, ubPtrTy});
        op = b.create<hivm_regbaseintrins::Vldsx1PostV256S8InstrOp>(loc, s8StructTy, ptr, byteOff, dist, postEn);
        Value i8Vec = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 0);
        newPtr = castToLiveUbPtr(b, loc, b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 1), ubPtrTy);
        vec = (vecTy == i8VecTy) ? i8Vec : b.create<mlir::vector::BitCastOp>(loc, vecTy, i8Vec).getResult();
        return std::make_pair(vec, newPtr);
    }
    if (elemType.isF32())
        op = b.create<hivm_regbaseintrins::Vldsx1PostV64F32InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
    else if (elemType.isF16())
        op = b.create<hivm_regbaseintrins::Vldsx1PostV128F16InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
    else if (elemType.isBF16())
        op = b.create<hivm_regbaseintrins::Vldsx1PostV128BF16InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
    else if (elemType.isInteger(8)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::Vldsx1PostV256U8InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
        else
            op = b.create<hivm_regbaseintrins::Vldsx1PostV256S8InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
    } else if (elemType.isInteger(16)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::Vldsx1PostV128U16InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
        else
            op = b.create<hivm_regbaseintrins::Vldsx1PostV128S16InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
    } else if (elemType.isInteger(32)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::Vldsx1PostV64U32InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
        else
            op = b.create<hivm_regbaseintrins::Vldsx1PostV64S32InstrOp>(loc, structTy, ptr, byteOff, dist, postEn);
    } else if (isAlignB64Elem(elemType)) {
        // AscendC: vlds((RegTensor<uint32_t>&)dst, uint32_t*&, stride*2, …).
        // byteOff stays stride_elems×8 (equivalent to (stride×2)×sizeof(u32)).
        // LLVM dialect structs require signless element types — use i32 carrier
        // (still the U32 intrinsic) then bitcast back to the semantic vecTy.
        Type i32 = IntegerType::get(b.getContext(), 32);
        Type i32VecTy = VectorType::get({64}, i32);
        Type i32StructTy = LLVM::LLVMStructType::getLiteral(b.getContext(), {i32VecTy, ubPtrTy});
        op = b.create<hivm_regbaseintrins::Vldsx1PostV64U32InstrOp>(loc, i32StructTy, ptr, byteOff, dist, postEn);
        Value i32Vec = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 0);
        newPtr = castToLiveUbPtr(b, loc, b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 1), ubPtrTy);
        vec = (vecTy == i32VecTy) ? i32Vec : b.create<mlir::vector::BitCastOp>(loc, vecTy, i32Vec).getResult();
        return std::make_pair(vec, newPtr);
    } else
        return failure();
    vec = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 0);
    newPtr = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 1);
    newPtr = castToLiveUbPtr(b, loc, newPtr, ubPtrTy);
    return std::make_pair(vec, newPtr);
}

/// Emit Align ``vstsx1.post`` and return HW ``newPtr``.
static FailureOr<Value> emitVstsx1Post(
    OpBuilder& b, Location loc, Value src, Value ptr, Value strideElems, Value preg, Type elemType)
{
    Value byteOff = strideElemsToByteOffI32(b, loc, strideElems, elemType);
    // DistVST NORM_B8/B16/B32 encodes 0/1/2; b64 uses the b32 carrier.
    const int64_t elemBytes = elementStorageBytes(elemType);
    const int64_t normDist = elemBytes == 1 ? 0 : elemBytes == 2 ? 1 : 2;
    Value dist = b.create<arith::ConstantIntOp>(loc, normDist, /*width=*/32);
    Value postEn = b.create<arith::ConstantIntOp>(loc, /*value=*/1, /*width=*/32);
    Type ubPtrTy = getUbPtrType(b.getContext());
    Operation* op = nullptr;
    if (isAlignPostByteAliasElem(elemType)) {
        Type i8VecTy = VectorType::get({256}, IntegerType::get(b.getContext(), 8));
        Value srcI8 =
            (src.getType() == i8VecTy) ? src : b.create<mlir::vector::BitCastOp>(loc, i8VecTy, src).getResult();
        op = b.create<hivm_regbaseintrins::Vstsx1PostV256S8InstrOp>(
            loc, ubPtrTy, srcI8, ptr, byteOff, dist, postEn, preg);
        return castToLiveUbPtr(b, loc, op->getResult(0), ubPtrTy);
    }
    if (elemType.isF32())
        op =
            b.create<hivm_regbaseintrins::Vstsx1PostV64F32InstrOp>(loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
    else if (elemType.isF16())
        op = b.create<hivm_regbaseintrins::Vstsx1PostV128F16InstrOp>(
            loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
    else if (elemType.isBF16())
        op = b.create<hivm_regbaseintrins::Vstsx1PostV128BF16InstrOp>(
            loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
    else if (elemType.isInteger(8)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::Vstsx1PostV256U8InstrOp>(
                loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
        else
            op = b.create<hivm_regbaseintrins::Vstsx1PostV256S8InstrOp>(
                loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
    } else if (elemType.isInteger(16)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::Vstsx1PostV128U16InstrOp>(
                loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
        else
            op = b.create<hivm_regbaseintrins::Vstsx1PostV128S16InstrOp>(
                loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
    } else if (elemType.isInteger(32)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::Vstsx1PostV64U32InstrOp>(
                loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
        else
            op = b.create<hivm_regbaseintrins::Vstsx1PostV64S32InstrOp>(
                loc, ubPtrTy, src, ptr, byteOff, dist, postEn, preg);
    } else if (isAlignB64Elem(elemType)) {
        // AscendC: vsts((RegTensor<uint32_t>&)src, uint32_t*&, stride*2,
        //               DIST_NORM_B32, expandedMask, POST). Caller must pass
        //               the ppack+pintlv_b32 expanded preg (see lowerStoreUpdate).
        // Signless i32 carrier for LLVM dialect; U32 intrinsic encodes unsigned.
        Type i32 = IntegerType::get(b.getContext(), 32);
        Type i32VecTy = VectorType::get({64}, i32);
        Value srcI32 =
            (src.getType() == i32VecTy) ? src : b.create<mlir::vector::BitCastOp>(loc, i32VecTy, src).getResult();
        op = b.create<hivm_regbaseintrins::Vstsx1PostV64U32InstrOp>(
            loc, ubPtrTy, srcI32, ptr, byteOff, dist, postEn, preg);
    } else
        return failure();
    return castToLiveUbPtr(b, loc, op->getResult(0), ubPtrTy);
}

static Type uregVectorType(OpBuilder& b)
{
    return VectorType::get({32}, b.getI8Type());
}

static Type dataVectorType(OpBuilder& b, Type elemType)
{
    if (elemType.isF32())
        return VectorType::get({64}, b.getF32Type());
    if (elemType.isF16())
        return VectorType::get({128}, b.getF16Type());
    if (elemType.isBF16())
        return VectorType::get({128}, b.getBF16Type());
    // AscendC ``TypeGet<double>`` → ``vector_u64``; AVE keeps f64, UnAlign post uses s64 bits.
    if (elemType.isF64())
        return VectorType::get({32}, b.getF64Type());
    // UnAlign has native ``vldus/vstus.post.f8e*``; AVE keeps the fp8 element.
    if (isa<Float8E4M3FNType>(elemType))
        return VectorType::get({256}, Float8E4M3FNType::get(b.getContext()));
    if (isa<Float8E5M2Type>(elemType))
        return VectorType::get({256}, Float8E5M2Type::get(b.getContext()));
    // Bool / HiFloat8 / e8m0 / fp4x2: no HIVM unalign mover for the IR type;
    // AVE payload is i8 (AscendC UnAlign UPDATE casts these to ``uint8_t`` /
    // ``int8_t``; ``TypeGet<bool>`` → ``vector_s8``).
    if (isBoolElementType(elemType) || isa<::tla::HiFloat8Type, ::tla::Float8E8M0Type, ::tla::Float4E2M1x2Type,
                                           ::tla::Float4E1M2x2Type, ::tla::Int4x2Type>(elemType))
        return VectorType::get({256}, IntegerType::get(b.getContext(), 8));
    if (auto intTy = dyn_cast<IntegerType>(elemType)) {
        unsigned width = intTy.getWidth();
        if (width != 8 && width != 16 && width != 32 && width != 64)
            return {};
        // AVE rejects ui*; carry unsigned as signless of the same width.
        Type aveElem = intTy.isUnsigned() ? IntegerType::get(b.getContext(), width) : Type(intTy);
        return VectorType::get({static_cast<int64_t>(256 / (width / 8))}, aveElem);
    }
    return {};
}

/// Emit load-side UnAlignReg Pre (AscendC ``LoadUnAlignPre`` / ``vldas``).
/// ``tla-unalign-ureg`` inserts ``load_unalign_pre`` before the first UPDATE.
static FailureOr<Value> emitVldas(OpBuilder& b, Location loc, Value ptr)
{
    auto op = b.create<hivm_regbaseintrins::VldasInstrOp>(loc, uregVectorType(b), ptr);
    return op.getResult();
}

/// Empty store ureg (AscendC ``UnalignRegForStore`` — no ``vldas`` / Pre).
/// HIVM ``hivm.init.vector.align.data``; same as AVE unaligned store path.
static Value emitEmptyStoreUreg(OpBuilder& b, Location loc)
{
    return b.create<hivm_regbaseintrins::InitVectorAlignDataInstrOp>(loc, uregVectorType(b)).getResult();
}

static Value getVectorHelperStorageIdentity(Value source, Region& outlinedRegion);

/// Emit ``vldus.post`` and return ``{vec, newUreg, newPtr}``.
///
/// ``$cceVL`` is the AscendC ``postUpdateStride`` in **bytes**
/// (``stride_elems × sizeof(T)``, same as Align UPDATE). HW ``newPtr`` is the
/// TLA ``next_ptr`` — no software pointer arithmetic. ``ureg`` is carried SSA
/// from ``load_unalign_pre`` / prior UPDATE ``ureg_out``.
static FailureOr<std::tuple<Value, Value, Value>> emitVldusPost(
    OpBuilder& b, Location loc, Value ptr, Value ureg, Value countBytes, Type elemType)
{
    Type vecTy = dataVectorType(b, elemType);
    if (!vecTy)
        return failure();
    Type structTy =
        LLVM::LLVMStructType::getLiteral(b.getContext(), {vecTy, uregVectorType(b), getUbPtrType(b.getContext())});
    Operation* op = nullptr;
    if (elemType.isF32())
        op = b.create<hivm_regbaseintrins::VldusPostF32InstrOp>(loc, structTy, ptr, ureg, countBytes);
    else if (elemType.isF16())
        op = b.create<hivm_regbaseintrins::VldusPostF16InstrOp>(loc, structTy, ptr, ureg, countBytes);
    else if (elemType.isBF16())
        op = b.create<hivm_regbaseintrins::VldusPostBF16InstrOp>(loc, structTy, ptr, ureg, countBytes);
    else if (elemType.isInteger(8)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::VldusPostU8InstrOp>(loc, structTy, ptr, ureg, countBytes);
        else
            op = b.create<hivm_regbaseintrins::VldusPostS8InstrOp>(loc, structTy, ptr, ureg, countBytes);
    } else if (elemType.isInteger(16)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::VldusPostU16InstrOp>(loc, structTy, ptr, ureg, countBytes);
        else
            op = b.create<hivm_regbaseintrins::VldusPostS16InstrOp>(loc, structTy, ptr, ureg, countBytes);
    } else if (elemType.isInteger(32)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::VldusPostU32InstrOp>(loc, structTy, ptr, ureg, countBytes);
        else
            op = b.create<hivm_regbaseintrins::VldusPostS32InstrOp>(loc, structTy, ptr, ureg, countBytes);
    } else if (elemType.isInteger(64))
        // HIVM exposes only the s64 post form; u64 bits share that path.
        op = b.create<hivm_regbaseintrins::VldusPostS64InstrOp>(loc, structTy, ptr, ureg, countBytes);
    else if (elemType.isF64()) {
        // No ``vldus.post.f64``; AscendC → ``vector_u64``, HIVM s64 post + bitcast.
        Type i64VecTy = VectorType::get({32}, IntegerType::get(b.getContext(), 64));
        Type s64StructTy = LLVM::LLVMStructType::getLiteral(
            b.getContext(), {i64VecTy, uregVectorType(b), getUbPtrType(b.getContext())});
        op = b.create<hivm_regbaseintrins::VldusPostS64InstrOp>(loc, s64StructTy, ptr, ureg, countBytes);
        Value i64Vec = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 0);
        Value newUreg = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 1);
        Value newPtr = castToLiveUbPtr(
            b, loc, b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 2), getUbPtrType(b.getContext()));
        Value f64Vec = b.create<mlir::vector::BitCastOp>(loc, vecTy, i64Vec).getResult();
        return std::make_tuple(f64Vec, newUreg, newPtr);
    } else if (isa<Float8E4M3FNType>(elemType))
        op = b.create<hivm_regbaseintrins::VldusPostF8E4InstrOp>(loc, structTy, ptr, ureg, countBytes);
    else if (isa<Float8E5M2Type>(elemType))
        op = b.create<hivm_regbaseintrins::VldusPostF8E5InstrOp>(loc, structTy, ptr, ureg, countBytes);
    else if (
        isBoolElementType(elemType) || isa<::tla::HiFloat8Type, ::tla::Float8E8M0Type, ::tla::Float4E2M1x2Type,
                                           ::tla::Float4E1M2x2Type, ::tla::Int4x2Type>(elemType))
        // No ``vldus.post`` for Bool/hif8/e8m0/fp4x2; AscendC → int8_t/uint8_t.
        op = b.create<hivm_regbaseintrins::VldusPostS8InstrOp>(loc, structTy, ptr, ureg, countBytes);
    else
        return failure();
    Value vec = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 0);
    Value newUreg = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 1);
    Value newPtr =
        castToLiveUbPtr(b, loc, b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 2), getUbPtrType(b.getContext()));
    return std::make_tuple(vec, newUreg, newPtr);
}

/// Emit ``vstus.post`` and return ``{newUreg, newPtr}``.
///
/// ``$eleSize`` is ``postUpdateStride`` in bytes (same rule as ``emitVldusPost``).
static FailureOr<std::pair<Value, Value>> emitVstusPost(
    OpBuilder& b, Location loc, Value src, Value ptr, Value ureg, Value countBytes, Type elemType)
{
    Type structTy = LLVM::LLVMStructType::getLiteral(b.getContext(), {uregVectorType(b), getUbPtrType(b.getContext())});
    Operation* op = nullptr;
    if (elemType.isF32())
        op = b.create<hivm_regbaseintrins::VstusPostF32InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    else if (elemType.isF16())
        op = b.create<hivm_regbaseintrins::VstusPostF16InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    else if (elemType.isBF16())
        op = b.create<hivm_regbaseintrins::VstusPostBF16InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    else if (elemType.isInteger(8)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::VstusPostU8InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
        else
            op = b.create<hivm_regbaseintrins::VstusPostS8InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    } else if (elemType.isInteger(16)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::VstusPostU16InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
        else
            op = b.create<hivm_regbaseintrins::VstusPostS16InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    } else if (elemType.isInteger(32)) {
        if (isUnsignedHivmElem(elemType))
            op = b.create<hivm_regbaseintrins::VstusPostU32InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
        else
            op = b.create<hivm_regbaseintrins::VstusPostS32InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    } else if (elemType.isInteger(64) || elemType.isF64()) {
        // HIVM s64 post; f64 AVE carrier may still be f64 — bitcast to s64 bits.
        Type i64VecTy = VectorType::get({32}, IntegerType::get(b.getContext(), 64));
        Value srcI64 =
            (src.getType() == i64VecTy) ? src : b.create<mlir::vector::BitCastOp>(loc, i64VecTy, src).getResult();
        op = b.create<hivm_regbaseintrins::VstusPostS64InstrOp>(loc, structTy, srcI64, ptr, countBytes, ureg);
    } else if (isa<Float8E4M3FNType>(elemType))
        op = b.create<hivm_regbaseintrins::VstusPostF8E4InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    else if (isa<Float8E5M2Type>(elemType))
        op = b.create<hivm_regbaseintrins::VstusPostF8E5InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    else if (
        isBoolElementType(elemType) || isa<::tla::HiFloat8Type, ::tla::Float8E8M0Type, ::tla::Float4E2M1x2Type,
                                           ::tla::Float4E1M2x2Type, ::tla::Int4x2Type>(elemType))
        // No ``vstus.post`` for Bool/hif8/e8m0/fp4x2; AscendC → int8_t/uint8_t.
        op = b.create<hivm_regbaseintrins::VstusPostS8InstrOp>(loc, structTy, src, ptr, countBytes, ureg);
    else
        return failure();
    Value newUreg = b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 0);
    Value newPtr =
        castToLiveUbPtr(b, loc, b.create<LLVM::ExtractValueOp>(loc, op->getResult(0), 1), getUbPtrType(b.getContext()));
    return std::make_pair(newUreg, newPtr);
}

static LogicalResult emitVstas(OpBuilder& b, Location loc, Value ureg, Value ptr)
{
    // ``hivm.vstas`` args: (data, base, offset, postUpdateEnable).
    // Non-post flush of the align ureg: offset=0, postUpdateEnable=0 (pointer
    // advance already done by ``vstus.post``). Same as AscendC ``vstar`` /
    // ``__builtin_cce_vstas_*(..., 0, 0 /* pre */)`` and HIVM ``buildVstasOp``.
    // Do not pass postUpdateEnable=1 here — that belongs to ``hivm.vstas.post``.
    Value zero = b.create<arith::ConstantIntOp>(loc, /*value=*/0, /*width=*/32);
    b.create<hivm_regbaseintrins::VstasInstrOp>(loc, ureg, ptr, zero, zero);
    return success();
}

static hivmave::VFPgeOp createAvePgeMask(OpBuilder& b, Location loc, VectorType maskType, hivmave::PgePattern pattern)
{
    return b.create<hivmave::VFPgeOp>(loc, maskType, pattern);
}

static hivmave::VFPltOp createAvePltMask(OpBuilder& b, Location loc, VectorType maskType, Value trueShape)
{
    return b.create<hivmave::VFPltOp>(loc, maskType, b.getIndexType(), trueShape);
}

static hivmave::LoadDist mapTlaLoadDistToAve(::LoadDist dist)
{
    switch (dist) {
        case ::LoadDist::norm:
            return hivmave::LoadDist::NORM;
        case ::LoadDist::brc_b32:
            return hivmave::LoadDist::BRC_B32;
        case ::LoadDist::brc_b8:
            return hivmave::LoadDist::BRC_B8;
        case ::LoadDist::ds_b8:
            return hivmave::LoadDist::DS_B8;
        case ::LoadDist::ds_b16:
            return hivmave::LoadDist::DS_B16;
        case ::LoadDist::unpack_b8:
            return hivmave::LoadDist::UNPK_B8;
        case ::LoadDist::unpack_b32:
            return hivmave::LoadDist::UNPK_B32;
        case ::LoadDist::dintlv_b8:
            return hivmave::LoadDist::DINTLV_B8;
        case ::LoadDist::dintlv_b32:
            return hivmave::LoadDist::DINTLV_B32;
        case ::LoadDist::dintlv_b16:
            return hivmave::LoadDist::DINTLV_B16;
        case ::LoadDist::us_b8:
            return hivmave::LoadDist::US_B8;
        case ::LoadDist::brc_b16:
            return hivmave::LoadDist::BRC_B16;
        case ::LoadDist::us_b16:
            return hivmave::LoadDist::US_B16;
        case ::LoadDist::unpack_b16:
            return hivmave::LoadDist::UNPK_B16;
        case ::LoadDist::unpack4_b8:
            return hivmave::LoadDist::UNPK4_B8;
        case ::LoadDist::e2b_b16:
            return hivmave::LoadDist::E2B_B16;
        case ::LoadDist::e2b_b32:
            return hivmave::LoadDist::E2B_B32;
        case ::LoadDist::blk:
            return hivmave::LoadDist::BLK;
        // MaskDist US/DS: IR may carry these enums, but lowering uses AIV bc
        // (NPU-IR i1→plds hardcodes NORM). Keep map entries for completeness.
        case ::LoadDist::us:
            return hivmave::LoadDist::US;
        case ::LoadDist::ds:
            return hivmave::LoadDist::DS;
    }
    llvm_unreachable("unsupported tla.load load_dist");
}

static hivmave::StoreDist mapTlaStoreDistToAve(::StoreDist dist)
{
    switch (dist) {
        case ::StoreDist::pack_b32:
            return hivmave::StoreDist::PK_B32;
        case ::StoreDist::pack_b16:
            return hivmave::StoreDist::PK_B16;
        case ::StoreDist::pack_b64:
            return hivmave::StoreDist::PK_B64;
        case ::StoreDist::pack4_b32:
            return hivmave::StoreDist::PK4_B32;
        case ::StoreDist::first_element_b8:
            return hivmave::StoreDist::ONEPT_B8;
        case ::StoreDist::first_element_b16:
            return hivmave::StoreDist::ONEPT_B16;
        case ::StoreDist::first_element_b32:
            return hivmave::StoreDist::ONEPT_B32;
        case ::StoreDist::intlv_b8:
            return hivmave::StoreDist::INTLV_B8;
        case ::StoreDist::intlv_b16:
            return hivmave::StoreDist::INTLV_B16;
        case ::StoreDist::intlv_b32:
            return hivmave::StoreDist::INTLV_B32;
        // If other StoreDist added, complete here
        default:
            return hivmave::StoreDist::NORM_B8;
    }
}

static bool isDualDestLoadDist(hivmave::LoadDist pattern)
{
    return pattern == hivmave::LoadDist::DINTLV_B8 || pattern == hivmave::LoadDist::DINTLV_B16 ||
           pattern == hivmave::LoadDist::DINTLV_B32;
}

static bool isDualSourceStoreDist(hivmave::StoreDist pattern)
{
    return pattern == hivmave::StoreDist::INTLV_B8 || pattern == hivmave::StoreDist::INTLV_B16 ||
           pattern == hivmave::StoreDist::INTLV_B32;
}

/// Shared Bool predicate (SoT: ``tla::isBoolElementType``).
static bool isBoolVectorElement(Type elementType)
{
    return isBoolElementType(elementType);
}

static hivmave::VFLoadOp createVFLoad(
    OpBuilder& b, Location loc, VectorType vecType, Value memref, Value index, hivmave::LoadDist pattern,
    bool unaligned)
{
    // Always pass ``pattern`` at create time. Dual-destination dists need two
    // result types up front (AVE's convenience builder only emits a single NORM
    // result), and single-destination non-NORM (e.g. BRC_B32) can use the same
    // pattern-in-create overload instead of create-then-setPattern.
    SmallVector<Type, 2> resultTypes;
    resultTypes.push_back(vecType);
    if (isDualDestLoadDist(pattern))
        resultTypes.push_back(vecType);

    auto load = b.create<hivmave::VFLoadOp>(loc, resultTypes, pattern, memref, ValueRange{index});
    if (unaligned)
        load->setAttr(hivmave::UnalignedAttr::name, hivmave::UnalignedAttr::get(b.getContext()));
    return load;
}

/// Byte address of a memref view: base + offset×elementStorageBytes.
static FailureOr<Value> memrefByteAddrI64(OpBuilder& b, Location loc, Value ubMemref)
{
    auto srcTy = dyn_cast<MemRefType>(ubMemref.getType());
    if (!srcTy)
        return failure();
    int64_t elemBytes = elementStorageBytes(srcTy.getElementType());
    if (elemBytes <= 0)
        return failure();
    auto meta = b.create<mlir::memref::ExtractStridedMetadataOp>(loc, ubMemref);
    Value basePtr = b.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, meta.getBaseBuffer());
    Value byteOffset = meta.getOffset();
    if (elemBytes != 1) {
        Value scale = b.create<arith::ConstantIndexOp>(loc, elemBytes);
        byteOffset = b.create<arith::MulIOp>(loc, byteOffset, scale);
    }
    Value byteAddr = b.create<arith::AddIOp>(loc, basePtr, byteOffset);
    return b.create<arith::IndexCastOp>(loc, b.getI64Type(), byteAddr).getResult();
}

// Map a 1/2/4/8-byte MaskSSA UB tile memref onto an i1 view of ``lanes``
// predicate bits at the same byte address (AscendC MaskDist::DIST_NORM /
// plds/psts.b8). UB element width types the pointer; the hardware transfer
// is always the bit-level plds/psts.b8 path.
static FailureOr<Value> materializeI1MaskMemrefFromUb(OpBuilder& b, Location loc, Value ubMemref, int64_t lanes)
{
    if (lanes <= 0 || lanes % 8 != 0)
        return failure();
    auto srcTy = dyn_cast<MemRefType>(ubMemref.getType());
    if (!srcTy)
        return failure();
    int64_t elemBytes = elementStorageBytes(srcTy.getElementType());
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4 && elemBytes != 8)
        return failure();
    auto addrOr = memrefByteAddrI64(b, loc, ubMemref);
    if (failed(addrOr))
        return failure();
    auto i1Ty = IntegerType::get(b.getContext(), 1);
    auto layout = StridedLayoutAttr::get(b.getContext(), /*offset=*/0, ArrayRef<int64_t>{1});
    auto i1MemrefTy = MemRefType::get({lanes}, i1Ty, layout, srcTy.getMemorySpace());
    return b.create<hivm::PointerCastOp>(loc, i1MemrefTy, *addrOr).getResult();
}

// Reinterpret UB storage at the same byte address into a fixed VL-wide view.
// Uses elementStorageBytes so Bool (i1, AscendC b8) scales as 1 byte/lane.
static FailureOr<Value> materializeDataMemref(OpBuilder& b, Location loc, Value source, Type elementType)
{
    auto type = dyn_cast<MemRefType>(source.getType());
    if (!type)
        return failure();
    int64_t dstBytes = elementStorageBytes(elementType);
    if (dstBytes <= 0 || 256 % dstBytes != 0)
        return failure();
    auto layout = StridedLayoutAttr::get(b.getContext(), 0, ArrayRef<int64_t>{1});
    auto resultType = MemRefType::get({256 / dstBytes}, elementType, layout, type.getMemorySpace());
    if (type == resultType)
        return source;
    if (type.getElementType() == elementType && type.getShape() == resultType.getShape())
        return source;
    auto addrOr = memrefByteAddrI64(b, loc, source);
    if (failed(addrOr))
        return failure();
    return b.create<hivm::PointerCastOp>(loc, resultType, *addrOr).getResult();
}

/// Same-shape signed/signless ``i8`` view for shared b8 ABI (MaskLoad US/DS,
/// DATA_BLOCK_COPY ``*_int8``, Bool AVE load/store). Pass signed/signless i8
/// through; cast Bool ``i1`` / ``ui8`` (and other non-i8 1-byte carriers) while
/// preserving shape/layout/memory space so mixed Bool+Int8 regions share one
/// memref signature. Never force ``memref<256xi8, strided<[1]>>`` — that
/// static-offset reshape crashes BiShengLIR and conflicts with Int8 tiles.
///
/// Callers must already hold a 1-byte-storage MemRef (Bool / ui8 / i8 tile).
static Value materializeSameShapeI8Memref(OpBuilder& b, Location loc, Value ubMemref)
{
    auto srcTy = cast<MemRefType>(ubMemref.getType());
    auto elem = srcTy.getElementType();
    // Signed / signless i8 only — unsigned must cast (shared stub is i8).
    if (auto it = dyn_cast<IntegerType>(elem)) {
        if (it.getWidth() == 8 && !it.isUnsigned())
            return ubMemref;
    }
    auto i8Ty = IntegerType::get(b.getContext(), 8);
    auto resultType = MemRefType::get(srcTy.getShape(), i8Ty, srcTy.getLayout(), srcTy.getMemorySpace());
    if (srcTy == resultType)
        return ubMemref;
    // 1-byte MemRef: memrefByteAddrI64 always succeeds under the caller contract.
    return b.create<hivm::PointerCastOp>(loc, resultType, *memrefByteAddrI64(b, loc, ubMemref)).getResult();
}

static std::string buildUniqueVectorHelperName(ModuleOp module, int& nextVectorRegionId)
{
    std::string helperName;
    do {
        helperName = "vector_region_" + std::to_string(nextVectorRegionId++);
    } while (module.lookupSymbol<func::FuncOp>(helperName));
    return helperName;
}

enum class VectorBinaryKind
{
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    And,
    Or,
    Xor,
    Relu,
    ShiftLeft,
    ShiftRight
};
enum class VectorRhsKind
{
    Vector,
    Scalar
};

static FailureOr<hivmave::CombiningKind> getAveReductionCombiningKind(::tla::ReduceOp reduceOp, Type elementType)
{
    auto kindAttr = reduceOp->getAttrOfType<StringAttr>("kind");
    if (!kindAttr)
        return reduceOp.emitError("tla.reduce requires string kind attribute"), failure();
    StringRef kind = kindAttr.getValue();
    if (kind == "add")
        return hivmave::CombiningKind::ADD;
    if (kind == "max") {
        if (auto intType = dyn_cast<IntegerType>(elementType))
            return intType.getSignedness() == IntegerType::Unsigned ? hivmave::CombiningKind::UMAX :
                                                                      hivmave::CombiningKind::MAX;
        if (isa<FloatType>(elementType))
            return hivmave::CombiningKind::MAX;
    }
    if (kind == "min") {
        if (auto intType = dyn_cast<IntegerType>(elementType))
            return intType.getSignedness() == IntegerType::Unsigned ? hivmave::CombiningKind::UMIN :
                                                                      hivmave::CombiningKind::MIN;
        if (isa<FloatType>(elementType))
            return hivmave::CombiningKind::MIN;
    }
    return reduceOp.emitError() << "tla.reduce supports only add, max, and min reductions, got \"" << kind << "\"",
           failure();
}

static bool isSupportedVectorReductionElementType(Type elementType)
{
    if (isa<Float16Type, Float32Type>(elementType))
        return true;
    auto intType = dyn_cast<IntegerType>(elementType);
    if (!intType)
        return false;
    switch (intType.getWidth()) {
        case 16:
        case 32:
            return true;
        default:
            return false;
    }
}

// The tla.tensor_descs a region uses but does not contain.
//
// tla-lower-tensor-desc is the sole descriptor producer, so this is a
// TensorDesc query and nothing here re-walks a tile_view / make_tensor producer
// chain. Both consumers -- the scalar capture in
// collectVectorHelperScalarOperands and the materialization in buildHelperFunc
// -- read the same list, so the two cannot drift apart.
static void collectExternalTensorDescs(::tla::VecFuncOp vecFuncOp, SmallVectorImpl<::tla::TensorDescOp>& descs)
{
    vecFuncOp.walk([&](Operation* op) {
        for (Value operand : op->getOperands()) {
            auto desc = operand.getDefiningOp<::tla::TensorDescOp>();
            if (desc && !vecFuncOp->isProperAncestor(desc.getOperation()) && !llvm::is_contained(descs, desc))
                descs.push_back(desc);
        }
    });
}

// A store destination tile is directly a tla.tensor_desc result after
// tla-lower-tensor-desc (the sole descriptor producer). Only look one level;
// the caller diagnoses when the value is not a descriptor result.
static ::tla::TensorDescOp findTensorDescProducer(Value tensorValue)
{
    return tensorValue ? tensorValue.getDefiningOp<::tla::TensorDescOp>() : nullptr;
}

static LogicalResult validateVectorReduction(::tla::ReduceOp reduceOp, Type elementType)
{
    if (!isSupportedVectorReductionElementType(elementType))
        return reduceOp.emitError() << "tla.reduce unsupported reduction element type " << elementType;
    return success();
}

enum class VectorUnaryKind
{
    Exp,
    Log,
    Sqrt,
    Abs,
    Relu,
    Neg,
    Not
};

struct TlaUnaryOperands {
    Value operand;
    Value mask;
};

struct VectorUnaryInfo {
    VectorUnaryKind kind;
    StringRef name;
    TlaUnaryOperands operands;
};

template <typename OpTy>
static TlaUnaryOperands getTlaUnaryOperands(OpTy op)
{
    return TlaUnaryOperands{op.getOperand(), op.getMask()};
}

static std::optional<VectorUnaryInfo> getVectorUnaryInfo(Operation* op)
{
    if (!op)
        return std::nullopt;
    if (auto o = dyn_cast<::tla::ExpOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Exp, "exp", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::LogOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Log, "log", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::SqrtOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Sqrt, "sqrt", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::AbsOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Abs, "abs", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::ReluOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Relu, "relu", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::NegOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Neg, "neg", getTlaUnaryOperands(o)};
    if (auto o = dyn_cast<::tla::BitwiseNotOp>(op))
        return VectorUnaryInfo{VectorUnaryKind::Not, "bitwise_not", getTlaUnaryOperands(o)};
    return std::nullopt;
}

static LogicalResult validateVectorUnaryElementType(Operation* op, VectorUnaryInfo info, Type elementType)
{
    switch (info.kind) {
        case VectorUnaryKind::Exp:
        case VectorUnaryKind::Log:
        case VectorUnaryKind::Sqrt:
            if (!isa<FloatType>(elementType))
                return op->emitError() << "tla." << info.name << " requires floating-point element type, got "
                                       << elementType;
            if (isa<BFloat16Type>(elementType))
                return op->emitError() << "tla." << info.name << " does not support bf16 element type yet";
            return success();
        case VectorUnaryKind::Relu:
            // ReluOp::verify checks the source type before signedness is erased.
            return success();
        case VectorUnaryKind::Abs:
        case VectorUnaryKind::Neg:
        case VectorUnaryKind::Not:
            if (auto floatType = dyn_cast<FloatType>(elementType)) {
                if (isa<BFloat16Type>(floatType))
                    return op->emitError() << "tla." << info.name << " does not support bf16 element type yet";
                if (floatType.isF16() || floatType.isF32())
                    return success();
                return op->emitError() << "tla." << info.name
                                       << " requires f16 or f32 floating-point element type, got " << elementType;
            }
            if (auto intType = dyn_cast<IntegerType>(elementType)) {
                unsigned width = intType.getWidth();
                if (width == 8 || width == 16 || width == 32)
                    return success();
                return op->emitError() << "tla." << info.name << " requires i8, i16, or i32 element type, got "
                                       << elementType;
            }
            return op->emitError() << "tla." << info.name << " requires f16/f32 or i8/i16/i32 element type, got "
                                   << elementType;
    }
    return failure();
}

// The lhs/rhs/mask operands of vector binary ops (mask may be null).
struct TlaBinaryOperands {
    Value lhs;
    Value rhs;
    Value mask;
};

struct VectorOpInfo {
    VectorBinaryKind kind;
    VectorRhsKind rhsKind;
    StringRef mnemonic;
    TlaBinaryOperands operands;
};

struct AnyVectorOperationInfo {
    std::optional<VectorOpInfo> binary;
    std::optional<VectorUnaryInfo> unary;
};

// Every tla binary op carries the same three operands, so the tables below read
// them generically: adding an op means adding one row, not editing a second
// parallel switch that fails silently when it is forgotten.
template <typename OpT>
static VectorOpInfo binaryInfo(OpT o, VectorBinaryKind kind, VectorRhsKind rhsKind, StringRef mnemonic)
{
    return VectorOpInfo{kind, rhsKind, mnemonic, TlaBinaryOperands{o.getLhs(), o.getRhs(), o.getMask()}};
}

/// Binary ops whose rhs is a whole vector.
static std::optional<VectorOpInfo> getVectorBinaryInfo(Operation* op)
{
    if (!op)
        return std::nullopt;
    using K = VectorBinaryKind;
    constexpr VectorRhsKind kVec = VectorRhsKind::Vector;
    return TypeSwitch<Operation*, std::optional<VectorOpInfo>>(op)
        .Case([](::tla::AddOp o) { return binaryInfo(o, K::Add, kVec, "add"); })
        .Case([](::tla::SubOp o) { return binaryInfo(o, K::Sub, kVec, "sub"); })
        .Case([](::tla::MulOp o) { return binaryInfo(o, K::Mul, kVec, "mul"); })
        .Case([](::tla::DivOp o) { return binaryInfo(o, K::Div, kVec, "div"); })
        .Case([](::tla::MaxOp o) { return binaryInfo(o, K::Max, kVec, "max"); })
        .Case([](::tla::MinOp o) { return binaryInfo(o, K::Min, kVec, "min"); })
        .Case([](::tla::BitwiseAndOp o) { return binaryInfo(o, K::And, kVec, "bitwise_and"); })
        .Case([](::tla::BitwiseOrOp o) { return binaryInfo(o, K::Or, kVec, "bitwise_or"); })
        .Case([](::tla::BitwiseXorOp o) { return binaryInfo(o, K::Xor, kVec, "bitwise_xor"); })
        .Case([](::tla::PreluOp o) { return binaryInfo(o, K::Relu, kVec, "prelu"); })
        .Case([](::tla::ShiftLeftOp o) { return binaryInfo(o, K::ShiftLeft, kVec, "shift_left"); })
        .Case([](::tla::ShiftRightOp o) { return binaryInfo(o, K::ShiftRight, kVec, "shift_right"); })
        .Default(std::optional<VectorOpInfo>{std::nullopt});
}

/// Binary ops whose rhs is a scalar splat across the vector.
static std::optional<VectorOpInfo> getVectorScalarBinaryInfo(Operation* op)
{
    if (!op)
        return std::nullopt;
    using K = VectorBinaryKind;
    constexpr VectorRhsKind kScalar = VectorRhsKind::Scalar;
    return TypeSwitch<Operation*, std::optional<VectorOpInfo>>(op)
        .Case([](::tla::AddsOp o) { return binaryInfo(o, K::Add, kScalar, "adds"); })
        .Case([](::tla::SubsOp o) { return binaryInfo(o, K::Sub, kScalar, "subs"); })
        .Case([](::tla::MulsOp o) { return binaryInfo(o, K::Mul, kScalar, "muls"); })
        .Case([](::tla::MaxsOp o) { return binaryInfo(o, K::Max, kScalar, "maxs"); })
        .Case([](::tla::MinsOp o) { return binaryInfo(o, K::Min, kScalar, "mins"); })
        .Case([](::tla::DivsOp o) { return binaryInfo(o, K::Div, kScalar, "divs"); })
        .Case([](::tla::LeakyReluOp o) { return binaryInfo(o, K::Relu, kScalar, "leaky_relu"); })
        .Case([](::tla::ShiftLeftsOp o) { return binaryInfo(o, K::ShiftLeft, kScalar, "shift_lefts"); })
        .Case([](::tla::ShiftRightsOp o) { return binaryInfo(o, K::ShiftRight, kScalar, "shift_rights"); })
        .Default(std::optional<VectorOpInfo>{std::nullopt});
}

static std::optional<AnyVectorOperationInfo> getAnyVectorOperationInfo(Operation* op)
{
    if (auto info = getVectorBinaryInfo(op))
        return AnyVectorOperationInfo{*info, std::nullopt};
    if (auto info = getVectorScalarBinaryInfo(op))
        return AnyVectorOperationInfo{*info, std::nullopt};
    if (auto info = getVectorUnaryInfo(op))
        return AnyVectorOperationInfo{std::nullopt, *info};
    return std::nullopt;
}

// The mask-register width (b8/b16/b32) matching the element type.
static hivmave::MaskWidth maskWidthForElement(Type elementType)
{
    unsigned bits = getBitSizeOfFixedWidthScalarType(elementType);
    if (bits <= 8)
        return hivmave::MaskWidth::B8;
    if (bits <= 16)
        return hivmave::MaskWidth::B16;
    return hivmave::MaskWidth::B32;
}

static std::optional<hivmave::CmpType> mapCmpMode(StringRef mode)
{
    return llvm::StringSwitch<std::optional<hivmave::CmpType>>(mode)
        .Case("lt", hivmave::CmpType::LT)
        .Case("le", hivmave::CmpType::LE)
        .Case("gt", hivmave::CmpType::GT)
        .Case("ge", hivmave::CmpType::GE)
        .Case("eq", hivmave::CmpType::EQ)
        .Case("ne", hivmave::CmpType::NE)
        .Default(std::nullopt);
}

// True for the tla ops that produce a vector compute result inside a vec.func
// region: element-wise binary/unary ops, bitwise ops, where/select,
// reductions, and gather.
static bool isVectorComputeOp(Operation* op)
{
    return getAnyVectorOperationInfo(op).has_value() || isa_and_nonnull<::tla::CmpOp>(op) ||
           isa_and_nonnull<::tla::WhereOp>(op) || isa_and_nonnull<::tla::SqueezeOp>(op) ||
           isa_and_nonnull<::tla::UnsqueezeOp>(op) || isa_and_nonnull<::tla::HistogramsOp>(op) ||
           isa_and_nonnull<::tla::ReduceOp>(op) || isa_and_nonnull<::tla::ReduceIntraBlockOp>(op) ||
           isa_and_nonnull<::tla::PairWiseOp>(op) || isa_and_nonnull<::tla::GatherOp>(op) ||
           isa_and_nonnull<::tla::CastOp>(op) || isa_and_nonnull<::tla::BitCastOp>(op) ||
           isa_and_nonnull<::tla::ExpSubOp, ::tla::InplaceMulAddOp>(op) ||
           isa_and_nonnull<::tla::AxpyOp, ::tla::SubAbsOp, ::tla::MulsCastOp>(op) ||
           isa_and_nonnull<::tla::PackOp>(op) || isa_and_nonnull<::tla::UnpackOp>(op) ||
           isa_and_nonnull<::tla::InterleaveOp, ::tla::DeInterleaveOp, ::tla::MullOp, ::tla::AddCOp, ::tla::SubCOp>(
               op) ||
           isa_and_nonnull<::tla::GenerateMaskOp>(op);
}

static void annotateAivLibraryCall(func::FuncOp callee)
{
    MLIRContext* ctx = callee.getContext();
    callee.setPrivate();
    callee->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
    callee->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
}

// 64-bit AVE library calls cannot legalize a <32xi64> result in the device
// backend; route them through linked bitcode helpers (same ABI as the shift
// stubs: <32xi64> vectors and the native b64 predicate).
static Value createInlineB64Call(
    OpBuilder& b, Location loc, ModuleOp module, StringRef name, TypeRange inputTypes, Type resultType,
    ValueRange operands)
{
    auto callee = module.lookupSymbol<func::FuncOp>(name);
    if (!callee) {
        OpBuilder moduleBuilder(module.getBodyRegion());
        auto fnType = FunctionType::get(b.getContext(), inputTypes, {resultType});
        callee = moduleBuilder.create<func::FuncOp>(loc, name, fnType);
        annotateAivLibraryCall(callee);
    }
    return b.create<func::CallOp>(loc, callee, operands).getResult(0);
}

// MaskLoadDist US/DS: one bc symbol per UB element byte width. Element type
// is address typing only; plds always takes __ubuf__ uint32_t*. Typed stubs
// keep the memref ABI exact (PointerCast-to-i8 views have tripped BiShengLIR).
static StringRef maskLoadUsDsCalleeName(int64_t elemBytes, hivmave::LoadDist pattern)
{
    const bool isUs = pattern == hivmave::LoadDist::US;
    switch (elemBytes) {
        case 1:
            return isUs ? "mask_load_us_b8" : "mask_load_ds_b8";
        case 2:
            return isUs ? "mask_load_us_b16" : "mask_load_ds_b16";
        case 4:
            return isUs ? "mask_load_us_b32" : "mask_load_ds_b32";
        case 8:
            return isUs ? "mask_load_us_b64" : "mask_load_ds_b64";
        default:
            return {};
    }
}

static func::FuncOp getOrCreateMaskLoadUsDsLibraryCall(
    ModuleOp module, Location loc, Type memRefType, hivmave::LoadDist pattern)
{
    auto memrefTy = dyn_cast<MemRefType>(memRefType);
    if (!memrefTy)
        return {};
    // Bool/i1 → 1 (b8), matching AscendC SupportBytes / sizeof(bool).
    int64_t elemBytes = elementStorageBytes(memrefTy.getElementType());
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4 && elemBytes != 8)
        return {};
    if (pattern != hivmave::LoadDist::US && pattern != hivmave::LoadDist::DS)
        return {};
    StringRef calleeName = maskLoadUsDsCalleeName(elemBytes, pattern);
    if (calleeName.empty())
        return {};
    // b8: callers must pass signed/signless i8 memref (Bool/ui8 cast at the
    // call site, preserving tile shape). Do not rewrite the arg to a forced
    // 256-lane carrier — that PointerCast path crashes BiShengLIR for MaskLoad
    // i8.
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateAivLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto pregTy = VectorType::get({256}, IntegerType::get(module.getContext(), 1));
    auto fnType = FunctionType::get(module.getContext(), {memRefType}, {pregTy});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateAivLibraryCall(callee);
    return callee;
}

static std::string getSqueezeLibraryCallName(Type elementType, bool storeReg)
{
    std::string name;
    if (elementType.isF32())
        name = "vsqueeze_float";
    else if (elementType.isF16())
        name = "vsqueeze_half";
    else if (auto intType = dyn_cast<IntegerType>(elementType)) {
        // Preserve the signedness when selecting the bitcode ABI stub.
        unsigned width = intType.getWidth();
        if (width == 8 || width == 16 || width == 32)
            name = std::string("vsqueeze_") + (intType.isUnsigned() ? "uint" : "int") + std::to_string(width) + "_t";
    }
    if (!name.empty() && storeReg)
        name += "_store_reg";
    return name;
}

static std::string getUnsqueezeLibraryCallName(Type elementType)
{
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
        unsigned width = intType.getWidth();
        // vusqz is sign-agnostic, but int*_t / uint*_t still need their own
        // VectorReg ABI stubs.
        if (width == 8 || width == 16 || width == 32)
            return std::string("vunsqueeze_") + (intType.isUnsigned() ? "uint" : "int") + std::to_string(width) + "_t";
    }
    return {};
}

static std::string getVectorPackLibraryCallName(Type srcElement, Type dstElement, ::RegPart part)
{
    auto src = cast<IntegerType>(srcElement);
    auto dst = cast<IntegerType>(dstElement);
    return std::string("pack_reg_uint") + std::to_string(dst.getWidth()) + "_t_" + (src.isUnsigned() ? "uint" : "int") +
           std::to_string(src.getWidth()) + "_t_" + (part == ::RegPart::high ? "high" : "low");
}

static std::string getVectorUnpackLibraryCallName(Type srcElement, Type dstElement, ::RegPart part)
{
    auto src = cast<IntegerType>(srcElement);
    auto dst = cast<IntegerType>(dstElement);
    return std::string("unpack_reg_") + (dst.isUnsigned() ? "uint" : "int") + std::to_string(dst.getWidth()) + "_t_" +
           (src.isUnsigned() ? "uint" : "int") + std::to_string(src.getWidth()) + "_t_" +
           (part == ::RegPart::high ? "high" : "low");
}

static func::FuncOp getOrCreateVectorPackUnpackLibraryCall(
    ModuleOp module, Location loc, VectorType srcType, VectorType dstType, StringRef name)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto type = FunctionType::get(module.getContext(), {srcType}, {dstType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, name, type);
    annotateAivLibraryCall(callee);
    return callee;
}

static std::string getHiFloat8CastLibraryCallName(Type srcElement, Type dstElement, ArrayRef<int32_t> trait)
{
    std::string name;
    if (isa<HiFloat8Type>(dstElement)) {
        name = srcElement.isF16() ? "lowp_cast_f16_to_hif8_" : "lowp_cast_f32_to_hif8_";
        name += static_cast<::RoundMode>(trait[2]) == ::RoundMode::cast_hybrid ? "hybrid_" : "round_";
        name += static_cast<::SatMode>(trait[1]) == ::SatMode::sat ? "sat" : "nosat";
    } else {
        name = dstElement.isF16() ? "lowp_cast_hif8_to_f16" : "lowp_cast_hif8_to_f32";
    }
    return name + "_slot" + std::to_string(trait[0]);
}

static StringRef getPacked4RoundSuffix(::RoundMode round)
{
    switch (round) {
        case ::RoundMode::cast_round:
            return "_round";
        case ::RoundMode::cast_floor:
            return "_floor";
        case ::RoundMode::cast_ceil:
            return "_ceil";
        case ::RoundMode::cast_trunc:
            return "_trunc";
        case ::RoundMode::cast_none:
        case ::RoundMode::cast_rint:
            return "_rint";
        default:
            llvm_unreachable("packed 4-bit rounding mode must be verified before lowering");
    }
}

static std::string getFp4CastLibraryCallName(Type srcElement, Type dstElement, ArrayRef<int32_t> trait)
{
    bool narrow = srcElement.isBF16();
    Type pairType = narrow ? dstElement : srcElement;
    std::string format = isa<Float4E2M1x2Type>(pairType) ? "f4e2m1x2" : "f4e1m2x2";
    std::string name = narrow ? "lowp_cast_bf16_to_" + format : "lowp_cast_" + format + "_to_bf16";
    if (narrow) {
        name += getPacked4RoundSuffix(static_cast<::RoundMode>(trait[2])).str();
        name += "_sat";
    }
    return name + "_slot" + std::to_string(trait[0]);
}

static func::FuncOp getOrCreateInlineAivLibraryCall(
    ModuleOp module, Location loc, ArrayRef<Type> inputTypes, VectorType resultType, StringRef name)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto type = FunctionType::get(module.getContext(), inputTypes, {resultType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, name, type);
    annotateAivLibraryCall(callee);
    callee->setAttr("hacc.always_inline", moduleBuilder.getUnitAttr());
    return callee;
}

static std::string getGatherRegLibraryCallName(Type elementType)
{
    if (elementType.isF32())
        return "gather_reg_float";
    if (elementType.isF16())
        return "gather_reg_half";
    if (elementType.isBF16())
        return "gather_reg_bfloat16_t";
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
        const unsigned w = intType.getWidth();
        const bool uns = intType.isUnsigned();
        if (w == 8)
            return uns ? "gather_reg_uint8_t" : "gather_reg_int8_t";
        if (w == 16)
            return uns ? "gather_reg_uint16_t" : "gather_reg_int16_t";
        if (w == 32)
            return uns ? "gather_reg_uint32_t" : "gather_reg_int32_t";
    }
    return {};
}

// mode / type mirror HistogramsParams / AscendC (BIN half vs FREQUENCY|ACCUMULATE).
static std::string getHistogramsLibraryCallName(HistogramsBin mode, HistogramsType type)
{
    const bool accumulate = type == HistogramsType::accumulate;
    const bool bin1 = mode == HistogramsBin::bin1;
    if (!accumulate && !bin1)
        return "histograms_frequency_bin0_u8";
    if (!accumulate && bin1)
        return "histograms_frequency_bin1_u8";
    if (accumulate && !bin1)
        return "histograms_accumulate_bin0_u8";
    return "histograms_accumulate_bin1_u8";
}

static func::FuncOp getOrCreateSqueezeLibraryCall(
    ModuleOp module, Location loc, VectorType vecType, VectorType pregType, StringRef calleeName)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateAivLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(module.getContext(), {vecType, pregType}, {vecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateAivLibraryCall(callee);
    return callee;
}

static func::FuncOp getOrCreateUnsqueezeLibraryCall(
    ModuleOp module, Location loc, VectorType vecType, VectorType pregType, StringRef calleeName)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateAivLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(module.getContext(), {pregType}, {vecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateAivLibraryCall(callee);
    return callee;
}

static func::FuncOp getOrCreateGatherRegLibraryCall(
    ModuleOp module, Location loc, VectorType srcVecType, VectorType indexVecType, StringRef calleeName)
{
    // Mixed AIC/AIV kernels require hivm.func_core_type on linked AIV bitcode
    // stubs; without it HIVMC reports "Unknown core type" for gather_reg_*.
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateAivLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(module.getContext(), {srcVecType, indexVecType}, {srcVecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateAivLibraryCall(callee);
    return callee;
}

static func::FuncOp getOrCreateHistogramsLibraryCall(
    ModuleOp module, Location loc, VectorType srcVecType, VectorType dstVecType, VectorType pregType,
    StringRef calleeName)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateAivLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(module.getContext(), {srcVecType, dstVecType, pregType}, {dstVecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateAivLibraryCall(callee);
    return callee;
}

static func::FuncOp getOrCreateMaskSelectLibraryCall(ModuleOp module, Location loc, VectorType pregType)
{
    constexpr llvm::StringLiteral calleeName = "tla_mask_select";
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateAivLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(module.getContext(), {pregType, pregType, pregType}, {pregType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateAivLibraryCall(callee);
    return callee;
}

/// Shared bc type suffix for DATA_BLOCK_COPY helpers in ``bc/Vector/{load,store}.cpp``.
/// AscendC ``DATA_BLOCK_COPY`` supports b8/b16/b32. Stub selection uses the TLA
/// element (so Bool / UInt8 stay distinct), but types whose AVE carrier is i8
/// (``aveVectorElementType`` → i8) share the ``int8`` vsldb/vsstb stubs — same
/// as master when suffix was taken from the already-lowered vector type.
static std::string dataBlockCopyElementSuffix(Type elementType)
{
    if (isBoolElementType(elementType))
        return "int8";
    // Logical TLA types that AVE already carries as i8 (not FP8E4/E5 AS_BYTES).
    if (isa<::tla::HiFloat8Type, ::tla::Float8E8M0Type>(elementType) || isFp4PairType(elementType) ||
        isInt4PairType(elementType))
        return "int8";
    if (elementType.isF32())
        return "float";
    if (elementType.isF16())
        return "half";
    if (elementType.isBF16())
        return "bf16";
    if (elementType.isInteger(32))
        return elementType.isUnsignedInteger() ? "uint32" : "int32";
    if (elementType.isInteger(16))
        return elementType.isUnsignedInteger() ? "uint16" : "int16";
    if (isa<Float8E4M3FNType>(elementType))
        return "fp8_e4m3fn";
    if (isa<Float8E5M2Type>(elementType))
        return "fp8_e5m2";
    if (elementType.isInteger(8))
        return elementType.isUnsignedInteger() ? "uint8" : "int8";
    return {};
}

static std::string getStoreWithStrideLibraryCallName(Type elementType)
{
    std::string suffix = dataBlockCopyElementSuffix(elementType);
    return suffix.empty() ? std::string{} : ("store_with_stride_" + suffix);
}

// The local VectorReg shim has no FP8 specializations, so strided block
// helpers pass those registers as bytes. SDK RegTensor, used by DINTLV_B8,
// provides native FP8 register types independently of this shim.
static bool blockStrideUsesByteRegister(Type elementType)
{
    return isa<Float8E4M3FNType, Float8E5M2Type>(elementType);
}

static VectorType byteRegisterTypeFor(VectorType vecType)
{
    return VectorType::get(vecType.getShape(), IntegerType::get(vecType.getContext(), 8));
}

static void annotateStoreWithStrideLibraryCall(func::FuncOp callee)
{
    MLIRContext* ctx = callee.getContext();
    callee.setPrivate();
    callee->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
    callee->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
}

static std::string getLoadWithStrideLibraryCallName(Type elementType)
{
    std::string suffix = dataBlockCopyElementSuffix(elementType);
    return suffix.empty() ? std::string{} : ("load_with_stride_" + suffix);
}

static func::FuncOp getOrCreateStoreWithStrideLibraryCall(
    ModuleOp module, Location loc, Type vecType, Type memRefType, StringRef calleeName)
{
    auto ctx = module.getContext();
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateStoreWithStrideLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(
        ctx, {vecType, memRefType, IntegerType::get(ctx, 32), VectorType::get({256}, IntegerType::get(ctx, 1))}, {});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateStoreWithStrideLibraryCall(callee);
    return callee;
}

// Strided block load (vsldb) library call: (memref, blockStride, repeatStride,
// preg) -> vecType. Annotated the same way as the store-with-stride call.
static func::FuncOp getOrCreateLoadWithStrideLibraryCall(
    ModuleOp module, Location loc, Type vecType, Type memRefType, StringRef calleeName)
{
    auto ctx = module.getContext();
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateStoreWithStrideLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(
        ctx,
        {memRefType, IntegerType::get(ctx, 32), IntegerType::get(ctx, 32),
         VectorType::get({256}, IntegerType::get(ctx, 1))},
        {vecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateStoreWithStrideLibraryCall(callee);
    return callee;
}

// vsldb.post / vsstb.post are not HIVMC ops. The bc helper returns the hardware
// pointer (store) or writes it to an i64 slot (load, so the VectorReg return
// ABI stays the single-result form already used by load_with_stride).
// Helpers already use the ``_mlir_ciface_*`` ABI in ``bc/Vector/{load,store}.cpp``
// (raw UB pointer, not a memref descriptor). Call that symbol directly.
// ``llvm.emit_c_interface`` would keep the bare name as the simd_vf callee,
// and bisheng then fails to find a body to inline.
static void annotateDataBlockCopyUpdateLibraryCall(func::FuncOp callee)
{
    MLIRContext* ctx = callee.getContext();
    callee.setPrivate();
    callee->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
}

// CCE LoadAlign supplies both typed registers through an explicit output slot.
// Use the raw-pointer symbol directly, as with the post-update helpers below.
static func::FuncOp getOrCreateLoadDintlvB8LibraryCall(ModuleOp module, Location loc, StringRef calleeName)
{
    MLIRContext* ctx = module.getContext();
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateDataBlockCopyUpdateLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto fnType = FunctionType::get(ctx, {getUbPtrType(ctx), LLVM::LLVMPointerType::get(ctx)}, {});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateDataBlockCopyUpdateLibraryCall(callee);
    return callee;
}

// Dual-source INTLV store mirrors Load DINTLV: emit HIVM ``vstsx2`` with the
// AVE StoreDist enum as ``dist`` (no AIV bc). Carriers are signless integer
// vectors of the Dist width — same bit-move model as AscendC's typed
// StoreAlign<DIST_INTLV_*> after TypeGet aliasing.
static VectorType dualTransferCarrierType(MLIRContext* ctx, int64_t distBytes)
{
    return VectorType::get({256 / distBytes}, IntegerType::get(ctx, distBytes * 8));
}

// Each unaligned instruction advances by at most one register. Split a
// partial transfer between the low and high halves in address order.
static std::pair<Value, Value> splitUnalignUpdateCount(OpBuilder& b, Location loc, Value countBytes)
{
    Value vlBytes = b.create<arith::ConstantIntOp>(loc, 256, 32);
    Value first = b.create<arith::MinUIOp>(loc, countBytes, vlBytes);
    Value second = b.create<arith::SubIOp>(loc, countBytes, first);
    return {first, second};
}

static Value bitcastToIntlvCarrier(OpBuilder& b, Location loc, Value src, int64_t distBytes)
{
    auto srcTy = dyn_cast<VectorType>(src.getType());
    if (!srcTy)
        return nullptr;
    auto carrierTy = dualTransferCarrierType(b.getContext(), distBytes);
    if (srcTy == carrierTy)
        return src;
    return b.create<mlir::vector::BitCastOp>(loc, carrierTy, src).getResult();
}

static Value emitVldsx2Post(
    OpBuilder& b, Location loc, Value ptr, Value stride, Type elemType, VectorType carrierTy, hivmave::LoadDist dist)
{
    namespace intr = hivm_regbaseintrins;
    Value byteOff = strideElemsToByteOffI32(b, loc, stride, elemType);
    Value postEnable = b.create<arith::ConstantIntOp>(loc, 1, 32);
    Type pairAndPtr =
        LLVM::LLVMStructType::getLiteral(b.getContext(), {carrierTy, carrierTy, getUbPtrType(b.getContext())});
    Value distribution = b.create<arith::ConstantIntOp>(loc, static_cast<uint32_t>(dist), 32);
    auto emit = [&](auto opTag) -> Value {
        return b.create<decltype(opTag)>(loc, pairAndPtr, ptr, byteOff, distribution, postEnable).getResult();
    };
    switch (carrierTy.getElementTypeBitWidth()) {
        case 8:
            return emit(intr::Vldsx2PostV256U8InstrOp{});
        case 16:
            return emit(intr::Vldsx2PostV128U16InstrOp{});
        case 32:
            return emit(intr::Vldsx2PostV64U32InstrOp{});
        default:
            llvm_unreachable("verified DINTLV carrier width");
    }
}

static LogicalResult emitVstsx2Intlv(
    OpBuilder& b, Location loc, hivmave::StoreDist pattern, Value data0, Value data1, Value ubPtr, Value mask)
{
    Value constantZero = b.create<arith::ConstantIntOp>(loc, 0, 32);
    Value distribution = b.create<arith::ConstantIntOp>(loc, static_cast<uint32_t>(pattern), 32);
    switch (pattern) {
        case hivmave::StoreDist::INTLV_B8:
            b.create<hivm_regbaseintrins::Vstsx2V256B8InstrOp>(
                loc, data0, data1, ubPtr, constantZero, distribution, constantZero, mask);
            return success();
        case hivmave::StoreDist::INTLV_B16:
            b.create<hivm_regbaseintrins::Vstsx2V128B16InstrOp>(
                loc, data0, data1, ubPtr, constantZero, distribution, constantZero, mask);
            return success();
        case hivmave::StoreDist::INTLV_B32:
            b.create<hivm_regbaseintrins::Vstsx2V64B32InstrOp>(
                loc, data0, data1, ubPtr, constantZero, distribution, constantZero, mask);
            return success();
        default:
            return failure();
    }
}

// PACK_B64: AscendC ``vsts(src, dst, 0, PK_B64, mask)``. AVE→bc has no
// ``masked_store_PK_B64_*`` stub, so emit HIVM ``vstsx1.v32{s,u}64`` with
// DistVST ``PK_B64`` (=10, same ordinal as AVE StoreDist::PK_B64).
static void emitVstsx1PackB64(OpBuilder& b, Location loc, Value data, Value ubPtr, Value mask)
{
    Value constantZero = b.create<arith::ConstantIntOp>(loc, 0, 32);
    Value distribution = b.create<arith::ConstantIntOp>(loc, static_cast<uint32_t>(hivmave::StoreDist::PK_B64), 32);
    Type i64 = IntegerType::get(b.getContext(), 64);
    auto i64VecTy = VectorType::get({32}, i64);
    Value srcI64 =
        (data.getType() == i64VecTy) ? data : b.create<mlir::vector::BitCastOp>(loc, i64VecTy, data).getResult();
    b.create<hivm_regbaseintrins::Vstsx1V32S64InstrOp>(
        loc, srcI64, ubPtr, constantZero, distribution, constantZero, mask);
}

static func::FuncOp getOrCreateLoadUpdateRetptrLibraryCall(
    ModuleOp module, Location loc, Type vecType, StringRef calleeName)
{
    auto ctx = module.getContext();
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateDataBlockCopyUpdateLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto ubPtr = getUbPtrType(ctx);
    auto slotPtr = LLVM::LLVMPointerType::get(ctx);
    auto fnType = FunctionType::get(
        ctx,
        {ubPtr, IntegerType::get(ctx, 32), IntegerType::get(ctx, 32), VectorType::get({256}, IntegerType::get(ctx, 1)),
         slotPtr},
        {vecType});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateDataBlockCopyUpdateLibraryCall(callee);
    return callee;
}

static func::FuncOp getOrCreateStoreUpdateRetptrLibraryCall(
    ModuleOp module, Location loc, Type vecType, StringRef calleeName)
{
    auto ctx = module.getContext();
    if (auto existing = module.lookupSymbol<func::FuncOp>(calleeName)) {
        annotateDataBlockCopyUpdateLibraryCall(existing);
        return existing;
    }
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto ubPtr = getUbPtrType(ctx);
    auto fnType = FunctionType::get(
        ctx,
        {vecType, ubPtr, IntegerType::get(ctx, 32), IntegerType::get(ctx, 32),
         VectorType::get({256}, IntegerType::get(ctx, 1))},
        {ubPtr});
    auto callee = moduleBuilder.create<func::FuncOp>(loc, calleeName, fnType);
    annotateDataBlockCopyUpdateLibraryCall(callee);
    return callee;
}

static Value castMaskToPregType(OpBuilder& b, Location loc, Value mask, VectorType pregVecType)
{
    if (mask.getType() == pregVecType)
        return mask;
    return b.create<UnrealizedConversionCastOp>(loc, pregVecType, mask).getResult(0);
}

static VectorType fullPregVecType(MLIRContext* ctx)
{
    return VectorType::get({256}, IntegerType::get(ctx, 1));
}

static hivmave::MaskWidthAttr maskWidthAttrForElement(OpBuilder& b, Type elementType)
{
    return hivmave::MaskWidthAttr::get(b.getContext(), maskWidthForElement(elementType));
}

// The semantic width of a MaskSSA is carried by !tla.mask<N>, independently
// of its lowered predicate-register container. A carrier crossing SCF uses the
// backend-native vector<256xi1> container, but !tla.mask<64> must still select
// B32 rather than being misclassified as B8 from that container width.
static hivmave::MaskWidth maskWidthForMaskType(::tla::MaskSSAType maskType)
{
    int64_t lanes = maskType.getPhysicalLanes();
    if (lanes <= 0)
        return hivmave::MaskWidth::B32;
    int64_t bytesPerLane = 256 / lanes;
    if (bytesPerLane <= 1)
        return hivmave::MaskWidth::B8;
    if (bytesPerLane <= 2)
        return hivmave::MaskWidth::B16;
    return hivmave::MaskWidth::B32;
}

static hivmave::MaskWidthAttr maskWidthAttrForMaskType(OpBuilder& b, ::tla::MaskSSAType maskType)
{
    return hivmave::MaskWidthAttr::get(b.getContext(), maskWidthForMaskType(maskType));
}

// AVE vectors are signless; peel unsigned from the tla operand when present.
static Type tlaElementOr(Type tlaType, Type fallback)
{
    if (auto v = dyn_cast<::tla::VectorSSAType>(tlaType))
        return v.getElementType();
    return fallback;
}

static bool isUnsignedInt(Type type)
{
    auto i = dyn_cast<IntegerType>(type);
    return i && i.isUnsigned();
}

static Value createInt64Max(
    OpBuilder& b, Location loc, ModuleOp module, VectorType vecType, Value lhs, Value rhs, Value mask,
    bool hasExplicitMask)
{
    auto wordType = VectorType::get({vecType.getNumElements() * 2}, b.getI32Type());
    auto pregType = fullPregVecType(b.getContext());
    Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
    Value zeroWords = b.create<hivmave::VFBroadcastScalarOp>(loc, wordType, zero).getRes();
    mask = castMaskToPregType(b, loc, mask, pregType);
    if (hasExplicitMask) {
        // Explicit b64 masks use every eighth bit. After splitting into words,
        // each half uses a b32 mask with one bit per original lane.
        mask = b.create<hivm_regbaseintrins::PpackInstrOp>(loc, pregType, mask, zero).getResult();
    }

    auto splitWords = [&](Value value) -> std::pair<Value, Value> {
        if (isa<VectorType>(value.getType())) {
            Value words = b.create<vector::BitCastOp>(loc, wordType, value);
            auto split =
                b.create<hivmave::VFDeInterleaveOp>(loc, TypeRange{wordType, wordType}, ValueRange{words, zeroWords});
            return {split->getResult(0), split->getResult(1)};
        }
        Value shift = b.create<arith::ConstantIntOp>(loc, 32, 64);
        Value upper = b.createOrFold<arith::ShRUIOp>(loc, value, shift);
        Value low = b.createOrFold<arith::TruncIOp>(loc, b.getI32Type(), value);
        Value high = b.createOrFold<arith::TruncIOp>(loc, b.getI32Type(), upper);
        return {
            b.create<hivmave::VFBroadcastScalarOp>(loc, wordType, low).getRes(),
            b.create<hivmave::VFBroadcastScalarOp>(loc, wordType, high).getRes()};
    };
    auto [lhsLow, lhsHigh] = splitWords(lhs);
    auto [rhsLow, rhsHigh] = splitWords(rhs);
    using CmpType = hivmave::CmpType;
    // Compare signed high words first, then compare unsigned low words at ties.
    Value highGt = b.create<hivmave::VFCmpOp>(loc, pregType, CmpType::GT, lhsHigh, rhsHigh, mask);
    Value highEq = b.create<hivmave::VFCmpOp>(loc, pregType, CmpType::EQ, lhsHigh, rhsHigh, mask);
    Value lowGt = b.create<hivmave::VFCmpOp>(loc, pregType, CmpType::UGT, lhsLow, rhsLow, mask);

    auto maskSelect = getOrCreateMaskSelectLibraryCall(module, loc, pregType);
    Value takeLhs = b.create<func::CallOp>(loc, maskSelect, ValueRange{lowGt, highGt, highEq}).getResult(0);

    auto selectWords = [&](Value x, Value y) -> Value {
        Value selected = b.create<hivmave::VFSelectOp>(loc, wordType, takeLhs, x, y);
        return b.create<hivmave::VFSelectOp>(loc, wordType, mask, selected, zeroWords);
    };
    Value low = selectWords(lhsLow, rhsLow);
    Value high = selectWords(lhsHigh, rhsHigh);
    auto joined = b.create<hivmave::VFInterleaveOp>(loc, TypeRange{wordType, wordType}, ValueRange{low, high});
    return b.create<vector::BitCastOp>(loc, vecType, joined->getResult(0));
}

// Build the AVE vector op for a tla binary op. The mask controls active lanes.
// Div uses TypeFn for signedness; unsigned max/min select vumax/vumin.
static Value createVectorBinaryResult(
    OpBuilder& b, Location loc, VectorBinaryKind kind, Type tlaOperandType, Type elementType, VectorType vecType,
    Value lhs, Value rhs, Value mask)
{
    const bool isUnsigned = isUnsignedInt(tlaElementOr(tlaOperandType, elementType));
    switch (kind) {
        case VectorBinaryKind::Add:
            return b.create<hivmave::VFAddOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::Sub:
            return b.create<hivmave::VFSubOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::Mul:
            return b.create<hivmave::VFMulOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::Div: {
            auto cast = isUnsigned ? hivm::TypeFn::cast_unsigned : hivm::TypeFn::cast_signed;
            return b.create<hivmave::VFDivOp>(loc, vecType, lhs, rhs, mask, hivm::TypeFnAttr::get(b.getContext(), cast))
                .getResult();
        }
        case VectorBinaryKind::Max:
            if (isUnsigned)
                return b.create<hivmave::VMaxUIOp>(loc, vecType, lhs, rhs, mask).getResult();
            return b.create<hivmave::VFMaxOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::Min:
            if (isUnsigned)
                return b.create<hivmave::VMinUIOp>(loc, vecType, lhs, rhs, mask).getResult();
            return b.create<hivmave::VFMinOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::Relu:
            return b.create<hivmave::VFPReluOp>(loc, vecType, lhs, rhs, mask).getResult();
        case VectorBinaryKind::And:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFAndOp>(loc, vecType, lhs, rhs, mask).getResult();
            if (isa<::tla::MaskSSAType>(tlaOperandType))
                return b
                    .create<hivmave::PregAndOp>(
                        loc, vecType, maskWidthAttrForMaskType(b, cast<::tla::MaskSSAType>(tlaOperandType)), lhs, rhs,
                        mask)
                    .getRes();
            return nullptr;
        case VectorBinaryKind::Or:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFOrOp>(loc, vecType, lhs, rhs, mask).getResult();
            if (isa<::tla::MaskSSAType>(tlaOperandType))
                return b
                    .create<hivmave::PregOrOp>(
                        loc, vecType, maskWidthAttrForMaskType(b, cast<::tla::MaskSSAType>(tlaOperandType)), lhs, rhs,
                        mask)
                    .getRes();
            return nullptr;
        case VectorBinaryKind::Xor:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFXorOp>(loc, vecType, lhs, rhs, mask).getResult();
            if (isa<::tla::MaskSSAType>(tlaOperandType))
                return b
                    .create<hivmave::PregXorOp>(
                        loc, vecType, maskWidthAttrForMaskType(b, cast<::tla::MaskSSAType>(tlaOperandType)), lhs, rhs,
                        mask)
                    .getRes();
            return nullptr;
        case VectorBinaryKind::ShiftLeft:
        case VectorBinaryKind::ShiftRight: {
            if (!isa<IntegerType>(elementType) || elementType.isInteger(64))
                return nullptr;
            if (kind == VectorBinaryKind::ShiftLeft)
                return isUnsigned ?
                           b.create<hivm_regbaseintrins::VshlUXInstrOp>(loc, vecType, lhs, rhs, mask).getRes() :
                           b.create<hivm_regbaseintrins::VshlSXInstrOp>(loc, vecType, lhs, rhs, mask).getRes();
            return isUnsigned ? b.create<hivm_regbaseintrins::VshrUXInstrOp>(loc, vecType, lhs, rhs, mask).getRes() :
                                b.create<hivm_regbaseintrins::VshrSXInstrOp>(loc, vecType, lhs, rhs, mask).getRes();
        }
    }
    return nullptr;
}

static int64_t maskElementBitWidthForLanes(int64_t lanes)
{
    if (lanes >= 256)
        return 8;
    if (lanes >= 128)
        return 16;
    return 32;
}

// AVE represents a semantic vector<Nxi1> predicate as a hardware
// vector<256xi1>. When those types differ, annotate the AVE producer with the
// semantic element width so HIVMAVE lowering still selects pge/plt.b8/b16/b32
// from N instead of the full container width.
//
// The bit width is carried by functionType (hivmave::FunctionDistType, i.e.
// #ave.func_dist_type<pb8|pb16|pb32>). Without it, readers fall back to the
// container width (VL_BITS / predicate size), so even an f32 operand gets a b8
// mask that covers the wrong elements.
static constexpr llvm::StringLiteral kFunctionType = "functionType";
static void annotateFullPregWidth(OpBuilder& b, Operation* op, VectorType resultType, int64_t semanticLanes)
{
    if (resultType.getNumElements() == semanticLanes)
        return;
    const int64_t bitWidth = maskElementBitWidthForLanes(semanticLanes);
    hivmave::FunctionDistType dist = bitWidth >= 32 ? hivmave::FunctionDistType::PB32 :
                                     bitWidth >= 16 ? hivmave::FunctionDistType::PB16 :
                                                      hivmave::FunctionDistType::PB8;
    op->setAttr(kFunctionType, hivmave::FunctionDistTypeAttr::get(b.getContext(), dist));
}

static Value createPredicatePge(
    OpBuilder& b, Location loc, VectorType resultType, int64_t semanticLanes, hivmave::PgePattern pattern)
{
    auto pge = createAvePgeMask(b, loc, resultType, pattern);
    annotateFullPregWidth(b, pge, resultType, semanticLanes);
    return pge.getRes();
}

static hivmave::VFPltOp createPredicatePlt(
    OpBuilder& b, Location loc, VectorType resultType, int64_t semanticLanes, Value trueShape)
{
    auto plt = createAvePltMask(b, loc, resultType, trueShape);
    annotateFullPregWidth(b, plt, resultType, semanticLanes);
    return plt;
}

// Reuse auto-generated full predicates at the vector-helper entry. A generated
// predicate has no data dependency, so placing it here makes it dominate every
// loop and branch in the helper. Explicit tla.create_mask stays at its source
// location; this cache applies only to omitted masks.
class AutoFullPredicateCache {
public:
    explicit AutoFullPredicateCache(Block* entry) : entry(entry)
    {}

    Value get(Location loc, int64_t semanticLanes)
    {
        if (Value mask = masks.lookup(semanticLanes))
            return mask;
        OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
        // Keep the cached predicate in the hardware register-container type.
        // A semantic-width cache value would be materialized as a conversion
        // to vector<Nxi1> at the entry and then converted back to the physical
        // predicate for every vector op. Explicit tla.create_mask lowers as the
        // physical predicate directly, so the hidden default must do the same.
        VectorType maskType = fullPregVecType(entry->getParent()->getContext());
        Value mask = createPredicatePge(entryBuilder, loc, maskType, semanticLanes, hivmave::PgePattern::ALL);
        masks[semanticLanes] = mask;
        return mask;
    }

    /// Private out-parameter slot for BC helpers. Always at the outlined
    /// helper entry — not the enclosing ``scf.for`` / ``scf.if`` region.
    /// ``lowerScfFor`` body callbacks run before the for-region is linked, so
    /// ``getInsertionBlock()->getParentOp()`` is null there; an escaped alloca
    /// left in the loop body would grow the AIV stack every trip after scf→cf.
    FailureOr<Value> createRetptrSlot(OpBuilder& b, Location loc, Type ptrTy, Type elemTy, unsigned alignment = 0)
    {
        if (!entry)
            return failure();
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(entry);
        Value one = b.create<LLVM::ConstantOp>(loc, b.getI32Type(), b.getI32IntegerAttr(1));
        return b.create<LLVM::AllocaOp>(loc, ptrTy, elemTy, one, alignment).getResult();
    }

    // Keep the 512-byte result slot outside runtime loops and branches.
    FailureOr<Value> createDintlvResultSlot(OpBuilder& b, Location loc, Type pairType)
    {
        if (!entry)
            return failure();
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(entry);
        Value one = b.create<LLVM::ConstantOp>(loc, b.getI32Type(), b.getI32IntegerAttr(1));
        Type slotPtrTy = LLVM::LLVMPointerType::get(b.getContext());
        return b.create<LLVM::AllocaOp>(loc, slotPtrTy, pairType, one, 256).getResult();
    }

private:
    Block* entry;
    DenseMap<uint64_t, Value> masks;
};

// An all-lanes-active predicate for a data or MaskSSA vector. MaskSSA keeps its
// semantic lane count in tlaOperandType even when its mapped value is the full
// predicate-register container.
static Value allTrueMaskFor(AutoFullPredicateCache& cache, Location loc, VectorType vecType, Type tlaOperandType)
{
    int64_t semanticLanes = vecType.getNumElements();
    if (auto maskType = dyn_cast<::tla::MaskSSAType>(tlaOperandType))
        semanticLanes = maskType.getPhysicalLanes();
    return cache.get(loc, semanticLanes);
}

// Casts that take this rounding operand use RINT for the default CAST_NONE.
static hivm::RoundModeAttr mapCastRoundMode(OpBuilder& b, ::RoundMode mode)
{
    hivm::RoundMode hv;
    switch (mode) {
        case ::RoundMode::cast_round:
            hv = hivm::RoundMode::ROUND;
            break;
        case ::RoundMode::cast_floor:
            hv = hivm::RoundMode::FLOOR;
            break;
        case ::RoundMode::cast_ceil:
            hv = hivm::RoundMode::CEIL;
            break;
        case ::RoundMode::cast_trunc:
            hv = hivm::RoundMode::TRUNC;
            break;
        case ::RoundMode::cast_none:
        case ::RoundMode::cast_rint:
            hv = hivm::RoundMode::RINT;
            break;
        case ::RoundMode::cast_odd:
            hv = hivm::RoundMode::ODD;
            break;
        default:
            // The HiFloat8 register helper handles CAST_HYBRID.
            return {};
    }
    return hivm::RoundModeAttr::get(b.getContext(), hv);
}

// Map the tla.cast register layout onto the AVE VCVT part (even/odd) attribute.
static hivmave::VCVT_PartTypeAttr mapCastPart(OpBuilder& b, ::RegSlot layout)
{
    auto part = layout == ::RegSlot::one ? hivmave::VCVT_PartType::PART_ODD : hivmave::VCVT_PartType::PART_EVEN;
    return hivmave::VCVT_PartTypeAttr::get(b.getContext(), part);
}

// Map the tla.cast register layout onto the AVE pack pattern (pp0..pp3) used by
// 4x-width int casts (i32<->i8). reg_slot zero/one/two/three -> pp0/pp1/pp2/pp3.
static hivmave::VCVT_PPTypeAttr mapCastPP(OpBuilder& b, ::RegSlot layout)
{
    hivmave::VCVT_PPType pp;
    switch (layout) {
        case ::RegSlot::one:
            pp = hivmave::VCVT_PPType::PP1;
            break;
        case ::RegSlot::two:
            pp = hivmave::VCVT_PPType::PP2;
            break;
        case ::RegSlot::three:
            pp = hivmave::VCVT_PPType::PP3;
            break;
        case ::RegSlot::zero:
        default:
            pp = hivmave::VCVT_PPType::PP0;
            break;
    }
    return hivmave::VCVT_PPTypeAttr::get(b.getContext(), pp);
}

// The two 8-bit floats AVE vtruncf / vextf actually accept -- not every 8-bit
// float MLIR knows. e8m0 has no AVE conversion path and must not slip in on
// width alone.
static bool isAveFloat8Type(Type t)
{
    return isa<Float8E4M3FNType, Float8E5M2Type>(t);
}

// HIVMAVE's float conversion ops model only the 2x even/odd selector. FP8
// <-> f32 is the one 4x float conversion: its hardware intrinsics take the
// pp0..pp3 immediate directly. Materialize those intrinsics here rather than
// silently folding reg_slot TWO/THREE into PART_EVEN in the generic AVE op.
static Value createFp8ToF32Result(
    OpBuilder& b, Location loc, VectorType dstVecType, Value src, Value mask, hivmave::VCVT_PPTypeAttr pp)
{
    Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext()));
    Value part = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(pp.getValue()), 32);
    Type elemType = cast<VectorType>(src.getType()).getElementType();
    if (isa<Float8E4M3FNType>(elemType))
        return b.create<hivm_regbaseintrins::VcvtffF8E4M32F32InstrOp>(loc, dstVecType, src, preg, part).getResult();
    return b.create<hivm_regbaseintrins::VcvtffF8E5M22F32InstrOp>(loc, dstVecType, src, preg, part).getResult();
}

static Value createF32ToFp8Result(
    OpBuilder& b, Location loc, VectorType dstVecType, Value src, Value mask, hivm::RoundModeAttr rnd, BoolAttr sat,
    hivmave::VCVT_PPTypeAttr pp)
{
    Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext()));
    Value round = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(rnd.getValue()), 32);
    Value saturation = b.create<arith::ConstantIntOp>(loc, sat.getValue() ? 1 : 0, 32);
    Value part = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(pp.getValue()), 32);
    Type elemType = dstVecType.getElementType();
    if (isa<Float8E4M3FNType>(elemType))
        return b
            .create<hivm_regbaseintrins::VcvtffF322F8E4M3InstrOp>(loc, dstVecType, src, preg, round, saturation, part)
            .getResult();
    return b.create<hivm_regbaseintrins::VcvtffF322F8E5M2InstrOp>(loc, dstVecType, src, preg, round, saturation, part)
        .getResult();
}

// E5M2 and f16 have the same sign and exponent layout (one sign bit and a
// five-bit, bias-15 exponent). Their two E5M2 mantissa bits become f16 bits
// 9:8, so widening the raw byte and shifting it left by eight is an exact
// conversion. Unlike the generic fp8 -> f32 -> f16 composition, this emits one
// 2x integer widen and one shift. The caller's even/odd slot selects which
// source elements become the result, identically to the composed route.
static Value createE5M2ToF16Result(
    OpBuilder& b, Location loc, VectorType srcVecType, VectorType dstVecType, Value src, Value mask,
    hivmave::VCVT_PartTypeAttr part)
{
    auto srcBitsType = VectorType::get(srcVecType.getShape(), b.getIntegerType(8));
    auto dstBitsType = VectorType::get(dstVecType.getShape(), b.getIntegerType(16));
    Value srcBits = b.create<mlir::vector::BitCastOp>(loc, srcBitsType, src).getResult();
    Value widened =
        b.create<hivmave::VFExtSIOp>(loc, dstBitsType, srcBits, mask, part, hivmave::VCVT_PPTypeAttr()).getResult();
    Value shift = b.create<arith::ConstantIntOp>(loc, 8, 16);
    Value sign = b.create<arith::ConstantOp>(loc, b.getI1Type(), b.getBoolAttr(false));
    Value shifted = b.create<hivmave::VFShlsOp>(loc, dstBitsType, widened, shift, mask, sign).getResult();
    return b.create<mlir::vector::BitCastOp>(loc, dstVecType, shifted).getResult();
}

// Element types the tla.cast lowering can emit AVE ops for: signed/signless
// integers i8/i16/i32/i64, floats f16/bf16/f32, and the fp8 formats
// f8E4M3FN/f8E5M2. Unsigned integers, i1 (bool) and f64 have no AVE cast path
// and are rejected (the front-end rejects them too; this guards hand-written /
// non-front-end IR).
static bool isSupportedCastElementType(Type t)
{
    if (auto f = dyn_cast<FloatType>(t))
        return f.getWidth() == 16 || f.getWidth() == 32 || isAveFloat8Type(t);
    if (auto i = dyn_cast<IntegerType>(t)) {
        unsigned w = i.getWidth();
        return w == 4 || w == 8 || w == 16 || w == 32 || w == 64;
    }
    return false;
}

// Unsigned cast routes are an explicit short list (frontend
// `_CAST_UNSIGNED_ROUTES`): a missing AVE arm reaches llvm_unreachable.
static FailureOr<Value> createUnsignedVectorCastResult(
    OpBuilder& b, AutoFullPredicateCache& fullMasks, Location loc, VectorType srcVecType, VectorType dstVecType,
    Type tlaSrcElem, Type tlaDstElem, Value src, Value mask, bool hasUserMask, hivmave::VCVT_PartTypeAttr part,
    hivmave::VCVT_PPTypeAttr pp, hivm::RoundModeAttr rnd, BoolAttr sat)
{
    unsigned sb = tlaSrcElem.getIntOrFloatBitWidth();
    unsigned db = tlaDstElem.getIntOrFloatBitWidth();
    const bool sUnsigned = tlaSrcElem.isUnsignedInteger();
    const bool dUnsigned = tlaDstElem.isUnsignedInteger();
    auto nullPart = hivmave::VCVT_PartTypeAttr();
    auto nullPP = hivmave::VCVT_PPTypeAttr();
    MLIRContext* ctx = b.getContext();

    if ((sUnsigned && sb == 64) || (dUnsigned && db == 64))
        return failure();

    Value aveSrc = src;
    VectorType aveSrcTy = srcVecType;
    // AVE integer carriers are signless; the TLA types select signedness.
    if (sUnsigned && !srcVecType.getElementType().isSignlessInteger(sb)) {
        aveSrcTy = VectorType::get(srcVecType.getShape(), b.getIntegerType(sb));
        aveSrc = b.create<mlir::vector::BitCastOp>(loc, aveSrcTy, src).getResult();
    }

    if (sUnsigned && sb == 8 && tlaDstElem.isF32()) {
        Value gated = aveSrc;
        Value legMask = mask;
        if (hasUserMask) {
            gated = b.create<hivmave::VFAndOp>(loc, aveSrcTy, aveSrc, aveSrc, mask).getResult();
            legMask = allTrueMaskFor(fullMasks, loc, aveSrcTy, aveSrcTy);
        }
        auto midType = VectorType::get(dstVecType.getShape(), b.getIntegerType(32));
        Value mid = b.create<hivmave::VFExtUIOp>(loc, midType, gated, legMask, nullPart, pp).getResult();
        Value midMask = hasUserMask ? allTrueMaskFor(fullMasks, loc, midType, midType) : legMask;
        return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, mid, midMask, rnd, nullPart).getResult();
    }

    if (sUnsigned && sb == 8 && tlaDstElem.isF16())
        return b.create<hivmave::VFUIntToFpOp>(loc, dstVecType, aveSrc, mask, hivm::RoundModeAttr(), part).getResult();
    if (tlaSrcElem.isF16() && dUnsigned && db == 8)
        return b.create<hivmave::VFFpToUIntOp>(loc, dstVecType, src, mask, rnd, sat, part).getResult();

    if (sUnsigned != dUnsigned && isa<IntegerType>(tlaSrcElem) && isa<IntegerType>(tlaDstElem)) {
        Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(ctx));
        if (dUnsigned && sb == 16 && db == 32) {
            Value slot = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(part.getValue()), 32);
            return b.create<hivm_regbaseintrins::VcvtiiS162U32InstrOp>(loc, dstVecType, aveSrc, preg, slot).getResult();
        }
        Value saturation = b.create<arith::ConstantIntOp>(loc, sat.getValue() ? 1 : 0, 32);
        if (dUnsigned && sb == 32 && db == 8) {
            Value slot = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(pp.getValue()), 32);
            return b.create<hivm_regbaseintrins::VcvtiiS322U8InstrOp>(loc, dstVecType, aveSrc, preg, saturation, slot)
                .getResult();
        }
        Value slot = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(part.getValue()), 32);
        if (dUnsigned && sb == 16 && db == 8)
            return b.create<hivm_regbaseintrins::VcvtiiS162U8InstrOp>(loc, dstVecType, aveSrc, preg, saturation, slot)
                .getResult();
        if (dUnsigned && sb == 32 && db == 16)
            return b.create<hivm_regbaseintrins::VcvtiiS322U16InstrOp>(loc, dstVecType, aveSrc, preg, saturation, slot)
                .getResult();
        if (sUnsigned && sb == 32 && db == 16)
            return b.create<hivm_regbaseintrins::VcvtiiU322S16InstrOp>(loc, dstVecType, aveSrc, preg, saturation, slot)
                .getResult();
        return failure();
    }

    if (!sUnsigned || !dUnsigned)
        return failure();

    if (db > sb) {
        if (db / sb >= 4)
            return b.create<hivmave::VFExtUIOp>(loc, dstVecType, aveSrc, mask, nullPart, pp).getResult();
        return b.create<hivmave::VFExtUIOp>(loc, dstVecType, aveSrc, mask, part, nullPP).getResult();
    }
    if (db == sb)
        return failure();

    auto uni = hivm::UnsignedModeAttr::get(ctx, hivm::UnsignedMode::UI2UI);
    if (sb / db >= 4)
        return b.create<hivmave::VFTruncIOp>(loc, dstVecType, aveSrc, mask, sat, nullPart, pp, uni).getResult();
    return b.create<hivmave::VFTruncIOp>(loc, dstVecType, aveSrc, mask, sat, part, nullPP, uni).getResult();
}

// Build the AVE op or regbase intrinsic for a tla.cast, dispatching by (src, dst) element kind.
// The trait supplies rounding, saturation and register layout; the mask (source
// width) predicates active lanes.
static FailureOr<Value> createVectorCastResult(
    OpBuilder& b, AutoFullPredicateCache& fullMasks, Location loc, VectorType srcVecType, VectorType dstVecType,
    Type tlaSrcElem, Type tlaDstElem, ArrayRef<int32_t> trait, Value src, Value mask, bool hasUserMask)
{
    // trait codes: [0] reg_slot, [1] sat_mode, [2] round_mode.
    Type s = srcVecType.getElementType();
    Type d = dstVecType.getElementType();
    auto rnd = mapCastRoundMode(b, static_cast<::RoundMode>(trait[2]));
    BoolAttr sat = b.getBoolAttr(static_cast<::SatMode>(trait[1]) == ::SatMode::sat);
    auto part = mapCastPart(b, static_cast<::RegSlot>(trait[0]));
    auto pp = mapCastPP(b, static_cast<::RegSlot>(trait[0]));

    bool sFloat = isa<FloatType>(s);
    bool dFloat = isa<FloatType>(d);
    unsigned sb = s.getIntOrFloatBitWidth();
    unsigned db = d.getIntOrFloatBitWidth();

    // AVE vextf has no rounding operand and its lowering only produces f32.
    // The direct f16 -> bf16 instruction preserves every lane without a part
    // selector. It has no saturation operand, so only UNKNOWN is accepted.
    if (s.isF16() && d.isBF16()) {
        if (static_cast<::SatMode>(trait[1]) != ::SatMode::unknown)
            return emitError(loc, "f16->bf16 cast only supports UNKNOWN sat_mode"), failure();
        Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext()));
        Value round = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(rnd.getValue()), 32);
        return b.create<hivm_regbaseintrins::VcvtffF162BF16InstrOp>(loc, dstVecType, src, preg, round).getResult();
    }
    if (s.isBF16() && d.isF16()) {
        Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext()));
        Value round = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(rnd.getValue()), 32);
        Value saturation = b.create<arith::ConstantIntOp>(loc, sat.getValue(), 32);
        return b.create<hivm_regbaseintrins::VcvtffBF162F16InstrOp>(loc, dstVecType, src, preg, saturation, round)
            .getResult();
    }

    if (tlaSrcElem.isUnsignedInteger() || tlaDstElem.isUnsignedInteger())
        return createUnsignedVectorCastResult(
            b, fullMasks, loc, srcVecType, dstVecType, tlaSrcElem, tlaDstElem, src, mask, hasUserMask, part, pp, rnd,
            sat);

    // The generic AVE lowering rewrites i64 casts into two-register library calls.
    // Emit the single-register instructions for the 32xi64 representation.
    if ((s.isSignlessInteger(64) && (d.isF32() || d.isSignlessInteger(32))) ||
        (d.isSignlessInteger(64) && (s.isF32() || s.isSignlessInteger(32)))) {
        Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext()));
        Value regPart = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(part.getValue()), 32);
        if (s.isSignlessInteger(32))
            return b.create<hivm_regbaseintrins::VcvtiiS322S64InstrOp>(loc, dstVecType, src, preg, regPart).getResult();
        if (d.isSignlessInteger(32)) {
            Value saturation = b.create<arith::ConstantIntOp>(loc, sat.getValue(), 32);
            return b.create<hivm_regbaseintrins::VcvtiiS642S32InstrOp>(loc, dstVecType, src, preg, saturation, regPart)
                .getResult();
        }

        Value round = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(rnd.getValue()), 32);
        if (s.isF32()) {
            Value saturation = b.create<arith::ConstantIntOp>(loc, sat.getValue(), 32);
            return b
                .create<hivm_regbaseintrins::VcvtfiF322S64InstrOp>(
                    loc, dstVecType, src, preg, round, saturation, regPart)
                .getResult();
        }
        return b.create<hivm_regbaseintrins::VcvtifS642F32InstrOp>(loc, dstVecType, src, preg, round, regPart)
            .getResult();
    }

    // Direct FP8 conversions use f32. The f16/bf16 routes use an exact
    // E5M2-to-f16 bit conversion or compose instructions through f32.
    if (isAveFloat8Type(s) || isAveFloat8Type(d)) {
        Type partner = isAveFloat8Type(s) ? d : s;
        if (isAveFloat8Type(s) && isAveFloat8Type(d))
            return failure(); // no fp8 <-> fp8 re-encode instruction

        // Every fp8 route that is not a single instruction has the same mask
        // problem: its instructions run in different lane domains -- f32 or
        // f16 for the composed route, i8 then i16 for the e5m2 shortcut --
        // while the mask is written in the source element domain. An all-true
        // or contiguous mask is insensitive to the mismatch; a strided one is
        // not, and lanes are silently dropped.
        //
        // Fold the predicate into the data once, here, where the mask means
        // what it says, and let the instructions run all-true. A masked-off
        // element becomes zero and converts to zero, which is what the
        // hardware already yields for inactive lanes.
        //
        // Only the multi-instruction routes need this. The direct fp8 <-> f32
        // conversions are a single instruction that consumes the mask in the
        // source domain already, so gating them would emit an and, two
        // bitcasts and a pge that nothing reads.
        const bool needsGatedMask = (isa<Float8E5M2Type>(s) && d.isF16()) || !partner.isF32();
        Value gatedSrc = src;
        Value legMask = mask;
        if (hasUserMask && needsGatedMask) {
            auto srcBitsType = VectorType::get(srcVecType.getShape(), b.getIntegerType(sb));
            Value srcBits = b.create<mlir::vector::BitCastOp>(loc, srcBitsType, src).getResult();
            // and-with-self under the predicate: keeps active lanes, zeros the rest.
            Value gated = b.create<hivmave::VFAndOp>(loc, srcBitsType, srcBits, srcBits, mask).getResult();
            gatedSrc = b.create<mlir::vector::BitCastOp>(loc, srcVecType, gated).getResult();
            legMask = allTrueMaskFor(fullMasks, loc, srcVecType, srcVecType);
        }
        // E5M2 -> f16 has an exact two-instruction form (see below). It places
        // its result exactly where the composed route does -- both select the
        // source elements `part` names -- so it is simply the cheaper way to
        // spell the same cast, and is always preferred.
        if (isa<Float8E5M2Type>(s) && d.isF16())
            return createE5M2ToF16Result(b, loc, srcVecType, dstVecType, gatedSrc, legMask, part);
        if (!partner.isF32()) {
            // f16 and bf16 have no fp8 instruction, so expand the cast into the
            // two the ISA does have, through f32. The user writes one .to();
            // this is where it becomes two. Exact in the widening direction
            // (fp8 -> f32 loses nothing) and single-rounded in the other, since
            // only the f32 -> fp8 leg rounds.
            if (!partner.isF16() && !partner.isBF16())
                return failure();
            // Every vector register is 256 bytes, so the f32 intermediate holds
            // 2048 / 32 lanes whatever the endpoints are.
            auto f32Ty = b.getF32Type();
            auto midType = VectorType::get({static_cast<int64_t>(2048 / 32)}, f32Ty);
            // vextf and vtruncf each address one half of a packed register.
            // A single composed route therefore drops the other half. Convert
            // both halves and OR their disjoint output lanes together. vor is
            // an integer operation, so merge the raw destination bits before
            // restoring the requested float type. The truncation is the only
            // rounded step.
            auto even = hivmave::VCVT_PartTypeAttr::get(b.getContext(), hivmave::VCVT_PartType::PART_EVEN);
            auto odd = hivmave::VCVT_PartTypeAttr::get(b.getContext(), hivmave::VCVT_PartType::PART_ODD);
            // The f8 <-> f32 leg is a 4x conversion, so its selector is the
            // pack pattern pp0..pp3 (which byte of each 32-bit slot), NOT the
            // 2x even/odd part. The generic AVE ops carry only `part`, and
            // HIVMAVEToAVEIntrin forwards that attribute verbatim as the pp
            // immediate -- PART_EVEN/PART_ODD are numerically PP0/PP1 -- so
            // driving this leg with even/odd silently reads or writes only
            // byte positions 0 and 1 of every 4-byte group and never touches
            // 2 and 3. Half the register is dropped and the rest lands on the
            // wrong lanes. Build the leg from the pp intrinsics instead.
            //
            // Pick the pattern so the composed cast matches every other 2x
            // widen/narrow in this function, where reg_slot selects the even
            // or the odd elements: the f32 half contributes the high pp bit
            // and reg_slot the low one, i.e. pp = 2 * half + (reg_slot == one).
            // Widening then gives dst[j] = src[2j + slot], and narrowing gives
            // dst[2j + slot] = src[j], which is also what the e5m2 -> f16
            // integer-widen path above produces.
            unsigned slotBit = (static_cast<::RegSlot>(trait[0]) == ::RegSlot::one) ? 1 : 0;
            auto ppForHalf = [&](unsigned half) {
                return hivmave::VCVT_PPTypeAttr::get(
                    b.getContext(), static_cast<hivmave::VCVT_PPType>(2 * half + slotBit));
            };
            Value dstEven;
            Value dstOdd;
            if (isAveFloat8Type(s)) {
                // fp8 -> f16/bf16. Widen the two source groups the destination
                // can hold, then narrow each onto the dst even / odd lanes.
                Value midEven = createFp8ToF32Result(b, loc, midType, gatedSrc, legMask, ppForHalf(0));
                Value midOdd = createFp8ToF32Result(b, loc, midType, gatedSrc, legMask, ppForHalf(1));
                dstEven = b.create<hivmave::VFTruncFOp>(loc, dstVecType, midEven, legMask, rnd, sat, even).getResult();
                dstOdd = b.create<hivmave::VFTruncFOp>(loc, dstVecType, midOdd, legMask, rnd, sat, odd).getResult();
            } else {
                // f16/bf16 -> fp8. Widen the source even / odd lanes, then
                // narrow each into its own byte position of the destination.
                Value midEven = b.create<hivmave::VFExtFOp>(loc, midType, gatedSrc, legMask, even).getResult();
                Value midOdd = b.create<hivmave::VFExtFOp>(loc, midType, gatedSrc, legMask, odd).getResult();
                dstEven = createF32ToFp8Result(b, loc, dstVecType, midEven, legMask, rnd, sat, ppForHalf(0));
                dstOdd = createF32ToFp8Result(b, loc, dstVecType, midOdd, legMask, rnd, sat, ppForHalf(1));
            }
            auto dstBitsType = VectorType::get(dstVecType.getShape(), b.getIntegerType(d.getIntOrFloatBitWidth()));
            Value dstEvenBits = b.create<mlir::vector::BitCastOp>(loc, dstBitsType, dstEven).getResult();
            Value dstOddBits = b.create<mlir::vector::BitCastOp>(loc, dstBitsType, dstOdd).getResult();
            Value dstMask = allTrueMaskFor(fullMasks, loc, dstBitsType, dstBitsType);
            Value dstBits = b.create<hivmave::VFOrOp>(loc, dstBitsType, dstEvenBits, dstOddBits, dstMask).getResult();
            return b.create<mlir::vector::BitCastOp>(loc, dstVecType, dstBits).getResult();
        }
    }

    // The native fp8<->f32 instructions select one of four byte positions in
    // each 32-bit slot. Keep every RegSlot value intact as pp0..pp3.
    if (isAveFloat8Type(s) && d.isF32())
        return createFp8ToF32Result(b, loc, dstVecType, src, mask, pp);
    if (s.isF32() && isAveFloat8Type(d))
        return createF32ToFp8Result(b, loc, dstVecType, src, mask, rnd, sat, pp);
    // A same-type float cast has no width to convert, so it is a pure re-round:
    // vtrc applies the round mode to every lane and nothing else. It has no
    // saturation operand and no even/odd part selector, so the trait's sat_mode
    // and reg_slot are simply ignored here (the frontend rejects the values
    // that would silently mean nothing).
    if (sFloat && dFloat && s == d)
        return b.create<hivmave::VFVtrcOp>(loc, dstVecType, src, rnd, mask).getResult();
    if (sFloat && dFloat) {
        if (db < sb)
            return b.create<hivmave::VFTruncFOp>(loc, dstVecType, src, mask, rnd, sat, part).getResult();
        // Widening float cast (e.g. f16 -> f32) takes no rounding/saturation.
        return b.create<hivmave::VFExtFOp>(loc, dstVecType, src, mask, part).getResult();
    }
    if (sFloat) {
        // The FP16 to Int32 instruction takes a part and a rounding mode.
        if (s.isF16() && d.isSignlessInteger(32))
            sat = {};
        // Equal-width conversions use every lane and have no part selector.
        auto partOrNull = sb == db ? hivmave::VCVT_PartTypeAttr() : part;
        return b.create<hivmave::VFFpToSIntOp>(loc, dstVecType, src, mask, rnd, sat, partOrNull).getResult();
    }
    if (dFloat) {
        // int -> float: the ISA does not allow #rnd and #part together. A same-width
        // source carries the round mode (rounding may be needed, e.g. i32->f32); a
        // width-changing widen/narrow carries the even/odd part with no round mode
        // (i16->f32 is exact). i64 sources carry both, matching the arith lowering.
        if (sb == db)
            return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, src, mask, rnd, hivmave::VCVT_PartTypeAttr())
                .getResult();
        if (sb == 64)
            return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, src, mask, rnd, part).getResult();
        return b.create<hivmave::VFSIntToFpOp>(loc, dstVecType, src, mask, hivm::RoundModeAttr(), part).getResult();
    }
    // int -> int (signed). A 2x width step (e.g. i32<->i16) uses the
    // even/odd `part`; a 4x step (i32<->i8) uses the pack-pattern `pp` (PP0)
    // instead, matching the arith->AVE lowering. Integer casts do not round.
    auto uni = hivm::UnsignedModeAttr::get(b.getContext(), hivm::UnsignedMode::SI2SI);
    if (db < sb) {
        if (sb / db >= 4)
            return b
                .create<hivmave::VFTruncIOp>(
                    loc, dstVecType, src, mask, sat, hivmave::VCVT_PartTypeAttr(), pp, hivm::UnsignedModeAttr())
                .getResult();
        return b.create<hivmave::VFTruncIOp>(loc, dstVecType, src, mask, sat, part, hivmave::VCVT_PPTypeAttr(), uni)
            .getResult();
    }
    if (db / sb >= 4)
        return b.create<hivmave::VFExtSIOp>(loc, dstVecType, src, mask, hivmave::VCVT_PartTypeAttr(), pp).getResult();
    return b.create<hivmave::VFExtSIOp>(loc, dstVecType, src, mask, part, hivmave::VCVT_PPTypeAttr()).getResult();
}

static FailureOr<Value> createVectorReductionResult(
    OpBuilder& b, Location loc, ::tla::ReduceOp reduceOp, Type elementType, VectorType vecType, Value operand,
    Value explicitMask)
{
    if (failed(validateVectorReduction(reduceOp, elementType)))
        return failure();
    auto aveKind = getAveReductionCombiningKind(reduceOp, elementType);
    if (failed(aveKind))
        return failure();
    // The active mask is supplied explicitly by the frontend; tla.reduce no longer
    // derives one from the operand's originShape.
    if (!explicitMask)
        return reduceOp.emitError("tla.reduce requires an explicit mask"), failure();

    return b.create<hivmave::ReductionOp>(loc, vecType, *aveKind, operand, explicitMask).getResult();
}

static Value createVectorUnaryResult(
    OpBuilder& b, Location loc, ModuleOp module, VectorUnaryKind kind, Type tlaOperandType, VectorType vecType,
    Value operand, Value mask, bool hasExplicitMask)
{
    switch (kind) {
        case VectorUnaryKind::Exp:
            return b.create<hivmave::VFExpOp>(loc, vecType, operand, mask).getResult();
        case VectorUnaryKind::Log:
            return b.create<hivmave::VFLnOp>(loc, vecType, operand, mask).getResult();
        case VectorUnaryKind::Sqrt:
            return b.create<hivmave::VFSqrtOp>(loc, vecType, operand, mask).getResult();
        case VectorUnaryKind::Abs:
            return b.create<hivmave::VFAbsOp>(loc, vecType, operand, mask).getResult();
        case VectorUnaryKind::Relu: {
            if (!vecType.getElementType().isInteger(64))
                return b.create<hivmave::VFReluOp>(loc, vecType, operand, mask).getResult();
            Value zero = b.create<arith::ConstantIntOp>(loc, 0, 64);
            return createInt64Max(b, loc, module, vecType, operand, zero, mask, hasExplicitMask);
        }
        case VectorUnaryKind::Neg:
            return b.create<hivmave::VFNegOp>(loc, vecType, operand, mask).getResult();
        case VectorUnaryKind::Not:
            if (isa<::tla::VectorSSAType>(tlaOperandType))
                return b.create<hivmave::VFNotOp>(loc, vecType, operand, mask).getResult();
            if (isa<::tla::MaskSSAType>(tlaOperandType))
                return b
                    .create<hivmave::PregNotOp>(
                        loc, vecType, maskWidthAttrForMaskType(b, cast<::tla::MaskSSAType>(tlaOperandType)), operand,
                        mask)
                    .getRes();
            return nullptr;
    }
    return nullptr;
}

// The per-op vector width bundle (one 256-byte register's worth of a given
// element type). Derived fresh for each op from its own operands/result rather
// than shared across the region, so a single vec.func body can mix element
// widths (as tla.cast requires).
struct VecLowerCtx {
    int64_t lanes;
    Type elementType;
    VectorType vecType;
    VectorType maskVecType;
};

// AVE rejects ui*; map unsigned tla elements to signless of the same width.
// Signedness for Compare/Cast/Div/Reduce/Max/Min stays on the tla type.
// AscendC TypeGet<bool> → vector_s8: Bool VectorSSA lowers as i8 carriers.
static Type aveVectorElementType(Type elementType)
{
    // Int4x2 registers carry one packed Byte per lane.
    if (isInt4PairType(elementType))
        return IntegerType::get(elementType.getContext(), 8);
    if (isa<::tla::Float8E8M0Type, ::tla::HiFloat8Type>(elementType) || isFp4PairType(elementType) ||
        isBoolElementType(elementType))
        return IntegerType::get(elementType.getContext(), 8);
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
        if (intType.isUnsigned())
            return IntegerType::get(elementType.getContext(), intType.getWidth());
    }
    return elementType;
}

// Build the per-op {lanes, elementType, vecType, maskVecType} for a given
// element type. Each op derives its own types this way rather than reusing a
// region-global width: a tla.cast may have produced operands whose element
// width (hence lane count, at a fixed 256-byte register) differs from the
// region's, and same-256-byte register can hold f32 (64), f16 (128) or i8
// (256) lanes.
static FailureOr<VecLowerCtx> deriveVecCtxForElement(Type elementType)
{
    auto lanesOr = getVectorLaneCount(elementType);
    if (failed(lanesOr) || *lanesOr <= 0)
        return failure();
    int64_t lanes = *lanesOr;
    return VecLowerCtx{
        lanes, elementType, VectorType::get({lanes}, aveVectorElementType(elementType)),
        fullPregVecType(elementType.getContext())};
}

// Return the value already mapped into the helper, or clone an arith.constant
// on demand (loop bounds / index math constants are pulled in lazily this way).
static Value lookupOrCloneScalarValue(OpBuilder& b, Value value, DenseMap<Value, Value>& valueMap)
{
    if (Value mapped = valueMap.lookup(value))
        return mapped;
    if (auto ptrToInt = value.getDefiningOp<::tla::PtrToIntOp>()) {
        if (Value mapped = lookupOrCloneScalarValue(b, ptrToInt.getPtr(), valueMap)) {
            if (mapped.getType().isInteger(64)) {
                valueMap[value] = mapped;
                return mapped;
            }
            if (isa<LLVM::LLVMPointerType>(mapped.getType())) {
                Value asI64 = b.create<LLVM::PtrToIntOp>(ptrToInt.getLoc(), b.getI64Type(), mapped);
                valueMap[value] = asI64;
                return asI64;
            }
        }
    }
    Operation* def = value.getDefiningOp();
    if (!def || def->getNumResults() != 1)
        return nullptr;
    if (isa<arith::ConstantOp>(def)) {
        Operation* cloned = b.clone(*def);
        valueMap[value] = cloned->getResult(0);
        return cloned->getResult(0);
    }
    return nullptr;
}

/// Map ``newPtr`` (as i64) onto the TLA next_ptr SSA and its ``tla.ptrtoint``.
static void mapUpdateNextPtrI64(Value tlaNextPtr, Value asI64, DenseMap<Value, Value>& valueMap)
{
    valueMap[tlaNextPtr] = asI64;
    for (Operation* user : tlaNextPtr.getUsers()) {
        if (auto ptrToInt = dyn_cast<::tla::PtrToIntOp>(user))
            valueMap[ptrToInt.getResult()] = asI64;
    }
}

/// Lower a ``!tla.ptr`` SCF carrier to helper ``i64`` (or nullptr on failure).
///
/// ``valueMap`` may still hold an abstract ``!tla.ptr`` for a TensorDesc /
/// make_tensor base that UPDATE has not yet rewritten via
/// ``mapUpdateNextPtrI64``. Chase ``tla.inttoptr`` / mapped llvm.ptr / i64
/// instead of treating that as a hard error (master used to keep going with
/// the abstract type until a later silent failure).
static Value materializeTlaPtrCarrierAsI64(OpBuilder& b, Location loc, Value tlaPtr, DenseMap<Value, Value>& valueMap)
{
    Type i64Ty = b.getI64Type();
    auto asI64 = [&](Value v) -> Value {
        if (!v)
            return nullptr;
        if (v.getType().isInteger(64))
            return v;
        if (isa<IndexType>(v.getType()))
            return b.create<arith::IndexCastOp>(loc, i64Ty, v).getResult();
        if (isa<LLVM::LLVMPointerType>(v.getType()))
            return b.create<LLVM::PtrToIntOp>(loc, i64Ty, v).getResult();
        return nullptr;
    };

    if (Value mapped = valueMap.lookup(tlaPtr)) {
        if (Value ok = asI64(mapped))
            return ok;
        // Still ``!tla.ptr`` (or other abstract): fall through to defining op.
    }

    if (auto intToPtr = tlaPtr.getDefiningOp<::tla::IntToPtrOp>()) {
        Value addr = lookupOrCloneScalarValue(b, intToPtr.getAddr(), valueMap);
        if (Value ok = asI64(addr)) {
            valueMap[tlaPtr] = ok;
            return ok;
        }
    }

    return nullptr;
}

// Materialize the descriptor-defined valid-lane count for an omitted store.
// `origin_shape` describes the logical tile extent even when the addressable UB
// subview is a full physical register wide.
static FailureOr<Value> getTlaTensorValidLaneCount(
    OpBuilder& b, Location loc, Value tensorValue, DenseMap<Value, Value>& valueMap)
{
    if (auto descOp = findTensorDescProducer(tensorValue)) {
        Value origin0 = lookupOrCloneScalarValue(b, descOp.getOriginShape0(), valueMap);
        Value origin1 = lookupOrCloneScalarValue(b, descOp.getOriginShape1(), valueMap);
        if (!origin0 || !origin1)
            return failure();
        return b.createOrFold<arith::MulIOp>(loc, origin0, origin1);
    }
    return failure();
}

static std::optional<int64_t> getStaticTlaTensorValidLaneCount(Value tensorValue)
{
    auto descOp = findTensorDescProducer(tensorValue);
    if (!descOp)
        return std::nullopt;
    auto origin0 = getConstantIntValue(descOp.getOriginShape0());
    auto origin1 = getConstantIntValue(descOp.getOriginShape1());
    if (!origin0 || !origin1 || *origin0 < 0 || *origin1 < 0 ||
        (*origin1 != 0 && *origin0 > std::numeric_limits<int64_t>::max() / *origin1))
        return std::nullopt;
    return *origin0 * *origin1;
}

struct IndexInterval {
    int64_t lower;
    int64_t upper;
};

// arith integer operations wrap at their SSA result width. An interval that
// exceeds that signed range cannot be propagated as ordinary int64 arithmetic:
// the wrapped value observed by a later index_cast may be completely different.
static bool intervalFitsValueType(Value value, const IndexInterval& interval)
{
    auto integerType = dyn_cast<IntegerType>(value.getType());
    if (!integerType)
        return true;
    unsigned width = integerType.getWidth();
    if (width == 0 || width > 64)
        return false;
    if (width == 64)
        return true;
    int64_t min = -(int64_t{1} << (width - 1));
    int64_t max = (int64_t{1} << (width - 1)) - 1;
    return interval.lower >= min && interval.upper <= max;
}

// Evaluate the small affine subset produced by tiled vector loops. This is
// deliberately conservative: unknown values retain the PLT tail path.
static std::optional<IndexInterval> getIndexInterval(Value value)
{
    if (std::optional<int64_t> constant = getConstantIntValue(value))
        return IndexInterval{*constant, *constant};

    if (auto arg = dyn_cast<BlockArgument>(value)) {
        auto forOp = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
        if (!forOp || arg.getArgNumber() != 0)
            return std::nullopt;
        auto lowerInterval = getIndexInterval(forOp.getLowerBound());
        auto upperInterval = getIndexInterval(forOp.getUpperBound());
        auto stepInterval = getIndexInterval(forOp.getStep());
        if (!lowerInterval || lowerInterval->lower != lowerInterval->upper || !upperInterval ||
            upperInterval->lower != upperInterval->upper || !stepInterval || stepInterval->lower != stepInterval->upper)
            return std::nullopt;
        int64_t lower = lowerInterval->lower;
        int64_t upper = upperInterval->lower;
        int64_t step = stepInterval->lower;
        if (step <= 0 || upper <= lower)
            return std::nullopt;
        int64_t span;
        int64_t lastOffset;
        int64_t last;
        if (__builtin_sub_overflow(upper, lower, &span) || __builtin_sub_overflow(span, int64_t{1}, &span) ||
            __builtin_mul_overflow(span / step, step, &lastOffset) || __builtin_add_overflow(lower, lastOffset, &last))
            return std::nullopt;
        IndexInterval result{lower, last};
        if (intervalFitsValueType(value, result))
            return result;
        return std::nullopt;
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        auto interval = getIndexInterval(cast.getIn());
        if (!interval)
            return std::nullopt;
        if (auto destType = dyn_cast<IntegerType>(cast.getType())) {
            unsigned width = destType.getWidth();
            if (width == 0 || width > 64)
                return std::nullopt;
            int64_t min = width == 64 ? std::numeric_limits<int64_t>::min() : -(int64_t{1} << (width - 1));
            int64_t max = width == 64 ? std::numeric_limits<int64_t>::max() : (int64_t{1} << (width - 1)) - 1;
            if (interval->lower < min || interval->upper > max)
                return std::nullopt;
        }
        return interval;
    }
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
        auto lhs = getIndexInterval(add.getLhs());
        auto rhs = getIndexInterval(add.getRhs());
        int64_t lower;
        int64_t upper;
        if (lhs && rhs && !__builtin_add_overflow(lhs->lower, rhs->lower, &lower) &&
            !__builtin_add_overflow(lhs->upper, rhs->upper, &upper)) {
            IndexInterval result{lower, upper};
            if (intervalFitsValueType(value, result))
                return result;
        }
    }
    if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
        auto lhs = getIndexInterval(sub.getLhs());
        auto rhs = getIndexInterval(sub.getRhs());
        int64_t lower;
        int64_t upper;
        if (lhs && rhs && !__builtin_sub_overflow(lhs->lower, rhs->upper, &lower) &&
            !__builtin_sub_overflow(lhs->upper, rhs->lower, &upper)) {
            IndexInterval result{lower, upper};
            if (intervalFitsValueType(value, result))
                return result;
        }
    }
    if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
        auto lhs = getIndexInterval(mul.getLhs());
        auto rhs = getIndexInterval(mul.getRhs());
        int64_t lower;
        int64_t upper;
        if (lhs && rhs && lhs->lower >= 0 && rhs->lower >= 0 &&
            !__builtin_mul_overflow(lhs->lower, rhs->lower, &lower) &&
            !__builtin_mul_overflow(lhs->upper, rhs->upper, &upper)) {
            IndexInterval result{lower, upper};
            if (intervalFitsValueType(value, result))
                return result;
        }
    }
    if (auto min = value.getDefiningOp<arith::MinSIOp>()) {
        auto lhs = getIndexInterval(min.getLhs());
        auto rhs = getIndexInterval(min.getRhs());
        if (lhs && rhs) {
            IndexInterval result{std::min(lhs->lower, rhs->lower), std::min(lhs->upper, rhs->upper)};
            if (intervalFitsValueType(value, result))
                return result;
        }
    }
    return std::nullopt;
}

static bool isProvablyFullVectorStore(::tla::StoreOp storeOp, int64_t lanes)
{
    auto descOp = findTensorDescProducer(storeOp.getDest());
    if (!descOp)
        return false;
    auto origin0 = getIndexInterval(descOp.getOriginShape0());
    auto origin1 = getIndexInterval(descOp.getOriginShape1());
    if (!origin0 || !origin1 || origin0->lower <= 0 || origin1->lower <= 0)
        return false;
    int64_t requiredRows = lanes / origin1->lower;
    if (lanes % origin1->lower != 0)
        ++requiredRows;
    return origin0->lower >= requiredRows;
}

// Packed stores use the source-predicate-to-destination-element geometry
// validated by tla.store. Keep this defensive lookup in the lowering too: a
// malformed program may reach the pass through a nonstandard pipeline.
static FailureOr<int64_t> getStorePredicateDestLaneMultiplier(::tla::StoreOp storeOp, Type sourceElementType)
{
    auto storeDist = storeOp.getStoreDist();
    ::StoreDist dist = storeDist ? storeDist->getStoreDist() : ::StoreDist::norm;
    auto destType = dyn_cast<::tla::TlaTensorType>(storeOp.getDest().getType());
    if (!destType)
        return failure();
    auto geometry = getPackedStorePredicateGeometry(dist, sourceElementType, destType.getPtr().getPointee());
    if (!geometry)
        return failure();
    return geometry->sourcePredicateLanesPerDestElement;
}

// An omitted vector-store mask is defined by the destination tile alone. Use
// the shared ALL predicate for a destination proven to cover the physical
// store width; retain a PLT tail predicate otherwise. Source-producing ops do
// not affect this default: callers use an explicit store mask for sparse or
// otherwise non-default writes.
static FailureOr<Value> createImplicitVectorStoreMask(
    OpBuilder& b, Location loc, ::tla::StoreOp storeOp, VecLowerCtx& opCtx, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    auto storeDist = storeOp.getStoreDist();
    ::StoreDist dist = storeDist ? storeDist->getStoreDist() : ::StoreDist::norm;
    auto destLaneMultiplierOr = getStorePredicateDestLaneMultiplier(storeOp, opCtx.elementType);
    if (failed(destLaneMultiplierOr))
        return storeOp.emitError("unsupported packed-store predicate geometry"), failure();
    int64_t destLaneMultiplier = *destLaneMultiplierOr;

    // PACK_B64 builds its implicit mask through PLT.b32 before
    // converting it to the hardware b64 predicate layout.
    bool supportsImplicitAll = getBitSizeOfFixedWidthScalarType(opCtx.elementType) <= 32;

    // FIRST_ELEMENT modes ignore predicate bits in the backend. Supplying a
    // destination-tail PLT would only add work and suggest semantics that the
    // instruction does not implement, so omitted masks use the shared ALL
    // singleton directly.
    if (supportsImplicitAll && (dist == ::StoreDist::first_element_b8 || dist == ::StoreDist::first_element_b16 ||
                                dist == ::StoreDist::first_element_b32))
        return fullMasks.get(loc, opCtx.lanes);

    int64_t requiredDestLanes = (opCtx.lanes + destLaneMultiplier - 1) / destLaneMultiplier;
    auto destStaticLanes = getStaticTlaTensorValidLaneCount(storeOp.getDest());

    if (supportsImplicitAll && isProvablyFullVectorStore(storeOp, requiredDestLanes))
        return fullMasks.get(loc, opCtx.lanes);

    if (destStaticLanes) {
        int64_t maxDestLanes = opCtx.lanes / destLaneMultiplier;
        int64_t activeLanes = *destStaticLanes >= maxDestLanes ? opCtx.lanes : *destStaticLanes * destLaneMultiplier;
        Value active = b.create<arith::ConstantIndexOp>(loc, activeLanes);
        return createPredicatePlt(b, loc, opCtx.maskVecType, opCtx.lanes, active).getRes();
    }

    auto destLanes = getTlaTensorValidLaneCount(b, loc, storeOp.getDest(), valueMap);
    if (failed(destLanes))
        return failure();
    Value destLimit = *destLanes;
    if (destLaneMultiplier != 1) {
        Value multiplier = b.create<arith::ConstantIndexOp>(loc, destLaneMultiplier);
        destLimit = b.create<arith::MulIOp>(loc, destLimit, multiplier);
    }
    return createPredicatePlt(b, loc, opCtx.maskVecType, opCtx.lanes, destLimit).getRes();
}

static FailureOr<Value> castScalarForVectorElement(Value scalar, Type elementType)
{
    if (scalar.getType() == elementType)
        return scalar;
    return failure();
}

static FailureOr<Value> materializeVectorScalarValue(
    OpBuilder& b, VectorOpInfo info, DenseMap<Value, Value>& valueMap, VecLowerCtx& ctx)
{
    TlaBinaryOperands operands = info.operands;
    Value scalar = lookupOrCloneScalarValue(b, operands.rhs, valueMap);
    if (!scalar)
        return failure();
    // Shift scalar ops deliberately carry i16, matching the vshls/vshrs
    // intrinsic ABI rather than the source vector element type.
    if (info.kind == VectorBinaryKind::ShiftLeft || info.kind == VectorBinaryKind::ShiftRight)
        return scalar;
    // Match the AVE (signless) ABI; ctx.elementType may still be unsigned.
    auto castScalar = castScalarForVectorElement(scalar, aveVectorElementType(ctx.elementType));
    if (failed(castScalar))
        return failure();
    return *castScalar;
}

static FailureOr<Value> createVectorScalarBinaryResult(
    OpBuilder& b, Location loc, ModuleOp module, VectorOpInfo info, VecLowerCtx& ctx, Value lhs, Value scalar,
    Value mask)
{
    if (info.kind == VectorBinaryKind::ShiftLeft || info.kind == VectorBinaryKind::ShiftRight) {
        if (!isa<IntegerType>(ctx.elementType) || ctx.elementType.isInteger(64) || !scalar.getType().isInteger(16))
            return failure();
        const bool isUnsigned = isUnsignedInt(ctx.elementType);
        if (info.kind == VectorBinaryKind::ShiftLeft)
            return isUnsigned ?
                       b.create<hivm_regbaseintrins::VshlsUXInstrOp>(loc, ctx.vecType, lhs, scalar, mask).getRes() :
                       b.create<hivm_regbaseintrins::VshlsSXInstrOp>(loc, ctx.vecType, lhs, scalar, mask).getRes();
        return isUnsigned ?
                   b.create<hivm_regbaseintrins::VshrsUXInstrOp>(loc, ctx.vecType, lhs, scalar, mask).getRes() :
                   b.create<hivm_regbaseintrins::VshrsSXInstrOp>(loc, ctx.vecType, lhs, scalar, mask).getRes();
    }

    if (info.kind == VectorBinaryKind::Relu)
        return b.create<hivmave::VFLRelusOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();

    if (info.kind == VectorBinaryKind::Add || info.kind == VectorBinaryKind::Mul ||
        info.kind == VectorBinaryKind::Max || info.kind == VectorBinaryKind::Min) {
        if (info.kind == VectorBinaryKind::Add)
            return b.create<hivmave::VFAddsOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
        if (info.kind == VectorBinaryKind::Mul)
            return b.create<hivmave::VFMulsOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
        if (info.kind == VectorBinaryKind::Max) {
            if (ctx.elementType.isSignlessInteger(64))
                return createInt64Max(
                    b, loc, module, ctx.vecType, lhs, scalar, mask, static_cast<bool>(info.operands.mask));
            if (isUnsignedInt(ctx.elementType))
                return b.create<hivmave::VMaxsUIOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
            return b.create<hivmave::VFMaxsOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
        }
        if (isUnsignedInt(ctx.elementType))
            return b.create<hivmave::VMinsUIOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
        return b.create<hivmave::VFMinsOp>(loc, ctx.vecType, lhs, scalar, mask).getResult();
    }

    Value rhs = b.create<hivmave::VFBroadcastScalarOp>(loc, ctx.vecType, scalar).getRes();
    return createVectorBinaryResult(
        b, loc, info.kind, info.operands.lhs.getType(), ctx.elementType, ctx.vecType, lhs, rhs, mask);
}

static FailureOr<Type> lowerSCFCarrierType(Type type)
{
    if (isa<::tla::UregType>(type)) {
        OpBuilder builder(type.getContext());
        return uregVectorType(builder);
    }
    if (auto vectorType = dyn_cast<::tla::VectorSSAType>(type)) {
        auto ctx = deriveVecCtxForElement(vectorType.getElementType());
        if (failed(ctx))
            return failure();
        return Type(ctx->vecType);
    }
    if (isa<::tla::MaskSSAType>(type))
        return Type(fullPregVecType(type.getContext()));
    return type;
}

static LogicalResult lowerNestedVectorBlock(
    Block* sourceBlock, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks);

// Materialize one tla.tensor_desc as an addressable subview inside the helper.
//
// tla-lower-tensor-desc is the sole descriptor producer, so the descriptor is
// consumed directly: its coord / stride slots carry the position and pitch and
// there is no producer chain to re-walk. Carves a lanes-wide (256-byte) flat
// subview of the helper's base-memref argument at the flat offset
// coord0 * stride0 + coord1 * stride1.
static LogicalResult materializeTensorDescSubview(
    ::tla::TensorDescOp descOp, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    Location loc = descOp.getLoc();
    Value baseMemref = valueMap.lookup(descOp.getResult());
    Value descBase = descOp.getBase();
    Value address = valueMap.lookup(descBase);
    if (!address) {
        if (auto intToPtr = descBase.getDefiningOp<::tla::IntToPtrOp>())
            address = lookupOrCloneScalarValue(b, intToPtr.getAddr(), valueMap);
    }
    if (address && !address.getType().isIntOrIndex() && !isa<LLVM::LLVMPointerType>(address.getType()))
        address = {};

    auto sourceType = baseMemref ? dyn_cast<MemRefType>(baseMemref.getType()) : MemRefType{};
    if (!baseMemref && address) {
        // A pointer selected inside the helper may have no base argument
        // at the call site. Its descriptor has an unknown backing extent.
        auto bridgedType = getBridgedTensorMemrefType(descOp.getResult());
        if (failed(bridgedType))
            return failure();
        sourceType = MemRefType::get(
            {ShapedType::kDynamic}, bridgedType->getElementType(), AffineMap(), bridgedType->getMemorySpace());
    }
    if (!baseMemref && !address)
        return descOp.emitError("failed to map tla.tensor_desc base in vector helper"), failure();
    if (!sourceType || sourceType.getRank() != 1)
        return descOp.emitError("expected rank-1 base memref for vector tensor_desc"), failure();
    auto lanesOr = getVectorLaneCount(sourceType.getElementType());
    if (failed(lanesOr))
        return descOp.emitError("unsupported element type for vector tensor_desc"), failure();

    // Ptr-backed windows share one effective-address model with UPDATE:
    // prefer the live address (``inttoptr`` or mapped UPDATE next_ptr i64),
    // then apply descriptor coords. ``valueMap`` may hold the storage-identity
    // seed helper arg (for ABI dedup / memref type only) — that seed must not
    // replace a mapped live address. If the live addr is not reachable in the
    // helper, keep the seed memref (ordinary lit / non-UPDATE paths).
    if (address) {
        if (isa<LLVM::LLVMPointerType>(address.getType()))
            address = b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), address);
        Value addrI64 = ::tla::castValueToI64(b, loc, address);
        SmallVector<Value, 1> dynamicSizes;
        if (!sourceType.hasStaticShape()) {
            for (int64_t dim : sourceType.getShape()) {
                if (dim == ShapedType::kDynamic)
                    dynamicSizes.push_back(b.create<arith::ConstantIndexOp>(loc, 0));
            }
        }
        baseMemref = b.create<hivm::PointerCastOp>(loc, sourceType, addrI64, dynamicSizes).getResult();
    }

    Value rowOff = lookupOrCloneScalarValue(b, descOp.getCoord0(), valueMap);
    Value colOff = lookupOrCloneScalarValue(b, descOp.getCoord1(), valueMap);
    Value stride0 = lookupOrCloneScalarValue(b, descOp.getStride0(), valueMap);
    Value stride1 = lookupOrCloneScalarValue(b, descOp.getStride1(), valueMap);
    auto info = decodeTensorTypeInfo(descOp.getResult().getType());
    if (failed(info))
        return failure();
    if (!rowOff || !colOff || !stride0 || !stride1)
        return descOp.emitError(
                   "tensor_desc offset or stride is not reachable "
                   "from the vector helper: it is neither a constant "
                   "nor a captured scalar argument"),
               failure();
    Value flatOffset;
    if (isLinearLayout(info->layoutTag)) {
        flatOffset = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, rowOff, stride0), b.create<arith::MulIOp>(loc, colOff, stride1));
    } else if (isNZFamilyLayout(info->layoutTag)) {
        auto shape0 = lookupOrCloneScalarValue(b, descOp.getShape0(), valueMap);
        auto shape1 = lookupOrCloneScalarValue(b, descOp.getShape1(), valueMap);
        auto shape2 = lookupOrCloneScalarValue(b, descOp.getShape2(), valueMap);
        auto shape3 = lookupOrCloneScalarValue(b, descOp.getShape3(), valueMap);
        stride0 = lookupOrCloneScalarValue(b, descOp.getStride0(), valueMap);
        stride1 = lookupOrCloneScalarValue(b, descOp.getStride1(), valueMap);
        auto stride2 = lookupOrCloneScalarValue(b, descOp.getStride2(), valueMap);
        auto stride3 = lookupOrCloneScalarValue(b, descOp.getStride3(), valueMap);
        if (!shape0 || !shape1 || !shape2 || !shape3 || !stride0 || !stride1 || !stride2 || !stride3)
            return failure();
        if (info->layoutTag == LayoutTag::zN) {
            Value pc0 = b.create<arith::RemSIOp>(loc, rowOff, shape0);
            Value pc1 = b.create<arith::DivSIOp>(loc, rowOff, shape0);
            Value pc2 = b.create<arith::RemSIOp>(loc, colOff, shape2);
            Value pc3 = b.create<arith::DivSIOp>(loc, colOff, shape2);
            Value t0 = b.create<arith::MulIOp>(loc, pc0, stride0);
            Value t1 = b.create<arith::MulIOp>(loc, pc1, stride1);
            Value t2 = b.create<arith::MulIOp>(loc, pc2, stride2);
            Value t3 = b.create<arith::MulIOp>(loc, pc3, stride3);
            Value sum01 = b.create<arith::AddIOp>(loc, t0, t1);
            Value sum23 = b.create<arith::AddIOp>(loc, t2, t3);
            flatOffset = b.create<arith::AddIOp>(loc, sum01, sum23);
        } else if (info->layoutTag == LayoutTag::zNUnAlign) {
            // zNUnAlign only used in on-chip memory, and the scalar unit in a
            // vf only supports integer arith to 32 bits. Keep the entire
            // address calculation in i32, then convert its final result to the
            // index type required by memref.reinterpret_cast.
            Type i32Type = b.getI32Type();
            Value rowOffI32 = b.create<arith::IndexCastOp>(loc, i32Type, rowOff);
            Value colOffI32 = b.create<arith::IndexCastOp>(loc, i32Type, colOff);
            Value shape2I32 = b.create<arith::IndexCastOp>(loc, i32Type, shape2);
            Value stride0I32 = b.create<arith::IndexCastOp>(loc, i32Type, stride0);
            Value stride1I32 = b.create<arith::IndexCastOp>(loc, i32Type, stride1);
            Value stride2I32 = b.create<arith::IndexCastOp>(loc, i32Type, stride2);
            Value stride3I32 = b.create<arith::IndexCastOp>(loc, i32Type, stride3);
            Value shape0OrgI32 = b.create<arith::DivSIOp>(loc, stride3I32, shape2I32);
            Value pc0I32 = b.create<arith::RemSIOp>(loc, rowOffI32, shape0OrgI32);
            Value pc1I32 = b.create<arith::DivSIOp>(loc, rowOffI32, shape0OrgI32);
            Value pc2I32 = b.create<arith::RemSIOp>(loc, colOffI32, shape2I32);
            Value pc3I32 = b.create<arith::DivSIOp>(loc, colOffI32, shape2I32);
            Value t0 = b.create<arith::MulIOp>(loc, pc0I32, stride0I32);
            Value t1 = b.create<arith::MulIOp>(loc, pc1I32, stride1I32);
            Value t2 = b.create<arith::MulIOp>(loc, pc2I32, stride2I32);
            Value t3 = b.create<arith::MulIOp>(loc, pc3I32, stride3I32);
            Value sum01 = b.create<arith::AddIOp>(loc, t0, t1);
            Value sum23 = b.create<arith::AddIOp>(loc, t2, t3);
            Value flatOffsetI32 = b.create<arith::AddIOp>(loc, sum01, sum23);
            flatOffset = b.create<arith::IndexCastOp>(loc, b.getIndexType(), flatOffsetI32);
        } else {
            return descOp->emitError() << "unsupported NZFamily layout to get flatOffset";
        }
    } else {
        return descOp->emitError() << "unsupported layout to get flatOffset";
    }
    valueMap[descOp.getResult()] = ::tla::materializeFlatReinterpretSubview(b, loc, baseMemref, flatOffset, *lanesOr);
    return success();
}

/// UB pointer for an UPDATE window at the origin. Prefer the live
/// ``tensor_desc(inttoptr(i64))`` address so UPDATE next windows that share
/// a storage-identity seed memref still advance.
static FailureOr<Value> resolveEffectiveUbPtr(
    Value tensor, OpBuilder& b, Location loc, DenseMap<Value, Value>& valueMap)
{
    if (auto desc = tensor.getDefiningOp<::tla::TensorDescOp>()) {
        Value base = desc.getBase();
        if (Value mappedBase = valueMap.lookup(base)) {
            if (isa<LLVM::LLVMPointerType>(mappedBase.getType()))
                return mappedBase;
            if (mappedBase.getType().isInteger(64))
                return intToUbPtr(b, loc, mappedBase);
        }
        if (auto intToPtr = base.getDefiningOp<::tla::IntToPtrOp>()) {
            Value addr = lookupOrCloneScalarValue(b, intToPtr.getAddr(), valueMap);
            if (addr) {
                Value ptr = intToUbPtr(b, loc, addr);
                valueMap[intToPtr.getResult()] = ptr;
                return ptr;
            }
        }
    }
    if (Value mapped = valueMap.lookup(tensor)) {
        if (isa<MemRefType>(mapped.getType())) {
            auto meta = b.create<mlir::memref::ExtractStridedMetadataOp>(loc, mapped);
            Value basePtr = b.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, meta.getBaseBuffer());
            Value elemOffset = meta.getOffset();
            auto mt = cast<MemRefType>(mapped.getType());
            // SoT: Bool i1 is b8 (1), not getByteSize(i1)==0.
            int64_t elemBytes = elementStorageBytes(mt.getElementType());
            Value byteOffset = elemOffset;
            if (elemBytes > 1) {
                Value scale = b.create<arith::ConstantIndexOp>(loc, elemBytes);
                byteOffset = b.create<arith::MulIOp>(loc, elemOffset, scale);
            }
            Value byteAddr = b.create<arith::AddIOp>(loc, basePtr, byteOffset);
            return intToUbPtr(b, loc, ::tla::castValueToI64(b, loc, byteAddr));
        }
        if (isa<LLVM::LLVMPointerType>(mapped.getType()))
            return mapped;
    }
    return failure();
}

// Re-create one vec.func body op inside the helper: tla ops become AVE vector
// ops; scf control flow and index arithmetic are carried verbatim. Each op
// derives its own vector/mask width from its operands or result element type,
// so a single region may mix element widths (e.g. across tla.cast).
static LogicalResult lowerLoad(
    ::tla::LoadOp loadOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *loadOp.getOperation();
    Location loc = op.getLoc();
    Value source = valueMap.lookup(loadOp.getSource());
    if (!source)
        return failure();

    // MaskSSA result: 1/2/4/8-byte UB → predicate.
    // DIST_US / DIST_DS: AscendC MaskDist via linked bc (plds dist=1/2).
    // IR i1→plds hardcodes dist=0, so US/DS must not go through VFLoad.
    if (auto maskType = dyn_cast<::tla::MaskSSAType>(loadOp.getResult().getType())) {
        int64_t lanes = maskType.getPhysicalLanes();
        hivmave::LoadDist pattern = hivmave::LoadDist::NORM;
        if (auto loadDistAttr = loadOp.getLoadDist())
            pattern = mapTlaLoadDistToAve(loadDistAttr->getLoadDist());
        if (pattern != hivmave::LoadDist::NORM && pattern != hivmave::LoadDist::US && pattern != hivmave::LoadDist::DS)
            return loadOp.emitError("mask load_dist must be norm|us|ds"), failure();

        if (pattern == hivmave::LoadDist::US || pattern == hivmave::LoadDist::DS) {
            auto srcMemTy = dyn_cast<MemRefType>(source.getType());
            if (!srcMemTy)
                return failure();
            int64_t elemBytes = elementStorageBytes(srcMemTy.getElementType());
            Value callArg = source;
            if (elemBytes == 1)
                callArg = materializeSameShapeI8Memref(b, loc, source);
            auto callee = getOrCreateMaskLoadUsDsLibraryCall(module, loc, callArg.getType(), pattern);
            if (!callee)
                return loadOp.emitError(
                           "MaskLoadDist US/DS requires a b8/b16/b32/b64 "
                           "scalar UB (AscendC LoadAlign<T, MaskDist>)"),
                       failure();
            auto call = b.create<func::CallOp>(loc, callee, ValueRange{callArg});
            Value loaded = castMaskToPregType(b, loc, call.getResult(0), fullPregVecType(b.getContext()));
            valueMap[loadOp.getResult()] = loaded;
            return success();
        }

        auto i1MemrefOr = materializeI1MaskMemrefFromUb(b, loc, source, lanes);
        if (failed(i1MemrefOr))
            return loadOp.emitError("failed to materialize i1 memref view for tla.load MaskSSA"), failure();
        VectorType semanticMaskType = VectorType::get({lanes}, b.getI1Type());
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        auto vfLoad = createVFLoad(b, loc, semanticMaskType, *i1MemrefOr, zero, pattern, /*unaligned=*/false);
        Value loaded = castMaskToPregType(b, loc, vfLoad.getRes(), fullPregVecType(b.getContext()));
        valueMap[loadOp.getResult()] = loaded;
        return success();
    }

    // The loaded vector's element type comes from the tile memref, not the
    // region-global width: a load feeding a differently-typed op keeps its own
    // dtype.
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType)
        return failure();
    auto opCtx = deriveVecCtxForElement(sourceType.getElementType());
    if (failed(opCtx))
        return failure();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    hivmave::LoadDist pattern = hivmave::LoadDist::NORM;
    if (auto loadDistAttr = loadOp.getLoadDist())
        pattern = mapTlaLoadDistToAve(loadDistAttr->getLoadDist());
    // Dual-dest load_dist needs a second vector result.
    Value r2 = loadOp.getResult2();
    bool hasSecondVec = static_cast<bool>(r2);
    bool dual = isDualDestLoadDist(pattern);
    if (dual != hasSecondVec)
        return loadOp.emitError(
                   "dintlv load_dist requires exactly two vector results; other "
                   "load_dist values require one vector result "
                   "(optional live !tla.ptr is separate)"),
               failure();
    // Bool is b8 move-only transport (TypeGet<bool>→vector_s8); allowed on NORM
    // and b8 Dist / DINTLV_B8 / BlockLoad the same as other SupportBytes<1> types.
    bool isBoolLoad = isBoolVectorElement(sourceType.getElementType()) ||
                      (isa<::tla::VectorSSAType>(loadOp.getResult().getType()) &&
                       isBoolVectorElement(cast<::tla::VectorSSAType>(loadOp.getResult().getType()).getElementType()));
    if (pattern == hivmave::LoadDist::DINTLV_B8) {
        // Encoded byte types use i8 carriers; select from the original TLA type.
        Type elementType = cast<::tla::TlaTensorType>(loadOp.getSource().getType()).getPtr().getPointee();
        auto meta = b.create<mlir::memref::ExtractStridedMetadataOp>(loc, source);
        Value basePtr = b.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, meta.getBaseBuffer());
        // Preserve the complete view offset; every supported element is one byte.
        Value byteAddr = b.create<arith::AddIOp>(loc, basePtr, meta.getOffset());
        Value ptr = intToUbPtr(b, loc, byteAddr);
        Type pairType = LLVM::LLVMStructType::getLiteral(b.getContext(), {opCtx->vecType, opCtx->vecType});
        Value pair;
        // Bool (i1) and signed/signless i8 share the s8 dual load; u8 uses u8.
        if (elementType.isInteger(8) || isBoolElementType(elementType)) {
            Value constantZero = b.create<arith::ConstantIntOp>(loc, 0, 32);
            Value distribution = b.create<arith::ConstantIntOp>(loc, static_cast<uint32_t>(pattern), 32);
            if (elementType.isInteger(8) && elementType.isUnsignedInteger())
                pair = b.create<hivm_regbaseintrins::Vldsx2V256U8InstrOp>(
                            loc, pairType, ptr, constantZero, distribution, constantZero)
                           .getResult();
            else
                pair = b.create<hivm_regbaseintrins::Vldsx2V256S8InstrOp>(
                            loc, pairType, ptr, constantZero, distribution, constantZero)
                           .getResult();
        } else {
            // Retain typed SDK loads for low-precision floating-point elements.
            std::string suffix;
            if (isa<Float8E4M3FNType>(elementType))
                suffix = "fp8_e4m3fn";
            else if (isa<Float8E5M2Type>(elementType))
                suffix = "fp8_e5m2";
            else if (isa<::tla::HiFloat8Type>(elementType))
                suffix = "hifloat8";
            else if (isa<::tla::Float8E8M0Type>(elementType))
                suffix = "fp8_e8m0";
            else if (isa<::tla::Float4E2M1x2Type>(elementType))
                suffix = "fp4x2_e2m1";
            else if (isa<::tla::Float4E1M2x2Type>(elementType))
                suffix = "fp4x2_e1m2";
            if (suffix.empty())
                return loadOp.emitError("unsupported element type for DINTLV_B8 load"), failure();
            auto callee = getOrCreateLoadDintlvB8LibraryCall(module, loc, "_mlir_ciface_load_dintlv_b8_" + suffix);
            auto slotOr = fullMasks.createDintlvResultSlot(b, loc, pairType);
            if (failed(slotOr))
                return loadOp.emitError("failed to allocate DINTLV_B8 result slot in outlined func entry"), failure();
            b.create<func::CallOp>(loc, callee, ValueRange{ptr, *slotOr});
            pair = b.create<LLVM::LoadOp>(loc, pairType, *slotOr);
        }
        valueMap[loadOp.getResult()] = b.create<LLVM::ExtractValueOp>(loc, pair, 0);
        valueMap[r2] = b.create<LLVM::ExtractValueOp>(loc, pair, 1);
        return success();
    }
    if (pattern == hivmave::LoadDist::DINTLV_B32 || pattern == hivmave::LoadDist::DINTLV_B16) {
        // NPU-IR's partial AVE conversion can leave this two-result load
        // unconverted when its consumer is a register helper call. Emit the
        // same native intrinsic here, with the complete view offset, so the
        // load composes directly with either native ops or helpers.
        auto metadata = b.create<mlir::memref::ExtractStridedMetadataOp>(loc, source);
        Value base = b.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, metadata.getBaseBuffer());
        Value elementBytes =
            b.create<arith::ConstantIndexOp>(loc, sourceType.getElementType().getIntOrFloatBitWidth() / 8);
        Value offset = b.create<arith::MulIOp>(loc, metadata.getOffset(), elementBytes);
        Value address = b.create<arith::AddIOp>(loc, base, offset);
        Value pointer = intToUbPtr(b, loc, ::tla::castValueToI64(b, loc, address));
        Value constantZero = b.create<arith::ConstantIntOp>(loc, 0, 32);
        Value distribution = b.create<arith::ConstantIntOp>(loc, static_cast<uint32_t>(pattern), 32);
        auto pairType = LLVM::LLVMStructType::getLiteral(b.getContext(), {opCtx->vecType, opCtx->vecType});
        Value pair;
        Type elemTy = sourceType.getElementType();
        if (elemTy.isF32()) {
            pair = b.create<hivm_regbaseintrins::Vldsx2V64F32InstrOp>(
                        loc, pairType, pointer, constantZero, distribution, constantZero)
                       .getResult();
        } else if (elemTy.isInteger(32)) {
            // AscendC DINTLV_B32: b32 deinterleave (i32/u32 share u32 mover).
            pair = b.create<hivm_regbaseintrins::Vldsx2V64U32InstrOp>(
                        loc, pairType, pointer, constantZero, distribution, constantZero)
                       .getResult();
        } else if (elemTy.isBF16()) {
            pair = b.create<hivm_regbaseintrins::Vldsx2V128BF16InstrOp>(
                        loc, pairType, pointer, constantZero, distribution, constantZero)
                       .getResult();
        } else if (elemTy.isF16()) {
            pair = b.create<hivm_regbaseintrins::Vldsx2V128F16InstrOp>(
                        loc, pairType, pointer, constantZero, distribution, constantZero)
                       .getResult();
        } else {
            // Integer b16 loads preserve bits; signed/unsigned share u16 mover.
            pair = b.create<hivm_regbaseintrins::Vldsx2V128U16InstrOp>(
                        loc, pairType, pointer, constantZero, distribution, constantZero)
                       .getResult();
        }
        valueMap[loadOp.getResult()] = b.create<LLVM::ExtractValueOp>(loc, pair, 0);
        valueMap[r2] = b.create<LLVM::ExtractValueOp>(loc, pair, 1);
        return success();
    }
    // Strided block load (vsldb): lower through a library call like the
    // store-with-stride path. The tile memref supplies the base address;
    // the gather reads all 8 DataBlocks, so the predicate is all-true.
    if (auto blockStrideAttr = loadOp.getBlockStrideAttr()) {
        Value blockStrideVal = b.create<arith::ConstantIntOp>(loc, blockStrideAttr.getInt(), 32);
        int64_t repeatStrideImm = loadOp.getRepeatStrideAttr() ? loadOp.getRepeatStrideAttr().getInt() : 0;
        Value repeatStrideVal = b.create<arith::ConstantIntOp>(loc, repeatStrideImm, 32);
        Type tlaElem = sourceType.getElementType();
        if (auto tlaTensor = dyn_cast<::tla::TlaTensorType>(loadOp.getSource().getType()))
            tlaElem = tlaTensor.getPtr().getPointee();
        std::string calleeName = getLoadWithStrideLibraryCallName(tlaElem);
        if (calleeName.empty())
            return loadOp.emitError("unsupported element type for tla.load with block_stride: ") << tlaElem, failure();
        Value loadSrc = source;
        if (isBoolElementType(tlaElem))
            // Bool rides load_with_stride_int8; same-shape i8 view matches Int8 ABI.
            loadSrc = materializeSameShapeI8Memref(b, loc, source);
        bool viaBytes = blockStrideUsesByteRegister(tlaElem);
        Type resultVecType = viaBytes ? byteRegisterTypeFor(opCtx->vecType) : Type(opCtx->vecType);
        auto callee = getOrCreateLoadWithStrideLibraryCall(module, loc, resultVecType, loadSrc.getType(), calleeName);
        if (!callee)
            return failure();
        // The gather reads all 8 DataBlocks, so the predicate is all-true;
        // build it via PgePattern::ALL like the other no-mask lowerings
        // (a dense<true> arith constant is not selectable in bisheng).
        Value allTrue = allTrueMaskFor(fullMasks, loc, opCtx->vecType, loadOp.getResult().getType());
        auto call = b.create<func::CallOp>(loc, callee, ValueRange{loadSrc, blockStrideVal, repeatStrideVal, allTrue});
        Value loaded = call.getResult(0);
        if (viaBytes)
            loaded = b.create<mlir::vector::BitCastOp>(loc, opCtx->vecType, loaded).getResult();
        valueMap[loadOp.getResult()] = loaded;
        return success();
    }

    bool unaligned = loadOp.getUnalignedUbAccess().value_or(false);
    if (isAlignB64Elem(sourceType.getElementType()) &&
        cast<::tla::VectorSSAType>(loadOp.getResult().getType()).getValidLanes() == 32) {
        Value loaded;
        if (unaligned) {
            auto addr = memrefByteAddrI64(b, loc, source);
            if (failed(addr))
                return loadOp.emitError("64-bit unaligned load requires an effective UB address"), failure();
            Value ptr = intToUbPtr(b, loc, *addr);
            auto ureg = emitVldas(b, loc, ptr);
            Value bytes = b.create<arith::ConstantIntOp>(loc, 256, 32);
            auto result = emitVldusPost(b, loc, ptr, *ureg, bytes, b.getI32Type());
            if (failed(result))
                return failure();
            loaded = std::get<0>(*result);
        } else {
            auto words = VectorType::get({64}, b.getI32Type());
            auto wordsMem = materializeDataMemref(b, loc, source, b.getI32Type());
            if (failed(wordsMem))
                return loadOp.emitError("failed to materialize i32 memref view for 64-bit VectorSSA load"), failure();
            loaded = createVFLoad(b, loc, words, *wordsMem, zero, pattern, false).getRes();
        }
        valueMap[loadOp.getResult()] = b.create<mlir::vector::BitCastOp>(loc, opCtx->vecType, loaded).getResult();
        return success();
    }

    Value loadMemref = source;
    if (isBoolLoad) {
        loadMemref = materializeSameShapeI8Memref(b, loc, source);
    } else if (unaligned && sourceType.getElementType().isUnsignedInteger()) {
        // The AVE unaligned load creates an LLVM struct and needs signless lanes.
        auto carrier = materializeDataMemref(b, loc, source, opCtx->vecType.getElementType());
        if (failed(carrier))
            return failure();
        loadMemref = *carrier;
    }
    auto vfLoad = createVFLoad(b, loc, opCtx->vecType, loadMemref, zero, pattern, unaligned);
    valueMap[loadOp.getResult()] = vfLoad.getRes();
    return success();
}

static LogicalResult materializeBlockStrides(
    Operation* op, Value blockStride, Value repeatStride, OpBuilder& b, DenseMap<Value, Value>& valueMap,
    Value& blockOut, Value& repeatOut)
{
    Location loc = op->getLoc();
    blockOut = valueMap.lookup(blockStride);
    if (!blockOut)
        blockOut = lookupOrCloneScalarValue(b, blockStride, valueMap);
    repeatOut = valueMap.lookup(repeatStride);
    if (!repeatOut)
        repeatOut = lookupOrCloneScalarValue(b, repeatStride, valueMap);
    if (!blockOut || !repeatOut)
        return op->emitError("failed to materialize DATA_BLOCK_COPY strides"), failure();
    blockOut = castToI32(b, loc, blockOut);
    repeatOut = castToI32(b, loc, repeatOut);
    return success();
}

static LogicalResult lowerLoadUpdate(
    ::tla::LoadUpdateOp loadOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = loadOp.getLoc();
    Value source = valueMap.lookup(loadOp.getSource());
    if (!source)
        return loadOp.emitError("load_update source is not mapped in vector helper"), failure();
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType)
        return loadOp.emitError("load_update source map is not a memref in vector helper"), failure();
    Type tlaSrcElem = sourceType.getElementType();
    if (auto tlaTensor = dyn_cast<::tla::TlaTensorType>(loadOp.getSource().getType()))
        tlaSrcElem = tlaTensor.getPtr().getPointee();
    Type tlaResultElem = tlaSrcElem;
    if (auto tlaVec = dyn_cast<::tla::VectorSSAType>(loadOp.getResult().getType()))
        tlaResultElem = tlaVec.getElementType();
    if (tlaSrcElem != tlaResultElem)
        return loadOp.emitError("load_update source and result element types must match"), failure();
    // Bool Align UPDATE: AscendC move-only b8 (emitted as s8 post alias).
    auto opCtx = deriveVecCtxForElement(tlaSrcElem);
    if (failed(opCtx))
        return loadOp.emitError("unsupported element type for load_update vector helper"), failure();
    auto ptrOr = resolveEffectiveUbPtr(loadOp.getSource(), b, loc, valueMap);
    if (failed(ptrOr))
        return loadOp.emitError(
                   "load_update requires an effective UB address "
                   "(tensor_desc over live inttoptr / materialized memref)"),
               failure();

    // DATA_BLOCK_COPY UPDATE: no HIVMC vsldb.post — AIV bc returns the pointer.
    if (Value blockStrideVal = loadOp.getBlockStride()) {
        Value blockStride;
        Value repeatStride;
        if (failed(materializeBlockStrides(
                loadOp, blockStrideVal, loadOp.getPostUpdateStride(), b, valueMap, blockStride, repeatStride)))
            return failure();
        std::string suffix = dataBlockCopyElementSuffix(tlaSrcElem);
        if (suffix.empty())
            return loadOp.emitError("unsupported element type for DATA_BLOCK_COPY load_update: ") << tlaSrcElem,
                   failure();
        bool viaBytes = blockStrideUsesByteRegister(tlaSrcElem);
        Type resultVecType = viaBytes ? byteRegisterTypeFor(opCtx->vecType) : Type(opCtx->vecType);
        auto callee = getOrCreateLoadUpdateRetptrLibraryCall(
            module, loc, resultVecType, ("_mlir_ciface_load_with_stride_post_update_retptr_" + suffix));
        Value allTrue = allTrueMaskFor(fullMasks, loc, opCtx->vecType, loadOp.getResult().getType());
        Value preg = castMaskToPregType(b, loc, allTrue, fullPregVecType(b.getContext()));
        auto slotOr = fullMasks.createRetptrSlot(b, loc, LLVM::LLVMPointerType::get(b.getContext()), b.getI64Type());
        if (failed(slotOr))
            return loadOp.emitError("failed to allocate load_update retptr slot in outlined func entry"), failure();
        auto call = b.create<func::CallOp>(loc, callee, ValueRange{*ptrOr, blockStride, repeatStride, preg, *slotOr});
        Value loaded = call.getResult(0);
        if (viaBytes)
            loaded = b.create<mlir::vector::BitCastOp>(loc, opCtx->vecType, loaded).getResult();
        Value newAddr = b.create<LLVM::LoadOp>(loc, b.getI64Type(), *slotOr);
        valueMap[loadOp.getResult()] = loaded;
        mapUpdateNextPtrI64(loadOp.getNextPtr(), newAddr, valueMap);
        return success();
    }

    Value stride = valueMap.lookup(loadOp.getPostUpdateStride());
    if (!stride)
        stride = lookupOrCloneScalarValue(b, loadOp.getPostUpdateStride(), valueMap);
    if (!stride)
        return loadOp.emitError("failed to materialize post_update_stride in vector helper"), failure();
    if (Value second = loadOp.getResult2()) {
        // B32 distribution also accepts b64 values as pairs of 32-bit words.
        int64_t distBytes = std::min<int64_t>(elementStorageBytes(tlaSrcElem), 4);
        auto carrierTy = dualTransferCarrierType(b.getContext(), distBytes);
        auto dist = mapTlaLoadDistToAve(loadOp.getLoadDist()->getLoadDist());
        Value packed = emitVldsx2Post(b, loc, *ptrOr, stride, tlaSrcElem, carrierTy, dist);
        auto extract = [&](int64_t index) -> Value {
            Value lane = b.create<LLVM::ExtractValueOp>(loc, packed, index);
            if (carrierTy == opCtx->vecType)
                return lane;
            return b.create<mlir::vector::BitCastOp>(loc, opCtx->vecType, lane);
        };
        valueMap[loadOp.getResult()] = extract(0);
        valueMap[second] = extract(1);
        Value next = b.create<LLVM::ExtractValueOp>(loc, packed, 2);
        mapUpdateNextPtrI64(loadOp.getNextPtr(), b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), next), valueMap);
        return success();
    }
    Type elemTy = sourceType.getElementType();
    FailureOr<std::pair<Value, Value>> emitted = emitVldsx1Post(b, loc, *ptrOr, stride, elemTy, opCtx->vecType);
    if (failed(emitted))
        return loadOp.emitError("unsupported element type for tla.load_update: ") << elemTy, failure();
    valueMap[loadOp.getResult()] = emitted->first;
    mapUpdateNextPtrI64(
        loadOp.getNextPtr(), b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), emitted->second), valueMap);
    return success();
}

static LogicalResult lowerLoadUnalignUpdate(
    ::tla::LoadUnalignUpdateOp op, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    Location loc = op.getLoc();
    // AscendC: HIVM $cceVL = postUpdateStride×sizeof(T); next_ptr = HW newPtr.
    Value postUpdateStride = valueMap.lookup(op.getPostUpdateStride());
    if (!postUpdateStride)
        postUpdateStride = lookupOrCloneScalarValue(b, op.getPostUpdateStride(), valueMap);
    if (!postUpdateStride)
        return op.emitError("failed to materialize post_update_stride"), failure();
    auto ptrOr = resolveEffectiveUbPtr(op.getSource(), b, loc, valueMap);
    if (failed(ptrOr))
        return op.emitError("load_unalign_update requires a live UB address"), failure();
    Value source = valueMap.lookup(op.getSource());
    auto sourceType = dyn_cast_or_null<MemRefType>(source ? source.getType() : Type());
    if (!sourceType)
        return failure();
    Type elemType = sourceType.getElementType();
    Value countBytes = strideElemsToByteOffI32(b, loc, postUpdateStride, elemType);
    Value uregIn = valueMap.lookup(op.getUregIn());
    if (!uregIn)
        return op.emitError("load_unalign_update ureg_in not mapped in vector helper"), failure();
    Value secondResult = op.getResult2();
    Type carrierElem = elemType;
    Value firstCount = countBytes;
    Value secondCount;
    if (secondResult) {
        int64_t distBytes = std::min<int64_t>(elementStorageBytes(elemType), 4);
        carrierElem = IntegerType::get(b.getContext(), distBytes * 8);
        std::tie(firstCount, secondCount) = splitUnalignUpdateCount(b, loc, countBytes);
    }
    auto emitted = emitVldusPost(b, loc, *ptrOr, uregIn, firstCount, carrierElem);
    if (failed(emitted))
        return op.emitError("unsupported element type for tla.load_unalign_update"), failure();
    auto [first, uregOut, nextPtr] = *emitted;
    if (secondResult) {
        // Both halves use the carrier validated by the first emission.
        auto [second, nextUreg, finalPtr] = *emitVldusPost(b, loc, nextPtr, uregOut, secondCount, carrierElem);
        auto pair = b.create<hivmave::VFDeInterleaveOp>(
            loc, TypeRange{first.getType(), first.getType()}, ValueRange{first, second});
        Type resultType = dataVectorType(b, elemType);
        auto decode = [&](Value bits) -> Value {
            return bits.getType() == resultType ? bits :
                                                  b.create<mlir::vector::BitCastOp>(loc, resultType, bits).getResult();
        };
        first = decode(pair->getResult(0));
        valueMap[secondResult] = decode(pair->getResult(1));
        uregOut = nextUreg;
        nextPtr = finalPtr;
    }
    valueMap[op.getResult()] = first;
    valueMap[op.getUregOut()] = uregOut;
    mapUpdateNextPtrI64(op.getNextPtr(), b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), nextPtr), valueMap);
    return success();
}

/// tla.bitcast preserves register bits while changing the element type and
/// lane count, lowering to MLIR's vector.bitcast.
static LogicalResult lowerBitCast(::tla::BitCastOp bitCastOp, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    Location loc = bitCastOp.getLoc();
    Type tlaSrcElem = bitCastOp.getSource().getType().getElementType();
    Type tlaDstElem = bitCastOp.getResult().getType().getElementType();
    if (isBoolVectorElement(tlaSrcElem) || isBoolVectorElement(tlaDstElem))
        return bitCastOp.emitError(
                   "Bool VectorSSA is move-only (load/store); "
                   "bitcast is not supported"),
               failure();
    Value src = valueMap.lookup(bitCastOp.getSource());
    if (!src)
        return failure();
    auto srcVecType = dyn_cast<VectorType>(src.getType());
    if (!srcVecType)
        return bitCastOp.emitError("tla.bitcast source is not a vector value"), failure();
    Type dstElementType = aveVectorElementType(tlaDstElem);
    // Values stay signless; signedness lives in the !tla.vector type only.
    if (auto intType = dyn_cast<IntegerType>(dstElementType))
        if (!intType.isSignless())
            dstElementType = b.getIntegerType(intType.getWidth());
    auto dstLanes = getVectorLaneCount(dstElementType);
    if (failed(dstLanes))
        return bitCastOp.emitError("unsupported register element width"), failure();
    auto dstVecType = VectorType::get({*dstLanes}, dstElementType);
    valueMap[bitCastOp.getResult()] = b.create<mlir::vector::BitCastOp>(loc, dstVecType, src).getResult();
    return success();
}

/// tla.cast: element-type conversion. The source vector already carries its
/// width; the destination width is one full 256-byte register's worth of the
/// target element type. The cast op picks the AVE cast (vtruncf / vfptosi /
/// vsitofp / vtrunci / ...) from the (src,dst) element kinds.
static LogicalResult lowerCast(
    ::tla::CastOp castOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = castOp.getLoc();
    Value src = valueMap.lookup(castOp.getSource());
    if (!src)
        return failure();
    auto srcVecType = dyn_cast<VectorType>(src.getType());
    if (!srcVecType)
        return castOp.emitError("tla.cast source is not a vector value"), failure();
    Type tlaSrcElem = castOp.getSource().getType().getElementType();
    Type tlaDstElem = castOp.getResult().getType().getElementType();
    auto dstLanesOr = getVectorLaneCount(tlaDstElem);
    if (failed(dstLanesOr))
        return castOp.emitError("unsupported tla.cast destination element type"), failure();
    if (isBoolVectorElement(tlaSrcElem) || isBoolVectorElement(tlaDstElem))
        return castOp.emitError(
                   "Bool VectorSSA is move-only (load/store); "
                   "cast is not supported"),
               failure();
    auto dstVecType = VectorType::get({*dstLanesOr}, aveVectorElementType(tlaDstElem));
    auto trait = castOp.getTrait();
    const bool toFp4 = isFp4PairType(tlaDstElem);
    const bool toInt4 = isInt4PairType(tlaDstElem);
    const bool hasUserMask = static_cast<bool>(castOp.getMask());
    // Packed 4-bit narrowing uses one mask bit per pair by default.
    const int64_t maskLanes = toFp4 || toInt4 ? 64 : srcVecType.getNumElements();
    Value mask = hasUserMask ? valueMap.lookup(castOp.getMask()) : fullMasks.get(loc, maskLanes);
    if (!mask)
        return failure();
    if (isInt4PairType(tlaSrcElem)) {
        // int4x2_t -> b16 has no rounding or saturation operand.
        auto round = static_cast<::RoundMode>(trait[2]);
        if (round != ::RoundMode::cast_none)
            return castOp.emitError("Int4x2 source casts require CAST_NONE"), failure();
        bool toF16 = tlaDstElem.isF16();
        bool toBF16 = tlaDstElem.isBF16();
        bool toI16 = tlaDstElem.isSignlessInteger(16);
        if (!toF16 && !toBF16 && !toI16)
            return castOp.emitError("Int4x2 source only widens to f16/bf16/i16"), failure();
        Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext()));
        Value pp = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(trait[0]), 32);
        Value result;
        if (toF16) {
            result = b.create<hivm_regbaseintrins::VcvtifS4x22F16InstrOp>(loc, dstVecType, src, preg, pp).getResult();
        } else if (toBF16) {
            result = b.create<hivm_regbaseintrins::VcvtifS4x22BF16InstrOp>(loc, dstVecType, src, preg, pp).getResult();
        } else {
            result = b.create<hivm_regbaseintrins::VcvtiiS4x22S16InstrOp>(loc, dstVecType, src, preg, pp).getResult();
        }
        valueMap[castOp.getResult()] = result;
        return success();
    }
    if (isa<::tla::Float8E8M0Type>(tlaDstElem)) {
        // Clear the sign before extracting BF16 bits [14:7]. After the left
        // shift, both infinities are 0xff00 and NaNs still have fraction bits.
        auto bitsType = VectorType::get(srcVecType.getShape(), b.getI16Type());
        Value bits = b.create<mlir::vector::BitCastOp>(loc, bitsType, src);
        Value one = b.create<arith::ConstantIntOp>(loc, 1, 16);
        Value eight = b.create<arith::ConstantIntOp>(loc, 8, 16);
        Value sign = b.create<arith::ConstantOp>(loc, b.getI1Type(), b.getBoolAttr(false));
        Value shifted = b.create<hivmave::VFShlsOp>(loc, bitsType, bits, one, mask, sign);
        Value exponent = b.create<hivmave::VFShrsOp>(loc, bitsType, shifted, eight, mask, sign);
        Value infinityBits = b.create<arith::ConstantIntOp>(loc, 0xff00, 16);
        Value isInfinity = b.create<hivmave::VFCmpS>(
            loc, fullPregVecType(b.getContext()), hivmave::CmpType::EQ, shifted, infinityBits, mask);
        Value maxFinite = b.create<arith::ConstantIntOp>(loc, 0xfe, 16);
        Value maxFiniteLanes = b.create<hivmave::VFBroadcastScalarOp>(loc, bitsType, maxFinite).getRes();
        Value encoded = b.create<hivmave::VFSelectOp>(loc, bitsType, isInfinity, maxFiniteLanes, exponent);
        auto part = mapCastPart(b, static_cast<::RegSlot>(trait[0]));
        auto unsignedMode = hivm::UnsignedModeAttr::get(b.getContext(), hivm::UnsignedMode::UI2UI);
        Value result = b.create<hivmave::VFTruncIOp>(
            loc, dstVecType, encoded, mask, b.getBoolAttr(true), part, hivmave::VCVT_PPTypeAttr(), unsignedMode);
        valueMap[castOp.getResult()] = result;
        return success();
    }
    std::string libraryCast;
    if (isa<HiFloat8Type>(tlaSrcElem) || isa<HiFloat8Type>(tlaDstElem)) {
        libraryCast = getHiFloat8CastLibraryCallName(tlaSrcElem, tlaDstElem, trait);
    } else if (isFp4PairType(tlaSrcElem) || toFp4) {
        libraryCast = getFp4CastLibraryCallName(tlaSrcElem, tlaDstElem, trait);
    } else if (toInt4) {
        // The Regbase and device LLVM definitions of the Int4x2 narrowing
        // intrinsics disagree on their operand or result types.
        libraryCast = tlaSrcElem.isF16() ? "lowp_cast_f16_to_i4x2" : "lowp_cast_i16_to_i4x2";
        if (tlaSrcElem.isF16())
            libraryCast += getPacked4RoundSuffix(static_cast<::RoundMode>(trait[2])).str();
        libraryCast += static_cast<::SatMode>(trait[1]) == ::SatMode::sat ? "_sat_slot" : "_nosat_slot";
        libraryCast += std::to_string(trait[0]);
    } else if (isa<Float8E8M0Type>(tlaSrcElem)) {
        libraryCast = "lowp_cast_e8m0_to_bf16_slot" + std::to_string(trait[0]);
    }
    if (!libraryCast.empty()) {
        auto pregType = fullPregVecType(b.getContext());
        Value preg = castMaskToPregType(b, loc, mask, pregType);
        auto callee = getOrCreateInlineAivLibraryCall(module, loc, {srcVecType, pregType}, dstVecType, libraryCast);
        valueMap[castOp.getResult()] = b.create<func::CallOp>(loc, callee, ValueRange{src, preg}).getResult(0);
        return success();
    }
    // Reject casts whose source or destination element type has no AVE cast path
    // (i1/bool, f64). Unsigned routes are gated by createUnsignedVectorCastResult.
    if (!isSupportedCastElementType(tlaSrcElem) || !isSupportedCastElementType(tlaDstElem))
        return castOp.emitError(
                   "unsupported tla.cast element type: only signed "
                   "integers (i8/i16/i32/i64), unsigned integers "
                   "(u8/u16/u32) and floats (f16/bf16/f32) "
                   "are supported; bool and f64 are not"),
               failure();
    auto result = createVectorCastResult(
        b, fullMasks, loc, srcVecType, dstVecType, tlaSrcElem, tlaDstElem, trait, src, mask, hasUserMask);
    if (failed(result))
        return castOp.emitError("unsupported tla.cast element type conversion"), failure();
    valueMap[castOp.getResult()] = *result;
    return success();
}

static LogicalResult lowerFull(
    ::tla::FullOp fullOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *fullOp.getOperation();
    Location loc = op.getLoc();
    // No vector operand to key off: the broadcast width comes from the result
    // VectorSSA element type.
    auto resultType = dyn_cast<::tla::VectorSSAType>(fullOp.getResult().getType());
    if (!resultType)
        return fullOp.emitError("expected !tla.vector result type"), failure();
    auto opCtx = deriveVecCtxForElement(resultType.getElementType());
    if (failed(opCtx))
        return fullOp.emitError("unsupported tla.full result element type"), failure();
    // AVE vectors are signless and Bool/hif8/e8m0/fp4x2 ride an i8 carrier.
    Type carrierElem = aveVectorElementType(resultType.getElementType());
    // An optional tla mask predicates the broadcast lanes; all-true when absent.
    Value mask;
    if (fullOp.getMask()) {
        mask = valueMap.lookup(fullOp.getMask());
        if (!mask)
            return failure();
    } else {
        mask = allTrueMaskFor(fullMasks, loc, opCtx->vecType, fullOp.getResult().getType());
    }

    // A one-lane vector fragment source (e.g. a tla.reduce result) is already
    // register-resident; broadcast lane 0 across the register with AVE
    // vector_broadcast. This stays entirely in the vector register file and
    // avoids reading a vector lane into a scalar register, which the SIMD
    // backend cannot legalize.
    if (Value mapped = valueMap.lookup(fullOp.getValue())) {
        if (auto srcVecType = dyn_cast<VectorType>(mapped.getType())) {
            if (srcVecType.getElementType() != carrierElem)
                return fullOp.emitError("tla.full source element type ")
                           << srcVecType.getElementType() << " does not match vector element type " << carrierElem,
                       failure();
            if (resultType.getElementType().isInteger(64)) {
                // The 64-bit AVE vector-broadcast ABI does not legalize either;
                // call the linked bitcode helper with the native b64 predicate.
                if (!fullOp.getMask()) {
                    Value lower = b.create<arith::ConstantIntOp>(loc, 0, 32);
                    mask =
                        b.create<hivm_regbaseintrins::PunpackInstrOp>(loc, opCtx->maskVecType, mask, lower).getResult();
                }
                std::string name =
                    resultType.getElementType().isUnsignedInteger() ? "dup_vec_uint64_t" : "dup_vec_int64_t";
                valueMap[fullOp.getResult()] = createInlineB64Call(
                    b, loc, module, name, {opCtx->vecType, opCtx->maskVecType}, opCtx->vecType, {mapped, mask});
                return success();
            }
            valueMap[fullOp.getResult()] =
                b.create<hivmave::VFBroadcastVectorOp>(loc, opCtx->vecType, mapped, mask, true).getRes();
            return success();
        }
    }

    // A plain scalar source broadcasts across the register via AVE broadcast
    // (mask-capable scalar splat, lowers to vdups.z).
    Value source = lookupOrCloneScalarValue(b, fullOp.getValue(), valueMap);
    if (!source)
        return failure();
    Type tlaElem = resultType.getElementType();
    if (source.getType() != carrierElem) {
        // Unsigned and the byte-carried TLA formats keep their TLA type until
        // here: the AVE scalar ABI is signless, and Bool/hif8/e8m0/fp4x2
        // broadcast an i8 byte.
        auto sourceInt = dyn_cast<IntegerType>(source.getType());
        if (source.getType() == tlaElem && tlaElem.isInteger(1)) {
            source = b.create<arith::ExtUIOp>(loc, carrierElem, source);
        } else if (source.getType() == tlaElem || (sourceInt && sourceInt.isUnsigned())) {
            source = b.create<UnrealizedConversionCastOp>(loc, carrierElem, source).getResult(0);
        }
    }
    if (source.getType() != carrierElem)
        return fullOp.emitError("tla.full scalar type ")
                   << source.getType() << " does not match vector element type " << carrierElem,
               failure();
    if (resultType.getElementType().isInteger(64)) {
        // vdups has no legalizable 64-bit AVE library call; the bitcode helper
        // broadcasts the scalar and takes the native b64 predicate (PGE has no
        // b64 mode, so an omitted mask is packed like tla.create_mask does).
        if (!fullOp.getMask()) {
            Value lower = b.create<arith::ConstantIntOp>(loc, 0, 32);
            mask = b.create<hivm_regbaseintrins::PunpackInstrOp>(loc, opCtx->maskVecType, mask, lower).getResult();
        }
        std::string name = resultType.getElementType().isUnsignedInteger() ? "dup_uint64_t" : "dup_int64_t";
        valueMap[fullOp.getResult()] = createInlineB64Call(
            b, loc, module, name, {carrierElem, opCtx->maskVecType}, opCtx->vecType, {source, mask});
        return success();
    }
    valueMap[fullOp.getResult()] =
        b.create<hivmave::VFBroadcastScalarMaskOp>(loc, opCtx->vecType, source, mask).getRes();
    return success();
}

static LogicalResult lowerArange(
    ::tla::ArangeOp arangeOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = arangeOp.getLoc();
    Value start = lookupOrCloneScalarValue(b, arangeOp.getStart(), valueMap);
    if (!start)
        return failure();
    // Width comes from the result VectorSSA element type.
    auto resultType = dyn_cast<::tla::VectorSSAType>(arangeOp.getResult().getType());
    if (!resultType)
        return arangeOp.emitError("expected !tla.vector result type"), failure();
    auto opCtx = deriveVecCtxForElement(resultType.getElementType());
    if (failed(opCtx))
        return arangeOp.emitError("unsupported tla.arange result element type"), failure();
    auto vciType = hivmave::VCIType::INCREASE;
    if (arangeOp.getOrder() == "decrease")
        vciType = hivmave::VCIType::DECREASE;
    else if (arangeOp.getOrder() != "increase")
        return arangeOp.emitError("unsupported tla.arange order: ") << arangeOp.getOrder(), failure();
    auto vciAttr = hivmave::VCITypeAttr::get(b.getContext(), vciType);
    Type resultElem = opCtx->elementType;
    if (isa<FloatType>(resultElem)) {
        // The float VCI form cannot take a non-constant start during device
        // compilation ("Unsupported Inst must be hoisted"), so generate the
        // lanes from zero and add the (constant) base in the float domain. This
        // also keeps the sequence out of the narrower integer helper type
        // (f16 -> i16, f32 -> i32) that would wrap the sign.
        Type intElem = IntegerType::get(b.getContext(), resultElem.getIntOrFloatBitWidth());
        auto intCtx = deriveVecCtxForElement(intElem);
        if (failed(intCtx))
            return arangeOp.emitError("unsupported tla.arange helper element type ") << intElem, failure();
        if (start.getType() != intElem)
            return arangeOp.emitError("tla.arange start type ")
                       << start.getType() << " does not match integer helper type " << intElem,
                   failure();
        Value zeroStart = b.create<arith::ConstantIntOp>(loc, /*value=*/0, intElem.getIntOrFloatBitWidth());
        Value intVec = b.create<hivmave::VFVCIOp>(loc, intCtx->vecType, zeroStart, vciAttr).getRes();
        Value intMask = allTrueMaskFor(fullMasks, loc, intCtx->vecType, arangeOp.getResult().getType());
        // CastParams defaults: RegSlot.ZERO / SatMode.NOSAT / RoundMode.CAST_NONE.
        // The integral -> float conversion is lossless, so the rounding mode is
        // not consumed on this path.
        int32_t trait[3] = {0, 2, 4};
        auto lanes = createVectorCastResult(
            b, fullMasks, loc, intCtx->vecType, opCtx->vecType, intElem, resultElem, trait, intVec, intMask,
            /*hasUserMask=*/false);
        if (failed(lanes))
            return arangeOp.emitError("failed to convert tla.arange lanes to ") << resultElem, failure();
        Value mask = allTrueMaskFor(fullMasks, loc, opCtx->vecType, arangeOp.getResult().getType());
        Value base = b.create<arith::SIToFPOp>(loc, resultElem, start).getResult();
        valueMap[arangeOp.getResult()] =
            b.create<hivmave::VFAddsOp>(loc, opCtx->vecType, *lanes, base, mask).getResult();
        return success();
    }
    if (start.getType() != resultElem)
        return arangeOp.emitError("tla.arange start type ")
                   << start.getType() << " does not match vector element type " << resultElem,
               failure();
    valueMap[arangeOp.getResult()] = b.create<hivmave::VFVCIOp>(loc, opCtx->vecType, start, vciAttr).getRes();
    return success();
}

static bool isShift(VectorBinaryKind kind)
{
    return kind == VectorBinaryKind::ShiftLeft || kind == VectorBinaryKind::ShiftRight;
}

static Value lowerShift64(
    OpBuilder& b, Location loc, ModuleOp module, const VectorOpInfo& info, Value lhs, Value amount, Value mask)
{
    const bool isUnsigned = isUnsignedInt(cast<::tla::VectorSSAType>(info.operands.lhs.getType()).getElementType());
    std::string name = (Twine(info.mnemonic) + (isUnsigned ? "_uint64_t" : "_int64_t")).str();
    VectorType pregType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregType);
    auto callee = module.lookupSymbol<func::FuncOp>(name);
    if (!callee) {
        OpBuilder moduleBuilder(module.getBodyRegion());
        auto type = FunctionType::get(b.getContext(), {lhs.getType(), amount.getType(), pregType}, {lhs.getType()});
        callee = moduleBuilder.create<func::FuncOp>(loc, name, type);
        annotateAivLibraryCall(callee);
    }
    return b.create<func::CallOp>(loc, callee, ValueRange{lhs, amount, preg}).getResult(0);
}

static LogicalResult lowerVectorBinary(
    Operation& op, const VectorOpInfo* info, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
    if (op.getNumResults() != 1)
        return failure();
    TlaBinaryOperands operands = info->operands;
    if (auto tlaVec = dyn_cast<::tla::VectorSSAType>(operands.lhs.getType());
        tlaVec && isBoolVectorElement(tlaVec.getElementType()))
        return op.emitError(
                   "Bool VectorSSA is move-only (load/store); "
                   "arithmetic is not supported"),
               failure();
    Value lhs = valueMap.lookup(operands.lhs);
    if (!lhs)
        return failure();
    Value rhs = valueMap.lookup(operands.rhs);
    if (!rhs)
        return failure();
    // Derive the vector width from the operands: a cast may have produced a
    // vector of a different lane width than the enclosing region's element type.
    auto opVecType = dyn_cast<VectorType>(lhs.getType());
    if (!opVecType)
        return failure();
    Value mask;
    if (operands.mask) {
        mask = valueMap.lookup(operands.mask);
        if (!mask)
            return failure();
    } else {
        mask = allTrueMaskFor(fullMasks, loc, opVecType, operands.lhs.getType());
    }
    // createVectorBinaryResult peels unsigned from the tla lhs type.
    Type elementType = tlaElementOr(operands.lhs.getType(), opVecType.getElementType());
    Value result;
    if (info->kind == VectorBinaryKind::Max && elementType.isSignlessInteger(64))
        result = createInt64Max(b, loc, module, opVecType, lhs, rhs, mask, static_cast<bool>(operands.mask));
    else if (isShift(info->kind) && opVecType.getElementType().isInteger(64))
        result = lowerShift64(b, loc, module, *info, lhs, rhs, mask);
    else
        result = createVectorBinaryResult(
            b, loc, info->kind, operands.lhs.getType(), opVecType.getElementType(), opVecType, lhs, rhs, mask);
    if (!result)
        return failure();
    valueMap[op.getResult(0)] = result;
    return success();
}

static LogicalResult lowerVectorScalarBinary(
    Operation& op, const VectorOpInfo* info, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
    if (op.getNumResults() != 1)
        return failure();
    TlaBinaryOperands operands = info->operands;
    if (auto tlaVec = dyn_cast<::tla::VectorSSAType>(operands.lhs.getType());
        tlaVec && isBoolVectorElement(tlaVec.getElementType()))
        return op.emitError(
                   "Bool VectorSSA is move-only (load/store); "
                   "arithmetic is not supported"),
               failure();
    Value lhs = valueMap.lookup(operands.lhs);
    if (!lhs)
        return failure();
    // Derive ctx from tla element so unsigned max/min still select VMaxsUI.
    auto lhsTy = dyn_cast<VectorType>(lhs.getType());
    if (!lhsTy)
        return failure();
    auto opCtx = deriveVecCtxForElement(tlaElementOr(operands.lhs.getType(), lhsTy.getElementType()));
    if (failed(opCtx))
        return failure();
    auto scalarOr = materializeVectorScalarValue(b, *info, valueMap, *opCtx);
    if (failed(scalarOr))
        return failure();
    Value mask;
    if (operands.mask) {
        mask = valueMap.lookup(operands.mask);
        if (!mask)
            return failure();
    } else {
        mask = fullMasks.get(loc, opCtx->lanes);
    }
    if (isShift(info->kind) && opCtx->elementType.isInteger(64)) {
        valueMap[op.getResult(0)] = lowerShift64(b, loc, module, *info, lhs, *scalarOr, mask);
        return success();
    }
    auto result = createVectorScalarBinaryResult(b, loc, module, *info, *opCtx, lhs, *scalarOr, mask);
    if (failed(result))
        return failure();
    valueMap[op.getResult(0)] = *result;
    return success();
}

/// Lower squeeze to vsqz, selecting the stub that records the byte count when requested.
static LogicalResult lowerSqueeze(
    ::tla::SqueezeOp squeezeOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = squeezeOp.getLoc();
    Value src = valueMap.lookup(squeezeOp.getSrc());
    Value mask = valueMap.lookup(squeezeOp.getMask());
    if (!src || !mask)
        return failure();
    auto srcTy = dyn_cast<VectorType>(src.getType());
    if (!srcTy)
        return failure();
    auto opCtx = deriveVecCtxForElement(srcTy.getElementType());
    if (failed(opCtx))
        return failure();
    // ``srcTy`` is the AVE (signless) vector type; recover the original tla
    // element type so unsigned squeeze picks the uint*_t stub.
    Type tlaElem = tlaElementOr(squeezeOp.getSrc().getType(), srcTy.getElementType());
    std::string calleeName = getSqueezeLibraryCallName(tlaElem, squeezeOp.getStoreReg());
    if (calleeName.empty())
        return squeezeOp.emitError("unsupported element type for tla.squeeze: ") << tlaElem, failure();
    VectorType pregVecType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregVecType);
    auto callee = getOrCreateSqueezeLibraryCall(module, loc, opCtx->vecType, pregVecType, calleeName);
    Value result = b.create<func::CallOp>(loc, callee, ValueRange{src, preg}).getResult(0);
    valueMap[squeezeOp.getResult()] = result;
    return success();
}

/// tla.unsqueeze: exclusive mask prefix sum via linked bitcode (vusqz).
static LogicalResult lowerUnsqueeze(
    ::tla::UnsqueezeOp unsqueezeOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    Location loc = unsqueezeOp.getLoc();
    Value mask = valueMap.lookup(unsqueezeOp.getMask());
    if (!mask)
        return failure();
    // No vector operand to key off: the register width comes from the result
    // VectorSSA element type.
    auto resultType = unsqueezeOp.getResult().getType();
    auto opCtx = deriveVecCtxForElement(resultType.getElementType());
    if (failed(opCtx))
        return unsqueezeOp.emitError("unsupported tla.unsqueeze result element type"), failure();
    std::string calleeName = getUnsqueezeLibraryCallName(resultType.getElementType());
    if (calleeName.empty())
        return unsqueezeOp.emitError("unsupported element type for tla.unsqueeze: ") << resultType.getElementType(),
               failure();
    VectorType pregVecType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregVecType);
    auto callee = getOrCreateUnsqueezeLibraryCall(module, loc, opCtx->vecType, pregVecType, calleeName);
    valueMap[unsqueezeOp.getResult()] = b.create<func::CallOp>(loc, callee, ValueRange{preg}).getResult(0);
    return success();
}

static LogicalResult lowerReduce(
    ::tla::ReduceOp reduceOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *reduceOp.getOperation();
    Location loc = op.getLoc();
    if (op.getNumResults() != 1)
        return failure();
    Value operand = valueMap.lookup(reduceOp->getOperand(0));
    if (!operand)
        return failure();
    // AVE operand is signless; CombiningKind UMAX/UMIN needs the tla element type.
    auto operandTy = dyn_cast<VectorType>(operand.getType());
    if (!operandTy)
        return failure();
    Type tlaElem = tlaElementOr(reduceOp->getOperand(0).getType(), operandTy.getElementType());
    auto opCtx = deriveVecCtxForElement(tlaElem);
    if (failed(opCtx))
        return failure();
    Value mask;
    if (reduceOp.getMask()) {
        mask = valueMap.lookup(reduceOp.getMask());
        if (!mask)
            return failure();
    }
    auto result = createVectorReductionResult(b, loc, reduceOp, opCtx->elementType, opCtx->vecType, operand, mask);
    if (failed(result))
        return failure();
    valueMap[op.getResult(0)] = *result;
    return success();
}

// tla.reduce_intra_block: per-32B-block group reduction
// lowered directly to the Vcg regbase instructions. The container value is
// already a full 256-byte register vector, so no pad is required: the
// instruction treats inactive (mask=0) elements as identity values (0 for ADD,
// -INF/+INF for MAX/MIN) and packs each block's result contiguously into the
// LSBs of the destination. The user mask is passed through unchanged.
static LogicalResult lowerReduceIntraBlock(
    ::tla::ReduceIntraBlockOp rdbOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *rdbOp.getOperation();
    Location loc = op.getLoc();
    Value operand = valueMap.lookup(rdbOp.getOperand());
    if (!operand)
        return failure();
    auto operandTy = dyn_cast<VectorType>(operand.getType());
    if (!operandTy)
        return failure();
    // The Vcg S/U variant selection follows the tla element type; the AVE
    // container element is signless.
    Type tlaElem = tlaElementOr(rdbOp->getOperand(0).getType(), operandTy.getElementType());
    auto opCtx = deriveVecCtxForElement(tlaElem);
    if (failed(opCtx))
        return failure();
    if (!rdbOp.getMask())
        return rdbOp.emitError("tla.reduce_intra_block requires an explicit mask"), failure();
    Value mask = valueMap.lookup(rdbOp.getMask());
    if (!mask)
        return failure();
    VectorType pregVecType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregVecType);

    auto kindAttr = rdbOp->getAttrOfType<StringAttr>("kind");
    if (!kindAttr)
        return rdbOp.emitError("tla.reduce_intra_block requires string kind attribute"), failure();
    StringRef kind = kindAttr.getValue();
    // Defensive mirror of the ReduceIntraBlockOp verifier constraints: the
    // Vcg dst is presented as the op's TLA result type, so a hand-written op
    // whose result type violates the widening/mirror rules would lower into
    // mismatched IR. Reject it here instead of ignoring the result type.
    auto tlaOperandTy = dyn_cast<::tla::VectorSSAType>(rdbOp.getOperand().getType());
    auto tlaResultTy = dyn_cast<::tla::VectorSSAType>(rdbOp.getResult().getType());
    if (!tlaResultTy)
        return rdbOp.emitError("tla.reduce_intra_block result must be !tla.vector"), failure();
    auto opElem = dyn_cast<IntegerType>(tlaElem);
    auto resElem = dyn_cast<IntegerType>(tlaResultTy.getElementType());
    bool widenAdd = (kind == "add") && opElem && (opElem.getWidth() == 16);
    if (widenAdd) {
        // The widening keeps the operand's sign category: i16 -> i32, ui16 -> ui32.
        if (!resElem || (resElem.getWidth() != 32) || (resElem.getSignedness() != opElem.getSignedness()))
            return rdbOp.emitError() << "add on i16/ui16 widens the result to the matching 32-bit type "
                                     << "(i16->i32, ui16->ui32), got " << tlaResultTy.getElementType(),
                   failure();
        // Lane count halves per 32-bit result lane, rounding up so a tail's
        // last valid element keeps its widened lane.
        if ((tlaOperandTy.getValidLanes() != ShapedType::kDynamic) &&
            (tlaResultTy.getValidLanes() != ((tlaOperandTy.getValidLanes() + 1) / 2)))
            return rdbOp.emitError() << "add on i16/ui16 keeps the 256B container, expected "
                                     << ((tlaOperandTy.getValidLanes() + 1) / 2) << " result lane(s), got "
                                     << tlaResultTy.getValidLanes(),
                   failure();
    } else if (
        (tlaResultTy.getElementType() != tlaElem) ||
        (tlaOperandTy && (tlaResultTy.getValidLanes() != tlaOperandTy.getValidLanes()))) {
        return rdbOp.emitError("result type must match operand type (packed mirror)"), failure();
    }
    bool isUnsigned = false;
    if (auto intType = dyn_cast<IntegerType>(tlaElem))
        isUnsigned = intType.isUnsigned();

    Value result;
    if (kind == "add") {
        // ADD on 16-bit sources accumulates into a 32-bit destination (the same
        // widening behavior as vcadd): each block sum is written as a 32-bit
        // value into the LSBs. The LLVM-side intrinsic signature requires the
        // source to have the destination's type, so the Vcgadd is emitted on the
        // 16-bit container and the result register is re-presented as a 64-lane
        // i32 vector with a zero-cost bitcast. No Vcvtii cast back: block sums
        // beyond the source type's range stay exact in the i32/ui32 result, and
        // truncation is the user's decision.
        result = isUnsigned ?
                     b.create<hivm_regbaseintrins::VcgaddUXInstrOp>(loc, opCtx->vecType, operand, preg).getRes() :
                     b.create<hivm_regbaseintrins::VcgaddSXInstrOp>(loc, opCtx->vecType, operand, preg).getRes();
        if (isa<IntegerType>(tlaElem) && (dyn_cast<IntegerType>(tlaElem).getWidth() == 16)) {
            VectorType wideType = VectorType::get({64}, b.getI32Type());
            result = b.create<mlir::vector::BitCastOp>(loc, wideType, result).getResult();
        }
    } else if (kind == "max") {
        result = isUnsigned ?
                     b.create<hivm_regbaseintrins::VcgmaxUInstrOp>(loc, opCtx->vecType, operand, preg).getRes() :
                     b.create<hivm_regbaseintrins::VcgmaxSInstrOp>(loc, opCtx->vecType, operand, preg).getRes();
    } else if (kind == "min") {
        result = isUnsigned ?
                     b.create<hivm_regbaseintrins::VcgminUInstrOp>(loc, opCtx->vecType, operand, preg).getRes() :
                     b.create<hivm_regbaseintrins::VcgminSInstrOp>(loc, opCtx->vecType, operand, preg).getRes();
    } else {
        return rdbOp.emitError() << "tla.reduce_intra_block supports only add, max, and min reductions, got \"" << kind
                                 << "\"",
               failure();
    }
    valueMap[op.getResult(0)] = result;
    return success();
}

// tla.pair_wise: adjacent-lane pairwise sum
// lowered straight to the Vcpadd regbase instruction. The destination keeps the
// full 256-byte container: the pairwise sums occupy its low half and the upper
// half is zero-filled by hardware. There is no AVE pair-wise op, so the
// instruction is emitted directly, like the Vcg group reductions above.
static LogicalResult lowerPairWise(
    ::tla::PairWiseOp pairOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = pairOp.getLoc();
    if (pairOp.getKind() != "add")
        return pairOp.emitError() << "tla.pair_wise supports only add, got \"" << pairOp.getKind() << "\"", failure();
    Value operand = valueMap.lookup(pairOp.getOperand());
    if (!operand)
        return failure();
    auto operandTy = dyn_cast<VectorType>(operand.getType());
    if (!operandTy)
        return failure();
    Type tlaElem = tlaElementOr(pairOp.getOperand().getType(), operandTy.getElementType());
    auto opCtx = deriveVecCtxForElement(tlaElem);
    if (failed(opCtx))
        return failure();
    if (!pairOp.getMask())
        return pairOp.emitError("tla.pair_wise requires an explicit mask"), failure();
    Value mask = valueMap.lookup(pairOp.getMask());
    if (!mask)
        return failure();
    VectorType pregVecType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregVecType);
    Value result = b.create<hivm_regbaseintrins::VcpaddInstrOp>(loc, opCtx->vecType, operand, preg).getRes();
    valueMap[pairOp.getResult()] = result;
    return success();
}

static LogicalResult lowerPack(::tla::PackOp packOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    Location loc = packOp.getLoc();
    Value src = valueMap.lookup(packOp.getSrc());
    if (!src)
        return failure();

    if (isa<::tla::VectorSSAType>(packOp.getSrc().getType())) {
        auto srcType = cast<::tla::VectorSSAType>(packOp.getSrc().getType());
        auto dstType = cast<::tla::VectorSSAType>(packOp.getResult().getType());
        auto srcCtx = deriveVecCtxForElement(srcType.getElementType());
        auto dstCtx = deriveVecCtxForElement(dstType.getElementType());
        if (failed(srcCtx) || failed(dstCtx))
            return failure();
        auto name = getVectorPackLibraryCallName(srcType.getElementType(), dstType.getElementType(), packOp.getPart());
        auto callee = getOrCreateVectorPackUnpackLibraryCall(module, loc, srcCtx->vecType, dstCtx->vecType, name);
        valueMap[packOp.getResult()] = b.create<func::CallOp>(loc, callee, ValueRange{src}).getResult(0);
        return success();
    }
    if (isa<::tla::MaskSSAType>(packOp.getSrc().getType())) {
        int64_t partValue = packOp.getPart() == ::RegPart::high ? 1 : 0;
        Value part = b.create<arith::ConstantIntOp>(loc, partValue, 32);
        VectorType pregType = fullPregVecType(b.getContext());
        Value pregSrc = castMaskToPregType(b, loc, src, pregType);
        auto packed = b.create<hivm_regbaseintrins::PpackInstrOp>(loc, pregType, pregSrc, part);
        valueMap[packOp.getResult()] = packed->getResult(0);
        return success();
    }
    return failure();
}

static LogicalResult lowerUnpack(
    ::tla::UnpackOp unpackOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    Value src = valueMap.lookup(unpackOp.getSrc());
    if (!src)
        return failure();
    // Preserve TLA signedness in the symbol; CANN selects sign/zero extension.
    auto srcType = cast<::tla::VectorSSAType>(unpackOp.getSrc().getType());
    auto dstType = cast<::tla::VectorSSAType>(unpackOp.getResult().getType());
    auto srcCtx = deriveVecCtxForElement(srcType.getElementType());
    auto dstCtx = deriveVecCtxForElement(dstType.getElementType());
    if (failed(srcCtx) || failed(dstCtx))
        return failure();
    auto name = getVectorUnpackLibraryCallName(srcType.getElementType(), dstType.getElementType(), unpackOp.getPart());
    auto callee =
        getOrCreateVectorPackUnpackLibraryCall(module, unpackOp.getLoc(), srcCtx->vecType, dstCtx->vecType, name);
    valueMap[unpackOp.getResult()] = b.create<func::CallOp>(unpackOp.getLoc(), callee, ValueRange{src}).getResult(0);
    return success();
}

static LogicalResult lowerMull(::tla::MullOp op, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    Value lhs = valueMap.lookup(op.getLhs());
    Value rhs = valueMap.lookup(op.getRhs());
    Value mask = valueMap.lookup(op.getMask());
    if (!lhs || !rhs || !mask)
        return failure();
    const bool isUnsigned = isUnsignedInt(op.getLhs().getType().getElementType());
    auto cast = isUnsigned ? hivm::TypeFn::cast_unsigned : hivm::TypeFn::cast_signed;
    auto attr = b.getNamedAttr("cast", hivm::TypeFnAttr::get(b.getContext(), cast));
    mask = castMaskToPregType(b, op.getLoc(), mask, fullPregVecType(b.getContext()));
    auto wide = b.create<hivmave::VFVMULLOp>(
        op.getLoc(), TypeRange{lhs.getType(), lhs.getType()}, ValueRange{lhs, rhs, mask}, ArrayRef{attr});
    valueMap[op.getLow()] = wide->getResult(0);
    valueMap[op.getHigh()] = wide->getResult(1);
    return success();
}

template <typename CarryOp>
static LogicalResult lowerB32CarryOp(
    CarryOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap, AutoFullPredicateCache& fullMasks,
    StringRef signedSymbol, StringRef unsignedSymbol)
{
    Value lhs = valueMap.lookup(op.getLhs());
    Value rhs = valueMap.lookup(op.getRhs());
    Value carry = valueMap.lookup(op.getCarryIn());
    Value mask = valueMap.lookup(op.getMask());
    if (!lhs || !rhs || !carry || !mask)
        return failure();
    Location loc = op.getLoc();
    Type vec = lhs.getType();
    auto preg = fullPregVecType(b.getContext());
    carry = castMaskToPregType(b, loc, carry, preg);
    mask = castMaskToPregType(b, loc, mask, preg);
    auto declareRawCall = [&](StringRef name, FunctionType fnType) {
        if (auto fn = module.lookupSymbol<func::FuncOp>(name))
            return fn;
        OpBuilder moduleBuilder(module.getBodyRegion());
        auto fn = moduleBuilder.create<func::FuncOp>(loc, name, fnType);
        fn.setPrivate();
        fn->setAttr(
            hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(b.getContext(), hivm::TFuncCoreType::AIV));
        return fn;
    };
    StringRef name = isUnsignedInt(op.getLhs().getType().getElementType()) ? unsignedSymbol : signedSymbol;
    Type ubPtr = getUbPtrType(b.getContext());
    // This nonescaping BC out parameter is promoted after the always-inline
    // helpers are linked. A kernel UB tensor would leave a real memory round trip.
    auto scratch = fullMasks.createRetptrSlot(b, loc, ubPtr, preg, /*alignment=*/32);
    if (failed(scratch))
        return op.emitError("could not reserve the carry predicate out parameter"), failure();
    auto helper = declareRawCall(name, b.getFunctionType({vec, vec, preg, preg, ubPtr}, {vec}));
    auto result = b.create<func::CallOp>(loc, helper, ValueRange{lhs, rhs, carry, mask, *scratch});
    valueMap[op->getResult(0)] = result.getResult(0);
    if (!op.getCarryOut().use_empty()) {
        auto read = declareRawCall("tla_b32_read_carry", b.getFunctionType({ubPtr}, {preg}));
        valueMap[op.getCarryOut()] = b.create<func::CallOp>(loc, read, ValueRange{*scratch}).getResult(0);
    }
    return success();
}

static LogicalResult lowerInterleave(
    ::tla::InterleaveOp interleaveOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *interleaveOp.getOperation();
    Location loc = op.getLoc();
    if (op.getNumResults() != 2)
        return failure();

    Value src0 = valueMap.lookup(interleaveOp.getSrc0());
    Value src1 = valueMap.lookup(interleaveOp.getSrc1());
    if (!src0 || !src1)
        return failure();

    if (isa<::tla::MaskSSAType>(interleaveOp.getSrc0().getType())) {
        auto maskType = cast<::tla::MaskSSAType>(interleaveOp.getSrc0().getType());
        auto pregType = fullPregVecType(b.getContext());
        src0 = castMaskToPregType(b, loc, src0, pregType);
        src1 = castMaskToPregType(b, loc, src1, pregType);
        auto pairType = LLVM::LLVMStructType::getLiteral(b.getContext(), {pregType, pregType});
        Operation* intrinsic = nullptr;
        switch (maskType.getPhysicalLanes()) {
            case 256:
                intrinsic = b.create<hivm_regbaseintrins::PintlvB8InstrOp>(loc, pairType, src0, src1);
                break;
            case 128:
                intrinsic = b.create<hivm_regbaseintrins::PintlvB16InstrOp>(loc, pairType, src0, src1);
                break;
            case 64:
                intrinsic = b.create<hivm_regbaseintrins::PintlvB32InstrOp>(loc, pairType, src0, src1);
                break;
            default:
                return interleaveOp.emitError("supports only b8, b16, and b32 predicate interleave"), failure();
        }
        Value aggregate = intrinsic->getResult(0);
        valueMap[interleaveOp.getDst0()] = b.create<LLVM::ExtractValueOp>(loc, aggregate, 0);
        valueMap[interleaveOp.getDst1()] = b.create<LLVM::ExtractValueOp>(loc, aggregate, 1);
        return success();
    }

    auto src0Type = dyn_cast<VectorType>(src0.getType());
    auto src1Type = dyn_cast<VectorType>(src1.getType());
    if (!src0Type || !src1Type || src0Type != src1Type)
        return failure();

    auto aveOp = b.create<hivmave::VFInterleaveOp>(loc, TypeRange{src0Type, src1Type}, ValueRange{src0, src1});

    valueMap[op.getResult(0)] = aveOp->getResult(0);
    valueMap[op.getResult(1)] = aveOp->getResult(1);
    return success();
}

static LogicalResult lowerDeInterleave(
    ::tla::DeInterleaveOp deInterleaveOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Operation& op = *deInterleaveOp.getOperation();
    Location loc = op.getLoc();
    if (op.getNumResults() != 2)
        return failure();

    Value src0 = valueMap.lookup(deInterleaveOp.getSrc0());
    Value src1 = valueMap.lookup(deInterleaveOp.getSrc1());
    if (!src0 || !src1)
        return failure();

    if (isa<::tla::MaskSSAType>(deInterleaveOp.getSrc0().getType())) {
        // SCF carriers are vector<256xi1>; choose PDintlv from the original
        // MaskSSA type rather than the converted carrier.
        auto maskType = cast<::tla::MaskSSAType>(deInterleaveOp.getSrc0().getType());
        auto pregType = fullPregVecType(b.getContext());
        src0 = castMaskToPregType(b, loc, src0, pregType);
        src1 = castMaskToPregType(b, loc, src1, pregType);
        auto pairType = LLVM::LLVMStructType::getLiteral(b.getContext(), {pregType, pregType});
        Operation* intrinsic = nullptr;
        switch (maskType.getPhysicalLanes()) {
            case 256:
                intrinsic = b.create<hivm_regbaseintrins::PDintlvB8InstrOp>(loc, pairType, src0, src1);
                break;
            case 128:
                intrinsic = b.create<hivm_regbaseintrins::PDintlvB16InstrOp>(loc, pairType, src0, src1);
                break;
            case 64:
                intrinsic = b.create<hivm_regbaseintrins::PDintlvB32InstrOp>(loc, pairType, src0, src1);
                break;
            default:
                return deInterleaveOp.emitError("supports only b8, b16, and b32 predicate deinterleave"), failure();
        }
        Value aggregate = intrinsic->getResult(0);
        valueMap[deInterleaveOp.getDst0()] = b.create<LLVM::ExtractValueOp>(loc, aggregate, 0);
        valueMap[deInterleaveOp.getDst1()] = b.create<LLVM::ExtractValueOp>(loc, aggregate, 1);
    } else {
        auto src0Type = dyn_cast<VectorType>(src0.getType());
        auto src1Type = dyn_cast<VectorType>(src1.getType());
        if (!src0Type || !src1Type || src0Type != src1Type)
            return failure();
        auto aveOp = b.create<hivmave::VFDeInterleaveOp>(loc, TypeRange{src0Type, src1Type}, ValueRange{src0, src1});
        valueMap[deInterleaveOp.getDst0()] = aveOp->getResult(0);
        valueMap[deInterleaveOp.getDst1()] = aveOp->getResult(1);
    }
    return success();
}

/// tla.gather: per-lane indexed load from a UB tile.
/// x (tile_view → rank-1 memref) → VFGatherOp base
/// y (loaded index vector)        → index_vec
/// mask (optional)                → mask (all-true if absent)
static LogicalResult lowerGather(
    ::tla::GatherOp gatherOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = gatherOp.getLoc();
    Value src = valueMap.lookup(gatherOp.getX());
    Value indexVec = valueMap.lookup(gatherOp.getY());
    if (!src || !indexVec)
        return failure();

    // Reg→Reg permute via linked bitcode (vselr). Not HIVMAVE vgather / vsel.
    if (isa<::tla::VectorSSAType>(gatherOp.getX().getType())) {
        auto srcTy = dyn_cast<VectorType>(src.getType());
        if (!srcTy)
            return failure();
        auto opCtx = deriveVecCtxForElement(srcTy.getElementType());
        if (failed(opCtx))
            return failure();
        std::string calleeName = getGatherRegLibraryCallName(srcTy.getElementType());
        if (calleeName.empty())
            return gatherOp.emitError("unsupported element type for Reg→Reg gather: ") << srcTy.getElementType(),
                   failure();

        unsigned indexBits = srcTy.getElementType().getIntOrFloatBitWidth();
        VectorType indexUType = VectorType::get(
            opCtx->vecType.getShape(), IntegerType::get(b.getContext(), indexBits, IntegerType::Unsigned));
        if (indexVec.getType() != indexUType)
            indexVec = b.create<UnrealizedConversionCastOp>(loc, indexUType, indexVec).getResult(0);

        auto callee = getOrCreateGatherRegLibraryCall(module, loc, opCtx->vecType, indexUType, calleeName);
        valueMap[gatherOp.getResult()] = b.create<func::CallOp>(loc, callee, ValueRange{src, indexVec}).getResult(0);
        return success();
    }

    // UB→Reg: per-lane indexed load from a UB tile.
    auto baseType = dyn_cast<MemRefType>(src.getType());
    if (!baseType || baseType.getRank() != 1)
        return failure();
    // Bool is move-only load/store; gather has no i8-carrier path and fails
    // LLVM translation for i1 AVE vgather.
    if (isBoolVectorElement(baseType.getElementType()) ||
        (isa<::tla::VectorSSAType>(gatherOp.getResult().getType()) &&
         isBoolVectorElement(cast<::tla::VectorSSAType>(gatherOp.getResult().getType()).getElementType())))
        return gatherOp.emitError(
                   "Bool VectorSSA is move-only (load/store); "
                   "gather is not supported"),
               failure();
    auto elemByteWidth = getElementByteWidth(baseType.getElementType());
    if (failed(elemByteWidth))
        return failure();
    int64_t numElems = 256 / *elemByteWidth;
    auto resultVecType = VectorType::get(numElems, baseType.getElementType());
    Value mask;
    if (gatherOp.getMask()) {
        mask = valueMap.lookup(gatherOp.getMask());
        if (!mask)
            return failure();
    } else {
        // Predicate follows the gathered vector semantic lane count.
        mask = fullMasks.get(loc, numElems);
    }
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    valueMap[gatherOp.getResult()] =
        b.create<hivmave::VFGatherOp>(loc, resultVecType, src, ValueRange(zero), indexVec, mask);
    return success();
}

// HIVMAVE's VFScatterOp has no backend implementation. Preserve the native
// Ascend C type/index/predicate contract through a typed, linked AIV helper.
static LogicalResult lowerScatter(
    ::tla::ScatterOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
    Value dst = valueMap.lookup(op.getDst());
    Value src = valueMap.lookup(op.getSrc());
    Value indices = valueMap.lookup(op.getIndices());
    if (!dst || !src || !indices)
        return failure();
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    auto srcType = dyn_cast<VectorType>(src.getType());
    auto indexType = dyn_cast<VectorType>(indices.getType());
    if (!dstType || dstType.getRank() != 1 || !srcType || !indexType)
        return failure();
    Type element = op.getSrc().getType().getElementType();
    // Defense in depth: verifier already rejects i1; do not use bit-width math
    // on Bool (would inflate 2048/1 and silently treat as i8 carrier).
    if (isBoolVectorElement(element) || isBoolVectorElement(dstType.getElementType()))
        return op.emitError("Bool VectorSSA is move-only (load/store); scatter is not supported"), failure();
    auto nativeSrcType = VectorType::get({2048 / element.getIntOrFloatBitWidth()}, element);
    if (srcType != nativeSrcType) {
        src = b.create<UnrealizedConversionCastOp>(loc, nativeSrcType, src).getResult(0);
        srcType = nativeSrcType;
    }
    Type indexElement = op.getIndices().getType().getElementType();
    auto nativeIndexType = VectorType::get({2048 / indexElement.getIntOrFloatBitWidth()}, indexElement);
    if (indexType != nativeIndexType) {
        indices = b.create<UnrealizedConversionCastOp>(loc, nativeIndexType, indices).getResult(0);
        indexType = nativeIndexType;
    }
    // One declaration per native type pair, independent of tile shape/offset.
    auto layout = StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, {ShapedType::kDynamic});
    auto abiDstType =
        MemRefType::get({ShapedType::kDynamic}, dstType.getElementType(), layout, dstType.getMemorySpace());
    if (dstType != abiDstType) {
        dst = b.create<mlir::memref::CastOp>(loc, abiDstType, dst);
        dstType = abiDstType;
    }
    std::string dataName;
    if (element.isF32())
        dataName = "float";
    else if (element.isF16())
        dataName = "half";
    else if (element.isBF16())
        dataName = "bfloat16_t";
    else if (auto integer = dyn_cast<IntegerType>(element))
        dataName = std::string(integer.isUnsigned() ? "uint" : "int") + std::to_string(integer.getWidth()) + "_t";
    else
        return op.emitOpError("unsupported scatter element type");
    unsigned indexBits = indexType.getElementType().getIntOrFloatBitWidth();
    std::string name = "scatter_" + dataName + "_u" + std::to_string(indexBits);
    auto pregType = fullPregVecType(b.getContext());
    int64_t lanes = 2048 / std::max(element.getIntOrFloatBitWidth(), 16u);
    Value mask = op.getMask() ? valueMap.lookup(op.getMask()) : fullMasks.get(loc, lanes);
    if (!mask)
        return failure();
    mask = castMaskToPregType(b, loc, mask, pregType);
    auto callee = module.lookupSymbol<func::FuncOp>(name);
    if (!callee) {
        OpBuilder moduleBuilder(module.getBodyRegion());
        auto fnType = FunctionType::get(b.getContext(), {dstType, srcType, indexType, pregType}, {});
        callee = moduleBuilder.create<func::FuncOp>(loc, name, fnType);
        annotateAivLibraryCall(callee);
    }
    b.create<func::CallOp>(loc, callee, ValueRange{dst, src, indices, mask});
    return success();
}

// tla.log mode=subnormal: vln treats subnormal inputs as zero, so the only
// subnormal-preserving sequence is the software log with FTZ disabled. Call
// the linked AIV bitcode helper instead of emitting ave.hir.vln.
static LogicalResult lowerVectorLogSubnormal(
    ::tla::LogOp logOp, Value operand, VectorType operandVecType, Value mask, OpBuilder& b, ModuleOp module,
    DenseMap<Value, Value>& valueMap)
{
    Location loc = logOp.getLoc();
    Type elementType = operandVecType.getElementType();
    if (!elementType.isF16() && !elementType.isF32())
        return logOp.emitError("tla.log subnormal mode requires f16 or f32, got ") << elementType, failure();

    VectorType pregType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregType);
    std::string calleeName = elementType.isF32() ? "log_subnormal_f32" : "log_subnormal_f16";
    auto callee = getOrCreateInlineAivLibraryCall(module, loc, {operandVecType, pregType}, operandVecType, calleeName);
    valueMap[logOp.getResult()] = b.create<func::CallOp>(loc, callee, ValueRange{operand, preg}).getResult(0);
    return success();
}

static LogicalResult lowerVectorUnary(
    Operation& op, const VectorUnaryInfo* info, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
    if (op.getNumResults() != 1)
        return failure();
    TlaUnaryOperands operands = info->operands;
    Type tlaOperandType = operands.operand.getType();
    if (auto tlaVec = dyn_cast<::tla::VectorSSAType>(tlaOperandType);
        tlaVec && isBoolVectorElement(tlaVec.getElementType()))
        return op.emitError(
                   "Bool VectorSSA is move-only (load/store); "
                   "arithmetic is not supported"),
               failure();
    Value operand = valueMap.lookup(operands.operand);
    if (!operand)
        return failure();
    auto operandVecType = dyn_cast<VectorType>(operand.getType());
    if (!operandVecType)
        return failure();

    // Validate the AVE carrier width. Bool is rejected above via
    // isBoolVectorElement before it can be mistaken for an i8 carrier.
    if (isa<::tla::VectorSSAType>(tlaOperandType)) {
        if (failed(validateVectorUnaryElementType(&op, *info, operandVecType.getElementType())))
            return failure();
    } else if (!isa<::tla::MaskSSAType>(tlaOperandType)) {
        return op.emitError("expected !tla.vector<NxT> or !tla.mask<N> operand");
    }

    Value mask;
    if (operands.mask) {
        mask = valueMap.lookup(operands.mask);
        if (!mask)
            return failure();
    } else {
        mask = allTrueMaskFor(fullMasks, loc, operandVecType, tlaOperandType);
    }
    // tla.log mode=subnormal: the FTZ path keeps the existing ave.hir.vln
    // lowering; only the subnormal-preserving mode needs the bitcode helper.
    if (auto logOp = dyn_cast<::tla::LogOp>(op)) {
        if (auto algo = logOp.getLogAlgoAttr(); algo && algo.getValue() == LogAlgo::subnormal)
            return lowerVectorLogSubnormal(logOp, operand, operandVecType, mask, b, module, valueMap);
    }
    Value result = createVectorUnaryResult(
        b, loc, module, info->kind, tlaOperandType, operandVecType, operand, mask, static_cast<bool>(operands.mask));
    if (!result)
        return failure();
    valueMap[op.getResult(0)] = result;
    return success();
}

/// tla.update_mask: tail mask + remaining count. Lowers to ave.hir.plt,
/// whose mask result drives masked stores and whose second result
/// (true_shape - lanes) is threaded back as the loop-carried tail counter.
/// The op's own dtype attr fixes the lane count (256 bytes / element size)
/// and hence the i1 mask width and the tail decrement.
static LogicalResult lowerUpdateMask(
    ::tla::UpdateMaskOp updateMaskOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = updateMaskOp.getLoc();
    // The true-shape operand may be a vec.func-external index constant (e.g.
    // tla.update_mask(1) building a single-lane mask): such constants are not
    // collected as helper arguments, so clone them inline like other scalar
    // operands instead of a bare valueMap lookup (which would miss them).
    Value trueShape = lookupOrCloneScalarValue(b, updateMaskOp.getTrueShape(), valueMap);
    if (!trueShape)
        return failure();
    auto opCtx = deriveVecCtxForElement(updateMaskOp.getDtype());
    if (failed(opCtx))
        return updateMaskOp.emitError("unsupported tla.update_mask dtype: ") << updateMaskOp.getDtype(), failure();
    auto plt = createPredicatePlt(b, loc, opCtx->maskVecType, opCtx->lanes, trueShape);
    Value mask = plt.getRes();
    if (opCtx->lanes == 32) {
        // Ascend C's single-register b64 predicate occupies every eighth bit.
        // plt.b32 produces every fourth bit; Scatter packs this b64 mask back
        // to b32 while splitting each source element into two 32-bit stores.
        Value lower = b.create<arith::ConstantIntOp>(loc, 0, 32);
        mask = b.create<hivm_regbaseintrins::PunpackInstrOp>(loc, opCtx->maskVecType, mask, lower).getResult();
    }
    valueMap[updateMaskOp.getMask()] = mask;
    // new_true_shape = true_shape - lanes, which is exactly what plt computes.
    // We materialize it with index arithmetic rather than consuming plt's second
    // result: that result is i32 in hardware but typed index, so carrying it
    // through the loop would leave an unfoldable i32<->index unrealized cast.
    Value lanesValue = b.create<arith::ConstantIndexOp>(loc, opCtx->lanes);
    valueMap[updateMaskOp.getNewTrueShape()] = b.create<arith::SubIOp>(loc, trueShape, lanesValue);
    return success();
}

/// tla.generate_mask: expand one data block of a register into a predicate
/// register. The source element width selects the lane count (b16 → mask<128>,
/// b32 → mask<64>) and the block range; lane L of the result is register bit
/// (256/sizeof(T)) * offset + L.
///
/// The backend instruction only selects integer operands, so a float or signed
/// source is first bitcast to the same-width integer vector: b32 →
/// vector<Nxi32>, b16 → vector<Nxi16>. The bit pattern is unchanged, which is
/// exactly what the instruction consumes.
static LogicalResult lowerGenerateMask(
    ::tla::GenerateMaskOp generateMaskOp, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    Location loc = generateMaskOp.getLoc();
    Value src = valueMap.lookup(generateMaskOp.getSrc());
    if (!src)
        return failure();
    auto srcType = dyn_cast<VectorType>(src.getType());
    if (!srcType)
        return generateMaskOp.emitError("expected a lowered vector source"), failure();
    // The TLA element type survives the unsigned→signless AVE mapping, so take
    // the width from it rather than the lowered carrier. Any integer or float
    // element works; only the bit width matters.
    Type tlaElem = tlaElementOr(generateMaskOp.getSrc().getType(), srcType.getElementType());
    if (!tlaElem.isIntOrFloat())
        return generateMaskOp.emitError("unsupported element type for tla.generate_mask: ") << tlaElem, failure();
    int64_t elementBits = tlaElem.getIntOrFloatBitWidth();
    if (elementBits != 16 && elementBits != 32)
        return generateMaskOp.emitError("unsupported element type for tla.generate_mask: ") << tlaElem, failure();

    // The instruction only selects the integer operands; reinterpret the bits.
    auto intSrcType = VectorType::get(srcType.getShape(), b.getIntegerType(elementBits));
    Value intSrc = src;
    if (srcType != intSrcType)
        intSrc = b.create<mlir::vector::BitCastOp>(loc, intSrcType, src).getResult();

    Value part = b.create<arith::ConstantIntOp>(loc, generateMaskOp.getOffset(), /*width=*/32);
    VectorType pregType = fullPregVecType(b.getContext());
    valueMap[generateMaskOp.getResult()] =
        b.create<hivm_regbaseintrins::MovvpInstrOp>(loc, pregType, intSrc, part).getResult();
    return success();
}

static LogicalResult lowerCmp(
    ::tla::CmpOp cmpOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = cmpOp.getLoc();
    Value lhs = valueMap.lookup(cmpOp.getLhs());
    if (!lhs)
        return failure();
    // The compare's operand width fixes both the input vectors and the i1 mask
    // result width. Prefer the tla element type so unsigned compares keep
    // unsigned semantics after aveVectorElementType maps the AVE vector to
    // signless.
    Type tlaElem = lhs.getType();
    if (auto tlaVec = dyn_cast<::tla::VectorSSAType>(cmpOp.getLhs().getType()))
        tlaElem = tlaVec.getElementType();
    else if (auto lhsTy = dyn_cast<VectorType>(lhs.getType()))
        tlaElem = lhsTy.getElementType();
    auto opCtx = deriveVecCtxForElement(tlaElem);
    if (failed(opCtx))
        return failure();
    auto cmpType = mapCmpMode(cmpOp.getMode());
    if (!cmpType)
        return cmpOp.emitError("unknown tla.cmp mode: ") << cmpOp.getMode(), failure();
    // With signless AVE vectors, HIVMCmpOpLowering's isUnsignedInteger path
    // never fires. Remap relational modes to the explicit unsigned CmpType
    // variants so VCmpU* is selected.
    if (auto intTy = dyn_cast<IntegerType>(tlaElem); intTy && intTy.isUnsigned()) {
        switch (*cmpType) {
            case hivmave::CmpType::LT:
                cmpType = hivmave::CmpType::ULT;
                break;
            case hivmave::CmpType::GT:
                cmpType = hivmave::CmpType::UGT;
                break;
            case hivmave::CmpType::LE:
                cmpType = hivmave::CmpType::ULE;
                break;
            case hivmave::CmpType::GE:
                cmpType = hivmave::CmpType::UGE;
                break;
            default:
                break;
        }
    }
    Value mask;
    if (cmpOp.getMask()) {
        mask = valueMap.lookup(cmpOp.getMask());
        if (!mask)
            return failure();
    } else {
        mask = fullMasks.get(loc, opCtx->lanes);
        // PGE has no b64 mode; the omitted-mask carrier is the b32 predicate.
        // Pack its low half into the native b64 predicate like tla.create_mask
        // does before the compare reaches the 64-bit helper.
        if (tlaElem.isInteger(64)) {
            Value lower = b.create<arith::ConstantIntOp>(loc, 0, 32);
            mask = b.create<hivm_regbaseintrins::PunpackInstrOp>(loc, opCtx->maskVecType, mask, lower).getResult();
        }
    }
    if (tlaElem.isInteger(64)) {
        // The 64-bit AVE cmp/select ABI does not legalize in the device
        // backend; call the linked bitcode helpers directly.
        const bool scalar = !isa<::tla::VectorSSAType>(cmpOp.getRhs().getType());
        const bool isUnsigned = tlaElem.isUnsignedInteger();
        std::string typeName = isUnsigned ? "uint64_t" : "int64_t";
        // Must match the extern "C" names exported by
        // csrc/mlir/bc/Vector/cmp_sel_reg.cpp.
        std::string calleeName = (scalar ? "cmps_" : "cmp_") + cmpOp.getMode().str() + "_" + typeName;
        Value rhs;
        if (scalar) {
            rhs = lookupOrCloneScalarValue(b, cmpOp.getRhs(), valueMap);
            if (!rhs)
                return failure();
            if (auto intTy = dyn_cast<IntegerType>(rhs.getType()); intTy && intTy.isUnsigned()) {
                Type signless = IntegerType::get(b.getContext(), intTy.getWidth());
                rhs = b.create<UnrealizedConversionCastOp>(loc, signless, rhs).getResult(0);
            }
        } else {
            rhs = valueMap.lookup(cmpOp.getRhs());
            if (!rhs)
                return failure();
        }
        SmallVector<Type, 3> inputTypes{
            opCtx->vecType, scalar ? Type(b.getI64Type()) : Type(opCtx->vecType), opCtx->maskVecType};
        valueMap[cmpOp.getResult()] =
            createInlineB64Call(b, loc, module, calleeName, inputTypes, opCtx->maskVecType, ValueRange{lhs, rhs, mask});
        return success();
    }
    if (isa<::tla::VectorSSAType>(cmpOp.getRhs().getType())) {
        Value rhs = valueMap.lookup(cmpOp.getRhs());
        if (!rhs)
            return failure();
        valueMap[cmpOp.getResult()] = b.create<hivmave::VFCmpOp>(loc, opCtx->maskVecType, *cmpType, lhs, rhs, mask);
    } else {
        Value rhs = lookupOrCloneScalarValue(b, cmpOp.getRhs(), valueMap);
        if (!rhs)
            return failure();
        // The AVE scalar ABI is signless; unsigned TLA scalars are bit-identical.
        if (auto intTy = dyn_cast<IntegerType>(rhs.getType()); intTy && intTy.isUnsigned()) {
            Type signless = IntegerType::get(b.getContext(), intTy.getWidth());
            rhs = b.create<UnrealizedConversionCastOp>(loc, signless, rhs).getResult(0);
        }
        auto scalarOr = castScalarForVectorElement(rhs, aveVectorElementType(opCtx->elementType));
        if (failed(scalarOr))
            return failure();
        Value scalar = *scalarOr;
        valueMap[cmpOp.getResult()] = b.create<hivmave::VFCmpS>(loc, opCtx->maskVecType, *cmpType, lhs, scalar, mask);
    }
    return success();
}

/// AscendC StoreAlign b64 (RegTraitNumOne):
/// ``ppack(tmp, mask, LOWEST); pintlv_b32(dst0, dst1, tmp, tmp);`` then
/// ``vsts(uint32_t, …, dst0)``. Expands each logical b64 enable into two
/// consecutive b32 DistVST slots. ``mask`` must already be in AscendC's
/// single-register b64 layout (every eighth bit) — ``create_mask`` /
/// ``update_mask`` apply ``Punpack`` for lanes==32; generated plt/pge masks
/// need that Punpack first.
static Value expandB64MaskForU32AlignStore(OpBuilder& b, Location loc, Value mask)
{
    VectorType pregType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregType);
    // HighLowPart::LOWEST == 0 (same encoding AscendC passes to ppack).
    Value lowest = b.create<arith::ConstantIntOp>(loc, /*value=*/0, /*width=*/32);
    Value packed = b.create<hivm_regbaseintrins::PpackInstrOp>(loc, pregType, preg, lowest).getResult();
    auto pairType = LLVM::LLVMStructType::getLiteral(b.getContext(), {pregType, pregType});
    auto pintlv = b.create<hivm_regbaseintrins::PintlvB32InstrOp>(loc, pairType, packed, packed);
    return b.create<LLVM::ExtractValueOp>(loc, pintlv.getResult(), 0);
}

static Value unpackB64Mask(OpBuilder& b, Location loc, Value mask)
{
    VectorType pregType = fullPregVecType(b.getContext());
    Value lower = b.create<arith::ConstantIntOp>(loc, 0, 32);
    return b.create<hivm_regbaseintrins::PunpackInstrOp>(
        loc, pregType, castMaskToPregType(b, loc, mask, pregType), lower);
}

static LogicalResult lowerStore(
    ::tla::StoreOp storeOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = storeOp.getLoc();
    Value dest = valueMap.lookup(storeOp.getDest());
    Value source = valueMap.lookup(storeOp.getSource());
    if (!dest || !source)
        return failure();

    // MaskSSA source: 1/2/4/8-byte UB ← i1 memref view ← masked_store <NORM_B8>.
    if (isa<::tla::MaskSSAType>(storeOp.getSource().getType())) {
        auto maskType = cast<::tla::MaskSSAType>(storeOp.getSource().getType());
        int64_t lanes = maskType.getPhysicalLanes();
        auto i1MemrefOr = materializeI1MaskMemrefFromUb(b, loc, dest, lanes);
        if (failed(i1MemrefOr))
            return storeOp.emitError("failed to materialize i1 memref view for tla.store MaskSSA"), failure();
        VectorType semanticMaskType = VectorType::get({lanes}, b.getI1Type());
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        Value allTrue = fullMasks.get(loc, lanes);
        Value storeVal = source;
        if (storeVal.getType() != semanticMaskType)
            storeVal = b.create<UnrealizedConversionCastOp>(loc, semanticMaskType, storeVal).getResult(0);
        Value storePred = allTrue;
        if (storePred.getType() != semanticMaskType)
            storePred = b.create<UnrealizedConversionCastOp>(loc, semanticMaskType, storePred).getResult(0);
        b.create<hivmave::VFMaskedStoreOp>(
            loc, hivmave::StoreDist::NORM_B8, *i1MemrefOr, ValueRange{zero}, storePred, storeVal);
        return success();
    }

    Type tlaStoreElem = cast<::tla::VectorSSAType>(storeOp.getSource().getType()).getElementType();
    auto opCtx = deriveVecCtxForElement(tlaStoreElem);
    if (failed(opCtx))
        return failure();
    bool isB64 = isAlignB64Elem(tlaStoreElem);
    if (storeOp.getUnalignedUbAccess().value_or(false)) {
        auto count = getTlaTensorValidLaneCount(b, loc, storeOp.getDest(), valueMap);
        if (failed(count))
            return failure();
        // The mapped view includes both the live UPDATE base and the tile offset.
        auto addr = memrefByteAddrI64(b, loc, dest);
        if (failed(addr))
            return storeOp.emitError("unaligned store requires an effective UB address"), failure();
        Value ptr = intToUbPtr(b, loc, *addr);
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        Value full = b.create<arith::ConstantIndexOp>(loc, opCtx->lanes);
        Value nonnegative = b.create<arith::MaxSIOp>(loc, *count, zero);
        Value lanes = b.create<arith::MinSIOp>(loc, nonnegative, full);
        Value bytes = strideElemsToByteOffI32(b, loc, lanes, tlaStoreElem);
        Type carrier = tlaStoreElem;
        if (isB64) {
            carrier = b.getI32Type();
            auto words = VectorType::get({64}, carrier);
            source = b.create<mlir::vector::BitCastOp>(loc, words, source).getResult();
        }
        Value ureg = emitEmptyStoreUreg(b, loc);
        auto result = emitVstusPost(b, loc, source, ptr, ureg, bytes, carrier);
        if (failed(result))
            return storeOp.emitError("unsupported element type for unaligned store: ") << tlaStoreElem, failure();
        return emitVstas(b, loc, result->first, result->second);
    }

    Value mask = storeOp.getMask() ? valueMap.lookup(storeOp.getMask()) : Value{};
    if (storeOp.getMask() && !mask)
        return failure();

    auto dist = storeOp.getStoreDist() ? storeOp.getStoreDist()->getStoreDist() : ::StoreDist::norm;
    if (isB64 && dist == ::StoreDist::norm) {
        auto words = VectorType::get({64}, b.getI32Type());
        source = b.create<mlir::vector::BitCastOp>(loc, words, source).getResult();
        if (mask) {
            mask = expandB64MaskForU32AlignStore(b, loc, mask);
        } else if (isProvablyFullVectorStore(storeOp, 32)) {
            mask = fullMasks.get(loc, 64);
        } else {
            auto count = getTlaTensorValidLaneCount(b, loc, storeOp.getDest(), valueMap);
            if (failed(count))
                return failure();
            Value two = b.create<arith::ConstantIndexOp>(loc, 2);
            Value activeWords = b.create<arith::MulIOp>(loc, *count, two);
            mask = createPredicatePlt(b, loc, fullPregVecType(b.getContext()), 64, activeWords).getRes();
        }
        auto wordsMemOr = materializeDataMemref(b, loc, dest, b.getI32Type());
        if (failed(wordsMemOr))
            return storeOp.emitError("failed to materialize i32 memref view for 64-bit VectorSSA store"), failure();
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        b.create<hivmave::VFMaskedStoreOp>(loc, *wordsMemOr, ValueRange{zero}, mask, source);
        return success();
    }

    auto sourceTy = dyn_cast<VectorType>(source.getType());
    if (!sourceTy)
        return failure();
    Type tlaDestElem = cast<::tla::TlaTensorType>(storeOp.getDest().getType()).getPtr().getPointee();
    Value storeDest = dest;
    // Same-shape i8 view only when both sides are Bool (TypeGet<bool> → vector_s8).
    if (isBoolVectorElement(tlaStoreElem) && isBoolVectorElement(tlaDestElem))
        storeDest = materializeSameShapeI8Memref(b, loc, dest);

    // Dual-source INTLV: same shape as Load DINTLV — HIVM vstsx2 + StoreDist.
    // AscendC asc_storealign_intlv ignores the predicate; pass all-true preg.
    if (auto storeDistAttr = storeOp.getStoreDist()) {
        hivmave::StoreDist pattern = mapTlaStoreDistToAve(storeDistAttr->getStoreDist());
        if (isDualSourceStoreDist(pattern)) {
            if (!storeOp.getSource2())
                return storeOp.emitError("intlv store_dist requires a second source"), failure();
            Value source2Val = valueMap.lookup(storeOp.getSource2());
            if (!source2Val)
                return failure();
            int64_t elemBytes = elementStorageBytes(tlaStoreElem);
            if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4)
                return storeOp.emitError("unsupported element type for intlv store"), failure();
            Value data0 = bitcastToIntlvCarrier(b, loc, source, elemBytes);
            Value data1 = bitcastToIntlvCarrier(b, loc, source2Val, elemBytes);
            if (!data0 || !data1)
                return storeOp.emitError("intlv store sources are not vectors"), failure();
            auto metadata = b.create<mlir::memref::ExtractStridedMetadataOp>(loc, storeDest);
            Value base = b.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, metadata.getBaseBuffer());
            Value elementBytes = b.create<arith::ConstantIndexOp>(loc, elemBytes);
            Value offset = b.create<arith::MulIOp>(loc, metadata.getOffset(), elementBytes);
            Value address = b.create<arith::AddIOp>(loc, base, offset);
            Value pointer = intToUbPtr(b, loc, ::tla::castValueToI64(b, loc, address));
            Value mask = fullMasks.get(loc, opCtx->lanes);
            if (failed(emitVstsx2Intlv(b, loc, pattern, data0, data1, pointer, mask)))
                return storeOp.emitError("unexpected dual store_dist"), failure();
            return success();
        }
    }

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    if (!mask) {
        auto implicitMask = createImplicitVectorStoreMask(b, loc, storeOp, *opCtx, valueMap, fullMasks);
        if (failed(implicitMask))
            return storeOp.emitError("failed to determine omitted tla.store active lanes"), failure();
        mask = *implicitMask;
    }
    if (storeOp.getBlockStride()) {
        Value blockStrideVal = lookupOrCloneScalarValue(b, storeOp.getBlockStride(), valueMap);
        if (!blockStrideVal)
            return failure();
        if (!isa<IntegerType>(blockStrideVal.getType()))
            blockStrideVal = b.create<arith::IndexCastOp>(loc, b.getI32Type(), blockStrideVal);
        std::string calleeName = getStoreWithStrideLibraryCallName(tlaStoreElem);
        if (calleeName.empty())
            return storeOp.emitError("unsupported element type for tla.store with BlockStoreParams: ") << tlaStoreElem,
                   failure();
        Value storeSrc = source;
        if (blockStrideUsesByteRegister(tlaStoreElem)) {
            storeSrc = b.create<mlir::vector::BitCastOp>(loc, byteRegisterTypeFor(sourceTy), source).getResult();
        }
        // Bool uses the i8 UB view (storeDest); other types keep dest == storeDest.
        auto callee =
            getOrCreateStoreWithStrideLibraryCall(module, loc, storeSrc.getType(), storeDest.getType(), calleeName);
        if (!callee)
            return failure();
        VectorType pregVecType = fullPregVecType(b.getContext());
        Value pregMask = castMaskToPregType(b, loc, mask, pregVecType);
        b.create<func::CallOp>(loc, callee, ValueRange{storeSrc, storeDest, blockStrideVal, pregMask});
    } else {
        auto storeDistAttr = storeOp.getStoreDist();
        if (storeDistAttr && storeDistAttr->getStoreDist() == ::StoreDist::pack_b64) {
            // No AIV ``masked_store_PK_B64_*``; emit HIVM vstsx1 with DistVST PK_B64.
            auto metadata = b.create<mlir::memref::ExtractStridedMetadataOp>(loc, storeDest);
            Value base = b.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, metadata.getBaseBuffer());
            Value elementBytes = b.create<arith::ConstantIndexOp>(loc, elementStorageBytes(tlaDestElem));
            Value offset = b.create<arith::MulIOp>(loc, metadata.getOffset(), elementBytes);
            Value address = b.create<arith::AddIOp>(loc, base, offset);
            Value pointer = intToUbPtr(b, loc, ::tla::castValueToI64(b, loc, address));
            Value pregMask = storeOp.getMask() ? castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext())) :
                                                 unpackB64Mask(b, loc, mask);
            emitVstsx1PackB64(b, loc, source, pointer, pregMask);
        } else if (storeDistAttr && storeDistAttr->getStoreDist() != ::StoreDist::norm) {
            // Bool uses the i8 UB view (storeDest) for b8 Dist / FIRST_ELEMENT.
            hivmave::StoreDist pattern = mapTlaStoreDistToAve(storeDistAttr->getStoreDist());
            b.create<hivmave::VFMaskedStoreOp>(loc, pattern, storeDest, ValueRange{zero}, mask, source);
        } else {
            // NORM mode (Bool uses i8 view of byte-addressed UB).
            b.create<hivmave::VFMaskedStoreOp>(loc, storeDest, ValueRange{zero}, mask, source);
        }
    }
    return success();
}

static LogicalResult lowerStoreUpdate(
    ::tla::StoreUpdateOp storeOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = storeOp.getLoc();
    Value dest = valueMap.lookup(storeOp.getDest());
    Value source = valueMap.lookup(storeOp.getSource());
    if (!dest || !source)
        return failure();
    auto sourceTy = dyn_cast<VectorType>(source.getType());
    if (!sourceTy)
        return failure();
    // Prefer TLA VectorSSA element (Bool=i1) over the AVE i8 carrier so UPDATE
    // cannot bypass the NORMAL-path block_stride move-only gate.
    Type tlaStoreElem = sourceTy.getElementType();
    if (auto tlaVec = dyn_cast<::tla::VectorSSAType>(storeOp.getSource().getType()))
        tlaStoreElem = tlaVec.getElementType();
    Type tlaDestElem = tlaStoreElem;
    if (auto tlaTensor = dyn_cast<::tla::TlaTensorType>(storeOp.getDest().getType()))
        tlaDestElem = tlaTensor.getPtr().getPointee();
    else if (auto destMem = dyn_cast<MemRefType>(dest.getType()))
        tlaDestElem = destMem.getElementType();
    if (tlaStoreElem != tlaDestElem)
        return storeOp.emitError("store_update source and destination element types must match"), failure();
    if (Value second = storeOp.getSource2()) {
        Value source2 = valueMap.lookup(second);
        Value stride = valueMap.lookup(storeOp.getPostUpdateStride());
        if (!stride)
            stride = lookupOrCloneScalarValue(b, storeOp.getPostUpdateStride(), valueMap);
        auto ptrOr = resolveEffectiveUbPtr(storeOp.getDest(), b, loc, valueMap);
        if (!source2 || !stride || failed(ptrOr))
            return storeOp.emitError("failed to materialize dual store_update operands"), failure();
        int64_t distBytes = std::min<int64_t>(elementStorageBytes(tlaStoreElem), 4);
        auto dist = mapTlaStoreDistToAve(storeOp.getStoreDist()->getStoreDist());
        Value data0 = bitcastToIntlvCarrier(b, loc, source, distBytes);
        Value data1 = bitcastToIntlvCarrier(b, loc, source2, distBytes);
        Value mask = fullMasks.get(loc, 256 / distBytes);
        if (!data0 || !data1 || failed(emitVstsx2Intlv(b, loc, dist, data0, data1, *ptrOr, mask)))
            return storeOp.emitError("failed to emit dual store_update"), failure();
        // VSTSx2 has no post-update instruction. Keep the same cursor SSA ABI
        // by advancing the effective address after the zero-offset store.
        Value address = b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), *ptrOr);
        Value byteOff = strideElemsToByteOffI32(b, loc, stride, tlaStoreElem);
        Value next = b.create<arith::AddIOp>(loc, address, castValueToI64(b, loc, byteOff));
        mapUpdateNextPtrI64(storeOp.getNextPtr(), next, valueMap);
        return success();
    }
    // Bool Align UPDATE: AscendC move-only b8 (AVE i8 carrier → s8 post).
    Value mask;
    auto opCtx = deriveVecCtxForElement(tlaStoreElem);
    if (failed(opCtx))
        return failure();
    bool generatedMaskNeedsB64Punpack = false;
    if (storeOp.getMask()) {
        mask = valueMap.lookup(storeOp.getMask());
        if (!mask)
            return failure();
    } else if (auto destStaticLanes = getStaticTlaTensorValidLaneCount(storeOp.getDest())) {
        int64_t activeLanes = *destStaticLanes >= opCtx->lanes ? opCtx->lanes : *destStaticLanes;
        if (activeLanes == opCtx->lanes)
            mask = fullMasks.get(loc, opCtx->lanes);
        else {
            Value active = b.create<arith::ConstantIndexOp>(loc, activeLanes);
            mask = createPredicatePlt(b, loc, opCtx->maskVecType, opCtx->lanes, active).getRes();
        }
        generatedMaskNeedsB64Punpack = isAlignB64Elem(sourceTy.getElementType());
    } else {
        auto destLanes = getTlaTensorValidLaneCount(b, loc, storeOp.getDest(), valueMap);
        if (failed(destLanes))
            return failure();
        mask = createPredicatePlt(b, loc, opCtx->maskVecType, opCtx->lanes, *destLanes).getRes();
        generatedMaskNeedsB64Punpack = isAlignB64Elem(sourceTy.getElementType());
    }
    VectorType pregVecType = fullPregVecType(b.getContext());
    Value pregMask;
    if (isAlignB64Elem(sourceTy.getElementType())) {
        // Match AscendC: b64 MaskReg → ppack+pintlv → u32 DistVST store preg.
        if (generatedMaskNeedsB64Punpack)
            mask = unpackB64Mask(b, loc, mask);
        pregMask = expandB64MaskForU32AlignStore(b, loc, mask);
    } else {
        pregMask = castMaskToPregType(b, loc, mask, pregVecType);
    }
    auto ptrOr = resolveEffectiveUbPtr(storeOp.getDest(), b, loc, valueMap);
    if (failed(ptrOr))
        return storeOp.emitError(
                   "store_update requires an effective UB address "
                   "(tensor_desc over live inttoptr / materialized memref)"),
               failure();

    // DATA_BLOCK_COPY UPDATE: no HIVMC vsstb.post — AIV bc returns the pointer.
    if (Value blockStrideVal = storeOp.getBlockStride()) {
        Value blockStride;
        Value repeatStride;
        if (failed(materializeBlockStrides(
                storeOp, blockStrideVal, storeOp.getPostUpdateStride(), b, valueMap, blockStride, repeatStride)))
            return failure();
        std::string suffix = dataBlockCopyElementSuffix(tlaStoreElem);
        if (suffix.empty())
            return storeOp.emitError("unsupported element type for DATA_BLOCK_COPY store_update: ") << tlaStoreElem,
                   failure();
        Value storeSrc = source;
        if (blockStrideUsesByteRegister(tlaStoreElem))
            storeSrc = b.create<mlir::vector::BitCastOp>(loc, byteRegisterTypeFor(sourceTy), source).getResult();
        auto callee = getOrCreateStoreUpdateRetptrLibraryCall(
            module, loc, storeSrc.getType(), ("_mlir_ciface_store_with_stride_post_update_retptr_" + suffix));
        auto call =
            b.create<func::CallOp>(loc, callee, ValueRange{storeSrc, *ptrOr, blockStride, repeatStride, pregMask});
        Value newAddr = b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), call.getResult(0));
        mapUpdateNextPtrI64(storeOp.getNextPtr(), newAddr, valueMap);
        return success();
    }

    Value stride = valueMap.lookup(storeOp.getPostUpdateStride());
    if (!stride)
        stride = lookupOrCloneScalarValue(b, storeOp.getPostUpdateStride(), valueMap);
    if (!stride)
        return storeOp.emitError("failed to materialize post_update_stride in vector helper"), failure();
    FailureOr<Value> newPtrOr = emitVstsx1Post(b, loc, source, *ptrOr, stride, pregMask, tlaStoreElem);
    if (failed(newPtrOr))
        return storeOp.emitError("unsupported element type for tla.store_update: ") << tlaStoreElem, failure();
    mapUpdateNextPtrI64(storeOp.getNextPtr(), b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), *newPtrOr), valueMap);
    return success();
}

/// Emit PSTU with the stream's ureg and return the updated ureg and UB pointer.
static FailureOr<std::pair<Value, Value>> emitPstu(
    OpBuilder& b, Location loc, Value mask, Value ptr, Value ureg, Type elemType)
{
    auto ubPtrTy = getUbPtrType(b.getContext());
    Type resultTy = LLVM::LLVMStructType::getLiteral(b.getContext(), {uregVectorType(b), ubPtrTy});
    Value preg = castMaskToPregType(b, loc, mask, fullPregVecType(b.getContext()));
    int64_t elemBytes = elementStorageBytes(elemType);
    Value result;
    if (elemBytes == 2)
        result = b.create<hivm_regbaseintrins::PSTUStoreS16InstOp>(loc, resultTy, preg, ptr, ureg).getResult();
    else if (elemBytes == 4)
        result = b.create<hivm_regbaseintrins::PSTUStoreS32InstOp>(loc, resultTy, preg, ptr, ureg).getResult();
    else
        return failure();
    Value nextUreg = b.create<LLVM::ExtractValueOp>(loc, result, 0);
    Value nextPtr = b.create<LLVM::ExtractValueOp>(loc, result, 1);
    return std::make_pair(nextUreg, nextPtr);
}

static LogicalResult lowerCompressedUpdate(
    ::tla::StoreUnalignUpdateOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& map);
static LogicalResult lowerCompressedPost(
    ::tla::StoreUnalignPostOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& map);

static LogicalResult lowerStoreUnalignUpdate(
    ::tla::StoreUnalignUpdateOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    if (op.getCompressed())
        return lowerCompressedUpdate(op, b, module, valueMap);
    Location loc = op.getLoc();
    Value src = valueMap.lookup(op.getSource());
    if (!src)
        return failure();
    auto ptrOr = resolveEffectiveUbPtr(op.getDest(), b, loc, valueMap);
    if (failed(ptrOr))
        return op.emitError("store_unalign_update requires a live UB address"), failure();
    Value uregIn = valueMap.lookup(op.getUregIn());
    if (!uregIn)
        return op.emitError("store_unalign_update ureg_in not mapped in vector helper"), failure();
    Type destElem = cast<::tla::TlaTensorType>(op.getDest().getType()).getPtr().getPointee();
    FailureOr<std::pair<Value, Value>> emitted = failure();
    if (isa<::tla::MaskSSAType>(op.getSource().getType())) {
        emitted = emitPstu(b, loc, src, *ptrOr, uregIn, destElem);
    } else {
        // The verifier checks matching TLA element types. Use the destination
        // type here so Bool uses its b8 carrier even if the AVE type is i8.
        Value stride = lookupOrCloneScalarValue(b, op.getPostUpdateStride(), valueMap);
        if (!stride)
            return op.emitError("failed to materialize post_update_stride"), failure();
        Value countBytes = strideElemsToByteOffI32(b, loc, stride, destElem);
        Value second;
        Value firstCount = countBytes;
        Value secondCount;
        Type carrierElem = destElem;
        if (Value secondSource = op.getSource2()) {
            Value other = valueMap.lookup(secondSource);
            if (!other)
                return op.emitError("dual unaligned store second source is not mapped"), failure();
            int64_t distBytes = std::min<int64_t>(elementStorageBytes(destElem), 4);
            Value even = bitcastToIntlvCarrier(b, loc, src, distBytes);
            Value odd = bitcastToIntlvCarrier(b, loc, other, distBytes);
            if (!even || !odd)
                return op.emitError("dual unaligned store sources must be vectors"), failure();
            auto pair =
                b.create<hivmave::VFInterleaveOp>(loc, TypeRange{even.getType(), odd.getType()}, ValueRange{even, odd});
            src = pair->getResult(0);
            second = pair->getResult(1);
            carrierElem = cast<VectorType>(src.getType()).getElementType();
            std::tie(firstCount, secondCount) = splitUnalignUpdateCount(b, loc, countBytes);
        }
        emitted = emitVstusPost(b, loc, src, *ptrOr, uregIn, firstCount, carrierElem);
        if (second && succeeded(emitted)) {
            auto [ureg, nextPtr] = *emitted;
            // The second emission uses the same carrier as the first.
            emitted = emitVstusPost(b, loc, second, nextPtr, ureg, secondCount, carrierElem);
        }
    }
    if (failed(emitted))
        return op.emitError("unsupported element type for tla.store_unalign_update"), failure();
    valueMap[op.getUregOut()] = emitted->first;
    mapUpdateNextPtrI64(op.getNextPtr(), b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), emitted->second), valueMap);
    // Stream-tail ``vstas`` comes from ``store_unalign_post`` only.
    return success();
}

static LogicalResult lowerLoadUnalignPre(::tla::LoadUnalignPreOp op, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    Location loc = op.getLoc();
    auto ptrOr = resolveEffectiveUbPtr(op.getSource(), b, loc, valueMap);
    if (failed(ptrOr))
        return op.emitError("load_unalign_pre requires a live UB address"), failure();
    FailureOr<Value> ureg = emitVldas(b, loc, *ptrOr);
    if (failed(ureg))
        return failure();
    valueMap[op.getUreg()] = *ureg;
    return success();
}

static LogicalResult lowerCreateUreg(::tla::CreateUregOp op, OpBuilder& b, DenseMap<Value, Value>& valueMap)
{
    valueMap[op.getUreg()] = emitEmptyStoreUreg(b, op.getLoc());
    return success();
}

static LogicalResult lowerStoreUnalignPost(
    ::tla::StoreUnalignPostOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    if (op.getCompressed())
        return lowerCompressedPost(op, b, module, valueMap);
    Location loc = op.getLoc();
    Value ureg = valueMap.lookup(op.getUreg());
    if (!ureg)
        return op.emitError("store_unalign_post ureg not mapped in vector helper"), failure();
    auto ptrOr = resolveEffectiveUbPtr(op.getDest(), b, loc, valueMap);
    if (failed(ptrOr))
        return op.emitError("store_unalign_post requires a live UB address"), failure();
    return emitVstas(b, loc, ureg, *ptrOr);
}

static func::FuncOp getOrCreateCompressionCall(
    ModuleOp module, Location loc, StringRef name, TypeRange args, TypeRange results)
{
    if (auto function = module.lookupSymbol<func::FuncOp>(name))
        return function;
    OpBuilder moduleBuilder(module.getBodyRegion());
    auto function =
        moduleBuilder.create<func::FuncOp>(loc, name, FunctionType::get(module.getContext(), args, results));
    annotateAivLibraryCall(function);
    return function;
}

static std::string compressionStoreName(Type elementType)
{
    std::string name = getSqueezeLibraryCallName(elementType, false);
    if (name.empty())
        return {};
    return "squeeze_store_" + name.substr(std::string("vsqueeze_").size());
}

static LogicalResult lowerClearSqueezeState(::tla::ClearSqueezeStateOp op, OpBuilder& b, ModuleOp module)
{
    auto function = getOrCreateCompressionCall(module, op.getLoc(), "squeeze_stream_clear", {}, {});
    b.create<func::CallOp>(op.getLoc(), function, ValueRange{});
    return success();
}

static LogicalResult lowerCompressedUpdate(
    ::tla::StoreUnalignUpdateOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& map)
{
    Value source = map.lookup(op.getSource());
    Value state = map.lookup(op.getUregIn());
    if (!source || !state)
        return op.emitOpError("compression source or ureg_in was not lowered");
    auto pointer = resolveEffectiveUbPtr(op.getDest(), b, op.getLoc(), map);
    if (failed(pointer))
        return op.emitOpError("compression store requires a live UB destination");
    Type element = cast<::tla::VectorSSAType>(op.getSource().getType()).getElementType();
    std::string name = compressionStoreName(element);
    if (name.empty())
        return op.emitOpError("unsupported compression store element type");
    auto function = getOrCreateCompressionCall(
        module, op.getLoc(), name, TypeRange{source.getType(), pointer->getType(), state.getType()},
        TypeRange{state.getType()});
    auto call = b.create<func::CallOp>(op.getLoc(), function, ValueRange{source, *pointer, state});
    map[op.getUregOut()] = call.getResult(0);
    if (!op.getNextPtr().use_empty())
        mapUpdateNextPtrI64(op.getNextPtr(), b.create<LLVM::PtrToIntOp>(op.getLoc(), b.getI64Type(), *pointer), map);
    return success();
}

static LogicalResult lowerCompressedPost(
    ::tla::StoreUnalignPostOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& map)
{
    Value state = map.lookup(op.getUreg());
    Value dest = op.getDest();
    Value pointer;
    if (isa<::tla::TlaTensorType>(dest.getType())) {
        auto resolved = resolveEffectiveUbPtr(dest, b, op.getLoc(), map);
        if (succeeded(resolved))
            pointer = *resolved;
    } else if (Value address = lookupOrCloneScalarValue(b, dest, map)) {
        pointer = intToUbPtr(b, op.getLoc(), address);
    }
    if (!state || !pointer)
        return op.emitOpError("compression stream tail requires a live UB destination and ureg");
    // VSTAR writes the buffered bytes for every element type.
    auto function = getOrCreateCompressionCall(
        module, op.getLoc(), "squeeze_store_post_uint8_t", TypeRange{pointer.getType(), state.getType()}, {});
    b.create<func::CallOp>(op.getLoc(), function, ValueRange{pointer, state});
    return success();
}

static scf::IfOp lowerCompressedForExit(scf::ForOp loop)
{
    // A compressed store updates the alignment register in place. With the
    // test in the loop header, the device compiler can copy that register
    // only on reentry and leave the tail write using stale state on exit.
    // Test after the body so both successors receive the current state.
    Location loc = loop.getLoc();
    OpBuilder b(loop);
    auto predicate = loop.getUnsignedCmp() ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
    Value enters = b.create<arith::CmpIOp>(loc, predicate, loop.getLowerBound(), loop.getUpperBound());
    auto guard = b.create<scf::IfOp>(loc, loop.getResultTypes(), enters, /*withElseRegion=*/true);

    SmallVector<Value> inits{loop.getLowerBound()};
    llvm::append_range(inits, loop.getInitArgs());
    b.setInsertionPointToEnd(guard.thenBlock());
    auto rotated = b.create<scf::WhileOp>(
        loc, TypeRange(ValueRange(inits)), inits, /*beforeBuilder=*/nullptr,
        [](OpBuilder& body, Location loc, ValueRange args) { body.create<scf::YieldOp>(loc, args); });
    rotated.getBefore().takeBody(loop.getRegion());
    b.create<scf::YieldOp>(loc, rotated.getResults().drop_front());

    auto yield = cast<scf::YieldOp>(rotated.getBeforeBody()->getTerminator());
    b.setInsertionPoint(yield);
    Value next = b.create<arith::AddIOp>(
        loc, rotated.getBeforeBody()->getArgument(0), loop.getStep(), arith::IntegerOverflowFlags::nsw);
    Value continues = b.create<arith::CmpIOp>(loc, predicate, next, loop.getUpperBound());
    SmallVector<Value> results{next};
    llvm::append_range(results, yield.getOperands());
    b.create<scf::ConditionOp>(loc, continues, results);
    yield.erase();

    b.setInsertionPointToEnd(guard.elseBlock());
    b.create<scf::YieldOp>(loc, loop.getInitArgs());
    loop.erase();
    return guard;
}

static LogicalResult lowerScfFor(
    scf::ForOp forOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = forOp.getLoc();
    Value lb = lookupOrCloneScalarValue(b, forOp.getLowerBound(), valueMap);
    Value ub = lookupOrCloneScalarValue(b, forOp.getUpperBound(), valueMap);
    Value step = lookupOrCloneScalarValue(b, forOp.getStep(), valueMap);
    if (!lb || !ub || !step)
        return failure();
    // Loop-carried `index` values (e.g. the tla.update_mask tail counter) are
    // carried across the loop as i64 instead: after scf->cf lowering the
    // downstream index->iN conversion only rewrites the induction variable, so
    // an index iter_arg would leave dangling index<->iN unrealized casts on the
    // carried value that ReconcileUnrealizedCasts cannot fold across the cf
    // block boundary. Casting at the boundaries with arith.index_cast keeps the
    // carried value a plain integer that lowers cleanly.
    Type i64Ty = b.getIntegerType(64);
    auto regionIterArgs = forOp.getRegionIterArgs();
    SmallVector<bool> wasIndex(regionIterArgs.size(), false);
    SmallVector<bool> wasPtr(regionIterArgs.size(), false);
    SmallVector<Value> mappedInits;
    for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
        Value mapped = valueMap.lookup(init);
        if (!mapped)
            mapped = lookupOrCloneScalarValue(b, init, valueMap);
        if (!mapped)
            return forOp.emitError("failed to map scf.for init_arg #") << idx << " (type " << init.getType() << ")",
                   failure();
        wasIndex[idx] = isa<IndexType>(init.getType());
        wasPtr[idx] = isa<::tla::PtrType>(init.getType());
        if (wasIndex[idx] && isa<IndexType>(mapped.getType()))
            mapped = b.create<arith::IndexCastOp>(loc, i64Ty, mapped);
        // !tla.ptr carriers: helper carries i64. UPDATE lowering maps next_ptr
        // to i64; IntToPtr inits may still be !llvm.ptr — normalize so
        // init/yield types match. If valueMap still holds abstract !tla.ptr
        // (e.g. TensorDesc base not yet rewritten), rematerialize instead of
        // hard-failing the common path.
        if (wasPtr[idx]) {
            if (isa<::tla::PtrType>(mapped.getType())) {
                mapped = materializeTlaPtrCarrierAsI64(b, loc, init, valueMap);
                if (!mapped)
                    return forOp.emitError("scf.for !tla.ptr init_arg #")
                               << idx << " could not materialize to i64 (!llvm.ptr / inttoptr addr)",
                           failure();
            } else if (isa<LLVM::LLVMPointerType>(mapped.getType())) {
                mapped = b.create<LLVM::PtrToIntOp>(loc, i64Ty, mapped);
            } else if (!mapped.getType().isInteger(64)) {
                return forOp.emitError("scf.for !tla.ptr init_arg #") << idx << " must lower to i64 or !llvm.ptr",
                       failure();
            }
            valueMap[init] = mapped;
        }
        mappedInits.push_back(mapped);
    }

    LogicalResult bodyStatus = success();
    auto newFor = b.create<scf::ForOp>(
        loc, lb, ub, step, mappedInits, [&](OpBuilder& nb, Location nloc, Value iv, ValueRange iterArgs) {
            DenseMap<Value, Value> nestedMap = valueMap;
            nestedMap[forOp.getInductionVar()] = iv;
            for (size_t i = 0; i < regionIterArgs.size(); ++i) {
                Value newArg = iterArgs[i];
                if (wasIndex[i])
                    newArg = nb.create<arith::IndexCastOp>(nloc, nb.getIndexType(), newArg);
                nestedMap[regionIterArgs[i]] = newArg;
            }
            if (failed(lowerNestedVectorBlock(forOp.getBody(), nb, module, nestedMap, fullMasks))) {
                bodyStatus = failure();
                nb.create<scf::YieldOp>(nloc, iterArgs);
                return;
            }
            auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            SmallVector<Value> yielded;
            for (auto [i, v] : llvm::enumerate(oldYield.getOperands())) {
                Value mapped = nestedMap.lookup(v);
                if (!mapped)
                    mapped = lookupOrCloneScalarValue(nb, v, nestedMap);
                if (!mapped) {
                    bodyStatus = failure();
                    break;
                }
                if (wasIndex[i] && isa<IndexType>(mapped.getType()))
                    mapped = nb.create<arith::IndexCastOp>(nloc, i64Ty, mapped);
                if (wasPtr[i]) {
                    if (isa<::tla::PtrType>(mapped.getType())) {
                        mapped = materializeTlaPtrCarrierAsI64(nb, nloc, v, nestedMap);
                        if (!mapped) {
                            bodyStatus = failure();
                            break;
                        }
                    } else if (isa<LLVM::LLVMPointerType>(mapped.getType())) {
                        mapped = nb.create<LLVM::PtrToIntOp>(nloc, i64Ty, mapped);
                    } else if (!mapped.getType().isInteger(64)) {
                        bodyStatus = failure();
                        break;
                    }
                    nestedMap[v] = mapped;
                }
                yielded.push_back(mapped);
            }
            if (failed(bodyStatus)) {
                nb.create<scf::YieldOp>(nloc, iterArgs);
                return;
            }
            nb.create<scf::YieldOp>(nloc, yielded);
        });
    if (failed(bodyStatus))
        return failure();
    ValueRange results = newFor.getResults();
    if (forOp
            .walk([](::tla::StoreUnalignUpdateOp store) {
                return store.getCompressed() ? WalkResult::interrupt() : WalkResult::advance();
            })
            .wasInterrupted())
        results = lowerCompressedForExit(newFor).getResults();
    // Only remap for *results*. Do not overwrite valueMap entries for yield
    // operands: those SSA values often exist outside the loop (e.g. ``b`` in
    // ``a = b``), and remapping them to for-results corrupts post-loop users
    // even on zero-trip loops. Stream-tail Post uses for results, covered here.
    for (auto [i, oldRes] : llvm::enumerate(forOp.getResults())) {
        Value newRes = results[i];
        if (wasIndex[i] && !oldRes.use_empty())
            newRes = b.create<arith::IndexCastOp>(loc, b.getIndexType(), newRes);
        valueMap[oldRes] = newRes;
    }
    return success();
}

/// scf.if: rebuild result-bearing conditionals after converting register
/// carrier types to physical builtin vectors. Each branch is lowered with an
/// independent value map, and its old scf.yield operands feed the new op.
static LogicalResult lowerScfIf(
    scf::IfOp ifOp, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap, AutoFullPredicateCache& fullMasks)
{
    Location loc = ifOp.getLoc();
    Value cond = lookupOrCloneScalarValue(b, ifOp.getCondition(), valueMap);
    if (!cond)
        return failure();

    SmallVector<Type> resultTypes;
    for (Value result : ifOp.getResults()) {
        auto loweredType = lowerSCFCarrierType(result.getType());
        if (failed(loweredType))
            return failure();
        resultTypes.push_back(*loweredType);
    }

    bool hasElse = !ifOp.getElseRegion().empty();
    auto newIf = b.create<scf::IfOp>(loc, resultTypes, cond, hasElse);
    auto lowerBranch = [&](Block* oldBlock, Block* newBlock) -> LogicalResult {
        DenseMap<Value, Value> branchMap = valueMap;
        if (!newBlock->empty() && newBlock->back().hasTrait<OpTrait::IsTerminator>())
            newBlock->back().erase();
        OpBuilder branchBuilder = OpBuilder::atBlockEnd(newBlock);
        if (failed(lowerNestedVectorBlock(oldBlock, branchBuilder, module, branchMap, fullMasks)))
            return failure();

        auto oldYield = dyn_cast<scf::YieldOp>(oldBlock->getTerminator());
        if (!oldYield)
            return failure();
        SmallVector<Value> yielded;
        for (Value operand : oldYield.getOperands()) {
            Value mapped = lookupOrCloneScalarValue(branchBuilder, operand, branchMap);
            if (!mapped)
                return failure();
            yielded.push_back(mapped);
        }
        branchBuilder.create<scf::YieldOp>(oldYield.getLoc(), yielded);
        return success();
    };

    if (failed(lowerBranch(ifOp.thenBlock(), newIf.thenBlock())))
        return failure();
    if (hasElse && failed(lowerBranch(ifOp.elseBlock(), newIf.elseBlock())))
        return failure();

    for (auto [oldResult, newResult] : llvm::zip(ifOp.getResults(), newIf.getResults()))
        valueMap[oldResult] = newResult;
    return success();
}

static Value vfPrintCount(OpBuilder& b, Location loc, Value count, int64_t capacity)
{
    Value value = ::tla::castValueToI64(b, loc, count);
    Value zero = b.create<arith::ConstantIntOp>(loc, 0, 64);
    Value limit = b.create<arith::ConstantIntOp>(loc, capacity, 64);
    value = b.create<arith::MaxSIOp>(loc, value, zero);
    value = b.create<arith::MinSIOp>(loc, value, limit);
    return b.create<arith::TruncIOp>(loc, b.getI32Type(), value);
}

static LogicalResult emitVfDumpCall(
    OpBuilder& b, Location loc, ModuleOp module, StringRef kind, Type elementType, ValueRange operands)
{
    std::string diagnostic;
    auto suffix = getPrintTensorHelperSuffix(elementType, diagnostic);
    if (failed(suffix))
        return emitError(loc, diagnostic);
    std::string name = ("tla_vf_print_" + kind + "_" + *suffix).str();
    auto callee = module.lookupSymbol<func::FuncOp>(name);
    if (!callee) {
        OpBuilder mb(module.getBodyRegion());
        SmallVector<Type> types;
        for (Value operand : operands)
            types.push_back(operand.getType());
        callee = mb.create<func::FuncOp>(loc, name, b.getFunctionType(types, {}));
        annotateAivLibraryCall(callee);
        callee->setAttr("hacc.always_inline", b.getUnitAttr());
    }
    b.create<func::CallOp>(loc, callee, operands);
    return success();
}

static LogicalResult lowerVfRegisterPrint(
    ::tla::PrintVectorOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    Value value = valueMap.lookup(op.getValue());
    Value length = lookupOrCloneScalarValue(b, op.getLength(), valueMap);
    if (!value || !length)
        return op.emitError("cannot capture VF register print operands");
    auto type = cast<::tla::VectorSSAType>(op.getValue().getType());
    Value count = vfPrintCount(b, op.getLoc(), length, 2048 / type.getElementType().getIntOrFloatBitWidth());
    auto id = op->getAttrOfType<IntegerAttr>("tla.vf_print_id");
    if (!id)
        return op.emitError("VF register print call ID was not assigned");
    Value callId = b.create<arith::ConstantIntOp>(op.getLoc(), id.getInt(), 32);
    return emitVfDumpCall(b, op.getLoc(), module, "reg", type.getElementType(), {value, count, callId});
}

static LogicalResult lowerVfTensorPrint(
    ::tla::PrintTensorOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap)
{
    auto desc = op.getValue().getDefiningOp<::tla::TensorDescOp>();
    auto type = cast<::tla::TlaTensorType>(op.getValue().getType());
    if (!desc || type.getPtr().getAddrspace() != AddressSpace::ub)
        return op.emitError("tensor printing inside a SIMD vec.func requires UB storage");
    Value value = valueMap.lookup(op.getValue());
    Value length = lookupOrCloneScalarValue(b, op.getLength(), valueMap);
    if (!value || !length)
        return op.emitError("cannot capture VF tensor print operands");
    Location loc = op.getLoc();
    int64_t rank = op.getShape().size();
    SmallVector<Value> shape;
    auto layout = type.getLayout().getLayoutTag();
    bool packed = layout != LayoutTag::RowMajor && layout != LayoutTag::ColumnMajor;
    for (unsigned i = 0; i < rank; ++i) {
        Value extent;
        if (op.getShape()[i] >= 0)
            extent = b.create<arith::ConstantIntOp>(loc, op.getShape()[i], 64);
        else {
            Value original = rank == 1 ? desc.getShape1() :
                             packed    ? (i == 0 ? desc.getOriginShape0() : desc.getOriginShape1()) :
                                         (i == 0 ? desc.getShape0() : desc.getShape1());
            extent = lookupOrCloneScalarValue(b, original, valueMap);
            if (!extent)
                return op.emitError("cannot capture VF tensor print shape");
            extent = ::tla::castValueToI64(b, loc, extent);
        }
        shape.push_back(extent);
    }
    Value elements = shape.front();
    if (rank == 2)
        elements = b.create<arith::MulIOp>(loc, elements, shape.back());
    length = b.create<arith::MinSIOp>(loc, length, elements);
    Type elementType = type.getPtr().getPointee();
    Value count = vfPrintCount(b, loc, length, 1952 * 8 / elementType.getIntOrFloatBitWidth());
    Value shape0 = b.create<arith::TruncIOp>(loc, b.getI32Type(), shape.front());
    Value shape1 = rank == 2 ? Value(b.create<arith::TruncIOp>(loc, b.getI32Type(), shape.back())) :
                               Value(b.create<arith::ConstantIntOp>(loc, 0, 32));
    Value rankValue = b.create<arith::ConstantIntOp>(loc, rank, 32);
    auto id = op->getAttrOfType<IntegerAttr>("tla.vf_print_id");
    if (!id)
        return op.emitError("VF tensor print call ID was not assigned");
    Value callId = b.create<arith::ConstantIntOp>(loc, id.getInt(), 32);
    return emitVfDumpCall(b, loc, module, "ub", elementType, {value, count, shape0, shape1, rankValue, callId});
}

// Intersect the input-domain mask with the shortest valid prefix known at compile time.
static Value limitMaskToValidLanes(
    OpBuilder& b, Location loc, Value mask, ValueRange inputs, AutoFullPredicateCache& fullMasks)
{
    Type element = cast<::tla::VectorSSAType>(inputs.front().getType()).getElementType();
    int64_t lanes = 2048 / element.getIntOrFloatBitWidth();
    int64_t validLanes = lanes;
    for (Value input : inputs) {
        int64_t count = cast<::tla::VectorSSAType>(input.getType()).getValidLanes();
        if (count != ShapedType::kDynamic)
            validLanes = std::min(validLanes, count);
    }
    auto pregType = fullPregVecType(b.getContext());
    Value preg = castMaskToPregType(b, loc, mask, pregType);
    if (validLanes == lanes)
        return preg;
    Value count = b.create<arith::ConstantIndexOp>(loc, validLanes);
    Value prefix = createPredicatePlt(b, loc, pregType, lanes, count).getRes();
    if (lanes == 32)
        prefix = unpackB64Mask(b, loc, prefix);
    return b
        .create<hivmave::PregAndOp>(
            loc, pregType, maskWidthAttrForElement(b, element), preg, prefix, fullMasks.get(loc, lanes))
        .getRes();
}

static LogicalResult lowerAxpy(
    ::tla::AxpyOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
    Value x = valueMap.lookup(op.getX());
    Value y = valueMap.lookup(op.getY());
    Value alpha = lookupOrCloneScalarValue(b, op.getAlpha(), valueMap);
    Value mask = valueMap.lookup(op.getMask());
    if (!x || !y || !alpha || !mask)
        return op.emitOpError("cannot map operands into the VF");
    Value preg = limitMaskToValidLanes(b, loc, mask, {op.getX(), op.getY()}, fullMasks);
    auto vectorType = cast<VectorType>(x.getType());
    Type element = op.getX().getType().getElementType();
    SmallVector<Type> types{vectorType, vectorType, alpha.getType(), preg.getType()};
    if (element.isInteger(64)) {
        StringRef symbol = element.isUnsignedInteger(64) ? "tla_axpy_u64" : "tla_axpy_i64";
        valueMap[op.getResult()] =
            createInlineB64Call(b, loc, module, symbol, types, vectorType, ValueRange{x, y, alpha, preg});
    } else {
        StringRef symbol = element.isF16() ? "tla_axpy_f16" : "tla_axpy_f32";
        auto callee = getOrCreateInlineAivLibraryCall(module, loc, types, vectorType, symbol);
        valueMap[op.getResult()] = b.create<func::CallOp>(loc, callee, ValueRange{x, y, alpha, preg}).getResult(0);
    }
    return success();
}

static LogicalResult lowerSubAbs(
    ::tla::SubAbsOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
    Value lhs = valueMap.lookup(op.getLhs());
    Value rhs = valueMap.lookup(op.getRhs());
    Value mask = valueMap.lookup(op.getMask());
    if (!lhs || !rhs || !mask)
        return op.emitOpError("cannot map operands into the VF");
    Value preg = limitMaskToValidLanes(b, loc, mask, {op.getLhs(), op.getRhs()}, fullMasks);
    auto vectorType = cast<VectorType>(lhs.getType());
    if (vectorType.getElementType().isInteger(64)) {
        valueMap[op.getResult()] = createInlineB64Call(
            b, loc, module, "tla_sub_abs_i64", {vectorType, vectorType, preg.getType()}, vectorType,
            ValueRange{lhs, rhs, preg});
    } else {
        valueMap[op.getResult()] = b.create<hivmave::VFAbsDiffOp>(loc, vectorType, lhs, rhs, preg).getResult();
    }
    return success();
}

static LogicalResult lowerMulsCast(
    ::tla::MulsCastOp op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
    Value src = valueMap.lookup(op.getSrc());
    Value scalar = lookupOrCloneScalarValue(b, op.getScalar(), valueMap);
    Value mask = valueMap.lookup(op.getMask());
    if (!src || !scalar || !mask)
        return op.emitOpError("cannot map operands into the VF");
    Value preg = limitMaskToValidLanes(b, loc, mask, {op.getSrc()}, fullMasks);
    auto resultType = VectorType::get({128}, b.getF16Type());
    StringRef symbol = op.getPart() == 0 ? "tla_muls_cast_even" : "tla_muls_cast_odd";
    auto callee = getOrCreateInlineAivLibraryCall(
        module, loc, {src.getType(), scalar.getType(), preg.getType()}, resultType, symbol);
    valueMap[op.getResult()] = b.create<func::CallOp>(loc, callee, ValueRange{src, scalar, preg}).getResult(0);
    return success();
}

static LogicalResult lowerNestedVectorOp(
    Operation& op, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap, AutoFullPredicateCache& fullMasks)
{
    Location loc = op.getLoc();
    // Encodings are transported as bytes downstream, but must never silently
    // acquire integer arithmetic semantics. Reinterpret explicitly for bit
    // operations or convert to a native computation dtype first.
    auto isE8M0Vector = [](Type type) {
        auto vec = dyn_cast<::tla::VectorSSAType>(type);
        return vec && isa<::tla::Float8E8M0Type>(vec.getElementType());
    };
    if (llvm::any_of(op.getOperandTypes(), isE8M0Vector) || llvm::any_of(op.getResultTypes(), isE8M0Vector)) {
        // AscendC b8 transport (SupportBytes<1>→uint8_t): NORM / byte Dist /
        // DINTLV_B8 / Block / UnAlign / UPDATE, BF16 exponent extraction and UInt8 byte views.
        if (!isa<
                ::tla::LoadOp, ::tla::StoreOp, ::tla::LoadUpdateOp, ::tla::StoreUpdateOp, ::tla::LoadUnalignUpdateOp,
                ::tla::StoreUnalignUpdateOp, ::tla::FullOp, ::tla::CastOp, ::tla::BitCastOp, scf::ForOp, scf::IfOp>(
                op)) {
            return op.emitError("E8M0 arithmetic is unsupported; use a UInt8 register view for byte operations");
        }
    }
    bool hasHiFloat8 = llvm::any_of(
                           op.getOperandTypes(),
                           [](Type type) {
                               auto vec = dyn_cast<::tla::VectorSSAType>(type);
                               return vec && isa<::tla::HiFloat8Type>(vec.getElementType());
                           }) ||
                       llvm::any_of(op.getResultTypes(), [](Type type) {
                           auto vec = dyn_cast<::tla::VectorSSAType>(type);
                           return vec && isa<::tla::HiFloat8Type>(vec.getElementType());
                       });
    auto isMoveOnlyEncodingOp = [](Operation& candidate) {
        return isa<
            ::tla::CastOp, ::tla::BitCastOp, ::tla::LoadOp, ::tla::StoreOp, ::tla::LoadUpdateOp, ::tla::StoreUpdateOp,
            ::tla::LoadUnalignUpdateOp, ::tla::StoreUnalignUpdateOp, ::tla::LoadUnalignPreOp, ::tla::CreateUregOp,
            ::tla::StoreUnalignPostOp, ::tla::WhereOp, ::tla::InterleaveOp, ::tla::DeInterleaveOp, ::tla::GatherOp,
            ::tla::FullOp, scf::ForOp, scf::IfOp>(candidate);
    };
    if (hasHiFloat8 && !isMoveOnlyEncodingOp(op))
        return op.emitError("HiFloat8 arithmetic is unsupported; convert or bitcast the register explicitly");
    // Packed pairs use byte carriers. Bitwise ops act on the whole carrier
    // without interpreting the two nibbles, so they are explicit byte-level
    // operations like where/bitcast rather than integer arithmetic.
    auto isFp4Vector = [](Type type) {
        auto vec = dyn_cast<::tla::VectorSSAType>(type);
        return vec && (isFp4PairType(vec.getElementType()) || isInt4PairType(vec.getElementType()));
    };
    bool hasFp4Vector =
        llvm::any_of(op.getOperandTypes(), isFp4Vector) || llvm::any_of(op.getResultTypes(), isFp4Vector);
    if (hasFp4Vector &&
        !isa<
            ::tla::CastOp, ::tla::BitCastOp, ::tla::LoadOp, ::tla::StoreOp, ::tla::LoadUpdateOp, ::tla::StoreUpdateOp,
            ::tla::LoadUnalignUpdateOp, ::tla::StoreUnalignUpdateOp, ::tla::LoadUnalignPreOp, ::tla::CreateUregOp,
            ::tla::StoreUnalignPostOp, ::tla::WhereOp, ::tla::InterleaveOp, ::tla::DeInterleaveOp, ::tla::GatherOp,
            ::tla::BitwiseNotOp, ::tla::BitwiseAndOp, ::tla::BitwiseOrOp, ::tla::BitwiseXorOp, ::tla::FullOp,
            scf::ForOp, scf::IfOp>(op))
        return op.emitError(
            "Packed 4-bit pair arithmetic is unsupported; only bitwise, select and layout ops are allowed");

    // make_shape / make_coord are dead after tla-lower-tensor-desc (their leaves
    // were folded into tensor_desc operands); skip them. (tla-finalize-memref
    // erases them.)
    if (isa<::tla::MakeShapeOp, ::tla::MakeCoordOp, ::tla::MakeStrideOp, ::tla::MakeLayoutOp>(op))
        return success();

    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        valueMap[constant.getResult()] = b.clone(op)->getResult(0);
        return success();
    }

    if (auto call = dyn_cast<::tla::CallExternOp>(op)) {
        SmallVector<Value> args;
        for (Value operand : call.getOperands()) {
            Value mapped = lookupOrCloneScalarValue(b, operand, valueMap);
            if (!mapped)
                return call.emitOpError("failed to map extern argument into the VF");
            args.push_back(mapped);
        }
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (!callee)
            return call.emitOpError("failed to find extern callee in module: ") << call.getCallee();
        auto lowered = b.create<func::CallOp>(loc, callee, args);
        for (auto [source, result] : llvm::zip(call.getResults(), lowered.getResults()))
            valueMap[source] = result;
        return success();
    }

    if (auto ptrToInt = dyn_cast<::tla::PtrToIntOp>(op)) {
        Value mapped = lookupOrCloneScalarValue(b, ptrToInt.getPtr(), valueMap);
        if (!mapped)
            return ptrToInt.emitError("failed to map tla.ptrtoint operand in vector helper"), failure();
        if (mapped.getType().isInteger(64))
            valueMap[ptrToInt.getResult()] = mapped;
        else if (isa<LLVM::LLVMPointerType>(mapped.getType()))
            valueMap[ptrToInt.getResult()] = b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), mapped);
        else
            return ptrToInt.emitError("expected i64 or !llvm.ptr for tla.ptrtoint"), failure();
        return success();
    }

    // Ptr-backed tiles: storage-identity outlines data memrefs onto the seed
    // UB helper arg for ABI dedup; the inttoptr still materializes the live
    // ``!llvm.ptr<6>`` so effective-address materialization / UPDATE `.post`
    // / SCF yield share end-to-end SSA.
    if (auto intToPtr = dyn_cast<::tla::IntToPtrOp>(op)) {
        Value addr = lookupOrCloneScalarValue(b, intToPtr.getAddr(), valueMap);
        if (!addr)
            return intToPtr.emitError("failed to map tla.inttoptr address in vector helper"), failure();
        valueMap[intToPtr.getResult()] = intToUbPtr(b, loc, addr);
        return success();
    }

    // tla.tensor_desc: materialize it as a subview of the helper's base memref
    // argument (see materializeTensorDescSubview).
    if (auto descOp = dyn_cast<::tla::TensorDescOp>(op))
        return materializeTensorDescSubview(descOp, b, valueMap);

    if (auto loadOp = dyn_cast<::tla::LoadOp>(op))
        return lowerLoad(loadOp, b, module, valueMap, fullMasks);

    if (auto loadUpdateOp = dyn_cast<::tla::LoadUpdateOp>(op))
        return lowerLoadUpdate(loadUpdateOp, b, module, valueMap, fullMasks);
    if (auto printOp = dyn_cast<::tla::DebugPrintOp>(op)) {
        SmallVector<Value> values;
        for (Value value : printOp.getValues()) {
            Value mapped = lookupOrCloneScalarValue(b, value, valueMap);
            if (!mapped)
                return printOp.emitError("cannot capture VF print operand");
            values.push_back(mapped);
        }
        return lowerVfScalarPrint(b, printOp, values);
    }
    if (auto printOp = dyn_cast<::tla::PrintVectorOp>(op))
        return lowerVfRegisterPrint(printOp, b, module, valueMap);
    if (auto printOp = dyn_cast<::tla::PrintTensorOp>(op))
        return lowerVfTensorPrint(printOp, b, module, valueMap);

    if (auto preOp = dyn_cast<::tla::LoadUnalignPreOp>(op))
        return lowerLoadUnalignPre(preOp, b, valueMap);
    if (auto clear = dyn_cast<::tla::ClearSqueezeStateOp>(op))
        return lowerClearSqueezeState(clear, b, module);
    if (auto emptyOp = dyn_cast<::tla::CreateUregOp>(op))
        return lowerCreateUreg(emptyOp, b, valueMap);
    if (auto postOp = dyn_cast<::tla::StoreUnalignPostOp>(op))
        return lowerStoreUnalignPost(postOp, b, module, valueMap);

    // Share the AR clear with compressed stores in the same vector function.
    if (isa<::tla::ArchClearArOp>(op)) {
        auto callee = getOrCreateCompressionCall(module, loc, "squeeze_stream_clear", {}, {});
        b.create<func::CallOp>(loc, callee, ValueRange{});
        return success();
    }

    if (auto loadUnalignUpdateOp = dyn_cast<::tla::LoadUnalignUpdateOp>(op))
        return lowerLoadUnalignUpdate(loadUnalignUpdateOp, b, valueMap);

    if (auto storeUnalignUpdateOp = dyn_cast<::tla::StoreUnalignUpdateOp>(op))
        return lowerStoreUnalignUpdate(storeUnalignUpdateOp, b, module, valueMap);

    // Preserve local-memory ordering while outlining a vec.func into its AVE
    // helper. Every operation in the source region must be recreated explicitly;
    // otherwise helper construction fails and leaves the whole vec.func behind.
    if (auto localMemBarOp = dyn_cast<::tla::LocalMemBarOp>(op)) {
        return lowerLocalMemBar(b, localMemBarOp);
    }

    if (auto bitCastOp = dyn_cast<::tla::BitCastOp>(op))
        return lowerBitCast(bitCastOp, b, valueMap);

    if (auto castOp = dyn_cast<::tla::CastOp>(op))
        return lowerCast(castOp, b, module, valueMap, fullMasks);

    if (auto axpy = dyn_cast<::tla::AxpyOp>(op))
        return lowerAxpy(axpy, b, module, valueMap, fullMasks);
    if (auto subAbs = dyn_cast<::tla::SubAbsOp>(op))
        return lowerSubAbs(subAbs, b, module, valueMap, fullMasks);
    if (auto mulsCast = dyn_cast<::tla::MulsCastOp>(op))
        return lowerMulsCast(mulsCast, b, module, valueMap, fullMasks);

    if (auto inplaceMulAddOp = dyn_cast<::tla::InplaceMulAddOp>(op)) {
        Value dst = valueMap.lookup(inplaceMulAddOp.getDst());
        Value multiplier = valueMap.lookup(inplaceMulAddOp.getMultiplier());
        Value addend = valueMap.lookup(inplaceMulAddOp.getAddend());
        if (!dst)
            return inplaceMulAddOp.emitError("inplace_mul_add failed to map dst operand");
        if (!multiplier)
            return inplaceMulAddOp.emitError("inplace_mul_add failed to map multiplier operand");
        if (!addend)
            return inplaceMulAddOp.emitError("inplace_mul_add failed to map addend operand");
        auto vecType = cast<VectorType>(dst.getType());
        Value mask = inplaceMulAddOp.getMask() ?
                         valueMap.lookup(inplaceMulAddOp.getMask()) :
                         allTrueMaskFor(fullMasks, loc, vecType, inplaceMulAddOp.getDst().getType());
        if (!mask)
            return inplaceMulAddOp.emitError("inplace_mul_add failed to materialize mask");
        auto pregType = fullPregVecType(b.getContext());
        Value preg = castMaskToPregType(b, loc, mask, pregType);
        Type element = vecType.getElementType();
        StringRef name = element.isF32() ? "inplace_mul_add_f32" :
                                           (element.isBF16() ? "inplace_mul_add_bf16" : "inplace_mul_add_f16");
        auto callee =
            getOrCreateInlineAivLibraryCall(module, loc, {vecType, vecType, vecType, pregType}, vecType, name);
        valueMap[inplaceMulAddOp.getResult()] =
            b.create<func::CallOp>(loc, callee, ValueRange{dst, multiplier, addend, preg}).getResult(0);
        return success();
    }

    if (auto expSubOp = dyn_cast<::tla::ExpSubOp>(op)) {
        Value lhs = valueMap.lookup(expSubOp.getLhs());
        Value rhs = valueMap.lookup(expSubOp.getRhs());
        Value mask = valueMap.lookup(expSubOp.getMask());
        if (!lhs || !rhs || !mask)
            return expSubOp.emitError("failed to map ExpSub operands"), failure();
        auto halfVec = VectorType::get({128}, b.getF16Type());
        auto floatVec = VectorType::get({64}, b.getF32Type());
        auto pregVec = fullPregVecType(b.getContext());
        bool halfInput = lhs.getType() == halfVec;
        if ((!halfInput && lhs.getType() != floatVec) || rhs.getType() != lhs.getType())
            return expSubOp.emitError("expected matching vector<128xf16> or vector<64xf32> inputs"), failure();
        Value preg = limitMaskToValidLanes(b, loc, mask, {expSubOp.getLhs(), expSubOp.getRhs()}, fullMasks);
        // Both FP32 slots use the same input positions. FP16 selects a BC entry
        // point with the corresponding compile-time Ascend C RegLayout.
        StringRef calleeName = "exp_sub_f32";
        if (halfInput)
            calleeName = expSubOp.getPart() == 0 ? "exp_sub_f16_even" : "exp_sub_f16_odd";
        SmallVector<Type> inputTypes{lhs.getType(), rhs.getType(), pregVec};
        auto callee = getOrCreateInlineAivLibraryCall(module, loc, inputTypes, floatVec, calleeName);
        valueMap[expSubOp.getResult()] = b.create<func::CallOp>(loc, callee, ValueRange{lhs, rhs, preg}).getResult(0);
        return success();
    }

    if (auto fullOp = dyn_cast<::tla::FullOp>(op))
        return lowerFull(fullOp, b, module, valueMap, fullMasks);

    if (auto arangeOp = dyn_cast<::tla::ArangeOp>(op))
        return lowerArange(arangeOp, b, module, valueMap, fullMasks);

    if (auto info = getVectorBinaryInfo(&op))
        return lowerVectorBinary(op, &*info, b, module, valueMap, fullMasks);

    if (auto info = getVectorScalarBinaryInfo(&op))
        return lowerVectorScalarBinary(op, &*info, b, module, valueMap, fullMasks);

    // tla.where: per-lane select. The mask controls which lanes take `x`; the
    // remaining lanes take `y`. VectorSSA lowers to ave.hir.vsel(mask, x, y),
    // while MaskSSA calls the linked psel bitcode helper.
    if (auto whereOp = dyn_cast<::tla::WhereOp>(op)) {
        Value mask = valueMap.lookup(whereOp.getMask());
        Value x = valueMap.lookup(whereOp.getX());
        Value y = valueMap.lookup(whereOp.getY());
        if (!mask || !x || !y)
            return failure();

        // After type conversion all MaskSSA values use vector<256xi1>, so the
        // original TLA operand type determines whether this is predicate select.
        if (isa<::tla::MaskSSAType>(whereOp.getX().getType())) {
            VectorType pregType = fullPregVecType(b.getContext());
            Value pregMask = castMaskToPregType(b, loc, mask, pregType);
            Value pregX = castMaskToPregType(b, loc, x, pregType);
            Value pregY = castMaskToPregType(b, loc, y, pregType);
            auto callee = getOrCreateMaskSelectLibraryCall(module, loc, pregType);
            valueMap[whereOp.getResult()] =
                b.create<func::CallOp>(loc, callee, ValueRange{pregX, pregY, pregMask}).getResult(0);
            return success();
        }

        // Result width follows the selected vectors' element type.
        auto xTy = dyn_cast<VectorType>(x.getType());
        if (!xTy)
            return failure();
        if (auto tlaVec = dyn_cast<::tla::VectorSSAType>(whereOp.getX().getType());
            tlaVec && tlaVec.getElementType().isInteger(64)) {
            // The 64-bit AVE vsel ABI does not legalize in the device backend;
            // call the linked bitcode helper with the native b64 predicate.
            // Name must match csrc/mlir/bc/Vector/cmp_sel_reg.cpp.
            std::string name = tlaVec.getElementType().isUnsignedInteger() ? "select_uint64_t" : "select_int64_t";
            SmallVector<Type, 3> inputTypes{fullPregVecType(b.getContext()), xTy, xTy};
            valueMap[whereOp.getResult()] =
                createInlineB64Call(b, loc, module, name, inputTypes, xTy, ValueRange{mask, x, y});
            return success();
        }
        auto opCtx = deriveVecCtxForElement(xTy.getElementType());
        if (failed(opCtx))
            return failure();
        valueMap[whereOp.getResult()] = b.create<hivmave::VFSelectOp>(loc, opCtx->vecType, mask, x, y);
        return success();
    }

    if (auto squeezeOp = dyn_cast<::tla::SqueezeOp>(op))
        return lowerSqueeze(squeezeOp, b, module, valueMap, fullMasks);

    if (auto unsqueezeOp = dyn_cast<::tla::UnsqueezeOp>(op))
        return lowerUnsqueeze(unsqueezeOp, b, module, valueMap);

    // tla.histograms: Reg FREQUENCY/ACCUMULATE histogram via linked bitcode
    // (dhistv2 / chistv2). Not lowered through HIVMAVE.
    if (auto histOp = dyn_cast<::tla::HistogramsOp>(op)) {
        Value src = valueMap.lookup(histOp.getSrc());
        if (!src)
            return failure();
        auto srcTy = dyn_cast<VectorType>(src.getType());
        if (!srcTy)
            return failure();
        auto srcElem = dyn_cast<IntegerType>(srcTy.getElementType());
        if (!srcElem || srcElem.getWidth() != 8)
            return histOp.emitError("tla.histograms src must be i8/ui8"), failure();

        auto histResultSSA = dyn_cast<VectorSSAType>(histOp.getResult().getType());
        if (!histResultSSA)
            return failure();
        Type resultElemType = histResultSSA.getElementType();
        auto resultElemInt = dyn_cast<IntegerType>(resultElemType);
        if (!resultElemInt || resultElemInt.getWidth() != 16)
            return histOp.emitError("tla.histograms result must be i16/ui16"), failure();

        auto srcCtx = deriveVecCtxForElement(srcElem);
        auto dstCtx = deriveVecCtxForElement(resultElemType);
        if (failed(srcCtx) || failed(dstCtx))
            return failure();

        std::string calleeName =
            getHistogramsLibraryCallName(histOp.getMode().getHistogramsBin(), histOp.getHistType().getHistogramsType());
        VectorType pregVecType = fullPregVecType(b.getContext());
        Value mask;
        if (histOp.getMask()) {
            mask = valueMap.lookup(histOp.getMask());
            if (!mask)
                return failure();
        } else {
            mask = fullMasks.get(loc, srcCtx->lanes);
        }
        Value preg = castMaskToPregType(b, loc, mask, pregVecType);

        // dst is required (in-place accumulator). Callers zero with
        // tla.full(0, ...) when starting a fresh histogram; no Pass-side zeroing.
        Value dst = valueMap.lookup(histOp.getDst());
        if (!dst)
            return failure();

        VectorType srcU8Type =
            VectorType::get(srcCtx->vecType.getShape(), IntegerType::get(b.getContext(), 8, IntegerType::Unsigned));
        VectorType dstU16Type =
            VectorType::get(dstCtx->vecType.getShape(), IntegerType::get(b.getContext(), 16, IntegerType::Unsigned));
        if (src.getType() != srcU8Type)
            src = b.create<UnrealizedConversionCastOp>(loc, srcU8Type, src).getResult(0);
        if (dst.getType() != dstU16Type)
            dst = b.create<UnrealizedConversionCastOp>(loc, dstU16Type, dst).getResult(0);

        auto callee = getOrCreateHistogramsLibraryCall(module, loc, srcU8Type, dstU16Type, pregVecType, calleeName);
        Value result = b.create<func::CallOp>(loc, callee, ValueRange{src, dst, preg}).getResult(0);
        if (result.getType() != dstCtx->vecType)
            result = b.create<UnrealizedConversionCastOp>(loc, dstCtx->vecType, result).getResult(0);
        valueMap[histOp.getResult()] = result;
        return success();
    }

    if (auto reduceOp = dyn_cast<::tla::ReduceOp>(op))
        return lowerReduce(reduceOp, b, module, valueMap, fullMasks);

    if (auto rdbOp = dyn_cast<::tla::ReduceIntraBlockOp>(op))
        return lowerReduceIntraBlock(rdbOp, b, module, valueMap, fullMasks);

    if (auto pairOp = dyn_cast<::tla::PairWiseOp>(op))
        return lowerPairWise(pairOp, b, module, valueMap, fullMasks);

    if (auto packOp = dyn_cast<::tla::PackOp>(op))
        return lowerPack(packOp, b, module, valueMap);
    if (auto unpackOp = dyn_cast<::tla::UnpackOp>(op))
        return lowerUnpack(unpackOp, b, module, valueMap);

    if (auto addc = dyn_cast<::tla::AddCOp>(op))
        return lowerB32CarryOp(addc, b, module, valueMap, fullMasks, "tla_addc_i32", "tla_addc_u32");
    if (auto subc = dyn_cast<::tla::SubCOp>(op))
        return lowerB32CarryOp(subc, b, module, valueMap, fullMasks, "tla_subc_i32", "tla_subc_u32");
    if (auto mull = dyn_cast<::tla::MullOp>(op))
        return lowerMull(mull, b, valueMap);
    if (auto interleaveOp = dyn_cast<::tla::InterleaveOp>(op))
        return lowerInterleave(interleaveOp, b, module, valueMap, fullMasks);

    if (auto deInterleaveOp = dyn_cast<::tla::DeInterleaveOp>(op))
        return lowerDeInterleave(deInterleaveOp, b, module, valueMap, fullMasks);

    if (auto gatherOp = dyn_cast<::tla::GatherOp>(op))
        return lowerGather(gatherOp, b, module, valueMap, fullMasks);

    if (auto scatterOp = dyn_cast<::tla::ScatterOp>(op))
        return lowerScatter(scatterOp, b, module, valueMap, fullMasks);

    if (auto info = getVectorUnaryInfo(&op))
        return lowerVectorUnary(op, &*info, b, module, valueMap, fullMasks);

    // tla.create_mask: build a mask vector from a fixed pattern ->
    // ave.hir.pge<PATTERN>. The op's own dtype attr fixes the lane count
    // (256 bytes / element size) and hence the i1 mask width.
    if (auto maskOp = dyn_cast<::tla::CreateMaskOp>(op)) {
        auto pattern = hivmave::symbolizePgePattern(maskOp.getPattern());
        if (!pattern)
            return maskOp.emitError("unknown tla.create_mask pattern: ") << maskOp.getPattern(), failure();
        auto opCtx = deriveVecCtxForElement(maskOp.getDtype());
        if (failed(opCtx))
            return maskOp.emitError("unsupported tla.create_mask dtype: ") << maskOp.getDtype(), failure();
        if (opCtx->lanes == 32) {
            // PGE has no b64 mode. Select the single-register fraction in b32
            // first, then expand its low half to the native b64 predicate.
            if (*pattern == hivmave::PgePattern::H)
                pattern = hivmave::PgePattern::VL16;
            else if (*pattern == hivmave::PgePattern::Q)
                pattern = hivmave::PgePattern::VL8;
        }
        Value mask = createPredicatePge(b, loc, opCtx->maskVecType, opCtx->lanes, *pattern);
        if (opCtx->lanes == 32) {
            Value lower = b.create<arith::ConstantIntOp>(loc, 0, 32);
            mask = b.create<hivm_regbaseintrins::PunpackInstrOp>(loc, opCtx->maskVecType, mask, lower).getResult();
        }
        valueMap[maskOp.getResult()] = mask;
        return success();
    }

    if (auto updateMaskOp = dyn_cast<::tla::UpdateMaskOp>(op))
        return lowerUpdateMask(updateMaskOp, b, module, valueMap, fullMasks);

    if (auto generateMaskOp = dyn_cast<::tla::GenerateMaskOp>(op))
        return lowerGenerateMask(generateMaskOp, b, valueMap);

    if (auto cmpOp = dyn_cast<::tla::CmpOp>(op))
        return lowerCmp(cmpOp, b, module, valueMap, fullMasks);

    if (auto storeOp = dyn_cast<::tla::StoreOp>(op))
        return lowerStore(storeOp, b, module, valueMap, fullMasks);

    if (auto storeUpdateOp = dyn_cast<::tla::StoreUpdateOp>(op))
        return lowerStoreUpdate(storeUpdateOp, b, module, valueMap, fullMasks);

    if (auto forOp = dyn_cast<scf::ForOp>(op))
        return lowerScfFor(forOp, b, module, valueMap, fullMasks);

    if (auto ifOp = dyn_cast<scf::IfOp>(op))
        return lowerScfIf(ifOp, b, module, valueMap, fullMasks);

    // Index/scalar arithmetic (arith.*): clone with mapped operands. This covers
    // both the arithmetic feeding offsets and conditions, and the coord/origin
    // arithmetic (addi/subi/minsi) that tla-lower-tensor-desc emits for
    // dynamic-coord tile_views. Static coords fold to constants (createOrFold)
    // and are handled by the arith::ConstantOp arm above.
    if (op.getDialect()->getNamespace() == arith::ArithDialect::getDialectNamespace()) {
        IRMapping mapper;
        for (Value operand : op.getOperands()) {
            Value mapped = lookupOrCloneScalarValue(b, operand, valueMap);
            if (!mapped)
                return failure();
            mapper.map(operand, mapped);
        }
        Operation* cloned = b.clone(op, mapper);
        for (auto [oldResult, newResult] : llvm::zip(op.getResults(), cloned->getResults()))
            valueMap[oldResult] = newResult;
        return success();
    }

    // GM/UB scalar accesses lowered by tla-lower-scalar-access before outlining.
    if (auto memLoad = dyn_cast<mlir::memref::LoadOp>(op)) {
        Value mem = valueMap.lookup(memLoad.getMemRef());
        if (!mem)
            return failure();
        SmallVector<Value, 2> indices;
        for (Value idx : memLoad.getIndices()) {
            Value mapped = lookupOrCloneScalarValue(b, idx, valueMap);
            if (!mapped)
                return failure();
            indices.push_back(mapped);
        }
        valueMap[memLoad.getResult()] = b.create<mlir::memref::LoadOp>(loc, mem, indices).getResult();
        return success();
    }
    if (auto memStore = dyn_cast<mlir::memref::StoreOp>(op)) {
        Value mem = valueMap.lookup(memStore.getMemRef());
        Value val = valueMap.lookup(memStore.getValue());
        if (!val)
            val = lookupOrCloneScalarValue(b, memStore.getValue(), valueMap);
        if (!mem || !val)
            return failure();
        SmallVector<Value, 2> indices;
        for (Value idx : memStore.getIndices()) {
            Value mapped = lookupOrCloneScalarValue(b, idx, valueMap);
            if (!mapped)
                return failure();
            indices.push_back(mapped);
        }
        b.create<mlir::memref::StoreOp>(loc, val, mem, indices);
        return success();
    }

    // Rebuild the descriptor view used by a scalar access from the base memref
    // passed to the helper.
    if (isa<mlir::memref::ExtractStridedMetadataOp, mlir::memref::ReinterpretCastOp,
            mlir::memref::ExtractAlignedPointerAsIndexOp>(op)) {
        IRMapping mapper;
        for (Value operand : op.getOperands()) {
            Value mapped = lookupOrCloneScalarValue(b, operand, valueMap);
            if (!mapped)
                return failure();
            mapper.map(operand, mapped);
        }
        Operation* cloned = b.clone(op, mapper);
        for (auto [oldResult, newResult] : llvm::zip(op.getResults(), cloned->getResults()))
            valueMap[oldResult] = newResult;
        return success();
    }

    // Keep profiling boundaries at their original position inside the helper.
    // The late profiling pass, not outlining, materializes hardware controls.
    if (isa<::tla::ProfStartOp, ::tla::ProfEndOp>(op)) {
        b.clone(op);
        return success();
    }

    if (op.hasTrait<OpTrait::IsTerminator>())
        return success();

    return failure();
}

static LogicalResult lowerNestedVectorBlock(
    Block* sourceBlock, OpBuilder& b, ModuleOp module, DenseMap<Value, Value>& valueMap,
    AutoFullPredicateCache& fullMasks)
{
    for (Operation& op : sourceBlock->getOperations()) {
        // Terminators are reproduced by the enclosing op (scf.for/scf.if) or by
        // buildHelperFunc's func.return.
        if (op.hasTrait<OpTrait::IsTerminator>())
            continue;
        if (failed(lowerNestedVectorOp(op, b, module, valueMap, fullMasks))) {
            // Avoid emitOpError: printing a half-rewritten op can assert in AsmPrinter.
            return op.emitError("unsupported op while outlining tla.vec.func into vector helper: ") << op.getName(),
                   failure();
        }
    }
    return success();
}

static Value stripHelperAddressCasts(Value address)
{
    while (Operation* def = address.getDefiningOp()) {
        if (!isa<arith::IndexCastOp, arith::ExtSIOp, arith::TruncIOp>(def))
            break;
        address = def->getOperand(0);
    }
    return address;
}

static Value getScfLoopCarrierInit(Value value)
{
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        Operation* parent = argument.getOwner()->getParentOp();
        unsigned index = argument.getArgNumber();
        if (auto loop = dyn_cast<scf::ForOp>(parent); loop && index > 0 && index <= loop.getNumRegionIterArgs())
            return loop.getInitArgs()[index - 1];
        if (auto loop = dyn_cast<scf::WhileOp>(parent); loop && index < loop.getInits().size())
            return loop.getInits()[index];
        return {};
    }
    if (auto loop = value.getDefiningOp<scf::ForOp>())
        return loop.getInitArgs()[cast<OpResult>(value).getResultNumber()];
    if (auto loop = value.getDefiningOp<scf::WhileOp>()) {
        unsigned index = cast<OpResult>(value).getResultNumber();
        if (index < loop.getInits().size())
            return loop.getInits()[index];
    }
    return {};
}

// Use the loop's initial address for the helper argument. Inside the helper,
// tensor accesses still use the address carried by the loop.
static Value chaseScfCarrierInit(Value address)
{
    address = stripHelperAddressCasts(address);
    DenseSet<Value> visited;
    while (visited.insert(address).second) {
        Value init = getScfLoopCarrierInit(address);
        if (!init)
            break;
        address = stripHelperAddressCasts(init);
    }
    return address;
}

// Follow the address through views, UPDATE and other pointer wrappers. Leave
// pointer arithmetic alone because the same SSA address seeds the helper ABI.
static Value getVectorHelperStorageParent(Value source)
{
    if (auto desc = source.getDefiningOp<::tla::TensorDescOp>())
        return desc.getBase();
    if (auto make = source.getDefiningOp<::tla::MakeTensorOp>())
        return make.getPtr();
    if (auto make = source.getDefiningOp<::tla::MakeTensorLikeOp>())
        return make.getPtr();
    if (auto tile = source.getDefiningOp<::tla::TileViewOp>())
        return tile.getSource();
    if (auto ptr = source.getDefiningOp<::tla::TensorPtrOp>())
        return ptr.getSrc();
    if (auto ptr = source.getDefiningOp<::tla::TensorPtrWithCoordOp>())
        return ptr.getSrc();
    if (auto load = source.getDefiningOp<::tla::LoadUpdateOp>())
        return load.getSource();
    if (auto load = source.getDefiningOp<::tla::LoadUnalignUpdateOp>())
        return load.getSource();
    if (auto store = source.getDefiningOp<::tla::StoreUpdateOp>())
        return store.getDest();
    if (auto store = source.getDefiningOp<::tla::StoreUnalignUpdateOp>())
        return store.getDest();
    if (auto toInt = source.getDefiningOp<::tla::PtrToIntOp>())
        return toInt.getPtr();
    if (auto toInt = source.getDefiningOp<LLVM::PtrToIntOp>())
        return toInt.getArg();
    if (auto toPtr = source.getDefiningOp<::tla::IntToPtrOp>())
        return toPtr.getAddr();
    if (auto toPtr = source.getDefiningOp<LLVM::IntToPtrOp>())
        return toPtr.getArg();
    if (auto extract = source.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>())
        return extract.getSource();
    if (auto cast = source.getDefiningOp<mlir::memref::CastOp>())
        return cast.getSource();
    return {};
}

// For results selected inside the helper, deduplicate only when every branch
// leads to the same seed. Results from outside already name a live address.
static Value getVectorHelperStorageIdentity(Value source, Region& outlinedRegion)
{
    DenseSet<Value> visited;
    SmallVector<Value, 2> pending{source};
    Value identity;
    while (!pending.empty()) {
        Value value = stripHelperAddressCasts(pending.pop_back_val());
        if (!visited.insert(value).second)
            continue;
        if (outlinedRegion.isAncestor(value.getParentRegion())) {
            if (auto conditional = value.getDefiningOp<scf::IfOp>()) {
                unsigned index = cast<OpResult>(value).getResultNumber();
                pending.push_back(conditional.thenYield().getOperand(index));
                pending.push_back(conditional.elseYield().getOperand(index));
                continue;
            }
            if (Value init = getScfLoopCarrierInit(value)) {
                pending.push_back(init);
                continue;
            }
        }
        if (Value parent = getVectorHelperStorageParent(value)) {
            pending.push_back(parent);
            continue;
        }
        if (auto pointerCast = value.getDefiningOp<hivm::PointerCastOp>())
            value = stripHelperAddressCasts(pointerCast.getSingleAddr());
        if (identity && identity != value)
            return source;
        identity = value;
    }
    return identity ? identity : source;
}

static FailureOr<MemRefType> getVectorHelperOperandType(Value source)
{
    if (auto type = dyn_cast<MemRefType>(source.getType()))
        return type;
    return getVectorHelperArgMemrefType(source);
}

// Add `source` as a helper operand unless another operand already represents
// the same allocation with the same helper memref type. Record every source's
// representative so scalar views can reuse a tensor operand's helper argument.
static void addVectorHelperOperand(
    Value source, Region& outlinedRegion, SmallVectorImpl<Value>& operands, DenseMap<Value, Value>& aliases)
{
    Value identity = getVectorHelperStorageIdentity(source, outlinedRegion);
    FailureOr<MemRefType> sourceType = getVectorHelperOperandType(source);
    for (Value operand : operands) {
        if (identity != getVectorHelperStorageIdentity(operand, outlinedRegion))
            continue;
        FailureOr<MemRefType> operandType = getVectorHelperOperandType(operand);
        // Same allocation seed: reuse the representative even when ``source`` is
        // an UPDATE next-window tile whose raw next_ptr cannot type a helper
        // memref by itself (typed via the seed inttoptr instead).
        if (succeeded(operandType) && (failed(sourceType) || *sourceType == *operandType)) {
            aliases[source] = operand;
            return;
        }
    }
    operands.push_back(source);
    aliases[source] = source;
}

// Scalar access lowering builds extract_strided_metadata + reinterpret_cast
// inside vec.func. Pass the first memref defined outside the outlined region to
// the helper and rebuild that view inside the helper.
static FailureOr<Value> getScalarAccessHelperMemref(
    Value source, Region& outlinedRegion, Operation* callSite, DominanceInfo& dominanceInfo)
{
    while (Operation* def = source.getDefiningOp()) {
        Region* defRegion = source.getParentRegion();
        if (!defRegion || !outlinedRegion.isAncestor(defRegion))
            break;
        if (!isa<mlir::memref::ExtractStridedMetadataOp, mlir::memref::ReinterpretCastOp>(def))
            break;
        source = def->getOperand(0);
    }

    // A dynamic inttoptr descriptor can materialize its hivm.pointer_cast at the
    // scalar access insertion point, which may be inside vec.func. Such a value
    // cannot be passed to the helper call that replaces the enclosing vec.func.
    if (!dominanceInfo.dominates(source, callSite))
        return failure();
    return source;
}

// Collect, in body order, the unique base memrefs that tla.load/tla.store/
// tla.gather chunks and bridged scalar accesses reference. These become the
// helper's arguments.
static LogicalResult collectVectorHelperOperands(
    Block* block, Region& outlinedRegion, Operation* callSite, DominanceInfo& dominanceInfo,
    SmallVectorImpl<Value>& operands, DenseMap<Value, Value>& aliases)
{
    auto collectWindow = [&](Value window) {
        if (auto desc = window.getDefiningOp<::tla::TensorDescOp>()) {
            Value base = desc.getBase();
            if (auto ptr = base.getDefiningOp<::tla::IntToPtrOp>()) {
                // Local addresses are computed in the helper. Capture a seed
                // only if it is available at the call site.
                if (outlinedRegion.isAncestor(ptr.getAddr().getParentRegion()) &&
                    !dominanceInfo.dominates(getVectorHelperStorageIdentity(window, outlinedRegion), callSite))
                    return;
            }
        }
        addVectorHelperOperand(window, outlinedRegion, operands, aliases);
    };
    for (Operation& op : block->getOperations()) {
        if (auto loadOp = dyn_cast<::tla::LoadOp>(op)) {
            collectWindow(loadOp.getSource());
            continue;
        }
        if (auto loadUpdateOp = dyn_cast<::tla::LoadUpdateOp>(op)) {
            collectWindow(loadUpdateOp.getSource());
            continue;
        }
        if (auto loadUnalignUpdateOp = dyn_cast<::tla::LoadUnalignUpdateOp>(op)) {
            collectWindow(loadUnalignUpdateOp.getSource());
            continue;
        }
        if (auto preOp = dyn_cast<::tla::LoadUnalignPreOp>(op)) {
            collectWindow(preOp.getSource());
            continue;
        }
        if (auto storeOp = dyn_cast<::tla::StoreOp>(op)) {
            collectWindow(storeOp.getDest());
            continue;
        }
        if (auto storeUpdateOp = dyn_cast<::tla::StoreUpdateOp>(op)) {
            collectWindow(storeUpdateOp.getDest());
            continue;
        }
        if (auto storeUnalignUpdateOp = dyn_cast<::tla::StoreUnalignUpdateOp>(op)) {
            collectWindow(storeUnalignUpdateOp.getDest());
            continue;
        }
        if (auto postOp = dyn_cast<::tla::StoreUnalignPostOp>(op)) {
            if (isa<::tla::TlaTensorType>(postOp.getDest().getType()))
                collectWindow(postOp.getDest());
            continue;
        }
        if (auto printOp = dyn_cast<::tla::PrintTensorOp>(op)) {
            collectWindow(printOp.getValue());
            continue;
        }
        if (auto gatherOp = dyn_cast<::tla::GatherOp>(op)) {
            // Only UB-tensor gather needs a helper memref argument.
            if (isa<::tla::TlaTensorType>(gatherOp.getX().getType()))
                collectWindow(gatherOp.getX());
            continue;
        }
        if (auto scatterOp = dyn_cast<::tla::ScatterOp>(op)) {
            collectWindow(scatterOp.getDst());
            continue;
        }
        // A memref-backed tensor.ptr in an extern argument reads the aligned address.
        if (auto addr = dyn_cast<mlir::memref::ExtractAlignedPointerAsIndexOp>(op)) {
            auto source = getScalarAccessHelperMemref(addr.getSource(), outlinedRegion, callSite, dominanceInfo);
            if (failed(source))
                return failure();
            addVectorHelperOperand(*source, outlinedRegion, operands, aliases);
            continue;
        }
        if (auto memLoad = dyn_cast<mlir::memref::LoadOp>(op)) {
            FailureOr<Value> source =
                getScalarAccessHelperMemref(memLoad.getMemRef(), outlinedRegion, callSite, dominanceInfo);
            if (failed(source))
                return failure();
            addVectorHelperOperand(*source, outlinedRegion, operands, aliases);
            continue;
        }
        if (auto memStore = dyn_cast<mlir::memref::StoreOp>(op)) {
            FailureOr<Value> source =
                getScalarAccessHelperMemref(memStore.getMemRef(), outlinedRegion, callSite, dominanceInfo);
            if (failed(source))
                return failure();
            addVectorHelperOperand(*source, outlinedRegion, operands, aliases);
            continue;
        }
        for (Region& region : op.getRegions())
            for (Block& nested : region)
                if (failed(collectVectorHelperOperands(
                        &nested, outlinedRegion, callSite, dominanceInfo, operands, aliases)))
                    return failure();
    }
    return success();
}

// Collect unique scalar values used inside the region but defined outside it
// (e.g. a sub_block_idx/block_idx computed at the top of the kernel, or a
// vector-scalar RHS constant). Passing them into the helper avoids cloning float
// constants into vector helpers where vector.broadcast can fold to illegal
// vector arith.constant ops before the HIVMAVE conversion pipeline.
// They are passed in as trailing scalar arguments rather than recomputed inside
// the outlined vector function.
static void collectVectorHelperScalarOperands(::tla::VecFuncOp vecFuncOp, SmallVectorImpl<Value>& scalars)
{
    vecFuncOp.walk([&](Operation* op) {
        for (Value operand : op->getOperands()) {
            Type operandType = operand.getType();
            if (!operandType.isIntOrIndex() && !isa<FloatType>(operandType))
                continue;
            // Index/integer constants defined outside vec.func are cloned inline by
            // lookupOrCloneScalarValue rather than passed as helper args. This keeps
            // static tile offsets/strides (now tensor_desc operands, materialized at
            // function scope by tla-lower-tensor-desc) as constants inside the helper
            // instead of bloating the helper signature. Float constants are still
            // passed as args (see the comment above re: vector.broadcast folding).
            if (operandType.isIntOrIndex() && operand.getDefiningOp<arith::ConstantOp>())
                continue;
            Region* defRegion = operand.getParentRegion();
            if (defRegion && !vecFuncOp.getBody().isAncestor(defRegion) && !llvm::is_contained(scalars, operand))
                scalars.push_back(operand);
        }
    });
    // The walk above visits tensor_descs written inside vec.func. A descriptor
    // produced in the parent scope and merely *used* here keeps its offset and
    // stride operands on an op the walk never reaches, yet buildHelperFunc has to
    // materialize that descriptor inside the helper. Capture those operands too,
    // under the same rule: constants are cloned inline, and anything else becomes
    // a trailing helper argument.
    SmallVector<::tla::TensorDescOp> externalDescs;
    collectExternalTensorDescs(vecFuncOp, externalDescs);
    for (::tla::TensorDescOp desc : externalDescs) {
        for (Value operand : desc->getOperands()) {
            Type operandType = operand.getType();
            if (!operandType.isIntOrIndex() && !isa<FloatType>(operandType))
                continue;
            if (operandType.isIntOrIndex() && operand.getDefiningOp<arith::ConstantOp>())
                continue;
            if (!llvm::is_contained(scalars, operand))
                scalars.push_back(operand);
        }
    }
}

// Build a vector_region helper for a tla.vec.func body. The helper receives one
// flat on-chip memref per referenced tensor (or a concrete memref for scalar
// accesses); the for/if control flow is carried inside the helper, where each
// tla.load/store is lowered to an AVE vload/masked-store over a 256-byte tile
// carved from the base address at the per-iteration offset.
static FailureOr<func::FuncOp> buildHelperFunc(
    ModuleOp module, func::FuncOp parentFunc, ::tla::VecFuncOp vecFuncOp, ArrayRef<Value> helperOperands,
    const DenseMap<Value, Value>& helperOperandAliases, ArrayRef<Value> scalarOperands, int& nextVectorRegionId,
    DenseMap<Value, Value>& loweredMemrefByValue)
{
    MLIRContext* ctx = module.getContext();
    Operation* vectorOp = vecFuncOp.getOperation();
    OpBuilder moduleBuilder(module.getBodyRegion());
    moduleBuilder.setInsertionPointAfter(parentFunc);

    Block* body = vecFuncOp.getBody().empty() ? nullptr : &vecFuncOp.getBody().front();
    if (!body)
        return vecFuncOp.emitOpError("vector helper needs a body"), failure();

    SmallVector<Type> functionInputs;
    functionInputs.reserve(helperOperands.size());
    for (Value operand : helperOperands) {
        // Already-bridged scalar-access memrefs keep their concrete type.
        if (auto mt = dyn_cast<MemRefType>(operand.getType())) {
            functionInputs.push_back(mt);
            continue;
        }
        auto operandType = getVectorHelperArgMemrefType(operand);
        if (failed(operandType))
            return vecFuncOp.emitOpError("failed to type helper memref operand"), failure();
        functionInputs.push_back(*operandType);
    }
    // Trailing scalar args: scalars captured from outside the region.
    for (Value scalar : scalarOperands)
        functionInputs.push_back(scalar.getType());
    auto funcType = moduleBuilder.getFunctionType(functionInputs, TypeRange{});

    // The per-iteration vector tile is one 256-byte register's worth of elements.
    // Each op inside the helper derives its own width from its operands/result,
    // so tiles may carry different element types within one region (e.g. a f32
    // load feeding a tla.cast to f16). Validate only that each tile operand is a
    // supported int/float type (the trailing scalar args are index/int and are
    // handled separately). This runs before the helper is created so a validation
    // failure leaks no partial IR.
    for (size_t i = 0; i < helperOperands.size(); ++i) {
        Type tileElementType = cast<MemRefType>(functionInputs[i]).getElementType();
        if (!isa<IntegerType>(tileElementType) && !isa<FloatType>(tileElementType))
            return vectorOp->emitError("unsupported element type for vector binary helper: ") << tileElementType;
        if (failed(getVectorLaneCount(tileElementType)))
            return vectorOp->emitError("unsupported element width for vector helper tile: ") << tileElementType;
    }

    std::string helperName = buildUniqueVectorHelperName(module, nextVectorRegionId);
    auto helper = moduleBuilder.create<func::FuncOp>(vectorOp->getLoc(), helperName, funcType);
    helper.setPrivate();
    helper->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
    helper->setAttr("hivm.vector_function", UnitAttr::get(ctx));
    helper->setAttr("no_inline", UnitAttr::get(ctx));
    // Do not stamp llvm.noalias on helper memref args: bump-plan disjointness
    // is a performance assumption without independent alias proof, and must
    // not be a correctness dependency. Same-base in-place load/store already
    // share one argument via helperOperandAliases.

    Block* entry = helper.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockBegin(entry);

    DenseMap<Value, Value> valueMap;
    for (auto [i, operand] : llvm::enumerate(helperOperands))
        valueMap[operand] = entry->getArgument(i);
    for (auto [source, representative] : helperOperandAliases)
        if (Value argument = valueMap.lookup(representative))
            valueMap[source] = argument;
    // Captured scalars map to their trailing block arguments.
    for (auto [j, scalar] : llvm::enumerate(scalarOperands))
        valueMap[scalar] = entry->getArgument(helperOperands.size() + j);
    // helperOperands holds one representative tensor_desc per unique base (see
    // addVectorHelperOperand). Every other vec.func-internal tensor_desc sharing
    // that base must resolve to the same helper memref arg so the TensorDescOp arm
    // in lowerNestedVectorOp can map it.
    DenseMap<Value, Value> baseToArg;
    DenseMap<Value, Value> identityToArg;
    Region& outlinedRegion = vecFuncOp.getBody();
    for (Value operand : helperOperands) {
        if (Value arg = valueMap.lookup(operand)) {
            identityToArg[getVectorHelperStorageIdentity(operand, outlinedRegion)] = arg;
            if (auto d = operand.getDefiningOp<::tla::TensorDescOp>())
                baseToArg[d.getBase()] = arg;
        }
    }
    vecFuncOp.walk([&](::tla::TensorDescOp desc) {
        if (valueMap.count(desc.getResult()))
            return;
        if (auto it = baseToArg.find(desc.getBase()); it != baseToArg.end()) {
            valueMap[desc.getResult()] = it->second;
            return;
        }
        // Distinct ``inttoptr(scf_i64)`` ops share chased seed identity.
        if (auto it = identityToArg.find(getVectorHelperStorageIdentity(desc.getResult(), outlinedRegion));
            it != identityToArg.end())
            valueMap[desc.getResult()] = it->second;
    });
    // A descriptor produced *outside* the region and used inside it needs the
    // same treatment as an internal one. lowerNestedVectorBlock never walks it
    // (the op lives in the parent), so without this its value would stay mapped
    // to the bare base-memref argument seeded above and every access would
    // silently address base + 0 -- the descriptor's offset lost.
    // addVectorHelperOperand keys operands by storage identity, so several
    // descriptors over one allocation collapse to a single argument and the
    // offset cannot come from the ABI; materialize it here instead, after
    // baseToArg has recorded the unoffset arguments.
    SmallVector<::tla::TensorDescOp> externalDescs;
    collectExternalTensorDescs(vecFuncOp, externalDescs);
    for (::tla::TensorDescOp desc : externalDescs) {
        // Only descriptors whose base reached the helper as an argument: the rest
        // are not addressable from in here and are not referenced by the body.
        if (!valueMap.count(desc.getResult()))
            continue;
        // A non-constant coord or stride cannot be reached from here (it is not a
        // helper argument), so this reports the unsupported descriptor instead of
        // falling back to offset 0.
        if (failed(materializeTensorDescSubview(desc, b, valueMap))) {
            vecFuncOp.emitOpError("failed to materialize external tensor_desc into vector helper");
            helper.erase();
            return failure();
        }
    }

    AutoFullPredicateCache fullMasks(&helper.getBody().front());
    if (failed(lowerNestedVectorBlock(body, b, module, valueMap, fullMasks))) {
        // Discard the partially-built helper so an unsupported construct fails
        // cleanly (the vec.func is left intact) instead of leaking malformed IR.
        vecFuncOp.emitOpError("failed to lower nested vector body into helper");
        helper.erase();
        return failure();
    }
    b.create<func::ReturnOp>(vectorOp->getLoc());
    return helper;
}

// tla.vec.func carries its execution model in a "mode" string attribute
// ("simd" / "simt", either case). SIMD is the historical default and is assumed
// when the attribute is absent.

class LowerVecFuncRegionPattern : public OpRewritePattern<::tla::VecFuncOp> {
public:
    LowerVecFuncRegionPattern(
        MLIRContext* context, ModuleOp module, int& nextVectorRegionId, DenseMap<Value, Value>& loweredMemrefByValue,
        bool& invalidScalarAccessBase)
        : OpRewritePattern<::tla::VecFuncOp>(context, /*benefit=*/2),
          module(module),
          nextVectorRegionId(nextVectorRegionId),
          loweredMemrefByValue(loweredMemrefByValue),
          invalidScalarAccessBase(invalidScalarAccessBase)
    {}

    LogicalResult matchAndRewrite(::tla::VecFuncOp vecFuncOp, PatternRewriter& rewriter) const override
    {
        // SIMD is the only mode this outlining path models: it maps the region onto
        // whole-vector AVE instructions. SIMT regions describe per-thread scalar
        // work and are lowered elsewhere.
        if (!isSimdVecFunc(vecFuncOp))
            return rewriter.notifyMatchFailure(vecFuncOp, "not a SIMD tla.vec.func");

        auto* body = vecFuncOp.getBody().empty() ? nullptr : &vecFuncOp.getBody().front();
        if (!body)
            return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.vec.func body");

        // Collect the load / binary compute / store ops (used for arg dedup and
        // graph validation); the helper builder walks the region itself to carry
        // the control flow structure.
        SmallVector<::tla::LoadOp, 4> loads;
        SmallVector<::tla::LoadUpdateOp, 4> loadUpdates;
        SmallVector<::tla::FullOp, 4> fulls;
        SmallVector<::tla::CreateMaskOp, 4> createMasks;
        SmallVector<::tla::UpdateMaskOp, 4> updateMasks;
        SmallVector<::tla::ArangeOp, 4> aranges;
        SmallVector<Operation*, 4> computeOps;
        SmallVector<::tla::StoreOp, 2> stores;
        SmallVector<::tla::ScatterOp, 2> scatters;
        SmallVector<::tla::StoreUpdateOp, 2> storeUpdates;
        SmallVector<Value, 4> unalignLoads;
        int unalignStores = 0;
        bool hasCompressedStore = false;
        bool hasDebugPrint = false;
        bool hasExtern = false;
        vecFuncOp->walk([&](Operation* op) {
            hasDebugPrint |= isa<::tla::DebugPrintOp, ::tla::PrintVectorOp, ::tla::PrintTensorOp>(op);
            if (isa<::tla::CallExternOp>(op)) {
                hasExtern = true;
            } else if (auto load = dyn_cast<::tla::LoadOp>(op)) {
                loads.push_back(load);
            } else if (auto loadUpdate = dyn_cast<::tla::LoadUpdateOp>(op)) {
                loadUpdates.push_back(loadUpdate);
            } else if (auto loadUnalignUpdate = dyn_cast<::tla::LoadUnalignUpdateOp>(op)) {
                unalignLoads.push_back(loadUnalignUpdate.getResult());
                if (Value second = loadUnalignUpdate.getResult2())
                    unalignLoads.push_back(second);
            } else if (auto full = dyn_cast<::tla::FullOp>(op)) {
                fulls.push_back(full);
            } else if (auto createMask = dyn_cast<::tla::CreateMaskOp>(op)) {
                createMasks.push_back(createMask);
            } else if (auto updateMask = dyn_cast<::tla::UpdateMaskOp>(op)) {
                updateMasks.push_back(updateMask);
            } else if (auto arange = dyn_cast<::tla::ArangeOp>(op)) {
                aranges.push_back(arange);
            } else if (auto scatter = dyn_cast<::tla::ScatterOp>(op)) {
                scatters.push_back(scatter);
            } else if (auto store = dyn_cast<::tla::StoreOp>(op)) {
                stores.push_back(store);
            } else if (auto storeUpdate = dyn_cast<::tla::StoreUpdateOp>(op)) {
                storeUpdates.push_back(storeUpdate);
            } else if (auto store = dyn_cast<::tla::StoreUnalignUpdateOp>(op)) {
                ++unalignStores;
                hasCompressedStore |= store.getCompressed();
            } else if (isVectorComputeOp(op)) {
                computeOps.push_back(op);
            }
            return WalkResult::advance();
        });
        if (hasDebugPrint && module->hasAttr("tla.ub_reserve_released"))
            return vecFuncOp.emitError("VF printing requires the default reserved UB");
        if (!hasExtern && stores.empty() && scatters.empty() && storeUpdates.empty() && unalignStores == 0 &&
            !hasDebugPrint) {
            // Scalar-only (or empty) VF cannot be outlined as a BiSheng helper — that
            // path requires tla.store or tla.store_update. If there is also no tile
            // load/compute, inline the body into the parent (same as tla.vector
            // flattening) so scalar_load/store + scf stay legal for later
            // convert-scf-to-cf.
            // tla.call_extern is opaque, it may have side effects, so if
            // there is no opaque extern or vector stores, flatten scalar-only
            // regions.
            if (!loads.empty() || !loadUpdates.empty() || !unalignLoads.empty() || !fulls.empty() ||
                !createMasks.empty() || !updateMasks.empty() || !aranges.empty() || !computeOps.empty())
                return rewriter.notifyMatchFailure(
                    vecFuncOp, "expected tla.vec.func body with a tla.store or tla.store_update");
            rewriter.inlineBlockBefore(body, vecFuncOp->getBlock(), vecFuncOp->getIterator());
            rewriter.eraseOp(vecFuncOp);
            return success();
        }

        // Validate the graph: every compute operand and store source must come from
        // a tla.load result or a prior compute result inside this region.
        DenseSet<Value> producedValues;
        vecFuncOp.walk([&](::tla::CallExternOp call) {
            producedValues.insert(call.getResults().begin(), call.getResults().end());
        });
        for (::tla::LoadOp load : loads) {
            producedValues.insert(load.getResult());
            if (Value result2 = load.getResult2())
                producedValues.insert(result2);
        }
        for (::tla::LoadUpdateOp loadUpdate : loadUpdates) {
            producedValues.insert(loadUpdate.getResult());
            if (Value second = loadUpdate.getResult2())
                producedValues.insert(second);
        }
        for (Value unalignLoad : unalignLoads)
            producedValues.insert(unalignLoad);
        for (::tla::FullOp full : fulls)
            producedValues.insert(full.getResult());
        for (::tla::CreateMaskOp createMask : createMasks)
            producedValues.insert(createMask.getResult());
        for (::tla::UpdateMaskOp updateMask : updateMasks)
            producedValues.insert(updateMask.getMask());
        for (::tla::ArangeOp arange : aranges)
            producedValues.insert(arange.getResult());
        auto isRegisterCarrier = [](Value value) {
            Type type = value.getType();
            return isa<::tla::VectorSSAType, ::tla::MaskSSAType>(type);
        };
        vecFuncOp.walk([&](scf::ForOp forOp) {
            for (BlockArgument arg : forOp.getRegionIterArgs())
                if (isRegisterCarrier(arg))
                    producedValues.insert(arg);
            for (Value result : forOp.getResults())
                if (isRegisterCarrier(result))
                    producedValues.insert(result);
        });
        vecFuncOp.walk([&](scf::IfOp ifOp) {
            for (Value result : ifOp.getResults())
                if (isRegisterCarrier(result))
                    producedValues.insert(result);
        });

        for (Operation* computeOp : computeOps) {
            if (isa<::tla::InterleaveOp, ::tla::DeInterleaveOp, ::tla::MullOp, ::tla::AddCOp, ::tla::SubCOp>(
                    computeOp)) {
                if (computeOp->getNumResults() != 2)
                    return rewriter.notifyMatchFailure(vecFuncOp, "unexpected two-result tla compute op shape");
            } else if (computeOp->getNumResults() != 1) {
                return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op shape");
            }
            if (auto anyInfo = getAnyVectorOperationInfo(computeOp)) {
                if (auto info = anyInfo->binary) {
                    // Vector operands must come from a load or prior compute op. A
                    // vector-scalar rhs may be captured or cloned into the helper.
                    TlaBinaryOperands ops = info->operands;
                    if (!ops.lhs || !ops.rhs || !producedValues.contains(ops.lhs))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp,
                            "expected binary op operand from load/create/update mask "
                            "or prior compute op");
                    if (info->rhsKind == VectorRhsKind::Vector && !producedValues.contains(ops.rhs))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp,
                            "expected binary op rhs from load/create/update mask "
                            "or prior compute op");
                    if (ops.mask && !producedValues.contains(ops.mask))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp,
                            "expected binary op mask from create/update mask, compare, "
                            "or prior mask compute op");
                } else if (auto unaryInfo = anyInfo->unary) {
                    TlaUnaryOperands ops = unaryInfo->operands;
                    if (!ops.operand || !producedValues.contains(ops.operand))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp,
                            "expected unary op operand from load/create/update mask "
                            "or prior compute op");
                    if (ops.mask && !producedValues.contains(ops.mask))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp,
                            "expected unary op mask from create/update mask, compare, "
                            "or prior mask compute op");
                } else {
                    return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op");
                }
            } else if (auto cmpOp = dyn_cast<::tla::CmpOp>(computeOp)) {
                if (!producedValues.contains(cmpOp.getLhs()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.cmp lhs from tla.load or prior compute op");
                if (isa<::tla::VectorSSAType>(cmpOp.getRhs().getType()) && !producedValues.contains(cmpOp.getRhs()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.cmp rhs from tla.load or prior compute op");
                if (cmpOp.getMask() && !producedValues.contains(cmpOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp,
                        "expected tla.cmp mask from create/update mask or "
                        "prior mask compute op");
            } else if (auto whereOp = dyn_cast<::tla::WhereOp>(computeOp)) {
                if (!producedValues.contains(whereOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp,
                        "expected tla.where mask from create/update mask, compare, "
                        "SCF carrier, or prior mask compute op");
                if (!producedValues.contains(whereOp.getX()) || !producedValues.contains(whereOp.getY()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.where operands from legal vector or mask producers");
            } else if (auto squeezeOp = dyn_cast<::tla::SqueezeOp>(computeOp)) {
                if (!producedValues.contains(squeezeOp.getSrc()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.squeeze src from tla.load or prior compute op");
                if (!producedValues.contains(squeezeOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp,
                        "expected tla.squeeze mask from create/update mask or "
                        "prior mask compute op");
            } else if (auto unsqueezeOp = dyn_cast<::tla::UnsqueezeOp>(computeOp)) {
                if (!producedValues.contains(unsqueezeOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp,
                        "expected tla.unsqueeze mask from create/update mask or "
                        "prior mask compute op");
            } else if (auto histOp = dyn_cast<::tla::HistogramsOp>(computeOp)) {
                if (!producedValues.contains(histOp.getSrc()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.histograms src from tla.load or prior compute op");
                if (!producedValues.contains(histOp.getDst()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.histograms dst from tla.full, tla.load, or prior compute op");
                if (histOp.getMask() && !producedValues.contains(histOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp,
                        "expected tla.histograms mask from create/update mask or "
                        "prior mask compute op");
            } else if (auto reduceOp = dyn_cast<::tla::ReduceOp>(computeOp)) {
                Value operand = reduceOp.getOperand();
                if (reduceOp.getMask() && !producedValues.contains(reduceOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.reduce mask from a legal mask producer");
                if (!producedValues.contains(operand))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.reduce operand from tla.load or prior compute op");
            } else if (auto reduceDataBlockOp = dyn_cast<::tla::ReduceIntraBlockOp>(computeOp)) {
                if (!producedValues.contains(reduceDataBlockOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.reduce_intra_block mask from a legal mask producer");
                if (!producedValues.contains(reduceDataBlockOp.getOperand()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.reduce_intra_block operand from tla.load or prior compute op");
            } else if (auto pairWiseOp = dyn_cast<::tla::PairWiseOp>(computeOp)) {
                if (pairWiseOp.getMask() && !producedValues.contains(pairWiseOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.pair_wise mask from a legal mask producer");
                if (!producedValues.contains(pairWiseOp.getOperand()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.pair_wise operand from tla.load or prior compute op");
            } else if (auto packOp = dyn_cast<::tla::PackOp>(computeOp)) {
                if (!producedValues.contains(packOp.getSrc()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.pack src from a legal register producer");
            } else if (auto unpackOp = dyn_cast<::tla::UnpackOp>(computeOp)) {
                if (!producedValues.contains(unpackOp.getSrc()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.unpack src from a legal register producer");
            } else if (isa<::tla::AddCOp, ::tla::SubCOp>(computeOp)) {
                for (Value input : computeOp->getOperands()) {
                    if (!producedValues.contains(input))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp, "expected carry op inputs from register producers");
                }
            } else if (auto mull = dyn_cast<::tla::MullOp>(computeOp)) {
                const Value inputs[] = {mull.getLhs(), mull.getRhs(), mull.getMask()};
                for (Value input : inputs) {
                    if (!producedValues.contains(input))
                        return rewriter.notifyMatchFailure(vecFuncOp, "expected Mull inputs from register producers");
                }
            } else if (auto interleaveOp = dyn_cast<::tla::InterleaveOp>(computeOp)) {
                if (!producedValues.contains(interleaveOp.getSrc0()) ||
                    !producedValues.contains(interleaveOp.getSrc1()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.interleave operands from tla.load or prior compute op");
            } else if (auto deInterleaveOp = dyn_cast<::tla::DeInterleaveOp>(computeOp)) {
                if (!producedValues.contains(deInterleaveOp.getSrc0()) ||
                    !producedValues.contains(deInterleaveOp.getSrc1()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.deinterleave operands from tla.load or prior compute op");
            } else if (auto generateMaskOp = dyn_cast<::tla::GenerateMaskOp>(computeOp)) {
                if (!producedValues.contains(generateMaskOp.getSrc()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.generate_mask src from tla.load or prior compute op");
            } else if (auto gatherOp = dyn_cast<::tla::GatherOp>(computeOp)) {
                if (isa<::tla::VectorSSAType>(gatherOp.getX().getType())) {
                    if (!producedValues.contains(gatherOp.getX()))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp, "expected tla.gather src from tla.load or prior compute op");
                }
                if (gatherOp.getMask() && !producedValues.contains(gatherOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.gather mask from a legal mask producer");
                if (!producedValues.contains(gatherOp.getY()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.gather y operand from tla.load or prior compute op");
            } else if (auto bitCastOp = dyn_cast<::tla::BitCastOp>(computeOp)) {
                if (!producedValues.contains(bitCastOp.getSource()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.bitcast source from tla.load or prior compute op");
            } else if (auto castOp = dyn_cast<::tla::CastOp>(computeOp)) {
                if (castOp.getMask() && !producedValues.contains(castOp.getMask()))
                    return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.cast mask from a legal mask producer");
                if (!producedValues.contains(castOp.getSource()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected tla.cast source from tla.load or prior compute op");
            } else if (isa<::tla::AxpyOp, ::tla::SubAbsOp, ::tla::MulsCastOp>(computeOp)) {
                for (Value input : computeOp->getOperands()) {
                    if (isRegisterCarrier(input) && !producedValues.contains(input))
                        return rewriter.notifyMatchFailure(
                            vecFuncOp, "expected register operands from prior register producers");
                }
            } else if (auto inplaceMulAddOp = dyn_cast<::tla::InplaceMulAddOp>(computeOp)) {
                for (OpOperand& operand : inplaceMulAddOp->getOpOperands())
                    if (!producedValues.contains(operand.get()))
                        return inplaceMulAddOp.emitError("inplace_mul_add operand #")
                               << operand.getOperandNumber()
                               << " requires a vector or mask produced inside the SIMD region";
            } else if (auto expSubOp = dyn_cast<::tla::ExpSubOp>(computeOp)) {
                if (!producedValues.contains(expSubOp.getLhs()) || !producedValues.contains(expSubOp.getRhs()) ||
                    !producedValues.contains(expSubOp.getMask()))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "expected mixed ExpSub inputs and half mask from prior vector ops");
            } else {
                return rewriter.notifyMatchFailure(vecFuncOp, "unexpected tla compute op");
            }
            for (Value result : computeOp->getResults())
                producedValues.insert(result);
        }
        for (::tla::ScatterOp scatter : scatters) {
            if (!producedValues.contains(scatter.getSrc()) || !producedValues.contains(scatter.getIndices()))
                return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.scatter registers from a legal producer");
            if (scatter.getMask() && !producedValues.contains(scatter.getMask()))
                return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.scatter mask from a legal mask producer");
        }
        for (::tla::StoreOp store : stores) {
            if (store.getMask() && !producedValues.contains(store.getMask()))
                return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.store mask from a legal mask producer");
            if (!producedValues.contains(store.getSource()))
                return rewriter.notifyMatchFailure(vecFuncOp, "expected tla.store source from tla.load or compute op");
        }
        for (::tla::StoreUpdateOp storeUpdate : storeUpdates) {
            if (Value second = storeUpdate.getSource2(); second && !producedValues.contains(second))
                return rewriter.notifyMatchFailure(
                    vecFuncOp, "expected tla.store_update second source from tla.load or compute op");
            if (storeUpdate.getMask() && !producedValues.contains(storeUpdate.getMask()))
                return rewriter.notifyMatchFailure(
                    vecFuncOp, "expected tla.store_update mask from a legal mask producer");
            if (!producedValues.contains(storeUpdate.getSource()))
                return rewriter.notifyMatchFailure(
                    vecFuncOp, "expected tla.store_update source from tla.load_update or compute op");
        }

        auto funcOp = vecFuncOp->getParentOfType<func::FuncOp>();
        if (!funcOp)
            return rewriter.notifyMatchFailure(vecFuncOp, "expected enclosing func.func");

        // The helper takes one flat on-chip memref per referenced tensor, in body
        // order. Its extent is static when known; address-backed dynamic cases use
        // a rank-1 dynamic memref whose descriptor size is an explicit zero
        // sentinel. The helper must only consume that value as an address-backed
        // source for fixed-size reinterpret_cast tiles, never as capacity metadata.
        // Compute the operand list once and use it for both the helper signature
        // and the call.
        SmallVector<Value> helperOperands;
        DenseMap<Value, Value> helperOperandAliases;
        DominanceInfo dominanceInfo(funcOp);
        if (failed(collectVectorHelperOperands(
                body, vecFuncOp.getBody(), vecFuncOp, dominanceInfo, helperOperands, helperOperandAliases))) {
            if (!invalidScalarAccessBase)
                vecFuncOp.emitOpError(
                    "cannot outline scalar access because its base memref does not "
                    "dominate the vector helper call site; materialize dynamic "
                    "pointer-backed storage outside tla.vec.func");
            invalidScalarAccessBase = true;
            return failure();
        }
        if (helperOperands.empty() && !hasDebugPrint && !hasExtern && !hasCompressedStore)
            return rewriter.notifyMatchFailure(vecFuncOp, "expected vector region tensor operands");
        // Scalars captured from outside the region (e.g. a sub_block_idx computed at
        // the top of the kernel) are passed as trailing scalar arguments.
        SmallVector<Value> scalarOperands;
        collectVectorHelperScalarOperands(vecFuncOp, scalarOperands);

        // Call-site ``hivm.pointer_cast`` consumes the inttoptr *address* SSA. If
        // that address is defined inside ``tla.vec.func`` (e.g. lowered
        // ``tla.ptr_add`` as ``addi``), materializing the helper operand would
        // create an external use and ``eraseOp(vecFuncOp)`` would assert. Reject
        // the outline with a match failure so later legalization reports a
        // normal diagnostic (same shape as pre-PR), instead of Abort.
        for (Value tensor : helperOperands) {
            if (isa<MemRefType>(tensor.getType()))
                continue;
            auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>();
            if (!descOp || !llvm::isa<::tla::PtrType>(descOp.getBase().getType()))
                continue;
            Value ptr = descOp.getBase();
            auto itp = ptr.getDefiningOp<::tla::IntToPtrOp>();
            if (!itp)
                return rewriter.notifyMatchFailure(vecFuncOp, "expected pointer lowered to tla.inttoptr boundary");
            // Prefer storage-identity seed (AllocPtr / outside ptr_add): SCF +
            // UPDATE next_ptr chase. Fall back to the inttoptr addr only when
            // identity itself is still an address SSA.
            Value identity = getVectorHelperStorageIdentity(tensor, vecFuncOp.getBody());
            Value callAddr = identity;
            if (!dominanceInfo.dominates(callAddr, vecFuncOp.getOperation())) {
                callAddr = stripHelperAddressCasts(chaseScfCarrierInit(itp.getAddr()));
                // Peel in-region ptrtoint of a live next_ptr back toward seed.
                DenseSet<Value> peelVisited;
                while (peelVisited.insert(callAddr).second &&
                       !dominanceInfo.dominates(callAddr, vecFuncOp.getOperation())) {
                    if (auto p2i = callAddr.getDefiningOp<::tla::PtrToIntOp>()) {
                        callAddr = getVectorHelperStorageIdentity(p2i.getPtr(), vecFuncOp.getBody());
                        continue;
                    }
                    if (auto p2i = callAddr.getDefiningOp<LLVM::PtrToIntOp>()) {
                        callAddr = getVectorHelperStorageIdentity(p2i.getArg(), vecFuncOp.getBody());
                        continue;
                    }
                    break;
                }
            }
            if (!dominanceInfo.dominates(callAddr, vecFuncOp.getOperation()))
                return rewriter.notifyMatchFailure(
                    vecFuncOp,
                    "cannot outline address-backed tile whose base address is defined "
                    "inside tla.vec.func");
        }

        auto helperOr = buildHelperFunc(
            module, funcOp, vecFuncOp, helperOperands, helperOperandAliases, scalarOperands, nextVectorRegionId,
            loweredMemrefByValue);
        if (failed(helperOr))
            return vecFuncOp.emitOpError("failed to build vector helper function"), failure();
        auto helper = *helperOr;

        // The for/if control flow lives inside the helper, so this is a single
        // call passing the helper memrefs that replaces the whole vec.func region.
        rewriter.setInsertionPoint(vecFuncOp);
        SmallVector<Value, 8> callOperands;
        callOperands.reserve(helperOperands.size());
        Value unknownExtent;
        for (Value tensor : helperOperands) {
            // Bridged scalar-access memref operands are passed as-is.
            if (isa<MemRefType>(tensor.getType())) {
                callOperands.push_back(tensor);
                continue;
            }
            auto type = getVectorHelperArgMemrefType(tensor);
            if (failed(type))
                return rewriter.notifyMatchFailure(vecFuncOp, "failed to type on-chip memref for vector helper call");
            // Materialize address-backed tla.tensor_desc operands at the call site.
            // tla-lower-tensor-desc is the sole descriptor producer, so every helper
            // operand here is a tensor_desc (raw memrefs were passed through above);
            // materialize its inttoptr base as a rank-1 helper arg when ptr-backed.
            Value ptr;
            if (auto descOp = tensor.getDefiningOp<::tla::TensorDescOp>()) {
                if (llvm::isa<::tla::PtrType>(descOp.getBase().getType()))
                    ptr = descOp.getBase();
            }
            FailureOr<Value> base = failure();
            if (ptr) {
                // Helper call ABI: in-loop ``tensor_desc(inttoptr(scf_i64))``
                // materializes from the seed init (dominates the call). Live
                // iter_arg addresses are rebound inside the helper via
                // materializeTensorDescParameteri / resolveEffectiveUbPtr.
                if (auto itp = ptr.getDefiningOp<::tla::IntToPtrOp>()) {
                    Value liveAddr = stripHelperAddressCasts(itp.getAddr());
                    Value seedAddr = liveAddr;
                    if (!dominanceInfo.dominates(liveAddr, vecFuncOp.getOperation()))
                        seedAddr = chaseScfCarrierInit(liveAddr);
                    // Once→for→once UnAlign: live addr may be in-region ptrtoint of
                    // next_ptr; peel to a dominating seed (same as outline check).
                    if (!dominanceInfo.dominates(seedAddr, vecFuncOp.getOperation())) {
                        Value identity = getVectorHelperStorageIdentity(tensor, vecFuncOp.getBody());
                        if (dominanceInfo.dominates(identity, vecFuncOp.getOperation())) {
                            OpBuilder::InsertionGuard guard(rewriter);
                            rewriter.setInsertionPoint(vecFuncOp);
                            if (identity.getType().isInteger(64) || isa<IndexType>(identity.getType()))
                                seedAddr = identity;
                            else if (isa<::tla::PtrType>(identity.getType()))
                                seedAddr = rewriter.create<::tla::PtrToIntOp>(
                                    vecFuncOp.getLoc(), rewriter.getI64Type(), identity);
                            else if (isa<LLVM::LLVMPointerType>(identity.getType()))
                                seedAddr = rewriter.create<LLVM::PtrToIntOp>(
                                    vecFuncOp.getLoc(), rewriter.getI64Type(), identity);
                            else if (isa<MemRefType>(identity.getType())) {
                                Location loc = vecFuncOp.getLoc();
                                Value address =
                                    rewriter.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, identity);
                                seedAddr = ::tla::castValueToI64(rewriter, loc, address);
                            }
                        }
                    }
                    if (seedAddr != liveAddr) {
                        // Seed is the dominating i64 carrier init; rebuild inttoptr
                        // at the call site (addr SSA is never itself an inttoptr).
                        OpBuilder::InsertionGuard guard(rewriter);
                        rewriter.setInsertionPoint(vecFuncOp);
                        ptr =
                            rewriter.create<::tla::IntToPtrOp>(vecFuncOp.getLoc(), itp.getResult().getType(), seedAddr);
                    }
                }
                if (!ptr.getDefiningOp<::tla::IntToPtrOp>())
                    return rewriter.notifyMatchFailure(vecFuncOp, "expected pointer lowered to tla.inttoptr boundary");
                SmallVector<Value, 1> dynamicSizes;
                if (!type->hasStaticShape()) {
                    if (type->getRank() != 1 || type->getDimSize(0) != ShapedType::kDynamic)
                        return rewriter.notifyMatchFailure(
                            vecFuncOp, "expected flattened rank-1 dynamic vector helper memref");
                    if (!unknownExtent)
                        unknownExtent = rewriter.create<arith::ConstantIndexOp>(vecFuncOp.getLoc(), 0);
                    dynamicSizes.push_back(unknownExtent);
                }
                base = materializePtrValueAsMemref(
                    rewriter, vecFuncOp.getLoc(), ptr, *type, vecFuncOp.getOperation(), dynamicSizes);
                if (failed(base))
                    return rewriter.notifyMatchFailure(
                        vecFuncOp, "failed to materialize address-backed vector helper operand");
            } else {
                base = materializeBaseMemref(
                    rewriter, vecFuncOp.getLoc(), tensor,
                    /*loweredMemrefByValue=*/nullptr);
                if (failed(base))
                    return rewriter.notifyMatchFailure(vecFuncOp, "failed to materialize vector helper memref");
            }
            auto arg = castMemrefToExpected(rewriter, vecFuncOp.getLoc(), *base, *type);
            if (failed(arg))
                return rewriter.notifyMatchFailure(vecFuncOp, "failed to cast helper operand to expected memref type");
            callOperands.push_back(*arg);
        }
        // Captured scalars are defined in the parent (before this region), so they
        // dominate the call — pass them directly as trailing call operands.
        for (Value scalar : scalarOperands)
            callOperands.push_back(scalar);

        if (hasDebugPrint) {
            Value workspace = getOrAppendDebugPrintWorkspaceArg(funcOp);
            emitVfDebugBoundary(rewriter, vecFuncOp.getLoc(), module, workspace, true);
            module->setAttr("tla.vf_debug", UnitAttr::get(rewriter.getContext()));
        }
        auto call = rewriter.create<func::CallOp>(vecFuncOp.getLoc(), helper, callOperands);
        call->setAttr("hivm.vector_function", UnitAttr::get(rewriter.getContext()));
        call->setAttr("no_inline", UnitAttr::get(rewriter.getContext()));
        if (hasDebugPrint)
            emitVfDebugBoundary(rewriter, vecFuncOp.getLoc(), module, Value{}, false);
        rewriter.eraseOp(vecFuncOp);
        return success();
    }

private:
    ModuleOp module;
    int& nextVectorRegionId;
    DenseMap<Value, Value>& loweredMemrefByValue;
    bool& invalidScalarAccessBase;
};

} // namespace

void populateSimdVectorPatterns(
    RewritePatternSet& patterns, ModuleOp module, int& nextVectorRegionId, DenseMap<Value, Value>& loweredMemrefByValue,
    bool& invalidScalarAccessBase)
{
    MLIRContext* ctx = patterns.getContext();
    patterns.add<LowerVecFuncRegionPattern>(
        ctx, module, nextVectorRegionId, loweredMemrefByValue, invalidScalarAccessBase);
    populateTlaCopyPatterns(patterns, loweredMemrefByValue);
}

namespace {

// Neither vector pass has a "skip functions I own nothing in" gate, and that is
// not an oversight in either direction: no such gate can exist as the pipeline
// stands.
//
// mlir::applyPatternsGreedily does not only apply patterns -- it folds and CSEs
// everything it walks. So the *driver*, not the pattern set, performs part of the
// lowering, and "run the pass but match nothing" is not a no-op. Gating this pass
// on its own ops (vec.func(simd) / tla.copy / tla.vector) leaves 7 of the 82
// end-to-end examples textually different; tests/lit/scalar_ops/sqrt_kernel_body
// has no vector work at all and still differs, purely by constant order. There is
// no predicate that makes either pass skippable, because what they contribute to
// those functions is not vector lowering.
//
// Both passes therefore run unconditionally over every AIV/MIX function, which
// only composes if neither reorders the other's constant pool. Both disable
// constant CSE (GreedyRewriteConfig::cseConstants) and leave constant dedup to
// the pipeline's later CSEPass, which is where it belongs.
//
// Measured against the single combined pass, over all 82 end-to-end examples:
// one .o differs, by 2 bytes, in basic_mixed_store_zNUnAlign. With the SIMT
// driver no longer gated it folds literal-zero tile offsets before this pass
// outlines the vector region, so vector_region_0 is captured with 8 operands
// instead of 12 and four `add i64 0, x` identities disappear -- provably
// semantics-preserving, and smaller. 10 .ll and 135 .mlir differ by constant
// order and the SSA renumbering cascading from it.
//
// What matters for mode separation is that the *patterns* are mode-guarded, and
// they are.

/// Lower the SIMD half: a `tla.vec.func(mode="simd")` region maps onto
/// whole-vector AVE instructions. Also owns tla.copy lowering, which shares this
/// pass's base-memref cache, and the end-of-lowering cleanup that must run after
/// both halves.
///
/// Runs after tla-vector-region-simt.
class TlaVectorRegionSimdPass : public PassWrapper<TlaVectorRegionSimdPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaVectorRegionSimdPass)

    StringRef getArgument() const override
    {
        return "tla-vector-region-simd";
    }
    StringRef getName() const override
    {
        return "TlaVectorRegionSimdPass";
    }
    StringRef getDescription() const override
    {
        return "Outline tla.vec.func(mode=simd) regions and lower fragment ops to vector IR.";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            arith::ArithDialect, func::FuncDialect, mlir::memref::MemRefDialect, hivm::HIVMDialect, hivmave::AVEDialect,
            vector::VectorDialect, hivm_regbaseintrins::HIVMRegbaseIntrinsDialect, LLVM::LLVMDialect, math::MathDialect,
            ::tla::TlaDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        nextVectorRegionId = 0;

        // Extern signatures were checked in TLA types before ptr lowering.
        // Convert declarations here, e.g. !tla.vector<128xf16> to vector<128xf16>.
        for (func::FuncOp func : module.getOps<func::FuncOp>()) {
            if (!func.isDeclaration())
                continue;
            SmallVector<Type> inputs, results;
            for (Type type : func.getArgumentTypes()) {
                auto lowered = lowerSCFCarrierType(type);
                if (failed(lowered)) {
                    func.emitOpError("cannot lower extern argument type ") << type;
                    signalPassFailure();
                    return;
                }
                inputs.push_back(*lowered);
            }
            for (Type type : func.getResultTypes()) {
                auto lowered = lowerSCFCarrierType(type);
                if (failed(lowered)) {
                    func.emitOpError("cannot lower extern result type ") << type;
                    signalPassFailure();
                    return;
                }
                results.push_back(*lowered);
            }
            func.setType(FunctionType::get(&getContext(), inputs, results));
        }

        // Snapshot the functions up front: lowering a vec.func appends a new
        // vector_region helper to the module, and that helper must not be fed back
        // through the lowering/folding driver (it already holds lowered AVE ops and
        // the carried scf control flow).
        SmallVector<func::FuncOp, 4> funcOps(module.getOps<func::FuncOp>());
        for (func::FuncOp funcOp : funcOps) {
            if (funcOp.isDeclaration())
                continue;
            if (funcOp->hasAttr(kHivmVectorFunctionAttrName))
                continue;
            // The SIMT pass runs first and outlines its own helpers, which carry a
            // SIMT calling convention rather than hivm.vector_function. Skipping
            // them keeps this pass seeing exactly the functions the combined pass
            // saw, which snapshotted the module before any lowering happened.
            if (funcOp->hasAttr(hivm_regbaseintrins::kDavinciCallingConvAttrName))
                continue;
            std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(funcOp.getOperation());
            if (coreType != hivm::TFuncCoreType::AIV && coreType != hivm::TFuncCoreType::MIX)
                continue;
            inlineVectorRegionWrappers(funcOp);
            // Fresh per-function lowering state: the base-memref handoff cache
            // shared by LowerCopyPattern (gm<->ub / ub->l1 cifax runtime calls) and
            // the vec.func helper operand materialization.
            ::tla::TlaTensorMemrefLowering lowering;
            RewritePatternSet patterns(&getContext());
            bool invalidScalarAccessBase = false;
            patterns.add<InlineVectorRegionWrapperPattern>(&getContext());
            populateSimdVectorPatterns(
                patterns, module, nextVectorRegionId, lowering.loweredMemrefByValue, invalidScalarAccessBase);
            GreedyRewriteConfig simdCfg;
            simdCfg.cseConstants = false;
            if (failed(mlir::applyPatternsGreedily(funcOp, std::move(patterns), simdCfg))) {
                signalPassFailure();
                return;
            }
            if (invalidScalarAccessBase) {
                signalPassFailure();
                return;
            }
        }

        // A pattern can decline a match without failing the greedy driver.
        // Never hand an unoutlined extern VF to the backend as successful IR.
        if (module
                .walk([&](::tla::CallExternOp call) {
                    call.emitOpError("failed to outline enclosing SIMD VF for extern @") << call.getCallee();
                    return WalkResult::interrupt();
                })
                .wasInterrupted()) {
            signalPassFailure();
            return;
        }

        // `module.getOps<func::FuncOp>()` visits only direct functions. Inline any
        // wrapper in nested modules so finalize never sees frontend scaffolding.
        SmallVector<::tla::VectorOp, 4> leftover;
        module.walk([&](::tla::VectorOp vectorOp) { leftover.push_back(vectorOp); });
        IRRewriter rewriter(module.getContext());
        for (::tla::VectorOp vectorOp : leftover) {
            if (!vectorOp)
                continue;
            if (vectorOp->getNumRegions() == 0 || vectorOp.getBody().empty()) {
                rewriter.eraseOp(vectorOp);
                continue;
            }
            Block* body = &vectorOp.getBody().front();
            rewriter.inlineBlockBefore(body, vectorOp->getBlock(), vectorOp->getIterator());
            rewriter.eraseOp(vectorOp);
        }

        // Barrier-only vec.func regions are inlined rather than outlined, and raw
        // TLAIR may place local barriers directly in a vector wrapper. Lower those
        // remaining operations here as well so the vector lowering is the single
        // owner of tla.local_mem_bar lowering.
        SmallVector<::tla::LocalMemBarOp, 4> localMemBars;
        module.walk([&](::tla::LocalMemBarOp op) { localMemBars.push_back(op); });
        for (::tla::LocalMemBarOp op : localMemBars) {
            if (!op->getBlock())
                continue;
            rewriter.setInsertionPoint(op);
            if (failed(lowerLocalMemBar(rewriter, op))) {
                signalPassFailure();
                return;
            }
            rewriter.eraseOp(op);
        }
    }

private:
    int nextVectorRegionId = 0;
};

} // namespace

std::unique_ptr<Pass> createTlaVectorRegionSimdPass()
{
    return std::make_unique<TlaVectorRegionSimdPass>();
}

void registerTlaVectorRegionSimdPass()
{
    PassRegistration<TlaVectorRegionSimdPass>();
}

} // namespace tla
