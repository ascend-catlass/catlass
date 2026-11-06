#include "PassesCommon.h"
#include "PassesInternal.h"
#include "Passes/TlaTensorToMemref.h"
#include "Passes/TlaDebugPrint.h"
#include "Passes/TlaVectorRegionCommon.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <cstdint>
#include <limits>

namespace tla {

constexpr StringLiteral kDebugPrintWorkspaceAttrName = "tla.debug_print.workspace";
constexpr StringLiteral kDebugPrintFormatCallAttrName = "tla.debug_print.format";
constexpr StringLiteral kPrintTensorWorkspaceAttrName = "tla.print_tensor.workspace";
constexpr StringLiteral kDebugTunnelStateAttrName = "tla.debug_tunnel.state";
constexpr StringLiteral kPrintTensorSupportedDtypes = "f16, f32, i8, i16, i32, u8, u16, u32";
constexpr StringLiteral kStringFormatCalleeName = "_mlir_ciface_tla_printf_format_string";
constexpr StringLiteral kValuesFormatCalleeName = "_mlir_ciface_tla_printf_format_values";
constexpr unsigned kFormattedScalarValueSlots = 8;

static std::string typeToString(Type type)
{
    std::string text;
    llvm::raw_string_ostream os(text);
    type.print(os);
    return os.str();
}

static std::string printTensorDiagnosticTypeToken(Type type)
{
    auto integerType = dyn_cast<IntegerType>(type);
    if (integerType && integerType.isUnsigned())
        return "u" + std::to_string(integerType.getWidth());
    return typeToString(type);
}

static FailureOr<StringRef> getDebugPrintScalarTypeToken(Type type)
{
    auto intType = dyn_cast<IntegerType>(type);
    if (intType) {
        unsigned width = intType.getWidth();
        if (width == 8 || width == 16 || width == 32) {
            if (intType.isSignless())
                return width == 8 ? StringRef("i8") : width == 16 ? StringRef("i16") : StringRef("i32");
            if (intType.isUnsigned())
                return width == 8 ? StringRef("u8") : width == 16 ? StringRef("u16") : StringRef("u32");
        }
    }
    if (type.isF16())
        return StringRef("f16");
    if (type.isF32())
        return StringRef("f32");
    return failure();
}

// Format text becomes an LLVM global, so derive a deterministic, symbol-safe
// suffix from the canonical backend printf payload instead of using user text.
static uint64_t hashDebugPrintCanonicalPayload(StringRef payload)
{
    uint64_t hash = 14695981039346656037ULL;
    for (char c : payload) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

static std::string toLowerHex64(uint64_t value)
{
    constexpr char kHex[] = "0123456789abcdef";
    std::string result(16, '0');
    for (int i = 15; i >= 0; --i) {
        result[i] = kHex[value & 0xf];
        value >>= 4;
    }
    return result;
}

static FailureOr<std::string> getFormattedDebugPrintCanonicalPayload(::tla::DebugPrintOp op)
{
    auto formatAttr = op->getAttrOfType<StringAttr>("format");
    if (!formatAttr)
        return failure();

    StringRef format = formatAttr.getValue();
    std::string payload;
    payload += "tla.debug_print.format\n";
    payload += "format:";
    payload += std::to_string(format.size());
    payload += ":";
    if (!format.empty())
        payload.append(format.data(), format.size());
    payload += "\ntypes:";
    for (auto item : llvm::enumerate(op.getValues())) {
        FailureOr<StringRef> typeToken = getDebugPrintScalarTypeToken(item.value().getType());
        if (failed(typeToken)) {
            op.emitError() << "unsupported tla.debug_print operand type " << typeToString(item.value().getType());
            return failure();
        }
        if (item.index() != 0)
            payload += ",";
        payload.append(typeToken->data(), typeToken->size());
    }
    return payload;
}

static FailureOr<std::string> getFormattedDebugPrintHelperSuffix(::tla::DebugPrintOp op)
{
    FailureOr<std::string> payload = getFormattedDebugPrintCanonicalPayload(op);
    if (failed(payload))
        return failure();
    return toLowerHex64(hashDebugPrintCanonicalPayload(*payload));
}

static FailureOr<std::string> getGeneratedDebugPrintFormat(::tla::DebugPrintOp op)
{
    auto formatAttr = op->getAttrOfType<StringAttr>("format");
    if (!formatAttr)
        return failure();

    StringRef format = formatAttr.getValue();
    std::string generated;
    unsigned fieldIndex = 0;
    for (size_t i = 0; i < format.size();) {
        char c = format[i];
        if (c == '{') {
            if (i + 1 >= format.size()) {
                op.emitError("malformed format string");
                return failure();
            }
            if (format[i + 1] == '{') {
                generated.push_back('{');
                i += 2;
                continue;
            }
            if (format[i + 1] == '}') {
                if (fieldIndex >= op.getValues().size()) {
                    op.emitError("format argument count mismatch");
                    return failure();
                }
                FailureOr<StringRef> typeToken = getDebugPrintScalarTypeToken(op.getValues()[fieldIndex].getType());
                if (failed(typeToken)) {
                    op.emitError() << "unsupported tla.debug_print operand type "
                                   << typeToString(op.getValues()[fieldIndex].getType());
                    return failure();
                }
                if (typeToken->starts_with("i"))
                    generated += "%d";
                else if (typeToken->starts_with("u"))
                    generated += "%u";
                else if (typeToken->starts_with("f"))
                    generated += "%f";
                else
                    return failure();
                ++fieldIndex;
                i += 2;
                continue;
            }
            op.emitError("unsupported format field");
            return failure();
        }
        if (c == '}') {
            if (i + 1 < format.size() && format[i + 1] == '}') {
                generated.push_back('}');
                i += 2;
                continue;
            }
            op.emitError("malformed format string");
            return failure();
        }
        if (c == '%')
            generated += "%%";
        else
            generated.push_back(c);
        ++i;
    }
    if (fieldIndex != op.getValues().size()) {
        op.emitError("format argument count mismatch");
        return failure();
    }
    return generated;
}

FailureOr<StringRef> getPrintTensorHelperSuffix(Type elementType, std::string& diagnostic)
{
    StringRef suffix = llvm::StringSwitch<StringRef>(typeToString(elementType))
                           .Case("f16", "f16")
                           .Case("f32", "f32")
                           .Case("i8", "i8")
                           .Case("i16", "i16")
                           .Case("i32", "i32")
                           .Case("ui8", "u8")
                           .Case("ui16", "u16")
                           .Case("ui32", "u32")
                           .Default("");
    if (!suffix.empty())
        return suffix;
    diagnostic = "unsupported dtype " + printTensorDiagnosticTypeToken(elementType) +
                 "; supported dtypes: " + kPrintTensorSupportedDtypes.str();
    return failure();
}

FailureOr<DebugPrintHelperSpec> getDebugPrintHelperSpec(
    Type type, DebugPrintPlacement placement, std::string& diagnostic)
{
    if (placement == DebugPrintPlacement::Invalid) {
        diagnostic = "unsupported tla.debug_print placement invalid for dtype " + printTensorDiagnosticTypeToken(type) +
                     "; expected cube or vector";
        return failure();
    }

    FailureOr<StringRef> suffix = getPrintTensorHelperSuffix(type, diagnostic);
    if (failed(suffix))
        return failure();
    StringRef callee = llvm::StringSwitch<StringRef>(*suffix)
                           .Case("i8", "_mlir_ciface_tla_printf_x_i8")
                           .Case("i16", "_mlir_ciface_tla_printf_x_i16")
                           .Case("i32", "_mlir_ciface_tla_printf_x_i32")
                           .Case("u8", "_mlir_ciface_tla_printf_x_u8")
                           .Case("u16", "_mlir_ciface_tla_printf_x_u16")
                           .Case("u32", "_mlir_ciface_tla_printf_x_u32")
                           .Case("f16", "_mlir_ciface_tla_printf_v_f16")
                           .Case("f32", "_mlir_ciface_tla_printf_v_f32")
                           .Default("");
    if (callee.empty()) {
        diagnostic = "missing tla.debug_print helper for dtype " + printTensorDiagnosticTypeToken(type);
        return failure();
    }
    return DebugPrintHelperSpec{callee};
}

namespace {

static constexpr bool isPrintTensorCallIdRepresentable(uint32_t ordinal)
{
    return ordinal <= 0xffff;
}

static_assert(isPrintTensorCallIdRepresentable(0xffff));
static_assert(!isPrintTensorCallIdRepresentable(0x10000));

static FailureOr<uint16_t> getPrintTensorCallId(uint32_t ordinal, std::string& diagnostic)
{
    if (isPrintTensorCallIdRepresentable(ordinal))
        return static_cast<uint16_t>(ordinal);
    diagnostic = "call ID exceeds the 16-bit descriptor range";
    return failure();
}

} // namespace

BlockArgument getOrAppendDebugPrintWorkspaceArg(func::FuncOp funcOp)
{
    MLIRContext* ctx = funcOp.getContext();
    for (BlockArgument arg : funcOp.getArguments()) {
        if (funcOp.getArgAttr(arg.getArgNumber(), kDebugPrintWorkspaceAttrName))
            return arg;
    }

    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type, 8> inputs(oldType.getInputs().begin(), oldType.getInputs().end());
    Type workspaceType = IntegerType::get(ctx, 64);
    inputs.push_back(workspaceType);
    funcOp.setType(FunctionType::get(ctx, inputs, oldType.getResults()));

    Block& entry = funcOp.getBody().front();
    unsigned argIndex = entry.getNumArguments();
    BlockArgument workspaceArg = entry.addArgument(workspaceType, funcOp.getLoc());
    funcOp.setArgAttr(argIndex, kDebugPrintWorkspaceAttrName, UnitAttr::get(ctx));
    funcOp.setArgAttr(
        argIndex, hacc::KernelArgTypeAttr::name, hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kWorkspace));
    return workspaceArg;
}

namespace {

static BlockArgument getOrPrependPrintTensorWorkspaceArg(func::FuncOp funcOp)
{
    MLIRContext* ctx = funcOp.getContext();
    for (BlockArgument arg : funcOp.getArguments()) {
        if (funcOp.getArgAttr(arg.getArgNumber(), kPrintTensorWorkspaceAttrName))
            return arg;
    }

    Type workspaceType = IntegerType::get(ctx, 64);
    funcOp.insertArgument(0, workspaceType, DictionaryAttr::get(ctx), funcOp.getLoc());
    BlockArgument workspaceArg = funcOp.getArgument(0);
    funcOp.setArgAttr(0, kPrintTensorWorkspaceAttrName, UnitAttr::get(ctx));
    funcOp.setArgAttr(
        0, hacc::KernelArgTypeAttr::name, hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kWorkspace));
    return workspaceArg;
}

static BlockArgument getOrAppendPrintTensorWorkspaceArg(func::FuncOp funcOp)
{
    MLIRContext* ctx = funcOp.getContext();
    for (BlockArgument arg : funcOp.getArguments()) {
        if (funcOp.getArgAttr(arg.getArgNumber(), kPrintTensorWorkspaceAttrName))
            return arg;
    }

    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type, 8> inputs(oldType.getInputs().begin(), oldType.getInputs().end());
    Type workspaceType = IntegerType::get(ctx, 64);
    inputs.push_back(workspaceType);
    funcOp.setType(FunctionType::get(ctx, inputs, oldType.getResults()));

    Block& entry = funcOp.getBody().front();
    unsigned argIndex = entry.getNumArguments();
    BlockArgument workspaceArg = entry.addArgument(workspaceType, funcOp.getLoc());
    funcOp.setArgAttr(argIndex, kPrintTensorWorkspaceAttrName, UnitAttr::get(ctx));
    funcOp.setArgAttr(
        argIndex, hacc::KernelArgTypeAttr::name, hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kWorkspace));
    return workspaceArg;
}

static BlockArgument getOrAppendDebugTunnelStateArg(func::FuncOp funcOp)
{
    MLIRContext* ctx = funcOp.getContext();
    for (BlockArgument arg : funcOp.getArguments()) {
        if (funcOp.getArgAttr(arg.getArgNumber(), kDebugTunnelStateAttrName))
            return arg;
    }
    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type, 8> inputs(oldType.getInputs().begin(), oldType.getInputs().end());
    Type stateType = IntegerType::get(ctx, 64);
    inputs.push_back(stateType);
    funcOp.setType(FunctionType::get(ctx, inputs, oldType.getResults()));
    Block& entry = funcOp.getBody().front();
    unsigned argIndex = entry.getNumArguments();
    BlockArgument stateArg = entry.addArgument(stateType, funcOp.getLoc());
    funcOp.setArgAttr(argIndex, kDebugTunnelStateAttrName, UnitAttr::get(ctx));
    funcOp.setArgAttr(
        argIndex, hacc::KernelArgTypeAttr::name, hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kWorkspace));
    return stateArg;
}

static bool isMixedSplitFunction(func::FuncOp funcOp)
{
    return funcOp->hasAttr(hivm::TPartOfMixAttr::name);
}

static BlockArgument getOrCreatePrintTensorWorkspaceArg(func::FuncOp funcOp, ModuleOp module)
{
    if (!isMixedSplitFunction(funcOp)) {
        return getOrPrependPrintTensorWorkspaceArg(funcOp);
    }

    StringRef name = funcOp.getSymName();
    constexpr StringLiteral aicSuffix = "_mix_aic";
    constexpr StringLiteral aivSuffix = "_mix_aiv";
    StringRef suffix = name.ends_with(aicSuffix) ? aicSuffix : aivSuffix;
    StringRef peerSuffix = suffix == aicSuffix ? aivSuffix : aicSuffix;
    std::string peerName = (name.drop_back(suffix.size()) + peerSuffix).str();
    if (auto peer = module.lookupSymbol<func::FuncOp>(peerName))
        (void)getOrAppendPrintTensorWorkspaceArg(peer);
    return getOrAppendPrintTensorWorkspaceArg(funcOp);
}

static BlockArgument getOrCreateDebugTunnelStateArg(func::FuncOp funcOp, ModuleOp module)
{
    if (!isMixedSplitFunction(funcOp))
        return getOrAppendDebugTunnelStateArg(funcOp);

    StringRef name = funcOp.getSymName();
    constexpr StringLiteral aicSuffix = "_mix_aic";
    constexpr StringLiteral aivSuffix = "_mix_aiv";
    StringRef suffix = name.ends_with(aicSuffix) ? aicSuffix : aivSuffix;
    StringRef peerSuffix = suffix == aicSuffix ? aivSuffix : aicSuffix;
    std::string peerName = (name.drop_back(suffix.size()) + peerSuffix).str();
    if (auto peer = module.lookupSymbol<func::FuncOp>(peerName))
        (void)getOrAppendDebugTunnelStateArg(peer);
    return getOrAppendDebugTunnelStateArg(funcOp);
}

static void annotatePrintfRuntimeCall(func::FuncOp funcOp)
{
    MLIRContext* ctx = funcOp.getContext();
    funcOp->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE), UnitAttr::get(ctx));
    funcOp->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIC_OR_AIV));
}

static func::FuncOp getOrCreateDebugRuntimeCall(ModuleOp module, StringRef name, ArrayRef<Type> operandTypes)
{
    if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
        annotatePrintfRuntimeCall(existing);
        return existing;
    }
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    auto func = builder.create<func::FuncOp>(module.getLoc(), name, builder.getFunctionType(operandTypes, {}));
    func.setPrivate();
    annotatePrintfRuntimeCall(func);
    return func;
}

static LLVM::GlobalOp getOrCreateStringFormatGlobal(
    ModuleOp module, OpBuilder& builder, Location loc, StringRef suffix, StringRef format)
{
    std::string name = "tla_debug_print_format_";
    name.append(suffix.data(), suffix.size());
    if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(name))
        return existing;

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    std::string value(format.data(), format.size());
    value.push_back('\0');
    auto globalType = LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8), value.size());
    return builder.create<LLVM::GlobalOp>(
        loc, globalType, /*isConstant=*/true, LLVM::Linkage::Internal, name,
        builder.getStringAttr(StringRef(value.data(), value.size())),
        /*alignment=*/0);
}

static Value createStringFormatPointer(OpBuilder& builder, Location loc, LLVM::GlobalOp global)
{
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, ptrType, global.getSymNameAttr());
    SmallVector<LLVM::GEPArg> indices{0, 0};
    return builder.create<LLVM::GEPOp>(loc, ptrType, global.getType(), globalPtr, indices);
}

static SmallVector<Type, 12> getFormattedValuesHelperTypes(MLIRContext* ctx)
{
    Type i64Type = IntegerType::get(ctx, 64);
    SmallVector<Type, 12> types{LLVM::LLVMPointerType::get(ctx), i64Type, i64Type};
    for (unsigned i = 0; i < kFormattedScalarValueSlots; ++i)
        types.push_back(i64Type);
    types.push_back(i64Type);
    return types;
}

static FailureOr<Value> createFormattedScalarSlot(OpBuilder& builder, Location loc, Value value)
{
    Type i64Type = builder.getI64Type();
    Type valueType = value.getType();
    auto intType = dyn_cast<IntegerType>(valueType);
    if (intType && (intType.getWidth() == 8 || intType.getWidth() == 16 || intType.getWidth() == 32)) {
        if (intType.isSignless())
            return builder.create<arith::ExtSIOp>(loc, i64Type, value).getResult();
        if (intType.isUnsigned()) {
            Type signlessType = builder.getIntegerType(intType.getWidth());
            Value signless = builder.create<UnrealizedConversionCastOp>(loc, signlessType, value).getResult(0);
            return builder.create<arith::ExtUIOp>(loc, i64Type, signless).getResult();
        }
    }
    if (valueType.isF16())
        value = builder.create<arith::ExtFOp>(loc, builder.getF32Type(), value);
    if (value.getType().isF32()) {
        Value bits = builder.create<arith::BitcastOp>(loc, builder.getI32Type(), value);
        return builder.create<arith::ExtUIOp>(loc, i64Type, bits).getResult();
    }
    return failure();
}

static FailureOr<SmallVector<Value, 12>> buildFormattedValuesCallOperands(
    OpBuilder& builder, Location loc, StringRef generatedFormat, Value formatPtr, ValueRange values, Value workspace)
{
    SmallVector<Value, 12> callOperands;
    callOperands.push_back(formatPtr);
    callOperands.push_back(builder.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(generatedFormat.size()), 64));
    callOperands.push_back(builder.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(values.size()), 64));

    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    SmallVector<Value, 8> slots(kFormattedScalarValueSlots, zero);
    for (auto item : llvm::enumerate(values)) {
        FailureOr<Value> slot = createFormattedScalarSlot(builder, loc, item.value());
        if (failed(slot))
            return failure();
        slots[item.index()] = *slot;
    }
    callOperands.append(slots.begin(), slots.end());
    callOperands.push_back(workspace);
    return callOperands;
}

} // namespace

static FailureOr<std::string> getVfPrintFormat(DebugPrintOp op)
{
    if (op.getFormat())
        return getGeneratedDebugPrintFormat(op);
    Type type = op.getValues().front().getType();
    return std::string(isa<FloatType>(type) ? "v=%f" : type.isUnsignedInteger() ? "x=%u" : "x=%d");
}

BlockArgument getOrAppendSimtPrintWorkspaceArg(func::FuncOp function)
{
    constexpr StringLiteral attribute = "tla.simt_print.workspace";
    for (BlockArgument argument : function.getArguments())
        if (function.getArgAttr(argument.getArgNumber(), attribute))
            return argument;
    unsigned index = function.getNumArguments();
    auto type = IntegerType::get(function.getContext(), 64);
    function.insertArgument(index, type, DictionaryAttr::get(function.getContext()), function.getLoc());
    function.setArgAttr(index, attribute, UnitAttr::get(function.getContext()));
    function.setArgAttr(
        index, hacc::KernelArgTypeAttr::name,
        hacc::KernelArgTypeAttr::get(function.getContext(), hacc::KernelArgType::kWorkspace));
    // Both mixed entries consume the same host argument blob. Reserve the
    // SIMT slot in the peer before later passes append legacy print contexts,
    // even though only AIV executes the SIMT function.
    if (isMixedSplitFunction(function)) {
        constexpr StringLiteral aicSuffix = "_mix_aic";
        constexpr StringLiteral aivSuffix = "_mix_aiv";
        StringRef name = function.getSymName();
        StringRef suffix = name.ends_with(aicSuffix) ? aicSuffix : aivSuffix;
        StringRef peerSuffix = suffix == aicSuffix ? aivSuffix : aicSuffix;
        auto module = function->getParentOfType<ModuleOp>();
        if (auto peer = module.lookupSymbol<func::FuncOp>((name.drop_back(suffix.size()) + peerSuffix).str()))
            (void)getOrAppendSimtPrintWorkspaceArg(peer);
    }
    return function.getArgument(index);
}

LogicalResult lowerSimtPrints(func::FuncOp function, Value workspace)
{
    ModuleOp module = function->getParentOfType<ModuleOp>();
    SmallVector<Operation*> prints;
    function.walk([&](Operation* operation) {
        if (isa<DebugPrintOp, PrintTensorOp>(operation))
            prints.push_back(operation);
    });
    for (Operation* operation : prints) {
        OpBuilder builder(operation);
        Location loc = operation->getLoc();
        auto constant = [&](int64_t value, unsigned width = 32) -> Value {
            return builder.create<arith::ConstantIntOp>(loc, value, width);
        };
        SmallVector<Value> operands{workspace};
        std::string name;
        if (auto scalar = dyn_cast<DebugPrintOp>(operation)) {
            auto format = getVfPrintFormat(scalar);
            if (failed(format))
                return failure();
            // Hex keeps this compile metadata unambiguous in textual MLIR and
            // avoids placing inaccessible device globals in a SIMT function.
            std::string blob;
            if (auto previous = module->getAttrOfType<StringAttr>("tla.simt_print.formats"))
                blob = previous.getValue().str();
            operands.push_back(constant(blob.size() / 2));
            constexpr char hex[] = "0123456789abcdef";
            for (unsigned char byte : *format) {
                blob += hex[byte >> 4];
                blob += hex[byte & 15];
            }
            blob += "00";
            module->setAttr("tla.simt_print.formats", builder.getStringAttr(blob));
            operands.push_back(constant(scalar.getValues().size()));
            for (unsigned slot = 0; slot < 8; ++slot) {
                if (slot >= scalar.getValues().size()) {
                    operands.push_back(constant(0, 64));
                    continue;
                }
                auto value = createFormattedScalarSlot(builder, loc, scalar.getValues()[slot]);
                if (failed(value))
                    return scalar.emitError("unsupported SIMT print scalar type");
                operands.push_back(*value);
            }
            name = "_mlir_ciface_tla_simt_printf";
        } else {
            auto tensor = cast<PrintTensorOp>(operation);
            // The outlining map has replaced the tensor descriptor with the
            // correctly offset, address-space-qualified flat memref parameter.
            Value source = operation->getOperand(0);
            auto type = dyn_cast<MemRefType>(source.getType());
            if (!type || !type.hasStaticShape())
                return tensor.emitError("SIMT tensor printing requires a static GM or UB view");
            auto space = dyn_cast<IntegerAttr>(type.getMemorySpace());
            if (!space || (space.getInt() != 1 && space.getInt() != 6))
                return tensor.emitError("SIMT tensor printing requires GM or UB storage");
            Value pointer = builder.create<::mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, source);
            pointer = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), pointer);
            auto pointerType = LLVM::LLVMPointerType::get(builder.getContext(), space.getInt());
            operands.push_back(builder.create<LLVM::IntToPtrOp>(loc, pointerType, pointer));
            operands.push_back(tensor.getLength());
            auto shape = tensor.getShape();
            int64_t rows = shape.size() == 2 ? shape[0] : 1;
            int64_t columns = shape.back();
            if (rows <= 0 || columns <= 0)
                return tensor.emitError("SIMT tensor printing requires a static positive shape");
            if (rows > std::numeric_limits<uint32_t>::max() || columns > std::numeric_limits<uint32_t>::max())
                return tensor.emitError("SIMT tensor print dimensions must be between 1 and 4294967295");
            operands.push_back(constant(std::min<int64_t>({rows * columns, type.getNumElements(), 262112})));
            Type element = type.getElementType();
            std::string diagnostic;
            auto suffix = getPrintTensorHelperSuffix(element, diagnostic);
            if (failed(suffix))
                return tensor.emitError(diagnostic);
            unsigned code = llvm::StringSwitch<unsigned>(*suffix)
                                .Case("f32", 0)
                                .Case("f16", 1)
                                .Case("i8", 2)
                                .Case("i32", 3)
                                .Case("u8", 4)
                                .Case("i16", 6)
                                .Case("u16", 7)
                                .Case("u32", 8)
                                .Default(0);
            operands.push_back(constant(code));
            operands.push_back(constant(rows));
            operands.push_back(constant(columns));
            operands.push_back(constant(shape.size()));
            name = space.getInt() == 1 ? "_mlir_ciface_tla_simt_print_gm" : "_mlir_ciface_tla_simt_print_ub";
        }
        SmallVector<Type> types;
        for (Value operand : operands)
            types.push_back(operand.getType());
        auto callee = getOrCreateDebugRuntimeCall(module, name, types);
        callee->setAttr(
            hivm_regbaseintrins::kDavinciCallingConvAttrName,
            hivm_regbaseintrins::SIMT_CallableAttr::get(builder.getContext()));
        builder.create<func::CallOp>(loc, callee, operands);
        operation->erase();
    }
    return success();
}

FailureOr<uint64_t> reserveVfPrintFormats(ModuleOp module, uint64_t bytes)
{
    uint32_t callId = 0;
    auto numbered = module.walk([&](Operation* operation) {
        if (!isa<PrintVectorOp, PrintTensorOp>(operation))
            return WalkResult::advance();
        auto vf = operation->getParentOfType<VecFuncOp>();
        if (!vf || !vecregion::isSimdVecFunc(vf))
            return WalkResult::advance();
        if (callId > 0xffff) {
            operation->emitError("VF print call ID exceeds the 16-bit descriptor range");
            return WalkResult::interrupt();
        }
        operation->setAttr("tla.vf_print_id", IntegerAttr::get(IntegerType::get(module.getContext(), 32), callId++));
        return WalkResult::advance();
    });
    if (numbered.wasInterrupted())
        return failure();
    if (callId)
        module->setAttr("tla.vf_print.count", IntegerAttr::get(IntegerType::get(module.getContext(), 32), callId));
    llvm::StringMap<uint64_t> offsets;
    auto result = module.walk([&](DebugPrintOp op) {
        auto vf = op->getParentOfType<VecFuncOp>();
        if (!vf || !vecregion::isSimdVecFunc(vf))
            return WalkResult::advance();
        auto format = getVfPrintFormat(op);
        if (failed(format))
            return WalkResult::interrupt();
        // CANN's 2024-byte VF buffer holds a 24-byte TLV header, 8-byte
        // argument slots and the NUL-terminated format. Its alignment helper
        // rounds up to 8 bytes, then adds another 8 bytes. Validate before
        // outlining so rejection propagates as a pass failure, not a mismatch.
        size_t recordBytes = llvm::alignTo(size_t{24} + 8 * op.getValues().size() + format->size() + 1, size_t{8}) + 8;
        if (recordBytes > 2024) {
            op.emitError("formatted VF print record exceeds the 2024-byte debug buffer");
            return WalkResult::interrupt();
        }
        if (bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - format->size() - 8) {
            op.emitError("VF print format reservation overflows the UB address range");
            return WalkResult::interrupt();
        }
        auto [entry, inserted] = offsets.try_emplace(*format, llvm::alignTo(bytes, uint64_t{4}));
        if (inserted)
            bytes = entry->second + llvm::alignTo(format->size() + 1, size_t{4});
        op->setAttr("tla.vf_format_offset", IntegerAttr::get(IntegerType::get(module.getContext(), 64), entry->second));
        return WalkResult::advance();
    });
    if (result.wasInterrupted())
        return failure();
    return bytes;
}

LogicalResult lowerVfScalarPrint(OpBuilder& builder, DebugPrintOp op, ValueRange values)
{
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();
    auto formatOr = getVfPrintFormat(op);
    if (failed(formatOr))
        return failure();
    const auto& format = *formatOr;
    size_t bytes = llvm::alignTo(format.size() + 1, size_t{4});
    auto global = module.lookupSymbol<LLVM::GlobalOp>("tla_ub_scratch");
    auto offset = op->getAttrOfType<IntegerAttr>("tla.vf_format_offset");
    if (!global || !offset)
        return op.emitError("VF print format was not reserved by tla-lower-ptr");
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 6);
    Value scratch = builder.create<LLVM::AddressOfOp>(loc, global);
    Value address = builder.create<LLVM::GEPOp>(
        loc, ptrType, builder.getI8Type(), scratch, ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(offset.getInt())});
    // Match CANN's VF string-literal lowering: an on-chip global initialized
    // with immediate word stores at the print site, with no GM access in VF.
    std::string padded = format;
    padded.resize(bytes, '\0');
    for (size_t offset = 0; offset < bytes; offset += 4) {
        uint32_t word = 0;
        for (unsigned byte = 0; byte < 4; ++byte)
            word |= static_cast<uint32_t>(static_cast<uint8_t>(padded[offset + byte])) << (8 * byte);
        Value value = builder.create<arith::ConstantIntOp>(loc, word, 32);
        Value ptr = builder.create<LLVM::GEPOp>(
            loc, ptrType, builder.getI8Type(), address, ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(offset)});
        builder.create<LLVM::StoreOp>(loc, value, ptr);
    }
    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    Value count = builder.create<arith::ConstantIntOp>(loc, values.size(), 32);
    SmallVector<Value> operands = {address, count};
    for (unsigned i = 0; i < 8; ++i) {
        if (i >= values.size()) {
            operands.push_back(zero);
            continue;
        }
        auto slot = createFormattedScalarSlot(builder, loc, values[i]);
        if (failed(slot))
            return op.emitError("VF scalar print requires a scalar value; print a register with a prefix length");
        operands.push_back(*slot);
    }
    SmallVector<Type> types;
    for (Value value : operands)
        types.push_back(value.getType());
    auto callee = getOrCreateDebugRuntimeCall(module, "_mlir_ciface_tla_vf_printf", types);
    builder.create<func::CallOp>(loc, callee, operands);
    return success();
}

void emitVfDebugBoundary(OpBuilder& builder, Location loc, ModuleOp module, Value workspace, bool begin)
{
    SmallVector<Value> operands;
    SmallVector<Type> types;
    if (begin) {
        operands.push_back(workspace);
        types.push_back(workspace.getType());
    }
    auto callee = getOrCreateDebugRuntimeCall(
        module, begin ? "_mlir_ciface_tla_vf_debug_begin" : "_mlir_ciface_tla_vf_debug_end", types);
    builder.create<func::CallOp>(loc, callee, operands);
}

namespace {

// Mixed modules link both core libraries. Give AIV implementations distinct
// symbols so a same-named AIC helper cannot supply its block/subblock encoding.
static std::string corePrintHelperName(func::FuncOp function, StringRef name)
{
    return name.str() +
           (isMixedSplitFunction(function) && getFunctionCoreType(function) == hivm::TFuncCoreType::AIV ? "_aiv" : "");
}

static LogicalResult lowerLegacyDebugPrint(::tla::DebugPrintOp op, PatternRewriter& rewriter, ModuleOp module)
{
    if (op->getNumResults() != 0 || op->getNumOperands() != 1)
        return op.emitError("tla.debug_print lowering requires exactly one operand and no results");

    Value value = op.getValues().front();
    Type valueType = value.getType();
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp)
        return op.emitError("tla.debug_print must be nested inside func.func");
    DebugPrintPlacement placement = DebugPrintPlacement::Invalid;
    if (std::optional<hivm::TFuncCoreType> coreType = getFunctionCoreType(funcOp)) {
        if (*coreType == hivm::TFuncCoreType::AIC)
            placement = DebugPrintPlacement::Cube;
        else if (*coreType == hivm::TFuncCoreType::AIV)
            placement = DebugPrintPlacement::Vector;
    }
    std::string diagnostic;
    FailureOr<DebugPrintHelperSpec> helper = getDebugPrintHelperSpec(valueType, placement, diagnostic);
    if (failed(helper))
        return op.emitError() << diagnostic;

    BlockArgument workspace = getOrAppendDebugPrintWorkspaceArg(funcOp);
    auto callee = getOrCreateDebugRuntimeCall(
        module, corePrintHelperName(funcOp, helper->callee), {value.getType(), workspace.getType()});
    rewriter.create<func::CallOp>(op.getLoc(), callee, ValueRange{value, workspace});
    rewriter.eraseOp(op);
    return success();
}

static LogicalResult lowerFormattedDebugPrint(::tla::DebugPrintOp op, PatternRewriter& rewriter, ModuleOp module)
{
    if (op->getNumResults() != 0)
        return op.emitError("formatted tla.debug_print lowering requires no results");

    auto formatAttr = op->getAttrOfType<StringAttr>("format");
    if (!formatAttr)
        return op.emitError("formatted tla.debug_print lowering requires a format attribute");

    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp)
        return op.emitError("tla.debug_print must be nested inside func.func");

    FailureOr<std::string> helperSuffix = getFormattedDebugPrintHelperSuffix(op);
    if (failed(helperSuffix))
        return failure();
    FailureOr<std::string> generatedFormat = getGeneratedDebugPrintFormat(op);
    if (failed(generatedFormat))
        return failure();

    Type i64Type = rewriter.getI64Type();
    BlockArgument workspace = getOrAppendDebugPrintWorkspaceArg(funcOp);

    if (op.getValues().empty()) {
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        LLVM::GlobalOp global =
            getOrCreateStringFormatGlobal(module, rewriter, op.getLoc(), *helperSuffix, *generatedFormat);
        Value formatPtr = createStringFormatPointer(rewriter, op.getLoc(), global);
        auto callee = getOrCreateDebugRuntimeCall(
            module, corePrintHelperName(funcOp, kStringFormatCalleeName), {ptrType, i64Type, workspace.getType()});
        SmallVector<Value, 3> callOperands;
        callOperands.push_back(formatPtr);
        callOperands.push_back(
            rewriter.create<arith::ConstantIntOp>(op.getLoc(), static_cast<int64_t>(generatedFormat->size()), 64));
        callOperands.push_back(workspace);
        auto call = rewriter.create<func::CallOp>(op.getLoc(), callee, callOperands);
        call->setAttr(kDebugPrintFormatCallAttrName, formatAttr);
        rewriter.eraseOp(op);
        return success();
    }

    LLVM::GlobalOp global =
        getOrCreateStringFormatGlobal(module, rewriter, op.getLoc(), *helperSuffix, *generatedFormat);
    Value formatPtr = createStringFormatPointer(rewriter, op.getLoc(), global);
    auto callee = getOrCreateDebugRuntimeCall(
        module, corePrintHelperName(funcOp, kValuesFormatCalleeName),
        getFormattedValuesHelperTypes(rewriter.getContext()));
    FailureOr<SmallVector<Value, 12>> callOperands =
        buildFormattedValuesCallOperands(rewriter, op.getLoc(), *generatedFormat, formatPtr, op.getValues(), workspace);
    if (failed(callOperands))
        return op.emitError("unsupported tla.debug_print operand type");

    auto call = rewriter.create<func::CallOp>(op.getLoc(), callee, *callOperands);
    call->setAttr(kDebugPrintFormatCallAttrName, formatAttr);
    rewriter.eraseOp(op);
    return success();
}

static LogicalResult lowerDebugPrint(::tla::DebugPrintOp op, PatternRewriter& rewriter, ModuleOp module)
{
    if (op->getAttrOfType<StringAttr>("format"))
        return lowerFormattedDebugPrint(op, rewriter, module);
    return lowerLegacyDebugPrint(op, rewriter, module);
}

static LogicalResult lowerPrintTensor(
    ::tla::PrintTensorOp op, PatternRewriter& rewriter, ModuleOp module, uint32_t callId)
{
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp)
        return op.emitError("tla.print_tensor must be nested inside func.func");

    auto descOp = op.getValue().getDefiningOp<::tla::TensorDescOp>();
    if (!descOp)
        return op.emitError("could not resolve the materialized tensor descriptor");
    FailureOr<::tla::TensorDescriptor> descOr = ::tla::descriptorFromTensorDescOp(descOp);
    if (failed(descOr))
        return op.emitError("could not decode the materialized tensor descriptor");
    const ::tla::TensorDescriptor& desc = *descOr;
    llvm::DenseMap<Value, Value> baseMemrefCache;
    FailureOr<Value> materialized =
        ::tla::getOrMaterializeDescriptorBaseMemref(rewriter, op.getLoc(), desc, op, baseMemrefCache);
    if (failed(materialized))
        return op.emitError("could not materialize the tensor base");
    auto tensorType = op.getValue().getType();
    Type elementType = tensorType.getPtr().getPointee();
    bool isL1 = tensorType.getPtr().getAddrspace() == AddressSpace::l1;
    bool isL0C = tensorType.getPtr().getAddrspace() == AddressSpace::l0c;
    bool isLocalDump = isL1 || isL0C;
    if (isLocalDump && !op.getLength().getDefiningOp<arith::ConstantOp>()) {
        if (isL1) {
            op.emitWarning(
                "dynamic L1 tensor print length is not runtime-validated; "
                "it must stay within 1..8 elements at an aligned address, "
                "or no native record is emitted");
        } else {
            op.emitWarning(
                "dynamic L0C tensor print length is not runtime-validated; "
                "it must equal 256 elements, or no native record is emitted");
        }
    }
    Value localRuntimeMemref;
    Value tensorPtr;
    if (isLocalDump) {
        auto materializedType = dyn_cast<MemRefType>((*materialized).getType());
        if (!materializedType)
            return op.emitError("could not materialize the local tensor memref");
        FailureOr<Value> runtimeMemref = ::tla::castMemrefToType(
            rewriter, op.getLoc(), *materialized, ::tla::getDynamicStridedMemrefType(materializedType));
        if (failed(runtimeMemref))
            return op.emitError("could not lower the local tensor memref ABI");
        localRuntimeMemref = *runtimeMemref;
    } else {
        tensorPtr = rewriter.create<::mlir::memref::ExtractAlignedPointerAsIndexOp>(op.getLoc(), *materialized);
    }
    Value elementOffset;
    if (::tla::isLinearLayout(desc.layoutTag)) {
        Value rowElements = rewriter.createOrFold<arith::MulIOp>(op.getLoc(), desc.coord[0], desc.stride[0]);
        Value colElements = rewriter.createOrFold<arith::MulIOp>(op.getLoc(), desc.coord[1], desc.stride[1]);
        elementOffset = rewriter.createOrFold<arith::AddIOp>(op.getLoc(), rowElements, colElements);
    } else {
        if (!::tla::isNZFamilyLayout(desc.layoutTag))
            return op.emitError("could not materialize the tensor physical offset");

        Value rowDivisor = desc.shape[0];
        if (desc.layoutTag == ::LayoutTag::zNUnAlign)
            rowDivisor = rewriter.createOrFold<arith::DivSIOp>(op.getLoc(), desc.stride[3], desc.shape[2]);
        Value packed0 = rewriter.createOrFold<arith::RemSIOp>(op.getLoc(), desc.coord[0], rowDivisor);
        Value packed1 = rewriter.createOrFold<arith::DivSIOp>(op.getLoc(), desc.coord[0], rowDivisor);
        Value packed2 = rewriter.createOrFold<arith::RemSIOp>(op.getLoc(), desc.coord[1], desc.shape[2]);
        Value packed3 = rewriter.createOrFold<arith::DivSIOp>(op.getLoc(), desc.coord[1], desc.shape[2]);
        Value term0 = rewriter.createOrFold<arith::MulIOp>(op.getLoc(), packed0, desc.stride[0]);
        Value term1 = rewriter.createOrFold<arith::MulIOp>(op.getLoc(), packed1, desc.stride[1]);
        Value term2 = rewriter.createOrFold<arith::MulIOp>(op.getLoc(), packed2, desc.stride[2]);
        Value term3 = rewriter.createOrFold<arith::MulIOp>(op.getLoc(), packed3, desc.stride[3]);
        Value rowElements = rewriter.createOrFold<arith::AddIOp>(op.getLoc(), term0, term1);
        Value colElements = rewriter.createOrFold<arith::AddIOp>(op.getLoc(), term2, term3);
        elementOffset = rewriter.createOrFold<arith::AddIOp>(op.getLoc(), rowElements, colElements);
    }
    Value elementBytes = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), elementType.getIntOrFloatBitWidth() / 8);
    Value byteOffset = rewriter.create<arith::MulIOp>(op.getLoc(), elementOffset, elementBytes);
    Value tensorI64;
    Value byteOffsetI64;
    if (isLocalDump) {
        byteOffsetI64 = ::tla::castValueToI64(rewriter, op.getLoc(), byteOffset);
    } else {
        tensorPtr = rewriter.create<arith::AddIOp>(op.getLoc(), tensorPtr, byteOffset);
        tensorI64 = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getI64Type(), tensorPtr);
    }
    Value count = op.getLength();
    Value shape0;
    Value shape1;
    if (isL0C) {
        shape0 = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 16, 64);
        shape1 = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 16, 64);
    } else if (::tla::isNZFamilyLayout(desc.layoutTag)) {
        shape0 = ::tla::castValueToI64(rewriter, op.getLoc(), desc.originShape[0]);
        shape1 = ::tla::castValueToI64(rewriter, op.getLoc(), desc.originShape[1]);
    } else if (op.getShape().size() == 1) {
        shape0 = ::tla::castValueToI64(rewriter, op.getLoc(), desc.shape[1]);
        shape1 = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
    } else {
        shape0 = ::tla::castValueToI64(rewriter, op.getLoc(), desc.shape[0]);
        shape1 = ::tla::castValueToI64(rewriter, op.getLoc(), desc.shape[1]);
    }
    Value shift = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 32, 64);
    Value packedShape1 = rewriter.create<arith::ShLIOp>(op.getLoc(), shape1, shift);
    Value packedShape = rewriter.create<arith::OrIOp>(op.getLoc(), shape0, packedShape1);
    Value encodedCallId = rewriter.create<arith::ConstantIntOp>(op.getLoc(), callId, 64);
    std::string diagnostic;
    FailureOr<StringRef> helperSuffix = getPrintTensorHelperSuffix(elementType, diagnostic);
    if (failed(helperSuffix))
        return op.emitError(diagnostic);
    std::string calleeName;
    if (isL1) {
        calleeName = "tla_print_tensor_l1_";
    } else if (isL0C) {
        calleeName = "tla_print_tensor_l0c_";
    } else {
        calleeName = tensorType.getPtr().getAddrspace() == AddressSpace::ub ? "_mlir_ciface_tla_print_tensor_ub_" :
                                                                              "_mlir_ciface_tla_print_tensor_gm_";
    }
    calleeName.append(helperSuffix->data(), helperSuffix->size());
    if (tensorType.getPtr().getAddrspace() == AddressSpace::gm)
        calleeName = corePrintHelperName(funcOp, calleeName);
    BlockArgument workspace = getOrCreatePrintTensorWorkspaceArg(funcOp, module);
    BlockArgument debugTunnelState = isLocalDump ? getOrCreateDebugTunnelStateArg(funcOp, module) : BlockArgument();
    SmallVector<Type> calleeTypes;
    SmallVector<Value> callOperands;
    if (isLocalDump) {
        calleeTypes = {workspace.getType(), localRuntimeMemref.getType(), byteOffsetI64.getType(),
                       count.getType(),     packedShape.getType(),        encodedCallId.getType()};
        callOperands = {workspace, localRuntimeMemref, byteOffsetI64, count, packedShape, encodedCallId};
    } else {
        calleeTypes = {
            workspace.getType(), tensorI64.getType(), count.getType(), packedShape.getType(), encodedCallId.getType()};
        callOperands = {workspace, tensorI64, count, packedShape, encodedCallId};
    }
    auto callee = getOrCreateDebugRuntimeCall(module, calleeName, calleeTypes);
    if (isLocalDump) {
        callee->setAttr(
            hivm::TFuncCoreTypeAttr::name,
            hivm::TFuncCoreTypeAttr::get(rewriter.getContext(), hivm::TFuncCoreType::AIC));
    }
    if (debugTunnelState) {
        callee->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
    }
    rewriter.create<func::CallOp>(op.getLoc(), callee, callOperands);
    rewriter.eraseOp(op);
    return success();
}

// Split entries consume the same packed host arguments. Consolidate contexts
// after all lowering passes have created their uses, preserving Tensor's
// established leading workspace in non-mixed kernels.
static void normalizePrintEntryGroup(ModuleOp module, ArrayRef<func::FuncOp> entries)
{
    bool scalar = false, tensor = false, simt = false, tunnel = false, mixed = false;
    for (auto function : entries) {
        mixed |= isMixedSplitFunction(function);
        for (BlockArgument argument : function.getArguments()) {
            unsigned index = argument.getArgNumber();
            scalar |= static_cast<bool>(function.getArgAttr(index, kDebugPrintWorkspaceAttrName));
            tensor |= static_cast<bool>(function.getArgAttr(index, kPrintTensorWorkspaceAttrName));
            simt |= static_cast<bool>(function.getArgAttr(index, "tla.simt_print.workspace"));
            tunnel |= static_cast<bool>(function.getArgAttr(index, kDebugTunnelStateAttrName));
        }
    }
    if (!scalar && !tensor && !simt && !tunnel)
        return;
    SmallVector<StringRef> layout;
    bool leadingFifo = tensor && !mixed;
    if (leadingFifo)
        layout.push_back("fifo");
    layout.push_back("user");
    if (simt)
        layout.push_back("simt");
    if ((scalar || tensor) && !leadingFifo)
        layout.push_back("fifo");
    if (tunnel)
        layout.push_back("tunnel");

    auto* context = module.getContext();
    for (auto function : entries) {
        SmallVector<BlockArgument> original(function.getArguments());
        auto role = [&](BlockArgument argument) -> StringRef {
            unsigned index = argument.getArgNumber();
            if (function.getArgAttr(index, kDebugPrintWorkspaceAttrName) ||
                function.getArgAttr(index, kPrintTensorWorkspaceAttrName))
                return "fifo";
            if (function.getArgAttr(index, "tla.simt_print.workspace"))
                return "simt";
            if (function.getArgAttr(index, kDebugTunnelStateAttrName))
                return "tunnel";
            return "user";
        };
        for (StringRef kind : layout) {
            if (kind == "user") {
                for (BlockArgument argument : original) {
                    if (role(argument) != "user")
                        continue;
                    function.insertArgument(
                        function.getNumArguments(), argument.getType(),
                        function.getArgAttrDict(argument.getArgNumber()), argument.getLoc());
                    argument.replaceAllUsesWith(function.getArguments().back());
                }
                continue;
            }
            unsigned index = function.getNumArguments();
            function.insertArgument(
                index, IntegerType::get(context, 64), DictionaryAttr::get(context), function.getLoc());
            function.setArgAttr(
                index, hacc::KernelArgTypeAttr::name,
                hacc::KernelArgTypeAttr::get(context, hacc::KernelArgType::kWorkspace));
            if (kind == "fifo") {
                if (scalar)
                    function.setArgAttr(index, kDebugPrintWorkspaceAttrName, UnitAttr::get(context));
                if (tensor)
                    function.setArgAttr(index, kPrintTensorWorkspaceAttrName, UnitAttr::get(context));
            } else {
                function.setArgAttr(
                    index,
                    kind == "simt" ? StringRef("tla.simt_print.workspace") : StringRef(kDebugTunnelStateAttrName),
                    UnitAttr::get(context));
            }
            for (BlockArgument argument : original)
                if (role(argument) == kind)
                    argument.replaceAllUsesWith(function.getArgument(index));
        }
        llvm::BitVector erase(function.getNumArguments());
        erase.set(0, original.size());
        function.eraseArguments(erase);
        function->setAttr("tla.print.layout", StringAttr::get(context, llvm::join(layout, ",")));
    }
}

static void normalizePrintArguments(ModuleOp module)
{
    llvm::StringMap<SmallVector<func::FuncOp>> groups;
    for (auto function : module.getOps<func::FuncOp>()) {
        if (!function->hasAttr("hacc.entry") || function.isDeclaration())
            continue;
        StringRef name = function.getSymName();
        if (isMixedSplitFunction(function) && (name.ends_with("_mix_aic") || name.ends_with("_mix_aiv")))
            name = name.drop_back(8);
        groups[name].push_back(function);
    }
    for (auto& group : groups)
        normalizePrintEntryGroup(module, group.second);
    if (groups.size() == 1) {
        if (auto layout = groups.begin()->second.front()->getAttr("tla.print.layout"))
            module->setAttr("tla.print.layout", layout);
    }
}

class TlaLowerDebugPrintPass : public PassWrapper<TlaLowerDebugPrintPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaLowerDebugPrintPass)

    StringRef getArgument() const override
    {
        return "tla-lower-debug-print";
    }
    StringRef getName() const override
    {
        return "TlaLowerDebugPrintPass";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<arith::ArithDialect, func::FuncDialect, hivm::HIVMDialect, LLVM::LLVMDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        SmallVector<::tla::DebugPrintOp, 8> ops;
        module.walk([&](::tla::DebugPrintOp op) { ops.push_back(op); });
        SmallVector<::tla::PrintTensorOp, 8> printOps;
        module.walk([&](::tla::PrintTensorOp op) { printOps.push_back(op); });
        for (::tla::DebugPrintOp op : ops) {
            if (!op || !op->getBlock())
                continue;
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            if (failed(lowerDebugPrint(op, rewriter, module))) {
                signalPassFailure();
                return;
            }
        }
        llvm::DenseMap<Operation*, uint32_t> nextCallId;
        llvm::SmallPtrSet<Operation*, 4> tunnelFunctionSet;
        SmallVector<func::FuncOp, 4> tunnelFunctions;
        for (::tla::PrintTensorOp op : printOps) {
            if (!op || !op->getBlock())
                continue;
            auto funcOp = op->getParentOfType<func::FuncOp>();
            if (!funcOp || funcOp.isPrivate()) {
                op.emitError("tla.print_tensor requires a non-private kernel entrypoint");
                signalPassFailure();
                return;
            }
            if (!funcOp.getBody().hasOneBlock()) {
                op.emitError("tla.print_tensor requires a single-block kernel CFG");
                signalPassFailure();
                return;
            }
            if (op->getBlock() != &op->getParentRegion()->front()) {
                op.emitError("tla.print_tensor must be in an entry CFG block");
                signalPassFailure();
                return;
            }
            for (Operation* ancestor = op->getParentOp(); ancestor && ancestor != funcOp.getOperation();
                 ancestor = ancestor->getParentOp()) {
                StringRef name = ancestor->getName().getStringRef();
                if (name == "scf.if" || name == "scf.for" || name == "scf.while") {
                    continue;
                }
                if (ancestor->getNumRegions() != 0) {
                    op.emitError("tla.print_tensor has an unrecognized multi-execution ancestor");
                    signalPassFailure();
                    return;
                }
            }
            auto vfCount = module->getAttrOfType<IntegerAttr>("tla.vf_print.count");
            uint32_t ordinal = nextCallId[funcOp.getOperation()]++ + (vfCount ? vfCount.getInt() : 0);
            std::string callIdDiagnostic;
            FailureOr<uint16_t> callId = getPrintTensorCallId(ordinal, callIdDiagnostic);
            if (failed(callId)) {
                op.emitError(callIdDiagnostic);
                signalPassFailure();
                return;
            }
            PatternRewriter rewriter(op.getContext());
            rewriter.setInsertionPoint(op);
            auto tensorType = op.getValue().getType();
            AddressSpace addressSpace = tensorType.getPtr().getAddrspace();
            bool usesDebugTunnel = addressSpace == AddressSpace::l1 || addressSpace == AddressSpace::l0c;
            if (failed(lowerPrintTensor(op, rewriter, module, *callId))) {
                signalPassFailure();
                return;
            }
            if (usesDebugTunnel && tunnelFunctionSet.insert(funcOp.getOperation()).second)
                tunnelFunctions.push_back(funcOp);
        }
        constexpr StringLiteral kInitDebugTunnel = "_mlir_ciface_init_debug";
        constexpr StringLiteral kFinishDebugTunnel = "_mlir_ciface_finish_debug";
        for (func::FuncOp funcOp : tunnelFunctions) {
            BlockArgument stateArg;
            for (BlockArgument arg : funcOp.getArguments()) {
                if (funcOp.getArgAttr(arg.getArgNumber(), kDebugTunnelStateAttrName)) {
                    stateArg = arg;
                    break;
                }
            }
            if (!stateArg) {
                funcOp.emitError("local tensor DebugTunnel lifecycle is missing its hidden state argument");
                signalPassFailure();
                return;
            }
            auto makeTunnelLifecycleCall = [&](StringRef name) {
                auto lifecycle = getOrCreateDebugRuntimeCall(module, name, ArrayRef<Type>{stateArg.getType()});
                lifecycle->setAttr(
                    hivm::TFuncCoreTypeAttr::name,
                    hivm::TFuncCoreTypeAttr::get(funcOp.getContext(), hivm::TFuncCoreType::AIC));
                lifecycle->setAttr("llvm.emit_c_interface", UnitAttr::get(funcOp.getContext()));
                return lifecycle;
            };
            auto init = makeTunnelLifecycleCall(kInitDebugTunnel);
            auto finish = makeTunnelLifecycleCall(kFinishDebugTunnel);
            OpBuilder entryBuilder(funcOp.getContext());
            entryBuilder.setInsertionPointToStart(&funcOp.getBody().front());
            entryBuilder.create<func::CallOp>(funcOp.getLoc(), init, ValueRange{stateArg});
            SmallVector<func::ReturnOp, 2> returns;
            funcOp.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
            if (returns.empty()) {
                funcOp.emitError("local tensor DebugTunnel lifecycle requires a kernel return");
                signalPassFailure();
                return;
            }
            for (func::ReturnOp returnOp : returns) {
                OpBuilder exitBuilder(returnOp);
                exitBuilder.create<func::CallOp>(returnOp.getLoc(), finish, ValueRange{stateArg});
            }
        }
        normalizePrintArguments(module);
    }
};

} // namespace

std::unique_ptr<Pass> createTlaLowerDebugPrintPass()
{
    return std::make_unique<TlaLowerDebugPrintPass>();
}

void registerTlaLowerDebugPrintPass()
{
    PassRegistration<TlaLowerDebugPrintPass>();
}

} // namespace tla
