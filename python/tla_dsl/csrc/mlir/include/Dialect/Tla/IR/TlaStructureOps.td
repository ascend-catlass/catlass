//===----------------------------------------------------------------------===//
// Function structure and debug/extern-call ops.
//
// Included by Dialect/Tla/IR/Tla.td, the file mlir-tblgen is run on.
//===----------------------------------------------------------------------===//

def Tla_FuncOp : Tla_Op<"func", []> {
  let summary = "Tla kernel function";
  let description = [{
    Defines a Tla kernel entry point. The body is an isolated region with Tla IR operations.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$function_type);
  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
}

def Tla_DebugPrintOp : Tla_Op<"debug_print", [MemoryEffects<[MemWrite]>]> {
  let summary = "Typed frontend debug print op";
  let description = [{
    Side-effecting TLA IR for legacy one-scalar debug print or compile-time
    string/formatted scalar print. The legacy assembly remains:

      tla.debug_print %value : type

    Formatted calls carry the validated user format string as metadata plus
    zero or more runtime scalar operands:

      tla.debug_print format "x={}"(%value) : (type)

    Formatted calls lower through the same private debug-print backend and
    scalar FIFO transport as legacy scalar debug print.

    Legacy operands support f16, f32, i8, i16, i32, u8, u16, and u32.
    Formatted operands support signless i32 and f32.
  }];
  let arguments = (ins Variadic<AnyType>:$values,
                       OptionalAttr<StrAttr>:$format);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Tla_CallExternOp : Tla_Op<"call_extern", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call a user-provided device function";
  let description = [{
    Calls one C ABI device function supplied by user LLVM bitcode. Pointer
    operands remain !tla.ptr until tla-lower-ptr converts them to i64 byte
    addresses. SIMD calls also accept full 256-byte registers of f16, bf16,
    f32, and 8/16/32/64-bit integer elements, and may return one such register.
    Each argument and result has its own element type.
    Calls conservatively model unknown memory reads and writes.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let hasVerifier = 1;
}

def Tla_PrintTensorOp : Tla_Op<"print_tensor", [MemoryEffects<[MemWrite]>]> {
  let summary = "Print a supported tensor prefix from GM, UB, L1, or L0C";
  let description = [{
    Side-effecting native tensor print marker. The Python API accepts rank-1 or
    rank-2 tensors with element type f16, f32, i8, i16, i32, u8, u16, or u32.
    GM is supported in AIC and AIV regions, UB in AIV regions, and the supported
    zN/nZ L1 and L0C layouts in AIC regions. Values are read as a contiguous
    physical prefix from the effective tensor address; ``shape`` is logical
    display metadata, not a request to gather through tensor strides. Dynamic
    shape leaves use ``-1`` and are materialized from the descriptor during
    lowering. ``length`` may be a runtime i64. ``dynamic_execution`` records
    that the site is nested under runtime control flow so the host warns rather
    than rejecting an aggregate FIFO bound that cannot be known statically.
  }];
  let arguments = (ins Tla_TensorType:$value, I64:$length,
                       DenseI64ArrayAttr:$shape,
                       OptionalAttr<UnitAttr>:$dynamic_execution);
  let assemblyFormat = [{
    $value `length` `=` $length `shape` `=` $shape attr-dict `:`
    type($value) `,` type($length)
  }];
  let hasVerifier = 1;
}

def Tla_PrintVectorOp : Tla_Op<"print_vector", [MemoryEffects<[MemWrite]>]> {
  let summary = "Print a register prefix from a real SIMD vector function";
  let arguments = (ins Tla_VectorSSAType:$value, I64:$length);
  let assemblyFormat = "$value `length` `=` $length attr-dict `:` qualified(type($value)) `,` type($length)";
  let hasVerifier = 1;
}

def Tla_ReturnOp : Tla_Op<"return", [Terminator]> {
  let summary = "Return";
  let description = [{
    Terminator for tla.func regions.
  }];
  let assemblyFormat = [{
    attr-dict
  }];
}
