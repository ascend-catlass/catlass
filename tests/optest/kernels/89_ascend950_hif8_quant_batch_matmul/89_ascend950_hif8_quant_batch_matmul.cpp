#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void Ascend950QuantBatchMatmulFixpipeDequant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_89_ascend950_hif8_quant_batch_matmul", tParams);
    // Raw uint8 inputs are interpreted as HiF8 by the JIT kernel.
    macros["CATLASS_JIT_ELEMENT_A"] = "hifloat8_t";
    macros["CATLASS_JIT_ELEMENT_B"] = "hifloat8_t";
    macros["L2_CACHE_HINT"] = "1";
    auto quantModeIt = tParams.element.find("QUANT_MODE");
    int quantMode = (quantModeIt != tParams.element.end()) ? static_cast<int>(quantModeIt->second) : 0;
    macros["CATLASS_JIT_QUANT_MODE"] = std::to_string(quantMode);
    auto* entry =
        JitCompiler::instance().getKernel("89_ascend950_hif8_quant_batch_matmul_impl.cpp", macros, JitKernelType::AIC);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
