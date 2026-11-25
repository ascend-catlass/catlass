#include <cstdint>
#include "kernel_operator.h"

extern "C" __aicore__ __attribute__((always_inline)) uint64_t tla_get_system_cycle()
{
    uint64_t ticks = 0;
    asm volatile("MOV %0, SYS_CNT" : "+l"(ticks));
    return ticks;
}
