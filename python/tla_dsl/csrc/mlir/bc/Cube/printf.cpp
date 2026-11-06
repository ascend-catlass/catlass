#include "../common.h"
#include "catlass/catlass.hpp"
#include "kernel_operator.h"

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
#include "../print_scalar.h"
#endif
