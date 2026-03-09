// Runtime SVE2 detection — compiled WITHOUT SVE2 flags so it's safe to call
// on any AArch64 hardware.
#include "optmath/sve2_kernels.hpp"

#if defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#endif

namespace optmath {
namespace sve2 {

bool is_available() {
#ifdef OPTMATH_USE_SVE2
    // Compiled with SVE2 support — check if hardware actually has it.
#if defined(__aarch64__)
    static const bool has_sve2 = (getauxval(AT_HWCAP2) & HWCAP2_SVE2) != 0;
    return has_sve2;
#else
    return false;
#endif
#else
    return false;
#endif
}

} // namespace sve2
} // namespace optmath
