#include "cpu.hpp"

#include <immintrin.h>  // _xgetbv
#include <intrin.h>     // __cpuid, __cpuidex

bool cpu_tiene_avx2_fma() {
    const auto bit = [](int registro, int n) { return ((registro >> n) & 1) != 0; };
    int r[4];  // EAX, EBX, ECX, EDX

    __cpuid(r, 0);
    if (r[0] < 7) return false;  // la hoja 7 de CPUID no existe

    __cpuid(r, 1);
    const bool fma = bit(r[2], 12), osxsave = bit(r[2], 27), avx = bit(r[2], 28);
    if (!fma || !osxsave || !avx) return false;

    // No basta con la CPU: el sistema operativo tiene que guardar los registros XMM (bit 1 de
    // XCR0) e YMM (bit 2) en los cambios de contexto. OSXSAVE indica que XGETBV se puede usar.
    if ((_xgetbv(0) & 0x6) != 0x6) return false;

    __cpuidex(r, 7, 0);
    return bit(r[1], 5);  // AVX2
}
