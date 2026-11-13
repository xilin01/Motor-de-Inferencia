// Detección en tiempo de ejecución de las extensiones que usa la variante AVX2.
#pragma once

// true si la CPU tiene AVX2 y FMA y el sistema operativo conserva los registros YMM.
bool cpu_tiene_avx2_fma();
