// Mini-marco de pruebas sin dependencias: COMPROBAR cuenta fallos y resumen() da el código de salida.
#pragma once
#include <cstdio>

inline int g_fallos = 0;

#define COMPROBAR(cond)                                                          \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ++g_fallos;                                                          \
            std::printf("FALLO %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
        }                                                                        \
    } while (0)

inline int resumen() {
    if (g_fallos == 0) std::printf("OK\n");
    else std::printf("%d fallos\n", g_fallos);
    return g_fallos == 0 ? 0 : 1;
}
