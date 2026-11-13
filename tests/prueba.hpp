// Mini-marco de pruebas sin dependencias: COMPROBAR cuenta fallos y resumen() da el código de salida.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <exception>

inline int g_fallos = 0;

// Una excepción sin capturar (por ejemplo, falta modelos/) termina el test en silencio: se muestra antes.
inline const std::terminate_handler g_terminar_anterior = std::set_terminate([] {
    try {
        if (std::current_exception()) std::rethrow_exception(std::current_exception());
    } catch (const std::exception& e) {
        std::printf("EXCEPCION sin capturar: %s\n", e.what());
    } catch (...) {
    }
    std::fflush(stdout);
    std::_Exit(1);
});

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
