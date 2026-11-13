// Los tests del modelo se lanzan una vez por variante del motor; la variante llega por argumentos.
#pragma once
#include "modelo.hpp"

#include <stdexcept>
#include <string>

// Lee [ingenua|avx2] [hilos] a partir de argv[primero]. Sin argumentos: todo ingenuo, un hilo.
inline Opciones leer_variante(int argc, char** argv, int primero) {
    Opciones op;
    if (argc > primero) {
        const std::string matmul = argv[primero];
        if (matmul == "avx2") op.matmul = Matmul::Avx2;
        else if (matmul != "ingenua") throw std::invalid_argument("variante desconocida: " + matmul);
    }
    if (argc > primero + 1) op.hilos = std::stoi(argv[primero + 1]);
    return op;
}
