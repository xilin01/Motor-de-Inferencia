// Los tests del modelo se lanzan una vez por variante del motor; la variante llega por argumentos.
#pragma once
#include "modelo.hpp"

#include <stdexcept>
#include <string>

// Lee [ingenua|avx2] a partir de argv[primero]. Sin argumento: todo ingenuo.
inline Opciones leer_variante(int argc, char** argv, int primero) {
    Opciones op;
    if (argc > primero) {
        const std::string matmul = argv[primero];
        if (matmul == "avx2") op.matmul = Matmul::Avx2;
        else if (matmul != "ingenua") throw std::invalid_argument("variante desconocida: " + matmul);
    }
    return op;
}
