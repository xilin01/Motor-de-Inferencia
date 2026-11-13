// Comprueba que el compilador acepta C++20 y que el mini-marco de pruebas detecta un fallo.
#include "prueba.hpp"
#include <span>
#include <vector>

int main() {
    std::vector<float> v{1.0f, 2.0f, 3.0f};
    std::span<const float> s(v);
    COMPROBAR(s.size() == 3);

    COMPROBAR(s.size() == 4);  // fallo provocado: debe contarse
    COMPROBAR(g_fallos == 1);
    g_fallos = 0;
    return resumen();
}
