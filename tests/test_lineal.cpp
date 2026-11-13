// La capa lineal AVX2 contra la ingenua con matrices aleatorias, y el cálculo por trozos de neuronas
// (el que usan los hilos) contra el cálculo de una vez.
#include "cpu.hpp"
#include "operaciones.hpp"
#include "prueba.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

struct Caso {
    size_t n_filas, n_in, n_out;
};

std::vector<float> aleatorio(size_t n, std::mt19937& azar) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& f : v) f = u(azar);
    return v;
}

// La capa entera calculada en trozos de 3 neuronas, como haría un reparto entre hilos.
std::vector<float> por_trozos(Lineal lineal, const Caso& c, const float* x, const float* w, const float* b) {
    std::vector<float> y(c.n_filas * c.n_out);
    for (size_t ini = 0; ini < c.n_out; ini += 3) lineal(x, c.n_filas, c.n_in, w, b, c.n_out, y.data(), ini, std::min(ini + 3, c.n_out));
    return y;
}

int main() {
    COMPROBAR(cpu_tiene_avx2_fma());
    if (!cpu_tiene_avx2_fma()) return resumen();

    std::mt19937 azar(1);
    // Tamaños reales del modelo y tamaños raros: n_in que no es múltiplo de 8 y n_out que no lo es de 4.
    for (const Caso& c : {Caso{1, 768, 2304}, Caso{3, 3072, 50}, Caso{2, 100, 7}, Caso{4, 13, 5}, Caso{1, 8, 4}, Caso{5, 7, 1}, Caso{2, 1, 9}}) {
        const std::vector<float> x = aleatorio(c.n_filas * c.n_in, azar);
        const std::vector<float> w = aleatorio(c.n_out * c.n_in, azar);
        const std::vector<float> sesgo = aleatorio(c.n_out, azar);

        for (const float* b : {sesgo.data(), static_cast<const float*>(nullptr)}) {
            std::vector<float> ingenua(c.n_filas * c.n_out), avx2(c.n_filas * c.n_out);
            lineal_ingenua(x.data(), c.n_filas, c.n_in, w.data(), b, c.n_out, ingenua.data(), 0, c.n_out);
            lineal_avx2(x.data(), c.n_filas, c.n_in, w.data(), b, c.n_out, avx2.data(), 0, c.n_out);

            // Mismo cálculo con las sumas en otro orden: solo puede diferir en el redondeo.
            float error = 0.0f;
            for (size_t i = 0; i < ingenua.size(); ++i) error = std::max(error, std::fabs(ingenua[i] - avx2[i]));
            COMPROBAR(error < 1e-3f);

            // Por trozos el resultado es idéntico bit a bit: cada neurona se calcula igual esté donde esté.
            COMPROBAR(por_trozos(lineal_ingenua, c, x.data(), w.data(), b) == ingenua);
            COMPROBAR(por_trozos(lineal_avx2, c, x.data(), w.data(), b) == avx2);
        }
    }
    return resumen();
}
