// El grupo de hilos reparte cada índice exactamente una vez, con cualquier tamaño de trozo y en
// muchas rondas seguidas (que es como lo usa el modelo: una ronda por capa lineal).
#include "hilos.hpp"
#include "prueba.hpp"

#include <atomic>
#include <vector>

int main() {
    for (int n_hilos : {1, 2, 4, 8, 20}) {
        Hilos hilos(n_hilos);
        COMPROBAR(hilos.n() == n_hilos);

        for (size_t total : {size_t{0}, size_t{1}, size_t{7}, size_t{768}, size_t{50257}}) {
            for (size_t trozo : {size_t{1}, size_t{4}, size_t{100}, size_t{100000}}) {
                std::vector<std::atomic<int>> veces(total);
                std::atomic<size_t> fuera_de_rango{0};
                hilos.paralelo(total, trozo, [&](size_t ini, size_t fin) {
                    if (ini >= fin || fin > total || fin - ini > trozo) ++fuera_de_rango;
                    for (size_t i = ini; i < fin; ++i) ++veces[i];
                });
                size_t mal = fuera_de_rango;
                for (const auto& v : veces) mal += v != 1;
                COMPROBAR(mal == 0);
            }
        }

        // Muchas rondas cortas: ningún trabajador se queda dormido ni repite una ronda.
        std::atomic<long long> suma{0};
        for (int ronda = 0; ronda < 5000; ++ronda) {
            hilos.paralelo(64, 4, [&](size_t ini, size_t fin) { suma += static_cast<long long>(fin - ini); });
        }
        COMPROBAR(suma == 5000LL * 64);
    }
    return resumen();
}
