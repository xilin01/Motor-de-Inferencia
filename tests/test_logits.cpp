// Compara los logits del motor con los de PyTorch en todas las posiciones de cada caso de referencia.
// El archivo modelos/ref_logits.bin lo genera tools/referencias.py.
//
// Tolerancia. El objetivo inicial era 1e-4 absoluto y no es alcanzable en float32: el propio PyTorch
// en float32 se aparta del mismo modelo en float64 entre 5,7e-5 y 2,9e-4 en los prompts cortos y
// 2,5e-3 con el contexto lleno. Umbrales acordados: 1e-3 en los prompts cortos (error medido del
// forward ingenuo: 7,9e-4) y 1e-2 con 1024 tokens (medido: 4,9e-3).
#include "modelo.hpp"
#include "prueba.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

// Mayor diferencia absoluta entre dos vectores.
float error_maximo(const std::vector<float>& a, const std::vector<float>& b) {
    float maximo = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) maximo = std::max(maximo, std::fabs(a[i] - b[i]));
    return maximo;
}

template <typename T>
std::vector<T> leer(std::ifstream& f, size_t n) {
    std::vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
    if (!f) throw std::runtime_error("modelos/ref_logits.bin truncado: vuelve a ejecutar tools/referencias.py");
    return v;
}

int main() {
    const Modelo modelo(cargar_pesos("modelos/gpt2_124M.bin"));
    const size_t V = modelo.config().n_vocab;

    std::ifstream f("modelos/ref_logits.bin", std::ios::binary);
    COMPROBAR(f.is_open());
    COMPROBAR(std::memcmp(leer<char>(f, 8).data(), "GPT2REFL", 8) == 0);
    const uint32_t n_casos = leer<uint32_t>(f, 1)[0];
    COMPROBAR(n_casos == 7);

    std::printf("tokens | motor vs PyTorch f32 | motor vs float64 | PyTorch f32 vs float64\n");
    for (uint32_t c = 0; c < n_casos; ++c) {
        const uint32_t n = leer<uint32_t>(f, 1)[0];
        const std::vector<int> ids = leer<int>(f, n);
        const std::vector<float> pytorch = leer<float>(f, n * V);  // PyTorch en float32
        const std::vector<float> exactos = leer<float>(f, n * V);  // PyTorch en float64, redondeado a float32

        const std::vector<float> logits = modelo.forward(ids);
        COMPROBAR(logits.size() == pytorch.size());
        const float e = error_maximo(logits, pytorch);
        std::printf("%6u | %20.2e | %16.2e | %22.2e\n", n, e, error_maximo(logits, exactos), error_maximo(pytorch, exactos));
        COMPROBAR(e <= (n == modelo.config().n_ctx ? 1e-2f : 1e-3f));
    }

    // Entradas inválidas: contexto vacío, demasiado largo e id fuera del vocabulario.
    const auto lanza = [&](const std::vector<int>& ids) {
        try { modelo.forward(ids); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    COMPROBAR(lanza({}));
    COMPROBAR(lanza(std::vector<int>(1025, 0)));
    COMPROBAR(lanza({50257}));
    COMPROBAR(lanza({-1}));
    return resumen();
}
