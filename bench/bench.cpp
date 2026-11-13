// Benchmark reproducible: tokens por segundo de la generación greedy con un prompt fijo.
// uso: bench [--tokens N] [--repeticiones R] [--sin-base]
//   --sin-base  omite la variante sin caché, que tarda minutos
#include "generar.hpp"
#include "tokenizador.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

const char* const PROMPT = "The history of computing began long before the first electronic machines, when";

struct Variante {
    const char* nombre;
    bool kv_cache;
    Opciones opciones;
};

// Cada paso de optimización se suma a los anteriores. El barrido de hilos va al final.
const Variante VARIANTES[] = {
    {"Ingenua, sin cache", false, {Matmul::Ingenua, 1}},
    {"+ KV cache", true, {Matmul::Ingenua, 1}},
    {"+ AVX2 y FMA", true, {Matmul::Avx2, 1}},
    {"+ hilos", true, {Matmul::Avx2, 2}},
    {"+ hilos", true, {Matmul::Avx2, 4}},
    {"+ hilos", true, {Matmul::Avx2, 6}},
    {"+ hilos", true, {Matmul::Avx2, 8}},
    {"+ hilos", true, {Matmul::Avx2, 14}},
    {"+ hilos", true, {Matmul::Avx2, 20}},
};

// Tokens por segundo de una generación de n_tokens: mediana de las repeticiones.
// El tiempo incluye procesar el prompt; no incluye cargar el modelo.
double medir(const Modelo& modelo, const std::vector<int>& prompt, int n_tokens, int repeticiones, bool kv_cache) {
    std::vector<double> velocidades;
    for (int r = 0; r < repeticiones; ++r) {
        const auto inicio = std::chrono::steady_clock::now();
        const std::vector<int> ids = generar(modelo, prompt, n_tokens, {}, -1, {}, kv_cache);
        const double segundos = std::chrono::duration<double>(std::chrono::steady_clock::now() - inicio).count();
        velocidades.push_back(static_cast<double>(ids.size() - prompt.size()) / segundos);
    }
    std::ranges::sort(velocidades);
    return velocidades[velocidades.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int n_tokens = 100, repeticiones = 3;
    bool con_base = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sin-base") == 0) con_base = false;
        else if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) n_tokens = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--repeticiones") == 0 && i + 1 < argc) repeticiones = std::atoi(argv[++i]);
        else n_tokens = 0;  // argumento desconocido: muestra el uso
    }
    if (n_tokens < 1 || repeticiones < 1) {
        std::fprintf(stderr, "uso: bench [--tokens N] [--repeticiones R] [--sin-base]\n");
        return 2;
    }

    try {
        const Tokenizador tok("modelos/gpt2_vocab.bin");
        Modelo modelo(cargar_pesos("modelos/gpt2_124M.bin"));
        const std::vector<int> prompt = tok.codificar(PROMPT);

        std::printf("prompt de %zu tokens, %d tokens generados, mediana de %d repeticiones\n\n", prompt.size(), n_tokens, repeticiones);
        std::printf("| Variante | Hilos | tokens/s | Frente a la base | Frente a 1 hilo |\n|---|---:|---:|---:|---:|\n");
        double base = 0.0, un_hilo = 0.0;  // velocidades de referencia; 0 mientras no se hayan medido
        for (const Variante& v : VARIANTES) {
            if (!v.kv_cache && !con_base) continue;
            modelo.configurar(v.opciones);
            const double velocidad = medir(modelo, prompt, n_tokens, repeticiones, v.kv_cache);
            if (!v.kv_cache) base = velocidad;
            if (v.opciones.matmul == Matmul::Avx2 && v.opciones.hilos == 1) un_hilo = velocidad;

            std::printf("| %s | %d | %.2f |", v.nombre, v.opciones.hilos, velocidad);
            if (base > 0.0) std::printf(" x%.1f |", velocidad / base);
            else std::printf(" |");
            if (v.opciones.hilos > 1) std::printf(" x%.2f |\n", velocidad / un_hilo);
            else std::printf(" |\n");
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
