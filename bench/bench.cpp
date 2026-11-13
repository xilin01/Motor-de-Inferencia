// Benchmark reproducible: tokens por segundo de la generación greedy con un prompt fijo.
// uso: bench [--tokens N] [--repeticiones R]
#include "generar.hpp"
#include "tokenizador.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

const char* const PROMPT = "The history of computing began long before the first electronic machines, when";

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
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--tokens") == 0) n_tokens = std::atoi(argv[i + 1]);
        if (std::strcmp(argv[i], "--repeticiones") == 0) repeticiones = std::atoi(argv[i + 1]);
    }
    if (n_tokens < 1 || repeticiones < 1) {
        std::fprintf(stderr, "uso: bench [--tokens N] [--repeticiones R]\n");
        return 2;
    }

    try {
        const Tokenizador tok("modelos/gpt2_vocab.bin");
        const Modelo modelo(cargar_pesos("modelos/gpt2_124M.bin"));
        const std::vector<int> prompt = tok.codificar(PROMPT);

        std::printf("prompt de %zu tokens, %d tokens generados, mediana de %d repeticiones\n\n", prompt.size(), n_tokens, repeticiones);
        std::printf("| Variante | tokens/s | Frente a la base |\n|---|---:|---:|\n");
        const double base = medir(modelo, prompt, n_tokens, repeticiones, false);
        const auto fila = [&](const char* nombre, double v) {
            std::printf("| %s | %.3f | x%.1f |\n", nombre, v, v / base);
            std::fflush(stdout);
        };
        fila("Ingenua, sin cache, 1 hilo", base);
        fila("KV cache, 1 hilo", medir(modelo, prompt, n_tokens, repeticiones, true));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
