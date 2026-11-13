// Compara los logits del motor con los de PyTorch en todas las posiciones de cada caso de referencia.
// El archivo modelos/ref_logits.bin lo genera tools/referencias.py.
//
// uso: test_logits [completo|incremental] [ingenua|avx2] [hilos]
//   completo:     Modelo::forward sobre el contexto entero (por defecto).
//   incremental:  Modelo::avanzar con KV cache, dando los tokens en trozos de 1 a 3.
//
// Tolerancia. El objetivo inicial era 1e-4 absoluto y no es alcanzable en float32: el propio PyTorch
// en float32 se aparta del mismo modelo en float64 entre 5,7e-5 y 2,9e-4 en los prompts cortos y
// 2,5e-3 con el contexto lleno. Umbrales acordados: 1e-3 en los prompts cortos (error medido del
// forward ingenuo: 7,9e-4) y 1e-2 con 1024 tokens (medido: 4,9e-3).
#include "modelo.hpp"
#include "prueba.hpp"
#include "variante.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

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

// Logits de todas las posiciones obtenidos con la caché, alimentando los tokens en trozos de
// tamaños variados para ejercitar tanto el paso de un token como el de varios.
std::vector<float> logits_incrementales(const Modelo& modelo, const std::vector<int>& ids) {
    const size_t trozos[] = {3, 1, 1, 2, 1};
    Cache cache;
    std::vector<float> logits;
    for (size_t i = 0, paso = 0; i < ids.size(); ++paso) {
        const size_t m = std::min(trozos[paso % std::size(trozos)], ids.size() - i);
        const std::vector<float> nuevos = modelo.avanzar(cache, std::span(ids).subspan(i, m));
        logits.insert(logits.end(), nuevos.begin(), nuevos.end());
        i += m;
    }
    return logits;
}

template <typename F>
bool lanza(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

int main(int argc, char** argv) {
    const std::string modo = argc > 1 ? argv[1] : "completo";
    if (modo != "completo" && modo != "incremental") {
        std::fprintf(stderr, "uso: test_logits [completo|incremental] [ingenua|avx2] [hilos]\n");
        return 2;
    }
    const bool incremental = modo == "incremental";

    Modelo modelo(cargar_pesos("modelos/gpt2_124M.bin"));
    const Opciones variante = leer_variante(argc, argv, 2);
    modelo.configurar(variante);
    const size_t V = modelo.config().n_vocab;

    std::ifstream f("modelos/ref_logits.bin", std::ios::binary);
    COMPROBAR(f.is_open());
    COMPROBAR(std::memcmp(leer<char>(f, 8).data(), "GPT2REFL", 8) == 0);
    const uint32_t n_casos = leer<uint32_t>(f, 1)[0];
    COMPROBAR(n_casos == 7);

    std::printf("modo %s, matmul %s, %d hilos\n", modo.c_str(), argc > 2 ? argv[2] : "ingenua", variante.hilos);
    std::printf("tokens | motor vs PyTorch f32 | motor vs float64 | PyTorch f32 vs float64\n");
    for (uint32_t c = 0; c < n_casos; ++c) {
        const uint32_t n = leer<uint32_t>(f, 1)[0];
        const std::vector<int> ids = leer<int>(f, n);
        const std::vector<float> pytorch = leer<float>(f, n * V);  // PyTorch en float32
        const std::vector<float> exactos = leer<float>(f, n * V);  // PyTorch en float64, redondeado a float32

        const std::vector<float> logits = incremental ? logits_incrementales(modelo, ids) : modelo.forward(ids);
        COMPROBAR(logits.size() == pytorch.size());
        const float e = error_maximo(logits, pytorch);
        std::printf("%6u | %20.2e | %16.2e | %22.2e\n", n, e, error_maximo(logits, exactos), error_maximo(pytorch, exactos));
        COMPROBAR(e <= (n == modelo.config().n_ctx ? 1e-2f : 1e-3f));

        // La caché no cambia ninguna operación, solo evita repetirlas: en los casos cortos se exige
        // que los logits incrementales sean idénticos bit a bit a los del forward completo.
        if (incremental && n <= 64) COMPROBAR(logits == modelo.forward(ids));

        // Los hilos se reparten neuronas de salida y cada neurona se calcula igual la coja quien la
        // coja: con varios hilos el resultado debe ser idéntico bit a bit al de un solo hilo.
        if (variante.hilos > 1 && n <= 64) {
            modelo.configurar({variante.matmul, 1});
            COMPROBAR(logits == modelo.forward(ids));
            modelo.configurar(variante);
        }
    }

    // Entradas inválidas: contexto vacío, demasiado largo e id fuera del vocabulario.
    if (incremental) {
        Cache cache;
        modelo.avanzar(cache, std::vector<int>{15496, 11});
        COMPROBAR(lanza([&] { modelo.avanzar(cache, {}); }));
        COMPROBAR(lanza([&] { modelo.avanzar(cache, std::vector<int>(1023, 0)); }));
        COMPROBAR(lanza([&] { modelo.avanzar(cache, std::vector<int>{50257}); }));
        COMPROBAR(cache.n == 2);  // una llamada rechazada deja la caché como estaba
        Cache llena;
        llena.n = modelo.config().n_ctx;
        COMPROBAR(lanza([&] { modelo.avanzar(llena, std::vector<int>{0}); }));
    } else {
        COMPROBAR(lanza([&] { modelo.forward({}); }));
        COMPROBAR(lanza([&] { modelo.forward(std::vector<int>(1025, 0)); }));
        COMPROBAR(lanza([&] { modelo.forward(std::vector<int>{50257}); }));
        COMPROBAR(lanza([&] { modelo.forward(std::vector<int>{-1}); }));
    }
    return resumen();
}
