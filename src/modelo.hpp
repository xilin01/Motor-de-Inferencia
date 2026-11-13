// GPT-2: del vector de ids de token a los logits del siguiente token.
#pragma once
#include "hilos.hpp"
#include "operaciones.hpp"
#include "pesos.hpp"

#include <memory>
#include <span>
#include <vector>

// Punteros a los pesos de un bloque transformer. Las matrices van como [salida, entrada].
struct Capa {
    const float *ln1_g, *ln1_b;            // LayerNorm previa a la atención
    const float *attn_w, *attn_b;          // [3C, C]: proyección conjunta a consulta, clave y valor
    const float *attn_proy_w, *attn_proy_b;  // [C, C]: proyección de salida de la atención
    const float *ln2_g, *ln2_b;            // LayerNorm previa al MLP
    const float *fc_w, *fc_b;              // [4C, C]: primera capa del MLP
    const float *mlp_proy_w, *mlp_proy_b;  // [C, 4C]: segunda capa del MLP
};

// KV cache: claves y valores de atención de las posiciones ya procesadas, por capa.
// Una caché vacía (recién construida) representa un contexto sin tokens.
struct Cache {
    size_t n = 0;                                    // posiciones ya procesadas
    std::vector<std::vector<float>> claves, valores;  // por capa: [n, C] en fila-mayor
};

enum class Matmul { Ingenua, Avx2 };

// Variante de cálculo. No cambia qué se calcula, solo cómo.
struct Opciones {
    Matmul matmul = Matmul::Ingenua;  // implementación de las capas lineales
    int hilos = 1;                    // hilos entre los que se reparte cada capa lineal
};

class Modelo {
public:
    // Se queda con los pesos y comprueba que cada tensor tiene la forma que dice la configuración.
    // Empieza con las opciones por defecto: todo ingenuo.
    explicit Modelo(Pesos pesos);

    // Cambia la variante de cálculo. Lanza std::runtime_error si se pide AVX2 y la CPU no lo tiene,
    // y std::invalid_argument si hilos no está entre 1 y 256.
    void configurar(const Opciones& opciones);

    // Forward ingenuo: recalcula todo el contexto. Devuelve los logits de todas las posiciones,
    // [n_tokens, n_vocab] en fila-mayor: la fila t predice el token t + 1.
    // Con solo_ultima se devuelve únicamente la última fila, [n_vocab]: es lo que necesita la
    // generación, y ahorra aplicar la cabeza final a las demás posiciones.
    // Lanza std::invalid_argument si ids está vacío, supera n_ctx o contiene un id inválido.
    std::vector<float> forward(std::span<const int> ids, bool solo_ultima = false) const;

    // Forward incremental: procesa solo los tokens nuevos, que ocupan las posiciones siguientes a
    // las cache.n ya procesadas, y añade sus claves y valores a la caché. Devuelve los logits de
    // los tokens nuevos, [nuevos.size(), n_vocab], o solo los del último con solo_ultima.
    // Lanza std::invalid_argument con las mismas condiciones que forward; la caché queda intacta.
    std::vector<float> avanzar(Cache& cache, std::span<const int> nuevos, bool solo_ultima = false) const;

    const Config& config() const { return pesos_.config; }

private:
    void comprobar(size_t n_previos, std::span<const int> ids) const;

    // Capa lineal completa con la variante elegida en configurar().
    void lineal(const float* x, size_t n_filas, size_t n_in, const float* w, const float* b, size_t n_out, float* y) const;

    Lineal nucleo_ = lineal_ingenua;
    std::unique_ptr<Hilos> hilos_;  // vacío con un solo hilo

    Pesos pesos_;
    const float *wte_, *wpe_, *lnf_g_, *lnf_b_;
    std::vector<Capa> capas_;
};
