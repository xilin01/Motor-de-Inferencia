// GPT-2: del vector de ids de token a los logits del siguiente token.
#pragma once
#include "pesos.hpp"

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

class Modelo {
public:
    // Se queda con los pesos y comprueba que cada tensor tiene la forma que dice la configuración.
    explicit Modelo(Pesos pesos);

    // Forward ingenuo: recalcula todo el contexto. Devuelve los logits de todas las posiciones,
    // [n_tokens, n_vocab] en fila-mayor: la fila t predice el token t + 1.
    // Con solo_ultima se devuelve únicamente la última fila, [n_vocab]: es lo que necesita la
    // generación, y ahorra aplicar la cabeza final a las demás posiciones.
    // Lanza std::invalid_argument si ids está vacío, supera n_ctx o contiene un id inválido.
    std::vector<float> forward(std::span<const int> ids, bool solo_ultima = false) const;

    const Config& config() const { return pesos_.config; }

private:
    Pesos pesos_;
    const float *wte_, *wpe_, *lnf_g_, *lnf_b_;
    std::vector<Capa> capas_;
};
