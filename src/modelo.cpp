#include "modelo.hpp"

#include "operaciones.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

Modelo::Modelo(Pesos pesos) : pesos_(std::move(pesos)) {
    const Config& c = pesos_.config;
    if (c.n_cabezas == 0 || c.n_embd % c.n_cabezas != 0) throw std::runtime_error("n_embd no es múltiplo de n_cabezas");
    const uint32_t C = c.n_embd;

    wte_ = pesos_.tensor("wte.weight", {c.n_vocab, C});
    wpe_ = pesos_.tensor("wpe.weight", {c.n_ctx, C});
    lnf_g_ = pesos_.tensor("ln_f.weight", {C});
    lnf_b_ = pesos_.tensor("ln_f.bias", {C});
    for (uint32_t i = 0; i < c.n_capas; ++i) {
        const std::string p = "h." + std::to_string(i) + ".";
        capas_.push_back({
            pesos_.tensor(p + "ln_1.weight", {C}),             pesos_.tensor(p + "ln_1.bias", {C}),
            pesos_.tensor(p + "attn.c_attn.weight", {3 * C, C}), pesos_.tensor(p + "attn.c_attn.bias", {3 * C}),
            pesos_.tensor(p + "attn.c_proj.weight", {C, C}),   pesos_.tensor(p + "attn.c_proj.bias", {C}),
            pesos_.tensor(p + "ln_2.weight", {C}),             pesos_.tensor(p + "ln_2.bias", {C}),
            pesos_.tensor(p + "mlp.c_fc.weight", {4 * C, C}),  pesos_.tensor(p + "mlp.c_fc.bias", {4 * C}),
            pesos_.tensor(p + "mlp.c_proj.weight", {C, 4 * C}), pesos_.tensor(p + "mlp.c_proj.bias", {C}),
        });
    }
}

std::vector<float> Modelo::forward(std::span<const int> ids) const {
    const Config& cfg = pesos_.config;
    const size_t T = ids.size();         // tokens del contexto
    const size_t C = cfg.n_embd;         // dimensión del modelo
    const size_t H = cfg.n_cabezas;      // cabezas de atención
    const size_t D = C / H;              // dimensión de cada cabeza
    if (T == 0 || T > cfg.n_ctx) throw std::invalid_argument("el contexto debe tener entre 1 y n_ctx tokens");
    for (int id : ids) {
        if (id < 0 || static_cast<uint32_t>(id) >= cfg.n_vocab) throw std::invalid_argument("id de token fuera del vocabulario");
    }

    // 1. Embeddings: x[t] = wte[token t] + wpe[posición t].
    std::vector<float> x(T * C);
    for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < C; ++c) x[t * C + c] = wte_[ids[t] * C + c] + wpe_[t * C + c];
    }

    std::vector<float> norm(T * C);      // salida de cada LayerNorm
    std::vector<float> qkv(T * 3 * C);   // por posición: consulta (C), clave (C) y valor (C)
    std::vector<float> mezcla(T * C);    // salida de la atención antes de proyectar
    std::vector<float> oculta(T * 4 * C);  // capa oculta del MLP
    std::vector<float> rama(T * C);      // lo que cada sub-bloque suma a x
    std::vector<float> pesos_at(T);      // pesos de atención de una consulta
    const float escala = 1.0f / std::sqrt(static_cast<float>(D));

    for (const Capa& capa : capas_) {
        // 2. Atención: x += proy(atención(ln_1(x))).
        for (size_t t = 0; t < T; ++t) layernorm(&x[t * C], capa.ln1_g, capa.ln1_b, C, &norm[t * C]);
        lineal_ingenua(norm.data(), T, C, capa.attn_w, capa.attn_b, 3 * C, qkv.data());

        for (size_t h = 0; h < H; ++h) {
            for (size_t t = 0; t < T; ++t) {
                // La cabeza h usa las componentes [h·D, (h+1)·D) de consulta, clave y valor.
                const float* q = &qkv[t * 3 * C + h * D];

                // Afinidad de la posición t con cada posición s <= t. La máscara causal es
                // simplemente no mirar s > t: softmax sobre los t + 1 primeros.
                for (size_t s = 0; s <= t; ++s) {
                    const float* k = &qkv[s * 3 * C + C + h * D];
                    float punto = 0.0f;
                    for (size_t d = 0; d < D; ++d) punto += q[d] * k[d];
                    pesos_at[s] = punto * escala;
                }
                softmax(pesos_at.data(), t + 1);

                // Salida: media de los valores ponderada por los pesos de atención.
                float* salida = &mezcla[t * C + h * D];
                for (size_t d = 0; d < D; ++d) salida[d] = 0.0f;
                for (size_t s = 0; s <= t; ++s) {
                    const float* v = &qkv[s * 3 * C + 2 * C + h * D];
                    for (size_t d = 0; d < D; ++d) salida[d] += pesos_at[s] * v[d];
                }
            }
        }
        lineal_ingenua(mezcla.data(), T, C, capa.attn_proy_w, capa.attn_proy_b, C, rama.data());
        for (size_t i = 0; i < T * C; ++i) x[i] += rama[i];

        // 3. MLP: x += proy(gelu(fc(ln_2(x)))).
        for (size_t t = 0; t < T; ++t) layernorm(&x[t * C], capa.ln2_g, capa.ln2_b, C, &norm[t * C]);
        lineal_ingenua(norm.data(), T, C, capa.fc_w, capa.fc_b, 4 * C, oculta.data());
        gelu(oculta.data(), T * 4 * C);
        lineal_ingenua(oculta.data(), T, 4 * C, capa.mlp_proy_w, capa.mlp_proy_b, C, rama.data());
        for (size_t i = 0; i < T * C; ++i) x[i] += rama[i];
    }

    // 4. LayerNorm final y cabeza: logits[t][v] = wte[v] · ln_f(x[t]). La cabeza reutiliza wte.
    for (size_t t = 0; t < T; ++t) layernorm(&x[t * C], lnf_g_, lnf_b_, C, &norm[t * C]);
    std::vector<float> logits(T * cfg.n_vocab);
    lineal_ingenua(norm.data(), T, C, wte_, nullptr, cfg.n_vocab, logits.data());
    return logits;
}
