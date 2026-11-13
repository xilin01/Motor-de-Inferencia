#include "modelo.hpp"

#include "cpu.hpp"

#include <algorithm>
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

void Modelo::configurar(const Opciones& opciones) {
    if (opciones.matmul == Matmul::Avx2 && !cpu_tiene_avx2_fma()) throw std::runtime_error("esta CPU no soporta AVX2 y FMA");
    if (opciones.hilos < 1 || opciones.hilos > 256) throw std::invalid_argument("el número de hilos debe estar entre 1 y 256");
    nucleo_ = opciones.matmul == Matmul::Avx2 ? lineal_avx2 : lineal_ingenua;
    if (opciones.hilos == 1) hilos_.reset();
    else if (!hilos_ || hilos_->n() != opciones.hilos) hilos_ = std::make_unique<Hilos>(opciones.hilos);
}

void Modelo::lineal(const float* x, size_t n_filas, size_t n_in, const float* w, const float* b, size_t n_out, float* y) const {
    // Se reparten todas las capas, también las pequeñas: medido, dejar alguna en un solo hilo
    // empeora el total (110 tokens/s repartiendo todas frente a 104 sin la de 768x768, 6 hilos).
    if (!hilos_) {
        nucleo_(x, n_filas, n_in, w, b, n_out, y, 0, n_out);
        return;
    }
    // Se reparten las neuronas de salida: cada hilo escribe posiciones distintas de y y solo lee
    // x y w, así que no hace falta ningún cerrojo. Trozos pequeños (8 por hilo) y reparto sobre
    // la marcha, porque los núcleos P y E no van a la misma velocidad. Múltiplos de 4 para no
    // romper los grupos de 4 neuronas del núcleo AVX2.
    const size_t trozo = (n_out / (static_cast<size_t>(hilos_->n()) * 8) / 4 + 1) * 4;
    hilos_->paralelo(n_out, trozo, [&](size_t ini, size_t fin) { nucleo_(x, n_filas, n_in, w, b, n_out, y, ini, fin); });
}

void Modelo::comprobar(size_t n_previos, std::span<const int> ids) const {
    const Config& cfg = pesos_.config;
    if (ids.empty() || n_previos + ids.size() > cfg.n_ctx) throw std::invalid_argument("el contexto debe tener entre 1 y n_ctx tokens");
    for (int id : ids) {
        if (id < 0 || static_cast<uint32_t>(id) >= cfg.n_vocab) throw std::invalid_argument("id de token fuera del vocabulario");
    }
}

std::vector<float> Modelo::forward(std::span<const int> ids, bool solo_ultima) const {
    const Config& cfg = pesos_.config;
    const size_t T = ids.size();         // tokens del contexto
    const size_t C = cfg.n_embd;         // dimensión del modelo
    const size_t H = cfg.n_cabezas;      // cabezas de atención
    const size_t D = C / H;              // dimensión de cada cabeza
    comprobar(0, ids);

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
        lineal(norm.data(), T, C, capa.attn_w, capa.attn_b, 3 * C, qkv.data());

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
        lineal(mezcla.data(), T, C, capa.attn_proy_w, capa.attn_proy_b, C, rama.data());
        for (size_t i = 0; i < T * C; ++i) x[i] += rama[i];

        // 3. MLP: x += proy(gelu(fc(ln_2(x)))).
        for (size_t t = 0; t < T; ++t) layernorm(&x[t * C], capa.ln2_g, capa.ln2_b, C, &norm[t * C]);
        lineal(norm.data(), T, C, capa.fc_w, capa.fc_b, 4 * C, oculta.data());
        gelu(oculta.data(), T * 4 * C);
        lineal(oculta.data(), T, 4 * C, capa.mlp_proy_w, capa.mlp_proy_b, C, rama.data());
        for (size_t i = 0; i < T * C; ++i) x[i] += rama[i];
    }

    // 4. LayerNorm final y cabeza: logits[t][v] = wte[v] · ln_f(x[t]). La cabeza reutiliza wte.
    for (size_t t = 0; t < T; ++t) layernorm(&x[t * C], lnf_g_, lnf_b_, C, &norm[t * C]);
    const size_t primera = solo_ultima ? T - 1 : 0;
    std::vector<float> logits((T - primera) * cfg.n_vocab);
    lineal(&norm[primera * C], T - primera, C, wte_, nullptr, cfg.n_vocab, logits.data());
    return logits;
}

// Mismos pasos que forward, pero solo para las M posiciones nuevas. Lo único que una posición
// necesita de las anteriores son sus claves y valores de atención, y esos no cambian al añadir
// tokens (la máscara causal impide que una posición dependa de las siguientes): se guardan.
std::vector<float> Modelo::avanzar(Cache& cache, std::span<const int> nuevos, bool solo_ultima) const {
    const Config& cfg = pesos_.config;
    const size_t N = cache.n;        // posiciones ya en la caché
    const size_t M = nuevos.size();  // posiciones nuevas: N .. N + M - 1
    const size_t C = cfg.n_embd;
    const size_t H = cfg.n_cabezas;
    const size_t D = C / H;
    comprobar(N, nuevos);
    cache.claves.resize(capas_.size());
    cache.valores.resize(capas_.size());

    // 1. Embeddings de los tokens nuevos, cada uno con su posición absoluta.
    std::vector<float> x(M * C);
    for (size_t i = 0; i < M; ++i) {
        for (size_t c = 0; c < C; ++c) x[i * C + c] = wte_[nuevos[i] * C + c] + wpe_[(N + i) * C + c];
    }

    std::vector<float> norm(M * C), qkv(M * 3 * C), mezcla(M * C), oculta(M * 4 * C), rama(M * C);
    std::vector<float> pesos_at(N + M);
    const float escala = 1.0f / std::sqrt(static_cast<float>(D));

    for (size_t l = 0; l < capas_.size(); ++l) {
        const Capa& capa = capas_[l];

        // 2. Atención. Consulta, clave y valor solo de las posiciones nuevas...
        for (size_t i = 0; i < M; ++i) layernorm(&x[i * C], capa.ln1_g, capa.ln1_b, C, &norm[i * C]);
        lineal(norm.data(), M, C, capa.attn_w, capa.attn_b, 3 * C, qkv.data());

        // ...y sus claves y valores pasan a la caché, a continuación de los que ya había.
        std::vector<float>& claves = cache.claves[l];
        std::vector<float>& valores = cache.valores[l];
        claves.resize((N + M) * C);
        valores.resize((N + M) * C);
        for (size_t i = 0; i < M; ++i) {
            std::copy_n(&qkv[i * 3 * C + C], C, &claves[(N + i) * C]);
            std::copy_n(&qkv[i * 3 * C + 2 * C], C, &valores[(N + i) * C]);
        }

        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < M; ++i) {
                const size_t t = N + i;  // posición absoluta de la consulta
                const float* q = &qkv[i * 3 * C + h * D];

                // Cada consulta nueva atiende a todas las posiciones s <= t: las de la caché y las nuevas.
                for (size_t s = 0; s <= t; ++s) {
                    const float* k = &claves[s * C + h * D];
                    float punto = 0.0f;
                    for (size_t d = 0; d < D; ++d) punto += q[d] * k[d];
                    pesos_at[s] = punto * escala;
                }
                softmax(pesos_at.data(), t + 1);

                float* salida = &mezcla[i * C + h * D];
                for (size_t d = 0; d < D; ++d) salida[d] = 0.0f;
                for (size_t s = 0; s <= t; ++s) {
                    const float* v = &valores[s * C + h * D];
                    for (size_t d = 0; d < D; ++d) salida[d] += pesos_at[s] * v[d];
                }
            }
        }
        lineal(mezcla.data(), M, C, capa.attn_proy_w, capa.attn_proy_b, C, rama.data());
        for (size_t i = 0; i < M * C; ++i) x[i] += rama[i];

        // 3. MLP: actúa posición a posición, no necesita nada de las anteriores.
        for (size_t i = 0; i < M; ++i) layernorm(&x[i * C], capa.ln2_g, capa.ln2_b, C, &norm[i * C]);
        lineal(norm.data(), M, C, capa.fc_w, capa.fc_b, 4 * C, oculta.data());
        gelu(oculta.data(), M * 4 * C);
        lineal(oculta.data(), M, 4 * C, capa.mlp_proy_w, capa.mlp_proy_b, C, rama.data());
        for (size_t i = 0; i < M * C; ++i) x[i] += rama[i];
    }
    cache.n = N + M;

    // 4. LayerNorm final y cabeza.
    for (size_t i = 0; i < M; ++i) layernorm(&x[i * C], lnf_g_, lnf_b_, C, &norm[i * C]);
    const size_t primera = solo_ultima ? M - 1 : 0;
    std::vector<float> logits((M - primera) * cfg.n_vocab);
    lineal(&norm[primera * C], M - primera, C, wte_, nullptr, cfg.n_vocab, logits.data());
    return logits;
}
