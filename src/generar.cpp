#include "generar.hpp"

#include "operaciones.hpp"

#include <algorithm>
#include <numeric>

int elegir_token(std::span<const float> logits, const Muestreo& muestreo, std::mt19937_64& azar) {
    const int n = static_cast<int>(logits.size());
    if (muestreo.temperatura <= 0.0f) {
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    // Top-k: quedarse con los k candidatos de mayor logit. nth_element los deja delante sin
    // ordenar el vector entero: coste lineal en vez de n·log(n).
    std::vector<int> candidatos(n);
    std::iota(candidatos.begin(), candidatos.end(), 0);
    if (muestreo.top_k > 0 && muestreo.top_k < n) {
        std::nth_element(candidatos.begin(), candidatos.begin() + muestreo.top_k, candidatos.end(),
                         [&](int a, int b) { return logits[a] > logits[b]; });
        candidatos.resize(muestreo.top_k);
    }

    // Temperatura: p_i = softmax(logit_i / T). T < 1 afila la distribución; T > 1 la aplana.
    std::vector<float> p(candidatos.size());
    for (size_t i = 0; i < p.size(); ++i) p[i] = logits[candidatos[i]] / muestreo.temperatura;
    softmax(p.data(), p.size());

    // Sorteo: un número uniforme en [0, 1) cae en el tramo del candidato i con probabilidad p_i.
    const float u = std::uniform_real_distribution<float>(0.0f, 1.0f)(azar);
    float acumulada = 0.0f;
    for (size_t i = 0; i < p.size(); ++i) {
        acumulada += p[i];
        if (u < acumulada) return candidatos[i];
    }
    return candidatos.back();  // el redondeo puede dejar la suma un pelo por debajo de 1
}

std::vector<int> generar(const Modelo& modelo, std::vector<int> ids, int n_nuevos, const Muestreo& muestreo,
                         int id_parada, const std::function<void(int)>& al_generar, bool kv_cache) {
    std::mt19937_64 azar(muestreo.semilla);
    Cache cache;
    for (int i = 0; i < n_nuevos && ids.size() < modelo.config().n_ctx; ++i) {
        // Con caché solo se procesan los tokens que aún no están en ella: el prompt entero en el
        // primer paso y después únicamente el último token generado. Sin caché, todo cada vez.
        const std::vector<float> logits = kv_cache ? modelo.avanzar(cache, std::span(ids).subspan(cache.n), true)
                                                   : modelo.forward(ids, true);
        const int id = elegir_token(logits, muestreo, azar);
        ids.push_back(id);
        if (al_generar) al_generar(id);
        if (id == id_parada) break;
    }
    return ids;
}
