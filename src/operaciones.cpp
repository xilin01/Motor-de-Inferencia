#include "operaciones.hpp"

#include <algorithm>
#include <cmath>

void lineal_ingenua(const float* x, size_t n_filas, size_t n_in, const float* w, const float* b, size_t n_out, float* y,
                    size_t o_ini, size_t o_fin) {
    for (size_t t = 0; t < n_filas; ++t) {
        const float* xt = x + t * n_in;
        for (size_t o = o_ini; o < o_fin; ++o) {
            const float* wo = w + o * n_in;  // fila o: los pesos de la neurona de salida o
            float suma = b ? b[o] : 0.0f;
            for (size_t k = 0; k < n_in; ++k) suma += wo[k] * xt[k];
            y[t * n_out + o] = suma;
        }
    }
}

void layernorm(const float* x, const float* g, const float* b, size_t n, float* y) {
    float media = 0.0f;
    for (size_t i = 0; i < n; ++i) media += x[i];
    media /= static_cast<float>(n);

    float varianza = 0.0f;  // varianza poblacional: se divide entre n, no entre n - 1
    for (size_t i = 0; i < n; ++i) varianza += (x[i] - media) * (x[i] - media);
    varianza /= static_cast<float>(n);

    const float inv = 1.0f / std::sqrt(varianza + 1e-5f);
    for (size_t i = 0; i < n; ++i) y[i] = g[i] * (x[i] - media) * inv + b[i];
}

void gelu(float* x, size_t n) {
    const float c = 0.7978845608028654f;  // sqrt(2 / pi)
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}

void softmax(float* x, size_t n) {
    // Restar el máximo no cambia el resultado y evita que exp() desborde.
    const float maximo = *std::max_element(x, x + n);
    float suma = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - maximo);
        suma += x[i];
    }
    for (size_t i = 0; i < n; ++i) x[i] /= suma;
}
