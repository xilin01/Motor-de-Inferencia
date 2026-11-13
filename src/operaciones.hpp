// Operaciones numéricas del transformer, en su versión ingenua: bucles directos, un float cada vez.
#pragma once
#include <cstddef>

// Capa lineal sobre n_filas vectores: y[t][o] = b[o] + Σ_k w[o][k] · x[t][k].
//   x: [n_filas, n_in]   w: [n_out, n_in]   b: [n_out] o nullptr   y: [n_filas, n_out]
// Todas las variantes de la multiplicación de matrices comparten esta firma.
using Lineal = void (*)(const float* x, size_t n_filas, size_t n_in, const float* w, const float* b, size_t n_out, float* y);

void lineal_ingenua(const float* x, size_t n_filas, size_t n_in, const float* w, const float* b, size_t n_out, float* y);

// LayerNorm de un vector: y = g · (x - media) / sqrt(varianza + 1e-5) + b.
void layernorm(const float* x, const float* g, const float* b, size_t n, float* y);

// GELU con la aproximación de tanh que usa GPT-2 ("gelu_new"), en el sitio.
void gelu(float* x, size_t n);

// Softmax en el sitio: x[i] = exp(x[i]) / Σ_j exp(x[j]).
void softmax(float* x, size_t n);
