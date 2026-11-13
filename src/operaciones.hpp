// Operaciones numéricas del transformer, en su versión ingenua: bucles directos, un float cada vez.
#pragma once
#include <cstddef>

// Capa lineal sobre n_filas vectores: y[t][o] = b[o] + Σ_k w[o][k] · x[t][k].
//   x: [n_filas, n_in]   w: [n_out, n_in]   b: [n_out] o nullptr   y: [n_filas, n_out]
// Solo se calculan las neuronas de salida o en [o_ini, o_fin): así varios hilos pueden repartirse
// la misma capa sin pisarse. Para la capa entera, o_ini = 0 y o_fin = n_out.
// Todas las variantes de la multiplicación de matrices comparten esta firma.
using Lineal = void (*)(const float* x, size_t n_filas, size_t n_in, const float* w, const float* b, size_t n_out, float* y,
                        size_t o_ini, size_t o_fin);

// Versión ingenua: tres bucles, un float cada vez.
void lineal_ingenua(const float* x, size_t n_filas, size_t n_in, const float* w, const float* b, size_t n_out, float* y,
                    size_t o_ini, size_t o_fin);

// Versión AVX2 + FMA: 8 floats por instrucción (src/lineal_avx2.cpp). Solo se puede llamar si
// cpu_tiene_avx2_fma() devuelve true.
void lineal_avx2(const float* x, size_t n_filas, size_t n_in, const float* w, const float* b, size_t n_out, float* y,
                 size_t o_ini, size_t o_fin);

// LayerNorm de un vector: y = g · (x - media) / sqrt(varianza + 1e-5) + b.
void layernorm(const float* x, const float* g, const float* b, size_t n, float* y);

// GELU con la aproximación de tanh que usa GPT-2 ("gelu_new"), en el sitio.
void gelu(float* x, size_t n);

// Softmax en el sitio: x[i] = exp(x[i]) / Σ_j exp(x[j]).
void softmax(float* x, size_t n);
