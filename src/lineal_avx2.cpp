// Capa lineal con AVX2 + FMA. Es el único archivo que se compila con /arch:AVX2; por eso no
// incluye cabeceras de la biblioteca estándar con código (ver docs/NOTAS.md, hito 6).
//
// Intrínsecos usados, comprobados en la Intel Intrinsics Guide (datos 3.6.9), cabecera immintrin.h:
//   _mm256_setzero_ps  (AVX)  vector de 8 floats a cero
//   _mm256_loadu_ps    (AVX)  carga 8 floats; la dirección no necesita alineación
//   _mm256_storeu_ps   (AVX)  guarda 8 floats; la dirección no necesita alineación
//   _mm256_fmadd_ps    (FMA)  a*b + c en los 8 floats
// La guía no habla del redondeo de fmadd. Que el producto no se redondea antes de sumar es la
// definición de "fused multiply-add" de IEEE 754-2008, no un dato sacado de la guía.
#include "operaciones.hpp"

#include <immintrin.h>

namespace {

const size_t ANCHO = 8;  // floats por registro de 256 bits

// Suma los 8 carriles de un acumulador, siempre en el mismo orden.
inline float sumar_carriles(__m256 v) {
    float c[ANCHO];
    _mm256_storeu_ps(c, v);
    return ((c[0] + c[4]) + (c[1] + c[5])) + ((c[2] + c[6]) + (c[3] + c[7]));
}

// Productos de los elementos que no llenan un registro (cuando n no es múltiplo de 8).
inline float cola(const float* w, const float* x, size_t k, size_t n) {
    float suma = 0.0f;
    for (; k < n; ++k) suma += w[k] * x[k];
    return suma;
}

}  // namespace

void lineal_avx2(const float* x, size_t n_filas, size_t n_in, const float* w, const float* b, size_t n_out, float* y,
                 size_t o_ini, size_t o_fin) {
    const size_t n_vec = n_in - n_in % ANCHO;  // parte de cada fila que se recorre de 8 en 8
    size_t o = o_ini;

    // Cuatro neuronas de salida a la vez. Cada una tiene su acumulador, así que las cuatro cadenas
    // de FMA son independientes y el procesador las solapa; además x[k..k+8) se carga una sola vez.
    for (; o + 4 <= o_fin; o += 4) {
        const float* w0 = w + o * n_in;
        const float* w1 = w0 + n_in;
        const float* w2 = w1 + n_in;
        const float* w3 = w2 + n_in;
        for (size_t t = 0; t < n_filas; ++t) {
            const float* xt = x + t * n_in;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (size_t k = 0; k < n_vec; k += ANCHO) {
                const __m256 xv = _mm256_loadu_ps(xt + k);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + k), xv, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + k), xv, a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + k), xv, a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + k), xv, a3);
            }
            float* yt = y + t * n_out + o;
            yt[0] = (b ? b[o] : 0.0f) + (sumar_carriles(a0) + cola(w0, xt, n_vec, n_in));
            yt[1] = (b ? b[o + 1] : 0.0f) + (sumar_carriles(a1) + cola(w1, xt, n_vec, n_in));
            yt[2] = (b ? b[o + 2] : 0.0f) + (sumar_carriles(a2) + cola(w2, xt, n_vec, n_in));
            yt[3] = (b ? b[o + 3] : 0.0f) + (sumar_carriles(a3) + cola(w3, xt, n_vec, n_in));
        }
    }

    // Neuronas sobrantes, de una en una. Mismo orden de operaciones que arriba: el resultado de
    // cada neurona no depende de en qué grupo caiga, y por tanto tampoco del reparto entre hilos.
    for (; o < o_fin; ++o) {
        const float* w0 = w + o * n_in;
        for (size_t t = 0; t < n_filas; ++t) {
            const float* xt = x + t * n_in;
            __m256 a0 = _mm256_setzero_ps();
            for (size_t k = 0; k < n_vec; k += ANCHO) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + k), _mm256_loadu_ps(xt + k), a0);
            y[t * n_out + o] = (b ? b[o] : 0.0f) + (sumar_carriles(a0) + cola(w0, xt, n_vec, n_in));
        }
    }
}
