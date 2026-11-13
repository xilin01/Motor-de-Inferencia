// Generación autorregresiva: elegir el siguiente token a partir de los logits y repetir.
#pragma once
#include "modelo.hpp"

#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <vector>

struct Muestreo {
    float temperatura = 0.0f;  // 0: greedy (siempre el token más probable)
    int top_k = 0;             // 0: sin recorte; k > 0: solo se sortea entre los k tokens más probables
    uint64_t semilla = 0;
};

// Elige un token a partir de los logits de la última posición.
int elegir_token(std::span<const float> logits, const Muestreo& muestreo, std::mt19937_64& azar);

// Añade hasta n_nuevos tokens a ids y devuelve la secuencia completa. Para antes si se llena el
// contexto o si sale id_parada (el token de fin de texto; -1 para no parar nunca).
// Llama a al_generar con cada token nuevo, para poder mostrarlo según sale.
// Con kv_cache cada paso procesa solo el último token; sin él se recalcula el contexto entero.
std::vector<int> generar(const Modelo& modelo, std::vector<int> ids, int n_nuevos, const Muestreo& muestreo,
                         int id_parada = -1, const std::function<void(int)>& al_generar = {}, bool kv_cache = true);
