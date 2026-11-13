// Cargador del archivo de pesos (formato en docs/FORMATO.md).
#pragma once
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

struct Tensor {
    std::vector<uint32_t> forma;
    std::vector<float> datos;  // orden fila-mayor: el último índice es el que varía más rápido
};

struct Config {
    uint32_t n_vocab;    // tamaño del vocabulario (50257)
    uint32_t n_ctx;      // longitud máxima de contexto (1024)
    uint32_t n_embd;     // dimensión del modelo (768)
    uint32_t n_capas;    // bloques transformer (12)
    uint32_t n_cabezas;  // cabezas de atención por bloque (12)
};

struct Pesos {
    Config config{};
    std::map<std::string, Tensor> tensores;

    // Puntero a los datos del tensor; lanza si falta o si su forma no es la esperada.
    const float* tensor(const std::string& nombre, std::initializer_list<uint32_t> forma) const;
};

// Lee y valida el archivo completo; lanza std::runtime_error si está truncado o corrupto.
Pesos cargar_pesos(const std::filesystem::path& ruta);
