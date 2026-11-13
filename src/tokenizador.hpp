// Tokenizador BPE a nivel de bytes de GPT-2: texto UTF-8 <-> ids de token.
#pragma once
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Paso 1 de la codificación: corta el texto en trozos (palabra con su espacio delante, número,
// puntuación, espacios) igual que el patrón de GPT-2. Cada trozo se tokeniza por separado.
std::vector<std::string_view> trocear(std::string_view texto);

class Tokenizador {
public:
    // Lee modelos/gpt2_vocab.bin (formato en docs/FORMATO.md); lanza std::runtime_error si es inválido.
    explicit Tokenizador(const std::filesystem::path& ruta_vocab);

    std::vector<int> codificar(std::string_view texto) const;

    // Concatena los bytes de cada token. El resultado puede no ser UTF-8 válido si los ids cortan
    // un carácter por la mitad. Lanza std::out_of_range con un id fuera del vocabulario.
    std::string decodificar(std::span<const int> ids) const;

    int fin_de_texto() const { return fin_de_texto_; }
    int n_tokens() const { return static_cast<int>(tokens_.size()); }

private:
    struct Fusion {
        uint32_t prioridad;  // posición en la lista de fusiones: menor se aplica antes
        int resultado;       // id del token fusionado
    };

    void bpe(std::string_view trozo, std::vector<int>& ids) const;

    std::vector<std::string> tokens_;                 // id -> bytes del token
    std::unordered_map<uint64_t, Fusion> fusiones_;   // (id izquierdo << 32 | id derecho) -> fusión
    int byte_a_id_[256];                              // token de un solo byte para cada valor de byte
    int fin_de_texto_ = 0;
};
