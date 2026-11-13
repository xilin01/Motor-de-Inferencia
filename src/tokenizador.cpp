#include "tokenizador.hpp"

#include "unicode_tablas.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace {

constexpr std::string_view FIN_DE_TEXTO = "<|endoftext|>";

// ---------------------------------------------------------------------------------------------
// Troceado previo. Equivale al patrón de GPT-2:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// std::regex no entiende \p{L} ni \p{N}, así que se recorre el texto a mano con tablas Unicode.
// ---------------------------------------------------------------------------------------------

enum class Clase { Letra, Numero, Espacio, Otro, Fin };

template <size_t N>
bool en_tabla(const Rango (&tabla)[N], char32_t c) {
    // Primer rango cuyo final no queda por debajo de c; c pertenece si además empieza antes o en c.
    auto it = std::ranges::lower_bound(tabla, c, {}, &Rango::fin);
    return it != std::end(tabla) && it->ini <= c;
}

Clase clasificar(char32_t c) {
    if (en_tabla(LETRAS, c)) return Clase::Letra;
    if (en_tabla(NUMEROS, c)) return Clase::Numero;
    if (en_tabla(ESPACIOS, c)) return Clase::Espacio;
    return Clase::Otro;
}

struct Caracter {
    char32_t codigo;
    size_t ini, fin;  // bytes [ini, fin) que ocupa en el texto
    Clase clase;
};

// Decodifica el carácter UTF-8 que empieza en texto[i].
// shortcut: un byte inválido cuenta como un carácter suelto de clase Otro y no se rechazan las
// codificaciones sobrelargas; basta para trocear, validar UTF-8 de verdad si el texto no es de fiar.
Caracter decodificar_utf8(std::string_view texto, size_t i) {
    const auto b = [&](size_t k) { return static_cast<unsigned char>(texto[k]); };
    const unsigned char b0 = b(i);
    size_t len = b0 < 0x80 ? 1 : (b0 >> 5) == 0b110 ? 2 : (b0 >> 4) == 0b1110 ? 3 : (b0 >> 3) == 0b11110 ? 4 : 0;
    if (len == 0 || i + len > texto.size()) return {0xFFFD, i, i + 1, Clase::Otro};

    // El primer byte aporta los bits que quedan tras su prefijo; cada byte de continuación (10xxxxxx), 6 bits.
    char32_t c = len == 1 ? b0 : b0 & (0xFF >> (len + 1));
    for (size_t k = 1; k < len; ++k) {
        if ((b(i + k) >> 6) != 0b10) return {0xFFFD, i, i + 1, Clase::Otro};
        c = (c << 6) | (b(i + k) & 0x3F);
    }
    return {c, i, i + len, clasificar(c)};
}

template <typename T>
T leer(std::ifstream& f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof v);
    if (!f) throw std::runtime_error("archivo de vocabulario truncado");
    return v;
}

uint64_t clave(int izq, int der) {
    return (static_cast<uint64_t>(izq) << 32) | static_cast<uint32_t>(der);
}

}  // namespace

std::vector<std::string_view> trocear(std::string_view texto) {
    std::vector<Caracter> c;
    for (size_t i = 0; i < texto.size(); i = c.back().fin) c.push_back(decodificar_utf8(texto, i));
    const size_t n = c.size();
    const auto clase = [&](size_t k) { return k < n ? c[k].clase : Clase::Fin; };

    std::vector<std::string_view> trozos;
    size_t i = 0;
    while (i < n) {
        size_t j = i;  // el trozo será [i, j)

        // 1. Contracciones inglesas: apóstrofo seguido de s, t, re, ve, m, ll o d.
        if (c[i].codigo == U'\'') {
            for (std::u32string_view sufijo : {U"s", U"t", U"re", U"ve", U"m", U"ll", U"d"}) {
                bool coincide = i + sufijo.size() < n;
                for (size_t k = 0; coincide && k < sufijo.size(); ++k) coincide = c[i + 1 + k].codigo == sufijo[k];
                if (coincide) {
                    j = i + 1 + sufijo.size();
                    break;
                }
            }
        }

        if (j == i) {
            // 2. Un espacio opcional y una racha de letras, de números o de "otros" (puntuación, símbolos).
            const size_t k = i + (c[i].codigo == U' ' ? 1 : 0);
            const Clase cl = clase(k);
            if (cl == Clase::Letra || cl == Clase::Numero || cl == Clase::Otro) {
                j = k;
                while (clase(j) == cl) ++j;
            } else {
                // 3. Racha de espacios. Si después viene texto, se deja fuera el último espacio
                //    para que se una a la palabra siguiente en la regla 2: es lo que hace \s+(?!\S).
                while (clase(j) == Clase::Espacio) ++j;
                if (j < n && j - i > 1) --j;
            }
        }

        trozos.push_back(texto.substr(c[i].ini, c[j - 1].fin - c[i].ini));
        i = j;
    }
    return trozos;
}

Tokenizador::Tokenizador(const std::filesystem::path& ruta_vocab) {
    std::ifstream f(ruta_vocab, std::ios::binary);
    if (!f) throw std::runtime_error("no se puede abrir el vocabulario (genera modelos/ con tools/exportar.py)");

    char magia[8];
    f.read(magia, sizeof magia);
    if (!f || std::string_view(magia, 8) != "GPT2VOCA") throw std::runtime_error("no es un archivo de vocabulario GPT2VOCA");
    if (leer<uint32_t>(f) != 1) throw std::runtime_error("versión de vocabulario no soportada");

    const uint32_t n_tokens = leer<uint32_t>(f);
    if (n_tokens < 256 || n_tokens > 1'000'000) throw std::runtime_error("número de tokens inválido");
    tokens_.resize(n_tokens);
    std::ranges::fill(byte_a_id_, -1);
    for (uint32_t id = 0; id < n_tokens; ++id) {
        std::string& t = tokens_[id];
        t.resize(leer<uint16_t>(f));
        f.read(t.data(), static_cast<std::streamsize>(t.size()));
        if (t.size() == 1) byte_a_id_[static_cast<unsigned char>(t[0])] = static_cast<int>(id);
    }
    if (!f) throw std::runtime_error("archivo de vocabulario truncado");
    // BPE a nivel de bytes exige un token para cada uno de los 256 bytes: así cualquier texto es codificable.
    if (std::ranges::count(byte_a_id_, -1) != 0) throw std::runtime_error("el vocabulario no cubre los 256 bytes");

    const uint32_t n_fusiones = leer<uint32_t>(f);
    for (uint32_t prioridad = 0; prioridad < n_fusiones; ++prioridad) {
        const uint32_t izq = leer<uint32_t>(f), der = leer<uint32_t>(f), res = leer<uint32_t>(f);
        if (izq >= n_tokens || der >= n_tokens || res >= n_tokens) throw std::runtime_error("fusión con id inválido");
        fusiones_.emplace(clave(static_cast<int>(izq), static_cast<int>(der)), Fusion{prioridad, static_cast<int>(res)});
    }

    const uint32_t fin = leer<uint32_t>(f);
    if (fin >= n_tokens) throw std::runtime_error("id de fin de texto inválido");
    fin_de_texto_ = static_cast<int>(fin);
}

// BPE sobre un trozo: empieza con un token por byte y aplica, mientras exista alguna, la fusión de
// mayor prioridad entre todos los pares adyacentes. Es el orden en que se aprendieron las fusiones.
void Tokenizador::bpe(std::string_view trozo, std::vector<int>& ids) const {
    std::vector<int> partes;
    for (unsigned char b : trozo) partes.push_back(byte_a_id_[b]);

    while (partes.size() > 1) {
        const Fusion* mejor = nullptr;
        int izq = 0, der = 0;
        for (size_t i = 0; i + 1 < partes.size(); ++i) {
            auto it = fusiones_.find(clave(partes[i], partes[i + 1]));
            if (it != fusiones_.end() && (!mejor || it->second.prioridad < mejor->prioridad)) {
                mejor = &it->second;
                izq = partes[i];
                der = partes[i + 1];
            }
        }
        if (!mejor) break;

        // Sustituye todas las apariciones del par, de izquierda a derecha, compactando en el sitio.
        size_t w = 0;
        for (size_t r = 0; r < partes.size();) {
            if (r + 1 < partes.size() && partes[r] == izq && partes[r + 1] == der) {
                partes[w++] = mejor->resultado;
                r += 2;
            } else {
                partes[w++] = partes[r++];
            }
        }
        partes.resize(w);
    }
    ids.insert(ids.end(), partes.begin(), partes.end());
}

std::vector<int> Tokenizador::codificar(std::string_view texto) const {
    std::vector<int> ids;
    while (true) {
        // "<|endoftext|>" escrito tal cual es el token especial, no texto que trocear.
        const size_t p = texto.find(FIN_DE_TEXTO);
        for (std::string_view trozo : trocear(texto.substr(0, p))) bpe(trozo, ids);
        if (p == std::string_view::npos) break;
        ids.push_back(fin_de_texto_);
        texto.remove_prefix(p + FIN_DE_TEXTO.size());
    }
    return ids;
}

std::string Tokenizador::decodificar(std::span<const int> ids) const {
    std::string texto;
    for (int id : ids) {
        if (id < 0 || id >= n_tokens()) throw std::out_of_range("id de token fuera del vocabulario");
        texto += tokens_[id];
    }
    return texto;
}
