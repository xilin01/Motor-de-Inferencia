#include "pesos.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>

static_assert(std::endian::native == std::endian::little, "el formato de pesos es little-endian");

namespace {

template <typename T>
T leer(std::ifstream& f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof v);
    if (!f) throw std::runtime_error("archivo de pesos truncado");
    return v;
}

}  // namespace

const float* Pesos::tensor(const std::string& nombre, std::initializer_list<uint32_t> forma) const {
    auto it = tensores.find(nombre);
    if (it == tensores.end()) throw std::runtime_error("falta el tensor " + nombre);
    if (!std::ranges::equal(it->second.forma, forma)) throw std::runtime_error("forma inesperada en " + nombre);
    return it->second.datos.data();
}

Pesos cargar_pesos(const std::filesystem::path& ruta) {
    std::ifstream f(ruta, std::ios::binary);
    if (!f) throw std::runtime_error("no se puede abrir el archivo de pesos (genera modelos/ con tools/exportar.py)");
    const uint64_t tam = std::filesystem::file_size(ruta);

    char magia[8];
    f.read(magia, sizeof magia);
    if (!f || std::memcmp(magia, "GPT2PESO", 8) != 0) throw std::runtime_error("no es un archivo de pesos GPT2PESO");
    if (leer<uint32_t>(f) != 1) throw std::runtime_error("versión de formato no soportada");

    Pesos p;
    p.config = leer<Config>(f);  // cinco uint32 seguidos, sin relleno
    const uint32_t n_tensores = leer<uint32_t>(f);

    for (uint32_t i = 0; i < n_tensores; ++i) {
        const uint32_t len = leer<uint32_t>(f);
        if (len == 0 || len > 256) throw std::runtime_error("nombre de tensor inválido");
        std::string nombre(len, '\0');
        f.read(nombre.data(), len);

        const uint32_t n_dims = leer<uint32_t>(f);
        if (n_dims == 0 || n_dims > 4) throw std::runtime_error("número de dimensiones inválido en " + nombre);
        Tensor t;
        uint64_t elementos = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            t.forma.push_back(leer<uint32_t>(f));
            elementos *= t.forma.back();
            // Un archivo corrupto no debe provocar una reserva gigante ni un desbordamiento del producto.
            if (elementos > tam / sizeof(float)) throw std::runtime_error("forma imposible en " + nombre);
        }
        const uint64_t suma_esperada = leer<uint64_t>(f);

        t.datos.resize(elementos);
        f.read(reinterpret_cast<char*>(t.datos.data()), static_cast<std::streamsize>(elementos * sizeof(float)));
        if (!f) throw std::runtime_error("archivo de pesos truncado en " + nombre);

        // Suma de control: suma de los patrones de 32 bits de cada float, módulo 2^64.
        uint64_t suma = 0;
        for (float v : t.datos) suma += std::bit_cast<uint32_t>(v);
        if (suma != suma_esperada) throw std::runtime_error("suma de control incorrecta en " + nombre);

        p.tensores.emplace(std::move(nombre), std::move(t));
    }
    return p;
}
