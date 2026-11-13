// El archivo exportado se carga, tiene la configuración y las formas de GPT-2 small, y un archivo corrupto se rechaza.
#include "pesos.hpp"
#include "prueba.hpp"

#include <fstream>
#include <stdexcept>

template <typename F>
bool lanza(F&& f) {
    try { f(); } catch (const std::runtime_error&) { return true; }
    return false;
}

int main() {
    const Pesos p = cargar_pesos("modelos/gpt2_124M.bin");  // verifica la suma de control de cada tensor

    COMPROBAR(p.config.n_vocab == 50257);
    COMPROBAR(p.config.n_ctx == 1024);
    COMPROBAR(p.config.n_embd == 768);
    COMPROBAR(p.config.n_capas == 12);
    COMPROBAR(p.config.n_cabezas == 12);
    COMPROBAR(p.tensores.size() == 148);  // 2 embeddings + 12 capas x 12 tensores + 2 de ln_f

    size_t parametros = 0;
    for (const auto& [nombre, t] : p.tensores) parametros += t.datos.size();
    COMPROBAR(parametros == 124439808);

    // Las capas lineales van traspuestas a [salida, entrada].
    COMPROBAR(p.tensor("wte.weight", {50257, 768}) != nullptr);
    COMPROBAR(p.tensor("h.0.attn.c_attn.weight", {2304, 768}) != nullptr);
    COMPROBAR(p.tensor("h.11.mlp.c_fc.weight", {3072, 768}) != nullptr);
    COMPROBAR(p.tensor("h.11.mlp.c_proj.weight", {768, 3072}) != nullptr);
    COMPROBAR(lanza([&] { p.tensor("h.0.mlp.c_fc.weight", {768, 3072}); }));
    COMPROBAR(lanza([&] { p.tensor("no.existe", {1}); }));

    // Archivos inválidos: magia incorrecta y archivo truncado tras la cabecera.
    const char* tmp = "build/test_pesos_corrupto.bin";
    std::ofstream(tmp, std::ios::binary) << "NOESPESO........................";
    COMPROBAR(lanza([&] { cargar_pesos(tmp); }));
    std::ofstream(tmp, std::ios::binary).write("GPT2PESO\1\0\0\0", 12);
    COMPROBAR(lanza([&] { cargar_pesos(tmp); }));
    COMPROBAR(lanza([&] { cargar_pesos("modelos/no_existe.bin"); }));
    return resumen();
}
