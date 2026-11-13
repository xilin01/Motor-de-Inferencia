// Compara codificar() con los ids del tokenizador de referencia y comprueba que decodificar() devuelve el texto.
#include "prueba.hpp"
#include "tokenizador.hpp"

#include <fstream>
#include <sstream>
#include <string>

std::string de_hex(const std::string& hex) {
    std::string bytes;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) bytes += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
    return bytes;
}

int main() {
    const Tokenizador tok("modelos/gpt2_vocab.bin");
    COMPROBAR(tok.n_tokens() == 50257);
    COMPROBAR(tok.fin_de_texto() == 50256);

    std::ifstream f("tests/datos/tokenizador.txt");
    COMPROBAR(f.is_open());
    int casos = 0;
    std::string comentario, hex, linea_ids;
    while (std::getline(f, comentario)) {
        if (comentario.starts_with("# Generado")) continue;
        if (!std::getline(f, hex) || !std::getline(f, linea_ids)) break;
        const std::string texto = de_hex(hex);
        std::vector<int> esperados;
        std::istringstream ss(linea_ids);
        for (int id; ss >> id;) esperados.push_back(id);

        const std::vector<int> ids = tok.codificar(texto);
        if (ids != esperados) {
            ++g_fallos;
            std::printf("FALLO al codificar %s\n  esperado:", comentario.c_str());
            for (int id : esperados) std::printf(" %d", id);
            std::printf("\n  obtenido:");
            for (int id : ids) std::printf(" %d", id);
            std::printf("\n");
        }
        COMPROBAR(tok.decodificar(ids) == texto);
        ++casos;
    }
    COMPROBAR(casos > 300);

    // UTF-8 inválido: no hay referencia posible, pero no debe romperse y debe conservar los bytes.
    for (std::string malo : {"\xff\xfe abc", "corte \xe4\xb8", "\xc3", "\x80\x80 \xf0\x9f\x98", "ok \xed\xa0\x80 fin"}) {
        COMPROBAR(tok.decodificar(tok.codificar(malo)) == malo);
    }

    // Ids fuera de rango.
    bool lanzo = false;
    try {
        const int malos[] = {15496, 50257};
        tok.decodificar(malos);
    } catch (const std::out_of_range&) {
        lanzo = true;
    }
    COMPROBAR(lanzo);

    std::printf("%d casos de referencia\n", casos);
    return resumen();
}
