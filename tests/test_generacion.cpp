// Muestreo (greedy, temperatura, top-k) sobre logits sintéticos y generación greedy contra PyTorch.
#include "generar.hpp"
#include "prueba.hpp"
#include "variante.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

std::vector<int> leer_ids(const std::string& linea) {
    std::vector<int> ids;
    std::istringstream ss(linea);
    for (int id; ss >> id;) ids.push_back(id);
    return ids;
}

void probar_muestreo() {
    // Logits ln(1), ln(2), ln(4), ln(8): con temperatura 1 las probabilidades son 1/15, 2/15, 4/15 y 8/15.
    const std::vector<float> logits{0.0f, std::log(2.0f), std::log(4.0f), std::log(8.0f)};
    std::mt19937_64 azar(1);
    const int N = 30000;
    const auto frecuencias = [&](const Muestreo& m) {
        std::vector<int> veces(logits.size(), 0);
        for (int i = 0; i < N; ++i) ++veces[elegir_token(logits, m, azar)];
        std::vector<float> f;
        for (int v : veces) f.push_back(static_cast<float>(v) / N);
        return f;
    };

    // Greedy: siempre el máximo.
    COMPROBAR(elegir_token(logits, {}, azar) == 3);

    // Temperatura 1: las frecuencias siguen al softmax.
    auto f = frecuencias({.temperatura = 1.0f});
    for (int i = 0; i < 4; ++i) COMPROBAR(std::fabs(f[i] - std::pow(2.0f, static_cast<float>(i)) / 15.0f) < 0.01f);

    // Temperatura 2: logits a la mitad, probabilidades proporcionales a 1, sqrt(2), 2 y 2·sqrt(2).
    f = frecuencias({.temperatura = 2.0f});
    const float suma = 3.0f + 3.0f * std::sqrt(2.0f);
    for (int i = 0; i < 4; ++i) COMPROBAR(std::fabs(f[i] - std::pow(std::sqrt(2.0f), static_cast<float>(i)) / suma) < 0.01f);

    // Temperatura muy baja: equivale a greedy.
    f = frecuencias({.temperatura = 0.01f});
    COMPROBAR(f[3] > 0.999f);

    // Top-k = 2: solo salen los dos mejores, con probabilidades 4/12 y 8/12.
    f = frecuencias({.temperatura = 1.0f, .top_k = 2});
    COMPROBAR(f[0] == 0.0f && f[1] == 0.0f);
    COMPROBAR(std::fabs(f[2] - 1.0f / 3.0f) < 0.01f);

    // Top-k = 1 es greedy con cualquier temperatura; top-k mayor que el vocabulario no recorta.
    f = frecuencias({.temperatura = 5.0f, .top_k = 1});
    COMPROBAR(f[3] == 1.0f);
    f = frecuencias({.temperatura = 1.0f, .top_k = 100});
    COMPROBAR(f[0] > 0.0f);
}

// uso: test_generacion [ingenua|avx2] [hilos]
int main(int argc, char** argv) {
    probar_muestreo();

    Modelo modelo(cargar_pesos("modelos/gpt2_124M.bin"));
    modelo.configurar(leer_variante(argc, argv, 1));

    // Greedy contra PyTorch: mismos tokens, uno a uno.
    std::ifstream f("tests/datos/greedy.txt");
    COMPROBAR(f.is_open());
    int casos = 0;
    std::string linea_prompt, linea_esperados;
    while (std::getline(f, linea_prompt)) {
        if (linea_prompt.starts_with("#")) continue;
        if (!std::getline(f, linea_esperados)) break;
        const std::vector<int> prompt = leer_ids(linea_prompt);
        std::vector<int> esperados = prompt;
        for (int id : leer_ids(linea_esperados)) esperados.push_back(id);
        const int n_nuevos = static_cast<int>(esperados.size() - prompt.size());

        COMPROBAR(generar(modelo, prompt, n_nuevos, {}, -1, {}, true) == esperados);   // con KV cache
        COMPROBAR(generar(modelo, prompt, n_nuevos, {}, -1, {}, false) == esperados);  // sin caché
        ++casos;
    }
    COMPROBAR(casos == 3);

    // Con el último prompt: misma semilla, misma salida; se avisa de cada token; se para en id_parada.
    const std::vector<int> prompt{15496, 11};  // "Hello,"
    const Muestreo sorteo{.temperatura = 0.8f, .top_k = 40, .semilla = 7};
    std::vector<int> avisados;
    const std::vector<int> a = generar(modelo, prompt, 4, sorteo, -1, [&](int id) { avisados.push_back(id); });
    COMPROBAR(a.size() == 6);
    COMPROBAR(a == generar(modelo, prompt, 4, sorteo));
    COMPROBAR(avisados == std::vector<int>(a.begin() + 2, a.end()));
    COMPROBAR(generar(modelo, prompt, 4, sorteo, a[2]).size() == 3);
    COMPROBAR(a == generar(modelo, prompt, 4, sorteo, -1, {}, false));  // el sorteo no depende de la caché

    // Contexto lleno: no se añade nada.
    COMPROBAR(generar(modelo, std::vector<int>(1024, 0), 5, {}).size() == 1024);
    return resumen();
}
