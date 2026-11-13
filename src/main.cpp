// CLI: gpt2 --prompt "..." --tokens 100
#include "cpu.hpp"
#include "generar.hpp"
#include "tokenizador.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace {

const char* const USO =
    "uso: gpt2 --prompt \"texto\" [opciones]\n"
    "  --tokens N        tokens a generar (por defecto 100)\n"
    "  --temperatura T   0 = greedy (por defecto); mayor = más variado\n"
    "  --top-k K         sortear solo entre los K tokens más probables (0 = todos)\n"
    "  --semilla S       semilla del sorteo (por defecto 0)\n"
    "  --modelo RUTA     archivo de pesos (por defecto modelos/gpt2_124M.bin)\n"
    "  --vocab RUTA      archivo de vocabulario (por defecto modelos/gpt2_vocab.bin)\n"
    "  --sin-cache       desactiva el KV cache: recalcula todo el contexto en cada token\n"
    "  --matmul M        ingenua o avx2 (por defecto avx2 si la CPU lo soporta)\n"
    "  --hilos N         hilos para las capas lineales (por defecto 6, o los que tenga la CPU si son menos)\n";

// Windows entrega los argumentos en UTF-16; el motor trabaja en UTF-8.
std::string a_utf8(const wchar_t* w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');  // n cuenta el terminador
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

// Longitud del mayor prefijo de s que no deja un carácter UTF-8 a medias. Un token puede acabar
// en mitad de un carácter (un emoji son 4 bytes) y la consola mostraría basura si se imprime así.
size_t prefijo_completo(const std::string& s) {
    // Retrocede hasta el byte inicial del último carácter (los de continuación son 10xxxxxx).
    size_t i = s.size();
    while (i > 0 && s.size() - i < 4 && (static_cast<unsigned char>(s[i - 1]) >> 6) == 0b10) --i;
    if (i == 0) return s.size();
    const unsigned char b = static_cast<unsigned char>(s[i - 1]);
    const size_t largo = b < 0x80 ? 1 : (b >> 5) == 0b110 ? 2 : (b >> 4) == 0b1110 ? 3 : (b >> 3) == 0b11110 ? 4 : 1;
    return s.size() - (i - 1) < largo ? i - 1 : s.size();
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    // La consola interpreta la salida con su página de códigos; se pone UTF-8 y se restaura al salir.
    const UINT cp_anterior = GetConsoleOutputCP();
    SetConsoleOutputCP(CP_UTF8);
    struct Restaurar {
        UINT cp;
        ~Restaurar() { SetConsoleOutputCP(cp); }
    } restaurar{cp_anterior};

    std::string prompt;
    std::filesystem::path ruta_modelo = "modelos/gpt2_124M.bin", ruta_vocab = "modelos/gpt2_vocab.bin";
    int n_tokens = 100;
    Muestreo muestreo;
    bool hay_prompt = false, kv_cache = true;
    Opciones opciones;  // por defecto, lo más rápido que soporte la máquina
    if (cpu_tiene_avx2_fma()) opciones.matmul = Matmul::Avx2;
    // En la máquina donde se midió, más de 6 hilos no mejora: el límite es la memoria (ver README).
    opciones.hilos = static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 1u, 6u));

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string opcion = a_utf8(argv[i]);
            if (opcion == "--sin-cache") {
                kv_cache = false;
                continue;
            }
            if (i + 1 >= argc) throw std::invalid_argument("falta el valor de " + opcion);
            const wchar_t* valor = argv[++i];
            if (opcion == "--prompt") {
                prompt = a_utf8(valor);
                hay_prompt = true;
            } else if (opcion == "--tokens") {
                n_tokens = std::stoi(valor);
            } else if (opcion == "--temperatura") {
                muestreo.temperatura = std::stof(valor);
            } else if (opcion == "--top-k") {
                muestreo.top_k = std::stoi(valor);
            } else if (opcion == "--semilla") {
                muestreo.semilla = std::stoull(valor);
            } else if (opcion == "--modelo") {
                ruta_modelo = valor;
            } else if (opcion == "--vocab") {
                ruta_vocab = valor;
            } else if (opcion == "--matmul") {
                const std::string m = a_utf8(valor);
                if (m != "ingenua" && m != "avx2") throw std::invalid_argument("--matmul debe ser ingenua o avx2");
                opciones.matmul = m == "avx2" ? Matmul::Avx2 : Matmul::Ingenua;
            } else if (opcion == "--hilos") {
                opciones.hilos = std::stoi(valor);
            } else {
                throw std::invalid_argument("opción desconocida: " + opcion);
            }
        }
        if (!hay_prompt) throw std::invalid_argument("falta --prompt");
        if (n_tokens < 0 || muestreo.temperatura < 0.0f || muestreo.top_k < 0) {
            throw std::invalid_argument("--tokens, --temperatura y --top-k no pueden ser negativos");
        }
    } catch (const std::exception& e) {  // incluye los errores de std::stoi y std::stof
        std::fprintf(stderr, "error: %s\n\n%s", e.what(), USO);
        return 2;
    }

    try {
        const Tokenizador tok(ruta_vocab);
        Modelo modelo(cargar_pesos(ruta_modelo));
        modelo.configurar(opciones);

        std::vector<int> ids = tok.codificar(prompt);
        if (ids.empty()) ids.push_back(tok.fin_de_texto());  // GPT-2 empieza los documentos con este token
        if (ids.size() >= modelo.config().n_ctx) throw std::invalid_argument("el prompt no cabe en el contexto del modelo");

        std::fputs(prompt.c_str(), stdout);
        std::string pendiente;  // bytes generados que aún no forman caracteres completos
        const auto inicio = std::chrono::steady_clock::now();
        const size_t n_prompt = ids.size();
        ids = generar(modelo, std::move(ids), n_tokens, muestreo, tok.fin_de_texto(), [&](int id) {
            if (id == tok.fin_de_texto()) return;
            pendiente += tok.decodificar({&id, 1});
            const size_t n = prefijo_completo(pendiente);
            std::fwrite(pendiente.data(), 1, n, stdout);
            std::fflush(stdout);
            pendiente.erase(0, n);
        }, kv_cache);
        const double segundos = std::chrono::duration<double>(std::chrono::steady_clock::now() - inicio).count();
        std::fputs(pendiente.c_str(), stdout);
        std::fputs("\n", stdout);
        std::fflush(stdout);  // que el texto salga antes que la línea de estadísticas de stderr

        const size_t generados = ids.size() - n_prompt;
        std::fprintf(stderr, "[%zu tokens de prompt, %zu generados en %.2f s: %.2f tokens/s]\n", n_prompt, generados, segundos,
                     segundos > 0 ? static_cast<double>(generados) / segundos : 0.0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
