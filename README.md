# Motor de inferencia de GPT-2 en C++20

Motor de inferencia de GPT-2 small (124 M de parámetros) escrito desde cero en C++20, sin
dependencias externas en el binario: tokenizador BPE, forward del transformer, generación de texto
y tres optimizaciones para CPU medidas paso a paso (KV cache, AVX2 + FMA, varios hilos).

El objetivo es didáctico: entender cómo se ejecuta un transformer y cómo se optimiza en CPU. Por
eso la versión ingenua se conserva junto a cada versión optimizada y se puede elegir al ejecutar.

```
> build\Release\gpt2.exe --prompt "Hello, my name is" --tokens 20
Hello, my name is John. I'm a writer, and I'm a writer. I'm a writer. I'm
```

## Resultados

Tokens generados por segundo. Cada fila añade una optimización a las anteriores. Todas las cifras
están medidas con `bench.exe` en la máquina descrita debajo; ninguna es estimada.

| Variante | Hilos | tokens/s | Frente a la base | Frente a 1 hilo |
|---|---:|---:|---:|---:|
| Ingenua, sin caché | 1 | 0,48 | x1,0 | |
| + KV cache | 1 | 18,31 | x37,8 | |
| + AVX2 y FMA | 1 | 62,63 | x129,5 | |
| + hilos | 2 | 91,08 | x188,3 | x1,45 |
| + hilos | 4 | 100,66 | x208,1 | x1,61 |
| + hilos | 6 | 114,99 | x237,7 | x1,84 |
| + hilos | 8 | 113,99 | x235,6 | x1,82 |
| + hilos | 14 | 115,54 | x238,8 | x1,84 |
| + hilos | 20 | 113,20 | x234,0 | x1,81 |

**Condiciones.** Intel Core i7-13700H (6 núcleos P + 8 núcleos E, 20 hilos lógicos), 32 GB de
DDR5 a 5200 MT/s en dos canales, Windows 11, enchufado a la corriente, plan de energía
"Equilibrado". MSVC 14.43, configuración Release (`/O2`). Prompt fijo de 13 tokens, 100 tokens
generados con greedy, mediana de 3 repeticiones.

**Lectura de la tabla.**

- *KV cache, x37,8.* Sin caché cada token recalcula todo el contexto: 6 250 posiciones procesadas
  para generar 100 tokens. Con caché son 112. La ganancia crece con la longitud del texto.
- *AVX2 + FMA, x3,4 sobre el paso anterior.* Ocho floats por instrucción darían x8 si el límite
  fuera la aritmética. No lo es: cada token recorre los 494 MB de pesos una vez y a 62,6 tokens/s
  eso son 31 GB/s leídos de RAM.
- *Hilos, x1,84 como máximo.* Se satura con 6 hilos. A 115 tokens/s se leen 57 GB/s, y dos
  canales de DDR5 a 5200 MT/s dan como máximo teórico 83 GB/s: los núcleos comparten el ancho de
  banda de memoria y añadir más no trae más datos por segundo.
- *Comprobación del límite de memoria.* Con el mismo núcleo AVX2 y un hilo, el motor hace 25 mil
  millones de multiplicaciones-suma por segundo procesando de golpe un prompt de 181 tokens (cada
  peso se reutiliza 181 veces desde la caché del procesador) y 8 mil millones generando token a
  token (cada peso se lee de RAM una vez).

**Variabilidad.** La fila sin caché es la menos estable: en dos medidas anteriores de una sola
repetición dio 0,39 y 0,41 tokens/s. Las demás filas se repitieron con diferencias de hasta el 5 %
(KV cache ingenuo) y menores del 3 % (AVX2 y hilos) entre ejecuciones del benchmark.

## Arquitectura

```
texto UTF-8
   │  Tokenizador::codificar      troceado previo + fusiones BPE
   ▼
ids de token
   │  Modelo::avanzar             embeddings, 12 bloques (atención + MLP), LayerNorm, cabeza
   ▼                              guarda claves y valores de atención en la caché
logits del siguiente token
   │  elegir_token                greedy, temperatura, top-k
   ▼
id nuevo ──► Tokenizador::decodificar ──► texto
   │
   └──────── vuelve a Modelo::avanzar con solo ese token
```

| Archivo | Contenido |
|---|---|
| `tools/exportar.py` | Descarga los pesos oficiales y los escribe en el formato binario propio |
| `tools/tablas_unicode.py` | Genera `src/unicode_tablas.hpp` (clases de caracteres del tokenizador) |
| `tools/referencias.py` | Genera con PyTorch y `transformers` los datos de referencia de los tests |
| `src/pesos.cpp` | Cargador del archivo de pesos, con validación y suma de control |
| `src/tokenizador.cpp` | BPE a nivel de bytes: codificar y decodificar |
| `src/operaciones.cpp` | Capa lineal ingenua, LayerNorm, GELU, softmax |
| `src/lineal_avx2.cpp` | Capa lineal con AVX2 + FMA (único archivo compilado con `/arch:AVX2`) |
| `src/cpu.cpp` | Detección de AVX2 y FMA en tiempo de ejecución |
| `src/hilos.cpp` | Grupo de hilos con reparto dinámico |
| `src/modelo.cpp` | `forward` (ingenuo, recalcula todo) y `avanzar` (incremental, con KV cache) |
| `src/generar.cpp` | Muestreo y bucle de generación |
| `src/main.cpp` | CLI |
| `bench/bench.cpp` | Benchmark |
| `tests/` | Tests, uno por área, lanzados con CTest |
| `docs/FORMATO.md` | Especificación de los formatos binarios |

Las variantes se eligen al ejecutar, no al compilar: un mismo binario contiene la versión ingenua
y las optimizadas, y los tests y el benchmark las recorren todas.

## Requisitos

- Windows 10 u 11 de 64 bits.
- Visual Studio 2022 con el componente de C++ (incluye CMake).
- Para la variante AVX2: CPU con AVX2 y FMA. Se comprueba al arrancar; sin ellas se usa la ingenua.
- Python 3 con `torch`, `transformers` y `numpy`, solo para exportar los pesos y generar las
  referencias. Versiones usadas: Python 3.13, torch 2.15, transformers 5.22, tokenizers 0.23.

## Cómo compilar

Desde "Developer PowerShell for VS 2022" (tiene `cmake` en el PATH), en la raíz del repositorio:

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

El ejecutable queda en `build\Release\gpt2.exe`. Se enlaza con `/MT`: solo depende de
`KERNEL32.dll`.

## Pesos

Los pesos no están en el repositorio. Se descargan de Hugging Face (unos 550 MB) y se exportan a
`modelos/`:

```
pip install torch transformers numpy
python tools/exportar.py
```

Genera `modelos/gpt2_124M.bin` (498 MB) y `modelos/gpt2_vocab.bin` (1 MB). El formato está
documentado en [docs/FORMATO.md](docs/FORMATO.md).

## Uso

Desde la raíz del repositorio:

```
build\Release\gpt2.exe --prompt "The meaning of life is" --tokens 100
build\Release\gpt2.exe --prompt "Érase una vez" --tokens 50 --temperatura 0.8 --top-k 40 --semilla 7
```

| Opción | Significado | Por defecto |
|---|---|---|
| `--prompt "texto"` | Texto inicial (UTF-8) | obligatorio |
| `--tokens N` | Tokens a generar | 100 |
| `--temperatura T` | 0 = greedy; mayor = más variado | 0 |
| `--top-k K` | Sortear solo entre los K tokens más probables; 0 = todos | 0 |
| `--semilla S` | Semilla del sorteo | 0 |
| `--sin-cache` | Desactiva el KV cache | caché activado |
| `--matmul ingenua\|avx2` | Implementación de las capas lineales | `avx2` si la CPU lo soporta |
| `--hilos N` | Hilos para las capas lineales | 6, o los de la CPU si son menos |
| `--modelo RUTA`, `--vocab RUTA` | Archivos de pesos y vocabulario | `modelos/...` |

El texto sale por la salida estándar; la línea de estadísticas (tokens por segundo), por la de
error. La generación se detiene antes si sale el token de fin de texto o se llena el contexto.

## Tests

```
python tools/referencias.py
ctest --test-dir build -C Release --output-on-failure
```

`referencias.py` genera `modelos/ref_logits.bin` (452 MB) con los logits de PyTorch. Los casos del
tokenizador y de la generación greedy ya están en `tests/datos/`. La batería completa tarda unos
4 minutos; casi todo son los tests de la versión ingenua.

| Test | Qué comprueba |
|---|---|
| `test_pesos` | Carga, formas, número de parámetros y rechazo de archivos corruptos |
| `test_tokenizador` | 349 textos (Unicode, emoji, espacios raros, contracciones): ids idénticos a los de `transformers` e ida y vuelta |
| `test_lineal` | Capa lineal AVX2 contra la ingenua; cálculo por trozos idéntico bit a bit |
| `test_hilos` | El grupo de hilos reparte cada índice exactamente una vez |
| `test_logits*` | Logits de todas las posiciones contra PyTorch en 7 casos, uno de ellos con el contexto lleno (1024 tokens), en cada variante |
| `test_generacion*` | Greedy idéntico a PyTorch; temperatura y top-k contra las probabilidades teóricas |

Además de comparar con PyTorch, los tests exigen igualdad **bit a bit** entre el forward completo y
el incremental (KV cache), y entre 1 hilo y 8 hilos.

### Tolerancia numérica

El objetivo inicial era un error absoluto máximo de 1e-4 frente a los logits de PyTorch. **No se
alcanza, y ningún motor en float32 puede alcanzarlo**: el propio PyTorch en float32 se aparta del
mismo modelo calculado en float64 más que eso. Error absoluto máximo medido sobre todos los
logits de cada caso:

| Tokens | Ingenua vs PyTorch f32 | AVX2 vs PyTorch f32 | PyTorch f32 vs PyTorch f64 |
|-------:|-----------------------:|--------------------:|---------------------------:|
| 5      | 2,90e-4                | 1,07e-4             | 8,39e-5                    |
| 5      | 3,81e-4                | 1,60e-4             | 9,16e-5                    |
| 40     | 2,67e-4                | 1,45e-4             | 9,92e-5                    |
| 31     | 7,93e-4                | 4,73e-4             | 2,90e-4                    |
| 19     | 2,52e-4                | 1,37e-4             | 1,60e-4                    |
| 1      | 3,24e-4                | 1,11e-4             | 5,72e-5                    |
| 1024   | 4,91e-3                | 5,32e-3             | 2,47e-3                    |

Los logits llegan a valer 334 en valor absoluto, así que el peor error relativo es del orden de
1e-5. La causa es el orden de las sumas en float32: en un experimento, acumulando las capas
lineales y la LayerNorm en `double`, el error del motor frente a float64 en los prompts cortos
bajó a 6,1e-5 o menos, por debajo del error propio de PyTorch.

Umbrales de los tests: **1e-3** en los prompts cortos y **1e-2** en el caso de 1024 tokens.

## Cómo reproducir el benchmark

```
build\Release\bench.exe
```

Hay que lanzarlo desde la raíz del repositorio, con el portátil enchufado y sin otros programas
pesados. Opciones: `--tokens N` (100), `--repeticiones R` (3) y `--sin-base` (omite la variante
sin caché, que tarda varios minutos).

Qué mide: tokens generados por segundo con muestreo greedy a partir de un prompt fijo de 13
tokens. El tiempo incluye procesar el prompt y no incluye cargar el modelo. Cada variante se mide
3 veces y se da la mediana. La salida es la tabla de arriba en Markdown.

## Limitaciones conocidas

- **Plataforma.** Solo se ha compilado y probado en Windows x64 con MSVC. `src/main.cpp` usa la
  API de Windows y `src/cpu.cpp` intrínsecos de MSVC. No hay versión para ARM.
- **Modelo.** Solo se ha probado GPT-2 small. El exportador rechaza modelos con otra activación u
  otro epsilon de LayerNorm.
- **Contexto.** Máximo 1024 tokens. Al llenarse, la generación se detiene: no hay ventana
  deslizante.
- **Una secuencia cada vez.** No hay procesamiento por lotes.
- **Tolerancia.** No se alcanza el objetivo inicial de 1e-4 frente a PyTorch (ver arriba).
- **Límite de memoria.** Los pesos están en float32 sin cuantizar y cada token recorre 494 MB.
  Es lo que impide escalar con los hilos.
- **Atención sin optimizar.** La atención es escalar y de un solo hilo en todas las variantes. Con
  los contextos del benchmark (13 a 113 tokens) pesa poco; con contextos cercanos a 1024 pesa más
  y la velocidad baja. No está medido.
- **Hilos.** No se fijan los hilos a núcleos concretos ni se distingue entre núcleos P y E. El
  valor por defecto de 6 hilos está ajustado a la máquina de medida.
- **Muestreo.** Solo greedy, temperatura y top-k: no hay top-p ni penalización de repetición. La
  misma semilla solo garantiza el mismo texto con el mismo compilador y biblioteca estándar.
- **Tokenizador.** Las tablas Unicode se generaron sondeando `tokenizers` 0.23; con caracteres de
  versiones de Unicode posteriores puede discrepar de un tokenizador más nuevo. El UTF-8 inválido
  se acepta byte a byte y no tiene referencia con la que compararse.
- **Carga.** El archivo de pesos se lee entero en memoria (unos 500 MB de RAM); no se usa `mmap`.

## Licencia

Código bajo licencia [MIT](LICENSE). Los pesos de GPT-2 son de OpenAI, se descargan de Hugging Face
y no se distribuyen con este repositorio.
