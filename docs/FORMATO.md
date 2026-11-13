# Formatos binarios

`tools/exportar.py` genera dos archivos en `modelos/`. Todos los enteros y los `float32` son
little-endian. No hay relleno ni alineación entre campos.

## `gpt2_124M.bin`: pesos

### Cabecera (36 bytes)

| Desplazamiento | Tipo      | Campo        | Valor en GPT-2 small |
|---------------:|-----------|--------------|----------------------|
| 0              | `char[8]` | magia        | `GPT2PESO`           |
| 8              | `uint32`  | versión      | 1                    |
| 12             | `uint32`  | `n_vocab`    | 50257                |
| 16             | `uint32`  | `n_ctx`      | 1024                 |
| 20             | `uint32`  | `n_embd`     | 768                  |
| 24             | `uint32`  | `n_capas`    | 12                   |
| 28             | `uint32`  | `n_cabezas`  | 12                   |
| 32             | `uint32`  | `n_tensores` | 148                  |

### Tensores

Después de la cabecera van `n_tensores` registros seguidos:

| Tipo              | Campo                                                              |
|-------------------|--------------------------------------------------------------------|
| `uint32`          | longitud del nombre en bytes (1 a 256)                             |
| `char[longitud]`  | nombre en ASCII, sin terminador                                    |
| `uint32`          | `n_dims` (1 a 4)                                                   |
| `uint32[n_dims]`  | forma                                                              |
| `uint64`          | suma de control                                                    |
| `float32[N]`      | datos en orden fila-mayor, con `N` igual al producto de la forma   |

**Suma de control.** Cada `float32` se reinterpreta como `uint32` y se suman todos módulo 2^64.
Detecta truncados, bytes corruptos y errores de orden de bytes. El cargador la comprueba siempre.

**Nombres.** Son los de Hugging Face sin el prefijo `transformer.`:

| Nombre                                 | Forma          | Uso                                         |
|----------------------------------------|----------------|---------------------------------------------|
| `wte.weight`                           | [50257, 768]   | embedding de tokens y cabeza final          |
| `wpe.weight`                           | [1024, 768]    | embedding de posiciones                     |
| `h.{i}.ln_1.weight`, `h.{i}.ln_1.bias` | [768]          | LayerNorm previa a la atención              |
| `h.{i}.attn.c_attn.weight`             | [2304, 768]    | proyección conjunta a consulta, clave, valor |
| `h.{i}.attn.c_attn.bias`               | [2304]         |                                             |
| `h.{i}.attn.c_proj.weight`             | [768, 768]     | proyección de salida de la atención         |
| `h.{i}.attn.c_proj.bias`               | [768]          |                                             |
| `h.{i}.ln_2.weight`, `h.{i}.ln_2.bias` | [768]          | LayerNorm previa al MLP                     |
| `h.{i}.mlp.c_fc.weight`                | [3072, 768]    | primera capa del MLP                        |
| `h.{i}.mlp.c_fc.bias`                  | [3072]         |                                             |
| `h.{i}.mlp.c_proj.weight`              | [768, 3072]    | segunda capa del MLP                        |
| `h.{i}.mlp.c_proj.bias`                | [768]          |                                             |
| `ln_f.weight`, `ln_f.bias`             | [768]          | LayerNorm final                             |

con `i` de 0 a 11.

**Diferencias con el `state_dict` original.**

- Las cuatro capas lineales de cada bloque se guardan **traspuestas**, con forma
  `[salida, entrada]`. En Hugging Face son `Conv1D` con forma `[entrada, salida]`. Así la fila `o`
  contiene todos los pesos de la neurona de salida `o` y calcularla es un producto escalar sobre
  memoria contigua: `y[o] = b[o] + Σ_k W[o][k]·x[k]`.
- `lm_head.weight` no se guarda: es el mismo tensor que `wte.weight`. El exportador lo comprueba.

## `gpt2_vocab.bin`: vocabulario

| Tipo                    | Campo                                                          |
|-------------------------|----------------------------------------------------------------|
| `char[8]`               | magia `GPT2VOCA`                                               |
| `uint32`                | versión (1)                                                    |
| `uint32`                | `n_tokens` (50257)                                             |
| por token, en orden de id | `uint16` longitud y después los bytes del token              |
| `uint32`                | `n_fusiones` (50000)                                           |
| por fusión, en orden de prioridad | `uint32` id izquierdo, `uint32` id derecho, `uint32` id resultante |
| `uint32`                | id de fin de texto (50256)                                     |

**Bytes reales.** El `vocab.json` original guarda cada token como texto en el que cada byte se ha
sustituido por un carácter visible (el espacio es `Ġ`). El exportador deshace esa sustitución y
guarda los bytes tal cual, de modo que decodificar es concatenar.

**Fusiones como ids.** El `merges.txt` original guarda pares de cadenas. Aquí cada fusión es un
trío de ids `(izquierda, derecha, resultado)`. La posición en la lista es la prioridad: la primera
fusión es la que se aplica antes.

## `ref_logits.bin`: referencia para los tests

Lo genera `tools/referencias.py` con PyTorch. No forma parte del motor.

| Tipo                                    | Campo                                               |
|-----------------------------------------|-----------------------------------------------------|
| `char[8]`                               | magia `GPT2REFL`                                    |
| `uint32`                                | `n_casos`                                           |
| por caso: `uint32`                      | `n_tokens`                                          |
| por caso: `int32[n_tokens]`             | ids de entrada                                      |
| por caso: `float32[n_tokens × n_vocab]` | logits de PyTorch en float32                        |
| por caso: `float32[n_tokens × n_vocab]` | logits de PyTorch en float64, redondeados a float32 |
