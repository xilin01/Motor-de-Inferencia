"""Genera src/unicode_tablas.hpp con los rangos de letras, números y espacios que distingue el pre-tokenizador de referencia."""
from pathlib import Path
from tokenizers.pre_tokenizers import PreTokenizer
from transformers import GPT2Tokenizer


RUTA_SALIDA = Path(__file__).resolve().parent.parent / 'src' / 'unicode_tablas.hpp'
SUSTITUTOS = range(0xD800, 0xE000) # No son caracteres: no se pueden codificar en UTF-8.
POR_LINEA = 6
Rangos = list[tuple[int, int]] # Intervalos cerrados de puntos de código, ordenados y disjuntos.


def clasificar(pre: PreTokenizer, c: str) -> str:
    """
    Deduce la clase de un carácter por cómo lo trocea el pre-tokenizador de referencia.
    Entre dos letras, una letra da 1 trozo, un espacio repetido da 4 y lo demás da 3; entre dos cifras, un número da 1 trozo.
    """
    if c == ' ': return 'espacio' # El patrón lo trata aparte (' ?'), así que la sonda de abajo no sirve para él.
    n = len(pre.pre_tokenize_str(f'a{c}{c}a'))
    if n == 1: return 'letra'
    if n == 4: return 'espacio'
    return 'numero' if len(pre.pre_tokenize_str(f'1{c}1')) == 1 else 'otro'


def a_rangos(puntos: list[int]) -> Rangos:
    rangos = []

    for p in puntos:
        if rangos and rangos[-1][1] == p - 1: rangos[-1] = (rangos[-1][0], p)
        else: rangos.append((p, p))
    return rangos


def tabla_cpp(nombre: str, rangos: Rangos) -> str:
    celdas = [f'{{0x{a:04X}, 0x{b:04X}}}' for a, b in rangos]
    lineas = [', '.join(celdas[i:i + POR_LINEA]) for i in range(0, len(celdas), POR_LINEA)]
    return f'inline constexpr Rango {nombre}[] = {{\n    ' + ',\n    '.join(lineas) + ',\n};\n'


if __name__ == '__main__':
    import tokenizers
    pre = GPT2Tokenizer.from_pretrained('gpt2').backend_tokenizer.pre_tokenizer
    puntos = {'letra': [], 'numero': [], 'espacio': [], 'otro': []}

    for p in range(0x110000):
        if p not in SUSTITUTOS: puntos[clasificar(pre, chr(p))].append(p)
    letras, numeros, espacios = a_rangos(puntos['letra']), a_rangos(puntos['numero']), a_rangos(puntos['espacio'])
    cabecera = (f'// GENERADO por tools/tablas_unicode.py a partir del pre-tokenizador de tokenizers {tokenizers.__version__}. No editar.\n'
                '// Rangos cerrados de puntos de código, ordenados: \\p{L}, \\p{N} y \\s del patrón de GPT-2.\n'
                '#pragma once\n\nstruct Rango {\n    char32_t ini, fin;\n};\n\n')
    RUTA_SALIDA.write_text(cabecera + '\n'.join([tabla_cpp('LETRAS', letras), tabla_cpp('NUMEROS', numeros), tabla_cpp('ESPACIOS', espacios)]), encoding='utf-8', newline='\n')
    print(f'[tablas] letras: {len(puntos["letra"])} en {len(letras)} rangos | numeros: {len(puntos["numero"])} en {len(numeros)} rangos | '
          f'espacios: {len(puntos["espacio"])} en {len(espacios)} rangos -> {RUTA_SALIDA}')
    print(f'[tablas] espacios: {" ".join(f"U+{p:04X}" for p in puntos["espacio"])}')
