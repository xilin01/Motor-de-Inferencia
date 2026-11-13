"""Genera con las implementaciones de referencia (transformers y PyTorch) los datos contra los que se prueban los tests de C++."""
import random
from pathlib import Path
from transformers import GPT2Tokenizer


RAIZ = Path(__file__).resolve().parent.parent
SEMILLA = 2106
N_ALEATORIOS = 300
TEXTOS = ['', ' ', '  ', 'a', 'Hello, world!', 'Hello  world', '  dos espacios delante', 'espacios detrás   ', 'línea uno\nlínea dos\n\n\nlínea cinco',
          '\ttabulador\ty\r\nretorno de carro', "I'm sure they've said it's fine, we'll see, he'd know, don't, you're", "'s 't 're 've 'm 'll 'd", "I'M SURE IT'S DON'T",
          "''s' 'll'll x'y !'s", 'El pingüino comió ñandú en Düsseldorf: ¿qué tal? ¡Genial!', 'naïve café résumé façade Ærøskøbing',
          'Precio: 1.234,56 € (el 12/11/2026 a las 09:30)', '3.14159 2^10=1024 1e-4 0x1F ½ ² ٣ Ⅷ ④', '北京是中华人民共和国的首都。', 'こんにちは世界、ありがとう！', '안녕하세요 세계',
          'Привет, мир! Как дела?', 'Γειά σου Κόσμε', 'مرحبا بالعالم', 'שלום עולם', 'नमस्ते दुनिया', 'สวัสดีชาวโลก', '😀 🎉 emoji 👍🏽 familia 👨‍👩‍👧‍👦 bandera 🇪🇸 corazón ❤️',
          'é combinado y é precompuesto', 'espacio duro, espacio　ideográfico, fino aquí y separador', 'ancho​cero y BOM﻿ dentro',
          'control \x00 \x1c \x7f \x85 fin', 'for (int i = 0; i < n; ++i) { x[i] += 1.0f; }  // código', 'https://example.com/ruta?a=1&b=dos#ancla',
          'snake_case camelCase SCREAMING_CASE kebab-case', '!!! ??? ... --- *** ((())) [[]] {{}} <<>>', 'a' * 300, ' ' * 50 + 'fin', 'ab ' * 100, '\n' * 20,
          'Texto antes<|endoftext|>Texto después', '<|endoftext|>', '<|endoftext|><|endoftext|> x', '<|endoftext', 'supercalifragilisticoespialidoso antidisestablishmentarianism',
          'The quick brown fox jumps over the lazy dog.', 'MAYÚSCULAS minúsculas MiXtO', '𝕳𝖊𝖑𝖑𝖔 𝒲ℴ𝓇𝓁𝒹 fuera del plano básico 𐍈', '\U0010ffff uso privado y último punto']
ALFABETO = (list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') + list(' ' * 12) + list("''.,;:!?-_()[]{}<>|/\\@#$%^&*+=~`\"") +
            list('\n\t\r\x0b\x0c\x1c\x85       　​‍️́̈') + list('áéíóúñüçßøæÁÉÑ') + list('αβγδΩЖдяш') +
            list('中文字日本語かなカナ한글') + list('مرحباשלוםनमस्ते') + list('½²³٣Ⅷ④०९') + list('😀🎉👍🏽❤👨👩🇪🇸𝕳𐍈') + ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", '<|endoftext|>'])


def textos_aleatorios() -> list[str]:
    """Cadenas al azar con semilla fija que mezclan alfabetos, espacios raros y contracciones, para encontrar casos que a mano no se me ocurren."""
    azar = random.Random(SEMILLA)
    return [''.join(azar.choices(ALFABETO, k=azar.randint(1, 60))) for _ in range(N_ALEATORIOS)]


def escribir_tokenizador(tok: GPT2Tokenizer, ruta: Path) -> None:
    """
    Escribe tres líneas por caso: un comentario legible, el texto en hexadecimal (UTF-8) y los ids de referencia separados por espacios.
    El hexadecimal evita tener que escapar saltos de línea y caracteres de control.
    """
    textos = TEXTOS + textos_aleatorios()
    lineas = ['# Generado por tools/referencias.py. Por caso: comentario, texto UTF-8 en hexadecimal, ids de referencia.']

    for texto in textos: lineas += [f'# {texto!r}', texto.encode('utf-8').hex(), ' '.join(str(i) for i in tok.encode(texto))]
    ruta.parent.mkdir(exist_ok=True)
    ruta.write_text('\n'.join(lineas) + '\n', encoding='utf-8', newline='\n')
    print(f'[tokenizador] {len(textos)} casos -> {ruta}')


if __name__ == '__main__':
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    escribir_tokenizador(tok, RAIZ / 'tests' / 'datos' / 'tokenizador.txt')
