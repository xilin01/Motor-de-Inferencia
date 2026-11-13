"""Genera con las implementaciones de referencia (transformers y PyTorch) los datos contra los que se prueban los tests de C++."""
import random
import struct
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


RAIZ = Path(__file__).resolve().parent.parent
SEMILLA = 2106
N_ALEATORIOS = 300
N_CTX = 1024
N_GREEDY = 12
PROMPTS_GREEDY = ['Hello, my name is', 'The capital of France is', '!']
TEXTOS = ['', ' ', '  ', 'a', 'Hello, world!', 'Hello  world', '  dos espacios delante', 'espacios detrás   ', 'línea uno\nlínea dos\n\n\nlínea cinco',
          '\ttabulador\ty\r\nretorno de carro', "I'm sure they've said it's fine, we'll see, he'd know, don't, you're", "'s 't 're 've 'm 'll 'd",
          "I'M SURE IT'S DON'T", "''s' 'll'll x'y !'s", 'El pingüino comió ñandú en Düsseldorf: ¿qué tal? ¡Genial!', 'naïve café résumé façade Ærøskøbing',
          'Precio: 1.234,56 € (el 12/11/2026 a las 09:30)', '3.14159 2^10=1024 1e-4 0x1F ½ ² ٣ Ⅷ ④', '北京是中华人民共和国的首都。', 'こんにちは世界、ありがとう！', '안녕하세요 세계',
          'Привет, мир! Как дела?', 'Γειά σου Κόσμε', 'مرحبا بالعالم', 'שלום עולם', 'नमस्ते दुनिया', 'สวัสดีชาวโลก',
          '😀 🎉 emoji 👍🏽 familia 👨‍👩‍👧‍👦 bandera 🇪🇸 corazón ❤️',
          'e\N{COMBINING ACUTE ACCENT} combinado y é precompuesto',
          'espacio\xa0duro, espacio\N{IDEOGRAPHIC SPACE}ideográfico, fino\N{THIN SPACE}aquí y\N{LINE SEPARATOR}separador',
          'ancho\N{ZERO WIDTH SPACE}cero y BOM\N{ZERO WIDTH NO-BREAK SPACE} dentro', 'control \x00 \x1c \x7f \x85 fin',
          'for (int i = 0; i < n; ++i) { x[i] += 1.0f; }  // código', 'https://example.com/ruta?a=1&b=dos#ancla',
          'snake_case camelCase SCREAMING_CASE kebab-case', '!!! ??? ... --- *** ((())) [[]] {{}} <<>>', 'a' * 300, ' ' * 50 + 'fin', 'ab ' * 100, '\n' * 20,
          'Texto antes<|endoftext|>Texto después', '<|endoftext|>', '<|endoftext|><|endoftext|> x', '<|endoftext',
          'supercalifragilisticoespialidoso antidisestablishmentarianism', 'The quick brown fox jumps over the lazy dog.', 'MAYÚSCULAS minúsculas MiXtO',
          '𝕳𝖊𝖑𝖑𝖔 𝒲ℴ𝓇𝓁𝒹 fuera del plano básico 𐍈', '\U0010ffff\U0000e000 uso privado y último punto']
INVISIBLES = ['\N{OGHAM SPACE MARK}', '\N{EM SPACE}', '\N{LINE SEPARATOR}', '\N{PARAGRAPH SEPARATOR}', '\N{NARROW NO-BREAK SPACE}',
              '\N{MEDIUM MATHEMATICAL SPACE}', '\N{IDEOGRAPHIC SPACE}', '\N{ZERO WIDTH SPACE}', '\N{ZERO WIDTH JOINER}', '\N{VARIATION SELECTOR-16}',
              '\N{COMBINING ACUTE ACCENT}', '\N{COMBINING DIAERESIS}']
ALFABETO = (list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') + list(' ' * 12) + list("''.,;:!?-_()[]{}<>|/@#$%^&*+=~`") + ['"', chr(92)] +
            list('\n\t\r\x0b\x0c\x1c\x85\xa0') + INVISIBLES + list('áéíóúñüçßøæÁÉÑ') + list('αβγδΩЖдяш') + list('中文字日本語かなカナ한글') + list('مرحباשלוםनमस्ते') +
            list('½²³٣Ⅷ④०९') + list('😀🎉👍🏽❤👨👩🇪🇸𝕳𐍈') + ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", '<|endoftext|>'])
PROMPTS = ['Hello, my name is', 'The capital of France is',
           'En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo',
           'def fibonacci(n):\n    if n < 2:\n        return n\n    return', '日本語のテキストと emoji 🎉 mezclados', '!']


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

    for texto in textos: lineas += [f'# {texto!a}', texto.encode('utf-8').hex(), ' '.join(str(i) for i in tok.encode(texto))]
    ruta.parent.mkdir(exist_ok=True)
    ruta.write_text('\n'.join(lineas) + '\n', encoding='utf-8', newline='\n')
    print(f'[tokenizador] {len(textos)} casos -> {ruta}')


def escribir_greedy(tok: GPT2Tokenizer, modelo: GPT2LMHeadModel, ruta: Path) -> None:
    """Escribe dos líneas por caso: los ids del prompt y los ids que añade la generación greedy, que es el argmax de los logits de la última posición."""
    lineas = ['# Generado por tools/referencias.py. Por caso: ids del prompt e ids generados con greedy por PyTorch.']

    for prompt in PROMPTS_GREEDY:
        ids = tok.encode(prompt)
        n = len(ids)

        for _ in range(N_GREEDY): ids.append(int(modelo(torch.tensor([ids])).logits[0, -1].argmax()))
        lineas += [' '.join(str(i) for i in ids[:n]), ' '.join(str(i) for i in ids[n:])]
        print(f'[greedy] {prompt!a} -> {tok.decode(ids[n:])!a}')
    ruta.write_text('\n'.join(lineas) + '\n', encoding='utf-8', newline='\n')
    print(f'[greedy] {len(PROMPTS_GREEDY)} casos -> {ruta}')


def escribir_logits(tok: GPT2Tokenizer, modelo: GPT2LMHeadModel, ruta: Path) -> None:
    """
    Escribe por caso los ids y los logits de PyTorch en todas las posiciones, calculados en float32 y en float64; el último caso llena el contexto máximo.
    Los de float64 se guardan redondeados a float32 y hacen de valor exacto: permiten medir el error propio de PyTorch en float32.
    """
    doble = GPT2LMHeadModel.from_pretrained('gpt2').double().eval()
    largo = tok.encode(' '.join(TEXTOS[4:36]) * 4)[:N_CTX]
    casos = [tok.encode(p) for p in PROMPTS] + [largo]

    if len(largo) != N_CTX: raise ValueError('El caso largo no llena el contexto.')
    with open(ruta, 'wb') as f:
        f.write(struct.pack('<8sI', b'GPT2REFL', len(casos)))
        for ids in casos:
            entrada = torch.tensor([ids])
            logits, exactos = modelo(entrada).logits[0], doble(entrada).logits[0]
            error = (logits.double() - exactos).abs().max().item()
            f.write(struct.pack(f'<I{len(ids)}i', len(ids), *ids))
            f.write(logits.numpy().astype('<f4').tobytes())
            f.write(exactos.numpy().astype('<f4').tobytes())
            print(f'[logits] {len(ids):4d} tokens | max |logit|: {logits.abs().max().item():7.2f} | error de PyTorch float32 frente a float64: {error:.2e}')
    print(f'[logits] {len(casos)} casos -> {ruta}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    modelo = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    escribir_tokenizador(tok, RAIZ / 'tests' / 'datos' / 'tokenizador.txt')
    escribir_greedy(tok, modelo, RAIZ / 'tests' / 'datos' / 'greedy.txt')
    escribir_logits(tok, modelo, RAIZ / 'modelos' / 'ref_logits.bin')
