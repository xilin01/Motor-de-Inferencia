"""Descarga GPT-2 small (124M) de Hugging Face y lo exporta a los formatos binarios propios descritos en docs/FORMATO.md."""
import struct
from pathlib import Path
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


MODELO_HF = 'gpt2'
DIR_SALIDA = Path(__file__).resolve().parent.parent / 'modelos'
VERSION = 1
SUFIJOS_CONV1D = ('c_attn.weight', 'c_proj.weight', 'c_fc.weight') # Capas lineales de Hugging Face con pesos [entrada, salida].


def escribir_pesos(modelo: GPT2LMHeadModel, ruta: Path) -> None:
    """
    Escribe la cabecera con la configuración y después cada tensor con nombre, forma, suma de control y datos float32.
    Las capas lineales se trasponen a [salida, entrada] para que cada neurona de salida sea una fila contigua.
    """
    cfg = modelo.config
    estado = {n.removeprefix('transformer.'): t.numpy() for n, t in modelo.state_dict().items()}
    cabeza = estado.pop('lm_head.weight')

    if not np.array_equal(cabeza, estado['wte.weight']): raise ValueError('La cabeza final no comparte pesos con wte.')
    if cfg.activation_function != 'gelu_new' or cfg.layer_norm_epsilon != 1e-5: raise ValueError('Activación o epsilon distintos de los del motor.')
    with open(ruta, 'wb') as f:
        f.write(struct.pack('<8s7I', b'GPT2PESO', VERSION, cfg.vocab_size, cfg.n_positions, cfg.n_embd, cfg.n_layer, cfg.n_head, len(estado)))
        for nombre, t in estado.items():
            datos = np.ascontiguousarray(t.T if nombre.endswith(SUFIJOS_CONV1D) else t, dtype='<f4')
            suma = int(datos.view(np.uint32).sum(dtype=np.uint64)) # Suma de los patrones de bits, módulo 2^64.
            f.write(struct.pack('<I', len(nombre)) + nombre.encode('ascii'))
            f.write(struct.pack(f'<I{datos.ndim}IQ', datos.ndim, *datos.shape, suma))
            f.write(datos.tobytes())
    print(f'[pesos] {len(estado)} tensores, {sum(t.size for t in estado.values())} parámetros -> {ruta}')


def bytes_a_unicode() -> dict[int, str]:
    """Tabla de GPT-2 que da a cada byte un carácter visible: los imprimibles de Latin-1 se quedan igual y el resto pasa a U+0100 en adelante."""
    imprimibles = [*range(ord('!'), ord('~') + 1), *range(ord('¡'), ord('¬') + 1), *range(ord('®'), ord('ÿ') + 1)]
    resto = [b for b in range(256) if b not in imprimibles]
    return {b: chr(b) for b in imprimibles} | {b: chr(256 + i) for i, b in enumerate(resto)}


def escribir_vocabulario(tok: GPT2Tokenizer, ruta: Path) -> None:
    """
    Escribe los bytes reales de cada token y las fusiones BPE como tríos de ids (izquierda, derecha, resultado) en orden de prioridad.
    Deshace la tabla de bytes_a_unicode, que solo existe para poder guardar el vocabulario original como JSON.
    """
    import json
    bpe = json.loads(tok.backend_tokenizer.to_str())['model']
    vocab, fusiones = bpe['vocab'], bpe['merges']
    car_a_byte = {c: b for b, c in bytes_a_unicode().items()}
    tokens = [b''] * len(vocab)

    for texto, i in vocab.items(): tokens[i] = bytes(car_a_byte[c] for c in texto)
    with open(ruta, 'wb') as f:
        f.write(struct.pack('<8sII', b'GPT2VOCA', VERSION, len(tokens)))
        for t in tokens: f.write(struct.pack('<H', len(t)) + t)
        f.write(struct.pack('<I', len(fusiones)))
        for a, b in fusiones: f.write(struct.pack('<III', vocab[a], vocab[b], vocab[a + b]))
        f.write(struct.pack('<I', tok.eos_token_id))
    print(f'[vocabulario] {len(tokens)} tokens, {len(fusiones)} fusiones -> {ruta}')


if __name__ == '__main__':
    DIR_SALIDA.mkdir(exist_ok=True)
    escribir_pesos(GPT2LMHeadModel.from_pretrained(MODELO_HF), DIR_SALIDA / 'gpt2_124M.bin')
    escribir_vocabulario(GPT2Tokenizer.from_pretrained(MODELO_HF), DIR_SALIDA / 'gpt2_vocab.bin')
