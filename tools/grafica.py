"""Dibuja la gráfica de tokens/s del README en docs/, una imagen para el tema claro de GitHub y otra para el oscuro."""
from pathlib import Path
import matplotlib.pyplot as plt


DIR_SALIDA = Path(__file__).resolve().parent.parent / 'docs'
# Medianas de bench.exe: las mismas cifras que la tabla de Resultados del README.
MEDIDAS = [('Ingenua, sin caché', 0.48), ('+ KV cache', 18.31), ('+ AVX2 y FMA', 62.63), ('+ 2 hilos', 91.08), ('+ 4 hilos', 100.66),
           ('+ 6 hilos', 114.99), ('+ 8 hilos', 113.99), ('+ 14 hilos', 115.54), ('+ 20 hilos', 113.20)]
TEMAS = {'claro': ('#2a78d6', '#0b0b0b', '#52514e'), 'oscuro': ('#3987e5', '#f0efec', '#c3c2b7')} # Barra, texto principal y texto secundario.


def dibujar(ruta: Path, barra: str, tinta: str, tinta_2: str) -> None:
    nombres, valores = zip(*MEDIDAS)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    barras = ax.barh(nombres, valores, height=0.55, color=barra)
    ax.bar_label(barras, labels=[f'{v:.2f}'.replace('.', ',') for v in valores], padding=6, color=tinta)
    ax.set_title('Tokens generados por segundo', loc='left', color=tinta, fontweight='semibold', pad=12)
    ax.invert_yaxis() # La variante base arriba, como en la tabla.
    ax.set_xlim(0, 130) # Hueco para la cifra de la barra más larga.
    ax.xaxis.set_visible(False) # Cada barra lleva su cifra: el eje sobra.
    ax.tick_params(axis='y', length=0, labelcolor=tinta_2)
    ax.spines[:].set_visible(False)
    fig.savefig(ruta, transparent=True, bbox_inches='tight', metadata={'Date': None}) # Sin fecha: no cambia al regenerarlo.
    plt.close(fig)


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'Segoe UI', 'svg.hashsalt': 'grafica'}) # Con la sal fija el SVG sale idéntico en cada ejecución.
    for tema, colores in TEMAS.items(): dibujar(DIR_SALIDA / f'tokens_por_segundo_{tema}.svg', *colores)
