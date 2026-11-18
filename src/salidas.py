"""Encontrar el archivo más reciente de una carpeta que matchea un patrón.

Se usa para no tener que escribir a mano, en los notebooks, el nombre exacto (con fecha)
del último Excel que generó cada vertical — mismo criterio que ya usaban los tests
(`sorted(carpeta.glob(patron))[-1]`), formalizado acá para reusarlo en los notebooks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union


def ultimo_archivo(carpeta: Union[str, Path], patron: str) -> Path:
    """El archivo que matchea `patron` en `carpeta` con el nombre "mayor" alfabéticamente.

    Los archivos de este proyecto se nombran `<AAAAMMDD>_<algo>.xlsx`, así que ordenar por
    nombre (no por fecha de modificación del archivo, que puede mentir si se copió o
    descargó de nuevo) equivale a ordenar por fecha de la corrida, y el último de la lista
    es el más reciente.

    Lanza `FileNotFoundError` con un mensaje claro (carpeta y patrón) si no hay ningún
    archivo que matchee, en vez de fallar más adelante con un error críptico de pandas.
    """
    carpeta = Path(carpeta)
    coincidencias = sorted(carpeta.glob(patron))
    if not coincidencias:
        raise FileNotFoundError(f"No hay ningún archivo que matchee '{patron}' en {carpeta}")
    return coincidencias[-1]
