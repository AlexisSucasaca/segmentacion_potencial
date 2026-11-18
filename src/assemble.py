"""Une las estimaciones (con la decisión ya aplicada) y los datos del cliente en una sola tabla.

Reemplaza a `format_potential_df_2` del código anterior. La hoja resultante trae cada
columna UNA vez: antes salían `des_ciudad_x` / `des_ciudad_y`,
`des_segmento_estrategico_x` / `_y` y `X`, `Y` repetidas, porque el merge volvía a traer
columnas que la estimación ya tenía. En los Excel del 2026-08-31 esos pares eran idénticos
en todas las filas, así que quedarse con uno no pierde información.

Orden de columnas (el mismo del Excel actual, sin repetidas):
    estimaciones + decisión  ->  des_ciudad, X, Y  ->  features en el orden de `features`
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import pandas as pd


def assemble_output(
    estimates: pd.DataFrame,
    client_data: pd.DataFrame,
    features: Iterable[str],
    id_col: str = "Cod Cliente DEX",
    geo_cols: Sequence[str] = ("des_ciudad",),
) -> pd.DataFrame:
    """Devuelve `estimates` + las columnas de `client_data` que aún no tiene.

    estimates    salida del modelo con `apply_decision` ya aplicada (una fila por cliente).
    client_data  una fila por cliente con X, Y y las features (y opcionalmente `geo_cols`).
    features     features del modelo, en el orden en que deben salir.
    geo_cols     columnas geográficas opcionales: se agregan solo si `client_data` las trae.

    Falla si `client_data` repite un id (el merge duplicaría filas en silencio) o si le
    falta X, Y o alguna feature.
    """
    features = list(dict.fromkeys(features))
    client_data = client_data.loc[:, ~client_data.columns.duplicated()]

    if id_col not in client_data.columns or id_col not in estimates.columns:
        raise KeyError(f"Falta la columna id '{id_col}' en estimates o en client_data.")
    dup = client_data[id_col][client_data[id_col].duplicated()]
    if len(dup):
        raise ValueError(
            f"client_data tiene {dup.nunique()} '{id_col}' repetidos (p. ej. {dup.iloc[0]!r}); "
            "el merge duplicaría filas."
        )

    geo = [c for c in geo_cols if c in client_data.columns]
    deseadas = list(dict.fromkeys([*geo, "X", "Y", *features]))
    faltan = [c for c in deseadas if c not in client_data.columns and c not in estimates.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en client_data: {faltan}")

    nuevas = [c for c in deseadas if c not in estimates.columns and c != id_col]
    out = estimates.merge(client_data[[id_col, *nuevas]], on=id_col, how="left", validate="m:1")
    if len(out) != len(estimates):                      # red de seguridad; validate ya lo cubre
        raise RuntimeError("El ensamblado cambió el número de filas.")
    return out
