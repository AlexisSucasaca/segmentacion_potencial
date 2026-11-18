"""Cruce con Equifax: urbanicidad, % de hogares por nivel socioeconómico y puntos de interés.

Reproduce, vectorizado, el bloque de los notebooks que va desde `equifax.csv` hasta
`pct_hogares_*` (usa `utils.match_nearest_location` con sus valores por defecto).

Cómo funciona (igual que antes)
--------------------------------
1. `equifax.csv` trae una fila por manzana, con varias filas por `id_origen` (varias
   manzanas comparten id). Se suman las columnas relevantes por `(id_origen, X, Y)`.
2. Se cruza con `urbanicidad.csv` por `id_origen` (columna `kpi_u` -> `score_urbanicidad`).
3. A cada cliente se le asigna la manzana de Equifax MÁS CERCANA (vecino más cercano en X, Y
   en grados, sin proyectar) — no un cruce por id, sino geográfico.
4. `total_ptos_de_interes` = suma de Educacion + Ent_Finan + Gastronomi + Grifos +
   Hosteleria + Mercado + Retails + Salud de esa manzana.
5. `pct_hogares_a..e` = HOG_A..E de esa manzana, cada uno sobre la suma de los cinco.

Importante: el cruce usa las coordenadas CRUDAS del cliente (`X`, `Y` tal como llegan de
BigQuery), no las corregidas. Un cliente con coordenadas invertidas queda emparejado con la
manzana de Equifax más cercana a su punto invertido, que normalmente no tiene sentido. Es
el mismo comportamiento de los notebooks; ver `agregar_features_geo(..., coords=...)` en
`features.py` para la misma disyuntiva en `DistanciaTiendaMass`.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

COLS_HOGARES = ["HOG_A", "HOG_B", "HOG_C", "HOG_D", "HOG_E"]
COLS_INTERES = ["Educacion", "Ent_Finan", "Gastronomi", "Grifos", "Hosteleria", "Mercado", "Retails", "Salud"]
COLS_PSICOGRAFICAS = ["AUSTERO", "FORMALISTA", "MODERNAS", "PROGRESIST", "SOFISTICAD", "CONSERVADO"]
COLS_SUMA = ["VIVIENDAS", "POB", *COLS_HOGARES, "AGUDIA1", "ABAAGU3", "ABAAGU4", *COLS_PSICOGRAFICAS, "TIPVIV4", *COLS_INTERES]

RENOMBRE = {
    "kpi_u": "score_urbanicidad", "POB": "poblacion", "VIVIENDAS": "nro_viviendas",
    "HOG_A": "nro_hogares_a", "HOG_B": "nro_hogares_b", "HOG_C": "nro_hogares_c",
    "HOG_D": "nro_hogares_d", "HOG_E": "nro_hogares_e",
    "AUSTERO": "nro_austero", "FORMALISTA": "nro_formalistas", "MODERNAS": "nro_modernas",
    "PROGRESIST": "nro_progresistas", "SOFISTICAD": "nro_sofisticados", "CONSERVADO": "nro_conservador",
    "Hosteleria": "nro_hosteleria", "Gastronomi": "nro_gastronomia",
}


def prepare_equifax_sum(equifax: pd.DataFrame, urbanicidad: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (id_origen, X, Y) de Equifax, sumada, con `score_urbanicidad` cruzado.

    Filas exactamente duplicadas de `equifax` (mismos valores en todas las columnas) se
    cuentan una sola vez, igual que `equifax.drop_duplicates()` en los notebooks.
    """
    faltan = [c for c in ["id_origen", "X", "Y", *COLS_SUMA] if c not in equifax.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en equifax: {faltan}")
    if "id_origen" not in urbanicidad.columns or "kpi_u" not in urbanicidad.columns:
        raise KeyError("urbanicidad debe traer 'id_origen' y 'kpi_u'.")

    eq = equifax.drop_duplicates()
    resumen = eq.groupby(["id_origen", "X", "Y"], as_index=False)[COLS_SUMA].sum()
    resumen = resumen.merge(urbanicidad[["id_origen", "kpi_u"]], on="id_origen", how="left")
    return resumen


def _nearest_index(client_x, client_y, ref_x, ref_y) -> np.ndarray:
    """Índice, en `ref_x`/`ref_y`, del punto más cercano a cada cliente (distancia en grados)."""
    tree = cKDTree(np.column_stack([ref_y, ref_x]))          # (lat, lon), como match_nearest_location
    _, idx = tree.query(np.column_stack([client_y, client_x]), k=1)
    return idx


def agregar_equifax(
    df_clientes: pd.DataFrame, equifax_sum: pd.DataFrame,
    x_col: str = "X", y_col: str = "Y",
) -> pd.DataFrame:
    """Agrega score_urbanicidad, total_ptos_de_interes y pct_hogares_a..e por vecino más cercano.

    `equifax_sum` es el resultado de `prepare_equifax_sum`. `df_clientes` debe traer
    coordenadas válidas en `x_col`/`y_col` (sin NaN: filtra antes con `dropna`, como hacían
    los notebooks). No modifica `df_clientes`.
    """
    faltan = [c for c in (x_col, y_col) if c not in df_clientes.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en df_clientes: {faltan}")
    if df_clientes[[x_col, y_col]].isna().any().any():
        raise ValueError(f"df_clientes tiene NaN en '{x_col}'/'{y_col}'; filtra antes de cruzar con Equifax.")

    idx = _nearest_index(df_clientes[x_col], df_clientes[y_col], equifax_sum["X"], equifax_sum["Y"])
    match = equifax_sum.iloc[idx].reset_index(drop=True)
    match.index = df_clientes.index

    out = df_clientes.copy()
    out["score_urbanicidad"] = match["kpi_u"]
    out["total_ptos_de_interes"] = match[COLS_INTERES].sum(axis=1)
    total_hogares = match[COLS_HOGARES].sum(axis=1)
    for c in COLS_HOGARES:
        out[f"pct_hogares_{c[-1].lower()}"] = match[c] / total_hogares
    for c, nuevo in RENOMBRE.items():
        if c in match.columns and nuevo not in out.columns:
            out[nuevo] = match[c]
    return out
