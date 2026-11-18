"""PM a nivel mercado: asigna `mercado_id` a cada cliente y agrega sus features por mercado.

Reproduce, sin cambiar resultados, el bloque de `pm_level_mercado...` que va desde el
maestro de mercados hasta `ub_mercado` (antes de aplicar el modelo).

`mercado_id`
------------
Es el `CODIMOD.REF` del maestro de mercados (`Maestro_de_mercado.xlsx`), cruzado por el
código del cliente. Si un cliente no aparece en el maestro, se usa `cluster_mercado_id.csv`
como respaldo (un `mercado_id` numérico, de un clustering geográfico anterior). Validado
contra las 29,345 filas de la hoja "IDs mercado" del Excel de referencia: coincide en el
100%, incluidos los 7 clientes que solo el respaldo resuelve.

Agregación por mercado
-----------------------
Cada mercado agrega a sus clientes así:

    segmento_trx                  segmento numérico (Oro+=1 .. Bronce=4) promediado por
                                   (cod_cliente_alicorp_actual_mode, mercado_id) y redondeado
                                   al entero más cercano (np.rint: al par en los empates .5)
    X, Y                          mediana
    des_ciudad                    moda de la `des_ciudad` propia de los clientes del mercado
    cod_cliente_alicorp_actual_mode  moda del DEX de los clientes del mercado
    count_pm                      cantidad de clientes
    pct_<segmento_estrategico>, pct_tip_doc_<tipo>, pct_cuadrante_<cuadrante>, pct_true/pct_false
                                   % de clientes del mercado en cada categoría
    equifax (score_urbanicidad, hogares, puntos de interés)
                                   de la manzana de Equifax más cercana a (X, Y) del mercado,
                                   descartada si la distancia supera `equifax_radio_km`
                                   (0.2 km en los notebooks; ver el parámetro más abajo)
    el resto de las features numéricas de cliente (scores de 1 km, DistanciaTiendaMass, etc.)
                                   mediana

Un mercado sin ninguna manzana de Equifax a `equifax_radio_km` o menos queda fuera del
resultado (la unión con Equifax es "inner", como en el notebook).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.equifax import COLS_HOGARES, COLS_INTERES, RENOMBRE, _nearest_index

SEG_TRX_MAP = {"1 Oro+": 1, "2 Oro": 2, "3 Plata": 3, "4 Bronce": 4}
SEG_TRX_MAP_INV = {v: k for k, v in SEG_TRX_MAP.items()}

POIS = ["Mercado Abastos", "Mercado Mayorista", "Colegio", "Universidad", "Terminal Buses",
        "Hospital", "Hotel", "Restaurante", "Banco", "Residencial", "Grifo", "Centro Comercial",
        "Iglesia", "Supermercado", "Comida Rapida", "Comp_BO_Tambo", "Comp_BO_Mass",
        "Comp_BO_Listo", "Comp_BO_Oxxo"]
NUMERICAS_MEDIANA = ["pct_venta_diaadia_insuma", "pct_descuento", *POIS,
                     "score_comercial", "score_competencia", "DistanciaTiendaMass"]
EQUIFAX_RADIO_KM_DEFAULT = 0.2


SIMPLE_TRX_SEG_MAP = {"1 Oro+": "1 Plata - Oro - Oro+", "2 Oro": "1 Plata - Oro - Oro+",
                      "3 Plata": "1 Plata - Oro - Oro+", "4 Bronce": "2 Bronce"}


def map_segmento_pm(ub_mercado: pd.DataFrame) -> pd.DataFrame:
    """El `segmento_trx` de 4 clases de `ub_mercado` a las 2 que usa el modelo de PM.

    Paso aparte, después de `aggregate_mercado` y antes de `model.fit_potential`, igual que
    en el notebook (el segmento se agrega en 4 clases y recién ahí se simplifica a 2).
    """
    out = ub_mercado.copy()
    out["segmento_trx"] = out["segmento_trx"].map(SIMPLE_TRX_SEG_MAP)
    return out


def reference_rows(
    ub_mercado: pd.DataFrame, class_order, id_col: str, features,
    geo_cols=("X", "Y", "des_ciudad"),
) -> pd.DataFrame:
    """Los mercados que ya están en la clase alta (`class_order[0]`), sin pasar por el modelo.

    PM es la única vertical cuyo Excel de referencia las incluye (usa `format_potential_df`,
    la versión CON estas filas; las demás verticales, incluida bodegas, usan `format_potential_df_2`,
    sin ellas — confirmado en el código de los notebooks). Hay que concatenar el resultado de
    esta función con la salida de `decision.apply_decision` antes de `assemble.assemble_output`:
    predicted, proba_*, final_pred, etc. quedan NaN en estas filas, igual que en el Excel.
    """
    top = class_order[0]
    cols = [id_col, *geo_cols, "segmento_trx", *features]
    faltan = [c for c in cols if c not in ub_mercado.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en ub_mercado para las filas de referencia: {faltan}")
    return ub_mercado.loc[ub_mercado["segmento_trx"] == top, cols].reset_index(drop=True)


def get_mode(serie: pd.Series):
    """La moda de una serie (la primera si hay empate); None si está vacía."""
    m = serie.mode()
    return m.iloc[0] if not m.empty else None


def assign_mercado_id(
    df_clientes: pd.DataFrame, maestro: pd.DataFrame, cluster_fallback: pd.DataFrame,
    id_col: str = "cod_cliente_dex",
) -> pd.Series:
    """`mercado_id` por cliente: `CODIMOD.REF` del maestro, con respaldo en `cluster_fallback`.

    `maestro` trae `Codigo cliente` y `CODIMOD.REF` (la hoja de `Maestro_de_mercado.xlsx`).
    `cluster_fallback` trae `Cod Cliente DEX` y `mercado_id` (`cluster_mercado_id.csv`).
    Los ids de cliente se comparan como texto con 10 dígitos (con ceros a la izquierda).
    """
    faltan = [c for c in ("Codigo cliente", "CODIMOD.REF") if c not in maestro.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en maestro: {faltan}")
    faltan = [c for c in ("Cod Cliente DEX", "mercado_id") if c not in cluster_fallback.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en cluster_fallback: {faltan}")

    ids = df_clientes[id_col].astype(str).str.zfill(10)
    del_maestro = maestro.set_index(maestro["Codigo cliente"].astype(str).str.zfill(10))["CODIMOD.REF"]
    del_maestro = del_maestro[~del_maestro.index.duplicated()]
    del_cluster = cluster_fallback.set_index(cluster_fallback["Cod Cliente DEX"].astype(str).str.zfill(10))["mercado_id"]
    del_cluster = del_cluster[~del_cluster.index.duplicated()]

    de_maestro = ids.map(del_maestro)
    de_cluster = ids.map(del_cluster)
    de_cluster_txt = de_cluster.map(lambda v: str(int(v)) if pd.notna(v) else None)
    resultado = de_maestro.where(de_maestro.notna(), de_cluster_txt)
    return resultado.astype(object).where(resultado.notna(), None).rename("mercado_id")


def _pct_por_categoria(df: pd.DataFrame, col: str, mercado_col: str, prefijo: str) -> pd.DataFrame:
    """% de clientes de cada mercado en cada valor de `col`, como `<prefijo><valor en minúscula>`."""
    tabla = (
        df.groupby(mercado_col)[col].value_counts(normalize=True).unstack().fillna(0)
    )
    tabla.columns = [f"{prefijo}{str(c).lower()}" for c in tabla.columns]
    return tabla.reset_index()


def aggregate_mercado(
    df_clientes: pd.DataFrame, equifax_sum: pd.DataFrame,
    mercado_col: str = "mercado_id", id_col: str = "Cod Cliente DEX",
    equifax_radio_km: float = EQUIFAX_RADIO_KM_DEFAULT,
) -> pd.DataFrame:
    """Agrega las features de cliente (ya calculadas, con Equifax de cliente incluido) por mercado.

    `df_clientes` es el resultado de `features.agregar_features_cliente`, con `mercado_col`
    ya asignado (`assign_mercado_id`) y sin filas sin mercado. `equifax_sum` es
    `src.equifax.prepare_equifax_sum(equifax, urbanicidad)`.

    `des_ciudad` del mercado es la moda de la `des_ciudad` propia de sus clientes (la que
    trae BigQuery). El notebook original también leía "250423 - DePara PM y Zona ventas.xlsx"
    con la intención de usar esa ciudad en su lugar, pero el cruce que hacía (un `merge` sin
    `on=`, que junta por las columnas que ambos lados comparten) terminaba usando
    `Cod Cliente DEX` Y `des_ciudad` como llave a la vez, así que no reemplazaba nada: el
    resultado final es el mismo que si ese archivo no se hubiera usado. Por eso no aparece
    como parámetro de esta función.
    """
    faltan = [c for c in (mercado_col, id_col, "segmento_trx", "cod_cliente_alicorp_actual",
                          "X", "Y", "des_segmento_estrategico", "tip_documento", "cuadrante",
                          "flg_cliente_perfecto", "des_ciudad") if c not in df_clientes.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en df_clientes: {faltan}")
    if df_clientes[mercado_col].isna().any():
        raise ValueError(f"df_clientes tiene filas sin '{mercado_col}'; descártalas antes de agregar.")

    df = df_clientes.copy()
    df["seg_trx_nbr"] = df["segmento_trx"].map(SEG_TRX_MAP)
    if df["seg_trx_nbr"].isna().any():
        malos = sorted(df.loc[df["seg_trx_nbr"].isna(), "segmento_trx"].dropna().unique().tolist())
        raise ValueError(f"'segmento_trx' tiene valores fuera de {list(SEG_TRX_MAP)}: {malos}")

    mean_coords = df.groupby(mercado_col)[["X", "Y"]].median().reset_index()
    ciudades = df.groupby(mercado_col)["des_ciudad"].apply(get_mode).rename("des_ciudad").reset_index()
    mode_dex = df.groupby(mercado_col)["cod_cliente_alicorp_actual"].apply(get_mode)
    df = df.merge(mode_dex.rename("cod_cliente_alicorp_actual_mode"), on=mercado_col, how="left")

    segmento = (
        np.rint(df.groupby(["cod_cliente_alicorp_actual_mode", mercado_col])["seg_trx_nbr"].mean())
        .map(SEG_TRX_MAP_INV).rename("segmento_trx").reset_index()
        .merge(mean_coords, on=mercado_col, how="left")
    )

    pct_estrategico = _pct_por_categoria(df, "des_segmento_estrategico", mercado_col, "pct_")
    pct_tip_doc = _pct_por_categoria(df, "tip_documento", mercado_col, "pct_tip_doc_")
    pct_cuadrante = _pct_por_categoria(df, "cuadrante", mercado_col, "pct_cuadrante_")
    pct_perfecto = _pct_por_categoria(df, "flg_cliente_perfecto", mercado_col, "pct_")
    count_pm = df.groupby(mercado_col)[id_col].count().rename("count_pm").reset_index()
    medianas = df.groupby(mercado_col)[NUMERICAS_MEDIANA].median().reset_index()

    ub_mercado = (
        segmento
        .merge(pct_estrategico, on=mercado_col, how="left")
        .merge(pct_tip_doc, on=mercado_col, how="left")
        .merge(count_pm, on=mercado_col, how="left")
        .merge(ciudades, on=mercado_col, how="left")
        .merge(medianas, on=mercado_col, how="left")
        .merge(pct_cuadrante, on=mercado_col, how="left")
        .merge(pct_perfecto, on=mercado_col, how="left")
    )

    idx = _nearest_index(ub_mercado["X"], ub_mercado["Y"], equifax_sum["X"], equifax_sum["Y"])
    match = equifax_sum.iloc[idx].reset_index(drop=True)
    dist_km = _haversine_km(ub_mercado["Y"].to_numpy(), ub_mercado["X"].to_numpy(),
                            match["Y"].to_numpy(), match["X"].to_numpy())
    ub_mercado = ub_mercado[dist_km <= equifax_radio_km].reset_index(drop=True)
    match = match[dist_km <= equifax_radio_km].reset_index(drop=True)

    ub_mercado["score_urbanicidad"] = match["kpi_u"]
    ub_mercado["nro_viviendas"] = match["VIVIENDAS"]
    ub_mercado["POB"] = match["POB"]
    ub_mercado["total_ptos_de_interes"] = match[COLS_INTERES].sum(axis=1)
    total_hogares = match[COLS_HOGARES].sum(axis=1)
    for c in COLS_HOGARES:
        ub_mercado[f"pct_hogares_{c[-1].lower()}"] = match[c] / total_hogares
    for c, nuevo in RENOMBRE.items():
        if c in match.columns and nuevo not in ub_mercado.columns:
            ub_mercado[nuevo] = match[c]
    return ub_mercado


def _haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    lat1, lng1, lat2, lng2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))
