"""Paso final de PM: cruza los mercados potenciales con ventas, familias top y segmentación,
y clasifica a cada cliente en No Potencial / Low / Medium / High.

Reproduce, sin cambiar resultados, `PDM_potencial_vs2.ipynb`. La única diferencia
deliberada: la segmentación estratégica (Abarrotero/Lavandero/Confitero→Impulso) no se
vuelve a pedir a BigQuery, se reusa la `des_segmento_estrategico` que ya trae el pipeline
de PM — la consulta de PM ya filtra exactamente esas tres categorías
(`data.QUERY_CONFIGS["pm_mercado"].filtro_segmento`), así que es la misma clasificación.

Flujo
-----
1. `map_segmento_pdm`: Confitero -> Impulso (Abarrotero y Lavandero quedan igual).
2. `familias_top_compradas`: cuántas familias "top 15" del cliente compró cada mes
   (cruza `pdm_2` con `top15` por `cod_cliente_alicorp_actual` + `des_familia`).
3. `armar_base_pdm`: junta `pdm_1` (ventas mensuales) con las familias top, el SKU por mes,
   el mercado y el segmento de cada cliente, y se queda solo con los mercados con
   `flag_potential = 1` en el Excel de PM.
4. `resumen_por_grupo` + `clasificar_potencial`: promedio por cliente, umbral por grupo
   (mercado para Abarrotero, distrito para Lavandero e Impulso) y la clasificación final.

`factor_umbral` (0.25 en el notebook original) es configurable — ver `configs/verticales.yaml`
→ `pm_mercado.pdm.factor_umbral`.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

SEGMENTO_MAP = {"Confitero": "Impulso", "Abarrotero": "Abarrotero", "Lavandero": "Lavandero"}
METRICAS = ["cnt_venta_real", "num_peso_real", "mnt_venta_real_bruto", "num_familias_top", "des_familia"]
# De las 5 métricas de `METRICAS`, solo estas 3 se usan para clasificar (igual que el notebook:
# num_peso_real y el umbral de SKU se calculan pero no entran en `suma_condiciones`).
METRICAS_CONDICION = {
    "mnt_venta_real_bruto": "cond_venta",
    "num_familias_top": "cond_familias",
    "cnt_venta_real": "cond_cnt",
}
FACTOR_UMBRAL_DEFAULT = 0.25
ETIQUETAS_POTENCIAL = {0: "No Potencial", 1: "Low", 2: "Medium", 3: "High"}


def map_segmento_pdm(des_segmento_estrategico: pd.Series) -> pd.Series:
    """Confitero -> Impulso; Abarrotero y Lavandero quedan igual."""
    return des_segmento_estrategico.map(SEGMENTO_MAP)


def familias_top_compradas(pdm_2: pd.DataFrame, top15: pd.DataFrame) -> pd.DataFrame:
    """Cuántas familias "top 15" del cliente-padre compró cada cliente-hijo, por periodo.

    `pdm_2`: compras por cliente (`cod_cliente_alicorp_actual`, `cod_cliente_dex`, `periodo`,
    `des_familia`). `top15`: `Cod DEX Hijo` (= `cod_cliente_alicorp_actual`), `des_familia`.
    """
    faltan = [c for c in ("cod_cliente_alicorp_actual", "cod_cliente_dex", "periodo", "des_familia") if c not in pdm_2.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en pdm_2: {faltan}")
    if not {"Cod DEX Hijo", "des_familia"} <= set(top15.columns):
        raise KeyError("top15 debe traer 'Cod DEX Hijo' y 'des_familia'.")

    merged = pdm_2.merge(
        top15[["Cod DEX Hijo", "des_familia"]],
        left_on=["cod_cliente_alicorp_actual", "des_familia"], right_on=["Cod DEX Hijo", "des_familia"],
        how="inner",
    )
    fam_top = merged.groupby(["periodo", "cod_cliente_dex"]).size().reset_index(name="num_familias_top")
    fam_top = fam_top[fam_top["cod_cliente_dex"] != "Sin Asignar"]
    fam_top = fam_top[pd.to_numeric(fam_top["cod_cliente_dex"], errors="coerce").notna()]
    fam_top["cod_cliente_dex"] = fam_top["cod_cliente_dex"].astype(int)
    return fam_top


def sku_por_mes(pdm_2: pd.DataFrame) -> pd.DataFrame:
    """Cantidad de familias (SKU) distintas compradas por cliente y periodo.

    La columna de salida se llama `des_familia`, igual que en el notebook original
    (`.agg("count")` no renombra la columna que cuenta) — pese al nombre, es un conteo
    numérico, y así se usa después como una de las `METRICAS`.
    """
    resumen = pdm_2.groupby(["periodo", "cod_cliente_dex"])[["des_familia"]].agg("count").reset_index()
    resumen = resumen[pd.to_numeric(resumen["cod_cliente_dex"], errors="coerce").notna()]
    resumen["cod_cliente_dex"] = resumen["cod_cliente_dex"].astype(int)
    return resumen


def armar_base_pdm(
    pdm_1: pd.DataFrame, pdm_2: pd.DataFrame, top15: pd.DataFrame,
    ids_mercado: pd.DataFrame, segmento_por_cliente: pd.DataFrame,
    potencial_pm: pd.DataFrame,
) -> pd.DataFrame:
    """Arma la base de `pdm_1` enriquecida, filtrada a los mercados con `flag_potential = 1`.

    `ids_mercado`: hoja "IDs mercado" del Excel de PM (`Cod Cliente DEX`, `mercado_id`), sin NaN.
    `segmento_por_cliente`: `Cod Cliente DEX` + `segmento` (salida de `map_segmento_pdm`
    aplicada a la `des_segmento_estrategico` de los clientes de PM).
    `potencial_pm`: hoja "Potencial" del Excel de PM, con `mercado_id` y `flag_potential`
    (ya filtrada a `flag_potential == 1`, como hace el notebook antes de este paso).
    """
    dex = ids_mercado.dropna().merge(segmento_por_cliente, on="Cod Cliente DEX", how="inner")

    fam_top = familias_top_compradas(pdm_2, top15)
    base = pdm_1.merge(fam_top, on=["periodo", "cod_cliente_dex"], how="left")
    base = base.merge(dex, left_on="cod_cliente_dex", right_on="Cod Cliente DEX", how="left")
    base = base.merge(sku_por_mes(pdm_2), on=["cod_cliente_dex", "periodo"], how="left")

    base = base.merge(potencial_pm, on="mercado_id", how="left")
    base = base[base["flag_potential"] == 1.0]
    base["num_familias_top"] = base["num_familias_top"].fillna(0)
    return base


def resumen_cliente(df: pd.DataFrame, id_cols: Sequence[str], metricas: Sequence[str] = METRICAS) -> pd.DataFrame:
    """Promedio de `metricas` por cliente (+ mercado o distrito, según `id_cols`)."""
    return df.groupby(list(id_cols))[list(metricas)].mean().reset_index()


def resumen_grupo(
    df: pd.DataFrame, group_col: str, metricas: Sequence[str] = METRICAS,
    factor_umbral: float = FACTOR_UMBRAL_DEFAULT,
) -> pd.DataFrame:
    """Media, desviación y umbral (`media + factor_umbral * desviación`) por grupo."""
    resumen = df.groupby(group_col)[list(metricas)].agg(["mean", "std"]).reset_index()
    resumen.columns = [group_col] + [f"{col}_{stat}" for col, stat in resumen.columns[1:]]
    for col in metricas:
        resumen[f"Threshold_{col}"] = resumen[f"{col}_mean"] + resumen[f"{col}_std"] * factor_umbral
    return resumen


def clasificar_potencial(cliente_summary: pd.DataFrame, group_summary: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Agrega `Potencial` (No Potencial/Low/Medium/High) según cuántas métricas están bajo su umbral."""
    out = cliente_summary.merge(group_summary, on=group_col, how="left")
    cond_cols = []
    for metrica, nombre_cond in METRICAS_CONDICION.items():
        out[nombre_cond] = (out[metrica] < out[f"Threshold_{metrica}"]).astype(int)
        cond_cols.append(nombre_cond)
    out["suma_condiciones"] = out[cond_cols].sum(axis=1)
    out["Potencial"] = out["suma_condiciones"].map(ETIQUETAS_POTENCIAL)
    return out.drop(columns=[*cond_cols, "suma_condiciones"])


def construir_pdm_final(
    pdm_1: pd.DataFrame, pdm_2: pd.DataFrame, top15: pd.DataFrame,
    ids_mercado: pd.DataFrame, segmento_por_cliente: pd.DataFrame, potencial_pm: pd.DataFrame,
    factor_umbral: float = FACTOR_UMBRAL_DEFAULT,
) -> Dict[str, pd.DataFrame]:
    """Punto de entrada: arma las tres hojas finales (Abarroteros, Lavanderos, Impulso)."""
    base = armar_base_pdm(pdm_1, pdm_2, top15, ids_mercado, segmento_por_cliente, potencial_pm)

    # El notebook original calcula `distrito_summary` solo con los clientes Lavandero y lo
    # reusa tal cual para Impulso (no calcula un umbral de distrito propio para Impulso).
    # Se preserva ese comportamiento, aunque parezca un descuido del original.
    distrito_summary = resumen_grupo(base[base["segmento"] == "Lavandero"], "des_distrito", factor_umbral=factor_umbral)

    salidas = {}
    especificaciones = {
        "Abarroteros": ("Abarrotero", ["cod_cliente_dex", "mercado_id"], "mercado_id", None),
        "Lavanderos": ("Lavandero", ["cod_cliente_dex", "des_distrito"], "des_distrito", distrito_summary),
        "Impulso": ("Impulso", ["cod_cliente_dex", "des_distrito"], "des_distrito", distrito_summary),
    }
    for hoja, (segmento, id_cols, group_col, grupo_fijo) in especificaciones.items():
        subset = base[base["segmento"] == segmento]
        cliente = resumen_cliente(subset, id_cols)
        grupo = grupo_fijo if grupo_fijo is not None else resumen_grupo(subset, group_col, factor_umbral=factor_umbral)
        salidas[hoja] = clasificar_potencial(cliente, grupo, group_col)
    return salidas
