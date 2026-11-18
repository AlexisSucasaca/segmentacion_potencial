"""Reglas de decisión del potencial: de probabilidades por clase a `final_pred` y `flag_potential`.

Convención de clases
--------------------
`class_order` va SIEMPRE de la clase más alta a la más baja, por ejemplo
    ["1 Plata - Oro - Oro+", "2 Bronce"]              (2 clases)
    ["1 Oro+", "2 Oro", "3 Plata", "4 Bronce"]        (bodegas)
La columna de probabilidad de cada clase se deriva del nombre: `proba_<clase>`.
Ya no existe un diccionario PROBA_COLUMNS aparte que pueda quedar desalineado
con los nombres reales de las clases.

Reglas disponibles (se elige una por vertical en configs/verticales.yaml)
------------------------------------------------------------------------
umbral_2clases
    Solo 2 clases. Un cliente que NO está en la clase alta se promueve a ella si
    P(clase alta) >= medium (etiqueta "Medium") o >= high (etiqueta "High").
    Es la regla con que se desarrollaron lavanderías, hoteles, panadería y gastro:
    un Medium SÍ marca potencial.

acumulada
    Para 2 o más clases. Recorre las clases de la más alta a la más baja con la
    probabilidad acumulada. High promueve hasta la primera clase donde el acumulado
    llega a `high`. Medium promueve UNA clase menos que la primera donde llega a
    `medium`. Con 2 clases eso deja a un Medium en su misma clase (no marca potencial).
    Es la regla de `potential_cls.predict_segment`, con la que se desarrollaron
    bodegas y PM mercado.

Las dos reglas están validadas contra los Excel del 2026-08-31 (tests/test_baseline.py).
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd

# `prediction_conf` (confianza en la clase predicha) usa siempre estos umbrales, en todas
# las verticales: en los notebooks classify_confidence se llamaba sin pasar high_thresh.
CONF_MEDIUM = 0.5
CONF_HIGH = 0.75

ACTUAL_COL = "segmento_trx"
PREDICTED_COL = "predicted"


def proba_col(cls: str) -> str:
    return f"proba_{cls}"


# --------------------------------------------------------------------------------------
# Reglas: cada una devuelve (final_pred: ndarray de objetos, final_pred_conf: ndarray de objetos)
# --------------------------------------------------------------------------------------
def _regla_umbral_2clases(
    actual: pd.Series, proba: np.ndarray, class_order: Sequence[str], high: float, medium: float
):
    if len(class_order) != 2:
        raise ValueError(
            f"La regla 'umbral_2clases' requiere exactamente 2 clases, recibió {len(class_order)}."
        )
    top = class_order[0]
    p_top = proba[:, 0]
    not_top = (actual != top).to_numpy()

    is_high = not_top & (p_top >= high)          # NaN >= x es False, igual que en el notebook
    is_medium = not_top & (p_top >= medium) & ~is_high

    final = np.where(is_high | is_medium, top, actual.to_numpy(dtype=object))
    conf = np.full(len(actual), None, dtype=object)
    conf[is_medium] = "Medium"
    conf[is_high] = "High"
    return final, conf


def _regla_acumulada(
    actual: pd.Series, proba: np.ndarray, class_order: Sequence[str], high: float, medium: float
):
    k = len(class_order)
    # np.cumsum propaga NaN hacia adelante, igual que sum() de Python en predict_segment.
    cum = np.cumsum(proba, axis=1)

    def first_true(mask: np.ndarray) -> np.ndarray:
        return np.where(mask.any(axis=1), mask.argmax(axis=1), -1)

    i_high = first_true(cum >= high)
    i_med = first_true(cum >= medium)

    rank_map = {c: i for i, c in enumerate(class_order)}
    rank = actual.map(rank_map).to_numpy(dtype=float)
    if np.isnan(rank).any():
        bad = sorted(actual[actual.map(rank_map).isna()].unique().tolist())
        raise ValueError(f"'{ACTUAL_COL}' contiene clases fuera de class_order: {bad}")
    rank = rank.astype(int)

    promote_high = (i_high >= 0) & (i_high < rank)
    promote_med = ~promote_high & (i_med >= 0) & (i_med < rank)

    final_idx = np.where(
        promote_high, i_high, np.where(promote_med, np.minimum(i_med + 1, k - 1), rank)
    )
    order = np.array(list(class_order), dtype=object)
    final = order[final_idx]

    conf = np.full(len(actual), None, dtype=object)
    conf[promote_med] = "Medium"
    conf[promote_high] = "High"
    return final, conf


REGLAS: Dict[str, Callable] = {
    "umbral_2clases": _regla_umbral_2clases,
    "acumulada": _regla_acumulada,
}


# --------------------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------------------
def classify_confidence(proba: pd.Series) -> pd.Series:
    """Low (<0.5), Medium (<0.75) o High. Una probabilidad NaN queda como NaN.

    (En el notebook original un NaN caía en la rama `else` y salía 'High'; en los Excel
    de referencia no hay ningún NaN en las filas donde se aplica, así que el resultado
    es idéntico y esta versión no inventa una confianza que no existe.)
    """
    p = pd.to_numeric(proba, errors="coerce").to_numpy(dtype=float)
    out = np.where(p < CONF_MEDIUM, "Low", np.where(p < CONF_HIGH, "Medium", "High")).astype(object)
    out[np.isnan(p)] = None
    return pd.Series(out, index=proba.index, dtype=object)


def apply_decision(
    df: pd.DataFrame,
    regla: str,
    class_order: Sequence[str],
    high: float = 0.75,
    medium: float = 0.5,
    actual_col: str = ACTUAL_COL,
    predicted_col: str = PREDICTED_COL,
) -> pd.DataFrame:
    """Agrega proba_pred_segment, prediction_conf, final_pred, final_pred_conf y flag_potential.

    `df` debe traer `actual_col`, `predicted_col` y una columna `proba_<clase>` por cada
    clase de `class_order` (de la más alta a la más baja). No modifica `df`.
    """
    if regla not in REGLAS:
        raise ValueError(f"Regla desconocida '{regla}'. Disponibles: {sorted(REGLAS)}")
    if not medium <= high:
        raise ValueError(f"medium ({medium}) no puede ser mayor que high ({high}).")
    class_order = list(class_order)
    if len(set(class_order)) != len(class_order):
        raise ValueError(f"class_order tiene clases repetidas: {class_order}")

    p_cols = [proba_col(c) for c in class_order]
    missing = [c for c in [actual_col, predicted_col, *p_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas para aplicar la decisión: {missing}")

    out = df.copy()
    actual = out[actual_col]
    proba = out[p_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # proba_pred_segment = probabilidad de la clase predicha (proba_<predicted>).
    pred_idx = out[predicted_col].map({c: i for i, c in enumerate(class_order)})
    if pred_idx.isna().any():
        bad = sorted(out.loc[pred_idx.isna(), predicted_col].dropna().unique().tolist())
        raise ValueError(f"'{predicted_col}' contiene clases fuera de class_order: {bad}")
    out["proba_pred_segment"] = proba[np.arange(len(out)), pred_idx.to_numpy(dtype=int)]
    out["prediction_conf"] = classify_confidence(out["proba_pred_segment"])

    final, conf = REGLAS[regla](actual, proba, class_order, high, medium)
    out["final_pred"] = final
    out["final_pred_conf"] = conf
    out["flag_potential"] = out["final_pred"].to_numpy(dtype=object) != actual.to_numpy(dtype=object)
    return out
