"""Hoja "Feature mean" y escritura del Excel de resultados.

Feature mean tiene dos modos, uno por vertical (configs/verticales.yaml):

modo="features"  (lavanderías, hoteles, panadería, gastro, bodegas)
    Una fila por feature con la media del VALOR de la feature:
        mean_<clase>            todas las filas de la clase
        mean_potential_<clase>  filas de la clase con flag_potential (todas menos la más alta)

modo="shap"      (PM mercado)
    Una fila por `shap_<feature>` numérica con la media del SHAP:
        mean_<clase>, mean_pot_<clase> (flag=1), mean_non_pot_<clase> (flag=0)
    Es lo que hacía el notebook de PM. La columna de la clase alta sale vacía por
    construcción: las filas de referencia de esa clase no tienen SHAP.

Las columnas salen de la clase más baja a la más alta, como en los Excel actuales,
aunque `class_order` se declare de la más alta a la más baja.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

ACTUAL_COL = "segmento_trx"
FLAG_COL = "flag_potential"
SHAP_PREFIX = "shap_"


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Si una columna aparece dos veces (p. ej. X, Y) se queda la primera."""
    return df.loc[:, ~df.columns.duplicated()]


def _as_float(s: pd.Series) -> pd.Series:
    """Convierte a float lo que sea numérico o booleano; lo demás queda NaN."""
    return pd.to_numeric(s.astype(object), errors="coerce").astype(float)


def _feature_names_from_shap(df_or_cols, rmv_feats: Iterable[str]) -> List[str]:
    cols = df_or_cols.columns if hasattr(df_or_cols, "columns") else df_or_cols
    rmv = set(rmv_feats)
    names = [str(c)[len(SHAP_PREFIX):] for c in cols if str(c).startswith(SHAP_PREFIX)]
    return [n for n in dict.fromkeys(names) if n not in rmv]


def compute_feature_mean(
    df: pd.DataFrame,
    class_order: Sequence[str],
    rmv_feats: Iterable[str] = (),
    modo: str = "features",
    features: Optional[Sequence[str]] = None,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Construye la hoja Feature mean.

    class_order   de la clase más alta a la más baja.
    rmv_feats     features a excluir (categóricas, geográficas).
    features      (modo features) lista explícita; por defecto se deduce de las columnas
                  `shap_*` de `df`, igual que los notebooks.
    reference_df  (modo features) filas de referencia con las features, para llenar la
                  media de las clases que no tienen filas en `df` (en las verticales de
                  2 clases `df` solo trae la clase objetivo, así que sin esto la media
                  de la clase alta sale vacía). Solo se usa para clases ausentes en `df`.
    """
    if modo not in ("features", "shap"):
        raise ValueError(f"modo debe ser 'features' o 'shap', recibió '{modo}'")
    df = _drop_duplicate_columns(df)
    sorted_classes = list(class_order)[::-1]            # de la más baja a la más alta
    potential_classes = sorted_classes[:-1]
    seg = df[ACTUAL_COL]
    # flag_potential puede venir como bool, 0/1/NaN (PM, con filas de referencia) u objeto.
    flag = _as_float(df[FLAG_COL]) if FLAG_COL in df.columns else pd.Series(0.0, index=df.index)
    is_pot = flag.eq(1).to_numpy()
    non_pot = flag.eq(0).to_numpy()

    if modo == "features":
        feats = list(features) if features is not None else _feature_names_from_shap(df, rmv_feats)
        feats = [f for f in dict.fromkeys(feats) if f in df.columns]
        data = pd.DataFrame({f: _as_float(df[f]) for f in feats}, index=df.index)
        ref_data = None
        if reference_df is not None:
            reference_df = _drop_duplicate_columns(reference_df)
            ref_data = pd.DataFrame(
                {f: _as_float(reference_df[f]) for f in feats if f in reference_df.columns}
            )
            ref_seg = reference_df[ACTUAL_COL].reset_index(drop=True)
        rows = {}
        for cls in sorted_classes:
            mask = (seg == cls).to_numpy()
            if mask.any():
                rows[f"mean_{cls}"] = data[mask].mean()
            elif ref_data is not None and (ref_seg == cls).any():
                rows[f"mean_{cls}"] = ref_data[(ref_seg == cls).to_numpy()].mean().reindex(feats)
            else:
                rows[f"mean_{cls}"] = pd.Series(np.nan, index=feats)
        for cls in potential_classes:
            mask = ((seg == cls).to_numpy()) & is_pot
            rows[f"mean_potential_{cls}"] = (
                data[mask].mean() if mask.any() else pd.Series(np.nan, index=feats)
            )
        out = pd.DataFrame(rows).reindex(feats)
        # Igual que en bodegas: se omite una feature sin ningún dato (categóricas, texto).
        mean_cols = [f"mean_{c}" for c in sorted_classes]
        out = out[out[mean_cols].notna().any(axis=1)]
        out = out.round(2).rename_axis("feature").reset_index()
        return out

    # ---- modo shap ---------------------------------------------------------------
    numeric = {
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    }
    shap_cols = [
        c for c in df.columns
        if str(c).startswith(SHAP_PREFIX) and str(c)[len(SHAP_PREFIX):] in numeric
    ]
    shap_cols = [c for c in shap_cols if c not in set(rmv_feats)]
    data = df[shap_cols].astype(float)
    rows = {}
    for cls in sorted_classes:
        mask = (seg == cls).to_numpy()
        rows[f"mean_{cls}"] = data[mask].mean() if mask.any() else pd.Series(np.nan, index=shap_cols)
    for cls in potential_classes:
        m = (seg == cls).to_numpy() & is_pot
        rows[f"mean_pot_{cls}"] = data[m].mean() if m.any() else pd.Series(np.nan, index=shap_cols)
    for cls in potential_classes:
        m = (seg == cls).to_numpy() & non_pot
        rows[f"mean_non_pot_{cls}"] = data[m].mean() if m.any() else pd.Series(np.nan, index=shap_cols)
    out = pd.DataFrame(rows).reindex(shap_cols).round(2)
    return out.rename_axis("feature").reset_index()


# --------------------------------------------------------------------------------------
# Escritura de Excel
# --------------------------------------------------------------------------------------
def _col_width(df: pd.DataFrame, i: int, sample: int = 100) -> int:
    header = str(df.columns[i])
    width = max(15, len(header) + 2)
    col = df.iloc[:sample, i]
    if len(col):
        width = max(width, int(col.astype(str).str.len().max()) + 1)
    return min(width, 50)


def write_excel(
    path: str,
    hojas: Dict[str, pd.DataFrame],
    hojas_con_formato: Iterable[str] = (),
) -> None:
    """Escribe un Excel con una hoja por entrada de `hojas` (en ese orden).

    En las hojas con formato: columnas `proba*` en 0.00 %, ancho automático
    (mín. 15, máx. 50), primera fila congelada. Usa xlsxwriter (el motor que ya usaba el
    notebook de PM) con el mismo contenido y nombres de hoja que el Excel anterior.
    """
    formato = set(hojas_con_formato)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        book = writer.book
        pct = book.add_format({"num_format": "0.00%"})
        for nombre, df in hojas.items():
            if df is None:
                continue
            df.to_excel(writer, sheet_name=nombre, index=False)
            if nombre not in formato:
                continue
            ws = writer.sheets[nombre]
            for i, col in enumerate(df.columns):
                ws.set_column(i, i, _col_width(df, i), pct if "proba" in str(col) else None)
            ws.freeze_panes(1, 0)
