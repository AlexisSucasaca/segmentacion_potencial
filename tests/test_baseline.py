"""Valida el post-proceso nuevo contra los Excel de referencia (línea base del 2026-08-31).

Qué debe coincidir EXACTO con el Excel actual (la regla de cada vertical no cambia):
    final_pred, final_pred_conf, flag_potential           -> las 6 verticales
    proba_pred_segment, prediction_conf                   -> bodegas y PM
    Feature mean                                          -> PM (SHAP) y, en las demás, cada
                                                             celda que el Excel actual sí calculó

Diferencias esperadas (bugs del código actual, corregidos aquí):
    * proba_pred_segment / prediction_conf en lavanderías, hoteles, panadería y gastro: el
      Excel actual trae un valor SHAP en las filas predichas como Bronce. Aquí se comparan
      solo las filas predichas como clase alta (correctas en ambos) y, para todas, que el
      valor nuevo sea la probabilidad de la clase predicha.
    * Feature mean: las filas X e Y del Excel actual traen texto (columnas X/Y duplicadas)
      o faltan (bodegas); aquí son numéricas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.decision import apply_decision, classify_confidence, proba_col
from src.export import compute_feature_mean

NOMBRES = list(load_config())

# Verticales cuyo Excel actual trae proba_pred_segment errónea en las filas predichas como
# clase baja (PROBA_COLUMNS con la clave '4 Bronce' cuando la clase se llama '2 Bronce').
BUG_PROBA_PRED = {"lavanderias", "hoteles", "panaderia", "gastro"}


def _estimates(baseline) -> pd.DataFrame:
    """Filas con predicción (en PM la hoja además trae filas de referencia sin predicción)."""
    main = baseline["main"]
    return main[main["predicted"].notna()].reset_index(drop=True)


def _decide(baseline, est: pd.DataFrame) -> pd.DataFrame:
    cfg = baseline["cfg"]
    d = cfg["decision"]
    return apply_decision(est, d["regla"], cfg["class_order"], d["high"], d["medium"])


def _none_if_nan(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), None)


def _describe_mismatch(est, res, mask, cols) -> str:
    bad = np.flatnonzero(mask)[:5]
    return "\n" + est.iloc[bad][cols].join(res.iloc[bad][["final_pred", "final_pred_conf", "flag_potential"]],
                                           rsuffix="_nuevo").to_string()


@pytest.mark.parametrize("baseline", NOMBRES, indirect=True)
def test_decision_reproduce_baseline(baseline):
    est = _estimates(baseline)
    res = _decide(baseline, est)
    cols = ["segmento_trx", "predicted", "final_pred", "final_pred_conf", "flag_potential"]

    m_pred = (res["final_pred"] != est["final_pred"]).to_numpy()
    assert not m_pred.any(), f"final_pred distinto en {m_pred.sum()} de {len(est)} filas" + \
        _describe_mismatch(est, res, m_pred, cols)

    m_conf = (_none_if_nan(res["final_pred_conf"]).fillna("∅") != _none_if_nan(est["final_pred_conf"]).fillna("∅")).to_numpy()
    assert not m_conf.any(), f"final_pred_conf distinto en {m_conf.sum()} de {len(est)} filas" + \
        _describe_mismatch(est, res, m_conf, cols)

    flag_ref = est["flag_potential"].astype(bool)
    m_flag = (res["flag_potential"].astype(bool) != flag_ref).to_numpy()
    assert not m_flag.any(), f"flag_potential distinto en {m_flag.sum()} de {len(est)} filas" + \
        _describe_mismatch(est, res, m_flag, cols)
    assert int(res["flag_potential"].sum()) == int(flag_ref.sum())


@pytest.mark.parametrize("baseline", NOMBRES, indirect=True)
def test_proba_pred_segment_y_prediction_conf(baseline):
    cfg = baseline["cfg"]
    est = _estimates(baseline)
    res = _decide(baseline, est)
    top = cfg["class_order"][0]

    # 1) El valor nuevo es, por construcción independiente, la probabilidad de la clase predicha.
    esperado = np.full(len(est), np.nan)
    for cls in cfg["class_order"]:
        m = (est["predicted"] == cls).to_numpy()
        esperado[m] = pd.to_numeric(est.loc[m, proba_col(cls)], errors="coerce").to_numpy(dtype=float)
    np.testing.assert_allclose(res["proba_pred_segment"].to_numpy(dtype=float), esperado, atol=1e-12)
    assert res["proba_pred_segment"].between(0, 1).all()

    # 2) Contra el Excel actual: todas las filas, salvo las del bug conocido.
    if cfg["nombre"] in BUG_PROBA_PRED:
        comparables = (est["predicted"] == top).to_numpy()
        assert comparables.any()
    else:
        comparables = np.ones(len(est), dtype=bool)
    ref_p = pd.to_numeric(est["proba_pred_segment"], errors="coerce").to_numpy(dtype=float)
    np.testing.assert_allclose(
        res["proba_pred_segment"].to_numpy(dtype=float)[comparables], ref_p[comparables], atol=1e-12
    )
    a = res["prediction_conf"].astype(object).to_numpy()[comparables]
    b = _none_if_nan(est["prediction_conf"]).to_numpy()[comparables]
    assert (a == b).all(), f"prediction_conf distinto en {(a != b).sum()} filas comparables"

    # 3) prediction_conf siempre consistente con la probabilidad de la clase predicha.
    conf_esperada = classify_confidence(pd.Series(esperado))
    assert (res["prediction_conf"].astype(object).to_numpy() == conf_esperada.astype(object).to_numpy()).all()


@pytest.mark.parametrize("baseline", NOMBRES, indirect=True)
def test_feature_mean_reproduce_baseline(baseline):
    cfg, main, base = baseline["cfg"], baseline["main"], baseline["fm"]
    fmc = cfg["feature_mean"]
    kwargs = {}
    if fmc["modo"] == "features":
        kwargs["features"] = [f for f in baseline["features"] if f not in set(fmc["rmv_feats"])]
    ours = compute_feature_mean(main, cfg["class_order"], fmc["rmv_feats"], fmc["modo"], **kwargs)

    assert list(ours.columns) == list(base.columns), "las columnas de Feature mean cambiaron"
    val_cols = [c for c in base.columns if c != "feature"]
    base_num = base.copy()
    for c in val_cols:
        base_num[c] = pd.to_numeric(base_num[c], errors="coerce")   # texto (X, Y duplicadas) -> NaN

    if fmc["modo"] == "shap":
        # PM: igualdad total, incluidas las celdas vacías de la clase alta.
        assert list(ours["feature"]) == list(base_num["feature"])
        np.testing.assert_allclose(
            ours[val_cols].to_numpy(dtype=float), base_num[val_cols].to_numpy(dtype=float),
            atol=1e-9, equal_nan=True,
        )
        return

    # Resto: cada feature que el Excel actual calculó bien debe coincidir; las X e Y del Excel
    # actual traen texto (o faltan) y aquí son numéricas.
    faltan = set(base["feature"]) - set(ours["feature"])
    assert not faltan, f"features que el Excel actual tiene y aquí no: {sorted(faltan)}"
    nuevas = set(ours["feature"]) - set(base["feature"])
    assert nuevas <= {"X", "Y"}, f"features nuevas inesperadas: {sorted(nuevas - {'X', 'Y'})}"

    ours_i = ours.set_index("feature")
    base_i = base_num.set_index("feature")
    for feat in base_i.index:
        fila_base = base_i.loc[feat, val_cols].to_numpy(dtype=float)
        if np.isnan(fila_base).all() and feat in {"X", "Y"}:
            continue                                        # texto en el Excel actual
        np.testing.assert_allclose(
            ours_i.loc[feat, val_cols].to_numpy(dtype=float), fila_base, atol=1e-9, equal_nan=True,
            err_msg=f"Feature mean distinto en '{feat}'",
        )
    for feat in nuevas:
        assert ours_i.loc[feat].notna().any(), f"'{feat}' debería ser numérica"
