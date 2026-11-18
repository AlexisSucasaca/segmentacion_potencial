"""Tests unitarios (sin Excel de referencia): documentan cómo se comporta cada regla."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from src.decision import apply_decision, classify_confidence
from src.export import compute_feature_mean, write_excel

ALTA, BAJA = "1 Plata - Oro - Oro+", "2 Bronce"
DOS = [ALTA, BAJA]
CUATRO = ["1 Oro+", "2 Oro", "3 Plata", "4 Bronce"]


def _dos_clases(p_alta, actual=BAJA):
    p = np.atleast_1d(p_alta).astype(float)
    return pd.DataFrame({
        "segmento_trx": actual, "predicted": np.where(p >= 0.5, ALTA, BAJA),
        f"proba_{ALTA}": p, f"proba_{BAJA}": 1 - p,
    })


def _fila(**p):
    """Un cliente Bronce de bodegas con probabilidades por clase."""
    d = {f"proba_{c}": p.get(c, 0.0) for c in CUATRO}
    pred = max(d, key=d.get).replace("proba_", "")
    return pd.DataFrame([{"segmento_trx": "4 Bronce", "predicted": pred, **d}])


# ---------------------------------------------------------------- umbral_2clases
@pytest.mark.parametrize("p, final, conf, flag", [
    (0.30, BAJA, None, False),
    (0.4999, BAJA, None, False),
    (0.50, ALTA, "Medium", True),      # el Medium SÍ marca potencial
    (0.65, ALTA, "Medium", True),
    (0.7499, ALTA, "Medium", True),
    (0.75, ALTA, "High", True),
    (0.95, ALTA, "High", True),
])
def test_umbral_2clases_bronce(p, final, conf, flag):
    r = apply_decision(_dos_clases(p), "umbral_2clases", DOS, high=0.75, medium=0.5).iloc[0]
    assert (r["final_pred"], r["final_pred_conf"], bool(r["flag_potential"])) == (final, conf, flag)


def test_umbral_2clases_clase_alta_no_cambia():
    r = apply_decision(_dos_clases(0.9, actual=ALTA), "umbral_2clases", DOS).iloc[0]
    assert r["final_pred"] == ALTA and r["final_pred_conf"] is None and not r["flag_potential"]


# ---------------------------------------------------------------- acumulada (PM: high=0.66)
@pytest.mark.parametrize("p, final, conf, flag", [
    (0.30, BAJA, None, False),
    (0.50, BAJA, "Medium", False),     # con 2 clases el Medium NO marca potencial
    (0.65, BAJA, "Medium", False),
    (0.66, ALTA, "High", True),
    (0.95, ALTA, "High", True),
])
def test_acumulada_2clases_pm(p, final, conf, flag):
    r = apply_decision(_dos_clases(p), "acumulada", DOS, high=0.66, medium=0.5).iloc[0]
    assert (r["final_pred"], r["final_pred_conf"], bool(r["flag_potential"])) == (final, conf, flag)


# ---------------------------------------------------------------- acumulada (bodegas, 4 clases)
@pytest.mark.parametrize("probas, final, conf, flag", [
    ({"3 Plata": 0.60, "4 Bronce": 0.40},                 "4 Bronce", "Medium", False),
    ({"2 Oro": 0.60, "3 Plata": 0.20, "4 Bronce": 0.20},  "3 Plata",  "High",   True),
    ({"1 Oro+": 0.55, "2 Oro": 0.10, "3 Plata": 0.15, "4 Bronce": 0.20}, "3 Plata", "High", True),
    ({"1 Oro+": 0.30, "2 Oro": 0.50, "3 Plata": 0.10, "4 Bronce": 0.10}, "2 Oro",   "High", True),
    ({"2 Oro": 0.30, "3 Plata": 0.40, "4 Bronce": 0.30},  "4 Bronce", "Medium", False),
    ({"2 Oro": 0.55, "3 Plata": 0.10, "4 Bronce": 0.35},  "3 Plata",  "Medium", True),  # 50% se cruza en Oro -> sube a Plata
])
def test_acumulada_bodegas(probas, final, conf, flag):
    r = apply_decision(_fila(**probas), "acumulada", CUATRO, high=0.75, medium=0.5).iloc[0]
    assert (r["final_pred"], r["final_pred_conf"], bool(r["flag_potential"])) == (final, conf, flag)


def test_acumulada_con_nan_en_clases_inferiores():
    """En bodegas cada modelo solo predice sus clases: un Plata no tiene proba_4 Bronce."""
    df = pd.DataFrame([{
        "segmento_trx": "3 Plata", "predicted": "2 Oro",
        "proba_1 Oro+": 0.2, "proba_2 Oro": 0.7, "proba_3 Plata": 0.1, "proba_4 Bronce": np.nan,
    }])
    r = apply_decision(df, "acumulada", CUATRO).iloc[0]
    assert r["final_pred"] == "2 Oro" and r["final_pred_conf"] == "High" and r["flag_potential"]


# ---------------------------------------------------------------- validaciones
def test_validaciones():
    with pytest.raises(ValueError, match="exactamente 2 clases"):
        apply_decision(_fila(**{"4 Bronce": 1.0}), "umbral_2clases", CUATRO)
    with pytest.raises(ValueError, match="Regla desconocida"):
        apply_decision(_dos_clases(0.6), "inventada", DOS)
    with pytest.raises(ValueError, match="no puede ser mayor"):
        apply_decision(_dos_clases(0.6), "acumulada", DOS, high=0.4, medium=0.6)
    with pytest.raises(KeyError, match="Faltan columnas"):
        apply_decision(_dos_clases(0.6).drop(columns=[f"proba_{BAJA}"]), "acumulada", DOS)
    malo = _dos_clases(0.6).assign(segmento_trx="9 Otro")
    with pytest.raises(ValueError, match="fuera de class_order"):
        apply_decision(malo, "acumulada", DOS)


def test_apply_decision_no_modifica_entrada():
    df = _dos_clases([0.2, 0.6, 0.9])
    antes = df.copy()
    apply_decision(df, "umbral_2clases", DOS)
    pd.testing.assert_frame_equal(df, antes)


def test_classify_confidence():
    s = classify_confidence(pd.Series([0.1, 0.5, 0.74, 0.75, 1.0, np.nan]))
    assert s.tolist() == ["Low", "Medium", "Medium", "High", "High", None]


# ---------------------------------------------------------------- Feature mean
def _df_fm():
    return pd.DataFrame({
        "segmento_trx": [BAJA] * 4,
        "flag_potential": [True, True, False, False],
        "shap_X": [0.1] * 4, "shap_ciudad": [0.2] * 4, "shap_score": [0.3] * 4,
        "X": [-76.0, -77.0, -75.0, -74.0], "ciudad": list("abcd"), "score": [1.0, 3.0, 5.0, 7.0],
    })


def test_feature_mean_features_columnas_y_valores():
    out = compute_feature_mean(_df_fm(), DOS, rmv_feats=["ciudad"], modo="features")
    assert list(out.columns) == ["feature", f"mean_{BAJA}", f"mean_{ALTA}", f"mean_potential_{BAJA}"]
    r = out.set_index("feature")
    assert list(r.index) == ["X", "score"]                      # 'ciudad' excluida
    assert r.loc["score", f"mean_{BAJA}"] == 4.0
    assert r.loc["score", f"mean_potential_{BAJA}"] == 2.0      # solo flag_potential
    assert r[f"mean_{ALTA}"].isna().all()                       # sin referencia queda vacía


def test_feature_mean_columna_duplicada_no_rompe():
    df = _df_fm()
    dup = pd.concat([df, df[["X"]]], axis=1)                    # X duplicada, como en los Excel actuales
    out = compute_feature_mean(dup, DOS, rmv_feats=["ciudad"], modo="features").set_index("feature")
    assert out.loc["X", f"mean_{BAJA}"] == -75.5                # numérico, no texto


def test_feature_mean_referencia_llena_la_clase_alta():
    ref = pd.DataFrame({"segmento_trx": [ALTA, ALTA], "X": [-70.0, -72.0], "score": [10.0, 20.0]})
    out = compute_feature_mean(_df_fm(), DOS, rmv_feats=["ciudad"], reference_df=ref).set_index("feature")
    assert out.loc["score", f"mean_{ALTA}"] == 15.0
    assert out.loc["score", f"mean_{BAJA}"] == 4.0              # la clase objetivo no cambia


def test_feature_mean_shap():
    out = compute_feature_mean(_df_fm(), DOS, modo="shap")
    assert list(out.columns) == [
        "feature", f"mean_{BAJA}", f"mean_{ALTA}", f"mean_pot_{BAJA}", f"mean_non_pot_{BAJA}"
    ]
    assert list(out["feature"]) == ["shap_X", "shap_score"]     # 'ciudad' no es numérica
    assert out.loc[0, f"mean_pot_{BAJA}"] == 0.1


# ---------------------------------------------------------------- Excel
def test_write_excel_hojas_formato_y_valores(tmp_path):
    df = pd.DataFrame({
        "id": [1, 2, 3], "proba_x": [0.1234, 0.5, np.nan], "texto": ["a", None, "c"],
        "flag": [True, False, True],
    })
    otra = pd.DataFrame({"a": [1, 2]})
    ruta = tmp_path / "salida.xlsx"
    write_excel(str(ruta), {"Principal": df, "Extra": otra, "Omitida": None}, hojas_con_formato=["Principal"])

    wb = load_workbook(ruta)
    assert wb.sheetnames == ["Principal", "Extra"]              # respeta el orden y omite None
    ws = wb["Principal"]
    assert ws.freeze_panes == "A2"
    assert ws["B2"].number_format == "0.00%" and ws["B2"].value == pytest.approx(0.1234)
    assert ws["A2"].number_format != "0.00%"
    assert ws["B4"].value is None and ws["C3"].value is None    # NaN y None quedan vacíos
    assert ws["D2"].value is True
    assert wb["Extra"].freeze_panes is None                     # hoja sin formato
    assert 15 <= ws.column_dimensions["A"].width <= 50
