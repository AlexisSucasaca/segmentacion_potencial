"""Tests de src/model.py (datos sintéticos, no necesitan ningún archivo tuyo).

El test principal compara contra `tests/legacy/potential_cls.py` (el código original): con los
mismos datos, semilla y parámetros, las probabilidades, los SHAP y las métricas de validación
deben coincidir. Se saltan si no están instalados lightgbm, shap y optuna.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("shap")
pytest.importorskip("optuna")

from sklearn.model_selection import train_test_split  # noqa: E402

from src import model as M  # noqa: E402
from src.assemble import assemble_output  # noqa: E402
from src.decision import apply_decision  # noqa: E402
from tests.legacy import potential_cls as legacy  # noqa: E402

NT = 60                                                    # árboles: suficiente para los tests
DOS = ["1 Plata - Oro - Oro+", "2 Bronce"]
CUATRO = ["1 Oro+", "2 Oro", "3 Plata", "4 Bronce"]
FEATS = ["x1", "x2", "x3", "ciudad", "seg"]
CATS = ["ciudad", "seg"]
PAR = {"lambda_l2": 1.0, "num_leaves": 8, "feature_fraction": 0.9, "bagging_fraction": 0.9,
       "bagging_freq": 3, "min_child_samples": 5, "n_jobs": 1}   # n_jobs=1: resultados reproducibles


def sintetico(n=700, orden=DOS, cortes=(0.25,), seed=0) -> pd.DataFrame:
    """Clientes con 3 features numéricas, 2 categóricas y una clase que depende de las numéricas."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": np.arange(n) + 1000,
        "x1": rng.normal(size=n), "x2": rng.normal(size=n),
        "x3": rng.integers(0, 10, n).astype(float),
        "ciudad": rng.choice(["Lima", "Cusco", "Piura", "Tacna"], n, p=[0.5, 0.2, 0.2, 0.1]),
        "seg": rng.choice(["A", "B", "C"], n),
    })
    score = df.x1 + 0.6 * df.x2 + 0.1 * df.x3 + rng.normal(scale=0.7, size=n)
    limites = np.quantile(score, 1 - np.cumsum(cortes))     # cortes de arriba hacia abajo
    clase = np.full(n, len(orden) - 1)
    for k, lim in reversed(list(enumerate(limites))):
        clase[score >= lim] = k
    df["segmento_trx"] = [orden[k] for k in clase]
    return df


def _ajustar(df, orden, **kw):
    clases_modelo = orden[::-1][:-1]
    return M.fit_potential(
        df, FEATS, CATS, orden, {c: PAR for c in clases_modelo},
        id_col="id", cluster_col="seg", n_estimators=NT, **kw,
    )


@pytest.fixture(autouse=True)
def _legacy_rapido(monkeypatch):
    monkeypatch.setattr(legacy, "N_ESTIMATORS", NT)


# ---------------------------------------------------------------- equivalencia con el original
def _legacy_estimates(df, orden):
    low_to_high = orden[::-1]
    partes, validacion = [], {}
    for i in range(len(low_to_high) - 1):
        cur, higher = low_to_high[i], low_to_high[i:]
        dff = df[df.segmento_trx.isin(higher)].copy()
        X = dff[FEATS].copy()
        y = dff.segmento_trx
        for c in CATS:
            X[c] = X[c].astype("category")
        Xt, Xv, yt, yv = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        params = {"objective": "multiclass", "num_class": len(higher), "metric": "multi_logloss",
                  "boosting_type": "gbdt", **PAR}
        validacion[cur] = legacy.validate_performance(Xt, yt, Xv, yv, CATS, params, cur)
        partes.append(legacy.run_potential_estimations(
            dff, target_class=cur, categorical_feats=CATS, features=FEATS,
            target_col="segmento_trx", cluster_col="seg", client_id_col="id",
            lgbm_params=params, add_shap=True))
    return pd.concat(partes, ignore_index=True), validacion


@pytest.mark.parametrize("orden, cortes", [(DOS, (0.25,)), (CUATRO, (0.1, 0.15, 0.25))])
def test_equivale_al_codigo_original(orden, cortes):
    df = sintetico(n=900, orden=orden, cortes=cortes)
    res = _ajustar(df, orden)
    ref, ref_val = _legacy_estimates(df, orden)

    assert list(res.estimates.columns) == list(ref.columns)
    pd.testing.assert_frame_equal(res.estimates, ref, check_exact=False, atol=1e-9, rtol=1e-9)

    for cls, v in ref_val.items():
        for nombre, met in v["benchmarks"].items():
            for k, val in met.items():
                assert res.validation[cls]["benchmarks"][nombre][k] == pytest.approx(val, abs=1e-9), \
                    f"{cls}/{nombre}/{k}"
        pd.testing.assert_frame_equal(res.validation[cls]["feature_importance"], v["feature_importance"])


# ---------------------------------------------------------------- forma del resultado
def test_estimaciones_forma_y_consistencia():
    df = sintetico()
    res = _ajustar(df, DOS)
    est = res.estimates
    objetivo = df[df.segmento_trx == "2 Bronce"]

    assert sorted(est["id"]) == sorted(objetivo["id"])            # cada cliente objetivo, una vez
    assert set(est["segmento_trx"]) == {"2 Bronce"}
    assert list(est.columns[:7]) == [
        "id", "segmento_trx", "seg", "predicted", "proba_1 Plata - Oro - Oro+", "proba_2 Bronce", "proba_potential",
    ]
    np.testing.assert_allclose(est["proba_1 Plata - Oro - Oro+"] + est["proba_2 Bronce"], 1.0)
    np.testing.assert_allclose(est["proba_potential"], est["proba_1 Plata - Oro - Oro+"])
    assert [c for c in est.columns if c.startswith("shap_")] == [f"shap_{f}" for f in FEATS]
    assert est.filter(like="shap_").notna().all().all()


def test_cuatro_clases_un_modelo_por_clase_menos_la_mas_alta():
    orden, cortes = CUATRO, (0.1, 0.15, 0.25)
    df = sintetico(n=1200, orden=orden, cortes=cortes)
    res = _ajustar(df, orden)
    assert list(res.params) == ["4 Bronce", "3 Plata", "2 Oro"]
    assert set(res.estimates["segmento_trx"]) == {"4 Bronce", "3 Plata", "2 Oro"}   # Oro+ no se estima
    assert len(res.estimates) == (df.segmento_trx != "1 Oro+").sum()
    assert res.params["4 Bronce"]["num_class"] == 4 and res.params["2 Oro"]["num_class"] == 2


def test_sin_shap():
    res = _ajustar(sintetico(), DOS, add_shap=False)
    assert not any(c.startswith("shap_") for c in res.estimates.columns)


def test_reproducible():
    df = sintetico()
    pd.testing.assert_frame_equal(_ajustar(df, DOS).estimates, _ajustar(df, DOS).estimates)


def test_validation_table():
    t = M.validation_table(_ajustar(sintetico(), DOS))
    assert list(t.columns[:2]) == ["clase", "metrica"] and set(t["metrica"]) == {"f1_macro", "auc"}
    assert {"model", "majority", "stratified", "uniform"} <= set(t.columns)
    assert t["model"].between(0, 100).all()


# ---------------------------------------------------------------- casos borde
def test_cluster_con_menos_de_3_en_la_clase_objetivo_no_estratifica():
    df = sintetico()
    bronce = df.index[df.segmento_trx == "2 Bronce"]
    df.loc[bronce[:2], "seg"] = "RARO"                            # solo 2 filas Bronce en 'RARO'
    res = _ajustar(df, DOS)
    assert len(res.estimates) == len(bronce)


def test_categoria_que_solo_aparece_en_una_mitad_no_rompe():
    df = sintetico()
    bronce = df.index[df.segmento_trx == "2 Bronce"]
    df.loc[bronce[0], "ciudad"] = "Ciudad_unica"                  # solo una fila la tiene
    res = _ajustar(df, DOS)
    assert len(res.estimates) == len(bronce) and res.estimates.filter(like="proba_").notna().all().all()


def test_errores_claros():
    df = sintetico()
    with pytest.raises(KeyError, match="hiperparámetros"):
        M.fit_potential(df, FEATS, CATS, DOS, {}, id_col="id", cluster_col="seg", n_estimators=NT)
    with pytest.raises(KeyError, match="Faltan columnas"):
        M.fit_potential(df.drop(columns=["x1"]), FEATS, CATS, DOS, {"2 Bronce": PAR},
                        id_col="id", cluster_col="seg", n_estimators=NT)
    con_nan = df.copy()
    con_nan.loc[0, "seg"] = np.nan
    with pytest.raises(ValueError, match="valores vacíos"):
        M.fit_potential(con_nan, FEATS, CATS, DOS, {"2 Bronce": PAR}, id_col="id", cluster_col="seg", n_estimators=NT)


def test_avisa_de_clases_fuera_de_class_order():
    df = sintetico()
    df.loc[df.index[:5], "segmento_trx"] = "9 Otro"
    with pytest.warns(UserWarning, match="fuera de class_order"):
        _ajustar(df, DOS)


def test_map_target():
    s = pd.Series(["1 Oro+", "2 Oro", "3 Plata", "4 Bronce", None])
    m = {"1 Oro+": "A", "2 Oro": "A", "3 Plata": "A", "4 Bronce": "B"}
    assert M.map_target(s, m).tolist()[:4] == ["A", "A", "A", "B"] and pd.isna(M.map_target(s, m).iloc[4])
    assert M.map_target(s, None) is s


def test_shap_acepta_lista_o_arreglo(monkeypatch):
    """Según la versión de shap, TreeExplainer devuelve lista por clase o arreglo 3D."""
    import shap
    rng = np.random.default_rng(1)
    sv = rng.normal(size=(6, 4, 2))                              # filas, features, clases

    class Falso:
        def __init__(self, formato): self.formato = formato
        def __call__(self, model): return self
        def shap_values(self, X): return sv if self.formato == "array" else [sv[:, :, 0], sv[:, :, 1]]

    class Modelo:
        classes_ = np.array(["a", "b"])

    X = pd.DataFrame(np.zeros((6, 4)))
    preds = np.array(["a", "b", "b", "a", "a", "b"])
    esperado = np.array([sv[i, :, 0 if p == "a" else 1] for i, p in enumerate(preds)])
    for formato in ("array", "lista"):
        monkeypatch.setattr(shap, "TreeExplainer", Falso(formato))
        np.testing.assert_allclose(M._shap_of_predicted_class(Modelo(), X, preds), esperado)


def test_optuna_smoke():
    en_rango = {**PAR, "num_leaves": 20, "min_child_samples": 10}     # dentro del espacio de búsqueda de Optuna
    res = M.fit_potential(
        sintetico(n=500), FEATS, CATS, DOS, {"2 Bronce": en_rango},
        id_col="id", cluster_col="seg", n_estimators=NT, optuna_trials=2,
    )
    assert {"lambda_l2", "num_leaves", "objective"} <= set(res.params["2 Bronce"])
    assert len(res.estimates) > 0


# ---------------------------------------------------------------- integración con el resto
def test_modelo_decision_ensamblado():
    df = sintetico()
    res = _ajustar(df, DOS)
    dec = apply_decision(res.estimates, "umbral_2clases", DOS, high=0.75, medium=0.5)
    # 2 clases con umbral_2clases: marca potencial exactamente a los Bronce con P(clase alta) >= 0.5
    assert (dec["flag_potential"] == (dec["proba_1 Plata - Oro - Oro+"] >= 0.5)).all()

    cliente = df.assign(X=-77.0, Y=-12.0, des_ciudad="Lima")
    out = assemble_output(dec, cliente, FEATS + ["X", "Y"], id_col="id")
    assert len(out) == len(dec) and not any(c.endswith(("_x", "_y")) for c in out.columns)
    assert out["flag_potential"].sum() == dec["flag_potential"].sum()
