"""Tests de src/pdm.py.

`test_equivale_al_notebook_original` reproduce, tal cual, la lógica de `PDM_potencial_vs2.ipynb`
(sin la consulta a BigQuery — ver el docstring de `src/pdm.py`) y compara contra `construir_pdm_final`
con datos sintéticos. Los demás tests son unitarios. `test_contra_csv_reales` usa una muestra de
tus `pdm_1.csv` / `pdm_2.csv` / `top15.csv` si están disponibles (`--insumos-dir`); `pdm_2.csv`
pesa ~280 MB, así que se lee con `nrows` para no demorar la corrida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pdm import (
    FACTOR_UMBRAL_DEFAULT, armar_base_pdm, clasificar_potencial, construir_pdm_final,
    familias_top_compradas, map_segmento_pdm, resumen_cliente, resumen_grupo, sku_por_mes,
)


# ---------------------------------------------------------------- map_segmento_pdm
def test_map_segmento_pdm():
    s = pd.Series(["Abarrotero", "Lavandero", "Confitero"])
    assert map_segmento_pdm(s).tolist() == ["Abarrotero", "Lavandero", "Impulso"]


# ---------------------------------------------------------------- datos sintéticos comunes
def _mundo(seed=0, n_clientes=40, n_periodos=3):
    rng = np.random.default_rng(seed)
    clientes = np.arange(1000, 1000 + n_clientes)
    mercados = rng.choice(["M1", "M2", "M3"], n_clientes)
    distritos = rng.choice(["Lima", "Callao", "Cusco"], n_clientes)
    segmentos_raw = rng.choice(["Abarrotero", "Lavandero", "Confitero"], n_clientes)
    dex_padre = rng.integers(9000, 9005, n_clientes)   # varios clientes por padre (como en la realidad)
    periodos = pd.date_range("2026-01-01", periods=n_periodos, freq="MS").strftime("%Y-%m-%d")

    pdm_1 = pd.DataFrame([
        {"periodo": p, "cod_cliente_alicorp_actual": int(dex_padre[i]), "cod_cliente_dex": int(clientes[i]),
         "des_distrito": distritos[i], "cnt_venta_real": rng.uniform(1, 50),
         "num_peso_real": rng.uniform(10, 500), "mnt_venta_real_bruto": rng.uniform(100, 5000)}
        for i in range(n_clientes) for p in periodos
    ])
    familias = [f"Familia {i}" for i in range(10)]
    pdm_2 = pd.DataFrame([
        {"periodo": p, "cod_cliente_alicorp_actual": int(dex_padre[i]), "cod_cliente_dex": int(clientes[i]),
         "des_familia": fam}
        for i in range(n_clientes) for p in periodos
        for fam in rng.choice(familias, rng.integers(1, 6), replace=False)
    ])
    top15 = pd.DataFrame({"Cod DEX Hijo": np.repeat(np.unique(dex_padre), 3),
                          "des_familia": [familias[j] for _ in np.unique(dex_padre) for j in range(3)]})

    ids_mercado = pd.DataFrame({"Cod Cliente DEX": clientes, "mercado_id": mercados})
    segmento_por_cliente = pd.DataFrame({
        "Cod Cliente DEX": clientes, "segmento": map_segmento_pdm(pd.Series(segmentos_raw)),
    })
    potencial_pm = pd.DataFrame({"mercado_id": ["M1", "M2", "M3"], "flag_potential": [1.0, 1.0, 0.0]})
    return pdm_1, pdm_2, top15, ids_mercado, segmento_por_cliente, potencial_pm


# ---------------------------------------------------------------- equivalencia con el notebook original
def _pdm_original(pdm_1, pdm_2, top15, ids_mercado, estrategico, potencial, factor=0.25):
    """Copia de la lógica de PDM_potencial_vs2.ipynb, con `estrategico` (Cod Cliente DEX +
    segmento, ya con Confitero->Impulso) reemplazando a la consulta de BigQuery."""
    dex = ids_mercado.dropna()
    dex = dex.merge(estrategico, on="Cod Cliente DEX", how="inner")

    merged = pd.merge(pdm_2, top15[["Cod DEX Hijo", "des_familia"]],
                      left_on=["cod_cliente_alicorp_actual", "des_familia"],
                      right_on=["Cod DEX Hijo", "des_familia"], how="inner")
    fam_top = merged.groupby(["periodo", "cod_cliente_dex"]).size().reset_index(name="num_familias_top")
    fam_top = fam_top[fam_top["cod_cliente_dex"] != "Sin Asignar"]
    fam_top = fam_top[pd.to_numeric(fam_top["cod_cliente_dex"], errors="coerce").notna()]
    fam_top["cod_cliente_dex"] = fam_top["cod_cliente_dex"].astype(int)

    pdm_1 = pdm_1.merge(fam_top, on=["periodo", "cod_cliente_dex"], how="left")
    pdm_1 = pdm_1.merge(dex, left_on="cod_cliente_dex", right_on="Cod Cliente DEX", how="left")

    summary_pdm_2 = pdm_2.groupby(["periodo", "cod_cliente_dex"])[["des_familia"]].agg("count").reset_index()
    summary_pdm_2 = summary_pdm_2[pd.to_numeric(summary_pdm_2["cod_cliente_dex"], errors="coerce").notna()]
    summary_pdm_2["cod_cliente_dex"] = summary_pdm_2["cod_cliente_dex"].astype(int)
    pdm_1 = pdm_1.merge(summary_pdm_2, on=["cod_cliente_dex", "periodo"], how="left")

    pdm_1 = pdm_1.merge(potencial, on="mercado_id", how="left")
    pdm_1 = pdm_1[pdm_1["flag_potential"] == 1.0]
    pdm_1["num_familias_top"] = pdm_1["num_familias_top"].fillna(0)

    abarrateros = pdm_1[pdm_1["segmento"] == "Abarrotero"]
    lavanderos = pdm_1[pdm_1["segmento"] == "Lavandero"]
    impulso = pdm_1[pdm_1["segmento"] == "Impulso"]
    cols = ["cnt_venta_real", "num_peso_real", "mnt_venta_real_bruto", "num_familias_top", "des_familia"]

    abarrateros_summary = abarrateros.groupby(["cod_cliente_dex", "mercado_id"])[cols].mean().reset_index()
    lavanderos_summary = lavanderos.groupby(["cod_cliente_dex", "des_distrito"])[cols].mean().reset_index()
    impulso_summary = impulso.groupby(["cod_cliente_dex", "des_distrito"])[cols].mean().reset_index()

    def resumen(df, group_col):
        s = df.groupby(group_col)[cols].agg(["mean", "std"]).reset_index()
        s.columns = [group_col] + [f"{c}_{st}" for c, st in s.columns[1:]]
        for c in cols:
            s[f"Threshold_{c}"] = s[f"{c}_mean"] + s[f"{c}_std"] * factor
        return s

    market_summary = resumen(abarrateros, "mercado_id")
    distrito_summary = resumen(lavanderos, "des_distrito")   # el notebook usa lavanderos para ambos

    def clasificar(summary, group_summary, group_col):
        summary = summary.merge(group_summary, on=group_col, how="left")
        summary["cond_venta"] = (summary["mnt_venta_real_bruto"] < summary["Threshold_mnt_venta_real_bruto"]).astype(int)
        summary["cond_familias"] = (summary["num_familias_top"] < summary["Threshold_num_familias_top"]).astype(int)
        summary["cond_cnt"] = (summary["cnt_venta_real"] < summary["Threshold_cnt_venta_real"]).astype(int)
        summary["suma_condiciones"] = summary[["cond_venta", "cond_familias", "cond_cnt"]].sum(axis=1)
        summary["Potencial"] = summary["suma_condiciones"].map(
            {0: "No Potencial", 1: "Low", 2: "Medium", 3: "High"})
        return summary.drop(columns=["cond_venta", "cond_familias", "cond_cnt", "suma_condiciones"])

    abarrateros_summary = clasificar(abarrateros_summary, market_summary, "mercado_id")
    lavanderos_summary = clasificar(lavanderos_summary, distrito_summary, "des_distrito")
    impulso_summary = clasificar(impulso_summary, distrito_summary, "des_distrito")
    return {"Abarroteros": abarrateros_summary, "Lavanderos": lavanderos_summary, "Impulso": impulso_summary}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_equivale_al_notebook_original(seed):
    pdm_1, pdm_2, top15, ids_mercado, segmento_por_cliente, potencial_pm = _mundo(seed=seed)
    ref = _pdm_original(pdm_1.copy(), pdm_2.copy(), top15.copy(), ids_mercado.copy(),
                        segmento_por_cliente.copy(), potencial_pm.copy())
    nuevo = construir_pdm_final(pdm_1, pdm_2, top15, ids_mercado, segmento_por_cliente, potencial_pm)

    assert set(nuevo) == set(ref)
    for hoja in ref:
        a = ref[hoja].sort_values(list(ref[hoja].columns[:2])).reset_index(drop=True)
        b = nuevo[hoja].sort_values(list(nuevo[hoja].columns[:2])).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


# ---------------------------------------------------------------- unitarios de cada pieza
def test_familias_top_compradas():
    pdm_2 = pd.DataFrame({
        "periodo": ["2026-01-01"] * 3, "cod_cliente_alicorp_actual": [1, 1, 1],
        "cod_cliente_dex": [10, 10, 11], "des_familia": ["A", "B", "A"],
    })
    top15 = pd.DataFrame({"Cod DEX Hijo": [1, 1], "des_familia": ["A", "C"]})
    out = familias_top_compradas(pdm_2, top15)
    assert out.set_index("cod_cliente_dex")["num_familias_top"].to_dict() == {10: 1, 11: 1}


def test_familias_top_compradas_descarta_sin_asignar_y_no_numericos():
    pdm_2 = pd.DataFrame({
        "periodo": ["2026-01-01"] * 2, "cod_cliente_alicorp_actual": [1, 1],
        "cod_cliente_dex": ["Sin Asignar", "abc"], "des_familia": ["A", "A"],
    })
    top15 = pd.DataFrame({"Cod DEX Hijo": [1], "des_familia": ["A"]})
    assert familias_top_compradas(pdm_2, top15).empty


def test_familias_top_compradas_valida_columnas():
    with pytest.raises(KeyError, match="Faltan columnas en pdm_2"):
        familias_top_compradas(pd.DataFrame({"x": [1]}), pd.DataFrame({"Cod DEX Hijo": [1], "des_familia": ["A"]}))
    with pytest.raises(KeyError, match="top15"):
        familias_top_compradas(
            pd.DataFrame({"cod_cliente_alicorp_actual": [1], "cod_cliente_dex": [1], "periodo": ["x"], "des_familia": ["A"]}),
            pd.DataFrame({"x": [1]}),
        )


def test_sku_por_mes():
    pdm_2 = pd.DataFrame({
        "periodo": ["2026-01-01"] * 3, "cod_cliente_dex": [10, 10, 11], "des_familia": ["A", "B", "A"],
    })
    out = sku_por_mes(pdm_2)
    assert out.set_index("cod_cliente_dex")["des_familia"].to_dict() == {10: 2, 11: 1}


def test_resumen_grupo_umbral():
    df = pd.DataFrame({"g": ["a", "a", "a"], "mnt_venta_real_bruto": [10.0, 20.0, 30.0],
                       "cnt_venta_real": [1.0, 2.0, 3.0], "num_peso_real": [1.0, 1.0, 1.0],
                       "num_familias_top": [1.0, 1.0, 1.0], "des_familia": [1.0, 1.0, 1.0]})
    out = resumen_grupo(df, "g", factor_umbral=0.5)
    fila = out.iloc[0]
    media, desv = df["mnt_venta_real_bruto"].mean(), df["mnt_venta_real_bruto"].std()
    assert fila["Threshold_mnt_venta_real_bruto"] == pytest.approx(media + desv * 0.5)


def test_clasificar_potencial_etiquetas():
    cliente = pd.DataFrame({
        "id": ["c1", "c2", "c3", "c4"], "g": ["a"] * 4,
        "mnt_venta_real_bruto": [0, 100, 0, 100], "num_familias_top": [0, 0, 100, 100], "cnt_venta_real": [100, 100, 100, 100],
    })
    grupo = pd.DataFrame({"g": ["a"], "Threshold_mnt_venta_real_bruto": [50], "Threshold_num_familias_top": [50],
                          "Threshold_cnt_venta_real": [50]})
    out = clasificar_potencial(cliente, grupo, "g")
    assert out.set_index("id")["Potencial"].to_dict() == {"c1": "Medium", "c2": "Low", "c3": "Low", "c4": "No Potencial"}
    assert not any(c.startswith("cond_") for c in out.columns) and "suma_condiciones" not in out.columns


def test_construir_pdm_final_hojas_y_factor_configurable():
    pdm_1, pdm_2, top15, ids_mercado, segmento_por_cliente, potencial_pm = _mundo()
    salidas = construir_pdm_final(pdm_1, pdm_2, top15, ids_mercado, segmento_por_cliente, potencial_pm)
    assert set(salidas) == {"Abarroteros", "Lavanderos", "Impulso"}
    for df in salidas.values():
        assert "Potencial" in df.columns and df["Potencial"].isin(
            ["No Potencial", "Low", "Medium", "High"]).all()
    # un factor enorme afloja tanto el umbral que casi todos quedan "High" (más venta/familias
    # que un umbral gigantesco siempre será "menor" al umbral); uno muy negativo, lo contrario
    salidas_laxo = construir_pdm_final(pdm_1, pdm_2, top15, ids_mercado, segmento_por_cliente,
                                       potencial_pm, factor_umbral=1000)
    salidas_estricto = construir_pdm_final(pdm_1, pdm_2, top15, ids_mercado, segmento_por_cliente,
                                           potencial_pm, factor_umbral=-1000)
    high_laxo = sum((df["Potencial"] == "High").sum() for df in salidas_laxo.values())
    high_estricto = sum((df["Potencial"] == "High").sum() for df in salidas_estricto.values())
    assert high_laxo > high_estricto


def test_construir_pdm_final_excluye_mercados_sin_potencial():
    pdm_1, pdm_2, top15, ids_mercado, segmento_por_cliente, potencial_pm = _mundo()
    salidas = construir_pdm_final(pdm_1, pdm_2, top15, ids_mercado, segmento_por_cliente, potencial_pm)
    assert "M3" not in salidas["Abarroteros"]["mercado_id"].values  # M3 tiene flag_potential=0


# ---------------------------------------------------------------- contra tus CSV reales
@pytest.fixture(scope="session")
def pdm_insumos_reales(request, baseline_dir):
    from pathlib import Path
    carpeta = Path(request.config.getoption("--insumos-dir") or baseline_dir)
    p1 = sorted(carpeta.glob("pdm_1*.csv"))
    p2 = sorted(carpeta.glob("pdm_2*.csv"))
    t15 = sorted(carpeta.glob("top15*.csv"))
    if not (p1 and p2 and t15):
        pytest.skip(f"Faltan pdm_1.csv, pdm_2.csv y/o top15.csv en {carpeta}")
    return {
        "pdm_1": pd.read_csv(p1[-1], nrows=20000),
        "pdm_2": pd.read_csv(p2[-1], nrows=20000),
        "top15": pd.read_csv(t15[-1]),
    }


def test_contra_csv_reales_no_falla_y_produce_las_3_hojas(pdm_insumos_reales):
    """Solo valida que el pipeline corre sobre datos reales (muestra) y produce las 3 hojas
    con las columnas esperadas — no hay Excel de referencia para este paso final."""
    d = pdm_insumos_reales
    clientes = d["pdm_1"]["cod_cliente_dex"].unique()
    ids_mercado = pd.DataFrame({"Cod Cliente DEX": clientes, "mercado_id": "M1"})
    segmento_por_cliente = pd.DataFrame({
        "Cod Cliente DEX": clientes,
        "segmento": np.resize(["Abarrotero", "Lavandero", "Impulso"], len(clientes)),
    })
    potencial_pm = pd.DataFrame({"mercado_id": ["M1"], "flag_potential": [1.0]})
    if "des_distrito" not in d["pdm_1"].columns:
        pytest.skip("pdm_1.csv de la muestra no trae des_distrito")

    salidas = construir_pdm_final(d["pdm_1"], d["pdm_2"], d["top15"], ids_mercado, segmento_por_cliente, potencial_pm)
    for hoja in ("Abarroteros", "Lavanderos", "Impulso"):
        assert hoja in salidas
        if not salidas[hoja].empty:
            assert "Potencial" in salidas[hoja].columns
