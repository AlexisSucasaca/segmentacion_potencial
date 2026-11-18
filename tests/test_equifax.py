"""Tests de src/equifax.py.

Unitarios: reproducen a mano el bucle original (groupby + vecino más cercano) con datos
sintéticos. Contra la línea base: usan equifax.csv y urbanicidad.csv reales
(`--insumos-dir`, por defecto la carpeta de `--baseline-dir`); equifax.csv pesa ~260 MB y
se cachea igual que los Excel grandes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equifax import COLS_HOGARES, COLS_INTERES, COLS_SUMA, agregar_equifax, prepare_equifax_sum
from src.config import load_config

CLIENTE = [n for n, c in load_config().items() if c["granularidad"] == "cliente"]
MAX_FILAS = 3000


# ---------------------------------------------------------------- datos sintéticos
def _equifax(seed=0, manzanas=40, filas_por_manzana=3):
    """Varias filas por manzana (id_origen, X, Y) para probar el agrupado, con duplicados exactos."""
    rng = np.random.default_rng(seed)
    filas = []
    for m in range(manzanas):
        x, y = -77.0 + rng.normal(scale=0.5), -12.0 + rng.normal(scale=0.5)
        for _ in range(filas_por_manzana):
            filas.append({"id_origen": 1000 + m, "X": x, "Y": y,
                          **{c: float(rng.integers(0, 20)) for c in COLS_SUMA}})
    df = pd.DataFrame(filas)
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)          # una fila exactamente duplicada


def _urbanicidad(equifax):
    ids = equifax["id_origen"].unique()
    rng = np.random.default_rng(1)
    return pd.DataFrame({"id_origen": ids, "kpi_u": rng.uniform(1, 10, len(ids))})


def test_prepare_equifax_sum_agrupa_y_descarta_duplicados_exactos():
    eq = _equifax()
    urb = _urbanicidad(eq)
    out = prepare_equifax_sum(eq, urb)
    assert len(out) == eq.drop_duplicates()[["id_origen", "X", "Y"]].drop_duplicates().shape[0]
    fila = out[out.id_origen == 1000].iloc[0]
    esperado = eq.drop_duplicates()
    esperado = esperado[esperado.id_origen == 1000]["VIVIENDAS"].sum()
    assert fila["VIVIENDAS"] == esperado
    assert set(urb.set_index("id_origen").loc[out.id_origen, "kpi_u"]) == set(out.kpi_u)


def test_prepare_equifax_sum_valida_columnas():
    with pytest.raises(KeyError, match="Faltan columnas"):
        prepare_equifax_sum(pd.DataFrame({"id_origen": [1]}), pd.DataFrame({"id_origen": [1], "kpi_u": [1.0]}))
    with pytest.raises(KeyError, match="id_origen.*kpi_u|kpi_u.*id_origen"):
        prepare_equifax_sum(_equifax(), pd.DataFrame({"id_origen": [1000]}))


def test_agregar_equifax_toma_la_manzana_mas_cercana_y_calcula_derivadas():
    eq = _equifax()
    resumen = prepare_equifax_sum(eq, _urbanicidad(eq))
    clientes = pd.DataFrame({"X": resumen["X"] + 1e-6, "Y": resumen["Y"] + 1e-6})   # justo sobre cada manzana
    out = agregar_equifax(clientes, resumen)

    np.testing.assert_allclose(out["score_urbanicidad"], resumen["kpi_u"])
    np.testing.assert_allclose(out["total_ptos_de_interes"], resumen[COLS_INTERES].sum(axis=1))
    total = resumen[COLS_HOGARES].sum(axis=1)
    for c in COLS_HOGARES:
        np.testing.assert_allclose(out[f"pct_hogares_{c[-1].lower()}"], resumen[c] / total)
    suma_pct = sum(out[f"pct_hogares_{c[-1].lower()}"] for c in COLS_HOGARES)
    np.testing.assert_allclose(suma_pct, 1.0)


def test_agregar_equifax_no_modifica_ni_pierde_columnas_del_cliente():
    eq = _equifax()
    resumen = prepare_equifax_sum(eq, _urbanicidad(eq))
    clientes = pd.DataFrame({"X": [-77.0], "Y": [-12.0], "id": ["c1"], "otra": [42]})
    out = agregar_equifax(clientes, resumen)
    assert out["id"].tolist() == ["c1"] and out["otra"].tolist() == [42]
    assert len(clientes.columns) == 4                                       # clientes no se modifica


def test_agregar_equifax_validaciones():
    eq = _equifax()
    resumen = prepare_equifax_sum(eq, _urbanicidad(eq))
    with pytest.raises(KeyError, match="Faltan columnas"):
        agregar_equifax(pd.DataFrame({"X": [-77.0]}), resumen)
    with pytest.raises(ValueError, match="NaN"):
        agregar_equifax(pd.DataFrame({"X": [-77.0, np.nan], "Y": [-12.0, -12.0]}), resumen)


# ---------------------------------------------------------------- integración con features.py
@pytest.mark.parametrize("nombre", CLIENTE)
def test_agregar_features_cliente_completo(nombre, config, baseline_dir, insumos, equifax_sum):
    """agregar_features_cliente junta scores de 1 km, Mass y Equifax en un solo llamado."""
    from src.features import agregar_features_cliente
    from tests.conftest import read_sheet

    cfg = config[nombre]
    candidatos = sorted(baseline_dir.glob(f"*{cfg['salida']}.xlsx"))
    if not candidatos:
        pytest.skip(f"No hay Excel de referencia para '{nombre}'")
    cols = ["Cod Cliente DEX", "X", "Y", "score_urbanicidad", "total_ptos_de_interes", "score_comercial"]
    ref = read_sheet(candidatos[-1], cfg["export"]["hoja_principal"], cols, max_rows=1500)
    bq = pd.DataFrame({"cod_cliente_dex": ref["Cod Cliente DEX"],
                       "val_coordenada_x": ref["X"], "val_coordenada_y": ref["Y"]})
    df, info = agregar_features_cliente(bq, insumos["puntos"], insumos["locales"], equifax_sum)
    assert info["sin_coordenadas_validas"] == 0
    assert {"score_urbanicidad", "total_ptos_de_interes", "score_comercial", "DistanciaTiendaMass"} <= set(df.columns)
    assert len(df) == len(ref)


# ---------------------------------------------------------------- contra la línea base
@pytest.mark.parametrize("nombre", CLIENTE)
def test_equifax_reproduce_la_linea_base(nombre, config, baseline_dir, equifax_sum):
    from tests.conftest import read_sheet

    cfg = config[nombre]
    candidatos = sorted(baseline_dir.glob(f"*{cfg['salida']}.xlsx"))
    if not candidatos:
        pytest.skip(f"No hay Excel de referencia para '{nombre}'")
    cols = ["Cod Cliente DEX", "X", "Y", "score_urbanicidad", "total_ptos_de_interes",
            *[f"pct_hogares_{c}" for c in "abcde"]]
    ref = read_sheet(candidatos[-1], cfg["export"]["hoja_principal"], cols, max_rows=MAX_FILAS)
    for c in cols[1:]:
        ref[c] = pd.to_numeric(ref[c], errors="coerce")

    out = agregar_equifax(ref[["X", "Y"]], equifax_sum)
    for c in cols[3:]:
        dif = np.abs(out[c].to_numpy(dtype=float) - ref[c].to_numpy(dtype=float))
        ok = (dif < 1e-6) | (np.isnan(out[c].to_numpy(dtype=float)) & np.isnan(ref[c].to_numpy(dtype=float)))
        assert ok.mean() >= 0.999, f"'{c}' distinto en {int((~ok).sum())} de {len(ref)} filas ({nombre})"
