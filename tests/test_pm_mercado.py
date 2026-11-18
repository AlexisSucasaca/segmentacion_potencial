"""Tests de src/pm_mercado.py.

Unitarios: con datos sintéticos, sin archivos tuyos. Contra la línea base: usan
Maestro_de_mercado.xlsx, cluster_mercado_id.csv, equifax.csv/urbanicidad.csv y el Excel de
PM (`--baseline-dir` / `--insumos-dir`). Como la base de clientes de PM ya no es exactamente
la del 2026-08-31, se compara con tolerancia (como en test_features.py), no celda a celda.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pm_mercado import SEG_TRX_MAP, aggregate_mercado, assign_mercado_id, get_mode, reference_rows

TOLERANCIA_FILAS = 0.02          # igual que en score_comercial/competencia: la base cambió desde el 31/08


# ---------------------------------------------------------------- get_mode
def test_get_mode():
    assert get_mode(pd.Series(["a", "a", "b"])) == "a"
    assert get_mode(pd.Series([], dtype=object)) is None


# ---------------------------------------------------------------- assign_mercado_id
def _maestro():
    return pd.DataFrame({"Codigo cliente": ["0000000001", "0000000002", "0000000003"],
                         "CODIMOD.REF": ["M001", "M002", None]})


def _cluster():
    return pd.DataFrame({"Cod Cliente DEX": [3, 4], "mercado_id": [77, 88]})


def test_assign_mercado_id_maestro_con_respaldo_de_cluster():
    clientes = pd.DataFrame({"cod_cliente_dex": ["1", "2", "3", "4", "5"]})
    out = assign_mercado_id(clientes, _maestro(), _cluster())
    assert out.tolist() == ["M001", "M002", "77", "88", None]


def test_assign_mercado_id_maestro_gana_sobre_cluster():
    """Un cliente con CODIMOD.REF no se toca aunque también esté en el respaldo."""
    maestro = pd.concat([_maestro(), pd.DataFrame({"Codigo cliente": ["0000000004"], "CODIMOD.REF": ["M004"]})])
    out = assign_mercado_id(pd.DataFrame({"cod_cliente_dex": ["4"]}), maestro, _cluster())
    assert out.tolist() == ["M004"]


def test_assign_mercado_id_valida_columnas():
    with pytest.raises(KeyError, match="maestro"):
        assign_mercado_id(pd.DataFrame({"cod_cliente_dex": ["1"]}), pd.DataFrame({"x": [1]}), _cluster())
    with pytest.raises(KeyError, match="cluster_fallback"):
        assign_mercado_id(pd.DataFrame({"cod_cliente_dex": ["1"]}), _maestro(), pd.DataFrame({"x": [1]}))


# ---------------------------------------------------------------- aggregate_mercado
def _clientes_sinteticos(n_mercados=4, por_mercado=5, seed=0):
    rng = np.random.default_rng(seed)
    filas = []
    for m in range(n_mercados):
        base_x, base_y = -77.0 + m, -12.0 + m
        for i in range(por_mercado):
            filas.append({
                "Cod Cliente DEX": f"c{m}_{i}", "mercado_id": f"M{m}",
                "segmento_trx": rng.choice(list(SEG_TRX_MAP), p=[0.05, 0.15, 0.3, 0.5]),
                "X": base_x + rng.normal(scale=1e-6), "Y": base_y + rng.normal(scale=1e-6),
                "des_ciudad": rng.choice(["Lima", "Lima", "Callao"]),
                "des_segmento_estrategico": rng.choice(["Abarrotero", "Lavandero"]),
                "tip_documento": rng.choice(["DNI", "RUC"]),
                "cuadrante": rng.choice(["Oportunidad", "Batalla", "Potencial", "Riesgo"]),
                "flg_cliente_perfecto": bool(rng.integers(0, 2)),
                "cod_cliente_alicorp_actual": f"dex{m}_{i % 2}",
                "pct_venta_diaadia_insuma": rng.uniform(0, 100), "pct_descuento": rng.uniform(0, 30),
                "score_comercial": rng.uniform(1, 10), "score_competencia": rng.uniform(1, 10),
                "DistanciaTiendaMass": rng.uniform(0, 5000),
                **{c: rng.integers(0, 5) for c in
                   ["Mercado Abastos", "Mercado Mayorista", "Colegio", "Universidad", "Terminal Buses",
                    "Hospital", "Hotel", "Restaurante", "Banco", "Residencial", "Grifo",
                    "Centro Comercial", "Iglesia", "Supermercado", "Comida Rapida",
                    "Comp_BO_Tambo", "Comp_BO_Mass", "Comp_BO_Listo", "Comp_BO_Oxxo"]},
            })
    return pd.DataFrame(filas)


def _equifax_sum(clientes, seed=1):
    """Una manzana por mercado, justo en su centro."""
    rng = np.random.default_rng(seed)
    centros = clientes.groupby("mercado_id")[["X", "Y"]].mean().reset_index()
    return pd.DataFrame({
        "id_origen": range(len(centros)), "mercado_id": centros["mercado_id"],
        "X": centros["X"], "Y": centros["Y"],
        "kpi_u": rng.uniform(1, 10, len(centros)),
        **{c: rng.integers(0, 100, len(centros)).astype(float) for c in
           ["VIVIENDAS", "POB", "HOG_A", "HOG_B", "HOG_C", "HOG_D", "HOG_E",
            "Educacion", "Ent_Finan", "Gastronomi", "Grifos", "Hosteleria", "Mercado", "Retails", "Salud"]},
    })


def test_aggregate_mercado_forma_y_valores():
    clientes = _clientes_sinteticos()
    es = _equifax_sum(clientes)
    out = aggregate_mercado(clientes, es)

    assert len(out) == clientes["mercado_id"].nunique()
    assert set(out["mercado_id"]) == set(clientes["mercado_id"])
    assert (out["count_pm"] == clientes.groupby("mercado_id").size().reindex(out["mercado_id"]).to_numpy()).all()
    assert out["segmento_trx"].isin(SEG_TRX_MAP).all()

    fila = out[out["mercado_id"] == "M0"].iloc[0]
    grupo = clientes[clientes["mercado_id"] == "M0"]
    assert fila["X"] == pytest.approx(grupo["X"].median())
    assert fila["des_ciudad"] == get_mode(grupo["des_ciudad"])
    pct_cols = [c for c in out.columns if c.startswith("pct_") and c[4:] in {"abarrotero", "lavandero"}]
    suma = out.loc[out["mercado_id"] == "M0", pct_cols].sum(axis=1).iloc[0]
    assert suma == pytest.approx(1.0)


def test_aggregate_mercado_pct_hogares_suma_uno():
    clientes = _clientes_sinteticos()
    out = aggregate_mercado(clientes, _equifax_sum(clientes))
    cols = [f"pct_hogares_{c}" for c in "abcde"]
    np.testing.assert_allclose(out[cols].sum(axis=1), 1.0)


def test_aggregate_mercado_descarta_mercados_lejos_de_equifax():
    clientes = _clientes_sinteticos(n_mercados=2)
    es = _equifax_sum(clientes)
    out_normal = aggregate_mercado(clientes, es)
    assert len(out_normal) == 2                                  # ambos mercados tienen manzana a 0 km
    out_casi_cero = aggregate_mercado(clientes, es, equifax_radio_km=0.001)
    assert len(out_casi_cero) == 2                                # ambos mercados, con jitter mínimo
    lejos = es.copy()
    lejos.loc[lejos["mercado_id"] == "M0", ["X", "Y"]] = [0.0, 0.0]   # aleja la manzana de M0
    out_uno_lejos = aggregate_mercado(clientes, lejos, equifax_radio_km=1.0)
    assert set(out_uno_lejos["mercado_id"]) == {"M1"}


def test_aggregate_mercado_valida_columnas_y_segmento():
    clientes = _clientes_sinteticos()
    es = _equifax_sum(clientes)
    with pytest.raises(KeyError, match="Faltan columnas"):
        aggregate_mercado(clientes.drop(columns=["des_ciudad"]), es)
    con_nan = clientes.copy()
    con_nan.loc[0, "mercado_id"] = None
    with pytest.raises(ValueError, match="sin.*mercado_id|mercado_id.*sin"):
        aggregate_mercado(con_nan, es)
    mal_seg = clientes.copy()
    mal_seg.loc[0, "segmento_trx"] = "9 Otro"
    with pytest.raises(ValueError, match="segmento_trx"):
        aggregate_mercado(mal_seg, es)


# ---------------------------------------------------------------- contra la línea base
def test_mercado_id_reproduce_la_linea_base(config, baseline_dir, pm_insumos):
    from tests.conftest import read_sheet

    cfg = config["pm_mercado"]
    candidatos = sorted(baseline_dir.glob(f"*{cfg['salida']}.xlsx"))
    if not candidatos:
        pytest.skip("No hay Excel de referencia para 'pm_mercado'")
    ids = read_sheet(candidatos[-1], "IDs mercado")
    ids["Cod Cliente DEX"] = ids["Cod Cliente DEX"].astype(str).str.zfill(10)

    calculado = assign_mercado_id(
        ids.rename(columns={"Cod Cliente DEX": "cod_cliente_dex"}),
        pm_insumos["maestro"], pm_insumos["cluster_fallback"], id_col="cod_cliente_dex",
    )
    esperado = ids["mercado_id"].astype(object)
    ok = (calculado.astype(str).to_numpy() == esperado.astype(str).to_numpy()) | (
        calculado.isna().to_numpy() & esperado.isna().to_numpy()
    )
    assert ok.mean() >= 0.999, f"mercado_id distinto en {int((~ok).sum())} de {len(ids)} clientes"


# ---------------------------------------------------------------- reference_rows
def test_reference_rows_solo_clase_alta_y_columnas_correctas():
    ub = pd.DataFrame({
        "mercado_id": ["M0", "M1", "M2"],
        "segmento_trx": ["1 Plata - Oro - Oro+", "2 Bronce", "1 Plata - Oro - Oro+"],
        "X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0], "des_ciudad": ["Lima", "Cusco", "Piura"],
        "score_urbanicidad": [7.0, 8.0, 9.0], "otra_columna": ["a", "b", "c"],
    })
    ref = reference_rows(ub, ["1 Plata - Oro - Oro+", "2 Bronce"], "mercado_id", ["score_urbanicidad"])
    assert ref["mercado_id"].tolist() == ["M0", "M2"]
    assert list(ref.columns) == ["mercado_id", "X", "Y", "des_ciudad", "segmento_trx", "score_urbanicidad"]
    assert "otra_columna" not in ref.columns


def test_reference_rows_concatena_con_las_estimaciones_dejando_nan_lo_modelado():
    from src.decision import apply_decision
    from src.assemble import assemble_output

    ALTA, BAJA = "1 Plata - Oro - Oro+", "2 Bronce"
    ub = pd.DataFrame({
        "mercado_id": ["M0", "M1", "M2"], "segmento_trx": [BAJA, BAJA, ALTA],
        "X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0], "des_ciudad": ["Lima", "Cusco", "Piura"],
        "score": [10.0, 20.0, 30.0],
        f"proba_{ALTA}": [0.8, 0.2, np.nan], f"proba_{BAJA}": [0.2, 0.8, np.nan],
        "predicted": [ALTA, BAJA, None],
    })
    est = apply_decision(ub[ub.segmento_trx == BAJA], "acumulada", [ALTA, BAJA], high=0.66, medium=0.5)
    ensambladas = assemble_output(est, ub, ["score"], id_col="mercado_id", geo_cols=("des_ciudad",))
    ref = reference_rows(ub, [ALTA, BAJA], "mercado_id", ["score"])
    salida = pd.concat([ensambladas, ref], ignore_index=True)

    assert len(salida) == 3
    fila_m2 = salida[salida["mercado_id"] == "M2"].iloc[0]
    assert fila_m2["segmento_trx"] == ALTA and fila_m2["score"] == 30.0
    assert pd.isna(fila_m2["final_pred"]) and pd.isna(fila_m2["flag_potential"])


def test_reference_rows_valida_columnas():
    ub = pd.DataFrame({"mercado_id": ["M0"], "segmento_trx": ["2 Bronce"], "X": [1.0], "Y": [2.0]})
    with pytest.raises(KeyError, match="Faltan columnas"):
        reference_rows(ub, ["1 Plata - Oro - Oro+", "2 Bronce"], "mercado_id", ["score"])
