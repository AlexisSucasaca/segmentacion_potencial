"""Tests del ensamblado con columnas limpias.

Contra la línea base: se parte cada Excel en (estimaciones + decisión) y (datos del cliente),
se vuelve a ensamblar y debe resultar la misma hoja que el Excel actual, salvo que cada
columna sale una sola vez (sin sufijos _x/_y ni X, Y repetidas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assemble import assemble_output
from src.config import load_config

# Verticales de cliente (PM es a nivel mercado y se ensambla en el paso de agregación).
CLIENTE = [n for n, c in load_config().items() if c["granularidad"] == "cliente"]
MAX_FILAS = 5000            # las verticales chicas se leen completas; bodegas, una muestra

SUFIJADAS = {"des_ciudad", "des_segmento_estrategico"}


# ---------------------------------------------------------------- unitarios
def _est():
    return pd.DataFrame({
        "id": [1, 2, 3], "segmento_trx": ["2 Bronce"] * 3,
        "des_segmento_estrategico": ["A", "B", "C"], "final_pred": ["x", "y", "z"],
    })


def _cli():
    return pd.DataFrame({
        "id": [3, 1, 2],                                   # otro orden: el merge debe alinear por id
        "des_ciudad": ["Lima", "Cusco", "Piura"], "X": [-77.0, -72.0, -80.0], "Y": [-12.0, -13.5, -5.2],
        "des_segmento_estrategico": ["C", "A", "B"], "score": [3.0, 1.0, 2.0],
    })


def test_ensamblado_sin_columnas_repetidas_ni_sufijos():
    out = assemble_output(_est(), _cli(), ["X", "Y", "des_ciudad", "des_segmento_estrategico", "score"], id_col="id")
    assert list(out.columns) == [
        "id", "segmento_trx", "des_segmento_estrategico", "final_pred", "des_ciudad", "X", "Y", "score",
    ]
    assert not any(c.endswith(("_x", "_y")) for c in out.columns)
    assert out["id"].tolist() == [1, 2, 3]                 # conserva el orden de las estimaciones
    assert out["des_ciudad"].tolist() == ["Cusco", "Piura", "Lima"]
    assert out["score"].tolist() == [1.0, 2.0, 3.0]


def test_ensamblado_no_pisa_columnas_de_la_estimacion():
    cli = _cli().assign(des_segmento_estrategico=["¡distinto!"] * 3)
    out = assemble_output(_est(), cli, ["score", "des_segmento_estrategico"], id_col="id")
    assert out["des_segmento_estrategico"].tolist() == ["A", "B", "C"]   # gana la de la estimación


def test_ensamblado_tolera_columnas_duplicadas_en_client_data():
    cli = pd.concat([_cli(), _cli()[["X", "Y"]]], axis=1)                 # X, Y repetidas
    out = assemble_output(_est(), cli, ["score"], id_col="id")
    assert list(out.columns).count("X") == 1 and list(out.columns).count("Y") == 1


def test_ensamblado_geo_es_opcional():
    out = assemble_output(_est(), _cli().drop(columns=["des_ciudad"]), ["score"], id_col="id")
    assert "des_ciudad" not in out.columns and "score" in out.columns


def test_ensamblado_falla_con_ids_repetidos():
    cli = pd.concat([_cli(), _cli().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="repetidos"):
        assemble_output(_est(), cli, ["score"], id_col="id")


def test_ensamblado_falla_si_faltan_features():
    with pytest.raises(KeyError, match="inexistente"):
        assemble_output(_est(), _cli(), ["score", "inexistente"], id_col="id")


def test_ensamblado_cliente_sin_datos_queda_vacio_pero_no_se_pierde():
    cli = _cli().iloc[:2]
    out = assemble_output(_est(), cli, ["score"], id_col="id")
    assert len(out) == 3 and out["score"].isna().sum() == 1


# ---------------------------------------------------------------- contra la línea base
def _quitar_sufijo(h: str) -> str:
    for suf in ("_x", "_y"):
        if h.endswith(suf) and h[: -len(suf)] in SUFIJADAS:
            return h[: -len(suf)]
    return h


@pytest.mark.parametrize("nombre", CLIENTE)
def test_ensamblado_reproduce_la_hoja_principal_sin_repetidas(nombre, config, baseline_dir):
    from tests.conftest import _header, read_sheet

    cfg = config[nombre]
    candidatos = sorted(baseline_dir.glob(f"*{cfg['salida']}.xlsx"))
    if not candidatos:
        pytest.skip(f"No hay Excel de referencia para '{nombre}'")
    path = candidatos[-1]
    hoja = cfg["export"]["hoja_principal"]
    id_col = cfg["id_col"]

    hdr = _header(path, hoja)
    i_flag = hdr.index("flag_potential")
    est_cols = list(dict.fromkeys(_quitar_sufijo(h) for h in hdr[: i_flag + 1]))
    cli_cols = list(dict.fromkeys(_quitar_sufijo(h) for h in hdr[i_flag + 1:]))
    features = [c for c in cli_cols if c not in ("des_ciudad", "X", "Y")]
    shap = {h[5:] for h in hdr if isinstance(h, str) and h.startswith("shap_")}
    assert set(features) - {"des_segmento_estrategico"} <= shap, "las features deberían coincidir con las shap_*"

    ref = read_sheet(path, hoja, max_rows=MAX_FILAS)
    ref = ref.rename(columns={"des_ciudad_x": "des_ciudad", "des_segmento_estrategico_x": "des_segmento_estrategico"})
    ref = ref.drop(columns=["des_ciudad_y", "des_segmento_estrategico_y"], errors="ignore")

    # Se parte el Excel en sus dos mitades y se vuelve a ensamblar.
    estimates = ref[[c for c in est_cols if c in ref.columns]]
    client_data = ref[[id_col, *[c for c in cli_cols if c in ref.columns]]]
    out = assemble_output(estimates, client_data, features, id_col=id_col)

    esperado_cols = [c for c in dict.fromkeys(_quitar_sufijo(h) for h in hdr) if c in ref.columns]
    assert list(out.columns) == esperado_cols
    assert not any(c.endswith(("_x", "_y")) for c in out.columns)
    assert len(out) == len(ref)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), ref[esperado_cols].reset_index(drop=True),
                                  check_dtype=False)
