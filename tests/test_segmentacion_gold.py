"""Tests de src/segmentacion_gold.py (datos sintéticos, sin archivos tuyos).

`test_equivale_al_notebook_original` reproduce, tal cual, las celdas 15-35 del notebook
original (sin la consulta a BigQuery) y compara contra `armar_df_final` + `aplicar_reglas_negocio`.
Los demás tests cubren cada pieza por separado y los dos modos de ejecución
(`completo` e `incremental`, con la alterna del trax como autoridad de boca y transaccional).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.segmentacion_gold import (
    COLUMNAS_GOLD, aplicar_reglas_negocio, aplicar_segmento_pa, armar_df_final, cargar_periodo_gold, clasificar_salidas,
    combinar_potencial, completar_incremental, construir_insert_gold, detectar_regresiones,
    detectar_salidas, flag_mercado_potencial, grupo_boca, preparar_carga_periodo, ultimo_dia_mes,
    union_encuestas, union_potencial_clientes, union_potencial_mercado_pdm, validar_combinaciones,
    preparar_segmento_pa, validar_incremental,
)


def test_ultimo_dia_mes():
    assert ultimo_dia_mes("2026-08-01") == pd.Timestamp("2026-08-31")
    assert ultimo_dia_mes("2026-02-05") == pd.Timestamp("2026-02-28")   # no bisiesto
    assert ultimo_dia_mes("2024-02-10") == pd.Timestamp("2024-02-29")   # bisiesto


# ---------------------------------------------------------------- uniones de potencial/encuestas
def test_union_potencial_clientes_filtra_por_umbral():
    v1 = pd.DataFrame({"Cod Cliente DEX": [1, 2], "predicted": ["2 Bronce", "1 Plata - Oro - Oro+"],
                       "flag_potential": [True, True], "proba_potential": [0.9, 0.5]})
    out = union_potencial_clientes({"lav": v1}, umbral_proba=0.65)
    assert out["cod_cliente_dex"].tolist() == [1] and out["flag_potential"].all()


def test_union_potencial_clientes_valida_columnas():
    with pytest.raises(KeyError, match="Faltan columnas"):
        union_potencial_clientes({"lav": pd.DataFrame({"x": [1]})})


def test_union_potencial_mercado_pdm():
    aba = pd.DataFrame({"cod_cliente_dex": [1, 2], "Potencial": ["Medium", "No Potencial"]})
    lav = pd.DataFrame({"cod_cliente_dex": [3], "Potencial": ["High"]})
    out = union_potencial_mercado_pdm({"aba": aba, "lav": lav})
    assert out.set_index("cod_cliente_dex")["flag_potential"].to_dict() == {1: True, 2: False, 3: True}


def test_combinar_potencial_sin_duplicar():
    a = pd.DataFrame({"cod_cliente_dex": [1], "des_potencial_predic": ["2 Bronce"], "flag_potential": [True]})
    b = pd.DataFrame({"cod_cliente_dex": [1], "des_potencial_predic": ["2 Bronce"], "flag_potential": [True]})
    assert len(combinar_potencial(a, b)) == 1


def test_flag_mercado_potencial():
    ids = pd.DataFrame({"Cod Cliente DEX": [1, 2, 3], "mercado_id": ["M1", "M1", "M2"]})
    pot = pd.DataFrame({"mercado_id": ["M1"], "flag_potential": [True]})
    out = flag_mercado_potencial(ids, pot)
    assert out.set_index("cod_cliente_dex")["flg_mercado_potencial"].to_dict() == {1: True, 2: True, 3: False}


def test_union_encuestas():
    e1 = pd.DataFrame({"external_store_id": [1, 2]})
    e2 = pd.DataFrame({"external_store_id": [2, 3]})
    out = union_encuestas({"bo": e1, "pm": e2})
    assert sorted(out["cod_cliente_dex"]) == [1, 2, 3] and out["flg_encuesta"].all()


# ---------------------------------------------------------------- equivalencia con el notebook original
def _mundo(seed=0, n=60):
    rng = np.random.default_rng(seed)
    bocas = rng.choice(["BO", "AB", "MM", "PA", "GA", "LV", "PM", "GO"], n)
    df_tranx = pd.DataFrame({
        "cod_cliente_dex": np.arange(1, n + 1),
        "des_segmentacion_transaccional": rng.choice(["1 Oro+", "2 Oro", "Sin asignar"], n),
        "cod_boca_salida": bocas,
        "cod_cliente_alicorp_actual": rng.integers(9000, 9010, n),
    })
    df_estrategico = pd.DataFrame({
        "id_interlocutor_origen": np.arange(1, n + 1),
        "segmento": rng.choice(["Clásica", "Estrella", "Golosinera", "Hotel/Casino", "Chifa", "Rara"], n),
        "cod_boca_salida": rng.choice(["AB", "BO"], n),   # para la Regla 2 (GO)
    })
    df_encuestas = union_encuestas({"x": pd.DataFrame({"external_store_id": rng.choice(np.arange(1, n + 1), n // 3, replace=False)})})
    df_potencial_mercado = pd.DataFrame({
        "cod_cliente_dex": np.arange(1, n + 1),
        "flg_mercado_potencial": rng.choice([True, False], n),
    })
    df_potencial = pd.DataFrame({
        "cod_cliente_dex": rng.choice(np.arange(1, n + 1), n // 2, replace=False),
    })
    df_potencial["des_potencial_predic"] = rng.choice(["High", "Low", "Medium", "2 Bronce", "1 Plata - Oro - Oro+"], len(df_potencial))
    df_potencial["flag_potential"] = True
    last_values = pd.DataFrame({"cod_cliente_dex": np.arange(1, n + 1).astype(str),
                                "des_oficina_venta_alicorp": rng.choice(["Lima", "Norte"], n)})
    return df_tranx, df_estrategico, df_encuestas, df_potencial_mercado, df_potencial, last_values


def _notebook_original(df_tranx, df_estrategico, df_encuestas, df_potencial_mercado, df_potencial, last_values, periodo):
    """Copia de las celdas 15-35 del notebook original."""
    df_estrategico = df_estrategico.rename(columns={
        "id_interlocutor_origen": "cod_cliente_dex", "segmento": "des_segmentacion_estrategica",
        "cod_boca_salida": "cod_boca_salida_vs2",
    })
    last_values = last_values.copy()
    last_values["cod_cliente_dex"] = pd.to_numeric(last_values["cod_cliente_dex"], errors="coerce").astype("Int64")
    last_values = last_values[["cod_cliente_dex", "des_oficina_venta_alicorp"]]

    df_final = df_tranx.merge(df_estrategico, on="cod_cliente_dex", how="left")
    df_final = df_final.merge(df_encuestas, on="cod_cliente_dex", how="left")
    df_final["flg_encuesta"] = df_final["flg_encuesta"].fillna(False)
    df_final = df_final.merge(df_potencial_mercado, on="cod_cliente_dex", how="left")
    df_final["flg_mercado_potencial"] = df_final["flg_mercado_potencial"].fillna(False)
    df_final = df_final.merge(df_potencial, on="cod_cliente_dex", how="left")
    df_final["flag_potential"] = df_final["flag_potential"].fillna(False)
    df_final = df_final.merge(last_values, on="cod_cliente_dex", how="left")
    df_final["periodo"] = pd.Timestamp(periodo)
    df_final["fec_corrida"] = ultimo_dia_mes(periodo)

    # Diferencia deliberada 5 del módulo: GO se reemplaza por su boca real ANTES de las reglas por boca
    # (el notebook original lo hacía después de la regla de AB/BO/MM).
    mask_go = df_final["cod_boca_salida"] == "GO"
    df_final.loc[mask_go, "cod_boca_salida"] = df_final.loc[mask_go, "cod_boca_salida_vs2"]
    df_final = df_final.drop(columns=["cod_boca_salida_vs2"], errors="ignore")

    valid_ab_bo_mm = ["Alejada", "Clásica", "Estrella"]
    mask = df_final["cod_boca_salida"].isin(["AB", "BO", "MM"])
    df_final.loc[mask & ~df_final["des_segmentacion_estrategica"].isin(valid_ab_bo_mm), "des_segmentacion_estrategica"] = "Clásica"

    valid_pm = ["Abarrotero", "Confitero", "Lavandero"]
    mask_pm = df_final["cod_boca_salida"] == "PM"
    df_final.loc[mask_pm & ~df_final["des_segmentacion_estrategica"].isin(valid_pm), "des_segmentacion_estrategica"] = "Abarrotero"

    valid_pa = ["Estadio 1", "Estadio 2", "Estadio 3", "Estadio 4"]
    mask_pa = df_final["cod_boca_salida"] == "PA"
    # Diferencia deliberada 4: en PA un segmento no válido/vacío queda "Por definir" (el original dejaba '').
    df_final.loc[mask_pa & ~df_final["des_segmentacion_estrategica"].isin(valid_pa), "des_segmentacion_estrategica"] = "Por definir"

    valid_lv = ["Hotel/Casino", "Lavanderia", "Otros Limpieza"]
    mask_lv = df_final["cod_boca_salida"] == "LV"
    df_final.loc[mask_lv & ~df_final["des_segmentacion_estrategica"].isin(valid_lv), "des_segmentacion_estrategica"] = "Otros Limpieza"

    valid_ga = ["Cebichería", "Chifa", "Clásica", "Comida Casera Criolla", "Otros Gastronomía", "Pollería", "Sanguchería"]
    mask_ga = df_final["cod_boca_salida"] == "GA"
    df_final.loc[mask_ga & ~df_final["des_segmentacion_estrategica"].isin(valid_ga), "des_segmentacion_estrategica"] = "Otros Gastronomía"

    df_final = df_final.drop_duplicates(subset=["cod_cliente_dex"], keep="first")
    df_final = df_final.rename(columns={"cod_cliente_dex": "id_interlocutor_origen"})
    df_final["id_interlocutor_origen"] = df_final["id_interlocutor_origen"].astype(str).str.zfill(10)
    df_final = df_final.rename(columns={"des_segmentacion_transaccional": "des_segmento_transaccional_TMP"})

    def convert_to_boolean(s):
        return s.replace({"FALSO": False, "VERDADERO": True, "False": False, "True": True})

    if df_final["flag_potential"].dtype == "object":
        df_final["flag_potential"] = convert_to_boolean(df_final["flag_potential"])
    if df_final["flg_mercado_potencial"].dtype == "object":
        df_final["flg_mercado_potencial"] = convert_to_boolean(df_final["flg_mercado_potencial"])

    mask_rule1 = (df_final["flg_encuesta"] == False) & (df_final["cod_boca_salida"] != "PM")  # noqa: E712
    df_final.loc[mask_rule1, "flag_potential"] = np.nan
    mask_rule2 = (df_final["flg_encuesta"] == True) & (df_final["cod_boca_salida"] != "PM") & (df_final["flag_potential"] == False)  # noqa: E712
    df_final.loc[mask_rule2, "des_potencial_predic"] = np.nan
    valid_values = ["High", "Low", "Medium"]
    mask_rule3_condition = (df_final["cod_boca_salida"] == "PM") & (df_final["flag_potential"] == True)  # noqa: E712
    mask_rule3_invalid = mask_rule3_condition & (df_final["des_potencial_predic"].isna() | (~df_final["des_potencial_predic"].isin(valid_values)))
    df_final.loc[mask_rule3_invalid, "des_potencial_predic"] = np.nan
    df_final.loc[mask_rule3_invalid, "flag_potential"] = False
    mask_rule4 = (df_final["cod_boca_salida"] == "PM") & (df_final["flag_potential"] == False)  # noqa: E712
    df_final.loc[mask_rule4, "des_potencial_predic"] = np.nan

    df_final = df_final.rename(columns={"des_segmento_transaccional_TMP": "des_segmentacion_transaccional"})
    df_final["des_segmentacion_transaccional"] = df_final["des_segmentacion_transaccional"].replace("Sin asignar", "Por definir")

    cond1 = (df_final["des_segmentacion_transaccional"] == "Por definir") & df_final["cod_boca_salida"].isin(["BO", "AB", "MM", "PM"])
    df_final["des_segmentacion_estrategica"] = np.where(cond1, "Por definir", df_final["des_segmentacion_estrategica"])
    cond2 = (df_final["cod_boca_salida"] == "GA") & (df_final["des_segmentacion_estrategica"] == "Clásica")
    df_final["des_segmentacion_estrategica"] = np.where(cond2, "Otros Gastronomía", df_final["des_segmentacion_estrategica"])
    df_final = df_final[df_final["cod_boca_salida"] != "Sin Asignar"]

    df_final["flag_potential"] = np.where(
        df_final["des_segmentacion_transaccional"] == "Por definir", np.nan, df_final["flag_potential"].fillna(False)
    )
    df_final["flag_potential"] = df_final["flag_potential"].astype("boolean")

    df_final = df_final.rename(columns={
        "id_interlocutor_origen": "cod_interlocutor",
        "des_segmentacion_transaccional": "des_segmento_transaccional",
        "des_segmentacion_estrategica": "des_segmento_estrategica",
        "flag_potential": "flg_potential",
        "des_oficina_venta_alicorp": "des_oficina_venta",
        "fec_corrida": "fec_proceso",
    })
    return df_final[COLUMNAS_GOLD]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_equivale_al_notebook_original(seed):
    fuentes = _mundo(seed=seed)
    ref = _notebook_original(*[f.copy() for f in fuentes], periodo="2026-06-01")
    df_final = armar_df_final(*fuentes, periodo="2026-06-01")
    nuevo = aplicar_reglas_negocio(df_final)

    ref = ref.sort_values("cod_interlocutor").reset_index(drop=True)
    nuevo = nuevo.sort_values("cod_interlocutor").reset_index(drop=True)
    pd.testing.assert_frame_equal(ref, nuevo, check_dtype=False)


def test_fec_proceso_es_ultimo_dia_del_mes():
    fuentes = _mundo()
    out = aplicar_reglas_negocio(armar_df_final(*fuentes, periodo="2026-08-05"))
    assert (out["fec_proceso"] == pd.Timestamp("2026-08-31")).all()
    assert (out["periodo"] == pd.Timestamp("2026-08-05")).all()


def test_sin_duplicados_en_cod_interlocutor():
    fuentes = _mundo()
    out = aplicar_reglas_negocio(armar_df_final(*fuentes, periodo="2026-06-01"))
    assert not out["cod_interlocutor"].duplicated().any()


def test_validar_combinaciones():
    df = pd.DataFrame({"cod_boca_salida": ["AB", "AB", "PA"], "des_segmento_estrategica": ["Clásica", "Rara", ""]})
    invalidas = validar_combinaciones(df)
    assert invalidas["des_segmento_estrategica"].tolist() == ["Rara"]


# ---------------------------------------------------------------- modo incremental
def _fila(cod, boca, transac, estrat, flg, pred, periodo, fec, oficina="Lima"):
    return {
        "periodo": pd.Timestamp(periodo), "cod_interlocutor": cod, "cod_boca_salida": boca,
        "des_segmento_transaccional": transac, "des_segmento_estrategica": estrat,
        "flg_encuesta": True, "flg_mercado_potencial": False, "flg_potential": flg,
        "des_potencial_predic": pred, "des_oficina_venta": oficina, "fec_proceso": pd.Timestamp(fec),
    }


def _gold_base():
    """Periodo anterior (2026-05-01). Cada cliente ejercita un caso."""
    f = lambda *a, **k: _fila(*a, periodo="2026-05-01", fec="2026-05-31", **k)  # noqa: E731
    return pd.DataFrame([
        f("0000000001", "BO", "1 Oro+", "Clásica", True, "2 Bronce"),            # conservado
        f("0000000002", "AB", "Por definir", "Por definir", None, None),         # se resuelve; conserva AB
        f("0000000003", "PA", "2 Oro", "Estadio 1", False, None),                # conserva potencial False
        f("0000000004", "BO", "2 Oro", "Clásica", True, "2 Bronce"),             # cambia de grupo: BO -> PM
        f("0000000005", "BO", "3 Plata", "Clásica", False, None),                # sale (no viene en la alterna)
        f("0000000006", "BO", "3 Plata", "Clásica", False, None),                # alterna lo trae "Por definir"
        f("0000000007", "MM", "3 Plata", "Clásica", True, "2 Bronce"),           # alterna trae BO; transaccional cambia
        f("0000000008", "Sin Asignar", "Por definir", "Por definir", None, None),  # alterna lo trae como BO
    ])


def _nuevo_calculado():
    """La alterna de 2026-06-01 (ya pasada por `aplicar_reglas_negocio`), con las bocas en vs2."""
    f = lambda *a, **k: _fila(*a, periodo="2026-06-01", fec="2026-06-30", **k)  # noqa: E731
    return pd.DataFrame([
        f("0000000001", "BO", "1 Oro+", "Estrella", True, "1 Plata - Oro - Oro+"),
        f("0000000002", "BO", "2 Oro", "Estrella", True, "1 Plata - Oro - Oro+"),
        f("0000000003", "PA", "2 Oro", "Estadio 2", True, "High"),
        f("0000000004", "PM", "2 Oro", "Abarrotero", True, "High", oficina="Sur"),
        f("0000000006", "BO", "Por definir", "Por definir", None, None),
        f("0000000007", "BO", "2 Oro", "Estrella", True, "1 Plata - Oro - Oro+"),
        f("0000000008", "BO", "2 Oro", "Clásica", True, "2 Bronce"),
        f("0000000009", "GA", "1 Oro+", "Chifa", True, "2 Bronce"),               # nuevo
        f("0000000010", "BO", "Por definir", "Por definir", None, None),          # nuevo "Por definir"
    ])


def _inc():
    return completar_incremental(_nuevo_calculado(), _gold_base()).set_index("cod_interlocutor")


def test_grupo_boca_colapsa_bo_mm_ab_go_y_sin_asignar():
    out = grupo_boca(pd.Series(["BO", "mm", " AB ", "GO", "Sin Asignar", "PM", "PA", None]))
    assert out.tolist()[:7] == ["BO", "BO", "BO", "BO", "BO", "PM", "PA"] and pd.isna(out.iloc[7])


def test_incremental_agrega_nuevos_y_no_arrastra_a_quien_no_viene_en_la_alterna():
    out = _inc()
    assert {"0000000009", "0000000010"} <= set(out.index)
    assert "0000000005" not in out.index                      # salió: no se arrastra su fila
    assert set(out.index) == set(_nuevo_calculado()["cod_interlocutor"])


def test_incremental_nuevo_por_definir_queda_sin_estrategica_ni_potencial():
    fila = _inc().loc["0000000010"]
    assert fila["des_segmento_estrategica"] == "Por definir"
    assert pd.isna(fila["flg_potential"]) and pd.isna(fila["des_potencial_predic"])


def test_incremental_conserva_estrategica_y_potencial_de_quien_ya_tenia_segmento():
    fila = _inc().loc["0000000001"]
    assert fila["des_segmento_estrategica"] == "Clásica"          # la alterna decía "Estrella"
    assert fila["flg_potential"] == True and fila["des_potencial_predic"] == "2 Bronce"  # noqa: E712


def test_incremental_conserva_potencial_false_aunque_el_fresco_diga_otra_cosa():
    fila = _inc().loc["0000000003"]
    assert fila["flg_potential"] == False and pd.isna(fila["des_potencial_predic"])  # noqa: E712
    assert fila["des_segmento_estrategica"] == "Estadio 1"


def test_incremental_completa_lo_que_estaba_por_definir_y_conserva_ab_de_gold():
    fila = _inc().loc["0000000002"]
    assert fila["des_segmento_transaccional"] == "2 Oro"
    assert fila["des_segmento_estrategica"] == "Estrella"
    assert fila["flg_potential"] == True and fila["des_potencial_predic"] == "1 Plata - Oro - Oro+"  # noqa: E712
    assert fila["cod_boca_salida"] == "AB"                         # la alterna colapsa AB -> BO


def test_incremental_el_transaccional_lo_manda_la_alterna():
    fila = _inc().loc["0000000007"]
    assert fila["des_segmento_transaccional"] == "2 Oro"          # gold tenía "3 Plata"
    assert fila["cod_boca_salida"] == "MM"                         # misma boca en vs2: se conserva MM
    assert fila["des_segmento_estrategica"] == "Clásica" and fila["des_potencial_predic"] == "2 Bronce"


def test_incremental_cambio_de_grupo_de_boca_toma_todo_lo_fresco():
    fila = _inc().loc["0000000004"]
    assert fila["cod_boca_salida"] == "PM"
    assert fila["des_segmento_transaccional"] == "2 Oro"
    assert fila["des_segmento_estrategica"] == "Abarrotero"
    assert fila["flg_potential"] == True and fila["des_potencial_predic"] == "High"  # noqa: E712
    assert fila["des_oficina_venta"] == "Sur"


def test_incremental_no_conserva_la_boca_de_gold_si_no_es_granular():
    """'Sin Asignar' en gold no se conserva: si la alterna ya trae una boca, esa es la boca."""
    assert _inc().loc["0000000008", "cod_boca_salida"] == "BO"


def test_incremental_un_cliente_con_segmento_no_regresa_a_por_definir():
    fila = _inc().loc["0000000006"]
    assert fila["des_segmento_transaccional"] == "3 Plata"
    assert fila["des_segmento_estrategica"] == "Clásica"
    assert fila["flg_potential"] == False                           # noqa: E712


def test_incremental_periodo_y_fec_proceso_son_los_de_la_alterna():
    out = _inc()
    assert (out["periodo"] == pd.Timestamp("2026-06-01")).all()
    assert (out["fec_proceso"] == pd.Timestamp("2026-06-30")).all()


def test_incremental_devuelve_las_columnas_de_gold_y_no_muta_las_entradas():
    nuevo, gold = _nuevo_calculado(), _gold_base()
    nuevo0, gold0 = nuevo.copy(), gold.copy()
    out = completar_incremental(nuevo, gold)
    assert list(out.columns) == COLUMNAS_GOLD
    pd.testing.assert_frame_equal(nuevo, nuevo0)
    pd.testing.assert_frame_equal(gold, gold0)


def test_incremental_potencial_false_nunca_tiene_etiqueta():
    out = _inc()
    assert out.loc[out["flg_potential"].eq(False).fillna(False).astype(bool), "des_potencial_predic"].isna().all()


def test_incremental_valida_columnas():
    with pytest.raises(KeyError, match="Faltan columnas"):
        completar_incremental(_nuevo_calculado(), pd.DataFrame({"cod_interlocutor": ["1"]}))


def test_incremental_una_boca_explicita_ab_o_mm_en_la_alterna_manda():
    """Si el trax algún día emite AB/MM explícitos, esa boca reemplaza a la de gold."""
    nuevo = _nuevo_calculado()
    nuevo.loc[nuevo["cod_interlocutor"] == "0000000007", "cod_boca_salida"] = "AB"
    assert completar_incremental(nuevo, _gold_base()).set_index("cod_interlocutor").loc["0000000007", "cod_boca_salida"] == "AB"


# ---------------------------------------------------------------- salidas, regresiones y validación
def test_detectar_salidas():
    salidas = detectar_salidas(_nuevo_calculado(), _gold_base())
    assert salidas["cod_interlocutor"].tolist() == ["0000000005"]


def test_clasificar_salidas_separa_las_esperadas_de_las_que_hay_que_revisar():
    salidas = pd.DataFrame({"cod_interlocutor": ["0000000005", "0000000011", "0000000012"],
                            "cod_boca_salida": ["BO"] * 3, "des_segmento_transaccional": ["2 Oro"] * 3})
    hist = pd.DataFrame({"cod_cliente_dex": [5, 11], "cod_grupo_precio_dex": ["ot", "BO"]})
    out = clasificar_salidas(salidas, hist).set_index("cod_interlocutor")
    assert out["motivo"].to_dict() == {"0000000005": "no_segmentable", "0000000011": "segmentable",
                                       "0000000012": "sin_historico"}
    assert out["esperada"].to_dict() == {"0000000005": True, "0000000011": False, "0000000012": False}


def test_detectar_regresiones():
    reg = detectar_regresiones(_nuevo_calculado(), _gold_base())
    assert reg["cod_interlocutor"].tolist() == ["0000000006"]
    assert reg["des_segmento_transaccional_ant"].tolist() == ["3 Plata"]


def test_validar_incremental_cuenta_cada_grupo():
    salidas = clasificar_salidas(
        detectar_salidas(_nuevo_calculado(), _gold_base()),
        pd.DataFrame({"cod_cliente_dex": [5], "cod_grupo_precio_dex": ["MA"]}),
    )
    v = validar_incremental(_nuevo_calculado(), _gold_base(), salidas)
    assert v == {
        "clientes_periodo_anterior": 8, "clientes_periodo_actual": 9, "nuevos": 2, "salidas": 1,
        "salidas_esperadas": 1, "salidas_a_revisar": 0,
        "cambio_grupo_boca": 1,            # 4: BO -> PM
        "boca_granular_conservada": 2,     # 2 (AB) y 7 (MM)
        "por_definir_resueltos": 2,        # 2 y 8 (8: "Sin Asignar" -> BO es el mismo grupo)
        "siguen_por_definir": 0,
        "conservados": 4,                  # 1, 3, 6 y 7
        "transaccional_cambia": 1,         # 7: "3 Plata" -> "2 Oro"
        "regresan_a_por_definir": 1,       # 6
    }


def test_validar_incremental_sin_clasificar_salidas_no_incluye_esos_campos():
    v = validar_incremental(_nuevo_calculado(), _gold_base())
    assert "salidas_esperadas" not in v and v["salidas"] == 1


# ---------------------------------------------------------------- modos y carga
def test_preparar_carga_periodo_modos():
    nuevo = _nuevo_calculado()
    assert len(preparar_carga_periodo(nuevo, "completo")) == len(nuevo)
    out_inc = preparar_carga_periodo(nuevo, "incremental", gold_periodo_anterior=_gold_base())
    assert len(out_inc) == len(nuevo)                              # nadie se arrastra; solo la alterna
    with pytest.raises(ValueError, match="necesita gold_periodo_anterior"):
        preparar_carga_periodo(nuevo, "incremental")
    with pytest.raises(ValueError, match="modo debe ser"):
        preparar_carga_periodo(nuevo, "otro")


def test_construir_insert_gold_es_un_insert_select_sin_merge():
    sql = construir_insert_gold("proyecto.gold.tabla", "proyecto.dev.staging")
    assert sql.startswith("INSERT INTO `proyecto.gold.tabla`") and "FROM `proyecto.dev.staging`" in sql
    assert "MERGE" not in sql and "CAST(periodo AS DATE) AS periodo" in sql


class _ClienteFalso:
    def __init__(self, filas_en_periodo):
        self.filas = filas_en_periodo

    def query(self, sql):
        class _R:
            def __init__(s, n): s.n = n
            def to_dataframe(s): return pd.DataFrame({"n": [s.n]})
        return _R(self.filas)


def test_cargar_periodo_gold_se_niega_si_el_periodo_ya_esta_cargado():
    with pytest.raises(ValueError, match="Ya existe al menos una fila"):
        cargar_periodo_gold(_nuevo_calculado(), _ClienteFalso(5), "p.d.gold", "p.d.staging")


def test_cargar_periodo_gold_se_niega_con_mas_de_un_periodo():
    df = pd.concat([_nuevo_calculado(), _gold_base()], ignore_index=True)
    with pytest.raises(ValueError, match="un solo periodo"):
        cargar_periodo_gold(df, _ClienteFalso(0), "p.d.gold", "p.d.staging")


# ---------------------------------------------------------------- PA: estratégica exclusiva de la tabla de estadios
def test_preparar_segmento_pa_normaliza_columnas_y_estadios():
    df = pd.DataFrame({"cod_cliente_dex": [1, 2, 3, 4], "estadio": ["Estadio 1", "estadio 2", 3, "4.0"]})
    out = preparar_segmento_pa(df)
    assert out.set_index("cod_interlocutor")["des_segmento_estrategica"].to_dict() == {
        "0000000001": "Estadio 1", "0000000002": "Estadio 2", "0000000003": "Estadio 3", "0000000004": "Estadio 4"}


def test_preparar_segmento_pa_descarta_lo_no_interpretable_con_aviso():
    df = pd.DataFrame({"cod_interlocutor": ["1", "2", None], "des_segmento_estrategica": ["Estadio 1", "Estadio 9", "Estadio 2"]})
    with pytest.warns(UserWarning, match="se descartan"):
        out = preparar_segmento_pa(df)
    assert out["cod_interlocutor"].tolist() == ["0000000001"]


def test_preparar_segmento_pa_rechaza_dos_estadios_para_un_cliente():
    df = pd.DataFrame({"cod_cliente_dex": [1, 1], "estadio": ["Estadio 1", "Estadio 2"]})
    with pytest.raises(ValueError, match="más de un estadio"):
        preparar_segmento_pa(df)


def test_preparar_segmento_pa_avisa_las_columnas_disponibles_si_no_las_encuentra():
    with pytest.raises(KeyError, match="Columnas disponibles"):
        preparar_segmento_pa(pd.DataFrame({"x": [1], "y": ["Estadio 1"]}))
    out = preparar_segmento_pa(pd.DataFrame({"x": [1], "y": ["Estadio 1"]}), col_cliente="x", col_segmento="y")
    assert len(out) == 1


def _pa_fresco():
    f = lambda *a, **k: _fila(*a, periodo="2026-06-01", fec="2026-06-30", **k)  # noqa: E731
    return pd.DataFrame([
        f("0000000001", "PA", "2 Oro", "Estadio 4", True, "High"),       # en tabla (Estadio 2)
        f("0000000002", "PA", "2 Oro", "Estadio 1", True, "High"),       # NO está en la tabla
        f("0000000003", "BO", "2 Oro", "Clásica", True, "High"),         # no es PA
        f("0000000004", "PA", "Por definir", "Por definir", None, None),  # en tabla (Estadio 3), transaccional pendiente
        f("0000000005", "PA", "2 Oro", "Por definir", True, "High"),     # en tabla (Estadio 4); en gold estaba "Por definir"
        f("0000000006", "PA", "Por definir", "Por definir", None, None),  # NUEVO, transaccional pendiente, en tabla (Estadio 2)
    ])


def _tabla_pa():
    return pd.DataFrame({"cod_interlocutor": ["0000000001", "0000000003", "0000000004", "0000000005", "0000000006"],
                         "des_segmento_estrategica": ["Estadio 2", "Estadio 1", "Estadio 3", "Estadio 4", "Estadio 2"]})


def test_aplicar_segmento_pa_solo_toca_pa_y_usa_unicamente_la_tabla():
    out, resumen = aplicar_segmento_pa(_pa_fresco(), _tabla_pa())
    out = out.set_index("cod_interlocutor")
    assert out.loc["0000000001", "des_segmento_estrategica"] == "Estadio 2"      # tabla, no la de la alterna
    assert out.loc["0000000002", "des_segmento_estrategica"] == "Por definir"    # PA sin tabla
    assert out.loc["0000000003", "des_segmento_estrategica"] == "Clásica"        # BO: intacto, aunque esté en la tabla
    assert resumen == {"pa_clientes": 5, "pa_con_tabla": 4, "pa_sin_tabla": 1}


def test_aplicar_segmento_pa_no_muta_la_entrada():
    fresco = _pa_fresco()
    antes = fresco.copy()
    aplicar_segmento_pa(fresco, _tabla_pa())
    pd.testing.assert_frame_equal(fresco, antes)


def _pa_gold():
    f = lambda *a, **k: _fila(*a, periodo="2026-05-01", fec="2026-05-31", **k)  # noqa: E731
    return pd.DataFrame([
        f("0000000001", "PA", "2 Oro", "Estadio 1", True, "High"),       # gold Estadio 1, tabla Estadio 2
        f("0000000002", "PA", "2 Oro", "Estadio 3", True, "High"),       # gold Estadio 3, no está en la tabla
        f("0000000003", "BO", "2 Oro", "Clásica", True, "High"),
        f("0000000004", "PA", "2 Oro", "Estadio 2", True, "High"),       # gold Estadio 2, tabla Estadio 3, transaccional hoy pendiente
        f("0000000005", "PA", "Por definir", "Por definir", None, None),  # estaba "Por definir"
    ])


def _inc_pa():
    fresco, resumen = aplicar_segmento_pa(_pa_fresco(), _tabla_pa())
    return completar_incremental(fresco, _pa_gold(), segmento_pa=_tabla_pa()).set_index("cod_interlocutor")


def test_incremental_pa_en_tabla_toma_siempre_el_estadio_de_la_tabla():
    out = _inc_pa()
    assert out.loc["0000000001", "des_segmento_estrategica"] == "Estadio 2"      # gold tenía Estadio 1
    assert out.loc["0000000001", "des_potencial_predic"] == "High"               # el potencial sigue su regla (conserva)


def test_incremental_pa_sin_tabla_conserva_su_estadio_de_gold():
    assert _inc_pa().loc["0000000002", "des_segmento_estrategica"] == "Estadio 3"   # no regresa a "Por definir"


def test_incremental_pa_con_segmento_en_gold_y_transaccional_pendiente_toma_el_estadio_y_conserva_lo_demas():
    """La regresión del transaccional está bloqueada: conserva su fila anterior; solo el estadio viene de la tabla."""
    fila = _inc_pa().loc["0000000004"]
    assert fila["des_segmento_estrategica"] == "Estadio 3"                         # gold tenía Estadio 2
    assert fila["des_segmento_transaccional"] == "2 Oro"
    assert fila["flg_potential"] == True and fila["des_potencial_predic"] == "High"  # noqa: E712


def test_incremental_pa_nuevo_con_transaccional_pendiente_toma_el_estadio_sin_potencial():
    fila = _inc_pa().loc["0000000006"]
    assert fila["des_segmento_estrategica"] == "Estadio 2"                         # la tabla manda aunque el transaccional sea pendiente
    assert pd.isna(fila["flg_potential"]) and pd.isna(fila["des_potencial_predic"])


def test_incremental_pa_que_estaba_por_definir_se_resuelve_con_la_tabla_y_toma_potencial_fresco():
    fila = _inc_pa().loc["0000000005"]
    assert fila["des_segmento_estrategica"] == "Estadio 4" and fila["des_segmento_transaccional"] == "2 Oro"
    assert fila["flg_potential"] == True and fila["des_potencial_predic"] == "High"  # noqa: E712


def test_incremental_sin_tabla_de_pa_se_comporta_como_antes():
    fresco = _pa_fresco()
    a = completar_incremental(fresco, _pa_gold())
    b = completar_incremental(fresco, _pa_gold(), segmento_pa=None)
    pd.testing.assert_frame_equal(a, b)


def test_validar_incremental_cuenta_los_cambios_de_pa():
    fresco, _ = aplicar_segmento_pa(_pa_fresco(), _tabla_pa())
    v = validar_incremental(fresco, _pa_gold(), segmento_pa=_tabla_pa())
    assert v["pa_estadio_cambia"] == 2          # 1 (1 -> 2) y 4 (2 -> 3)
    assert v["pa_por_definir_resueltos"] == 1   # 5
    assert "pa_estadio_cambia" not in validar_incremental(fresco, _pa_gold())


def test_preparar_carga_periodo_pasa_la_tabla_de_pa_al_incremental():
    fresco, _ = aplicar_segmento_pa(_pa_fresco(), _tabla_pa())
    out = preparar_carga_periodo(fresco, "incremental", gold_periodo_anterior=_pa_gold(), segmento_pa=_tabla_pa())
    assert out.set_index("cod_interlocutor").loc["0000000001", "des_segmento_estrategica"] == "Estadio 2"
