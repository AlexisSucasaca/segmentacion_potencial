"""Tests de src/segmentacion_gold.py (datos sintéticos, sin archivos tuyos).

`test_equivale_al_notebook_original` reproduce, tal cual, las celdas 15-35 del notebook
original (sin la consulta a BigQuery) y compara contra `armar_df_final` + `aplicar_reglas_negocio`.
Los demás tests cubren cada pieza por separado y los dos modos de ejecución.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.segmentacion_gold import (
    COLUMNAS_GOLD, aplicar_reglas_negocio, armar_df_final, build_merge_query, combinar_potencial,
    filtrar_incremental, flag_mercado_potencial, preparar_para_upsert, ultimo_dia_mes,
    union_encuestas, union_potencial_clientes, union_potencial_mercado_pdm, validar_combinaciones,
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

    valid_ab_bo_mm = ["Alejada", "Clásica", "Estrella"]
    mask = df_final["cod_boca_salida"].isin(["AB", "BO", "MM"])
    df_final.loc[mask & ~df_final["des_segmentacion_estrategica"].isin(valid_ab_bo_mm), "des_segmentacion_estrategica"] = "Clásica"

    mask_go = df_final["cod_boca_salida"] == "GO"
    df_final.loc[mask_go, "cod_boca_salida"] = df_final.loc[mask_go, "cod_boca_salida_vs2"]
    df_final = df_final.drop(columns=["cod_boca_salida_vs2"], errors="ignore")

    valid_pm = ["Abarrotero", "Confitero", "Lavandero"]
    mask_pm = df_final["cod_boca_salida"] == "PM"
    df_final.loc[mask_pm & ~df_final["des_segmentacion_estrategica"].isin(valid_pm), "des_segmentacion_estrategica"] = "Abarrotero"

    valid_pa = ["Estadio 1", "Estadio 2", "Estadio 3", "Estadio 4"]
    mask_pa = df_final["cod_boca_salida"] == "PA"
    df_final.loc[mask_pa & ~df_final["des_segmentacion_estrategica"].isin(valid_pa), "des_segmentacion_estrategica"] = ""

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
def _gold_base():
    return pd.DataFrame({
        "periodo": pd.Timestamp("2026-05-01"), "cod_interlocutor": ["0000000001", "0000000002", "0000000003"],
        "cod_boca_salida": ["BO", "BO", "PA"], "des_segmento_transaccional": ["1 Oro+", "2 Oro", "2 Oro"],
        "des_segmento_estrategica": ["Clásica", None, "Estadio 1"],
        "flg_encuesta": [True, True, False], "flg_mercado_potencial": [False, False, False],
        "flg_potential": [True, None, False], "des_potencial_predic": ["2 Bronce", None, None],
        "des_oficina_venta": ["Lima", "Lima", "Norte"], "fec_proceso": pd.Timestamp("2026-05-31"),
    })


def _nuevo_calculado():
    return pd.DataFrame({
        "periodo": pd.Timestamp("2026-06-01"),
        "cod_interlocutor": ["0000000001", "0000000002", "0000000003", "0000000004"],
        "cod_boca_salida": ["BO", "BO", "PA", "GA"],
        "des_segmento_transaccional": ["1 Oro+", "2 Oro", "2 Oro", "1 Oro+"],
        "des_segmento_estrategica": ["Clásica", "Estrella", "Estadio 1", "Chifa"],
        "flg_encuesta": [True, True, True, True], "flg_mercado_potencial": [False, False, False, False],
        "flg_potential": [True, True, False, True], "des_potencial_predic": ["2 Bronce", "1 Plata - Oro - Oro+", None, "2 Bronce"],
        "des_oficina_venta": ["Lima", "Callao", "Norte", "Sur"], "fec_proceso": pd.Timestamp("2026-06-30"),
    })


def test_filtrar_incremental_agrega_clientes_nuevos():
    out = filtrar_incremental(_nuevo_calculado(), _gold_base())
    assert "0000000004" in out["cod_interlocutor"].values


def test_filtrar_incremental_devuelve_toda_la_tabla_mas_los_nuevos():
    """Opción B: la salida siempre trae TODOS los clientes de gold + los nuevos (no solo un delta)."""
    out = filtrar_incremental(_nuevo_calculado(), _gold_base())
    assert set(out["cod_interlocutor"]) == {"0000000001", "0000000002", "0000000003", "0000000004"}


def test_filtrar_incremental_completa_solo_lo_vacio():
    out = filtrar_incremental(_nuevo_calculado(), _gold_base()).set_index("cod_interlocutor")
    # cliente 2: no tenía des_segmento_estrategica ni flg_potential/des_potencial_predic -> se completan
    fila2 = out.loc["0000000002"]
    assert fila2["des_segmento_estrategica"] == "Estrella"
    assert fila2["flg_potential"] == True and fila2["des_potencial_predic"] == "1 Plata - Oro - Oro+"  # noqa: E712
    # cliente 1: ya tenía todo -> se conserva igual (no se pisa con el valor "nuevo" recalculado)
    assert out.loc["0000000001", "des_segmento_estrategica"] == "Clásica"
    assert out.loc["0000000001", "flg_potential"] == True  # noqa: E712


def test_filtrar_incremental_nunca_cambia_boca_salida_de_existentes():
    """cod_boca_salida de un cliente existente sale siempre de gold, nunca del cálculo
    fresco (acá con boca coincidente, para que el cliente sí reciba un relleno)."""
    out = filtrar_incremental(_nuevo_calculado(), _gold_base()).set_index("cod_interlocutor")
    assert out.loc["0000000002", "cod_boca_salida"] == "BO"


def test_filtrar_incremental_emparejamiento_por_boca_salida():
    """Si el cálculo fresco trae al cliente con otra boca, no se rellena nada para ese
    cliente (queda como estaba en gold, incompleto): la boca de gold no cambia, y el valor
    fresco fue calculado para una boca distinta — mezclarlos crearía una combinación
    inconsistente."""
    nuevo = _nuevo_calculado()
    nuevo.loc[nuevo["cod_interlocutor"] == "0000000002", "cod_boca_salida"] = "AB"
    out = filtrar_incremental(nuevo, _gold_base()).set_index("cod_interlocutor")
    assert pd.isna(out.loc["0000000002", "des_segmento_estrategica"])   # sigue como en gold: sin dato
    assert pd.isna(out.loc["0000000002", "flg_potential"])
    assert out.loc["0000000002", "cod_boca_salida"] == "BO"


def test_filtrar_incremental_nunca_produce_combinacion_invalida_en_filas_completadas():
    """Toda fila que SÍ quedó con segmento asignado (no NaN) es una combinación válida.
    Una fila que sigue incompleta (nunca se pudo rellenar) puede seguir "sin asignar" —
    eso no es una combinación inválida nueva, es la misma incompletitud que ya traía gold."""
    from src.segmentacion_gold import validar_combinaciones

    nuevo = _nuevo_calculado()
    nuevo.loc[nuevo["cod_interlocutor"] == "0000000002", "cod_boca_salida"] = "AB"
    out = filtrar_incremental(nuevo, _gold_base())
    con_segmento = out[out["des_segmento_estrategica"].notna()]
    assert validar_combinaciones(con_segmento).empty


def test_filtrar_incremental_periodo_y_fec_proceso_de_toda_la_tabla():
    """Opción B: periodo/fec_proceso se pisan en TODA la tabla, incluidos los clientes que
    ya tenían todo completo (no solo en los que recibieron un dato nuevo)."""
    out = filtrar_incremental(_nuevo_calculado(), _gold_base())
    assert (out["periodo"] == pd.Timestamp("2026-06-01")).all()
    assert (out["fec_proceso"] == pd.Timestamp("2026-06-30")).all()
    # incluido el cliente 1, que en gold tenía periodo 2026-05-01 y ya estaba completo
    assert out.set_index("cod_interlocutor").loc["0000000001", "periodo"] == pd.Timestamp("2026-06-01")


def test_filtrar_incremental_no_pisa_datos_ya_existentes_con_el_recalculo():
    """Aunque el cálculo fresco traiga un valor distinto para un cliente que ya tenía datos,
    se conserva el de gold — el modo incremental solo rellena huecos, nunca reemplaza."""
    nuevo = _nuevo_calculado()
    nuevo.loc[nuevo["cod_interlocutor"] == "0000000001", "des_segmento_estrategica"] = "Estrella"  # distinto a gold ("Clásica")
    out = filtrar_incremental(nuevo, _gold_base()).set_index("cod_interlocutor")
    assert out.loc["0000000001", "des_segmento_estrategica"] == "Clásica"   # el de gold, no "Estrella"


def test_filtrar_incremental_valida_columnas():
    with pytest.raises(KeyError, match="Faltan columnas"):
        filtrar_incremental(_nuevo_calculado(), pd.DataFrame({"x": [1]}))


def test_preparar_para_upsert_modos():
    nuevo = _nuevo_calculado()
    assert len(preparar_para_upsert(nuevo, "completo")) == len(nuevo)
    out_inc = preparar_para_upsert(nuevo, "incremental", df_gold_actual=_gold_base())
    assert len(out_inc) == 4   # 3 de gold (conservados/completados) + 1 nuevo
    with pytest.raises(ValueError, match="necesita df_gold_actual"):
        preparar_para_upsert(nuevo, "incremental")
    with pytest.raises(ValueError, match="modo debe ser"):
        preparar_para_upsert(nuevo, "otro")

# ---------------------------------------------------------------- MERGE SQL
def test_build_merge_query_contiene_las_clausulas():
    sql = build_merge_query("proyecto.dev.staging", "proyecto.gold.tabla")
    assert "MERGE `proyecto.gold.tabla`" in sql
    assert "USING `proyecto.dev.staging`" in sql
    assert "ON destino.cod_interlocutor = nuevo.cod_interlocutor" in sql
    assert "WHEN MATCHED THEN" in sql and "WHEN NOT MATCHED THEN" in sql
    assert "INSERT (periodo, cod_interlocutor" in sql
    assert "cod_interlocutor = nuevo.cod_interlocutor" not in sql.split("UPDATE SET")[1].split("WHEN NOT")[0]


# ---------------------------------------------------------------- upsert_gold
def test_upsert_gold_convierte_periodo_y_fec_proceso_a_date(monkeypatch):
    """BigQuery rechaza DATETIME en una columna DATE: periodo/fec_proceso deben llegar a
    to_gbq como datetime.date, no como pd.Timestamp."""
    import datetime
    import sys
    import types

    from src.segmentacion_gold import upsert_gold

    capturado = {}

    def fake_to_gbq(df, destination_table, project_id, if_exists):
        capturado["df"] = df.copy()

    fake_module = types.ModuleType("pandas_gbq")
    fake_module.to_gbq = fake_to_gbq
    monkeypatch.setitem(sys.modules, "pandas_gbq", fake_module)

    class FakeQueryResult:
        def result(self):
            return None

    class FakeClient:
        def query(self, sql):
            return FakeQueryResult()

    df = _nuevo_calculado()  # ya tiene periodo/fec_proceso como pd.Timestamp
    upsert_gold(df, FakeClient(), "proyecto.dev.staging", "proyecto.gold.tabla", "proyecto")

    subido = capturado["df"]
    assert all(isinstance(v, datetime.date) and not isinstance(v, datetime.datetime) for v in subido["periodo"])
    assert all(isinstance(v, datetime.date) and not isinstance(v, datetime.datetime) for v in subido["fec_proceso"])
    assert list(subido.columns) == COLUMNAS_GOLD
