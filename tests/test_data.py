"""Tests de src/data.py.

Compara, línea por línea y normalizando espacios, la query que genera `build_query_clientes`
contra la `query_clientes` real de cada notebook (una copia literal en
`tests/legacy/notebook_queries.py`). La única diferencia esperada es el LEFT JOIN a
`equifax_encuesta_cliente`, que se omite a propósito: sus columnas no se seleccionan, así
que no cambia el resultado (ver el docstring de `src/data.py`).
"""
from __future__ import annotations

import re

import pytest

from src.data import (
    QUERY_CONFIGS, build_query_clientes, build_query_puntos, query_clientes_de,
)
from tests.legacy.notebook_queries import QUERIES_ORIGINALES


def _normalizar(sql: str) -> str:
    sql = re.sub(
        r"left\s+join\s+`acpe-dev-uc-ml\.dev\.equifax_encuesta_cliente`\s+f\s+"
        r"on\s+a\.cod_cliente_dex\s*=\s*lpad\(cast\(f\.id_interlocutor_origen\s+as\s+string\),\s*10,\s*'0'\)",
        "", sql, flags=re.I,
    )
    sql = re.sub(r"\s+", "", sql).lower().rstrip(";")
    sql = sql.replace("b.cod_interlocutor", "cod_interlocutor")
    return sql


@pytest.mark.parametrize("vertical", sorted(QUERY_CONFIGS))
def test_query_coincide_con_el_notebook_original(vertical):
    mia = _normalizar(query_clientes_de(vertical))
    original = _normalizar(QUERIES_ORIGINALES[vertical])
    assert mia == original


def test_todas_las_verticales_tienen_config():
    from src.config import load_config
    assert set(QUERY_CONFIGS) == set(load_config())


def test_grupo_precio_y_segmento_por_vertical():
    """Deja constancia explícita de los dos hallazgos no obvios al revisar los notebooks."""
    assert QUERY_CONFIGS["hoteles"].grupo_precio == "LV"          # no tiene grupo propio
    assert QUERY_CONFIGS["hoteles"].filtro_segmento == ["Hotel/Casino"]
    assert QUERY_CONFIGS["panaderia"].filtro_segmento is None      # no filtra por segmento
    assert "estadio 1" in QUERY_CONFIGS["panaderia"].columna_segmento.lower()


def test_flg_cliente_perfecto_solo_bodegas_y_pm():
    con_flag = {n for n, c in QUERY_CONFIGS.items() if c.incluir_flg_cliente_perfecto}
    assert con_flag == {"bodegas", "pm_mercado"}
    assert "flg_cliente_perfecto" in query_clientes_de("bodegas")
    assert "flg_cliente_perfecto" not in query_clientes_de("lavanderias")


def test_sql_lista_escapa_comillas():
    from src.data import QueryConfig
    cfg = QueryConfig(grupo_precio="XX", filtro_segmento=["O'Brien"])
    assert "O''Brien" in build_query_clientes(cfg)


def test_periodo_es_parametrizable():
    from src.data import QueryConfig
    cfg = QueryConfig(grupo_precio="LV", filtro_segmento=["Lavanderia"], periodo_segmentacion="2027-01-01")
    assert "2027-01-01" in build_query_clientes(cfg)
    assert "2026-07-01" not in build_query_clientes(cfg).split("ventas AS")[0]


def test_query_clientes_de_vertical_desconocida():
    with pytest.raises(KeyError, match="no está en QUERY_CONFIGS"):
        query_clientes_de("no_existe")


def test_query_puntos_igual_en_todas_las_verticales():
    """La query de puntos comerciales no depende de la vertical (un único WHERE geográfico)."""
    q = build_query_puntos()
    assert "acpe-dev-uc-sandbox-aa.scraping.puntos_comerciales" in q
    assert "-18.4" in q and "-0.03" in q and "-81.4" in q and "-68.7" in q
