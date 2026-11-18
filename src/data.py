"""Query de clientes y puntos comerciales, parametrizada por vertical.

Reproduce la query de BigQuery de los seis notebooks. Solo cambian, por vertical:
  - `grupo_precio` (cod_grupo_precio_dex): LV, GA, PA, BO, PM. Hoteles usa el mismo que
    lavanderías ('LV') — no tiene grupo propio, se separa por `filtro_segmento`.
  - `filtro_segmento`: la lista de valores de `des_segmento_estrategica` a incluir con un
    `IN (...)`, o None para no filtrar (panadería no filtra: agrupa después con un CASE WHEN
    en la propia columna, ver `SQL_SEGMENTO_PANADERIA`).
  - `incluir_flg_cliente_perfecto`: solo bodegas y PM seleccionan esa columna.

El periodo de segmentación ('2026-08-01') y el rango de ventas ('2026-02-01' a '2026-08-01')
quedan fijos, como en los notebooks; pásalos si necesitas otro periodo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

PERIODO_SEGMENTACION_DEFAULT = "2026-09-01"
VENTAS_DESDE_DEFAULT = "2026-03-01"
VENTAS_HASTA_DEFAULT = "2026-09-01"

# El CASE WHEN que usa panadería para agrupar des_segmento_estrategica en vez de filtrarla.
SQL_SEGMENTO_PANADERIA = (
    "CASE WHEN b.des_segmento_estrategica IN ('Estadio 1', 'Estadio 2', 'Estadio 3', 'Estadio 4') "
    "THEN b.des_segmento_estrategica ELSE '?' END"
)


@dataclass(frozen=True)
class QueryConfig:
    grupo_precio: str
    filtro_segmento: Optional[Sequence[str]] = None   # None = sin filtro (panadería)
    columna_segmento: str = "b.des_segmento_estrategica"  # panadería pasa SQL_SEGMENTO_PANADERIA
    incluir_flg_cliente_perfecto: bool = False
    periodo_segmentacion: str = PERIODO_SEGMENTACION_DEFAULT
    ventas_desde: str = VENTAS_DESDE_DEFAULT
    ventas_hasta: str = VENTAS_HASTA_DEFAULT


def _sql_lista(valores: Sequence[str]) -> str:
    escapados = [v.replace("'", "''") for v in valores]
    return "(" + ", ".join(f"'{v}'" for v in escapados) + ")"


def build_query_clientes(cfg: QueryConfig) -> str:
    """La query `query_clientes` de los notebooks, con los parámetros de `cfg`."""
    flg = ",\n        a.flg_cliente_perfecto" if cfg.incluir_flg_cliente_perfecto else ""
    filtro = (
        f"\n        AND b.des_segmento_estrategica IN {_sql_lista(cfg.filtro_segmento)}"
        if cfg.filtro_segmento is not None else ""
    )
    return f"""
WITH clientes AS (
    SELECT DISTINCT
        a.cod_cliente_dex,
        a.tip_documento,
        a.des_provincia,
        a.des_ciudad,
        a.des_distrito,
        a.val_coordenada_x,
        a.val_coordenada_y,
        {cfg.columna_segmento} as des_segmento_estrategico,
        b.des_segmento_transaccional AS segmento_trx,
        e.val_score_urbanicidad,
        g.cod_cliente_alicorp_actual{flg}
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_cliente_dex` a
    LEFT JOIN
    (SELECT DISTINCT * FROM `acpe-dev-uc-sandbox-aa.dev_segmentacion.seg_tabla_alterna_estrategica_transaccional`
     WHERE periodo = '{cfg.periodo_segmentacion}') b
        ON a.cod_cliente_dex = b.cod_interlocutor
    LEFT JOIN `acpe-dev-uc-ml.dev.equifax_urbanicidad_cliente` e
        ON a.cod_cliente_dex = LPAD(CAST(e.id_interlocutor_origen AS STRING), 10, '0')
    LEFT JOIN `acpe-dev-uc-sandbox-aa.dev.ultimos_valores` g
        ON a.cod_cliente_dex = g.cod_cliente_dex
    WHERE
        a.cod_grupo_precio_dex = '{cfg.grupo_precio}'
        AND b.des_segmento_transaccional IN ('1 Oro+', '2 Oro', '3 Plata', '4 Bronce'){filtro}
),

ventas AS (
    SELECT
        cod_cliente_dex,
        ROUND(ABS(
            SUM(CASE WHEN des_autoventa IN ('Dia a Dia', 'Insuma') THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_venta_diaadia_insuma,
        ROUND(ABS(
            SUM(CASE WHEN cod_clase_posicion != 'TNR' THEN mnt_total ELSE 0 END)
            / NULLIF(SUM(CASE WHEN cod_clase_posicion = 'TNR' THEN mnt_total ELSE 0 END), 0) * 100
        ), 2) AS pct_descuento
    FROM `acpe-prod-dlk-shrd-gld.delivery_modelo_dimensional.cr_venta_cliente_dex_diario`
    WHERE periodo >= '{cfg.ventas_desde}'
      AND periodo <= '{cfg.ventas_hasta}'
    GROUP BY cod_cliente_dex
)

SELECT
    c.*,
    v.pct_venta_diaadia_insuma,
    v.pct_descuento
FROM clientes c
LEFT JOIN ventas v
    ON c.cod_cliente_dex = v.cod_cliente_dex
"""


def build_query_puntos(lat_range=(-18.4, -0.03), lng_range=(-81.4, -68.7)) -> str:
    """La query `query_puntos` de los notebooks (igual en las seis verticales)."""
    return f"""
SELECT
    nombre,
    categoria,
    tipo,
    segmento,
    latitud,
    longitud
FROM `acpe-dev-uc-sandbox-aa.scraping.puntos_comerciales`
WHERE latitud  BETWEEN {lat_range[0]} AND {lat_range[1]}
  AND longitud BETWEEN {lng_range[0]} AND {lng_range[1]}
"""


# Config de cada vertical, ya con los valores confirmados contra los seis notebooks.
QUERY_CONFIGS = {
    "lavanderias": QueryConfig(grupo_precio="LV", filtro_segmento=["Lavanderia"]),
    "hoteles": QueryConfig(grupo_precio="LV", filtro_segmento=["Hotel/Casino"]),
    "panaderia": QueryConfig(grupo_precio="PA", filtro_segmento=None, columna_segmento=SQL_SEGMENTO_PANADERIA),
    "gastro": QueryConfig(
        grupo_precio="GA",
        filtro_segmento=["Comida Casera Criolla", "Sanguchería", "Chifa", "Cebichería", "Pollería", "Otros Gastronomía"],
    ),
    "bodegas": QueryConfig(
        grupo_precio="BO", filtro_segmento=["Golosinera", "Alejada", "Estrella", "Clásica"],
        incluir_flg_cliente_perfecto=True,
    ),
    "pm_mercado": QueryConfig(
        grupo_precio="PM", filtro_segmento=["Abarrotero", "Lavandero", "Confitero"],
        incluir_flg_cliente_perfecto=True,
    ),
}


def query_clientes_de(vertical: str) -> str:
    if vertical not in QUERY_CONFIGS:
        raise KeyError(f"Vertical '{vertical}' no está en QUERY_CONFIGS. Disponibles: {sorted(QUERY_CONFIGS)}")
    return build_query_clientes(QUERY_CONFIGS[vertical])


def fetch_clientes(vertical: str, client) -> "pandas.DataFrame":  # noqa: F821
    """Corre la query de clientes de `vertical` con un `bigquery.Client` ya autenticado.

    BigQuery devuelve las columnas `NUMERIC` (aquí, las que salen de `ROUND()`) como
    `Decimal`, que scikit-learn y LightGBM no aceptan; se convierten a `float64` acá.
    """
    df = client.query(query_clientes_de(vertical)).to_dataframe()
    for c in ("pct_venta_diaadia_insuma", "pct_descuento", "val_score_urbanicidad"):
        if c in df.columns:
            df[c] = df[c].astype(float)
    return df


def fetch_puntos(client) -> "pandas.DataFrame":  # noqa: F821
    """Corre la query de puntos comerciales (igual para las seis verticales)."""
    return client.query(build_query_puntos()).to_dataframe()
