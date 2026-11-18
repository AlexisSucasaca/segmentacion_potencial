"""Tests de src/features.py.

* Unitarios y de equivalencia con el bucle original (datos sintéticos, sin archivos tuyos).
* Contra la línea base: recalcula los 15 + 4 conteos de 1 km y `DistanciaTiendaMass` de los
  clientes de los Excel de referencia usando el export de puntos comerciales y locales.csv
  (`--insumos-dir`, por defecto la misma carpeta de los Excel).
  Si esas tablas cambiaron desde el 2026-08-31 pueden aparecer diferencias: revisa el dato
  antes de asumir un error de código.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import BallTree

from src import features as F
from src.config import load_config

CLIENTE = [n for n, c in load_config().items() if c["granularidad"] == "cliente"]
CATS = list(F.PESOS_TRAFICO) + list(F.CATEGORIAS_COMPETENCIA)
MAX_FILAS = 5000
TOLERANCIA_FILAS = 0.001            # fracción de filas que puede diferir por cambios en las tablas


# ---------------------------------------------------------------- coordenadas
def test_corregir_coordenadas():
    x = [-77.0, -12.0, -12.0, np.nan, 0.0, "-77.5", -100.0]
    y = [-12.0, -77.0, -12.0, -77.0, 5.0, "-12.5", -12.0]
    lat, lng = F.corregir_coordenadas(x, y)
    np.testing.assert_allclose(lat, [-12, -12, np.nan, np.nan, np.nan, -12.5, np.nan])
    np.testing.assert_allclose(lng, [-77, -77, np.nan, np.nan, np.nan, -77.5, np.nan])


def test_rangos_de_peru_no_se_solapan():
    """Por eso la corrección nunca es ambigua."""
    assert F.LAT_RANGE[1] < F.LNG_RANGE[0] or F.LNG_RANGE[1] < F.LAT_RANGE[0]


def test_web_mercator_valores_conocidos():
    x, y = F.a_web_mercator([0, 180, 0], [0, 0, 45])
    np.testing.assert_allclose(x, [0, 20037508.342789244, 0], atol=1e-6)
    np.testing.assert_allclose(y, [0, 0, 5621521.486192066], atol=1e-6)


def test_web_mercator_igual_a_pyproj():
    pyproj = pytest.importorskip("pyproj")
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(-81.4, -68.7, 200), rng.uniform(-18.4, -0.03, 200)
    px, py = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    x, y = F.a_web_mercator(lon, lat)
    np.testing.assert_allclose(x, px, atol=1e-6)
    np.testing.assert_allclose(y, py, atol=1e-6)


def test_distancia_tienda_mass():
    d = F.distancia_tienda_mass([0.0, 1.0], [0.0, 0.0], [0.0], [0.0])   # en el ecuador 1° = 111,319.49 m
    np.testing.assert_allclose(d, [0.0, 111319.4908], atol=1e-3)


# ---------------------------------------------------------------- conteos en radio
def _haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _mundo(n_clientes=300, n_puntos=4000, seed=0):
    rng = np.random.default_rng(seed)
    centro = np.array([-12.05, -77.03])
    puntos = pd.DataFrame({
        "nombre": "p", "segmento": None,
        "latitud": centro[0] + rng.normal(scale=0.02, size=n_puntos),
        "longitud": centro[1] + rng.normal(scale=0.02, size=n_puntos),
    })
    puntos["categoria"] = rng.choice(CATS + ["Otra"], n_puntos)           # incluye una categoría que se ignora
    puntos["tipo"] = np.where(puntos["categoria"].isin(F.CATEGORIAS_COMPETENCIA), "competencia", "trafico")
    clientes = pd.DataFrame({
        "id": np.arange(n_clientes) + 500,
        "lat_corr": centro[0] + rng.normal(scale=0.03, size=n_clientes),
        "lng_corr": centro[1] + rng.normal(scale=0.03, size=n_clientes),
    })
    return clientes, puntos


def test_conteos_en_radio_igual_a_fuerza_bruta():
    clientes, puntos = _mundo()
    got = F.conteos_en_radio(clientes.lat_corr, clientes.lng_corr, puntos.latitud, puntos.longitud,
                             puntos.categoria, CATS, radio_km=1.0, chunk=70)     # chunk chico: prueba los bloques
    esperado = np.zeros_like(got)
    for i, c in clientes.iterrows():
        cerca = _haversine_km(c.lat_corr, c.lng_corr, puntos.latitud.values, puntos.longitud.values) <= 1.0
        for j, cat in enumerate(CATS):
            esperado[i, j] = int((cerca & (puntos.categoria.values == cat)).sum())
    np.testing.assert_array_equal(got, esperado)
    assert got.sum() > 0


def test_conteos_sin_puntos_o_sin_vecinos():
    z = F.conteos_en_radio([-12.0], [-77.0], [], [], pd.Series([], dtype=object), CATS)
    assert z.shape == (1, len(CATS)) and z.sum() == 0
    lejos = F.conteos_en_radio([-3.0], [-70.0], [-12.0], [-77.0], pd.Series(["Colegio"]), CATS)
    assert lejos.sum() == 0


# ---------------------------------------------------------------- scores: equivalencia con el bucle original
def _legacy_scores(df_clientes: pd.DataFrame, df_puntos: pd.DataFrame) -> pd.DataFrame:
    """Bloque original de los notebooks (celda de scores), sin cambios de lógica."""
    PESOS_BO = F.PESOS_TRAFICO
    CATEGORIAS_TRAFICO = list(PESOS_BO.keys())
    CATEGORIAS_COMPETENCIA = list(F.CATEGORIAS_COMPETENCIA)
    RADIO_RAD = 1.0 / 6371
    df_trafico = df_puntos[df_puntos["tipo"] == "trafico"].reset_index(drop=True)
    df_competencia = df_puntos[df_puntos["tipo"] == "competencia"].reset_index(drop=True)
    tree_trafico = BallTree(np.radians(df_trafico[["latitud", "longitud"]].values), metric="haversine")
    tree_comp = BallTree(np.radians(df_competencia[["latitud", "longitud"]].values), metric="haversine")
    coords_clientes = np.radians(df_clientes[["lat_corr", "lng_corr"]].values)
    indices_trafico = tree_trafico.query_radius(coords_clientes, r=RADIO_RAD)
    indices_comp = tree_comp.query_radius(coords_clientes, r=RADIO_RAD)
    cols_breakdown = CATEGORIAS_TRAFICO + CATEGORIAS_COMPETENCIA
    resultados = []
    for i, row in enumerate(df_clientes.itertuples()):
        idx_t = indices_trafico[i]
        puntos_t = df_trafico.iloc[idx_t] if len(idx_t) > 0 else pd.DataFrame()
        score_comercial_raw = 0.0
        breakdown = {c: 0 for c in cols_breakdown}
        if len(puntos_t) > 0:
            for cat, peso in PESOS_BO.items():
                n = (puntos_t["categoria"] == cat).sum()
                breakdown[cat] = int(n)
                score_comercial_raw += n * peso
        idx_c = indices_comp[i]
        puntos_c = df_competencia.iloc[idx_c] if len(idx_c) > 0 else pd.DataFrame()
        score_competencia_raw = 0.0
        if len(puntos_c) > 0:
            for cat in CATEGORIAS_COMPETENCIA:
                n = (puntos_c["categoria"] == cat).sum()
                breakdown[cat] = int(n)
                score_competencia_raw += n
        resultados.append({"id": row.id, "score_comercial_raw": score_comercial_raw,
                           "score_competencia_raw": score_competencia_raw, **breakdown})
    df_scores = pd.DataFrame(resultados)

    def normalizar(serie, min_val=1, max_val=10):
        s_min, s_max = serie.min(), serie.max()
        if s_max == s_min:
            return pd.Series([min_val] * len(serie))
        return (min_val + (max_val - min_val) * (serie - s_min) / (s_max - s_min)).round(1)

    df_scores["score_comercial"] = normalizar(df_scores["score_comercial_raw"])
    df_scores["score_competencia"] = normalizar(df_scores["score_competencia_raw"])
    med_com = df_scores["score_comercial"].median()
    med_cmp = df_scores["score_competencia"].median()

    def cuadrante(row):
        alto_com = row["score_comercial"] >= med_com
        alto_cmp = row["score_competencia"] >= med_cmp
        if alto_com and not alto_cmp:
            return "Oportunidad"
        elif alto_com and alto_cmp:
            return "Batalla"
        elif not alto_com and not alto_cmp:
            return "Potencial"
        else:
            return "Riesgo"

    df_scores["cuadrante"] = df_scores.apply(cuadrante, axis=1)
    return df_scores.drop(columns=["score_comercial_raw", "score_competencia_raw"])


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_scores_equivalen_al_bucle_original(semilla):
    clientes, puntos = _mundo(seed=semilla)
    nuevo = F.calcular_scores(clientes, puntos, "id")
    viejo = _legacy_scores(clientes, puntos)
    assert list(nuevo.columns) == list(viejo.columns)
    pd.testing.assert_frame_equal(nuevo, viejo, check_dtype=False)
    assert nuevo["cuadrante"].nunique() == 4


def test_normalizar_y_cuadrante():
    assert F.normalizar(pd.Series([5, 5, 5])).tolist() == [1.0, 1.0, 1.0]
    assert F.normalizar(pd.Series([0, 5, 10])).tolist() == [1.0, 5.5, 10.0]
    com = pd.Series([1.0, 9.0, 9.0, 1.0])
    cmp_ = pd.Series([1.0, 1.0, 9.0, 9.0])
    assert F.cuadrante(com, cmp_).tolist() == ["Potencial", "Oportunidad", "Batalla", "Riesgo"]


def test_cliente_sin_puntos_cerca_tiene_score_minimo():
    clientes, puntos = _mundo(n_clientes=50)
    clientes.loc[0, ["lat_corr", "lng_corr"]] = [-3.0, -70.0]              # en plena selva, sin puntos
    out = F.calcular_scores(clientes, puntos, "id")
    fila = out.iloc[0]
    assert fila[CATS].sum() == 0 and fila["score_comercial"] == 1.0 and fila["score_competencia"] == 1.0


# ---------------------------------------------------------------- agregar_features_geo
def _clientes_bq(clientes):
    # como sale de BigQuery: x = longitud, y = latitud, como texto
    return pd.DataFrame({
        "cod_cliente_dex": clientes["id"],
        "val_coordenada_x": clientes["lng_corr"].astype(str), "val_coordenada_y": clientes["lat_corr"].astype(str),
        "des_ciudad": "Lima",
    })


def _locales(n=5, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"latitud": -12.05 + rng.normal(scale=0.03, size=n), "longitud": -77.03 + rng.normal(scale=0.03, size=n)})


def test_agregar_features_geo_descarta_invalidos_e_informa():
    clientes, puntos = _mundo(n_clientes=60)
    bq = _clientes_bq(clientes)
    bq.loc[0, "val_coordenada_x"] = None                                   # sin coordenada
    bq.loc[1, ["val_coordenada_x", "val_coordenada_y"]] = ["10.0", "10.0"] # fuera de Perú
    df, info = F.agregar_features_geo(bq, puntos, _locales())
    assert info == {"clientes_originales": 60, "sin_coordenadas_validas": 2, "con_coordenadas_validas": 58,
                    "coordenadas_intercambiadas": 0}
    assert len(df) == 58 and "val_coordenada_x" not in df.columns
    assert {"X", "Y", "DistanciaTiendaMass", "score_comercial", "score_competencia", "cuadrante", "des_ciudad"} <= set(df.columns)
    assert set(CATS) <= set(df.columns)


def test_coordenadas_intercambiadas_crudas_vs_corregidas():
    clientes, puntos = _mundo(n_clientes=60)
    bq = _clientes_bq(clientes)
    bq.loc[5, ["val_coordenada_x", "val_coordenada_y"]] = bq.loc[5, ["val_coordenada_y", "val_coordenada_x"]].values  # x=lat, y=lng
    locales = _locales()
    crudas, info = F.agregar_features_geo(bq, puntos, locales, coords="crudas")
    corr, _ = F.agregar_features_geo(bq, puntos, locales, coords="corregidas")
    assert info["coordenadas_intercambiadas"] == 1

    ref = clientes.iloc[5]
    fila_c = crudas.loc[crudas.cod_cliente_dex == ref["id"]].iloc[0]
    fila_r = corr.loc[corr.cod_cliente_dex == ref["id"]].iloc[0]
    assert fila_c.X == pytest.approx(ref["lat_corr"]) and fila_c.Y == pytest.approx(ref["lng_corr"])       # crudas: intercambiadas
    assert fila_r.X == pytest.approx(ref["lng_corr"]) and fila_r.Y == pytest.approx(ref["lat_corr"])       # corregidas: X=lng, Y=lat
    assert fila_c.DistanciaTiendaMass > 1e6 > fila_r.DistanciaTiendaMass                                   # >1,000 km vs cerca

    # el resto de las columnas no cambia entre modos: los scores siempre usan coordenadas corregidas
    cols = [c for c in crudas.columns if c not in ("X", "Y", "DistanciaTiendaMass")]
    pd.testing.assert_frame_equal(crudas[cols], corr[cols])
    normales = crudas.cod_cliente_dex != ref["id"]
    pd.testing.assert_series_equal(crudas.loc[normales, "DistanciaTiendaMass"], corr.loc[normales, "DistanciaTiendaMass"])


def test_agregar_features_geo_validaciones():
    clientes, puntos = _mundo(n_clientes=10)
    bq = _clientes_bq(clientes)
    with pytest.raises(ValueError, match="coords"):
        F.agregar_features_geo(bq, puntos, _locales(), coords="otra")
    with pytest.raises(KeyError, match="Faltan columnas"):
        F.agregar_features_geo(bq.drop(columns=["val_coordenada_y"]), puntos, _locales())
    with pytest.raises(ValueError, match="repetidos"):
        F.agregar_features_geo(pd.concat([bq, bq.iloc[[0]]]), puntos, _locales())


# ---------------------------------------------------------------- contra la línea base
@pytest.mark.parametrize("nombre", CLIENTE)
def test_features_geo_reproducen_la_linea_base(nombre, config, baseline_dir, insumos):
    from tests.conftest import read_sheet

    cfg = config[nombre]
    candidatos = sorted(baseline_dir.glob(f"*{cfg['salida']}.xlsx"))
    if not candidatos:
        pytest.skip(f"No hay Excel de referencia para '{nombre}'")
    ref = read_sheet(candidatos[-1], cfg["export"]["hoja_principal"],
                     ["Cod Cliente DEX", "X", "Y", "DistanciaTiendaMass", *CATS], max_rows=MAX_FILAS)
    for c in ref.columns:
        ref[c] = pd.to_numeric(ref[c], errors="coerce")

    bq = pd.DataFrame({"cod_cliente_dex": ref["Cod Cliente DEX"],
                       "val_coordenada_x": ref["X"], "val_coordenada_y": ref["Y"]})
    df, info = F.agregar_features_geo(bq, insumos["puntos"], insumos["locales"], coords="crudas")
    assert info["sin_coordenadas_validas"] == 0                     # el Excel ya solo trae clientes válidos
    df = df.set_index("cod_cliente_dex").loc[ref["Cod Cliente DEX"]].reset_index()

    # 1) los 15 + 4 conteos de 1 km coinciden (incluida la corrección de coordenadas intercambiadas)
    distintas = (df[CATS].to_numpy() != ref[CATS].to_numpy()).any(axis=1)
    assert distintas.mean() <= TOLERANCIA_FILAS, (
        f"{distintas.sum()} de {len(df)} clientes con conteos distintos (puntos: {insumos['puntos_archivo']})")

    # 2) DistanciaTiendaMass con coordenadas crudas, exacta en metros
    dif = np.abs(df["DistanciaTiendaMass"].to_numpy() - ref["DistanciaTiendaMass"].to_numpy())
    assert (dif > 1e-6).mean() <= TOLERANCIA_FILAS, f"DistanciaTiendaMass distinta en {(dif > 1e-6).sum()} clientes"

    # 3) con coordenadas corregidas solo cambian los clientes intercambiados (X, Y y distancia)
    corr, _ = F.agregar_features_geo(bq, insumos["puntos"], insumos["locales"], coords="corregidas")
    corr = corr.set_index("cod_cliente_dex").loc[ref["Cod Cliente DEX"]].reset_index()
    cambia = (np.abs(corr["DistanciaTiendaMass"].to_numpy() - df["DistanciaTiendaMass"].to_numpy()) > 1e-6)
    assert cambia.sum() == info["coordenadas_intercambiadas"]
    if cambia.any():
        assert (df.loc[cambia, "DistanciaTiendaMass"] > 1e6).all()          # crudas: miles de km
        assert (corr.loc[cambia, "DistanciaTiendaMass"] < 1e6).all()        # corregidas: razonable
