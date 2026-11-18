"""Features geográficas de cada cliente: scores de entorno comercial y distancia a Mass.

Reproduce el bloque de los notebooks que va desde el resultado de BigQuery hasta
`DistanciaTiendaMass`, vectorizado (antes recorría cliente por cliente):

  1. Corrige coordenadas invertidas (lat/lng) según el rango de Perú.
  2. Cuenta puntos comerciales en 1 km a la redonda (15 categorías de tráfico y 4 de competencia).
  3. score_comercial (conteos ponderados) y score_competencia, normalizados a 1-10, y el
     `cuadrante` (Oportunidad / Batalla / Potencial / Riesgo) según la mediana de ambos.
  4. Distancia en metros a la tienda Mass más cercana.

Todavía no incluye el cruce con Equifax (urbanicidad, % de hogares, puntos de interés).

Sobre las coordenadas
---------------------
Los scores de 1 km siempre usan las coordenadas corregidas (como antes). Las que van como
features `X`, `Y` y a `DistanciaTiendaMass` dependen de `coords`:

  "crudas"      las de BigQuery tal cual (igual que los notebooks). Si un cliente trae latitud y
                longitud intercambiadas, su distancia a Mass sale absurda (~12,000 km).
  "corregidas"  las mismas que usan los scores.

En los Excel del 2026-08-31 hay entre 0.05 % y 0.25 % de clientes con coordenadas invertidas.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

from src.equifax import agregar_equifax

LAT_RANGE = (-18.4, -0.03)          # Perú
LNG_RANGE = (-81.4, -68.7)
EARTH_RADIUS_KM = 6371.0
WEB_MERCATOR_R = 6378137.0          # EPSG:3857, el mismo que usaba geopandas

# Peso de cada categoría de tráfico en el score comercial.
PESOS_TRAFICO: Dict[str, float] = {
    "Mercado Abastos": 2.0, "Mercado Mayorista": 2.0, "Colegio": 3.0, "Universidad": 2.0,
    "Terminal Buses": 3.0, "Hospital": 2.0, "Hotel": 1.0, "Restaurante": 1.0, "Banco": 2.0,
    "Residencial": 3.0, "Grifo": 2.0, "Centro Comercial": 2.0, "Iglesia": 1.0,
    "Supermercado": 2.0, "Comida Rapida": 1.0,
}
CATEGORIAS_COMPETENCIA: Tuple[str, ...] = ("Comp_BO_Tambo", "Comp_BO_Mass", "Comp_BO_Listo", "Comp_BO_Oxxo")
RADIO_KM = 1.0


# --------------------------------------------------------------------------------------
# Coordenadas
# --------------------------------------------------------------------------------------
def corregir_coordenadas(x, y) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (lat, lng) a partir de dos columnas que pueden venir en cualquier orden.

    En la base lo normal es x = longitud e y = latitud. Si x cae en el rango de latitud de
    Perú e y en el de longitud, el cliente viene intercambiado y se corrige; si viene lo
    normal, se deja; en cualquier otro caso (o con vacíos) el resultado es NaN. Los rangos
    de Perú no se solapan, así que no hay casos ambiguos.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    in_lat = lambda v: (v >= LAT_RANGE[0]) & (v <= LAT_RANGE[1])   # noqa: E731
    in_lng = lambda v: (v >= LNG_RANGE[0]) & (v <= LNG_RANGE[1])   # noqa: E731
    intercambiado = in_lat(x) & in_lng(y)          # x trae la latitud, y la longitud
    normal = in_lng(x) & in_lat(y)                 # x trae la longitud, y la latitud
    lat = np.where(intercambiado, x, np.where(normal, y, np.nan))
    lng = np.where(intercambiado, y, np.where(normal, x, np.nan))
    return lat, lng


def a_web_mercator(lon, lat) -> Tuple[np.ndarray, np.ndarray]:
    """Grados -> metros en EPSG:3857 (fórmula cerrada; no requiere pyproj ni geopandas)."""
    lon = np.radians(np.asarray(lon, dtype=float))
    lat = np.radians(np.asarray(lat, dtype=float))
    return WEB_MERCATOR_R * lon, WEB_MERCATOR_R * np.log(np.tan(np.pi / 4 + lat / 2))


def distancia_tienda_mass(lon, lat, mass_lon, mass_lat) -> np.ndarray:
    """Distancia en metros (EPSG:3857, como antes) a la tienda Mass más cercana."""
    mx, my = a_web_mercator(mass_lon, mass_lat)
    cx, cy = a_web_mercator(lon, lat)
    dist, _ = cKDTree(np.column_stack([mx, my])).query(np.column_stack([cx, cy]), k=1)
    return dist


# --------------------------------------------------------------------------------------
# Puntos comerciales en un radio
# --------------------------------------------------------------------------------------
def conteos_en_radio(
    lat, lng, puntos_lat, puntos_lng, puntos_categoria: pd.Series,
    categorias: Sequence[str], radio_km: float = RADIO_KM, chunk: int = 20000,
) -> np.ndarray:
    """Matriz (clientes x categorías) con los puntos a `radio_km` o menos (haversine).

    Los puntos de categorías que no estén en `categorias` se ignoran. Se procesa por
    bloques para acotar la memoria en bases grandes (bodegas: 126 mil clientes).
    """
    categorias = list(categorias)
    k = len(categorias)
    lat = np.asarray(lat, dtype=float)
    lng = np.asarray(lng, dtype=float)
    out = np.zeros((len(lat), k), dtype=np.int64)
    if len(puntos_lat) == 0 or len(lat) == 0:
        return out

    codigo = {c: i for i, c in enumerate(categorias)}
    codes = pd.Series(puntos_categoria).map(codigo).fillna(-1).to_numpy(dtype=np.int64)
    tree = BallTree(np.radians(np.column_stack([puntos_lat, puntos_lng])), metric="haversine")
    r = radio_km / EARTH_RADIUS_KM

    for ini in range(0, len(lat), chunk):
        coords = np.radians(np.column_stack([lat[ini:ini + chunk], lng[ini:ini + chunk]]))
        vecinos = tree.query_radius(coords, r=r)
        largos = np.fromiter((len(v) for v in vecinos), dtype=np.int64, count=len(vecinos))
        if largos.sum() == 0:
            continue
        idx = np.concatenate(vecinos)
        filas = np.repeat(np.arange(len(vecinos)), largos)
        c = codes[idx]
        ok = c >= 0
        out[ini:ini + len(vecinos)] = np.bincount(
            filas[ok] * k + c[ok], minlength=len(vecinos) * k
        ).reshape(len(vecinos), k)
    return out


# --------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------
def normalizar(serie: pd.Series, min_val: float = 1, max_val: float = 10) -> pd.Series:
    """Mín-máx a [min_val, max_val] con un decimal. Si todos los valores son iguales -> min_val."""
    s_min, s_max = serie.min(), serie.max()
    if s_max == s_min:
        return pd.Series(float(min_val), index=serie.index)
    return (min_val + (max_val - min_val) * (serie - s_min) / (s_max - s_min)).round(1)


def cuadrante(score_comercial: pd.Series, score_competencia: pd.Series) -> pd.Series:
    """Oportunidad / Batalla / Potencial / Riesgo según la mediana de cada score."""
    alto_com = score_comercial >= score_comercial.median()
    alto_cmp = score_competencia >= score_competencia.median()
    return pd.Series(
        np.select(
            [alto_com & ~alto_cmp, alto_com & alto_cmp, ~alto_com & ~alto_cmp],
            ["Oportunidad", "Batalla", "Potencial"],
            default="Riesgo",
        ),
        index=score_comercial.index, dtype=object,
    )


def calcular_scores(
    clientes: pd.DataFrame, puntos: pd.DataFrame, id_col: str,
    lat_col: str = "lat_corr", lng_col: str = "lng_corr",
    pesos: Optional[Dict[str, float]] = None,
    competencia: Sequence[str] = CATEGORIAS_COMPETENCIA, radio_km: float = RADIO_KM,
) -> pd.DataFrame:
    """Conteos por categoría, score_comercial, score_competencia y cuadrante de cada cliente.

    `puntos` trae `tipo` ('trafico' / 'competencia'), `categoria`, `latitud` y `longitud`.
    Los scores se normalizan sobre TODOS los clientes recibidos, así que hay que pasar la
    base completa de la vertical, no un subconjunto.
    """
    pesos = dict(PESOS_TRAFICO if pesos is None else pesos)
    cats_trafico, cats_comp = list(pesos), list(competencia)
    trafico = puntos[puntos["tipo"] == "trafico"]
    comp = puntos[puntos["tipo"] == "competencia"]

    n_trafico = conteos_en_radio(
        clientes[lat_col], clientes[lng_col], trafico["latitud"], trafico["longitud"],
        trafico["categoria"], cats_trafico, radio_km,
    )
    n_comp = conteos_en_radio(
        clientes[lat_col], clientes[lng_col], comp["latitud"], comp["longitud"],
        comp["categoria"], cats_comp, radio_km,
    )

    out = pd.DataFrame({id_col: clientes[id_col].to_numpy()})
    out = pd.concat(
        [out, pd.DataFrame(n_trafico, columns=cats_trafico), pd.DataFrame(n_comp, columns=cats_comp)], axis=1
    )
    raw_com = n_trafico @ np.array([pesos[c] for c in cats_trafico], dtype=float)
    raw_cmp = n_comp.sum(axis=1).astype(float)
    out["score_comercial"] = normalizar(pd.Series(raw_com))
    out["score_competencia"] = normalizar(pd.Series(raw_cmp))
    out["cuadrante"] = cuadrante(out["score_comercial"], out["score_competencia"])
    return out


# --------------------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------------------
def agregar_features_geo(
    df_clientes: pd.DataFrame, puntos: pd.DataFrame, locales: pd.DataFrame,
    id_col: str = "cod_cliente_dex", x_col: str = "val_coordenada_x", y_col: str = "val_coordenada_y",
    coords: str = "crudas", radio_km: float = RADIO_KM,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Agrega scores de 1 km, cuadrante, X, Y y DistanciaTiendaMass al resultado de BigQuery.

    Los clientes sin coordenadas válidas se descartan (se informa cuántos). Las columnas
    `x_col` / `y_col` se reemplazan por `X` (longitud) e `Y` (latitud).
    `locales` trae las tiendas Mass con `latitud` y `longitud`.
    """
    if coords not in ("crudas", "corregidas"):
        raise ValueError(f"coords debe ser 'crudas' o 'corregidas', recibió '{coords}'")
    faltan = [c for c in (id_col, x_col, y_col) if c not in df_clientes.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en df_clientes: {faltan}")

    df = df_clientes.copy()
    df["_x"] = pd.to_numeric(df[x_col], errors="coerce")
    df["_y"] = pd.to_numeric(df[y_col], errors="coerce")
    lat, lng = corregir_coordenadas(df["_x"], df["_y"])
    df["lat_corr"], df["lng_corr"] = lat, lng

    n_original = len(df)
    df = df.dropna(subset=["lat_corr", "lng_corr"]).reset_index(drop=True)
    info = {
        "clientes_originales": n_original,
        "sin_coordenadas_validas": n_original - len(df),
        "con_coordenadas_validas": len(df),
        "coordenadas_intercambiadas": int((df["_x"] == df["lat_corr"]).sum()),
    }
    if df[id_col].duplicated().any():
        raise ValueError(f"df_clientes tiene '{id_col}' repetidos.")

    scores = calcular_scores(df, puntos, id_col, radio_km=radio_km)
    df = df.merge(scores, on=id_col, how="left", validate="1:1")

    if coords == "crudas":
        df["X"], df["Y"] = df["_x"], df["_y"]
    else:
        df["X"], df["Y"] = df["lng_corr"], df["lat_corr"]
    df["DistanciaTiendaMass"] = distancia_tienda_mass(
        df["X"], df["Y"], locales["longitud"], locales["latitud"]
    )
    df = df.drop(columns=["_x", "_y", "lat_corr", "lng_corr", x_col, y_col])
    return df, info


def agregar_features_cliente(
    df_clientes: pd.DataFrame, puntos: pd.DataFrame, locales: pd.DataFrame, equifax_sum: pd.DataFrame,
    id_col: str = "cod_cliente_dex", x_col: str = "val_coordenada_x", y_col: str = "val_coordenada_y",
    coords: str = "crudas", radio_km: float = RADIO_KM,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """`agregar_features_geo` + el cruce con Equifax (urbanicidad, % de hogares, puntos de interés).

    El cruce con Equifax usa las coordenadas crudas, como en los notebooks, sin importar
    `coords` (ver el aviso en `src/equifax.py`). `equifax_sum` es el resultado de
    `src.equifax.prepare_equifax_sum(equifax, urbanicidad)`.
    """
    df, info = agregar_features_geo(df_clientes, puntos, locales, id_col, x_col, y_col, coords, radio_km)
    # las coordenadas crudas ya no quedan en `df` cuando coords='corregidas'; se recuperan del original
    crudas = (
        df_clientes.set_index(id_col)[[x_col, y_col]]
        .loc[df[id_col]].reset_index(drop=True)
        .rename(columns={x_col: "_eq_x", y_col: "_eq_y"})
    )
    crudas["_eq_x"] = pd.to_numeric(crudas["_eq_x"], errors="coerce")
    crudas["_eq_y"] = pd.to_numeric(crudas["_eq_y"], errors="coerce")
    df = pd.concat([df.reset_index(drop=True), crudas], axis=1)
    df = agregar_equifax(df, equifax_sum, x_col="_eq_x", y_col="_eq_y").drop(columns=["_eq_x", "_eq_y"])
    return df, info
