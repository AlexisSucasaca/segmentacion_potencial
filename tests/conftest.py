"""Fixtures de los tests contra la línea base (los Excel de referencia por vertical).

Uso:
    pytest tests -q --baseline-dir "C:/ruta/a/los/excels"
    # o con la variable de entorno POTENCIAL_BASELINE_DIR

Para cada vertical se busca `*<salida>.xlsx` en esa carpeta (la fecha del nombre no importa;
si hay varios se usa el más reciente). Si no existe, sus tests se saltan.

Los Excel de bodegas pesan >100 MB: se leen en modo `read_only` y solo las columnas
necesarias, y el resultado se cachea en `<baseline-dir>/.cache_tests/` (o en la carpeta temporal del sistema si esa carpeta es de solo lectura).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import pytest
from openpyxl import load_workbook

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import load_config  # noqa: E402
from src.decision import proba_col  # noqa: E402

COLS_DECISION = [
    "segmento_trx", "predicted", "proba_pred_segment", "prediction_conf",
    "final_pred", "final_pred_conf", "flag_potential",
]


def pytest_addoption(parser):
    parser.addoption(
        "--baseline-dir",
        default=os.environ.get("POTENCIAL_BASELINE_DIR"),
        help="Carpeta con los Excel de referencia (uno por vertical).",
    )
    parser.addoption(
        "--insumos-dir",
        default=os.environ.get("POTENCIAL_INSUMOS_DIR"),
        help="Carpeta con locales.csv, urbanicidad.csv, el export de puntos comerciales, etc. "
             "Por defecto, la misma de --baseline-dir.",
    )


# --------------------------------------------------------------------------------------
# Lectura rápida de una hoja
# --------------------------------------------------------------------------------------
def _header(path: Path, sheet: str) -> List[str]:
    wb = load_workbook(path, read_only=True)
    try:
        return list(next(wb[sheet].iter_rows(max_row=1, values_only=True)))
    finally:
        wb.close()


def read_sheet(
    path: Path, sheet: str, columns: Optional[Iterable[str]] = None, max_rows: Optional[int] = None
) -> pd.DataFrame:
    """Lee una hoja con openpyxl en modo read_only. Con nombres de columna repetidos
    (X, Y en los Excel actuales) se queda con la primera aparición. `max_rows` limita las
    filas de datos leídas (útil para no cargar los 122 mil clientes de bodegas)."""
    wb = load_workbook(path, read_only=True)
    try:
        it = wb[sheet].iter_rows(values_only=True)
        hdr = list(next(it))
        first: Dict[str, int] = {}
        for i, h in enumerate(hdr):
            first.setdefault(h, i)
        names = list(first) if columns is None else [c for c in dict.fromkeys(columns) if c in first]
        idx = [first[n] for n in names]
        filas = []
        for n, r in enumerate(it):
            if max_rows is not None and n >= max_rows:
                break
            filas.append([r[i] for i in idx])
        return pd.DataFrame(filas, columns=names)
    finally:
        wb.close()


def _cached(path: Path, sheet: str, columns: Optional[List[str]], cache_dir: Path) -> pd.DataFrame:
    st = path.stat()
    # la versión de pandas va en la clave: un pickle de pandas 3 no se puede leer con pandas 2
    key = (f"{path.name}|{st.st_size}|{int(st.st_mtime)}|{sheet}|{sorted(columns) if columns else 'ALL'}"
           f"|pandas{pd.__version__}")
    pkl = cache_dir / (hashlib.md5(key.encode()).hexdigest() + ".pkl")
    if pkl.exists():
        try:
            return pd.read_pickle(pkl)
        except Exception:                      # caché corrupta o de otra versión: se vuelve a leer el Excel
            pass
    df = read_sheet(path, sheet, columns)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_pickle(pkl)
    except OSError:
        pass                                   # carpeta de solo lectura: se lee sin caché
    return df


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def config() -> Dict[str, dict]:
    return load_config()


@pytest.fixture(scope="session")
def baseline_dir(request) -> Path:
    d = request.config.getoption("--baseline-dir")
    if not d:
        pytest.skip("Falta --baseline-dir (o POTENCIAL_BASELINE_DIR) con los Excel de referencia.")
    return Path(d)


@pytest.fixture(scope="session")
def _baseline_cache() -> dict:
    return {}


@pytest.fixture
def baseline(request, config, baseline_dir, _baseline_cache):
    """Carga (una sola vez por sesión) la línea base de la vertical parametrizada.

    Devuelve dict con: cfg, main (hoja principal), fm (hoja Feature mean), features
    (nombres de feature deducidos de las columnas shap_*, en su orden)."""
    nombre = request.param
    if nombre in _baseline_cache:
        return _baseline_cache[nombre]
    cfg = config[nombre]
    candidatos = sorted(baseline_dir.glob(f"*{cfg['salida']}.xlsx"))
    if not candidatos:
        pytest.skip(f"No hay Excel de referencia para '{nombre}' (*{cfg['salida']}.xlsx) en {baseline_dir}")
    path = candidatos[-1]
    hoja = cfg["export"]["hoja_principal"]
    cache_dir = baseline_dir / ".cache_tests"
    if not os.access(baseline_dir, os.W_OK):                     # p. ej. carpeta montada de solo lectura
        cache_dir = Path(tempfile.gettempdir()) / "potencial_cache_tests"

    hdr = _header(path, hoja)
    shap_feats = [h[5:] for h in hdr if isinstance(h, str) and h.startswith("shap_")]
    shap_feats = list(dict.fromkeys(shap_feats))
    proba_cols = [proba_col(c) for c in cfg["class_order"]]

    if cfg["feature_mean"]["modo"] == "shap":
        columnas = None                                   # PM es chico: se lee todo
    else:
        columnas = COLS_DECISION + proba_cols + shap_feats  # solo lo necesario (bodegas pesa >100 MB)
    main = _cached(path, hoja, columnas, cache_dir)
    fm = _cached(path, "Feature mean", None, cache_dir)

    out = {"cfg": cfg, "main": main, "fm": fm, "features": shap_feats, "path": path}
    _baseline_cache[nombre] = out
    return out


@pytest.fixture(scope="session")
def equifax_sum(request, baseline_dir):
    """equifax.csv + urbanicidad.csv, ya resumidos (src.equifax.prepare_equifax_sum).

    equifax.csv pesa ~260 MB: se cachea en `.cache_tests/` igual que los Excel."""
    from src.equifax import prepare_equifax_sum

    carpeta = Path(request.config.getoption("--insumos-dir") or baseline_dir)
    eq = sorted(carpeta.glob("equifax*.csv"))
    urb = sorted(carpeta.glob("urbanicidad*.csv"))
    if not eq or not urb:
        pytest.skip(f"Faltan equifax.csv y/o urbanicidad.csv en {carpeta}")
    path = eq[-1]
    cache_dir = carpeta / ".cache_tests" if os.access(carpeta, os.W_OK) else Path(tempfile.gettempdir()) / "potencial_cache_tests"
    st = path.stat()
    key = f"equifax_sum|{path.name}|{st.st_size}|{int(st.st_mtime)}|pandas{pd.__version__}"
    pkl = cache_dir / (hashlib.md5(key.encode()).hexdigest() + ".pkl")
    if pkl.exists():
        try:
            return pd.read_pickle(pkl)
        except Exception:
            pass
    resumen = prepare_equifax_sum(pd.read_csv(path), pd.read_csv(urb[-1]))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        resumen.to_pickle(pkl)
    except OSError:
        pass
    return resumen


@pytest.fixture(scope="session")
def pm_insumos(request, baseline_dir) -> dict:
    """Maestro de mercados y su respaldo de clustering, para armar mercado_id de PM."""
    carpeta = Path(request.config.getoption("--insumos-dir") or baseline_dir)
    maestro = sorted(carpeta.glob("Maestro*mercado*.xlsx"))
    cluster = sorted(carpeta.glob("cluster_mercado_id*.csv"))
    if not maestro or not cluster:
        pytest.skip(f"Faltan Maestro_de_mercado.xlsx y/o cluster_mercado_id.csv en {carpeta}")
    m = pd.read_excel(maestro[-1], usecols=range(14)).dropna(axis=1, how="all")
    return {"maestro": m, "cluster_fallback": pd.read_csv(cluster[-1])}


@pytest.fixture(scope="session")
def insumos(request, baseline_dir) -> dict:
    """Archivos de entrada de las features geográficas (puntos comerciales y tiendas Mass)."""
    carpeta = Path(request.config.getoption("--insumos-dir") or baseline_dir)
    locales = sorted(carpeta.glob("locales*.csv"))
    puntos = sorted(carpeta.glob("puntos_comerciales*.csv")) or sorted(carpeta.glob("resultados_*.csv"))
    if not locales or not puntos:
        pytest.skip(f"Faltan locales.csv y/o el export de puntos comerciales en {carpeta}")
    return {
        "locales": pd.read_csv(locales[-1], sep=";", encoding="utf-8-sig"),
        "puntos": pd.read_csv(puntos[-1]),
        "puntos_archivo": puntos[-1].name,
    }
