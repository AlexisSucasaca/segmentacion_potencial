"""Tests de src/metrics_log.py (datos sintéticos, sin archivos tuyos)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.metrics_log import log_metrics, metrics_rows, ultima_referencia
from src.model import ModelResult


def _resultado(auc_bronce=80.0):
    val = {
        "4 Bronce": {"benchmarks": {
            "model": {"accuracy": 0.6, "f1_macro": 0.55, "auc": auc_bronce / 100},
            "majority": {"accuracy": 0.4, "f1_macro": 0.3, "auc": 0.5},
            "stratified": {"accuracy": 0.35, "f1_macro": 0.32, "auc": 0.5},
            "uniform": {"accuracy": 0.25, "f1_macro": 0.24, "auc": 0.5},
        }},
        "3 Plata": {"benchmarks": {
            "model": {"accuracy": 0.7, "f1_macro": 0.65, "auc": 0.75},
            "majority": {"accuracy": 0.5, "f1_macro": 0.4, "auc": 0.5},
            "stratified": {"accuracy": 0.45, "f1_macro": 0.42, "auc": 0.5},
            "uniform": {"accuracy": 0.3, "f1_macro": 0.28, "auc": 0.5},
        }},
    }
    return ModelResult(estimates=pd.DataFrame(), validation=val, params={}, class_order=["1 Oro+", "2 Oro", "3 Plata", "4 Bronce"])


def test_metrics_rows_forma():
    filas = metrics_rows("bodegas", _resultado(), fecha="2026-09-29 10:00:00")
    assert len(filas) == 6  # 2 clases x 3 métricas
    assert set(filas["clase"]) == {"4 Bronce", "3 Plata"}
    assert set(filas["metrica"]) == {"accuracy", "f1_macro", "auc"}
    fila = filas[(filas.clase == "4 Bronce") & (filas.metrica == "auc")].iloc[0]
    assert fila["modelo"] == 0.8 and fila["majority"] == 0.5


def test_log_metrics_crea_y_acumula(tmp_path):
    ruta = tmp_path / "metricas.csv"
    log_metrics("lavanderias", _resultado(), ruta, fecha="2026-09-29 10:00:00")
    assert ruta.exists()
    log_metrics("bodegas", _resultado(), ruta, fecha="2026-09-29 11:00:00")
    historial = pd.read_csv(ruta)
    assert set(historial["vertical"]) == {"lavanderias", "bodegas"}
    assert len(historial) == 12  # 6 filas x 2 corridas


def test_log_metrics_no_borra_corridas_anteriores(tmp_path):
    ruta = tmp_path / "metricas.csv"
    log_metrics("bodegas", _resultado(auc_bronce=70), ruta, fecha="2026-09-01 00:00:00")
    log_metrics("bodegas", _resultado(auc_bronce=85), ruta, fecha="2026-09-29 00:00:00")
    historial = pd.read_csv(ruta)
    assert len(historial[historial["vertical"] == "bodegas"]) == 12
    assert sorted(historial["fecha"].unique()) == ["2026-09-01 00:00:00", "2026-09-29 00:00:00"]


def test_ultima_referencia(tmp_path):
    ruta = tmp_path / "metricas.csv"
    assert ultima_referencia(ruta, "bodegas").empty  # sin archivo todavía
    log_metrics("bodegas", _resultado(auc_bronce=70), ruta, fecha="2026-09-01 00:00:00")
    log_metrics("lavanderias", _resultado(auc_bronce=99), ruta, fecha="2026-09-15 00:00:00")
    log_metrics("bodegas", _resultado(auc_bronce=85), ruta, fecha="2026-09-29 00:00:00")

    ref = ultima_referencia(ruta, "bodegas")
    assert (ref["fecha"] == "2026-09-29 00:00:00").all()
    assert ref[(ref.clase == "4 Bronce") & (ref.metrica == "auc")]["modelo"].iloc[0] == pytest.approx(0.85)


def test_ultima_referencia_vertical_sin_historial(tmp_path):
    ruta = tmp_path / "metricas.csv"
    log_metrics("bodegas", _resultado(), ruta, fecha="2026-09-01 00:00:00")
    assert ultima_referencia(ruta, "pm_mercado").empty
