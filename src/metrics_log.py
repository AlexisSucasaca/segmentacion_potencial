"""Registro histórico de las métricas de validación de cada corrida.

Cada vez que se entrena un modelo (`model.fit_potential`), sus métricas de validación
(accuracy, f1_macro, auc del modelo y de los tres baselines dummy) se pueden anexar a un
CSV persistente, con fecha y vertical, para tener un historial y detectar si una corrida
nueva quedó peor que las anteriores (el "modelo de referencia").
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd

from src.model import ModelResult

COLUMNAS = ["fecha", "vertical", "clase", "metrica", "modelo", "majority", "stratified", "uniform"]


def metrics_rows(vertical: str, resultado: ModelResult, fecha=None) -> pd.DataFrame:
    """Una fila por (clase, métrica) con el valor del modelo y de los tres baselines."""
    fecha = fecha or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    filas = []
    for clase, val in resultado.validation.items():
        bench = val["benchmarks"]
        for metrica in ("accuracy", "f1_macro", "auc"):
            filas.append({
                "fecha": fecha, "vertical": vertical, "clase": clase, "metrica": metrica,
                "modelo": bench["model"][metrica],
                **{nombre: bench[nombre][metrica] for nombre in ("majority", "stratified", "uniform")},
            })
    return pd.DataFrame(filas, columns=COLUMNAS)


def log_metrics(vertical: str, resultado: ModelResult, ruta_log: Union[str, Path], fecha=None) -> pd.DataFrame:
    """Agrega las métricas de esta corrida a `ruta_log` (lo crea si no existe) y las devuelve.

    Nunca borra corridas anteriores: cada llamada solo agrega filas nuevas.
    """
    nuevas = metrics_rows(vertical, resultado, fecha)
    ruta_log = Path(ruta_log)
    if ruta_log.exists():
        previas = pd.read_csv(ruta_log)
        combinado = pd.concat([previas, nuevas], ignore_index=True)
    else:
        ruta_log.parent.mkdir(parents=True, exist_ok=True)
        combinado = nuevas
    combinado.to_csv(ruta_log, index=False)
    return nuevas


def ultima_referencia(ruta_log: Union[str, Path], vertical: str) -> pd.DataFrame:
    """La corrida más reciente registrada para `vertical` (para comparar contra la nueva)."""
    ruta_log = Path(ruta_log)
    if not ruta_log.exists():
        return pd.DataFrame(columns=COLUMNAS)
    historial = pd.read_csv(ruta_log)
    historial = historial[historial["vertical"] == vertical]
    if historial.empty:
        return historial
    return historial[historial["fecha"] == historial["fecha"].max()]
