"""Carga y validación de configs/verticales.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.decision import REGLAS

RAIZ = Path(__file__).resolve().parents[1]
CONFIG_PATH = RAIZ / "configs" / "verticales.yaml"


def load_config(path: Path | str | None = None) -> Dict[str, Dict[str, Any]]:
    """Devuelve {nombre_vertical: config}. Falla con un mensaje claro si algo no cuadra."""
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        verticales = yaml.safe_load(f)["verticales"]
    for nombre, cfg in verticales.items():
        _resolver_modelo(cfg)
        _validar(nombre, cfg)
        cfg["nombre"] = nombre
    return verticales


def get_vertical(nombre: str, path: Path | str | None = None) -> Dict[str, Any]:
    verticales = load_config(path)
    if nombre not in verticales:
        raise KeyError(f"Vertical '{nombre}' no existe. Disponibles: {sorted(verticales)}")
    return verticales[nombre]


def _resolver_modelo(cfg: Dict[str, Any]) -> None:
    """Junta `features` + `features_extra` (y lo mismo con las categóricas) en una sola lista."""
    m = cfg["modelo"]
    m["features"] = list(dict.fromkeys([*m["features"], *m.pop("features_extra", [])]))
    m["categoricas"] = list(dict.fromkeys([*m["categoricas"], *m.pop("categoricas_extra", [])]))


def _validar(nombre: str, cfg: Dict[str, Any]) -> None:
    dec = cfg["decision"]
    if dec["regla"] not in REGLAS:
        raise ValueError(f"[{nombre}] regla desconocida '{dec['regla']}'. Disponibles: {sorted(REGLAS)}")
    if dec["regla"] == "umbral_2clases" and len(cfg["class_order"]) != 2:
        raise ValueError(f"[{nombre}] 'umbral_2clases' requiere exactamente 2 clases.")
    if dec["medium"] > dec["high"]:
        raise ValueError(f"[{nombre}] medium ({dec['medium']}) > high ({dec['high']}).")
    if cfg["feature_mean"]["modo"] not in ("features", "shap"):
        raise ValueError(f"[{nombre}] feature_mean.modo debe ser 'features' o 'shap'.")

    m = cfg["modelo"]
    con_modelo = list(cfg["class_order"])[:0:-1]           # todas las clases menos la más alta
    if set(m["lgbm"]) != set(con_modelo):
        raise ValueError(
            f"[{nombre}] modelo.lgbm debe tener hiperparámetros para {sorted(con_modelo)}, "
            f"tiene {sorted(m['lgbm'])}."
        )
    fuera = [c for c in m["categoricas"] if c not in m["features"]]
    if fuera:
        raise ValueError(f"[{nombre}] categóricas que no están en features: {fuera}")
    if m["cluster_col"] not in m["features"]:
        raise ValueError(f"[{nombre}] cluster_col '{m['cluster_col']}' no está en features.")
    if len(set(m["features"])) != len(m["features"]):
        raise ValueError(f"[{nombre}] features repetidas.")
    if m["target_map"] is not None and not set(cfg["class_order"]) <= set(m["target_map"].values()):
        raise ValueError(f"[{nombre}] target_map no produce todas las clases de class_order.")
