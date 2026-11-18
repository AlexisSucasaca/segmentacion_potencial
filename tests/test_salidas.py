"""Tests de src/salidas.py (archivos temporales, sin datos tuyos)."""
from __future__ import annotations

import pytest

from src.salidas import ultimo_archivo


def _tocar(carpeta, *nombres):
    for n in nombres:
        (carpeta / n).write_text("x")


def test_ultimo_archivo_toma_el_mas_reciente_por_nombre(tmp_path):
    _tocar(tmp_path, "20260731_potential_estimations_lavanderia_nuevo.xlsx",
          "20260831_potential_estimations_lavanderia_nuevo.xlsx",
          "20260601_potential_estimations_lavanderia_nuevo.xlsx")
    out = ultimo_archivo(tmp_path, "*potential_estimations_lavanderia_nuevo.xlsx")
    assert out.name == "20260831_potential_estimations_lavanderia_nuevo.xlsx"


def test_ultimo_archivo_no_confunde_con_otra_vertical(tmp_path):
    _tocar(tmp_path, "20260831_potential_estimations_lavanderia_nuevo.xlsx",
          "20260930_potential_estimations_hotel_nuevo.xlsx")
    out = ultimo_archivo(tmp_path, "*potential_estimations_lavanderia_nuevo.xlsx")
    assert out.name == "20260831_potential_estimations_lavanderia_nuevo.xlsx"


def test_ultimo_archivo_falla_con_mensaje_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="lavanderia"):
        ultimo_archivo(tmp_path, "*lavanderia*.xlsx")


def test_ultimo_archivo_orden_no_depende_de_mtime(tmp_path):
    """El más 'nuevo' por nombre gana, aunque el archivo se haya tocado en otro orden
    (por ejemplo, si se volvió a descargar o copiar más tarde)."""
    _tocar(tmp_path, "20260601_x.xlsx")
    _tocar(tmp_path, "20260831_x.xlsx")
    (tmp_path / "20260601_x.xlsx").write_text("tocado de nuevo, más tarde que el otro")
    out = ultimo_archivo(tmp_path, "*x.xlsx")
    assert out.name == "20260831_x.xlsx"
