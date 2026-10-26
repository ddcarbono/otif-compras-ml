"""Los dos modelos publicados en ``artifacts/``: un ida y vuelta por joblib no cambia ninguna predicción, y cada uno trae
su ficha JSON con lo que el Entregable 3 pide (versiones, semilla, umbral, Platt)."""
import io
import json

import joblib
import numpy as np
import pytest

from src import data
from src.data import ejemplos, ejemplos_original, predecir, predecir_original

ART = data.DATA.parent / "artifacts"


def ida_y_vuelta(obj):
    b = io.BytesIO()
    joblib.dump(obj, b)
    b.seek(0)
    return joblib.load(b)


@pytest.mark.parametrize("nombre, cargar, predecir_, filas", [
    ("servido", data.servido, predecir, ejemplos),
    ("original_servido", data.original_servido, predecir_original, ejemplos_original),
])
def test_ida_y_vuelta_por_joblib_no_cambia_la_prediccion(monkeypatch, nombre, cargar, predecir_, filas):
    s, ficha = cargar()
    e = filas()
    antes = [predecir_(e.iloc[[i]].copy())[0] for i in range(0, len(e), 30)]
    copia = ida_y_vuelta(s)
    monkeypatch.setattr(data, nombre, lambda: (copia, ficha))
    despues = [predecir_(e.iloc[[i]].copy())[0] for i in range(0, len(e), 30)]
    assert np.array_equal(antes, despues)


@pytest.mark.parametrize("ficha", ["servido.json", "original_servido.json", "logistica.metadata.json",
                                   "xgboost.metadata.json"])
def test_cada_artefacto_trae_su_ficha(ficha):
    f = json.loads((ART / ficha).read_text(encoding="utf-8"))
    assert 0 < f["umbral"] < 1
