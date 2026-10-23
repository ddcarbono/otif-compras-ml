"""Carga y caché de lo que lee el tablero (``data/``). Se lee una sola vez al arrancar; los callbacks filtran en memoria.
No importa scikit-learn ni el dataset completo: en Render solo existe ``data/``."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


@lru_cache(maxsize=None)
def resumen():
    return json.loads((DATA / "resumen.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def diccionario():
    return pd.read_csv(DATA / "diccionario.csv")


@lru_cache(maxsize=None)
def muestra():
    m = pd.read_parquet(DATA / "muestra_eda.parquet")
    m["fecha_pedido"] = pd.to_datetime(m.fecha_pedido)
    return m


@lru_cache(maxsize=None)
def agregado(nombre):
    return pd.read_csv(DATA / "agregados" / f"{nombre}.csv")


@lru_cache(maxsize=None)
def serie_mensual():
    s = agregado("serie_mensual").copy()
    s["fecha_pedido"] = pd.to_datetime(s.fecha_pedido)
    return s


def nombre(v):
    d = diccionario()
    return dict(zip(d.variable, d.nombre)).get(v, v)


def unidad(v):
    d = diccionario()
    return dict(zip(d.variable, d.unidad.fillna("—"))).get(v, "—")


def es(v, d=0):
    """Número con punto de miles y coma decimal."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.{d}f}".translate(str.maketrans(",.", ".,"))


def pct(v, d=1):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{es(100 * v, d)} %"
