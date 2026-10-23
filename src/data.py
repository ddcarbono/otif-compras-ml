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
def modelo(nombre):
    """Tablas del modelo base en ``data/modelos``; ``resumen`` devuelve el JSON."""
    if nombre == "resumen":
        return json.loads((DATA / "modelos" / "resumen.json").read_text(encoding="utf-8"))
    return pd.read_csv(DATA / "modelos" / f"{nombre}.csv")


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


# --- predicción en vivo ------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def servido():
    """El modelo publicado (``artifacts/servido.joblib``, lo escribe ``src.prediccion``) y su ficha. Se carga al primer
    uso: importa scikit-learn, que el resto del tablero no necesita."""
    import joblib
    art = DATA.parent / "artifacts"
    return joblib.load(art / "servido.joblib"), json.loads((art / "servido.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def ejemplos():
    return pd.read_parquet(DATA / "prediccion" / "ejemplos.parquet")


@lru_cache(maxsize=None)
def rangos():
    return pd.read_csv(DATA / "prediccion" / "rangos.csv").set_index("campo")


def predecir(fila):
    """Probabilidad de llegar tarde de una posición (``fila``: un DataFrame de una fila con las numéricas, las
    categóricas y las tasas ``te_*``) y la contribución de cada variable al logit: coeficiente × valor en la logística,
    valores SHAP exactos de árbol (``pred_contribs``) en XGBoost."""
    s, ficha = servido()
    cat = s["cat"].transform(fila[s["cat_cols"]])
    cat = cat.toarray() if hasattr(cat, "toarray") else cat
    X = np.hstack([np.asarray(s["num"].transform(fila[s["num_cols"]]), dtype=float), np.asarray(cat, dtype=float),
                   fila[s["te_cols"]].to_numpy(dtype=float)])
    m = s["modelo"]
    p = float(m.predict_proba(X)[0, 1])
    if hasattr(m, "coef_"):
        contrib = X[0] * m.coef_[0]
    else:
        import xgboost
        contrib = m.get_booster().predict(xgboost.DMatrix(X, nthread=1), pred_contribs=True)[0][:-1]
    c = pd.DataFrame({"columna": s["nombres"], "contribucion": contrib})
    c["variable"] = c.columna.str.split("__").str[1].str.replace(r"^te_", "", regex=True)
    c["bloque"] = c.columna.str.split("__").str[0]
    # una categórica one-hot suma la contribución de sus columnas bajo el nombre de la variable
    cat = c.bloque == "cat"
    for v in s["cat_cols"]:
        c.loc[cat & c.variable.str.startswith(v + "_"), "variable"] = v
    c = c.groupby("variable", as_index=False).contribucion.sum()
    return p, ficha["umbral"], c.reindex(c.contribucion.abs().sort_values(ascending=False).index)


# --- predicción en vivo con el modelo original ---------------------------------------------------------------------------
HOLGURAS = ["prov", "provmat", "provga", "mat"]


@lru_cache(maxsize=None)
def original_servido():
    """El modelo original publicado (``artifacts/original_servido.joblib``, lo escribe ``original/servir_original.py``):
    el LightGBM del modelo y el del plazo log-normal, en texto, y su ficha. Se carga al primer uso."""
    import joblib
    import lightgbm as lgb
    art = DATA.parent / "artifacts"
    s = joblib.load(art / "original_servido.joblib")
    s["booster"] = lgb.Booster(model_str=s["modelo"])
    s["booster_lognormal"] = lgb.Booster(model_str=s["lognormal"])
    return s, json.loads((art / "original_servido.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def ejemplos_original():
    return pd.read_parquet(DATA / "prediccion" / "ejemplos_original.parquet")


@lru_cache(maxsize=None)
def rangos_original():
    return pd.read_csv(DATA / "prediccion" / "rangos_original.csv").set_index("campo")


def predecir_original(fila):
    """Probabilidad de llegar tarde con el modelo original (``fila``: una posición de ``ejemplos_original``, quizá con
    campos cambiados), la probabilidad del plazo log-normal sola (1 − Φ(z)) y la contribución al logit de cada variable
    (valores SHAP exactos del LightGBM en el corte c = 1) y de cada efecto por entidad.

    Las holguras y z se recalculan con el plazo de la fila: son el plazo al liberar menos el plazo real previo, y cambiar
    uno sin las otras daría una posición imposible."""
    from math import erfc, sqrt
    s, ficha = original_servido()
    x = fila.copy()
    for k in HOLGURAS:
        for q in ("med", "q80"):
            x[f"{k}_holgura_{q}"] = x.plazo_habil_al_liberar - x[f"{k}_lt_{q}"]
    for c, cats in s["categorias"].items():
        x[c] = pd.Categorical(x[c], categories=cats)
    mu = float(s["booster_lognormal"].predict(x[s["sin_plazo"]])[0])
    z = (np.log1p(max(float(x.plazo_habil_al_liberar.iloc[0]), 0) + 1.5) - mu) / s["s"]
    x = x.assign(dist_mu=mu, dist_z=z)
    X = x[s["columnas"]].assign(corte=s["corte"])
    efectos = x[s["efectos"]].iloc[0].astype(float)
    logit = float(s["booster"].predict(X, raw_score=True)[0]) + efectos.sum()
    a, b = s["platt"]
    p = float(1 / (1 + np.exp(-(a * logit + b))))
    shap = s["booster"].predict(X, pred_contrib=True)[0]
    c = pd.DataFrame({"variable": list(X.columns) + list(efectos.index),
                      "contribucion": np.r_[shap[:-1], efectos.to_numpy()]})
    return p, ficha["umbral"], 0.5 * erfc(z / sqrt(2)), c.reindex(c.contribucion.abs().sort_values(ascending=False).index)
