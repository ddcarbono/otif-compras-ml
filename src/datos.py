"""Datos del proyecto: ruta, semilla, partición, tipos de variable y preprocesamiento.

El dataset completo no está en el repositorio (342 MB, datos de una empresa con uso académico autorizado). Se lee de la
variable de entorno ``OTIF_CSV`` o, si no está, de ``data/otif_depurado.csv``. Si junto al CSV hay un
``otif_depurado.parquet`` se usa ese, que carga en segundos; ``cargar`` lo escribe la primera vez.

La codificación jerárquica por objetivo es la del modelado del proyecto (misma fórmula, M = 20).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")   # evita el aviso de joblib al contar núcleos en Windows

RAIZ = Path(__file__).resolve().parents[1]
CSV = Path(os.environ.get("OTIF_CSV", RAIZ / "data" / "otif_depurado.csv"))
SEMILLA = 2026
OBJETIVO = "y_tarde"
PARTICIONES = {"entrenamiento": "2021-2024", "validacion": "2025", "prueba": "2026"}

DIC = pd.read_csv(RAIZ / "data" / "diccionario.csv")
NOMBRE = dict(zip(DIC.variable, DIC.nombre))
GRUPO = dict(zip(DIC.variable, DIC.grupo))
TIPO = dict(zip(DIC.variable, DIC.tipo))
UNIDAD = dict(zip(DIC.variable, DIC.unidad.fillna("—")))
NUMERICAS = [v for v, t in TIPO.items() if t in ("numérica", "binaria")]
CATEGORICAS = [v for v, t in TIPO.items() if t == "categórica"]
ENTIDADES = [v for v, t in TIPO.items() if t == "entidad"]
PREDICTORAS = NUMERICAS + CATEGORICAS + ENTIDADES

# Entidad -> nivel superior para la codificación por objetivo. El último de cada cadena cae a la tasa global.
JERARQUIA = {"proveedor": "pv_region", "material": "grupo_articulos", "imp_ceco": "sociedad", "imp_cuenta": "sociedad",
             "sp_creador": "grupo_compras", "creador_oc": "grupo_compras", "ultimo_liberador": "estrategia_lib",
             "centro": "sociedad", "estrategia_lib": "sociedad", "grupo_compras": "sociedad", "sociedad": "linea",
             "grupo_articulos": "clase_compras", "pv_region": None, "linea": None, "clase_compras": None}
M = 20


def cargar(ruta=CSV):
    """Lee el dataset depurado con los tipos del diccionario y la fecha de pedido como fecha."""
    ruta = Path(ruta)
    pq = ruta.with_suffix(".parquet")
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(ruta, sep=";", decimal=",", low_memory=False)
        for c in CATEGORICAS + ENTIDADES + ["oc_id", "particion"]:
            df[c] = df[c].astype("string")
        df.to_parquet(pq, index=False)
    for c in ["fecha_pedido", "fecha_prediccion", "fecha_prometida"]:
        df[c] = pd.to_datetime(df[c])
    df[NUMERICAS] = df[NUMERICAS].replace([np.inf, -np.inf], np.nan)
    # SAP marca «sin valor» con un espacio (sin imputación, no va a inspección): es una categoría, no un vacío
    for c in CATEGORICAS + ENTIDADES:
        df[c] = df[c].astype("string").str.strip().replace("", "(en blanco)").str.replace(r"\.0$", "", regex=True)
    return df


def particion(df, nombre):
    """Filas de una partición: 'entrenamiento', 'validacion' o 'prueba'."""
    return df[df["particion"] == nombre]


def _codificar(c, ent, y_ent, apl, glob):
    padre = JERARQUIA.get(c)
    prior = _codificar(padre, ent, y_ent, apl, glob) if padre else np.full(len(apl[c]), glob)
    t = pd.DataFrame({"k": ent[c].values, "y": y_ent.values}).groupby("k").y.agg(["sum", "count"])
    s = apl[c].map(t["sum"]).fillna(0).to_numpy()
    n = apl[c].map(t["count"]).fillna(0).to_numpy()
    return (s + M * prior) / (n + M)


def codificar_objetivo(df_ent, y_ent, df_apl, columnas):
    """Tasa de retraso de cada valor, suavizada hacia la de su nivel superior:

        tasa(v) = (retrasos(v) + M · tasa(padre)) / (n(v) + M)

    Se estima con (df_ent, y_ent) y se aplica a df_apl. Un valor que no estaba en df_ent recibe la tasa de su padre."""
    ent = {c: df_ent[c].astype(str) for c in JERARQUIA if c in df_ent.columns}
    apl = {c: df_apl[c].astype(str) for c in JERARQUIA if c in df_apl.columns}
    return pd.DataFrame({f"te_{c}": _codificar(c, ent, y_ent, apl, float(y_ent.mean())) for c in columnas},
                        index=df_apl.index)


def codificar_fuera_de_pliegue(df, y, grupos, columnas, pliegues=5):
    """La misma codificación en el entrenamiento, fuera de pliegue y agrupada por OC: una fila no ve su objetivo ni el
    de las posiciones de su misma orden."""
    out = pd.DataFrame(index=df.index, columns=[f"te_{c}" for c in columnas], dtype=float)
    for i_e, i_a in GroupKFold(n_splits=pliegues).split(df, y, grupos):
        out.iloc[i_a] = codificar_objetivo(df.iloc[i_e], y.iloc[i_e], df.iloc[i_a], columnas).to_numpy()
    return out


class CodificadorJerarquico(BaseEstimator, TransformerMixin):
    """Transformador de scikit-learn para las entidades. En ``fit_transform`` codifica fuera de pliegue (agrupado por
    OC, que se pasa en ``grupos``); en ``transform`` usa todo lo visto en ``fit``. Necesita las columnas de la jerarquía
    (las de nivel superior) en la entrada."""

    def __init__(self, columnas=None, grupos=None):
        self.columnas = columnas
        self.grupos = grupos

    def fit(self, X, y):
        self.X_, self.y_ = X.copy(), pd.Series(np.asarray(y), index=X.index)
        return self

    def transform(self, X):
        return codificar_objetivo(self.X_, self.y_, X, self.columnas).to_numpy()

    def fit_transform(self, X, y, **kw):
        self.fit(X, y)
        g = self.grupos if self.grupos is not None else np.arange(len(X))
        return codificar_fuera_de_pliegue(X, self.y_, pd.Series(np.asarray(g), index=X.index), self.columnas).to_numpy()

    def get_feature_names_out(self, entrada=None):
        return np.array([f"te_{c}" for c in self.columnas])


def a_objeto(X):
    """Texto de pandas (con pd.NA) a objeto con np.nan, que es lo que entienden los imputadores de scikit-learn."""
    return X.astype(object).where(X.notna(), np.nan)


def preprocesador(grupos=None):
    """El preprocesamiento del modelo, para ir dentro de un ``Pipeline`` y ajustarse solo con el entrenamiento:

    - numéricas: mediana y estandarización (los vacíos estructurales ya están marcados con tiene_*);
    - categóricas de pocos valores: one-hot, con las categorías que no estaban en el entrenamiento en ceros;
    - entidades: codificación jerárquica por objetivo, fuera de pliegue en el ajuste."""
    jer, pendientes = set(), list(ENTIDADES)            # las entidades y toda su cadena de niveles superiores
    while pendientes:
        c = pendientes.pop()
        if c and c not in jer:
            jer.add(c)
            pendientes.append(JERARQUIA.get(c))
    jer = sorted(jer)
    return ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]), NUMERICAS),
        ("cat", Pipeline([("objeto", FunctionTransformer(a_objeto, feature_names_out="one-to-one")),
                          ("imputar", SimpleImputer(strategy="constant", fill_value="(vacío)")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30))]), CATEGORICAS),
        ("ent", CodificadorJerarquico(columnas=ENTIDADES, grupos=grupos), jer),
    ])
