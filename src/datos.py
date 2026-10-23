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
ENTRADA = PREDICTORAS + ["oc_id"]          # lo que recibe el preprocesador: oc_id solo agrupa, no es predictora

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
    try:
        import pyarrow  # noqa: F401  (sin pyarrow, como en srvgdai, se lee el CSV cada vez)
    except ImportError:
        pq = None
    if pq is not None and pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(ruta, sep=";", decimal=",", low_memory=False)
        for c in CATEGORICAS + ENTIDADES + ["oc_id", "particion"]:
            df[c] = df[c].astype("string")
        if pq is not None:
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


def tablas_objetivo(df_ent, y_ent):
    """Retrasos y filas por valor de cada columna de la jerarquía, y la tasa global: lo único que hay que guardar para
    codificar después (no el entrenamiento entero)."""
    y = pd.Series(np.asarray(y_ent), index=df_ent.index)
    tablas = {c: y.groupby(df_ent[c].astype(str).to_numpy()).agg(["sum", "count"]) for c in JERARQUIA if c in df_ent}
    return tablas, float(y.mean())


def aplicar_tablas(tablas, glob, df_apl, columnas):
    """La codificación jerárquica de ``df_apl`` con tablas ya estimadas (la misma fórmula de ``codificar_objetivo``)."""
    def cod(c):
        padre = JERARQUIA.get(c)
        prior = cod(padre) if padre else np.full(len(df_apl), glob)
        k = df_apl[c].astype(str)
        s = k.map(tablas[c]["sum"]).fillna(0).to_numpy()
        n = k.map(tablas[c]["count"]).fillna(0).to_numpy()
        return (s + M * prior) / (n + M)
    return pd.DataFrame({f"te_{c}": cod(c) for c in columnas}, index=df_apl.index)


class CodificadorJerarquico(BaseEstimator, TransformerMixin):
    """Transformador de scikit-learn para las entidades. Recibe las columnas de la jerarquía más ``oc_id``, que usa solo
    para agrupar. En ``fit_transform`` codifica fuera de pliegue y agrupado por OC (una fila no ve su objetivo ni el de
    su orden); en ``transform`` aplica las tablas estimadas con todo lo visto en ``fit``. Guarda solo esas tablas."""

    def __init__(self, columnas=None, col_grupo="oc_id"):
        self.columnas = columnas
        self.col_grupo = col_grupo

    def fit(self, X, y):
        self.tablas_, self.glob_ = tablas_objetivo(X, y)
        return self

    def transform(self, X):
        return aplicar_tablas(self.tablas_, self.glob_, X, self.columnas).to_numpy()

    def fit_transform(self, X, y, **kw):
        self.fit(X, y)
        y = pd.Series(np.asarray(y), index=X.index)
        return codificar_fuera_de_pliegue(X, y, X[self.col_grupo], self.columnas).to_numpy()

    def get_feature_names_out(self, entrada=None):
        return np.array([f"te_{c}" for c in self.columnas])


def log_signo(X):
    """signo(x) · log(1 + |x|): comprime las colas largas (montos, cantidades, conteos) sin cambiar el orden ni el signo.
    Es monótona, así que a los árboles no les cambia nada; a los modelos lineales les quita el peso de los extremos."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def a_objeto(X):
    """Texto de pandas (con pd.NA) a objeto con np.nan, que es lo que entienden los imputadores de scikit-learn."""
    return X.astype(object).where(X.notna(), np.nan)


def preprocesador():
    """El preprocesamiento del modelo, para ir dentro de un ``Pipeline`` y ajustarse solo con el entrenamiento:

    - numéricas: signo · log(1 + |x|), mediana y estandarización (los vacíos estructurales ya están marcados con tiene_*);
    - categóricas de pocos valores: one-hot, con las categorías que no estaban en el entrenamiento en ceros;
    - entidades: codificación jerárquica por objetivo, fuera de pliegue y agrupada por OC en el ajuste.

    La entrada es ``df[ENTRADA]``: las predictoras más ``oc_id``, que solo sirve para agrupar."""
    jer, pendientes = set(), list(ENTIDADES)            # las entidades y toda su cadena de niveles superiores
    while pendientes:
        c = pendientes.pop()
        if c and c not in jer:
            jer.add(c)
            pendientes.append(JERARQUIA.get(c))
    jer = sorted(jer)
    return ColumnTransformer([
        ("num", Pipeline([("log", FunctionTransformer(log_signo, feature_names_out="one-to-one")),
                          ("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]), NUMERICAS),
        ("cat", Pipeline([("objeto", FunctionTransformer(a_objeto, feature_names_out="one-to-one")),
                          ("imputar", SimpleImputer(strategy="constant", fill_value="(vacío)")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30))]), CATEGORICAS),
        ("ent", CodificadorJerarquico(columnas=ENTIDADES), jer + ["oc_id"]),
    ])
