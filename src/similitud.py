"""Qué tan distinta es la evaluación del entrenamiento: partición cronológica contra partición aleatoria por orden.

**Validación adversaria.** Se juntan dos conjuntos, se etiqueta de dónde viene cada posición y se entrena un clasificador
(XGBoost, validación cruzada de 5 pliegues agrupada por orden) para distinguirlos con las predictoras. Si el AUC es 0,5
los conjuntos son intercambiables; cuanto más sube, más distinto es lo que el modelo tendrá que predecir de lo que vio.

- **Cronológica**: entrenamiento 2021-2024 contra validación 2025 (lo que hace el proyecto).
- **Cronológica sin variables de reloj** (``RELOJ``): la misma, sin las predictoras que fechan la fila por sí solas.
- **Aleatoria**: dentro de 2021-2025, un conjunto de órdenes al azar del mismo tamaño que 2025 contra el resto (lo que
  haría una partición 80/20 sin fecha).

Usa las numéricas y las categóricas; deja fuera las entidades, porque su codificación necesita el objetivo. La prueba
2026 no se usa. Escribe ``resultados/similitud/{auc,importancia}.csv``. Uso: ``python -m src.similitud``.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

from . import datos

SALIDA = datos.RAIZ / "resultados" / "similitud"
# variables de reloj: crecen con el calendario (conteos acumulados, antigüedades) o fijan la fecha (tasa de cambio, mes).
# Separan cualquier corte cronológico por sí solas; la tercera variante las quita para ver la deriva del resto.
RELOJ = ["prov_n_previas", "mat_n_previas", "cmp_n_previas", "sol_n_previas", "pv_antiguedad_dias", "ps_antiguedad_dias",
         "prov_mat_oc_previas", "prov_centro_oc_previas", "trm_usd", "trm_usd_var30", "trm_usd_var90", "trm_eur",
         "trm_eur_var30", "trm_eur_var90", "trm_usd_vol30", "mes"]


def pre(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("log", FunctionTransformer(datos.log_signo, feature_names_out="one-to-one")),
                          ("imputar", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("objeto", FunctionTransformer(datos.a_objeto, feature_names_out="one-to-one")),
                          ("imputar", SimpleImputer(strategy="constant", fill_value="(vacío)")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30))]), cat)])


def adversaria(a, b, nombre, sin=()):
    num = [c for c in datos.NUMERICAS if c not in sin]
    cat = [c for c in datos.CATEGORICAS if c not in sin]
    df = pd.concat([a.assign(_b=0), b.assign(_b=1)], ignore_index=True)
    y, g = df._b.to_numpy(), df.oc_id.to_numpy()
    p = np.zeros(len(df))
    imp = []
    for tr, te in GroupKFold(5).split(df, y, g):
        t = pre(num, cat)
        Xtr = t.fit_transform(df.iloc[tr][num + cat])
        m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=4,
                          random_state=datos.SEMILLA).fit(Xtr, y[tr])
        p[te] = m.predict_proba(t.transform(df.iloc[te][num + cat]))[:, 1]
        imp.append(pd.Series(m.feature_importances_, index=t.get_feature_names_out()))
    i = pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False).head(15).rename("importancia").reset_index()
    i.columns = ["columna", "importancia"]
    i.insert(0, "particion", nombre)
    return {"particion": nombre, "filas_a": len(a), "filas_b": len(b), "AUC": roc_auc_score(y, p)}, i


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    # solo las columnas que se usan: con el dataset entero, las copias de cada variante no caben en 16 GB
    df = datos.cargar()[datos.NUMERICAS + datos.CATEGORICAS + ["oc_id", "particion"]]
    df = df[df.particion.ne("prueba")]
    ent, val = datos.particion(df, "entrenamiento"), datos.particion(df, "validacion")
    pool = pd.concat([ent, val])
    oc = pool.oc_id.unique()
    rng = np.random.default_rng(datos.SEMILLA)
    elegidas = set(rng.choice(oc, size=val.oc_id.nunique(), replace=False))
    filas, imps = [], []
    for nombre, a, b, sin in [
            ("cronológica: 2021-2024 contra 2025", ent, val, ()),
            ("cronológica, sin variables de reloj", ent, val, RELOJ),
            ("aleatoria por orden, mismo tamaño", pool[~pool.oc_id.isin(elegidas)], pool[pool.oc_id.isin(elegidas)], ())]:
        r, i = adversaria(a, b, nombre, sin)
        print(r, flush=True)
        filas.append(r)
        imps.append(i)
    pd.DataFrame(filas).to_csv(SALIDA / "auc.csv", index=False)
    pd.concat(imps).to_csv(SALIDA / "importancia.csv", index=False)


if __name__ == "__main__":
    main()
