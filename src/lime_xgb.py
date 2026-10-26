"""LIME frente a SHAP en el XGBoost elegido, en la prueba 2026 (el Entregable 2 pide la comparación).

LIME (Ribeiro, Singh y Guestrin 2016) explica una predicción ajustando una regresión lineal ponderada a perturbaciones de
la fila: muestrea vecinos, los pondera por cercanía y lee los coeficientes. SHAP (TreeSHAP) reparte el logit exacto entre
las variables.

LIME se calcula en el espacio de las variables originales, no en las columnas preprocesadas: con las 594 columnas del
preprocesador (500 de ellas one-hot) LIME perturba cada indicador por separado, arma filas con varias categorías de una
misma variable encendidas a la vez y reparte peso entre indicadores de categorías que la fila no tiene; sumar eso por
variable infla las categóricas. Aquí cada variable es una sola columna categórica para LIME:

- una categórica o una entidad, con su código;
- una numérica, con su cuartil en el entrenamiento más un tramo propio para el vacío (el vacío informa: el 11 % de las
  posiciones no tiene solicitud de pedido y el 34 % de los materiales no tiene peso).

Un vecino que conserva el tramo de la fila conserva su valor exacto; uno que cambia de tramo toma un valor del
entrenamiento al azar dentro del tramo nuevo. Así la fila sin perturbar es la posición real, el modelo recibe filas
posibles y el peso de LIME de cada variable es el efecto de *tener el valor de la fila* frente a lo que pasa en el
entrenamiento, que es lo que mide SHAP.

Escribe en ``resultados/prueba_2026/``:

- ``lime_local.csv``: las tres posiciones de ``shap_local.csv`` (verdadero positivo, falso negativo y falso positivo, las
  de puntaje mediano), con el peso LIME y el valor SHAP de sus 8 variables de mayor |SHAP|;
- ``lime_acuerdo.csv``: en 200 posiciones de 2026 al azar, cuánto coinciden: correlación de Spearman entre |LIME| y |SHAP|
  por variable, cuántas de las 5 principales comparten y en cuántas de las 5 principales de SHAP coincide el signo;
- ``lime_estabilidad.csv``: LIME repetido con cinco semillas en las mismas tres posiciones: cuántas de sus 5 principales
  se mantienen (SHAP es exacto: no cambia).

Uso (ml_venv, con OTIF_CSV y ``artifacts/xgboost.joblib`` de ``src.modelo_final``): ``python -m src.lime_xgb``.
"""
import json

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from . import datos
from .modelo_final import contribuciones

P26 = datos.RAIZ / "resultados" / "prueba_2026"
MUESTRAS = 3000          # vecinos por explicación
FONDO = 20_000           # filas del entrenamiento que fijan tramos, frecuencias y valores de reemplazo


class Tramos:
    """Pasa las variables originales a códigos enteros (lo que perturba LIME) y de vuelta a una fila para el modelo."""

    def __init__(self, fondo, semilla):
        self.rng = np.random.default_rng(semilla)
        self.cortes, self.niveles, self.reserva = {}, {}, {}
        for v in datos.PREDICTORAS:
            x = fondo[v]
            if v in datos.NUMERICAS:
                q = np.unique(np.nanquantile(x.astype(float), [0.25, 0.5, 0.75]))
                self.cortes[v] = q
                c = self._numerica(v, x.to_numpy(float))
                self.reserva[v] = {k: x.to_numpy(float)[c == k] for k in np.unique(c)}
            else:
                self.niveles[v] = pd.Index(x.astype(object).unique())

    def _numerica(self, v, x):
        c = np.searchsorted(self.cortes[v], x, side="right")
        return np.where(np.isnan(x), len(self.cortes[v]) + 1, c)

    def codigos(self, X):
        """Filas × variables, enteros."""
        cols = []
        for v in datos.PREDICTORAS:
            if v in datos.NUMERICAS:
                cols.append(self._numerica(v, X[v].to_numpy(float)))
            else:
                k = self.niveles[v].get_indexer(X[v].astype(object))
                cols.append(np.where(k < 0, len(self.niveles[v]), k))          # nivel nuevo: un código propio
        return np.column_stack(cols)

    def filas(self, Z, fila):
        """Códigos perturbados -> DataFrame para el pipeline; donde el código es el de ``fila`` va su valor exacto."""
        z0 = self.codigos(fila)[0]
        out = {}
        for j, v in enumerate(datos.PREDICTORAS):
            z = Z[:, j].astype(int)
            igual = z == z0[j]
            if v in datos.NUMERICAS:
                x = np.full(len(z), np.nan)
                for k in np.unique(z[~igual]):
                    pool = self.reserva[v].get(k)
                    m = ~igual & (z == k)
                    if pool is not None and len(pool):
                        x[m] = self.rng.choice(pool, m.sum())
                x[igual] = fila[v].to_numpy(float)[0]
                out[v] = x
            else:
                niv = np.append(self.niveles[v].to_numpy(object), None)
                x = niv[np.minimum(z, len(niv) - 1)]
                x[igual] = fila[v].iloc[0]
                out[v] = x
        X = pd.DataFrame(out)
        X["oc_id"] = fila["oc_id"].iloc[0]
        return X


def main():
    cfg = json.loads((P26 / "config_xgb136.json").read_text(encoding="utf-8"))
    pipe = joblib.load(datos.RAIZ / "artifacts" / "xgboost.joblib")
    df = datos.cargar()
    ent, pru = df[df.particion.ne("prueba")], datos.particion(df, "prueba").reset_index(drop=True)
    fondo = ent[datos.ENTRADA].sample(FONDO, random_state=datos.SEMILLA)
    tramos = Tramos(fondo, datos.SEMILLA)
    Zf = tramos.codigos(fondo)
    y = pru[datos.OBJETIVO].astype(int).to_numpy()
    p = pipe.predict_proba(pru[datos.ENTRADA])[:, 1]
    shap, _ = contribuciones(pipe, pru[datos.ENTRADA])
    todas = list(range(len(datos.PREDICTORAS)))

    def explicar(i, semilla):
        fila = pru.loc[[i], datos.ENTRADA]
        ex = LimeTabularExplainer(Zf, feature_names=datos.PREDICTORAS, class_names=["a tiempo", "tarde"],
                                  categorical_features=todas, discretize_continuous=False, random_state=semilla)
        e = ex.explain_instance(tramos.codigos(fila)[0], lambda Z: pipe.predict_proba(tramos.filas(Z, fila)),
                                num_features=len(todas), num_samples=MUESTRAS)
        w = pd.Series(0.0, index=datos.PREDICTORAS)
        for j, peso in e.as_map()[1]:
            w.iloc[j] = peso
        return w

    # las tres posiciones de shap_local.csv
    marcada = p >= cfg["umbral"]
    filas, estab = [], []
    for caso, m in [("verdadero positivo", (y == 1) & marcada), ("falso negativo", (y == 1) & ~marcada),
                    ("falso positivo", (y == 0) & marcada)]:
        idx = np.flatnonzero(m)
        i = idx[np.argsort(p[idx])[len(idx) // 2]]
        l = explicar(i, datos.SEMILLA)
        s = shap.iloc[i]
        for v in s.abs().sort_values(ascending=False).index[:8]:
            filas.append({"caso": caso, "probabilidad": p[i], "variable": v, "shap": s[v], "lime": l.get(v, 0.0)})
        tops = [set(explicar(i, datos.SEMILLA + k).abs().sort_values(ascending=False).index[:5]) for k in range(5)]
        comun = set.intersection(*tops)
        estab.append({"caso": caso, "semillas": 5, "en las 5 principales con todas las semillas": len(comun),
                      "distintas que aparecen entre las 5 principales": len(set.union(*tops))})
        print(caso, "| SHAP:", list(s.abs().sort_values(ascending=False).index[:5]),
              "| LIME:", list(l.abs().sort_values(ascending=False).index[:5]), flush=True)
    pd.DataFrame(filas).to_csv(P26 / "lime_local.csv", index=False)
    pd.DataFrame(estab).to_csv(P26 / "lime_estabilidad.csv", index=False)

    # acuerdo en 200 posiciones al azar
    rng = np.random.default_rng(datos.SEMILLA)
    acuerdo = []
    for i in rng.choice(len(pru), 200, replace=False):
        l, s = explicar(i, datos.SEMILLA), shap.iloc[i]
        l = l.reindex(s.index).fillna(0.0)
        ts, tl = s.abs().sort_values(ascending=False).index[:5], l.abs().sort_values(ascending=False).index[:5]
        acuerdo.append({"spearman_abs": spearmanr(s.abs(), l.abs()).correlation,
                        "comparten_de_5": len(set(ts) & set(tl)),
                        "mismo_signo_en_5_de_shap": int((np.sign(s[ts]) == np.sign(l[ts])).sum())})
    a = pd.DataFrame(acuerdo)
    resumen = pd.DataFrame({"medida": ["Spearman |LIME| contra |SHAP| por variable", "de las 5 principales, comparten",
                                       "de las 5 de SHAP, mismo signo en LIME"],
                            "media": a.mean().to_numpy(), "p25": a.quantile(0.25).to_numpy(),
                            "p75": a.quantile(0.75).to_numpy(), "posiciones": len(a)})
    resumen.to_csv(P26 / "lime_acuerdo.csv", index=False)
    print(resumen.round(3).to_string(index=False))
    print(pd.DataFrame(estab).to_string(index=False))


if __name__ == "__main__":
    main()
