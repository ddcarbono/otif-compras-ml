"""Estabilidad ante la semilla y curva de convergencia de XGBoost, en los 8 bloques externos (sin tocar la prueba 2026).

- ``semillas.csv``: la mejor combinación de XGBoost, el bosque, la logística y el SVM (la de la tabla de líderes), con
  los hiperparámetros que ya eligió su optimizador, reajustada con 10 semillas más (2027 a 2036) en cada bloque. Solo
  cambia la semilla del modelo: el remuestreo SMOTE es el de la caché (semilla del proyecto). Responde si la ventaja de
  0,009 de XGBoost sobre el bosque es mayor que lo que mueve la semilla.
- ``convergencia.csv``: XGBoost (mejor combinación) en cada bloque, con el AUC y la log-loss en entrenamiento y en el
  semestre externo después de cada árbol. Es la brecha entrenamiento-validación por iteración.

Escribe en ``resultados/estabilidad/``. Uso (srvgdai, después de ``experimentos preparar``):
``python -m src.estabilidad [--trabajos 8]``.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import datos, experimentos, validacion
from .experimentos import TRABAJOS, construir, nombre_trabajo, puntuar

SALIDA = datos.RAIZ / "resultados" / "estabilidad"
SEMILLAS = range(2027, 2037)
MODELOS = ("xgboost", "random_forest", "logistica", "svm")


def mejores():
    t = pd.read_csv(datos.RAIZ / "resultados" / "tabla_maestra.csv").sort_values("AUC", ascending=False)
    t = t.groupby("modelo").head(1).set_index("modelo")
    return {m: (t.loc[m, "balanceo"], t.loc[m, "optimizador"]) for m in MODELOS}


def _params(modelo, balanceo, opt, bloque):
    r = json.loads((TRABAJOS / f"{nombre_trabajo(modelo, balanceo, opt, bloque)}.json").read_text(encoding="utf-8"))
    return r["params"]


def con_semilla(modelo, balanceo, opt, bloque, semilla):
    experimentos.S = semilla        # construir() lee la semilla del módulo al construir el estimador
    Xe, ye, Xb, yb = validacion.cargar(bloque, "externo", balanceo)
    est, kw = construir(modelo, balanceo, _params(modelo, balanceo, opt, bloque), ye)
    est.fit(Xe, ye, **{k: v for k, v in kw.items() if v is not None})
    return {"modelo": modelo, "bloque": bloque, "semilla": semilla, "AUC": roc_auc_score(yb, puntuar(est, Xb))}


def convergencia(balanceo, opt, bloque):
    Xe, ye, Xb, yb = validacion.cargar(bloque, "externo", balanceo)
    est, kw = construir("xgboost", balanceo, _params("xgboost", balanceo, opt, bloque), ye)
    est.set_params(eval_metric=["auc", "logloss"])
    est.fit(Xe, ye, eval_set=[(Xe, ye), (Xb, yb)], verbose=False)
    r = est.evals_result()
    n = len(r["validation_0"]["auc"])
    return pd.DataFrame({"bloque": bloque, "arbol": np.arange(1, n + 1),
                         "AUC_entrenamiento": r["validation_0"]["auc"], "AUC_externo": r["validation_1"]["auc"],
                         "logloss_entrenamiento": r["validation_0"]["logloss"],
                         "logloss_externo": r["validation_1"]["logloss"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trabajos", type=int, default=8)
    a = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    m = mejores()
    with ProcessPoolExecutor(max_workers=a.trabajos) as ex:
        conv = [ex.submit(convergencia, *m["xgboost"], b) for b in validacion.BLOQUES]
        fut = [ex.submit(con_semilla, mod, *m[mod], b, s) for mod in MODELOS for b in validacion.BLOQUES
               for s in SEMILLAS]
        filas = []
        for i, f in enumerate(as_completed(fut), 1):
            filas.append(f.result())
            if i % 40 == 0:
                print(f"{i} de {len(fut)} ajustes con semilla", flush=True)
        pd.concat([f.result() for f in conv]).to_csv(SALIDA / "convergencia.csv", index=False)
    pd.DataFrame(filas).sort_values(["modelo", "bloque", "semilla"]).to_csv(SALIDA / "semillas.csv", index=False)
    print("escrito", SALIDA)


if __name__ == "__main__":
    main()
