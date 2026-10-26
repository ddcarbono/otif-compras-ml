"""Las 112 combinaciones del Entregable 2: 7 modelos × 4 balanceos × 4 optimizadores, cada una en los 8 bloques externos
de ``validacion`` (896 trabajos).

Un **trabajo** es (modelo, balanceo, optimizador, bloque): el optimizador gasta E evaluaciones en la ventana interna
(entrena con el remuestreo que toque y mide el ROC-AUC en el semestre de validación), el umbral de máximo F1 se fija con
las puntuaciones de la mejor configuración en esa misma validación, y la mejor configuración se reajusta con la ventana
externa y se mide en el bloque. Escribe ``resultados/experimentos/trabajos/<modelo>__<balanceo>__<optimizador>__<bloque>.json``
con las métricas, los tiempos y el historial del optimizador. Si el archivo existe, el trabajo no se repite (reanudable).

Recortes de costo acordados para srvgdai (2026-10-22): logística y SVM lineal con ``SGDClassifier`` (pérdidas
``log_loss`` y ``hinge``), KNN sobre 30 componentes principales y Random Forest de 100 árboles. Todo en un hilo por
trabajo; el paralelismo es entre trabajos.

Balanceo ``pesos`` (``class_weight``): ``balanced`` en árbol, bosque y SGD; ``scale_pos_weight`` = negativos/positivos en
XGBoost; pesos por muestra en Naive Bayes; en KNN, los votos de cada clase se dividen por su frecuencia en el
entrenamiento (KNN no tiene pesos de clase en scikit-learn).

Uso, desde la raíz del repositorio y con ``OTIF_CSV`` definido::

    python -m src.experimentos preparar [--trabajos 8] [--bloques 2025-S2]  # matrices y remuestreos (cache/)
    python -m src.experimentos piloto [--E 2]                # un trabajo por modelo en el bloque más grande
    python -m src.experimentos correr --E 20 --trabajos 30   # los 896 trabajos
    python -m src.experimentos tabla                         # resultados/tabla_maestra.csv
"""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from . import datos, optimizadores, validacion  # noqa: E402

S = datos.SEMILLA
SALIDA = datos.RAIZ / "resultados" / "experimentos"
TRABAJOS = SALIDA / "trabajos"

# espacios de búsqueda: (nombre, tipo, extremos u opciones, escala log)
ESPACIOS = {
    "knn": [("n_neighbors", "int", (5, 300), True), ("weights", "cat", ("uniform", "distance"), False)],
    "naive_bayes": [("var_smoothing", "float", (1e-12, 1e-1), True)],
    "logistica": [("alpha", "float", (1e-7, 1e-2), True), ("penalty", "cat", ("l1", "l2"), False)],
    "arbol": [("max_depth", "int", (3, 30), False), ("min_samples_leaf", "int", (5, 2000), True),
              ("criterion", "cat", ("gini", "entropy"), False)],
    "random_forest": [("max_depth", "int", (5, 40), False), ("min_samples_leaf", "int", (1, 200), True),
                      ("max_features", "cat", ("sqrt", "log2", 0.05), False)],
    "xgboost": [("n_estimators", "int", (100, 600), False), ("learning_rate", "float", (0.01, 0.3), True),
                ("max_depth", "int", (3, 10), False), ("subsample", "float", (0.5, 1.0), False),
                ("colsample_bytree", "float", (0.3, 1.0), False), ("min_child_weight", "float", (1, 100), True)],
    "svm": [("alpha", "float", (1e-7, 1e-2), True), ("penalty", "cat", ("l1", "l2"), False)],
}
MODELOS = list(ESPACIOS)
# orden de lanzamiento: lo caro primero, para que el final no quede esperando un trabajo largo
MEMORIA_MIN_GB = float(os.environ.get("OTIF_MEMORIA_MIN_GB", 6))  # memoria libre para arrancar un trabajo
COSTO = {"knn": 5, "random_forest": 6, "xgboost": 7, "svm": 3, "logistica": 3, "arbol": 1, "naive_bayes": 0}


class KNNPesos(BaseEstimator, ClassifierMixin):
    """KNN cuyos votos se dividen por la frecuencia de cada clase en el entrenamiento: el ``class_weight`` de KNN."""

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors, self.weights = n_neighbors, weights

    def fit(self, X, y):
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights).fit(X, y)
        self.classes_ = self.knn_.classes_
        self.w_ = 1 / np.bincount(y) * len(y) / 2
        return self

    def predict_proba(self, X):
        p = self.knn_.predict_proba(X) * self.w_
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def construir(modelo, balanceo, p, y):
    """El estimador con los parámetros ``p``; devuelve (estimador, argumentos extra de fit)."""
    pesos = balanceo == "pesos"
    cw = "balanced" if pesos else None
    if modelo == "knn":
        k = (KNNPesos if pesos else KNeighborsClassifier)(n_neighbors=p["n_neighbors"], weights=p["weights"])
        return Pipeline([("pca", PCA(30, svd_solver="randomized", random_state=S)), ("knn", k)]), {}
    if modelo == "naive_bayes":
        sw = None
        if pesos:
            sw = (len(y) / (2 * np.bincount(y)))[y]
        return GaussianNB(var_smoothing=p["var_smoothing"]), {"sample_weight": sw}
    if modelo in ("logistica", "svm"):
        return SGDClassifier(loss="log_loss" if modelo == "logistica" else "hinge", alpha=p["alpha"],
                             penalty=p["penalty"], class_weight=cw, max_iter=30, tol=1e-3, random_state=S), {}
    if modelo == "arbol":
        return DecisionTreeClassifier(class_weight=cw, random_state=S, **p), {}
    if modelo == "random_forest":
        return RandomForestClassifier(n_estimators=100, class_weight=cw, n_jobs=1, random_state=S, **p), {}
    if modelo == "xgboost":
        spw = float((y == 0).sum() / (y == 1).sum()) if pesos else 1.0
        return XGBClassifier(tree_method="hist", n_jobs=1, random_state=S, scale_pos_weight=spw,
                             eval_metric="logloss", **p), {}
    raise ValueError(modelo)


def puntuar(est, X):
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


def umbral_f1(y, s):
    pr, rc, u = precision_recall_curve(y, s)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(u[np.argmax(f1)])


def pico_gb():
    if sys.platform.startswith("win"):
        return psutil.Process().memory_info().peak_wset / 2 ** 30
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def nombre_trabajo(modelo, balanceo, opt, bloque):
    return f"{modelo}__{balanceo}__{opt}__{bloque}"


def trabajo(modelo, balanceo, opt, bloque, E, destino=TRABAJOS):
    """Corre un trabajo y escribe su JSON. Devuelve el nombre y los segundos."""
    t0 = time.time()
    archivo = destino / f"{nombre_trabajo(modelo, balanceo, opt, bloque)}.json"
    while psutil.virtual_memory().available < MEMORIA_MIN_GB * 2 ** 30:  # no arrancar sin memoria para el trabajo
        time.sleep(30)
    Xi, yi, Xv, yv = validacion.cargar(bloque, "interno", balanceo)
    mejor = {"valor": -1.0, "s": None}

    def objetivo(p):
        est, kw = construir(modelo, balanceo, p, yi)
        est.fit(Xi, yi, **{k: v for k, v in kw.items() if v is not None})
        s = puntuar(est, Xv)
        v = roc_auc_score(yv, s)
        if v > mejor["valor"]:
            mejor.update(valor=v, s=s)
        return v

    p, hist = optimizadores.optimizar(opt, ESPACIOS[modelo], E, objetivo, S)
    t_opt = time.time() - t0
    umbral = umbral_f1(yv, mejor["s"])
    del Xi, yi, Xv, yv

    Xe, ye, Xb, yb = validacion.cargar(bloque, "externo", balanceo)
    est, kw = construir(modelo, balanceo, p, ye)
    t = time.time()
    est.fit(Xe, ye, **{k: v for k, v in kw.items() if v is not None})
    t_fit = time.time() - t
    t = time.time()
    s = puntuar(est, Xb)
    t_pred = time.time() - t
    fila = {"modelo": modelo, "balanceo": balanceo, "optimizador": opt, "bloque": bloque, "E": E,
            "AUC": roc_auc_score(yb, s), "PR_AUC": average_precision_score(yb, s), "F1": f1_score(yb, s >= umbral),
            "marcadas": float((s >= umbral).mean()), "umbral": umbral, "AUC_interno": mejor["valor"], "params": p,
            "filas_entrena": int(len(ye)), "filas_bloque": int(len(yb)), "segundos_optimizar": t_opt,
            "segundos_reajuste": t_fit, "ms_inferencia_por_1000": 1000 * t_pred / len(yb) * 1000,
            "segundos_total": time.time() - t0, "pico_gb": pico_gb(), "historial": hist}
    destino.mkdir(parents=True, exist_ok=True)
    tmp = archivo.with_suffix(".tmp")
    tmp.write_text(json.dumps(fila, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(archivo)
    return archivo.stem, fila["segundos_total"]


def _seguro(*a):
    try:
        return trabajo(*a)
    except Exception:
        return f"ERROR {a[:4]}", traceback.format_exc()


def _pool(n):
    kw = {"max_tasks_per_child": 1} if sys.version_info >= (3, 11) else {}  # memoria limpia y pico propio por trabajo
    return ProcessPoolExecutor(max_workers=n, **kw)


# --- preparar: matrices por ventana, después SMOTE y ADASYN ------------------------------------------------------------
_DF = None


def _preparar(bloque, fase):
    global _DF
    if validacion.listo(bloque, fase):
        return f"{bloque} {fase} ya estaba"
    if _DF is None:
        _DF = datos.cargar()
    return f"{bloque} {fase}: {validacion.preparar(_DF, bloque, fase):.0f} s"


def _remuestrear(bloque, fase, metodo):
    if validacion.listo(bloque, fase, metodo):
        return f"{bloque} {fase} {metodo} ya estaba"
    return f"{bloque} {fase} {metodo}: {validacion.remuestrear(bloque, fase, metodo):.0f} s"


def preparar(n, bloques=validacion.BLOQUES):
    tareas = [(b, f) for b in reversed(bloques) for f in ("externo", "interno")]
    with _pool(n) as ex:
        for r in as_completed([ex.submit(_preparar, *t) for t in tareas]):
            print(r.result(), flush=True)
    tareas = [(b, f, m) for m in ("adasyn", "smote") for b in reversed(bloques) for f in ("externo", "interno")]
    with _pool(n) as ex:
        for r in as_completed([ex.submit(_remuestrear, *t) for t in tareas]):
            print(r.result(), flush=True)


def correr(E, n, modelos=MODELOS, balanceos=validacion.BALANCEOS, opts=optimizadores.OPTIMIZADORES,
           bloques=validacion.BLOQUES, destino=TRABAJOS):
    tareas = [(m, b, o, k) for m in modelos for b in balanceos for o in opts for k in bloques
              if not (destino / f"{nombre_trabajo(m, b, o, k)}.json").exists()]
    # lo caro y los bloques grandes primero
    tareas.sort(key=lambda t: (-COSTO[t[0]], -validacion.BLOQUES.index(t[3]), t[1] == "ninguno"))
    print(f"{len(tareas)} trabajos pendientes, {n} a la vez, E = {E}", flush=True)
    t0, hechos = time.time(), 0
    with _pool(n) as ex:
        for r in as_completed([ex.submit(_seguro, *t, E, destino) for t in tareas]):
            nombre, x = r.result()
            hechos += 1
            if nombre.startswith("ERROR"):
                print(nombre, x, flush=True)
            else:
                print(f"[{hechos}/{len(tareas)}] {nombre}: {x:.0f} s (reloj {(time.time() - t0) / 3600:.2f} h)",
                      flush=True)


def tabla(destino=TRABAJOS):
    filas = [json.loads(f.read_text(encoding="utf-8")) for f in destino.glob("*.json")]
    df = pd.DataFrame(filas).drop(columns="historial")
    df.to_csv(SALIDA / "por_bloque.csv", index=False)
    g = df.groupby(["modelo", "balanceo", "optimizador"])
    t = g.agg(bloques=("bloque", "size"), AUC=("AUC", "mean"), AUC_desv=("AUC", "std"), F1=("F1", "mean"),
              F1_desv=("F1", "std"), PR_AUC=("PR_AUC", "mean"), AUC_interno=("AUC_interno", "mean"),
              horas_cpu=("segundos_total", lambda s: s.sum() / 3600),
              ms_inferencia_por_1000=("ms_inferencia_por_1000", "mean"), pico_gb=("pico_gb", "max")).reset_index()
    # hiperparámetros finales: los que eligió el optimizador en el bloque más reciente (2025-S2), con los que se
    # reentrena el modelo que se usa después
    ultimo = df[df.bloque == validacion.BLOQUES[-1]].set_index(["modelo", "balanceo", "optimizador"])
    t = t.join(ultimo[["E", "params"]].rename(columns={"E": "evaluaciones", "params": "hiperparametros_finales"}),
               on=["modelo", "balanceo", "optimizador"])
    t["hiperparametros_finales"] = t.hiperparametros_finales.map(lambda d: json.dumps(d, default=float))
    t["semilla"] = datos.SEMILLA
    t = t.sort_values("AUC", ascending=False)
    t.to_csv(datos.RAIZ / "resultados" / "tabla_maestra.csv", index=False)
    print(t.head(15).to_string(index=False))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["preparar", "piloto", "correr", "tabla"])
    ap.add_argument("--E", type=int, default=20)
    ap.add_argument("--trabajos", type=int, default=8)
    ap.add_argument("--modelos", nargs="*", default=MODELOS)
    ap.add_argument("--bloques", nargs="*", default=validacion.BLOQUES)
    a = ap.parse_args()
    if a.accion == "preparar":
        preparar(a.trabajos, a.bloques)
    elif a.accion == "piloto":
        # un trabajo por modelo en el bloque más grande, con SMOTE (la matriz más grande) y búsqueda aleatoria
        piloto = SALIDA / "piloto"
        correr(a.E, min(a.trabajos, len(a.modelos)), modelos=a.modelos, balanceos=["smote"], opts=["random"],
               bloques=["2025-S2"], destino=piloto)
        for f in sorted(piloto.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            ev = [h["segundos"] for h in r["historial"]]
            print(f"{r['modelo']:14s} {np.diff([0] + ev).mean():7.0f} s por evaluación, reajuste {r['segundos_reajuste']:6.0f} s, "
                  f"inferencia {r['ms_inferencia_por_1000']:6.1f} ms/1000, pico {r['pico_gb']:.1f} GB, AUC {r['AUC']:.3f}")
    elif a.accion == "correr":
        correr(a.E, a.trabajos, modelos=a.modelos)
    else:
        tabla()


if __name__ == "__main__":
    main()
