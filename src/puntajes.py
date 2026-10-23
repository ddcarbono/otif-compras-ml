"""Puntajes por posición de la mejor combinación de cada modelo en cada bloque externo, para las pruebas que necesitan
las predicciones y no solo el AUC (DeLong, intervalos BCa).

Para cada uno de los 7 modelos toma su mejor combinación (balanceo y optimizador, por la media del AUC externo de
``resultados/tabla_maestra.csv``) y, en cada uno de los 8 bloques, la reajusta con los hiperparámetros que ya eligió su
optimizador (los del JSON del trabajo) sobre la misma matriz externa. No optimiza nada otra vez. Como todo lleva la
semilla del proyecto, el AUC reproducido tiene que coincidir con el del JSON hasta 1e-5 (KNN se aparta en la sexta
cifra: sus empates de distancia dependen del orden de las sumas en coma flotante); si no, se detiene. Cada reajuste se
guarda en ``resultados/puntajes/parciales/`` al terminar, y una corrida nueva salta los que ya están.

Escribe ``resultados/puntajes/<bloque>.csv.gz`` con ``y``, ``grupo`` (la orden de cada fila, como un entero propio del
bloque: no viaja el número de la orden) y una columna de puntaje por modelo. La prueba 2026 no se usa.
Uso (srvgdai, después de ``experimentos preparar``): ``python -m src.puntajes [--trabajos 8]``.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import datos, validacion
from .experimentos import TRABAJOS, construir, nombre_trabajo, puntuar

SALIDA = datos.RAIZ / "resultados" / "puntajes"
PARCIALES = SALIDA / "parciales"
TOLERANCIA = 1e-5


def mejores():
    t = pd.read_csv(datos.RAIZ / "resultados" / "tabla_maestra.csv").sort_values("AUC", ascending=False)
    return t.groupby("modelo").head(1)[["modelo", "balanceo", "optimizador"]].itertuples(index=False)


def puntaje(modelo, balanceo, opt, bloque):
    hecho = PARCIALES / f"{modelo}_{bloque}.npy"
    if hecho.exists():
        s = np.load(hecho)
        return modelo, bloque, s, None
    r = json.loads((TRABAJOS / f"{nombre_trabajo(modelo, balanceo, opt, bloque)}.json").read_text(encoding="utf-8"))
    Xe, ye, Xb, yb = validacion.cargar(bloque, "externo", balanceo)
    est, kw = construir(modelo, balanceo, r["params"], ye)
    est.fit(Xe, ye, **{k: v for k, v in kw.items() if v is not None})
    s = puntuar(est, Xb)
    auc = roc_auc_score(yb, s)
    if abs(auc - r["AUC"]) > TOLERANCIA:
        raise RuntimeError(f"{modelo} {bloque}: AUC reproducido {auc:.6f} contra {r['AUC']:.6f} del trabajo")
    np.save(hecho, s.astype(np.float64))
    return modelo, bloque, s.astype(np.float64), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trabajos", type=int, default=8)
    a = ap.parse_args()
    PARCIALES.mkdir(parents=True, exist_ok=True)
    df = datos.cargar()[["oc_id", "particion", "fecha_pedido", datos.OBJETIVO]]
    salida = {}
    for b in validacion.BLOQUES:
        eva = validacion.ventanas(df, b)["externo"][1]
        salida[b] = pd.DataFrame({"y": eva[datos.OBJETIVO].to_numpy(np.int8),
                                  "grupo": pd.factorize(eva.oc_id)[0].astype(np.int32)})
    tareas = [(m.modelo, m.balanceo, m.optimizador, b) for m in mejores() for b in validacion.BLOQUES]
    with ProcessPoolExecutor(max_workers=a.trabajos) as ex:
        for f in as_completed([ex.submit(puntaje, *t) for t in tareas]):
            modelo, bloque, s, auc = f.result()
            assert len(s) == len(salida[bloque])
            salida[bloque][modelo] = s
            print(f"{modelo:14s} {bloque}: " + ("ya estaba" if auc is None else f"AUC {auc:.4f} (coincide con el trabajo)"),
                  flush=True)
    for b, d in salida.items():
        assert (np.asarray(validacion.cargar(b, "externo", "ninguno")[3]) == d.y.to_numpy()).all()
        d.to_csv(SALIDA / f"{b}.csv.gz", index=False)
    print("escrito", SALIDA)


if __name__ == "__main__":
    main()
