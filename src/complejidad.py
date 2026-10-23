"""Complejidad medida: tiempo de ajuste y de inferencia de los 7 modelos contra el número de filas de entrenamiento, con
los hiperparámetros fijos (los que eligió Optuna sin balanceo en el bloque 2025-S2) y un hilo por modelo.

Usa la matriz ya preprocesada del bloque externo 2025-S2 (``cache/``, la deja ``experimentos preparar``): entrena con
1/8, 1/4, 1/2 y todas sus filas (311.865) y predice las 42.320 del bloque. La pendiente de log(tiempo) contra log(n) es
el exponente empírico que se compara con la complejidad teórica en el capítulo de fundamentos.

Uso (srvgdai): ``nice -n 19 python -m src.complejidad`` -> ``resultados/complejidad.csv``.
"""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import datos, experimentos, validacion  # noqa: E402

BLOQUE = "2025-S2"
FRACCIONES = [0.125, 0.25, 0.5, 1.0]


def medir(modelo):
    X, y, Xb, yb = validacion.cargar(BLOQUE, "externo", "ninguno")
    p = json.loads((experimentos.TRABAJOS / f"{modelo}__ninguno__optuna__{BLOQUE}.json").read_text(encoding="utf-8"))["params"]
    rng = np.random.default_rng(datos.SEMILLA)
    orden = rng.permutation(len(y))
    filas = []
    for f in FRACCIONES:
        idx = np.sort(orden[: int(f * len(y))])
        Xs, ys = np.asarray(X[idx]), y[idx]
        est, kw = experimentos.construir(modelo, "ninguno", p, ys)
        t = time.perf_counter()
        est.fit(Xs, ys)
        ajuste = time.perf_counter() - t
        t = time.perf_counter()
        experimentos.puntuar(est, Xb)
        pred = time.perf_counter() - t
        filas.append({"modelo": modelo, "filas": len(ys), "columnas": X.shape[1], "segundos_ajuste": ajuste,
                      "ms_inferencia_por_1000": 1e6 * pred / len(yb), "params": json.dumps(p)})
        print(filas[-1], flush=True)
    return filas


def main():
    with ProcessPoolExecutor(len(experimentos.MODELOS)) as ex:
        filas = [f for r in ex.map(medir, experimentos.MODELOS) for f in r]
    d = pd.DataFrame(filas)
    d.to_csv(datos.RAIZ / "resultados" / "complejidad.csv", index=False)
    for m, g in d.groupby("modelo"):
        b = np.polyfit(np.log(g.filas), np.log(g.segundos_ajuste), 1)[0]
        print(f"{m:14s} exponente {b:.2f}")


if __name__ == "__main__":
    main()
