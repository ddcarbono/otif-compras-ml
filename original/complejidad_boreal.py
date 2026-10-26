"""Tiempo de ajuste y de predicción de BOREAL contra n (filas) y p (columnas), para contrastar el análisis O(·) del
capítulo del modelo original con lo medido (el Entregable 3 lo pide, como la sección de complejidad del Entregable 2).

Datos sintéticos como los de ``tests/test_boreal.py`` (retraso log-normal con un efecto real por proveedor), con
hiperparámetros fijos y pocos árboles: interesa cómo crece el tiempo, no el tiempo de una corrida completa. Se mide en
un solo proceso (``BOREAL_PROCESOS=1``) y se repite cada punto tres veces (mediana).

- n de 2.000 a 64.000 con p = 10;
- p de 5 a 80 con n = 16.000.

El exponente empírico es la pendiente de log(tiempo) contra log(n) o log(p). La teoría dice O(n·p) por árbol para el
ajuste (histogramas) y O(profundidad) por fila para predecir, más O(n) por paso de Newton en los efectos.

Escribe ``resultados/original/complejidad_boreal.csv`` (una fila por punto) y ``complejidad_boreal_pendientes.csv``.
Uso (desde la raíz del repositorio): ``python original/complejidad_boreal.py``.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BOREAL_PROCESOS", "1")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modelo_original_otif import Boreal  # noqa: E402

SALIDA = AQUI.parent / "resultados" / "original"
REPETICIONES = 3


def datos(n, p, semilla=7):
    rng = np.random.default_rng(semilla)
    prov = rng.integers(0, max(20, n // 50), n)
    efecto = rng.normal(0, 1.0, prov.max() + 1)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{j}" for j in range(p)])
    dias = np.exp(0.9 * X.x0 - 0.6 * X.x1 + efecto[prov] + rng.normal(0, 0.8, n)) - 1.5
    X = X.assign(proveedor=prov.astype(str), material="m", sp_creador="s", creador_oc="c")
    return X, (dias > 1).astype(int).to_numpy(), dias.to_numpy()


def medir(n, p):
    X, y, dias = datos(n, p)
    ajuste, prediccion = [], []
    for _ in range(REPETICIONES):
        m = Boreal(n_estimators=50, arboles_efectos=20, vueltas=1, num_leaves=31, min_child_samples=20,
                   learning_rate=0.1)
        t0 = time.perf_counter()
        m.fit(X, y, dias=dias)
        t1 = time.perf_counter()
        m.predict_proba(X)
        t2 = time.perf_counter()
        ajuste.append(t1 - t0)
        prediccion.append(t2 - t1)
    return {"n": n, "p": p, "segundos_ajuste": float(np.median(ajuste)),
            "ms_prediccion_por_1000": 1e6 * float(np.median(prediccion)) / n}


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas = []
    for n in [2000, 4000, 8000, 16000, 32000, 64000]:
        filas.append(dict(medir(n, 10), barrido="n"))
        print(filas[-1], flush=True)
    for p in [5, 10, 20, 40, 80]:
        filas.append(dict(medir(16000, p), barrido="p"))
        print(filas[-1], flush=True)
    d = pd.DataFrame(filas)
    d.to_csv(SALIDA / "complejidad_boreal.csv", index=False)
    pend = []
    for b in ("n", "p"):
        s = d[d.barrido == b]
        for c in ("segundos_ajuste", "ms_prediccion_por_1000"):
            pend.append({"barrido": b, "medida": c, "exponente": float(np.polyfit(np.log(s[b]), np.log(s[c]), 1)[0])})
    pend = pd.DataFrame(pend)
    pend.to_csv(SALIDA / "complejidad_boreal_pendientes.csv", index=False)
    print(pend.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
