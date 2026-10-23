"""Validación anidada de las 112 combinaciones y caché de matrices preprocesadas.

**Bucle externo**: ventana creciente por semestre. Ocho bloques de evaluación, de 2022-S1 a 2025-S2; cada uno entrena con
todos los semestres anteriores (desde 2021-S1). **Bucle interno**: dentro de la ventana de entrenamiento de cada bloque,
el último semestre valida y los anteriores entrenan; ahí los optimizadores eligen hiperparámetros. La prueba 2026 no
entra en nada de esto. Como la partición es por fecha de la orden, ninguna orden queda partida entre entrenamiento y
evaluación.

El preprocesamiento (``datos.preprocesador``) no tiene hiperparámetros: se ajusta una vez por ventana con su
entrenamiento y su salida se guarda en disco en float32 (``cache/<bloque>/<fase>_*.npy``). SMOTE y ADASYN también se
calculan una vez por ventana, solo sobre su entrenamiento, y se guardan. Cada trabajo de ``experimentos`` lee esas
matrices con ``mmap_mode="r"`` y solo ajusta el modelo: así entran 30 trabajos a la vez en la memoria de srvgdai.

Las clases sintéticas de SMOTE y ADASYN se generan sobre la matriz ya codificada (one-hot incluido). Las columnas
one-hot de una fila sintética quedan entre 0 y 1: es la aproximación usual cuando SMOTENC sería demasiado costoso con
42 categóricas.
"""
import time

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from scipy import sparse

from . import datos

BLOQUES = [f"{a}-S{s}" for a in range(2022, 2026) for s in (1, 2)]
BALANCEOS = ["ninguno", "smote", "adasyn", "pesos"]
CACHE = datos.RAIZ / "cache"


def semestre(df):
    f = df.fecha_pedido
    return f.dt.year.astype(str) + "-S" + np.where(f.dt.month <= 6, "1", "2")


def anterior(s):
    a, k = int(s[:4]), int(s[-1])
    return f"{a}-S1" if k == 2 else f"{a - 1}-S2"


def ventanas(df, bloque):
    """Las dos particiones de un bloque: ``interno`` (entrena hasta el semestre previo al anterior y valida con el
    anterior) y ``externo`` (entrena con todo lo anterior al bloque y evalúa en el bloque)."""
    df = df[df.particion.ne("prueba")]
    sem = semestre(df)
    val = anterior(bloque)
    return {"interno": (df[sem < val], df[sem == val]), "externo": (df[sem < bloque], df[sem == bloque])}


def _denso(X):
    return (X.toarray() if sparse.issparse(X) else np.asarray(X)).astype(np.float32)


def preparar(df, bloque, fase):
    """Preprocesa una ventana y guarda X e y de entrenamiento y de evaluación. Devuelve segundos."""
    t = time.time()
    d = CACHE / bloque
    d.mkdir(parents=True, exist_ok=True)
    ent, eva = ventanas(df, bloque)[fase]
    pre = datos.preprocesador()
    X = _denso(pre.fit_transform(ent[datos.ENTRADA], ent.y_tarde))
    np.save(d / f"{fase}_X_ent.npy", X)
    np.save(d / f"{fase}_y_ent.npy", ent.y_tarde.to_numpy(np.int8))
    np.save(d / f"{fase}_X_eva.npy", _denso(pre.transform(eva[datos.ENTRADA])))
    np.save(d / f"{fase}_y_eva.npy", eva.y_tarde.to_numpy(np.int8))
    return time.time() - t


def remuestrear(bloque, fase, metodo):
    """SMOTE o ADASYN sobre el entrenamiento ya preprocesado de una ventana (5 vecinos, clase minoritaria hasta igualar
    a la mayoritaria). Devuelve segundos."""
    t = time.time()
    d = CACHE / bloque
    X, y = np.load(d / f"{fase}_X_ent.npy"), np.load(d / f"{fase}_y_ent.npy")
    m = SMOTE(random_state=datos.SEMILLA) if metodo == "smote" else ADASYN(random_state=datos.SEMILLA)
    Xr, yr = m.fit_resample(X, y)
    np.save(d / f"{fase}_X_ent_{metodo}.npy", Xr.astype(np.float32))
    np.save(d / f"{fase}_y_ent_{metodo}.npy", yr.astype(np.int8))
    return time.time() - t


def listo(bloque, fase, metodo=None):
    suf = f"_{metodo}" if metodo else ""
    return (CACHE / bloque / f"{fase}_y_ent{suf}.npy").exists()


def cargar(bloque, fase, balanceo):
    """Matrices de un trabajo, en modo memmap. ``pesos`` usa el entrenamiento sin remuestrear (los pesos los pone el
    modelo)."""
    d = CACHE / bloque
    suf = f"_{balanceo}" if balanceo in ("smote", "adasyn") else ""
    leer = lambda n: np.load(d / f"{fase}_{n}.npy", mmap_mode="r")  # noqa: E731
    return leer(f"X_ent{suf}"), np.asarray(leer(f"y_ent{suf}")), leer("X_eva"), np.asarray(leer("y_eva"))
