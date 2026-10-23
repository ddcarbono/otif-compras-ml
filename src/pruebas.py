"""Pruebas estadísticas sobre predicciones: DeLong para dos AUC correlacionados, delta de Cliff e intervalos BCa.

- ``delong``: compara dos AUC medidos sobre las mismas filas (DeLong, DeLong y Clarke-Pearson, 1988), con el algoritmo
  rápido de Sun y Xu (2014). Un empate entre una tardía y una puntual cuenta 0,5. Validada contra ``pROC::roc.test``
  (``paired = TRUE``, ``method = "delong"``) en ``src/validar_delong.R``; los valores de pROC quedan en
  ``tests/datos/delong_proc.csv`` para que pytest compare sin R.
- ``cliff``: delta de Cliff entre dos muestras, P(a > b) − P(a < b), con la escala de Romano y otros (2006).
- ``bca``: intervalo bootstrap corregido por sesgo y acelerado (Efron, 1987). El remuestreo es por conglomerado (la
  orden) dentro de cada estrato (el bloque), y la aceleración sale de un jackknife por grupos de conglomerados.
"""
import numpy as np
from scipy import stats


# --- DeLong ------------------------------------------------------------------------------------------------------------
def _rangos_medios(x):
    """Rango medio de cada valor (1 = menor), con los empates repartidos: lo que hace que un empate cuente 0,5."""
    return stats.rankdata(x, method="average")


def _componentes(y, s):
    """Componentes estructurales de DeLong para un puntaje: V10 (una por tardía) y V01 (una por puntual)."""
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = _rangos_medios(pos), _rangos_medios(neg), _rangos_medios(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:m] - tx) / n            # para cada tardía, la fracción de puntuales que quedan por debajo
    v01 = 1 - (tz[m:] - ty) / m        # para cada puntual, la fracción de tardías que quedan por encima
    return auc, v10, v01


def delong(y, s1, s2):
    """AUC de los dos puntajes, su diferencia (1 − 2), el error estándar, z y el p bilateral."""
    y = np.asarray(y).astype(int)
    a1, v10_1, v01_1 = _componentes(y, np.asarray(s1, float))
    a2, v10_2, v01_2 = _componentes(y, np.asarray(s2, float))
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    S = s10 / len(v10_1) + s01 / len(v01_1)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    dif = a1 - a2
    ee = float(np.sqrt(var)) if var > 0 else 0.0
    z = dif / ee if ee > 0 else 0.0
    return {"AUC_1": a1, "AUC_2": a2, "diferencia": dif, "ee": ee, "z": z, "p": 2 * stats.norm.sf(abs(z))}


def auc(y, s):
    return _componentes(np.asarray(y).astype(int), np.asarray(s, float))[0]


# --- Cliff ---------------------------------------------------------------------------------------------------------------
def cliff(a, b):
    """Delta de Cliff y su magnitud: despreciable (< 0,147), pequeña (< 0,33), mediana (< 0,474) o grande."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float((np.sign(a[:, None] - b[None, :])).mean())
    m = abs(d)
    etiqueta = "despreciable" if m < 0.147 else "pequeña" if m < 0.33 else "mediana" if m < 0.474 else "grande"
    return d, etiqueta


# --- BCa -----------------------------------------------------------------------------------------------------------------
def bca(estratos, estadistico, B=1000, alfa=0.05, semilla=2026, grupos_jack=200):
    """Intervalo BCa de ``estadistico`` remuestreando conglomerados dentro de cada estrato.

    ``estratos``: lista de (conglomerado, datos) por estrato, con ``conglomerado`` un entero por fila y ``datos`` un
    arreglo o DataFrame con las filas. ``estadistico``: función que recibe la lista de ``datos`` remuestreados (uno por
    estrato) y devuelve un número o un vector (un intervalo por componente, todos con las mismas réplicas). Devuelve
    (estimación, inferior, superior), cada uno del tamaño de lo que devuelve ``estadistico``."""
    rng = np.random.default_rng(semilla)
    idx = []
    for g, _ in estratos:
        g = np.asarray(g)
        orden = np.argsort(g, kind="stable")
        unicos, inicio = np.unique(g[orden], return_index=True)
        idx.append(np.split(orden, inicio[1:]))            # filas de cada conglomerado
    tomar = lambda d, i: d.iloc[i] if hasattr(d, "iloc") else d[i]  # noqa: E731
    theta = np.atleast_1d(estadistico([d for _, d in estratos])).astype(float)
    boot = np.empty((B, len(theta)))
    for b in range(B):
        muestra = []
        for (g, d), filas in zip(estratos, idx):
            elegidos = rng.integers(0, len(filas), len(filas))
            muestra.append(tomar(d, np.concatenate([filas[k] for k in elegidos])))
        boot[b] = np.atleast_1d(estadistico(muestra))
    z0 = stats.norm.ppf(np.clip((boot < theta).mean(0) + 0.5 * (boot == theta).mean(0), 1e-6, 1 - 1e-6))
    # aceleración: jackknife quitando un grupo de conglomerados a la vez (en todos los estratos por igual)
    jack = []
    asignacion = [rng.integers(0, grupos_jack, len(filas)) for filas in idx]
    for k in range(grupos_jack):
        muestra = []
        for (g, d), filas, asig in zip(estratos, idx, asignacion):
            quedan = [f for f, a in zip(filas, asig) if a != k]
            muestra.append(tomar(d, np.concatenate(quedan)))
        jack.append(np.atleast_1d(estadistico(muestra)))
    jack = np.asarray(jack)
    u = jack.mean(0) - jack
    den = 6 * ((u ** 2).sum(0)) ** 1.5
    a = np.where(den > 0, (u ** 3).sum(0) / np.where(den > 0, den, 1), 0.0)
    inf, sup = np.empty(len(theta)), np.empty(len(theta))
    for j in range(len(theta)):
        zq = stats.norm.ppf([alfa / 2, 1 - alfa / 2])
        q = stats.norm.cdf(z0[j] + (z0[j] + zq) / (1 - a[j] * (z0[j] + zq)))
        inf[j], sup[j] = np.quantile(boot[:, j], q)
    if len(theta) == 1:
        return float(theta[0]), float(inf[0]), float(sup[0])
    return theta, inf, sup
