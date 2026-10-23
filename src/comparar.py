"""Resume los 896 trabajos de ``experimentos`` (``resultados/experimentos/trabajos/*.json``) en las tablas del capítulo
de los 112 y del tablero, y hace la comparación de Friedman y Nemenyi entre modelos.

Uso: ``python -m src.comparar`` desde la raíz; escribe ``resultados/experimentos/*.csv``. No usa el dataset.

- ``lideres.csv``: una fila por combinación (modelo, balanceo, optimizador) con la media y la desviación de AUC, F1 y
  PR-AUC en los 8 bloques externos, el AUC interno, las horas de CPU y la inferencia.
- ``curvas_optimizador.csv``: para cada optimizador y número de evaluación, la distancia media entre el mejor AUC interno
  encontrado hasta ahí y el mejor que encontró cualquier optimizador con el mismo modelo, balanceo y bloque (el
  arrepentimiento). En segundos, lo mismo contra el tiempo de reloj, solo para XGBoost (los tiempos de los demás
  modelos no son comparables entre sí).
- ``friedman.csv`` y ``rangos.csv``: Friedman sobre los 8 bloques con la mejor combinación de cada modelo, elegida por su
  **AUC interno medio** (elegirla por el AUC externo sería mirar el resultado que se quiere comparar), y la diferencia
  crítica de Nemenyi; ``wilcoxon.csv``: el mejor contra cada uno, con Holm.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from . import datos
from .plots import BALANCEO, MODELO, OPTIMIZADOR  # noqa: F401  (los nombres legibles viven con las figuras)

R = datos.RAIZ / "resultados" / "experimentos"
# q de Nemenyi (rango studentizado / √2) con α = 0,05, para k = 2..10 (Demšar, 2006, tabla 5)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def cargar():
    filas = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((R / "trabajos").glob("*.json"))]
    return filas


def lideres(filas):
    df = pd.DataFrame([{k: v for k, v in f.items() if k not in ("historial", "params")} for f in filas])
    g = df.groupby(["modelo", "balanceo", "optimizador"])
    t = g.agg(AUC=("AUC", "mean"), AUC_desv=("AUC", "std"), F1=("F1", "mean"), F1_desv=("F1", "std"),
              PR_AUC=("PR_AUC", "mean"), AUC_interno=("AUC_interno", "mean"),
              horas_cpu=("segundos_total", lambda s: s.sum() / 3600),
              ms_inferencia_por_1000=("ms_inferencia_por_1000", "mean")).reset_index()
    t = t.sort_values("AUC", ascending=False).reset_index(drop=True)
    t.insert(0, "puesto", np.arange(1, len(t) + 1))
    return df, t


def curvas(filas):
    h = pd.DataFrame([{"modelo": f["modelo"], "balanceo": f["balanceo"], "optimizador": f["optimizador"],
                       "bloque": f["bloque"], "eval": r["eval"], "valor": r["valor"], "segundos": r["segundos"]}
                      for f in filas for r in f["historial"]])
    claves = ["modelo", "balanceo", "optimizador", "bloque"]
    h = h.sort_values(claves + ["eval"])
    h["mejor"] = h.groupby(claves).valor.cummax()
    techo = h.groupby(["modelo", "balanceo", "bloque"]).valor.max().rename("techo")
    h = h.join(techo, on=["modelo", "balanceo", "bloque"])
    h["arrepentimiento"] = h.techo - h.mejor
    # una búsqueda que terminó antes (la rejilla con 16 o 18 puntos) conserva su último valor hasta la evaluación 20
    completo = (h.set_index(claves + ["eval"]).arrepentimiento.unstack("eval").ffill(axis=1).stack()
                .rename("arrepentimiento").reset_index())
    por_eval = completo.groupby(["optimizador", "eval"]).arrepentimiento.agg(["mean", "median"]).reset_index()
    x = h[h.modelo == "xgboost"]
    rejilla = np.geomspace(60, x.segundos.max(), 40)
    tiempo = []
    for (o, *_), d in x.groupby(claves[2:3] + ["balanceo", "bloque"]):
        a = np.interp(rejilla, d.segundos, d.arrepentimiento, left=np.nan, right=d.arrepentimiento.iloc[-1])
        tiempo.append(pd.DataFrame({"optimizador": o, "segundos": rejilla, "arrepentimiento": a}))
    por_tiempo = pd.concat(tiempo).groupby(["optimizador", "segundos"]).arrepentimiento.mean().reset_index()
    return por_eval, por_tiempo


def friedman(df, t):
    """Friedman y Nemenyi entre los 7 modelos, cada uno con su combinación de mayor AUC interno medio."""
    elegida = t.sort_values("AUC_interno", ascending=False).groupby("modelo").head(1)
    m = df.merge(elegida[["modelo", "balanceo", "optimizador"]]).pivot(index="bloque", columns="modelo", values="AUC")
    rangos = m.rank(axis=1, ascending=False)
    est, p = stats.friedmanchisquare(*[m[c] for c in m.columns])
    n, k = m.shape
    iman = (n - 1) * est / (n * (k - 1) - est)          # corrección de Iman y Davenport, F con (k-1, (k-1)(n-1)) gl
    p_iman = stats.f.sf(iman, k - 1, (k - 1) * (n - 1))
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
    r = rangos.mean().rename("rango_medio").reset_index().merge(elegida[["modelo", "balanceo", "optimizador", "AUC"]])
    r = r.sort_values("rango_medio")
    res = pd.DataFrame([{"k": k, "bloques": n, "chi2": est, "p": p, "F_iman": iman, "p_iman": p_iman, "CD": cd}])
    return res, r, m


def contra_mejor(m, r):
    """Wilcoxon pareado (exacto, dos colas) del modelo de mejor rango contra cada uno de los demás en los 8 bloques, con
    la corrección de Holm por las 6 comparaciones. Más potente que Nemenyi, que compara todos contra todos."""
    mejor = r.modelo.iloc[0]
    filas = []
    for c in r.modelo.iloc[1:]:
        d = m[mejor] - m[c]
        filas.append({"mejor": mejor, "contra": c, "diferencia_media": d.mean(), "bloques_gana": int((d > 0).sum()),
                      "p": stats.wilcoxon(m[mejor], m[c], alternative="two-sided").pvalue})
    w = pd.DataFrame(filas).sort_values("p").reset_index(drop=True)
    k = len(w)
    w["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * p for i, p in enumerate(w.p)]))
    return w


def main():
    filas = cargar()
    df, t = lideres(filas)
    df.to_csv(R / "por_bloque.csv", index=False)
    t.to_csv(R / "lideres.csv", index=False)
    e, s = curvas(filas)
    e.to_csv(R / "curvas_optimizador.csv", index=False)
    s.to_csv(R / "curvas_optimizador_tiempo.csv", index=False)
    res, r, m = friedman(df, t)
    res.to_csv(R / "friedman.csv", index=False)
    r.to_csv(R / "rangos.csv", index=False)
    m.reset_index().to_csv(R / "auc_por_bloque_modelos.csv", index=False)
    w = contra_mejor(m, r)
    w.to_csv(R / "wilcoxon.csv", index=False)
    print(f"{len(filas)} trabajos; {len(t)} combinaciones")
    print(res.round(4).to_string(index=False))
    print(r.round(3).to_string(index=False))
    print(w.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
