"""Validación estadística de BOREAL contra el XGBoost de las 112 en la prueba 2026 (R3 del Entregable 3).

La prueba 2026 tiene orden en el tiempo, así que además de DeLong (el AUC de todo el año) se usan las pruebas que el
enunciado indica para datos temporales:

- **DeLong** pareado sobre el ROC-AUC de las 51.941 posiciones (``src/pruebas.py``).
- **Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold (HLN)** sobre la serie semanal de la diferencia de
  pérdida logarítmica media, d_t = pérdida(XGBoost) − pérdida(BOREAL): d > 0 es que BOREAL se equivoca menos. Con
  horizonte h = 1 la varianza larga es la de d con rezagos de Newey-West (Bartlett, L = ⌊T^(1/3)⌋), y el estadístico
  se compara con una t de T − 1 grados.
- **Wilcoxon de rangos con signo** sobre las diferencias de AUC mes a mes, y **delta de Cliff** entre los AUC
  mensuales de los dos modelos como tamaño del efecto (no despreciable desde |δ| ≥ 0,147).

Como sensibilidad, fuera de Holm, Diebold-Mariano se repite solo con las semanas de 200 posiciones o más: en la media
semanal una semana de 10 posiciones pesa lo mismo que una de 2.000, y las últimas semanas de la prueba tienen pocas
órdenes ya resueltas.

Los p de las pruebas de los dos contrastes (la propuesta completa y el modelo con las mismas 136 variables) se ajustan
juntos por Holm. La fecha es la de liberación de la orden (``fecha_prediccion``).

Escribe ``resultados/prueba_2026/validacion_original.csv`` y ``validacion_original_mensual.csv``.
Uso (ml_venv, con OTIF_CSV): ``python -m src.validacion_original <predicciones_prueba_2x2.csv>``.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from . import datos, pruebas

P26 = datos.RAIZ / "resultados" / "prueba_2026"
CONTRASTES = [("propuesta completa contra el ganador de las 112", "BOREAL 136 + nuevas", "XGBoost 136"),
              ("modelo, con las 136", "BOREAL 136", "XGBoost 136")]


def perdida_log(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def diebold_mariano_hln(d, h=1):
    """DM con varianza de Newey-West y la corrección HLN; devuelve el estadístico y el p bilateral (t de T − 1)."""
    d = np.asarray(d, float)
    T = len(d)
    L = max(h - 1, int(np.floor(T ** (1 / 3))))
    e = d - d.mean()
    lrv = e @ e / T + 2 * sum((1 - k / (L + 1)) * (e[k:] @ e[:-k]) / T for k in range(1, L + 1))
    dm = d.mean() / np.sqrt(lrv / T)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(hln), float(2 * stats.t.sf(abs(hln), T - 1)), T


def holm(p):
    p = np.asarray(p, float)
    orden, m, acum, out = np.argsort(p), len(p), 0.0, np.empty(len(p))
    for k, i in enumerate(orden):
        acum = max(acum, (m - k) * p[i])
        out[i] = min(1.0, acum)
    return out


def main():
    q = pd.read_csv(sys.argv[1], sep=";", decimal=",", dtype={"oc_id": str})
    pru = datos.particion(datos.cargar(), "prueba")
    d = pru[["oc_id", "posicion", "fecha_prediccion", datos.OBJETIVO]].merge(
        q.drop(columns="y_tarde"), on=["oc_id", "posicion"], how="left", validate="1:1")
    y = d[datos.OBJETIVO].astype(int).to_numpy()
    semana = d.fecha_prediccion.dt.to_period("W")
    mes = d.fecha_prediccion.dt.to_period("M")
    filas, mensual = [], []
    for nombre, a, b in CONTRASTES:
        dl = pruebas.delong(y, d[a].to_numpy(), d[b].to_numpy())
        filas.append({"contraste": nombre, "prueba": "DeLong (AUC de 2026)", "unidades": len(d),
                      "efecto": float(pruebas.auc(y, d[a]) - pruebas.auc(y, d[b])), "medida del efecto": "ΔAUC",
                      "estadístico": dl["z"], "p": dl["p"], "en Holm": True})
        g = pd.Series(perdida_log(y, d[b].to_numpy()) - perdida_log(y, d[a].to_numpy())).groupby(semana.to_numpy())
        dif = g.mean()
        est, p, T = diebold_mariano_hln(dif.to_numpy())
        filas.append({"contraste": nombre, "prueba": "Diebold-Mariano con HLN (pérdida log, semanal)", "unidades": T,
                      "efecto": float(dif.mean()), "medida del efecto": "Δ pérdida log media por semana",
                      "estadístico": est, "p": p, "en Holm": True})
        # sensibilidad, fuera de Holm: las semanas con menos de 200 posiciones (las colas de la prueba, con pocas órdenes
        # ya resueltas) pesan lo mismo que las completas en la media semanal; se repite solo con las semanas completas
        llenas = dif[g.size() >= 200]
        est, p, T = diebold_mariano_hln(llenas.to_numpy())
        filas.append({"contraste": nombre, "prueba": "Diebold-Mariano con HLN, semanas con 200 posiciones o más "
                      "(sensibilidad)", "unidades": T, "efecto": float(llenas.mean()),
                      "medida del efecto": "Δ pérdida log media por semana", "estadístico": est, "p": p, "en Holm": False})
        m = pd.DataFrame([{"mes": str(k), "posiciones": len(g), "AUC " + a: pruebas.auc(g[datos.OBJETIVO], g[a]),
                           "AUC " + b: pruebas.auc(g[datos.OBJETIVO], g[b])}
                          for k, g in d.groupby(mes) if g[datos.OBJETIVO].nunique() == 2 and len(g) >= 500])
        w = stats.wilcoxon(m["AUC " + a] - m["AUC " + b])
        delta, magnitud = pruebas.cliff(m["AUC " + a], m["AUC " + b])
        filas.append({"contraste": nombre, "prueba": "Wilcoxon (ΔAUC mensual) y delta de Cliff", "unidades": len(m),
                      "efecto": delta, "medida del efecto": f"δ de Cliff ({magnitud})",
                      "estadístico": float(w.statistic), "p": float(w.pvalue), "en Holm": True})
        mensual.append(m.assign(contraste=nombre))
    r = pd.DataFrame(filas)
    r.loc[r["en Holm"], "p (Holm)"] = holm(r.loc[r["en Holm"], "p"].to_numpy())
    r.to_csv(P26 / "validacion_original.csv", index=False)
    pd.concat(mensual).to_csv(P26 / "validacion_original_mensual.csv", index=False)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
