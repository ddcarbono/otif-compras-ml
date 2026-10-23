"""Calibración y errores de XGBoost (la mejor combinación de las 112) en los 8 bloques externos, sin tocar la prueba 2026.

Lee los puntajes de ``resultados/puntajes/`` (``src.puntajes``) y el umbral de máximo F1 que cada trabajo fijó en su
validación interna (el JSON del trabajo: no se elige ningún umbral aquí). Escribe en ``resultados/diagnostico/``:

- ``por_bloque.csv``: prevalencia, AUC, Brier, ECE, F1, precisión, sensibilidad y fracción marcada en cada bloque;
- ``fiabilidad.csv``: la curva de fiabilidad de los 8 bloques juntos (deciles de probabilidad: predicha contra observada);
- ``platt.csv``: Platt ajustado en el bloque anterior y aplicado al siguiente (7 pares), contra el puntaje crudo. Es la
  decisión de calibrar o no, tomada con validación, antes de mirar 2026;
- ``por_linea.csv``: las métricas por línea de negocio, con los 8 bloques juntos;
- ``nuevos.csv``: proveedor nuevo (no aparece en el entrenamiento de su bloque) frente a conocido, por bloque y en total;
- ``perfil_errores.csv``: falsos negativos frente a verdaderos positivos (y falsos frente a verdaderos negativos) en
  puntaje, proveedor nuevo, historial y plazo.

Uso (ml_venv, con OTIF_CSV apuntando al dataset): ``python -m src.diagnostico_validacion``.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from . import datos, validacion
from .experimentos import TRABAJOS, nombre_trabajo
from .modelo_base import ece

ENTRADA = datos.RAIZ / "resultados" / "puntajes"
SALIDA = datos.RAIZ / "resultados" / "diagnostico"
MODELO = "xgboost"
PERFIL = ["prov_tasa_tarde_previa", "prov_n_previas", "plazo_habil_al_liberar", "dias_habiles_liberacion"]


def combinacion():
    t = pd.read_csv(datos.RAIZ / "resultados" / "tabla_maestra.csv")
    f = t[t.modelo == MODELO].sort_values("AUC", ascending=False).iloc[0]
    return f.balanceo, f.optimizador


def metricas(y, s, u):
    m = s >= u
    return {"filas": len(y), "prevalencia": y.mean(), "AUC": roc_auc_score(y, s) if 0 < y.mean() < 1 else np.nan,
            "Brier": brier_score_loss(y, s), "F1": f1_score(y, m, zero_division=0),
            "precision": precision_score(y, m, zero_division=0), "sensibilidad": recall_score(y, m, zero_division=0),
            "marcadas": m.mean()}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    balanceo, opt = combinacion()
    df = datos.cargar()[["oc_id", "particion", "fecha_pedido", datos.OBJETIVO, "linea", "proveedor"] + PERFIL]
    partes = []
    for b in validacion.BLOQUES:
        ent, eva = validacion.ventanas(df, b)["externo"]
        p = pd.read_csv(ENTRADA / f"{b}.csv.gz")
        assert (p.y.to_numpy() == eva[datos.OBJETIVO].to_numpy()).all()
        r = json.loads((TRABAJOS / f"{nombre_trabajo(MODELO, balanceo, opt, b)}.json").read_text(encoding="utf-8"))
        partes.append(eva[["linea", "proveedor"] + PERFIL].assign(
            bloque=b, y=p.y.to_numpy(), s=p[MODELO].to_numpy(), umbral=r["umbral"],
            nuevo=~eva.proveedor.isin(set(ent.proveedor)).to_numpy()))
    d = pd.concat(partes, ignore_index=True)
    d["marcada"] = d.s >= d.umbral

    # calibración por bloque y curva de fiabilidad
    pb = pd.DataFrame([{"bloque": b, "umbral": g.umbral.iloc[0], **metricas(g.y, g.s, g.umbral.iloc[0]),
                        "ECE": ece(g.y, g.s)} for b, g in d.groupby("bloque")])
    pb.to_csv(SALIDA / "por_bloque.csv", index=False)
    d["decil"] = pd.qcut(d.s.rank(method="first"), 10, labels=False) + 1
    d.groupby("decil").agg(filas=("y", "size"), predicha=("s", "mean"), observada=("y", "mean")).reset_index() \
        .to_csv(SALIDA / "fiabilidad.csv", index=False)

    # Platt: ajustado en el bloque anterior, aplicado al siguiente
    filas = []
    for previo, b in zip(validacion.BLOQUES, validacion.BLOQUES[1:]):
        a, g = d[d.bloque == previo], d[d.bloque == b]
        pl = LogisticRegression(C=1e6).fit(logit(a.s.to_numpy()), a.y)
        q = pl.predict_proba(logit(g.s.to_numpy()))[:, 1]
        filas.append({"ajuste": previo, "bloque": b, "ECE_crudo": ece(g.y, g.s), "ECE_platt": ece(g.y, q),
                      "Brier_crudo": brier_score_loss(g.y, g.s), "Brier_platt": brier_score_loss(g.y, q),
                      "pendiente": pl.coef_[0, 0], "intercepto": pl.intercept_[0]})
    pd.DataFrame(filas).to_csv(SALIDA / "platt.csv", index=False)

    # errores por línea y por proveedor nuevo o conocido (cada fila con el umbral de su bloque)
    def resumen(g):
        m = g.marcada
        return pd.Series({"filas": len(g), "prevalencia": g.y.mean(),
                          "AUC": roc_auc_score(g.y, g.s) if 0 < g.y.mean() < 1 else np.nan,
                          "F1": f1_score(g.y, m, zero_division=0), "precision": precision_score(g.y, m, zero_division=0),
                          "sensibilidad": recall_score(g.y, m, zero_division=0), "ECE": ece(g.y, g.s)})
    d.groupby("linea").apply(resumen).reset_index().sort_values("filas", ascending=False) \
        .to_csv(SALIDA / "por_linea.csv", index=False)
    nv = d.assign(proveedor_tipo=np.where(d.nuevo, "nuevo", "conocido"))
    pd.concat([nv.groupby(["bloque", "proveedor_tipo"]).apply(resumen).reset_index(),
               nv.groupby("proveedor_tipo").apply(resumen).reset_index().assign(bloque="todos")]) \
        .to_csv(SALIDA / "nuevos.csv", index=False)

    # perfil de los errores
    d["caso"] = np.select([d.y.eq(1) & d.marcada, d.y.eq(1) & ~d.marcada, d.y.eq(0) & d.marcada],
                          ["verdadero positivo", "falso negativo", "falso positivo"], "verdadero negativo")
    agg = {"filas": ("y", "size"), "puntaje_mediano": ("s", "median"), "proveedor_nuevo": ("nuevo", "mean")}
    agg.update({f"{c}_mediana": (c, "median") for c in PERFIL})
    d.groupby("caso").agg(**agg).reset_index().to_csv(SALIDA / "perfil_errores.csv", index=False)
    print(pb.round(3).to_string(index=False))
    print(pd.read_csv(SALIDA / "platt.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
