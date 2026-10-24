"""El modelo elegido en las 112 (XGBoost) en la prueba 2026, frente a la logística base y a la propuesta del Entregable 3.

Lee las probabilidades por posición de la comparación 2x2 (``maestria/machine_learning/otif/evaluar_2x2_otif.py``, ya con
Platt): «XGBoost 136» y «BOREAL 136 + nuevas», con los umbrales de máximo F1 congelados en la validación 2025
(``resultados/prueba_2026/config_*.json``). La logística base se predice aquí con ``artifacts/logistica.joblib``
(entrenada con 2021-2025, umbral de su metadata). Nada se elige mirando 2026; es la tercera mirada a la prueba y se lee
como exploratoria.

Escribe en ``resultados/prueba_2026/`` (mismos formatos que ``resultados/modelo_base/``, para las mismas figuras):

- ``metricas.csv``: ROC-AUC, PR-AUC y F1 con su intervalo BCa 95 % (1.000 réplicas que remuestrean órdenes), precisión,
  sensibilidad, Brier y ECE de los tres modelos, más la regla del proveedor y el Dummy de ``modelo_base``;
- ``diferencias.csv``: XGBoost − logística y propuesta − XGBoost, con DeLong y el intervalo BCa de la diferencia;
- ``curvas.csv``, ``calibracion.csv`` y ``confusion.csv`` (esta, de XGBoost a su umbral);
- ``segmentos.csv``: XGBoost y la propuesta por línea de negocio y por orden con alguna entidad nueva o todo conocido;
- ``dos_por_dos.csv`` y ``contrastes.csv``: el ROC-AUC de las seis celdas de la 2x2 (modelo × información, con la
  ablación del log-normal) y sus diferencias en pareja, con BCa por órdenes;
- ``brecha.csv``: el AUC de XGBoost, la logística y la propuesta en cada etapa (validación interna, bloques externos,
  validación 2025 y prueba 2026).

Uso (ml_venv, con OTIF_CSV): ``python -m src.prueba_2026 <predicciones_prueba_2x2.csv>``.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_curve)

from . import datos, pruebas
from .modelo_base import ece

P26 = datos.RAIZ / "resultados" / "prueba_2026"
MB = datos.RAIZ / "resultados" / "modelo_base"
XGB, PROP, LOG = "XGBoost (las 112)", "BOREAL (propuesta)", "Regresión logística"
CELDAS = ["XGBoost 136", "XGBoost 136 + nuevas", "Boosting mixto 136 (ablación)", "Boosting mixto 136 + nuevas (ablación)", "BOREAL 136",
          "BOREAL 136 + nuevas"]
CONTRASTES = [("propuesta completa contra el ganador de las 112", "BOREAL 136 + nuevas", "XGBoost 136"),
              ("modelo, con las 136", "BOREAL 136", "XGBoost 136"),
              ("modelo, con 136 + nuevas", "BOREAL 136 + nuevas", "XGBoost 136 + nuevas"),
              ("información, en XGBoost", "XGBoost 136 + nuevas", "XGBoost 136"),
              ("información, en la propuesta", "BOREAL 136 + nuevas", "BOREAL 136"),
              ("log-normal, con las 136", "BOREAL 136", "Boosting mixto 136 (ablación)"),
              ("log-normal, con 136 + nuevas", "BOREAL 136 + nuevas", "Boosting mixto 136 + nuevas (ablación)")]


def calibracion(y, p, modelo):
    t = pd.DataFrame({"y": y, "p": p})
    t["decil"] = pd.qcut(t.p.rank(method="first"), 10, labels=False) + 1
    g = t.groupby("decil").agg(predicha=("p", "mean"), observada=("y", "mean"), posiciones=("y", "size")).reset_index()
    return g.assign(modelo=modelo)


def curvas(y, p, modelo, puntos=200):
    f, v, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    i, j = np.linspace(0, len(f) - 1, puntos).astype(int), np.linspace(0, len(pr) - 1, puntos).astype(int)
    return pd.concat([pd.DataFrame({"modelo": modelo, "curva": "ROC", "x": f[i], "y": v[i]}),
                      pd.DataFrame({"modelo": modelo, "curva": "PR", "x": rc[j], "y": pr[j]})])


def main():
    q = pd.read_csv(sys.argv[1], sep=";", decimal=",", dtype={"oc_id": str})
    cfg = {k: json.loads((P26 / f"config_{k}.json").read_text(encoding="utf-8")) for k in ("xgb136", "prop_nuevas")}
    df = datos.cargar()
    pru = datos.particion(df, "prueba")
    d = pru[["oc_id", "posicion", "linea", datos.OBJETIVO]].merge(q.rename(columns={"y_tarde": "y_2x2"}),
                                                                  on=["oc_id", "posicion"], how="left", validate="1:1")
    assert (d[datos.OBJETIVO] == d.y_2x2).all()
    meta = json.loads((datos.RAIZ / "artifacts" / "logistica.metadata.json").read_text(encoding="utf-8"))
    d[LOG] = joblib.load(datos.RAIZ / "artifacts" / "logistica.joblib").predict_proba(pru[meta["entrada"]])[:, 1]
    d[XGB], d[PROP] = d["XGBoost 136"], d["BOREAL 136 + nuevas"]
    modelos = {XGB: cfg["xgb136"]["umbral"], PROP: cfg["prop_nuevas"]["umbral"], LOG: meta["umbral"]}
    y = d.y_tarde.to_numpy().astype(int)
    grupo = pd.factorize(d.oc_id)[0]

    # un solo bootstrap por órdenes para las métricas y las diferencias
    nombres = list(modelos)

    def estadistico(m):
        m = m[0]
        yy = m.y_tarde.to_numpy().astype(int)
        auc = [pruebas.auc(yy, m[k].to_numpy()) for k in nombres]
        pr = [average_precision_score(yy, m[k]) for k in nombres]
        f1 = [f1_score(yy, m[k] >= u) for k, u in modelos.items()]
        c = {k: pruebas.auc(yy, m[k].to_numpy()) for k in CELDAS}
        return np.r_[auc, pr, f1, auc[0] - auc[2], auc[1] - auc[0], [c[k] for k in CELDAS],
                     [c[a] - c[b] for _, a, b in CONTRASTES]]
    est, inf, sup = pruebas.bca([(grupo, d)], estadistico, B=1000)
    k = len(nombres)
    filas = []
    for i, (n, u) in enumerate(modelos.items()):
        p = d[n].to_numpy()
        filas.append({"modelo": n, "partición": "prueba 2026", "umbral": u, "ROC-AUC": est[i], "PR-AUC": est[k + i],
                      "F1": est[2 * k + i], "precisión": precision_score(y, p >= u),
                      "sensibilidad": recall_score(y, p >= u), "Brier": brier_score_loss(y, p), "ECE": ece(y, p),
                      "marcadas": (p >= u).mean(), "ROC-AUC inf": inf[i], "ROC-AUC sup": sup[i],
                      "PR-AUC inf": inf[k + i], "PR-AUC sup": sup[k + i], "F1 inf": inf[2 * k + i],
                      "F1 sup": sup[2 * k + i]})
    mb = pd.read_csv(MB / "metricas.csv")
    mb = mb[(mb["partición"] == "prueba 2026") & mb.modelo.isin(["Regla del proveedor", "Dummy (tasa del entrenamiento)"])]
    pd.concat([pd.DataFrame(filas), mb[[c for c in filas[0] if c in mb.columns]]]).to_csv(P26 / "metricas.csv",
                                                                                         index=False)
    dif = []
    for j, (a, b) in enumerate([(XGB, LOG), (PROP, XGB)]):
        dl = pruebas.delong(y, d[a], d[b])
        dif.append({"comparación": f"{a} − {b}", "diferencia": est[3 * k + j], "BCa inf": inf[3 * k + j],
                    "BCa sup": sup[3 * k + j], "z DeLong": dl["z"], "p DeLong": dl["p"]})
    pd.DataFrame(dif).to_csv(P26 / "diferencias.csv", index=False)

    # la 2x2 (modelo x información) con la ablación del log-normal
    o = 3 * k + 2
    pd.DataFrame({"celda": CELDAS, "ROC-AUC": est[o:o + len(CELDAS)], "inf": inf[o:o + len(CELDAS)],
                  "sup": sup[o:o + len(CELDAS)]}).to_csv(P26 / "dos_por_dos.csv", index=False)
    o += len(CELDAS)
    pd.DataFrame([{"contraste": n, "diferencia": a + " − " + b, "ΔROC-AUC": est[o + j], "inf": inf[o + j],
                   "sup": sup[o + j]} for j, (n, a, b) in enumerate(CONTRASTES)]).to_csv(P26 / "contrastes.csv",
                                                                                    index=False)

    pd.concat([curvas(y, d[n].to_numpy(), n) for n in nombres]).to_csv(P26 / "curvas.csv", index=False)
    pd.concat([calibracion(y, d[n].to_numpy(), n) for n in nombres]).to_csv(P26 / "calibracion.csv", index=False)
    m = d[XGB] >= modelos[XGB]
    pd.DataFrame([{"real": r, "predicho": pr, "posiciones": int(((y == yr) & (m == mr)).sum())}
                  for r, yr in [("tarde", 1), ("a tiempo", 0)] for pr, mr in [("tarde", True), ("a tiempo", False)]]
                 ).to_csv(P26 / "confusion.csv", index=False)

    seg = []
    for col, nom in [("linea", "línea"), ("nuevo", "entidad nueva")]:
        for v, g in d.groupby(col):
            for n in (XGB, PROP):
                yy, p = g.y_tarde.to_numpy().astype(int), g[n].to_numpy()
                seg.append({"segmento": nom, "valor": {0: "todo conocido", 1: "alguna nueva"}.get(v, v), "modelo": n,
                            "posiciones": len(g), "prevalencia": yy.mean(), "ROC-AUC": pruebas.auc(yy, p),
                            "F1": f1_score(yy, p >= modelos[n]), "ECE": ece(yy, p)})
    pd.DataFrame(seg).to_csv(P26 / "segmentos.csv", index=False)

    # brecha: el AUC del mismo modelo en cada etapa
    t = pd.read_csv(datos.RAIZ / "resultados" / "tabla_maestra.csv")
    xg = t[(t.modelo == "xgboost") & (t.balanceo == "ninguno") & (t.optimizador == "optuna")].iloc[0]
    todo = pd.read_csv(MB / "metricas.csv")
    lv = todo[(todo.modelo == LOG) & (todo["partición"] == "validación 2025")].iloc[0]
    pd.DataFrame([
        {"modelo": XGB, "validación interna (8 bloques)": xg.AUC_interno, "bloques externos (8)": xg.AUC,
         "validación 2025": cfg["xgb136"]["auc_validacion"], "prueba 2026": est[0]},
        {"modelo": PROP, "validación interna (8 bloques)": np.nan, "bloques externos (8)": np.nan,
         "validación 2025": cfg["prop_nuevas"]["auc_validacion"], "prueba 2026": est[1]},
        {"modelo": LOG, "validación interna (8 bloques)": np.nan, "bloques externos (8)": np.nan,
         "validación 2025": lv["ROC-AUC"], "prueba 2026": est[2]}]).to_csv(P26 / "brecha.csv", index=False)
    print(pd.read_csv(P26 / "metricas.csv").round(4).to_string(index=False))
    print(pd.read_csv(P26 / "diferencias.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
