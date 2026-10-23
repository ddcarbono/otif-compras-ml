"""Modelo base: regresión logística con regularización L2, frente a dos líneas base triviales.

Protocolo (la prueba 2026 se mira una sola vez, al final):

1. **Elegir C** con validación cruzada temporal dentro del entrenamiento 2021-2024: tres pliegues de ventana creciente
   por año (entrena con 2021 y valida con 2022; con 2021-2022 y valida con 2023; con 2021-2023 y valida con 2024). Gana
   el C con mayor AUC medio. Cada pliegue reajusta todo el *pipeline* (preprocesamiento incluido).
2. **Ajustar con 2021-2024** y, en la validación 2025, elegir el umbral que maximiza el F1 y medir la calibración.
   La curva de aprendizaje sale aquí: AUC de entrenamiento y de validación con 10 %, 25 %, 50 % y 100 % de las órdenes.
3. **Reentrenar con 2021-2025** y medir la prueba 2026 una vez, con el umbral de la validación e intervalos al 95 % por
   *bootstrap* de órdenes (las posiciones de una orden no son independientes).

Líneas base: ``DummyClassifier`` (predice la tasa del entrenamiento; con el umbral de validación marca todas las
posiciones) y la regla del proveedor (su tasa previa de retraso, conocida al crear la orden).

Escribe ``resultados/modelo_base/*.csv`` y ``resumen.json``, y el modelo en ``artifacts/logistica.joblib`` con su
``metadata.json``. Uso: ``python -m src.modelo_base [--bootstrap 200]`` desde la raíz, con ``OTIF_CSV`` definido.
Tarda unos 20 minutos.
"""
import argparse
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

from . import datos

REJILLA_C = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
FRACCIONES = [0.1, 0.25, 0.5, 1.0]
SALIDA = datos.RAIZ / "resultados" / "modelo_base"


def logistica(c):
    return Pipeline([("pre", datos.preprocesador()),
                     ("modelo", LogisticRegression(C=c, max_iter=3000, random_state=datos.SEMILLA))])


def ece(y, p, tramos=10):
    """Error de calibración esperado: diferencia media, ponderada por filas, entre la probabilidad predicha y la tasa
    observada en cada decil de probabilidad."""
    t = pd.DataFrame({"y": y, "p": p})
    t["tramo"] = pd.qcut(t.p.rank(method="first"), tramos, labels=False)
    g = t.groupby("tramo").agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    return float((g.n * (g.obs - g.pred).abs()).sum() / g.n.sum())


def umbral_f1(y, p):
    pr, rc, u = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(u[np.argmax(f1)])


def metricas(y, p, umbral):
    m = p >= umbral
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p), "F1": f1_score(y, m),
            "precisión": precision_score(y, m, zero_division=0), "sensibilidad": recall_score(y, m),
            "exactitud": accuracy_score(y, m), "Brier": brier_score_loss(y, p), "ECE": ece(y, p),
            "marcadas": float(m.mean())}


def bootstrap(y, p, grupos, umbral, b, rng):
    """Intervalo al 95 % de cada métrica, remuestreando órdenes completas."""
    y, p = np.asarray(y), np.asarray(p)
    _, inv = np.unique(np.asarray(grupos), return_inverse=True)
    filas = pd.Series(np.arange(len(y))).groupby(inv).apply(np.array).to_list()
    vals = []
    for _ in range(b):
        idx = np.concatenate([filas[k] for k in rng.integers(0, len(filas), len(filas))])
        vals.append(metricas(y[idx], p[idx], umbral))
    v = pd.DataFrame(vals)
    return v.quantile(0.025), v.quantile(0.975)


def regla_proveedor(df, ref):
    """Tasa previa de retraso del proveedor; sin historial, la tasa del conjunto de referencia."""
    return df.prov_tasa_tarde_previa.fillna(ref.y_tarde.mean()).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=200)
    a = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(datos.SEMILLA)
    t0 = time.time()
    df = datos.cargar()
    ent, val, pru = (datos.particion(df, p) for p in ["entrenamiento", "validacion", "prueba"])
    anio = ent.fecha_pedido.dt.year

    # 1. C por validación cruzada temporal dentro de 2021-2024
    filas = []
    for c in REJILLA_C:
        for hasta in [2021, 2022, 2023]:
            tr, te = ent[anio <= hasta], ent[anio == hasta + 1]
            m = logistica(c).fit(tr[datos.ENTRADA], tr.y_tarde)
            filas.append({"C": c, "valida": hasta + 1, "filas_entrena": len(tr),
                          "AUC": roc_auc_score(te.y_tarde, m.predict_proba(te[datos.ENTRADA])[:, 1])})
            print(f"C={c} valida {hasta + 1}: AUC {filas[-1]['AUC']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    cv = pd.DataFrame(filas)
    cv.to_csv(SALIDA / "cv_C.csv", index=False)
    resumen_cv = cv.groupby("C").AUC.agg(["mean", "std"])
    c_elegido = float(resumen_cv["mean"].idxmax())

    # 2. Ajuste con 2021-2024: umbral, calibración y curva de aprendizaje en la validación 2025
    oc = ent.oc_id.unique()
    rng_oc = np.random.default_rng(datos.SEMILLA)
    orden_oc = rng_oc.permutation(oc)
    curva = []
    for f in FRACCIONES:
        sub = ent[ent.oc_id.isin(set(orden_oc[: int(f * len(oc))]))]
        pre = datos.preprocesador()
        Xs = pre.fit_transform(sub[datos.ENTRADA], sub.y_tarde)         # codificación fuera de pliegue: sin verse
        lr = LogisticRegression(C=c_elegido, max_iter=3000, random_state=datos.SEMILLA).fit(Xs, sub.y_tarde)
        p_val_f = lr.predict_proba(pre.transform(val[datos.ENTRADA]))[:, 1]
        curva.append({"fracción": f, "filas": len(sub), "AUC entrenamiento": roc_auc_score(sub.y_tarde, lr.predict_proba(Xs)[:, 1]),
                      "AUC validación": roc_auc_score(val.y_tarde, p_val_f)})
        print(f"curva {f}: {curva[-1]}", flush=True)
        if f == 1.0:
            p_val = p_val_f
    pd.DataFrame(curva).to_csv(SALIDA / "curva_aprendizaje.csv", index=False)

    dummy = DummyClassifier(strategy="prior").fit(ent[datos.ENTRADA], ent.y_tarde)
    preds_val = {"Dummy (tasa del entrenamiento)": dummy.predict_proba(val[datos.ENTRADA])[:, 1],
                 "Regla del proveedor": regla_proveedor(val, ent), "Regresión logística": p_val}
    umbral = {k: umbral_f1(val.y_tarde, p) for k, p in preds_val.items()}
    # el Dummy da la misma probabilidad a todas: su único umbral útil es marcarlas todas ("todo llega tarde"); con el de
    # la validación, una prior reentrenada apenas por debajo no marcaría ninguna
    umbral["Dummy (tasa del entrenamiento)"] = 0.0

    # 3. Reentrenar con 2021-2025 y medir la prueba 2026 una sola vez
    ev = pd.concat([ent, val])
    final = logistica(c_elegido).fit(ev[datos.ENTRADA], ev.y_tarde)
    dummy = DummyClassifier(strategy="prior").fit(ev[datos.ENTRADA], ev.y_tarde)
    preds_pru = {"Dummy (tasa del entrenamiento)": dummy.predict_proba(pru[datos.ENTRADA])[:, 1],
                 "Regla del proveedor": regla_proveedor(pru, ev),
                 "Regresión logística": final.predict_proba(pru[datos.ENTRADA])[:, 1]}

    tabla = []
    for nombre in preds_val:
        for part, y, p, g in [("validación 2025", val.y_tarde, preds_val[nombre], val.oc_id),
                              ("prueba 2026", pru.y_tarde, preds_pru[nombre], pru.oc_id)]:
            fila = {"modelo": nombre, "partición": part, "umbral": umbral[nombre], **metricas(y.to_numpy(), p, umbral[nombre])}
            if part == "prueba 2026":
                lo, hi = bootstrap(y, p, g, umbral[nombre], a.bootstrap, rng)
                for k in ["ROC-AUC", "PR-AUC", "F1"]:
                    fila[f"{k} inf"], fila[f"{k} sup"] = lo[k], hi[k]
            tabla.append(fila)
    pd.DataFrame(tabla).to_csv(SALIDA / "metricas.csv", index=False)

    # diagnósticos de la logística en la prueba
    y, p, u = pru.y_tarde.to_numpy(), preds_pru["Regresión logística"], umbral["Regresión logística"]
    m = p >= u
    pd.DataFrame({"real": ["tarde", "tarde", "a tiempo", "a tiempo"], "predicho": ["tarde", "a tiempo", "tarde", "a tiempo"],
                  "posiciones": [int((y & m).sum()), int((y & ~m).sum()), int((~y.astype(bool) & m).sum()),
                                 int((~y.astype(bool) & ~m).sum())]}).to_csv(SALIDA / "confusion.csv", index=False)
    curvas = []
    for nombre, pp in preds_pru.items():
        fpr, tpr, _ = roc_curve(y, pp)
        k = np.unique(np.linspace(0, len(fpr) - 1, 300).astype(int))
        curvas.append(pd.DataFrame({"modelo": nombre, "curva": "ROC", "x": fpr[k], "y": tpr[k]}))
        pr, rc, _ = precision_recall_curve(y, pp)
        k = np.unique(np.linspace(0, len(pr) - 1, 300).astype(int))
        curvas.append(pd.DataFrame({"modelo": nombre, "curva": "PR", "x": rc[k], "y": pr[k]}))
    pd.concat(curvas).to_csv(SALIDA / "curvas.csv", index=False)
    cal = pd.DataFrame({"y": y, "p": p})
    cal["decil"] = pd.qcut(cal.p.rank(method="first"), 10, labels=False) + 1
    cal.groupby("decil").agg(predicha=("p", "mean"), observada=("y", "mean"), posiciones=("y", "size")).reset_index() \
        .to_csv(SALIDA / "calibracion.csv", index=False)

    nombres = final.named_steps["pre"].get_feature_names_out()
    coef = pd.DataFrame({"columna": nombres, "coeficiente": final.named_steps["modelo"].coef_[0]})
    coef["bloque"] = coef.columna.str.split("__").str[0]
    coef["variable"] = coef.columna.str.split("__").str[1].str.replace(r"^te_", "", regex=True)
    coef["razón de odds por 1 desv."] = np.exp(coef.coeficiente)
    coef.reindex(coef.coeficiente.abs().sort_values(ascending=False).index).to_csv(SALIDA / "coeficientes.csv", index=False)

    resumen = {"C": c_elegido, "rejilla_C": REJILLA_C, "umbral_f1_validacion": umbral, "semilla": datos.SEMILLA,
               "bootstrap": a.bootstrap, "minutos": round((time.time() - t0) / 60, 1),
               "filas": {"entrenamiento": len(ent), "validacion": len(val), "prueba": len(pru)}}
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")

    art = datos.RAIZ / "artifacts"
    art.mkdir(exist_ok=True)
    joblib.dump(final, art / "logistica.joblib", compress=3)
    meta = {"modelo": "regresión logística L2 (modelo base)", "entrenado_con": "órdenes de 2021-2025",
            "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                          "pandas": pd.__version__, "numpy": np.__version__},
            "semilla": datos.SEMILLA, "hiperparametros": {"C": c_elegido, "penalty": "l2", "solver": "lbfgs"},
            "umbral": umbral["Regresión logística"], "entrada": datos.ENTRADA,
            "metricas_prueba": {k: v for k, v in tabla[-1].items() if isinstance(v, float)}}
    (art / "logistica.metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    # ida y vuelta: el modelo leído del disco predice exactamente lo mismo
    assert np.array_equal(joblib.load(art / "logistica.joblib").predict_proba(pru[datos.ENTRADA].head(2000)),
                          final.predict_proba(pru[datos.ENTRADA].head(2000)))
    print(f"listo en {resumen['minutos']} min; C = {c_elegido}")


if __name__ == "__main__":
    main()
