"""Por qué XGBoost y no la logística: cifras de operación, de forma del efecto y de costo, en la prueba 2026.

Usa las probabilidades de XGBoost de la 2x2 (columna «XGBoost 136», con Platt congelado en 2025) y las de la logística
base (``artifacts/logistica.joblib``, entrenada con 2021-2025). Escribe en ``resultados/prueba_2026/``:

- ``presupuesto.csv``: con el mismo presupuesto de alertas (el 10, 20 y 30 % de las posiciones de mayor riesgo de cada
  modelo), cuántas tardías atrapa cada uno, en total, por mes y en órdenes con alguna tardía atrapada;
- ``forma_plazo.csv``: el efecto de los días hábiles de plazo al liberar sobre el logit, por tramo de plazo y por tercio
  de la tasa previa de retraso del proveedor. En XGBoost es la media de su valor SHAP; en la logística, el coeficiente
  por la variable transformada (centrado), que no puede depender del proveedor;
- ``forma_clase.csv``: lo mismo con la clase de compras en lugar del proveedor;
- ``costo.csv``: horas de CPU de las 8 ventanas, segundos de un ajuste (el de 2025-S2, 311.865 filas, un hilo) y
  milisegundos de inferencia por 1.000 posiciones de los dos, de la tabla
  maestra de las 112.

Uso (ml_venv, con OTIF_CSV): ``python -m src.xgb_vs_logistica <predicciones_prueba_2x2.csv>``.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

from . import datos
from .modelo_final import contribuciones

P26 = datos.RAIZ / "resultados" / "prueba_2026"
ART = datos.RAIZ / "artifacts"
TRAMOS = [-np.inf, 0, 2, 5, 10, 20, np.inf]
ETIQUETAS = ["≤ 0", "1-2", "3-5", "6-10", "11-20", "> 20"]


def presupuesto(d, modelos):
    filas = []
    meses = d.fecha_pedido.dt.to_period("M").nunique()
    for frac in (0.10, 0.20, 0.30):
        for n in modelos:
            k = int(round(frac * len(d)))
            marca = d[n].rank(method="first", ascending=False) <= k
            atrapa = marca & d.y.eq(1)
            filas.append({"presupuesto": frac, "modelo": n, "alertas": k, "tardías atrapadas": int(atrapa.sum()),
                          "fracción de las tardías": atrapa.sum() / d.y.sum(), "precisión": atrapa.sum() / k,
                          "órdenes con una tardía atrapada": d.oc_id[atrapa].nunique(),
                          "tardías atrapadas por mes": atrapa.sum() / meses, "meses": meses})
    return pd.DataFrame(filas)


def forma(d, shap, logit_log, grupo, nombre_grupo):
    t = pd.DataFrame({"tramo": pd.cut(d.plazo_habil_al_liberar, TRAMOS, labels=ETIQUETAS), "grupo": grupo,
                      "xgboost": shap, "logistica": logit_log})
    g = t.groupby(["grupo", "tramo"], observed=True).agg(posiciones=("xgboost", "size"), xgboost=("xgboost", "mean"),
                                                         logistica=("logistica", "mean")).reset_index()
    return g.rename(columns={"grupo": nombre_grupo})


def main():
    q = pd.read_csv(sys.argv[1], sep=";", decimal=",", dtype={"oc_id": str})
    df = datos.cargar()
    pru = datos.particion(df, "prueba")
    d = pru[["oc_id", "posicion", "fecha_pedido", "plazo_habil_al_liberar", "prov_tasa_tarde_previa", "clase_compras",
             datos.OBJETIVO]].merge(q[["oc_id", "posicion", "XGBoost 136"]], on=["oc_id", "posicion"], how="left",
                                    validate="1:1").rename(columns={datos.OBJETIVO: "y", "XGBoost 136": "XGBoost"})
    meta = json.loads((ART / "logistica.metadata.json").read_text(encoding="utf-8"))
    log = joblib.load(ART / "logistica.joblib")
    d["Regresión logística"] = log.predict_proba(pru[meta["entrada"]])[:, 1]
    presupuesto(d, ["XGBoost", "Regresión logística"]).to_csv(P26 / "presupuesto.csv", index=False)

    # forma del efecto del plazo: SHAP de XGBoost contra el término lineal de la logística
    xgb = joblib.load(ART / "xgboost.joblib")
    s, _ = contribuciones(xgb, pru[datos.ENTRADA])
    pre = log.named_steps["pre"]
    nombres = list(pre.get_feature_names_out())
    j = nombres.index("num__plazo_habil_al_liberar")
    x = pre.named_transformers_["num"].transform(pru[datos.NUMERICAS])[:, datos.NUMERICAS.index("plazo_habil_al_liberar")]
    lin = log.named_steps["modelo"].coef_[0, j] * np.asarray(x, dtype=float)
    lin = lin - lin.mean()
    tercio = pd.qcut(d.prov_tasa_tarde_previa.rank(method="first"), 3,
                     labels=["proveedor puntual", "proveedor medio", "proveedor que se retrasa"])
    tercio = tercio.astype(str).where(d.prov_tasa_tarde_previa.notna(), "proveedor sin historial")
    forma(d, s["plazo_habil_al_liberar"].to_numpy(), lin, tercio.to_numpy(), "proveedor").to_csv(
        P26 / "forma_plazo.csv", index=False)
    top = d.clase_compras.value_counts().head(4).index
    clase = d.clase_compras.where(d.clase_compras.isin(top), "otras").astype(str)
    forma(d, s["plazo_habil_al_liberar"].to_numpy(), lin, clase.to_numpy(), "clase de compras").to_csv(
        P26 / "forma_clase.csv", index=False)

    t = pd.read_csv(datos.RAIZ / "resultados" / "tabla_maestra.csv")
    filas = []
    for m, b, o in [("xgboost", "ninguno", "optuna"), ("logistica", "pesos", "grid")]:
        f = t[(t.modelo == m) & (t.balanceo == b) & (t.optimizador == o)].iloc[0]
        r = json.loads((datos.RAIZ / "resultados" / "experimentos" / "trabajos" / f"{m}__{b}__{o}__2025-S2.json")
                       .read_text(encoding="utf-8"))
        filas.append({"modelo": m, "balanceo": b, "optimizador": o, "horas_cpu_8_bloques": f.horas_cpu,
                      "segundos_un_ajuste": r["segundos_reajuste"], "filas_un_ajuste": r["filas_entrena"],
                      "ms_inferencia_por_1000": f.ms_inferencia_por_1000, "pico_gb": f.pico_gb, "AUC_8_bloques": f.AUC})
    pd.DataFrame(filas).to_csv(P26 / "costo.csv", index=False)
    print(pd.read_csv(P26 / "presupuesto.csv").round(3).to_string(index=False))
    print(pd.read_csv(P26 / "forma_plazo.csv").round(3).to_string(index=False))
    print(pd.read_csv(P26 / "costo.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
