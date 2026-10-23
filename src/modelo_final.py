"""El modelo elegido en las 112 (XGBoost sin balanceo, Optuna) entrenado con 2021-2025, con Platt, para servirlo en el
tablero y explicarlo con SHAP en la prueba 2026.

Es la celda «XGBoost 136» de la comparación 2x2 (``maestria/machine_learning/otif/evaluar_2x2_otif.py``): los
hiperparámetros del trabajo del bloque más reciente (``xgboost__ninguno__optuna__2025-S2.json``), Platt y umbral de
máximo F1 ajustados en la validación 2025 y congelados antes de la prueba (``resultados/prueba_2026/config_xgb136.json``,
copia de ``config_2x2/xgb136.json``). Aquí no se elige nada: se reentrena lo mismo para tener el modelo en un archivo, y
se comprueba que sus probabilidades en 2026 son las de la 2x2.

Escribe:

- ``artifacts/xgboost.joblib`` (no se publica: su codificador guarda la tasa de cada código de entidad) y
  ``artifacts/xgboost.metadata.json``; con ``python -m src.prediccion --modelo xgboost`` pasa a ser el modelo servido;
- ``resultados/prueba_2026/reproduccion.json``: la comparación con las probabilidades de la 2x2;
- ``resultados/prueba_2026/shap_global.csv``: la media del valor SHAP absoluto de cada variable en 2026 (TreeSHAP exacto,
  en la escala del margen, antes de Platt; un one-hot suma sus columnas);
- ``resultados/prueba_2026/shap_local.csv``: las 8 contribuciones mayores de tres posiciones de 2026 (un verdadero positivo,
  un falso negativo y un falso positivo, los de puntaje mediano de su caso), sin identificadores.

Uso (ml_venv, con OTIF_CSV): ``python -m src.modelo_final <predicciones_prueba_2x2.csv>``.
"""
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from . import datos
from .calibrado import XGBoostPlatt, denso

P26 = datos.RAIZ / "resultados" / "prueba_2026"
TRABAJO = datos.RAIZ / "resultados" / "experimentos" / "trabajos" / "xgboost__ninguno__optuna__2025-S2.json"
COLUMNA = "XGBoost 136"


def contribuciones(pipe, X):
    """SHAP por variable (filas × variables) y el valor base, a partir de las columnas preprocesadas."""
    import xgboost
    pre = pipe.named_steps["pre"]
    c = pipe.named_steps["modelo"].get_booster().predict(xgboost.DMatrix(denso(pre.transform(X))), pred_contribs=True)
    nombres = pd.Series(pre.get_feature_names_out())
    var = nombres.str.split("__").str[1].str.replace(r"^te_", "", regex=True)
    cat = nombres.str.split("__").str[0].eq("cat")
    for v in datos.CATEGORICAS:
        var[cat & var.str.startswith(v + "_")] = v
    s = pd.DataFrame(c[:, :-1], columns=nombres).T.groupby(var.to_numpy()).sum().T
    return s, c[:, -1]


def main():
    ruta = sys.argv[1]
    cfg = json.loads((P26 / "config_xgb136.json").read_text(encoding="utf-8"))
    prm = json.loads(TRABAJO.read_text(encoding="utf-8"))["params"]
    df = datos.cargar()
    ent, pru = df[df.particion.ne("prueba")], datos.particion(df, "prueba")
    pipe = Pipeline([("pre", datos.preprocesador()), ("modelo", XGBoostPlatt(prm, tuple(cfg["platt"]), datos.SEMILLA))])
    pipe.fit(ent[datos.ENTRADA], ent[datos.OBJETIVO].astype(int))

    # comprobación contra la 2x2
    p = pipe.predict_proba(pru[datos.ENTRADA])[:, 1]
    q = pd.read_csv(ruta, sep=";", decimal=",", dtype={"oc_id": str})
    q = pru[["oc_id", "posicion"]].merge(q, on=["oc_id", "posicion"], how="left", validate="1:1")
    y = pru[datos.OBJETIVO].astype(int).to_numpy()
    rep = {"filas": len(y), "AUC_aqui": roc_auc_score(y, p), "AUC_2x2": roc_auc_score(y, q[COLUMNA]),
           "dif_max": float(np.abs(p - q[COLUMNA]).max()), "correlacion": float(np.corrcoef(p, q[COLUMNA])[0, 1]),
           "misma_decision": float(((p >= cfg["umbral"]) == (q[COLUMNA] >= cfg["umbral"])).mean())}
    P26.mkdir(parents=True, exist_ok=True)
    (P26 / "reproduccion.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(rep, flush=True)

    art = datos.RAIZ / "artifacts"
    joblib.dump(pipe, art / "xgboost.joblib", compress=3)
    (art / "xgboost.metadata.json").write_text(json.dumps({
        "modelo": "XGBoost sin balanceo (Optuna) con Platt", "entrenado_con": "órdenes de 2021-2025",
        "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__, "xgboost": __import__("xgboost").__version__},
        "semilla": datos.SEMILLA, "hiperparametros": prm, "platt": cfg["platt"], "umbral": cfg["umbral"],
        "entrada": list(datos.ENTRADA), "metricas_prueba": {"ROC-AUC": rep["AUC_2x2"]}}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    # SHAP en 2026
    s, base = contribuciones(pipe, pru[datos.ENTRADA])
    g = s.abs().mean().sort_values(ascending=False)
    pd.DataFrame({"variable": g.index, "shap_abs_medio": g.to_numpy(),
                  "shap_medio": s.mean()[g.index].to_numpy()}).to_csv(P26 / "shap_global.csv", index=False)
    marcada = p >= cfg["umbral"]
    filas = []
    for caso, m in [("verdadero positivo", (y == 1) & marcada), ("falso negativo", (y == 1) & ~marcada),
                    ("falso positivo", (y == 0) & marcada)]:
        idx = np.flatnonzero(m)
        i = idx[np.argsort(p[idx])[len(idx) // 2]]
        top = s.iloc[i].reindex(s.iloc[i].abs().sort_values(ascending=False).index).head(8)
        for v, c in top.items():
            x = pru.iloc[i][v] if v in pru.columns else np.nan
            filas.append({"caso": caso, "probabilidad": p[i], "base": base[i], "variable": v, "contribucion": c,
                          "valor": x if not datos.TIPO.get(v) == "entidad" else "(código de entidad)"})
    pd.DataFrame(filas).to_csv(P26 / "shap_local.csv", index=False)
    print(g.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
