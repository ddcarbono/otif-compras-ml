"""Comparacion 2x2 en la prueba 2026 (Daniel, 2026-10-23): modelo {XGBoost ganador de los 112, propuesta original} x
variables {las 136 del dataset, 136 + nuevas}, mas la ablacion de la propuesta. Separa el aporte del modelo (filas, con la
misma informacion) del aporte de la informacion (columnas, con el mismo modelo). CUARTA mirada a la prueba 2026 (la tercera fue con LightGBM de aprendiz):
exploratoria.

Celdas (--celda):
  xgb136, xgb_nuevas   XGBoost ganador: xgboost / ninguno / optuna, la mejor de la tabla maestra de los 112 (AUC medio
                       0,852 en los 8 bloques), con los hiperparametros del bloque mas reciente (2025-S2, elegidos en su
                       ventana interna; JSON del trabajo en otif-compras-ml/resultados/experimentos/trabajos/) y
                       datos.preprocesador() de ese repo, importado sin tocarlo. Las variables nuevas pasan por el mismo
                       tratamiento de las numericas (signo·log, mediana, escala).
  prop136, prop_nuevas Propuesta: K3 (k3_congelado.json: 766 arboles al validar, 881 al reentrenar) + modelo log-normal del
                       plazo (mejorar_otif.py, variante nuevas_dist): un LightGBM de regresion, sin las variables del plazo
                       prometido, estima mu = E[log(1 + plazo real)] (fuera de pliegue por OC al entrenar) y K3 recibe
                       dist_mu y dist_z = (log(1 + plazo prometido + 1,5) - mu) / s, con P(tarde) = 1 - Phi(dist_z) si el
                       plazo fuera log-normal. Es una pieza del modelo (usa las mismas variables), no informacion nueva.
  k3_136, k3_nuevas    ablacion: K3 sin el log-normal, con 136 y con 136 + nuevas.
Implementacion propia (2026-10-25): BOREAL y el regresor del log-normal son arboles_propios.BoostingHistograma (solo
numpy), con los hiperparametros congelados de original_congelado.json, PRM_EFECTOS y k3_congelado.json. Las celdas xgb*
no cambian (XGBoost es el rival, no el modelo propio) y se reusan sus archivos. Los efectos ordenados y los pliegues del
log-normal se reparten entre procesos (BOREAL_PROCESOS, por defecto 1).
Variables nuevas: las de variables_nuevas_otif.py que pasaron la regla de mejorar_otif.py (las 32).

Fases:
  --fase validacion --celda C   entrena con 2021-2024, predice 2025, ajusta Platt y el umbral de maximo F1 con toda la
                                validacion (como modelar_otif.py) y escribe config_2x2/<C>.json. Se commitea antes de la prueba.
  --fase prueba --celda C       entrena con 2021-2025 y guarda el logit crudo de 2026 en <datos>/2x2/<C>_logit.csv; en
                                prop_nuevas guarda además las piezas del modelo (<datos>/2x2/prop_nuevas_piezas.joblib,
                                con las tablas de efectos por código: no se publica) para servir_original.py.
  --fase platt --celda C        aplica a ese logit el Platt congelado en config_2x2/<C>.json -> <datos>/2x2/<C>.csv.
  --fase comparar               une las celdas y mide en pareja (bootstrap por OC), en toda la prueba y por entidad nueva.
Las celdas son independientes: se corren en procesos paralelos.

Uso (ml_venv): python evaluar_2x2_otif.py <ruta>/otif_depurado.csv --fase validacion|prueba --celda C
               python evaluar_2x2_otif.py <ruta>/otif_depurado.csv --fase comparar [--bootstrap 200]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import multiprocessing as mp
import os

from arboles_propios import BoostingHistograma, Discretizador
from modelo_original_otif import (CORTE_OBJETIVO, ENTIDADES_RE, PRM_EFECTOS, BoostingMixtoOrdinal, columnas,
                                  efectos_ordenados, escalar, sigmoide)
from otif_comun import cargar, categorias_fijas, codificar_particion, con_intervalo, marcar_nuevos

RUTA = Path(sys.argv[1])
arg = lambda n, d: type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
FASE, CELDA, B = arg("--fase", ""), arg("--celda", ""), arg("--bootstrap", 200)
CELDAS = ["xgb136", "xgb_nuevas", "prop136", "prop_nuevas", "k3_136", "k3_nuevas"]
NOMBRE = {"xgb136": "XGBoost 136", "xgb_nuevas": "XGBoost 136 + nuevas", "prop136": "BOREAL 136",
          "prop_nuevas": "BOREAL 136 + nuevas", "k3_136": "Boosting mixto 136 (ablación)",
          "k3_nuevas": "Boosting mixto 136 + nuevas (ablación)"}
assert FASE == "comparar" or (FASE in ("validacion", "prueba", "platt") and CELDA in CELDAS), "--fase y --celda"
AQUI = Path(__file__).parent
REPO = AQUI.parent  # en la copia de otif-compras-ml/original/, el repositorio es la carpeta de arriba
DIR_CFG = AQUI / "config_2x2"
DIR_PRED = RUTA.with_name("2x2")
DIR_CFG.mkdir(exist_ok=True)
DIR_PRED.mkdir(exist_ok=True)
cfg_o = json.loads((AQUI / "original_congelado.json").read_text(encoding="utf-8"))
cfg_k3 = json.loads((AQUI / "k3_congelado.json").read_text(encoding="utf-8"))
TRABAJO = REPO / "resultados" / "experimentos" / "trabajos" / "xgboost__ninguno__optuna__2025-S2.json"
SEMILLA = 2026
salida = []
t0 = time.time()
reloj = lambda: f"{(time.time() - t0) / 60:.1f} min"


def anotar(texto):
    print(texto, flush=True)
    salida.append(texto)


def ece(y, p, tramos=10):
    t = pd.qcut(p, tramos, labels=False, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "t": t}).groupby("t").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g.n * (g.y - g.p).abs()).sum() / g.n.sum())


# --- datos -------------------------------------------------------------------------------------------------------------
df, numericas, pocas, entidades = cargar(RUTA)
df["oc_id"] = df.oc_id.astype(str)
# en srvgdai no hay pyarrow: ahí va la copia en csv.gz
_gz = RUTA.with_name("variables_nuevas.csv.gz")
nv = (pd.read_csv(_gz) if _gz.exists() else pd.read_parquet(RUTA.with_name("variables_nuevas.parquet"))).astype({"oc_id": str})
NUEVAS = [c for c in nv.columns if c not in ("oc_id", "posicion")]
df = df.merge(nv, on=["oc_id", "posicion"], how="left", validate="1:1")


def partir():
    """(filas de entrenamiento, filas a predecir) de la fase."""
    if FASE == "validacion":
        return df.particion.eq("entrenamiento").to_numpy(), df.particion.eq("validacion").to_numpy()
    return df.particion.isin(["entrenamiento", "validacion"]).to_numpy(), df.particion.eq("prueba").to_numpy()


def xgb(e, v, con_nuevas):
    """Logit del XGBoost ganador entrenado en las filas e, para las filas v."""
    sys.path.insert(0, str(REPO))
    from src import datos as datos_repo
    from xgboost import XGBClassifier
    dr = datos_repo.cargar(RUTA)
    dr["oc_id"] = dr.oc_id.astype(str)
    dr = df[["oc_id", "posicion"]].merge(dr[["oc_id", "posicion"] + datos_repo.PREDICTORAS], on=["oc_id", "posicion"],
                                         how="left", validate="1:1")
    pre = datos_repo.preprocesador()
    Xe = pre.fit_transform(dr.loc[e, datos_repo.ENTRADA], df.y_tarde[e].astype(int))
    Xv = pre.transform(dr.loc[v, datos_repo.ENTRADA])
    Xe, Xv = [np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32) for x in (Xe, Xv)]
    if con_nuevas:
        num = Pipeline([("log", FunctionTransformer(datos_repo.log_signo)), ("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler())])
        # prov_carga_rel trae infinitos (mediana previa de abiertas = -1); LightGBM los tolera, el imputador no
        nuevas_ = lambda filas: df.loc[filas, NUEVAS].replace([np.inf, -np.inf], np.nan)
        Xe = np.hstack([Xe, num.fit_transform(nuevas_(e)).astype(np.float32)])
        Xv = np.hstack([Xv, num.transform(nuevas_(v)).astype(np.float32)])
    prm = json.loads(TRABAJO.read_text(encoding="utf-8"))["params"]
    m = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEMILLA, eval_metric="logloss", **prm)
    m.fit(Xe, df.y_tarde[e].astype(int).to_numpy())
    anotar(f"   XGBoost {prm}: {Xe.shape[1]} columnas, {Xe.shape[0]} filas | {reloj()}")
    return m.predict(Xv, output_margin=True)


_LN = {}
PIEZAS = {}


def _regresion(tarea):
    """Un ajuste del regresor del plazo: (filas de ajuste, filas a predecir: 'e' + indices o 'v')."""
    i_a, destino = tarea
    c = _LN
    m = BoostingHistograma(n_estimators=escalar(550), semilla=SEMILLA, objetivo="cuadratica", **PRM_EFECTOS)
    m.fit(c["Te"][:, i_a], c["lt"][i_a], c["cat"])
    if isinstance(destino, str):
        return destino, m.predict_raw(c["Tv"]), m
    return destino, m.predict_raw(c["Te"][:, destino]), None


def lognormal(ent, eva, cols):
    """dist_mu y dist_z del modelo log-normal del plazo real, fuera de pliegue por OC en ent (como mejorar_otif.py)."""
    sin_plazo = [c for c in cols if "plazo" not in c and "holgura" not in c and c != "nace_vencida"]
    Xe, Xv = categorias_fijas(ent[sin_plazo], eva[sin_plazo], pocas)
    lt = np.log1p((ent.plazo_habil_al_liberar + ent.dias_retraso).clip(lower=0).to_numpy())
    ce, cat = columnas(Xe)
    disc = Discretizador().fit(ce, cat)
    _LN.update(Te=disc.transform(ce), Tv=disc.transform(columnas(Xv)[0]), lt=lt, cat=cat)
    tareas = [(i_a, i_b) for i_a, i_b in GroupKFold(n_splits=3).split(Xe, lt, ent.oc_id)] + [(np.arange(len(lt)), "v")]
    procesos = int(os.environ.get("BOREAL_PROCESOS", "1"))
    if procesos > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(min(procesos, len(tareas))) as pool:
            resultados = pool.map(_regresion, tareas)
    else:
        resultados = map(_regresion, tareas)
    mu_e = np.zeros(len(ent))
    for destino, pred, modelo in resultados:
        if isinstance(destino, str):
            mu_v, regresor = pred, modelo
        else:
            mu_e[destino] = pred
    _LN.clear()
    s = float(np.std(lt - mu_e))
    PIEZAS.update(regresor=regresor, disc_ln=disc, s=s, sin_plazo=sin_plazo)
    z = lambda d, mu: (np.log1p(d.plazo_habil_al_liberar.clip(lower=0).to_numpy() + 1.5) - mu) / s
    ze, zv = z(ent, mu_e), z(eva, mu_v)
    anotar(f"   log-normal del plazo: s = {s:.3f}; AUC de 1 - Phi(z) solo en lo predicho "
           f"{roc_auc_score(eva.y_tarde.astype(int), norm.sf(zv)):.4f} | {reloj()}")
    return ent.assign(dist_mu=mu_e, dist_z=ze), eva.assign(dist_mu=mu_v, dist_z=zv)


def k3(e, v, arboles, con_nuevas, con_dist):
    """Logit de K3 (efectos ordenados por semestre como evaluar_k3_otif.py), entrenado en e, para v."""
    ent, eva = codificar_particion(df[e].copy(), df[v].copy(), entidades)
    ent, eva = ent.reset_index(drop=True), eva.reset_index(drop=True)
    cols = numericas + [f"te_{c}" for c in entidades if c not in ENTIDADES_RE] + pocas + (NUEVAS if con_nuevas else [])
    if con_dist:
        ent, eva = lognormal(ent, eva, cols)
        cols = cols + ["dist_mu", "dist_z"]
    f = pd.to_datetime(ent.fecha_pedido)
    sem = (f.dt.year.astype(str) + "-S" + np.where(f.dt.month <= 6, "1", "2")).to_numpy()
    X = categorias_fijas(ent[cols], ent[cols].iloc[:0], pocas)[0]
    g = {c: ent[c] for c in ENTIDADES_RE}
    y_bin = (ent.dias_retraso.to_numpy() > CORTE_OBJETIVO).astype(int)
    ordenado = efectos_ordenados(X, y_bin, {c: s.to_numpy() for c, s in g.items()}, ent.oc_id.to_numpy(), sem,
                                 cfg_o["prm"], centrar=cfg_o["centrar"], aviso=lambda s: anotar(s + f" | {reloj()}"))
    m = BoostingMixtoOrdinal(cfg_o["prm"], arboles=arboles, monotono=cfg_o["monotono"], centrar=cfg_o["centrar"])
    m.fit(X, ent.dias_retraso, g, ent.oc_id, efectos_ent=ordenado)
    Xv = categorias_fijas(ent[cols], eva[cols], pocas)[1]
    anotar(f"   K3: {len(cols)} columnas, {arboles} árboles | {reloj()}")
    PIEZAS.update(m=m, cols=cols)
    return m.logit(Xv, {c: eva[c] for c in ENTIDADES_RE})


def logit_celda(celda, e, v):
    arboles = cfg_k3["arboles_validacion"] if FASE == "validacion" else int(round(cfg_k3["arboles_validacion"] * 1.15))
    return {"xgb136": lambda: xgb(e, v, False), "xgb_nuevas": lambda: xgb(e, v, True),
            "prop136": lambda: k3(e, v, arboles, False, True), "prop_nuevas": lambda: k3(e, v, arboles, True, True),
            "k3_136": lambda: k3(e, v, arboles, False, False), "k3_nuevas": lambda: k3(e, v, arboles, True, False)}[celda]()


if FASE == "validacion":
    e, v = partir()
    y = df.y_tarde[v].astype(int).to_numpy()
    anotar(f"evaluar_2x2_otif.py --fase validacion --celda {CELDA} | 2021-2024 {e.sum()} -> 2025 {v.sum()}")
    z = logit_celda(CELDA, e, v)
    pl = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y)
    a, b = float(pl.coef_[0, 0]), float(pl.intercept_[0])
    p = sigmoide(a * z + b)
    prec, sens, u = precision_recall_curve(y, p)
    umbral = float(u[np.argmax(2 * prec[:-1] * sens[:-1] / np.maximum(prec[:-1] + sens[:-1], 1e-12))])
    cfg = dict(celda=CELDA, nombre=NOMBRE[CELDA], platt=[a, b], umbral=umbral, auc_validacion=round(roc_auc_score(y, z), 4),
               f1_validacion=round(f1_score(y, p >= umbral), 4), variables_nuevas=NUEVAS if "nuevas" in CELDA else [],
               xgb_trabajo=TRABAJO.name if CELDA.startswith("xgb") else None)
    (DIR_CFG / f"{CELDA}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    anotar(f"   AUC validación {cfg['auc_validacion']} | Platt ({a:.3f}, {b:.3f}) | umbral {umbral:.3f} (F1 {cfg['f1_validacion']}) "
           f"| congelado en config_2x2/{CELDA}.json | {reloj()}")

elif FASE == "prueba":
    # Entrena con 2021-2025 y guarda el logit crudo de 2026, sin Platt: puede correr en paralelo con la validación,
    # porque no lee nada de ella. La fase platt aplica después lo congelado en config_2x2/ (solo de 2025).
    e, v = partir()
    anotar(f"evaluar_2x2_otif.py --fase prueba --celda {CELDA} | 2021-2025 {e.sum()} -> 2026 {v.sum()} (logit crudo)")
    z = logit_celda(CELDA, e, v)
    df.loc[v, ["oc_id", "posicion", "y_tarde"]].assign(z=z).to_csv(DIR_PRED / f"{CELDA}_logit.csv", sep=";", decimal=",",
                                                                    index=False)
    if CELDA == "prop_nuevas":
        import joblib
        joblib.dump(PIEZAS, DIR_PRED / "prop_nuevas_piezas.joblib", compress=3)
        anotar("   piezas del modelo en 2x2/prop_nuevas_piezas.joblib (no se publica)")
    anotar(f"   guardado 2x2/{CELDA}_logit.csv | {reloj()}")

elif FASE == "platt":
    cfg = json.loads((DIR_CFG / f"{CELDA}.json").read_text(encoding="utf-8"))
    assert cfg["variables_nuevas"] in ([], NUEVAS), "las variables nuevas no son las congeladas"
    a, b = cfg["platt"]
    q = pd.read_csv(DIR_PRED / f"{CELDA}_logit.csv", sep=";", decimal=",", dtype={"oc_id": str})
    q.assign(p=sigmoide(a * q.z.to_numpy() + b)).drop(columns="z").to_csv(DIR_PRED / f"{CELDA}.csv", sep=";", decimal=",",
                                                                          index=False)
    anotar(f"evaluar_2x2_otif.py --fase platt --celda {CELDA} | Platt ({a:.3f}, {b:.3f}) de config_2x2/{CELDA}.json "
           f"-> 2x2/{CELDA}.csv")

else:
    pru = df[df.particion.eq("prueba")][["oc_id", "posicion", "y_tarde"]].reset_index(drop=True)
    y = pru.y_tarde.astype(int).to_numpy()
    leer = lambda ruta, col: pru[["oc_id", "posicion"]].merge(
        pd.read_csv(ruta, sep=";", decimal=",").astype({"oc_id": str}), on=["oc_id", "posicion"], how="left", validate="1:1")[col].to_numpy()
    p, u = {}, {}
    for c in CELDAS:
        p[NOMBRE[c]] = leer(DIR_PRED / f"{c}.csv", "p")
        u[NOMBRE[c]] = json.loads((DIR_CFG / f"{c}.json").read_text(encoding="utf-8"))["umbral"]
    p_lgb = leer(RUTA.with_name("predicciones_prueba.csv"), "LightGBM + Platt")
    assert all(np.isfinite(v_).all() for v_ in p.values())
    anotar(f"evaluar_2x2_otif.py --fase comparar | CUARTA mirada a la prueba 2026 (aprendiz propio), exploratoria | {len(y)} filas (tasa {y.mean():.3f})")
    nuevo, _ = marcar_nuevos(RUTA, df[df.particion.ne("prueba")], df[df.particion.eq("prueba")].reset_index(drop=True), entidades)
    oc = pru.oc_id.to_numpy()
    rng = np.random.default_rng(SEMILLA)
    N = NOMBRE
    pares = [(N["prop136"], N["xgb136"], "modelo, con las 136"),
             (N["prop_nuevas"], N["xgb_nuevas"], "modelo, con 136 + nuevas"),
             (N["xgb_nuevas"], N["xgb136"], "información, en XGBoost"),
             (N["prop_nuevas"], N["prop136"], "información, en la propuesta"),
             (N["prop_nuevas"], N["xgb136"], "propuesta completa contra el ganador del E2"),
             (N["prop_nuevas"], N["k3_nuevas"], "ablación: aporte del log-normal (con nuevas)"),
             (N["prop136"], N["k3_136"], "ablación: aporte del log-normal (con 136)")]
    dif = lambda fn: (lambda yy, pp: fn(yy, pp[:, 0]) - fn(yy, pp[:, 1]))
    for seg, s in [("toda la prueba", np.ones(len(y), bool)), ("con algún valor nuevo", nuevo), ("todo conocido", ~nuevo)]:
        yy, gg = y[s], oc[s]
        anotar(f"\n{seg} ({s.sum()} filas, {len(np.unique(gg))} OC, tasa {yy.mean():.3f}):")
        anotar("   celda | ROC-AUC [IC 95 %] | PR-AUC [IC 95 %] | F1 al umbral de validación [IC 95 %] | Brier | ECE")
        for k in p:
            anotar(f"   {k:28} {con_intervalo(yy, p[k][s], gg, roc_auc_score, rng, B)} | "
                   f"{con_intervalo(yy, p[k][s], gg, average_precision_score, rng, B)} | "
                   f"{con_intervalo(yy, p[k][s], gg, lambda a_, b_, k=k: f1_score(a_, b_ >= u[k]), rng, B)} | "
                   f"{brier_score_loss(yy, p[k][s]):.4f} | {ece(yy, p[k][s]):.4f}")
        anotar(f"   (referencia fuera de la comparación: LightGBM + Platt, ROC-AUC {roc_auc_score(yy, p_lgb[s]):.3f})")
        for a_, b_, que in pares:
            pp = np.c_[p[a_][s], p[b_][s]]
            f1d = lambda q, r, a_=a_, b_=b_: f1_score(q, r[:, 0] >= u[a_]) - f1_score(q, r[:, 1] >= u[b_])
            anotar(f"   {que:46} {a_} − {b_}: ROC-AUC {con_intervalo(yy, pp, gg, dif(roc_auc_score), rng, B)} | "
                   f"PR-AUC {con_intervalo(yy, pp, gg, dif(average_precision_score), rng, B)} | F1 {con_intervalo(yy, pp, gg, f1d, rng, B)}")
    pd.concat([pru, pd.DataFrame({"nuevo": nuevo.astype(int), **p})], axis=1).to_csv(
        RUTA.with_name("predicciones_prueba_2x2.csv"), sep=";", decimal=",", index=False)

sufijo = f"{FASE}_{CELDA}" if CELDA else FASE
(AQUI / f"evaluar_2x2_otif_{sufijo}.txt").write_text("\n".join(salida) + "\n", encoding="utf-8")
