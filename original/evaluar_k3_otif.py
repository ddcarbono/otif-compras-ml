"""K3 (modelo original con efectos ordenados en el tiempo) llevado a la prueba 2026: SEGUNDA mirada a la prueba, declarada
y exploratoria. La primera fue evaluar_original_otif.py --fase prueba (empate); K3 se eligio despues de verla, en la
validacion 2025 (sondear_frio_otif.py: parada temprana en 2025-S1, 766 arboles).

K3: el BoostingMixtoOrdinal congelado (original_congelado.json: hiperparametros, monotonia, correccion C), pero los
arboles entrenan con efectos ordenados: cada fila recibe la suma de efectos estimados solo con los semestres anteriores
al suyo (0 en el primero y para la entidad que aun no aparecia). Al predecir se usan los efectos con todo el entrenamiento.

--fase validacion  entrena con 2021-2024 con 766 arboles, ajusta Platt y el umbral de maximo F1 con TODA la validacion
                   2025 (como modelar_otif.py) y escribe k3_congelado.json, que se commitea antes de la prueba.
--fase prueba      entrena con 2021-2025 (arboles por 1,15, como modelar_otif.py), mide una vez en 2026 y compara en pareja
                   contra el LightGBM actual + Platt (predicciones_prueba.csv) y contra el original
                   (predicciones_prueba_original.csv), en toda la prueba y por entidad nueva o conocida.

Uso (ml_venv): python evaluar_k3_otif.py <ruta>/otif_depurado.csv --fase validacion|prueba [--bootstrap 200]
Deja evaluar_k3_otif_<fase>.txt aqui y, en la prueba, predicciones_prueba_k3.csv junto al dataset.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss, precision_recall_curve,
                             roc_auc_score)

from modelo_original_otif import CORTE_OBJETIVO, ENTIDADES_RE, SEMILLA, BoostingMixtoOrdinal, sigmoide
from otif_comun import cargar, categorias_fijas, codificar_particion, con_intervalo, marcar_nuevos

RUTA = sys.argv[1]
arg = lambda n, d: type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
FASE, B = arg("--fase", ""), arg("--bootstrap", 200)
assert FASE in ("validacion", "prueba"), "--fase validacion o --fase prueba"
AQUI = Path(__file__).parent
CONGELADO = AQUI / "k3_congelado.json"
cfg_o = json.loads((AQUI / "original_congelado.json").read_text(encoding="utf-8"))
ARBOLES_K3 = 766            # parada temprana en 2025-S1, sondear_frio_otif.txt
UMBRAL_ACTUAL = 0.334       # LightGBM + Platt, modelar_otif.txt
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


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def ajustar_k3(ent, arboles):
    """K3 sobre ent (ya codificado). Devuelve el modelo y X/grupos para predecir."""
    f = pd.to_datetime(ent.fecha_pedido)
    sem = (f.dt.year.astype(str) + "-S" + np.where(f.dt.month <= 6, "1", "2")).to_numpy()
    X = categorias_fijas(ent[cols], ent[cols].iloc[:0], pocas)[0]
    g = {e: ent[e] for e in ENTIDADES_RE}
    y_bin = (ent.dias_retraso.to_numpy() > CORTE_OBJETIVO).astype(int)
    ordenado = np.zeros(len(ent))
    for t in sorted(set(sem))[1:]:
        antes, este = sem < t, sem == t
        aux = BoostingMixtoOrdinal(cfg_o["prm"], centrar=cfg_o["centrar"])
        aux._estimar_efectos(X[antes], y_bin[antes], {e: v[antes] for e, v in g.items()}, ent.oc_id[antes], np.zeros(antes.sum()))
        ordenado[este] = aux.desplazamiento({e: v[este] for e, v in g.items()})
        anotar(f"   efectos ordenados para {t} con {antes.sum()} filas anteriores | {reloj()}")
    m = BoostingMixtoOrdinal(cfg_o["prm"], arboles=arboles, monotono=cfg_o["monotono"], centrar=cfg_o["centrar"])
    return m.fit(X, ent.dias_retraso, g, ent.oc_id, efectos_ent=ordenado)


df, numericas, pocas, entidades = cargar(RUTA)
cols = numericas + [f"te_{c}" for c in entidades if c not in ENTIDADES_RE] + pocas

if FASE == "validacion":
    ent, val = codificar_particion(df[df.particion.eq("entrenamiento")].copy(), df[df.particion.eq("validacion")].copy(), entidades)
    ent, val = ent.reset_index(drop=True), val.reset_index(drop=True)
    anotar(f"evaluar_k3_otif.py --fase validacion | entrenamiento 2021-2024 {len(ent)} | validación 2025 {len(val)} | {ARBOLES_K3} árboles")
    m = ajustar_k3(ent, ARBOLES_K3)
    X_val = categorias_fijas(ent[cols], val[cols], pocas)[1]
    y_val = val.y_tarde.astype(int).to_numpy()
    z = m.logit(X_val, {e: val[e] for e in ENTIDADES_RE})
    platt = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y_val)
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    p = sigmoide(a * z + b)
    prec, sens, u = precision_recall_curve(y_val, p)
    umbral = float(u[np.argmax(2 * prec[:-1] * sens[:-1] / np.maximum(prec[:-1] + sens[:-1], 1e-12))])
    anotar(f"K3 en validación 2025: AUC {roc_auc_score(y_val, z):.4f} | no monótonas {m.fraccion_no_monotona(X_val, {e: val[e] for e in ENTIDADES_RE}):.2%} "
           f"| Platt sigmoide({a:.4f}·logit + {b:.4f}) | Brier {brier_score_loss(y_val, sigmoide(z)):.4f} -> {brier_score_loss(y_val, p):.4f} "
           f"| ECE {ece(y_val, sigmoide(z)):.4f} -> {ece(y_val, p):.4f} | umbral de máximo F1 {umbral:.3f} (F1 {f1_score(y_val, p >= umbral):.3f}) | {reloj()}")
    CONGELADO.write_text(json.dumps(dict(variante="K3 efectos ordenados", base="original_congelado.json",
                                         arboles_validacion=ARBOLES_K3, platt=[a, b], umbral_f1=umbral,
                                         auc_validacion=round(roc_auc_score(y_val, z), 4)), indent=2), encoding="utf-8")
    anotar(f"congelado en {CONGELADO.name}: commitear antes de --fase prueba")

else:
    cfg = json.loads(CONGELADO.read_text(encoding="utf-8"))
    ent, pru = codificar_particion(df[df.particion.ne("prueba")].copy(), df[df.particion.eq("prueba")].copy(), entidades)
    ent, pru = ent.reset_index(drop=True), pru.reset_index(drop=True)
    arboles = int(round(cfg["arboles_validacion"] * 1.15))
    y = pru.y_tarde.astype(int).to_numpy()
    anotar(f"evaluar_k3_otif.py --fase prueba | SEGUNDA mirada a la prueba 2026, exploratoria | entrenamiento 2021-2025 "
           f"{len(ent)} | prueba {len(pru)} (tasa {y.mean():.3f}) | {arboles} árboles")
    m = ajustar_k3(ent, arboles)
    X_pru = categorias_fijas(ent[cols], pru[cols], pocas)[1]
    g_pru = {e: pru[e] for e in ENTIDADES_RE}
    a, b = cfg["platt"]
    p_k3 = sigmoide(a * m.logit(X_pru, g_pru) + b)
    anotar(f"ajustado | no monótonas {m.fraccion_no_monotona(X_pru, g_pru):.2%} | {reloj()}")
    nuevo, _ = marcar_nuevos(RUTA, ent, pru, entidades)
    t = pru[["oc_id", "posicion"]].merge(pd.read_csv(Path(RUTA).with_name("predicciones_prueba.csv"), sep=";", decimal=","),
                                         on=["oc_id", "posicion"], how="left", validate="1:1").merge(
        pd.read_csv(Path(RUTA).with_name("predicciones_prueba_original.csv"), sep=";", decimal=",")[["oc_id", "posicion", "original + Platt"]],
        on=["oc_id", "posicion"], how="left", validate="1:1")
    assert t["LightGBM + Platt"].notna().all() and (t.y_tarde.to_numpy() == y).all(), "predicciones_prueba.csv no cuadra"
    p_act, p_org = t["LightGBM + Platt"].to_numpy(), t["original + Platt"].to_numpy()
    oc = pru.oc_id.to_numpy()
    rng = np.random.default_rng(SEMILLA)
    u_k, u_a, u_o = cfg["umbral_f1"], UMBRAL_ACTUAL, json.loads((AQUI / "original_congelado.json").read_text())["umbral_f1"]
    dif = lambda fn: (lambda yy, pp: fn(yy, pp[:, 0]) - fn(yy, pp[:, 1]))
    for seg, s in [("toda la prueba", np.ones(len(y), bool)), ("con algún valor nuevo", nuevo), ("todo conocido", ~nuevo)]:
        yy, gg = y[s], oc[s]
        anotar(f"\n{seg} ({s.sum()} filas, {len(np.unique(gg))} OC, tasa {yy.mean():.3f}):")
        anotar("   modelo | ROC-AUC [IC 95 %] | PR-AUC [IC 95 %] | F1 al umbral de validación [IC 95 %] | Brier | log loss | ECE")
        for nombre, p, u in [("K3 + Platt", p_k3, u_k), ("LightGBM actual + Platt", p_act, u_a), ("original + Platt", p_org, u_o)]:
            anotar(f"   {nombre:24} {con_intervalo(yy, p[s], gg, roc_auc_score, rng, B)} | "
                   f"{con_intervalo(yy, p[s], gg, average_precision_score, rng, B)} | "
                   f"{con_intervalo(yy, p[s], gg, lambda a_, b_: f1_score(a_, b_ >= u), rng, B)} | "
                   f"{brier_score_loss(yy, p[s]):.4f} | {log_loss(yy, np.clip(p[s], 1e-6, 1 - 1e-6)):.4f} | {ece(yy, p[s]):.4f}")
        for ref, pr, ur in [("actual", p_act, u_a), ("original", p_org, u_o)]:
            pp = np.c_[p_k3[s], pr[s]]
            f1d = lambda a_, b_: f1_score(a_, b_[:, 0] >= u_k) - f1_score(a_, b_[:, 1] >= ur)
            anotar(f"   diferencia K3 - {ref}: ROC-AUC {con_intervalo(yy, pp, gg, dif(roc_auc_score), rng, B)} | PR-AUC "
                   f"{con_intervalo(yy, pp, gg, dif(average_precision_score), rng, B)} | F1 {con_intervalo(yy, pp, gg, f1d, rng, B)}")
    pd.DataFrame({"oc_id": pru.oc_id, "posicion": pru.posicion, "y_tarde": y, "nuevo": nuevo.astype(int), "K3 + Platt": p_k3}).to_csv(
        Path(RUTA).with_name("predicciones_prueba_k3.csv"), sep=";", decimal=",", index=False)

(AQUI / f"evaluar_k3_otif_{FASE}.txt").write_text("\n".join(salida) + "\n", encoding="utf-8")
