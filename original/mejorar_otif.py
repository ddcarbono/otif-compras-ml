"""Busqueda de mejora con validacion por cuatro origenes (2026-10-23).

Por que otra validacion: en las rondas anteriores (revision_literatura/inventario_otif.md) las ganancias medidas en la
validacion 2025 (+0,003 a +0,007 de AUC) no se sostuvieron en la prueba 2026. Un solo ano de validacion es una sola
muestra de la deriva. Aqui cada pieza se mide en cuatro origenes que avanzan en el tiempo: entrena con las OC creadas
antes de T y mide en el semestre que empieza en T, con T = 2024-01-01, 2024-07-01, 2025-01-01 y 2025-07-01. La prueba
2026 no se toca.

REGLA, fijada antes de correr: una pieza pasa si la diferencia de AUC contra su referencia, juntando los cuatro
semestres, tiene intervalo al 95 % por encima de 0 Y gana en al menos 3 de los 4 semestres.

Aprendiz: el LightGBM actual (hiperparametros de modelar_otif.py, ensayo 33) con 550 arboles fijos en todas las variantes
y todos los origenes (474 fue la parada en 2025 con 2021-2024; fijo para no elegir mirando cada semestre).

Variantes (--variantes, separadas por coma):
  base             las 136 predictoras del dataset, como modelar_otif.py
  nuevas           + todas las de variables_nuevas_otif.py
  plazo            + solo plazo real y holgura empirica (*_lt_*, *_holgura_*)
  reciente         + solo lo reciente del proveedor (prov_tasa_ult*, prov_tasa_90d, prov_n_90d, prov_retraso_ult10, prov_carga_rel)
  pareja           + solo las parejas (provmat_*, provga_*, provcen_* sin lt ni holgura)
  nuevas_semillas  nuevas, promedio del logit de 5 semillas (submuestreo de filas y columnas distinto)
  nuevas_dist      nuevas + modelo de probabilidad del plazo: un LightGBM de regresion predice mu = E[log(1 + plazo real)]
                   (fuera de pliegue por OC en el entrenamiento) y la variable z = (log(1 + plazo prometido + 1,5) - mu) / s,
                   con s la desviacion de los residuos, que bajo un plazo log-normal da P(tarde) = 1 - Phi(z).
                   Tambien se reporta el AUC de 1 - Phi(z) solo.
Referencias: nuevas, plazo, reciente y pareja contra base; nuevas_semillas y nuevas_dist contra nuevas.

Techo (diagnostico): en cada semestre, la fraccion de OC con posiciones de desenlace mixto y el AUC del oraculo que
predice cada posicion con la tasa de las OTRAS posiciones de su OC. Mide cuanto del retraso es de la orden entera.

Uso (ml_venv): python mejorar_otif.py <ruta>/otif_depurado.csv [--variantes base,nuevas,...] [--bootstrap 200] [--sufijo x]
Deja mejorar_otif[_sufijo].txt aqui y los logit por semestre en <datos>/mejorar/logits_<variante>.parquet.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold

from otif_comun import cargar, categorias_fijas, codificar_particion

RUTA = Path(sys.argv[1])
arg = lambda n, d: type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
VARIANTES = arg("--variantes", "base,nuevas,plazo,reciente,pareja,nuevas_semillas,nuevas_dist").split(",")
B, SUFIJO, SEMILLA = arg("--bootstrap", 200), arg("--sufijo", ""), 2026
ARBOLES = 550
PRM = dict(learning_rate=0.01698, num_leaves=110, min_child_samples=143, subsample=0.981, subsample_freq=1,
           colsample_bytree=0.3655, reg_lambda=0.007801, reg_alpha=2.399, cat_smooth=1.542, min_data_per_group=28)
ORIGENES = ["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"]
REFERENCIA = {"nuevas": "base", "plazo": "base", "reciente": "base", "pareja": "base",
              "nuevas_semillas": "nuevas", "nuevas_dist": "nuevas"}
AQUI = Path(__file__).parent
SAL = RUTA.with_name("mejorar")
SAL.mkdir(exist_ok=True)
salida = []
t0 = time.time()
reloj = lambda: f"{(time.time() - t0) / 60:.1f} min"


def anotar(texto):
    print(texto, flush=True)
    salida.append(texto)


df, numericas, pocas, entidades = cargar(RUTA)
nv = pd.read_parquet(RUTA.with_name("variables_nuevas.parquet"))
df = df.merge(nv, on=["oc_id", "posicion"], how="left", validate="1:1")
NUEVAS = [c for c in nv.columns if c not in ("oc_id", "posicion")]
GRUPO = {"plazo": [c for c in NUEVAS if "_lt_" in c or "_holgura_" in c],
         "reciente": ["prov_tasa_ult10", "prov_tasa_ult30", "prov_tasa_90d", "prov_n_90d", "prov_retraso_ult10", "prov_carga_rel"],
         "pareja": [c for c in NUEVAS if c.startswith(("provmat_", "provga_", "provcen_")) and "_lt_" not in c and "_holgura_" not in c]}
df = df[df.particion.ne("prueba")].reset_index(drop=True)
df["fecha_pedido"] = pd.to_datetime(df.fecha_pedido)
base_cols = numericas + [f"te_{c}" for c in entidades] + pocas
anotar(f"mejorar_otif.py | {len(df)} filas 2021-2025 (sin la prueba) | {len(NUEVAS)} variables nuevas | variantes {VARIANTES} "
       f"| {ARBOLES} arboles fijos | {reloj()}")


def ajustar(X, y, Xa, semilla=SEMILLA, objetivo=None):
    prm = dict(PRM, random_state=semilla, verbose=-1, n_jobs=-1, n_estimators=ARBOLES)
    if objetivo == "regresion":
        return lgb.LGBMRegressor(**prm).fit(X, y).predict(Xa)
    return lgb.LGBMClassifier(**prm).fit(X, y).predict(Xa, raw_score=True)


def con_ic(y, s, oc, fn, rng):
    """Valor e intervalo al 95 % por bootstrap de OC; s puede tener varias columnas (fn las recibe todas)."""
    cod, inv = np.unique(oc, return_inverse=True)
    filas = pd.Series(np.arange(len(y))).groupby(inv).apply(np.array).to_list()
    vals = [fn(y[i], s[i]) for i in (np.concatenate([filas[k] for k in rng.integers(0, len(cod), len(cod))]) for _ in range(B))]
    return fn(y, s), *np.percentile(vals, [2.5, 97.5])


logits = {v: [] for v in VARIANTES}
extra = {"dist_solo": []}
techo = []
for T in ORIGENES:
    T = pd.Timestamp(T)
    ent = df[df.fecha_pedido < T].copy()
    ev = df[(df.fecha_pedido >= T) & (df.fecha_pedido < T + pd.DateOffset(months=6))].copy()
    ent, ev = codificar_particion(ent, ev, entidades)
    ent, ev = ent.reset_index(drop=True), ev.reset_index(drop=True)
    y, yv = ent.y_tarde.astype(int).to_numpy(), ev.y_tarde.astype(int).to_numpy()
    anotar(f"\norigen {T.date()}: entrena {len(ent)} (tasa {y.mean():.3f}) | mide {len(ev)} (tasa {yv.mean():.3f}, "
           f"{ev.oc_id.nunique()} OC) | {reloj()}")
    # techo: tasa de las otras posiciones de la misma OC
    g = ev.groupby("oc_id").y_tarde
    s_, n_ = g.transform("sum").to_numpy(), g.transform("size").to_numpy()
    multi = n_ > 1
    loo = (s_[multi] - yv[multi]) / (n_[multi] - 1)
    mixtas = ev[multi].groupby("oc_id").y_tarde.agg(lambda t: 0 < t.mean() < 1).mean()
    techo.append((T.date(), multi.mean(), mixtas, roc_auc_score(yv[multi], loo)))
    anotar(f"   techo: {multi.mean():.1%} de las filas en OC de varias posiciones; {mixtas:.1%} de esas OC con desenlace mixto; "
           f"AUC del oraculo de la OC {techo[-1][3]:.3f}")
    meta = ev[["oc_id", "posicion"]].assign(origen=str(T.date()), y=yv)

    def X_de(cols):
        return categorias_fijas(ent[cols], ev[cols], pocas)

    for v in VARIANTES:
        if v == "base":
            cols = base_cols
        elif v in ("nuevas", "nuevas_semillas", "nuevas_dist"):
            cols = base_cols + NUEVAS
        else:
            cols = base_cols + GRUPO[v]
        Xe, Xv = X_de(cols)
        if v == "nuevas_semillas":
            z = np.mean([ajustar(Xe, y, Xv, semilla=SEMILLA + k) for k in range(5)], axis=0)
        elif v == "nuevas_dist":
            # modelo del plazo real: mu = E[log(1 + lt)], sin el plazo prometido entre las variables
            lt_e = np.log1p((ent.plazo_habil_al_liberar + ent.dias_retraso).clip(lower=0).to_numpy())
            sin_plazo = [c for c in cols if "plazo" not in c and "holgura" not in c and c != "nace_vencida"]
            Xe_r, Xv_r = X_de(sin_plazo)
            mu_e = np.zeros(len(ent))
            for i_a, i_b in GroupKFold(n_splits=3).split(Xe_r, lt_e, ent.oc_id):
                mu_e[i_b] = ajustar(Xe_r.iloc[i_a], lt_e[i_a], Xe_r.iloc[i_b], objetivo="regresion")
            mu_v = ajustar(Xe_r, lt_e, Xv_r, objetivo="regresion")
            s = float(np.std(lt_e - mu_e))
            umbral_e = np.log1p(ent.plazo_habil_al_liberar.clip(lower=0).to_numpy() + 1.5)
            umbral_v = np.log1p(ev.plazo_habil_al_liberar.clip(lower=0).to_numpy() + 1.5)
            ze, zv = (umbral_e - mu_e) / s, (umbral_v - mu_v) / s
            p_solo = norm.sf(zv)
            extra["dist_solo"].append(meta.assign(z=np.log(np.clip(p_solo, 1e-9, 1) / np.clip(1 - p_solo, 1e-9, 1))))
            anotar(f"   plazo log-normal: s = {s:.3f}; AUC de 1 - Phi(z) solo {roc_auc_score(yv, p_solo):.4f}")
            z = ajustar(Xe.assign(dist_z=ze, dist_mu=mu_e), y, Xv.assign(dist_z=zv, dist_mu=mu_v))
        else:
            z = ajustar(Xe, y, Xv)
        logits[v].append(meta.assign(z=z))
        anotar(f"   {v:16} AUC {roc_auc_score(yv, z):.4f} | log loss {log_loss(yv, 1 / (1 + np.exp(-z))):.4f} | {reloj()}")

for v in VARIANTES:
    pd.concat(logits[v]).to_parquet(SAL / f"logits_{v}{'_' + SUFIJO if SUFIJO else ''}.parquet", index=False)

# --- resumen con la regla ------------------------------------------------------------------------------------------
rng = np.random.default_rng(SEMILLA)
anotar("\nAUC por semestre (origen = inicio del semestre medido):")
anotar("   variante         | " + " | ".join(ORIGENES))
for v in VARIANTES:
    anotar(f"   {v:16} | " + " | ".join(f"{roc_auc_score(t.y, t.z):.4f}    " for t in logits[v]))
anotar("\nDiferencias contra la referencia (intervalo por bootstrap de OC, juntando los cuatro semestres; el AUC se calcula "
       "dentro de cada semestre y se promedia):")
for v, ref in REFERENCIA.items():
    if v not in VARIANTES or ref not in VARIANTES:
        continue
    a, r = pd.concat(logits[v]).reset_index(drop=True), pd.concat(logits[ref]).reset_index(drop=True)
    assert (a.oc_id.to_numpy() == r.oc_id.to_numpy()).all()
    por = [roc_auc_score(x.y, x.z) - roc_auc_score(w.y, w.z) for x, w in zip(logits[v], logits[ref])]
    origen = a.origen.to_numpy()

    def d_auc(yy, s):
        return np.mean([roc_auc_score(yy[s[:, 2] == o], s[s[:, 2] == o, 0]) - roc_auc_score(yy[s[:, 2] == o], s[s[:, 2] == o, 1])
                        for o in np.unique(s[:, 2])])
    s = np.c_[a.z.to_numpy(), r.z.to_numpy(), pd.factorize(origen)[0]]
    m, lo, hi = con_ic(a.y.to_numpy(), s, (a.origen + "|" + a.oc_id.astype(str)).to_numpy(), d_auc, rng)
    gana = sum(d > 0 for d in por)
    pasa = lo > 0 and gana >= 3
    anotar(f"   {v:16} - {ref:8}: {m:+.4f} [{lo:+.4f}; {hi:+.4f}] | por semestre {', '.join(f'{d:+.4f}' for d in por)} "
           f"| gana en {gana}/4 | {'PASA' if pasa else 'no pasa'}")

(AQUI / f"mejorar_otif{'_' + SUFIJO if SUFIJO else ''}.txt").write_text("\n".join(salida) + "\n", encoding="utf-8")
