"""El modelo original (celda «Propuesta 136 + nuevas» de la 2x2) entrenado con 2021-2025, para servirlo en el tablero.

Repite la fase ``prueba`` de ``evaluar_2x2_otif.py --celda prop_nuevas`` (K3 con 881 árboles, el modelo log-normal del
plazo y las 32 variables nuevas; Platt y umbral de ``config_2x2/prop_nuevas.json``, congelados en 2025), pero guarda las
piezas en vez de solo las probabilidades. Aquí no se elige nada. Comprueba que sus probabilidades en 2026 son las de la 2x2.

Lo publicado no lleva códigos de entidad, como en ``src/prediccion.py``: el tablero recibe el LightGBM del modelo y el
del plazo log-normal, y cada posición de ejemplo trae sus entidades ya convertidas en números (las tasas ``te_*`` y los
efectos aleatorios ``efecto_*``). Las tablas de efectos por código no salen de esta máquina.

Escribe:

- ``artifacts/original_servido.joblib`` y ``artifacts/original_servido.json`` (ficha, umbral, Platt, reproducción);
- ``data/prediccion/ejemplos_original.parquet``: 300 posiciones de la prueba 2026 (150 a tiempo y 150 tarde), que el
  modelo no vio al entrenar, con sus variables, sus efectos y el resultado real; sin ``oc_id`` ni códigos;
- ``data/prediccion/rangos_original.csv``: el rango de desarrollo de cada campo del formulario (percentiles 1 y 99 y la
  mediana de 2021-2025).

Uso (ml_venv, desde esta carpeta): ``python servir_original.py <ruta>/otif_depurado.csv``.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from modelo_original_otif import CORTE_OBJETIVO, ENTIDADES_RE, PRM_EFECTOS, BoostingMixtoOrdinal, sigmoide
from otif_comun import cargar, categorias_fijas, codificar_particion

RUTA = Path(sys.argv[1])
AQUI = Path(__file__).parent
REPO = AQUI.parent
SEMILLA = 2026
cfg = json.loads((AQUI / "config_2x2" / "prop_nuevas.json").read_text(encoding="utf-8"))
cfg_o = json.loads((AQUI / "original_congelado.json").read_text(encoding="utf-8"))
cfg_k3 = json.loads((AQUI / "k3_congelado.json").read_text(encoding="utf-8"))
ARBOLES = int(round(cfg_k3["arboles_validacion"] * 1.15))          # como la fase prueba de la 2x2
CAMPOS = ["plazo_habil_al_liberar", "plazo_prometido_dias", "prov_lt_med", "prov_tasa_tarde_previa",
          "efecto_proveedor", "efecto_material"]
t0 = time.time()
reloj = lambda: f"{(time.time() - t0) / 60:.1f} min"

# --- datos, como evaluar_2x2_otif.py -----------------------------------------------------------------------------------
df, numericas, pocas, entidades = cargar(RUTA)
df["oc_id"] = df.oc_id.astype(str)
nv = pd.read_parquet(RUTA.with_name("variables_nuevas.parquet")).astype({"oc_id": str})
NUEVAS = [c for c in nv.columns if c not in ("oc_id", "posicion")]
assert NUEVAS == cfg["variables_nuevas"], "las variables nuevas no son las congeladas"
df = df.merge(nv, on=["oc_id", "posicion"], how="left", validate="1:1")
e = df.particion.isin(["entrenamiento", "validacion"]).to_numpy()
v = df.particion.eq("prueba").to_numpy()
ent, eva = codificar_particion(df[e].copy(), df[v].copy(), entidades)
ent, eva = ent.reset_index(drop=True), eva.reset_index(drop=True)
cols = numericas + [f"te_{c}" for c in entidades if c not in ENTIDADES_RE] + pocas + NUEVAS
print(f"2021-2025 {len(ent)} -> 2026 {len(eva)} | {len(cols)} columnas | {reloj()}", flush=True)

# --- modelo log-normal del plazo (lognormal() de evaluar_2x2_otif.py, guardando el regresor) ---------------------------
sin_plazo = [c for c in cols if "plazo" not in c and "holgura" not in c and c != "nace_vencida"]
Xe_l, Xv_l = categorias_fijas(ent[sin_plazo], eva[sin_plazo], pocas)
lt = np.log1p((ent.plazo_habil_al_liberar + ent.dias_retraso).clip(lower=0).to_numpy())
reg = lambda X, y: lgb.LGBMRegressor(n_estimators=550, random_state=SEMILLA, verbose=-1, n_jobs=-1,
                                     **dict(PRM_EFECTOS, subsample_freq=1)).fit(X, y)
mu_e = np.zeros(len(ent))
for i_a, i_b in GroupKFold(n_splits=3).split(Xe_l, lt, ent.oc_id):
    mu_e[i_b] = reg(Xe_l.iloc[i_a], lt[i_a]).predict(Xe_l.iloc[i_b])
regresor = reg(Xe_l, lt)
mu_v = regresor.predict(Xv_l)
s = float(np.std(lt - mu_e))
z = lambda d, mu: (np.log1p(d.plazo_habil_al_liberar.clip(lower=0).to_numpy() + 1.5) - mu) / s
ent, eva = ent.assign(dist_mu=mu_e, dist_z=z(ent, mu_e)), eva.assign(dist_mu=mu_v, dist_z=z(eva, mu_v))
cols = cols + ["dist_mu", "dist_z"]
print(f"log-normal: s = {s:.3f} | {reloj()}", flush=True)

# --- K3 (k3() de evaluar_2x2_otif.py, guardando el modelo) -------------------------------------------------------------
f = pd.to_datetime(ent.fecha_pedido)
sem = (f.dt.year.astype(str) + "-S" + np.where(f.dt.month <= 6, "1", "2")).to_numpy()
X = categorias_fijas(ent[cols], ent[cols].iloc[:0], pocas)[0]
g = {c: ent[c] for c in ENTIDADES_RE}
y_bin = (ent.dias_retraso.to_numpy() > CORTE_OBJETIVO).astype(int)
ordenado = np.zeros(len(ent))
for t in sorted(set(sem))[1:]:
    antes, este = sem < t, sem == t
    aux = BoostingMixtoOrdinal(cfg_o["prm"], centrar=cfg_o["centrar"])
    aux._estimar_efectos(X[antes], y_bin[antes], {c: s_[antes] for c, s_ in g.items()}, ent.oc_id[antes],
                         np.zeros(antes.sum()))
    ordenado[este] = aux.desplazamiento({c: s_[este] for c, s_ in g.items()})
    print(f"   efectos ordenados {t} | {reloj()}", flush=True)
m = BoostingMixtoOrdinal(cfg_o["prm"], arboles=ARBOLES, monotono=cfg_o["monotono"], centrar=cfg_o["centrar"])
m.fit(X, ent.dias_retraso, g, ent.oc_id, efectos_ent=ordenado)
Xv = categorias_fijas(ent[cols], eva[cols], pocas)[1]
gv = {c: eva[c] for c in ENTIDADES_RE}
a, b = cfg["platt"]
p = sigmoide(a * m.logit(Xv, gv) + b)
print(f"K3: {ARBOLES} árboles | {reloj()}", flush=True)

# --- reproducción contra la 2x2 ----------------------------------------------------------------------------------------
q = eva[["oc_id", "posicion"]].merge(
    pd.read_csv(RUTA.with_name("predicciones_prueba_2x2.csv"), sep=";", decimal=",", dtype={"oc_id": str}),
    on=["oc_id", "posicion"], how="left", validate="1:1")[cfg["nombre"]].to_numpy()
rep = {"filas": int(len(p)), "dif_max": float(np.abs(p - q).max()), "correlacion": float(np.corrcoef(p, q)[0, 1]),
       "misma_decision": float(np.mean((p >= cfg["umbral"]) == (q >= cfg["umbral"])))}
print("reproducción:", rep, flush=True)
assert rep["misma_decision"] > 0.99, "el modelo servido no reproduce la 2x2"

# --- lo publicado ------------------------------------------------------------------------------------------------------
cats = {c: list(X[c].cat.categories) for c in pocas}
efectos = {f"efecto_{c}": m.desplazamiento({k: (gv[k] if k == c else pd.Series(["\0"] * len(eva))) for k in ENTIDADES_RE})
           for c in ENTIDADES_RE}
servido = {"modelo": m.modelo.booster_.model_to_string(num_iteration=m.arboles_usados),
           "lognormal": regresor.booster_.model_to_string(), "columnas": cols, "sin_plazo": sin_plazo, "categorias": cats,
           "s": s, "platt": [a, b], "umbral": cfg["umbral"], "corte": float(CORTE_OBJETIVO),
           "efectos": [f"efecto_{c}" for c in ENTIDADES_RE]}
art = REPO / "artifacts"
joblib.dump(servido, art / "original_servido.joblib", compress=3)
(art / "original_servido.json").write_text(json.dumps({
    "modelo": "BOREAL: boosting mixto + plazo log-normal + 32 variables nuevas, con Platt", "entrenado_con": "órdenes de 2021-2025",
    "umbral": cfg["umbral"], "arboles": m.arboles_usados, "platt": [a, b], "reproduccion_2x2": rep,
    "versiones": {"lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__}},
    ensure_ascii=False, indent=1), encoding="utf-8")

ej = eva.assign(**efectos)
ej = ej.groupby("y_tarde", group_keys=False).apply(lambda d: d.sample(150, random_state=SEMILLA))
salida = ej[cols + list(efectos) + ["y_tarde", "dias_retraso"]].drop(columns=["dist_mu", "dist_z"]).reset_index(drop=True)
assert not ({"oc_id", "posicion"} | set(entidades)) & set(salida.columns)   # ni la orden ni los códigos de entidad
d = REPO / "data" / "prediccion"
salida.to_parquet(d / "ejemplos_original.parquet", index=False)
rg = ent.assign(**{f"efecto_{c}": m.desplazamiento({k: (g[k] if k == c else pd.Series(["\0"] * len(ent)))
                                                     for k in ENTIDADES_RE}) for c in ENTIDADES_RE})
pd.DataFrame([{"campo": c, "p01": rg[c].quantile(0.01), "mediana": rg[c].median(), "p99": rg[c].quantile(0.99)}
              for c in CAMPOS]).to_csv(d / "rangos_original.csv", index=False)
print(f"escrito artifacts/original_servido.joblib, {len(salida)} ejemplos y {len(CAMPOS)} rangos | {reloj()}")
