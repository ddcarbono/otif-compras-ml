"""BOREAL, el modelo original (celda «Propuesta 136 + nuevas» de la 2x2), entrenado con 2021-2025, para servirlo en el
tablero.

Toma las piezas que guarda ``evaluar_2x2_otif.py --fase prueba --celda prop_nuevas`` (``<datos>/2x2/prop_nuevas_piezas
.joblib``: el boosting mixto ordinal, el regresor del plazo log-normal y sus discretizadores; Platt y umbral de
``config_2x2/prop_nuevas.json``, congelados en 2025). Aquí no se entrena ni se elige nada. Comprueba que las
probabilidades en 2026 son las de la 2x2.

Lo publicado no lleva códigos de entidad, como en ``src/prediccion.py``: el tablero recibe los dos boosting (árboles y
discretizadores, sin las tablas de efectos) y cada posición de ejemplo trae sus entidades ya convertidas en números (las
tasas ``te_*`` y los efectos aleatorios ``efecto_*``). Las tablas de efectos por código no salen de esta máquina.

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
import numpy as np
import pandas as pd

from modelo_original_otif import ENTIDADES_RE, columnas, sigmoide
from otif_comun import cargar, categorias_fijas, codificar_particion

RUTA = Path(sys.argv[1])
AQUI = Path(__file__).parent
REPO = AQUI.parent
SEMILLA = 2026
cfg = json.loads((AQUI / "config_2x2" / "prop_nuevas.json").read_text(encoding="utf-8"))
CAMPOS = ["plazo_habil_al_liberar", "plazo_prometido_dias", "prov_lt_med", "prov_tasa_tarde_previa",
          "efecto_proveedor", "efecto_material"]
t0 = time.time()
reloj = lambda: f"{(time.time() - t0) / 60:.1f} min"

# --- datos, como evaluar_2x2_otif.py -----------------------------------------------------------------------------------
df, numericas, pocas, entidades = cargar(RUTA)
df["oc_id"] = df.oc_id.astype(str)
_gz = RUTA.with_name("variables_nuevas.csv.gz")
nv = (pd.read_csv(_gz) if _gz.exists() else pd.read_parquet(RUTA.with_name("variables_nuevas.parquet"))).astype({"oc_id": str})
NUEVAS = [c for c in nv.columns if c not in ("oc_id", "posicion")]
assert NUEVAS == cfg["variables_nuevas"], "las variables nuevas no son las congeladas"
df = df.merge(nv, on=["oc_id", "posicion"], how="left", validate="1:1")
e = df.particion.isin(["entrenamiento", "validacion"]).to_numpy()
v = df.particion.eq("prueba").to_numpy()
ent, eva = codificar_particion(df[e].copy(), df[v].copy(), entidades)
ent, eva = ent.reset_index(drop=True), eva.reset_index(drop=True)

pz = joblib.load(RUTA.with_name("2x2") / "prop_nuevas_piezas.joblib")
m, regresor, disc_ln, s, sin_plazo, cols = pz["m"], pz["regresor"], pz["disc_ln"], pz["s"], pz["sin_plazo"], pz["cols"]
print(f"2021-2025 {len(ent)} -> 2026 {len(eva)} | {len(cols)} columnas | {reloj()}", flush=True)

# --- 2026 con las piezas -----------------------------------------------------------------------------------------------
def dist(d, Xd):
    mu = regresor.predict_raw(disc_ln.transform(columnas(Xd)[0]))
    return d.assign(dist_mu=mu, dist_z=(np.log1p(d.plazo_habil_al_liberar.clip(lower=0).to_numpy() + 1.5) - mu) / s)


eva = dist(eva, categorias_fijas(ent[sin_plazo], eva[sin_plazo], pocas)[1])
Xv = categorias_fijas(ent[cols[:-2]].assign(dist_mu=0.0, dist_z=0.0), eva[cols], pocas)[1]
gv = {c: eva[c] for c in ENTIDADES_RE}
a, b = cfg["platt"]
p = sigmoide(a * m.logit(Xv, gv) + b)

q = eva[["oc_id", "posicion"]].merge(pd.read_csv(RUTA.with_name("2x2") / "prop_nuevas.csv", sep=";", decimal=",",
                                                 dtype={"oc_id": str}),
                                     on=["oc_id", "posicion"], how="left", validate="1:1")["p"].to_numpy()
rep = {"filas": int(len(p)), "dif_max": float(np.abs(p - q).max()), "correlacion": float(np.corrcoef(p, q)[0, 1]),
       "misma_decision": float(np.mean((p >= cfg["umbral"]) == (q >= cfg["umbral"])))}
print("reproducción:", rep, flush=True)
assert rep["misma_decision"] > 0.99, "el modelo servido no reproduce la 2x2"

# --- lo publicado: los árboles y discretizadores, sin niveles ni efectos por código ------------------------------------
sin_codigos = lambda d: {e: pd.Series(["\0"] * len(d)) for e in ENTIDADES_RE}
efectos = {f"efecto_{c}": m.desplazamiento({k: (gv[k] if k == c else sin_codigos(eva)[k]) for k in ENTIDADES_RE})
           for c in ENTIDADES_RE}
servido = {"modelo": m.modelo, "disc": m.disc, "lognormal": regresor, "disc_lognormal": disc_ln, "columnas": cols,
           "sin_plazo": sin_plazo, "s": s, "platt": [a, b], "umbral": cfg["umbral"],
           "corte_tramo": m.cortes.index(m.corte_objetivo) + 1, "efectos": [f"efecto_{c}" for c in ENTIDADES_RE]}
art = REPO / "artifacts"
joblib.dump(servido, art / "original_servido.joblib", compress=3)
(art / "original_servido.json").write_text(json.dumps({
    "modelo": "BOREAL: boosting mixto ordinal + plazo log-normal + 32 variables nuevas, con Platt (implementación propia, numpy)",
    "entrenado_con": "órdenes de 2021-2025", "umbral": cfg["umbral"], "arboles": m.arboles_usados, "platt": [a, b],
    "semilla": SEMILLA, "hiperparametros": {k: float(v) for k, v in m.prm.items()}, "reproduccion_2x2": rep, "versiones": {"pandas": pd.__version__, "numpy": np.__version__}},
    ensure_ascii=False, indent=1), encoding="utf-8")

ej = eva.assign(**efectos)
ej = ej.groupby("y_tarde", group_keys=False).apply(lambda d: d.sample(150, random_state=SEMILLA))
salida = ej[cols + list(efectos) + ["y_tarde", "dias_retraso"]].drop(columns=["dist_mu", "dist_z"]).reset_index(drop=True)
assert not ({"oc_id", "posicion"} | set(entidades)) & set(salida.columns)   # ni la orden ni los códigos de entidad
d = REPO / "data" / "prediccion"
salida.to_parquet(d / "ejemplos_original.parquet", index=False)
g = {c: ent[c] for c in ENTIDADES_RE}
rg = ent.assign(**{f"efecto_{c}": m.desplazamiento({k: (g[k] if k == c else sin_codigos(ent)[k]) for k in ENTIDADES_RE})
                   for c in ENTIDADES_RE})
pd.DataFrame([{"campo": c, "p01": rg[c].quantile(0.01), "mediana": rg[c].median(), "p99": rg[c].quantile(0.99)}
              for c in CAMPOS]).to_csv(d / "rangos_original.csv", index=False)
print(f"escrito artifacts/original_servido.joblib, {len(salida)} ejemplos y {len(CAMPOS)} rangos | {reloj()}")

# --- SHAP global en 2026, para compararlo con el de XGBoost (resultados/prueba_2026/shap_global.csv) ------------------
# 5.000 posiciones al azar; por variable, la media de |SHAP| del boosting en c = 1 y la de |efecto| de cada entidad
mu = np.random.default_rng(SEMILLA).choice(len(eva), min(5000, len(eva)), replace=False)
T = m._ampliar(m._discretizar(Xv.iloc[mu]), m.corte_objetivo)
sh, _ = m.modelo.shap(T)
g = pd.DataFrame(sh, columns=list(cols) + ["corte"]).drop(columns="corte")
for c in ENTIDADES_RE:
    g[c] = np.asarray(efectos[f"efecto_{c}"])[mu]
(pd.DataFrame({"variable": g.columns, "shap_abs_medio": g.abs().mean().to_numpy(), "shap_medio": g.mean().to_numpy()})
   .sort_values("shap_abs_medio", ascending=False)
   .to_csv(REPO / "resultados" / "prueba_2026" / "shap_global_original.csv", index=False))
print(f"SHAP global de BOREAL en {len(mu)} posiciones de 2026 | {reloj()}")
