"""Variables nuevas para OTIF (2026-10-23): plazo real del proveedor, historial reciente y por pareja.

Los 34 candidatos de revision_literatura/inventario_otif.md cambiaron la estructura del modelo, no la informacion. Este
script agrega informacion que el dataset no trae:
  - plazo real: lo que de verdad tardo cada entrega previa, en dias habiles desde la liberacion hasta la primera entrada
        lt = plazo_habil_al_liberar + dias_retraso      (exacto: las dos cuentan dias habiles con los festivos)
    mediana y percentil 80 de las ultimas 20 entregas del proveedor, de la pareja proveedor-material, proveedor-grupo de
    articulos y del material; y la holgura empirica = plazo_habil_al_liberar - esa mediana (o ese percentil). Hoy solo
    se compara la promesa contra el plazo del maestro (mc_plazo_prometido_menos_maestro), que es un dato administrativo.
  - lo reciente: tasa de retraso del proveedor en sus ultimas 10 y 30 posiciones resueltas y en los ultimos 90 dias, y
    el retraso medio de las ultimas 10. El dataset solo trae la tasa acumulada desde 2021.
  - parejas: tasa de proveedor-material, proveedor-grupo de articulos y proveedor-centro, encogida hacia la tasa reciente
    del proveedor: (tardes + 5 * tasa_prov) / (n + 5). El dataset solo cuenta cuantas OC tuvo la pareja.
  - carga relativa: prov_pos_abiertas sobre la mediana de las que tuvo abiertas en sus posiciones anteriores.

Sin mirar al futuro (como construir_otif.py, el historial se corta en fecha_pedido de la fila):
  - la primera entrada se reconstruye con entrada = prometida + dias_retraso dias habiles;
  - la etiqueta de una posicion previa se conoce el dia de su entrada o, si llego tarde, al vencer el dia de tolerancia
    (prometida + 2 dias habiles), lo que ocurra primero; su plazo real, el dia de la entrada; mas un dia de margen;
  - una posicion previa cuenta solo si se conocia ANTES de fecha_pedido. Las de la misma OC nunca cuentan (su entrada es
    posterior a la creacion de la orden).
  El historial sale solo de las posiciones del dataset (2021 en adelante): los primeros meses tienen poca historia, igual
  para todos los modelos que se comparan.

Uso (ml_venv): python variables_nuevas_otif.py <ruta>/otif_posiciones.csv
Deja variables_nuevas.parquet junto al dataset (llave oc_id + posicion) y variables_nuevas_otif.txt aqui.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RUTA = Path(sys.argv[1])
AQUI = Path(__file__).parent
ENCOGER = 5
salida = []
t0 = time.time()


def anotar(texto):
    print(texto, flush=True)
    salida.append(texto)


def pascua(a):
    # copiada de construir_otif.py, que no se puede importar (corre al cargarse)
    b, c = divmod(a, 100); d, e = divmod(b, 4); g = (8 * b + 13) // 25
    h = (19 * (a % 19) + b - d - g + 15) % 30; j, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * j - h - k) % 7; n = (a % 19 + 11 * h + 22 * m) // 451
    mes, dia = divmod(h + m - 7 * n + 114, 31)
    return pd.Timestamp(a, mes, dia + 1)


def festivos(a):
    lunes = lambda f: f + pd.Timedelta(days=(7 - f.dayofweek) % 7)
    fijos = [(1, 1), (5, 1), (7, 20), (8, 7), (12, 8), (12, 25)]
    movibles = [(1, 6), (3, 19), (6, 29), (8, 15), (10, 12), (11, 1), (11, 11)]
    p_ = pascua(a)
    f = [pd.Timestamp(a, m, d) for m, d in fijos] + [lunes(pd.Timestamp(a, m, d)) for m, d in movibles]
    f += [p_ - pd.Timedelta(days=3), p_ - pd.Timedelta(days=2)] + [lunes(p_ + pd.Timedelta(days=k)) for k in (39, 60, 68)]
    return f


FEST = np.array(sorted(set(sum([festivos(a) for a in range(2019, 2029)], []))), dtype="datetime64[D]")

cols = ["oc_id", "posicion", "proveedor", "material", "centro", "grupo_articulos", "fecha_pedido", "fecha_prometida",
        "plazo_habil_al_liberar", "dias_retraso", "prov_pos_abiertas", "prov_tasa_tarde_previa"]
x = pd.read_csv(RUTA, sep=";", decimal=",", low_memory=False, usecols=cols)
for c in ["fecha_pedido", "fecha_prometida"]:
    x[c] = pd.to_datetime(x[c])
for c in ["proveedor", "material", "centro", "grupo_articulos"]:
    x[c] = x[c].fillna("").astype(str).str.strip()
anotar(f"variables_nuevas_otif.py | {len(x)} posiciones | {RUTA.name}")

# --- cuando se conoce cada desenlace ---------------------------------------------------------------------------------
ok = x.dias_retraso.notna() & x.fecha_prometida.notna()
prom = x.fecha_prometida.values.astype("datetime64[D]")
d = np.nan_to_num(x.dias_retraso.to_numpy()).astype(int)
entrada = np.busday_offset(prom, d, roll="backward", holidays=FEST)
vence = np.busday_offset(prom, 2, roll="backward", holidays=FEST)
x["conoce_etiqueta"] = pd.to_datetime(np.minimum(entrada, vence)) + pd.Timedelta(days=1)
x["conoce_plazo"] = pd.to_datetime(entrada) + pd.Timedelta(days=1)
x["tarde"] = (x.dias_retraso > 1).astype(float)
x["plazo_real"] = x.plazo_habil_al_liberar + x.dias_retraso
x.loc[~ok, ["conoce_etiqueta", "conoce_plazo"]] = pd.NaT
anotar(f"   desenlace conocido en {ok.mean():.1%}; plazo real (dias habiles) p10/p50/p90: "
       f"{np.nanpercentile(x.plazo_real, [10, 50, 90]).round(1).tolist()} | negativo (entrada antes de liberar) {np.mean(x.plazo_real < 0):.2%}")
x["plazo_real"] = x.plazo_real.clip(lower=0)
fp = x.fecha_pedido.to_numpy(dtype="datetime64[ns]")


def historia(llave, cuando, valor):
    """Por fila: (indices de eventos de su llave conocidos antes de fecha_pedido) como (orden, j): los eventos de la llave
    ordenados por fecha de conocimiento y cuantos de ellos (j) son anteriores a la fila."""
    ev = pd.DataFrame({"k": llave, "t": cuando, "v": valor}).dropna(subset=["t", "v"])
    ev = ev[ev.k.ne("")].sort_values(["k", "t"])
    grupos = {k: (g.t.to_numpy(dtype="datetime64[ns]"), g.v.to_numpy(float)) for k, g in ev.groupby("k", sort=False)}
    return grupos


def estadisticos(llave, grupos, ventana_n=None, dias=None, cuantiles=None):
    """Media (o cuantiles) de los ultimos ventana_n eventos, o de los de los ultimos `dias`, conocidos antes de cada fila.
    Devuelve (estadistico, n usados)."""
    n_out = np.zeros(len(llave))
    out = np.full((len(llave), len(cuantiles) if cuantiles else 1), np.nan)
    filas = pd.Series(np.arange(len(llave))).groupby(np.asarray(llave)).apply(np.array)
    for k, idx in filas.items():
        if k == "" or k not in grupos:
            continue
        t, v = grupos[k]
        cs = np.concatenate([[0.0], np.cumsum(v)])
        j = np.searchsorted(t, fp[idx], side="left")
        if dias is not None:
            i0 = np.searchsorted(t, fp[idx] - np.timedelta64(dias, "D"), side="left")
        else:
            i0 = np.maximum(j - ventana_n, 0)
        n = j - i0
        n_out[idx] = n
        if cuantiles is None:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[idx, 0] = np.where(n > 0, (cs[j] - cs[i0]) / np.maximum(n, 1), np.nan)
        else:
            for r, a, b in zip(idx, i0, j):
                if b > a:
                    out[r] = np.percentile(v[a:b], cuantiles)
    return (out[:, 0] if cuantiles is None else out), n_out


nuevas = pd.DataFrame({"oc_id": x.oc_id, "posicion": x.posicion})
prov = x.proveedor.to_numpy()
llaves = {"prov": prov,
          "provmat": np.where(x.material.ne(""), x.proveedor + "|" + x.material, ""),
          "provga": np.where(x.grupo_articulos.ne(""), x.proveedor + "|" + x.grupo_articulos, ""),
          "provcen": x.proveedor + "|" + x.centro,
          "mat": x.material.to_numpy()}

# lo reciente del proveedor
g_et = historia(prov, x.conoce_etiqueta, x.tarde)
nuevas["prov_tasa_ult10"], _ = estadisticos(prov, g_et, ventana_n=10)
nuevas["prov_tasa_ult30"], _ = estadisticos(prov, g_et, ventana_n=30)
nuevas["prov_tasa_90d"], nuevas["prov_n_90d"] = estadisticos(prov, g_et, dias=90)
g_d = historia(prov, x.conoce_plazo, x.dias_retraso)
nuevas["prov_retraso_ult10"], _ = estadisticos(prov, g_d, ventana_n=10)
anotar(f"   reciente del proveedor listo | {(time.time() - t0) / 60:.1f} min")

# plazo real y holgura empirica
for nombre in ["prov", "provmat", "provga", "mat"]:
    g = historia(llaves[nombre], x.conoce_plazo, x.plazo_real)
    q, n = estadisticos(llaves[nombre], g, ventana_n=20, cuantiles=[50, 80])
    nuevas[f"{nombre}_lt_med"], nuevas[f"{nombre}_lt_q80"], nuevas[f"{nombre}_lt_n"] = q[:, 0], q[:, 1], n
    nuevas[f"{nombre}_holgura_med"] = x.plazo_habil_al_liberar.to_numpy() - q[:, 0]
    nuevas[f"{nombre}_holgura_q80"] = x.plazo_habil_al_liberar.to_numpy() - q[:, 1]
    anotar(f"   plazo real de {nombre}: con historia {np.mean(n > 0):.1%} de las filas | {(time.time() - t0) / 60:.1f} min")

# parejas, encogidas hacia la tasa reciente del proveedor (o la acumulada si no hay reciente)
ref = nuevas.prov_tasa_ult30.fillna(x.prov_tasa_tarde_previa).fillna(np.nanmean(x.tarde)).to_numpy()
for nombre in ["provmat", "provga", "provcen"]:
    g = historia(llaves[nombre], x.conoce_etiqueta, x.tarde)
    tasa, n = estadisticos(llaves[nombre], g, ventana_n=10 ** 9)
    nuevas[f"{nombre}_n"] = n
    nuevas[f"{nombre}_tasa"] = (np.nan_to_num(tasa) * n + ENCOGER * ref) / (n + ENCOGER)
    anotar(f"   pareja {nombre}: con historia {np.mean(n > 0):.1%} | {(time.time() - t0) / 60:.1f} min")

# carga relativa: abiertas hoy sobre la mediana de las abiertas en sus posiciones anteriores (por fecha_pedido)
orden = np.lexsort((fp, prov))
med = np.full(len(x), np.nan)
ab = x.prov_pos_abiertas.to_numpy(float)
for k, idx in pd.Series(orden).groupby(prov[orden]).apply(np.array).items():
    if k == "":
        continue
    v, f = ab[idx], fp[idx]
    for m_, (r, fr) in enumerate(zip(idx, f)):
        prev = v[:np.searchsorted(f, fr, side="left")]
        if len(prev):
            med[r] = np.median(prev[-50:])
nuevas["prov_carga_rel"] = ab / (med + 1)
anotar(f"   carga relativa lista | {(time.time() - t0) / 60:.1f} min")

anotar(f"\n{nuevas.shape[1] - 2} variables nuevas; vacios por variable:")
for c in nuevas.columns[2:]:
    anotar(f"   {c:22} vacio {nuevas[c].isna().mean():6.1%} | p50 {np.nanmedian(nuevas[c]):8.2f}")
nuevas.to_parquet(RUTA.with_name("variables_nuevas.parquet"), index=False)
anotar(f"guardado {RUTA.with_name('variables_nuevas.parquet').name} | {(time.time() - t0) / 60:.1f} min")
(AQUI / "variables_nuevas_otif.txt").write_text("\n".join(salida) + "\n", encoding="utf-8")
