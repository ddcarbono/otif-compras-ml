"""Piezas comunes del modelado OTIF (las usan sondear_senal_otif.py y modelar_otif.py).

- cargar: lee otif_depurado.csv, lleva los infinitos a vacio y separa las columnas por tipo.
- Codificacion jerarquica por objetivo: cada valor recibe su tasa de retraso
  suavizada hacia la de su nivel superior, y esta hacia la global:
      tasa(v) = (retrasos(v) + M * tasa(padre)) / (n(v) + M),   M = 20
  Un valor que no estaba en el entrenamiento recibe la tasa de su padre. En el entrenamiento se calcula fuera de pliegue
  (5 pliegues agrupados por OC), para que una fila no vea su objetivo ni el de las posiciones de su misma OC.
- marcar_nuevos: filas con alguna entidad que no estaba en el entrenamiento, mirando el seudonimo original de
  otif_posiciones.csv (en el depurado una persona nueva ya va agrupada como "(otras <grupo>)", que puede existir antes).
- con_intervalo: metrica con intervalo al 95 % por bootstrap de OC (las posiciones de una OC no son independientes).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

NO_PREDICTORAS = {"oc_id", "posicion", "fecha_pedido", "fecha_prediccion", "fecha_prometida", "y_tarde", "y_otif_falla",
                  "dias_retraso", "particion"}
# entidad -> nivel superior. El ultimo de cada cadena cae a la tasa global.
JERARQUIA = {"proveedor": "pv_region", "material": "grupo_articulos", "imp_ceco": "sociedad", "imp_cuenta": "sociedad",
             "sp_creador": "grupo_compras", "creador_oc": "grupo_compras", "ultimo_liberador": "estrategia_lib",
             "centro": "sociedad", "estrategia_lib": "sociedad", "grupo_compras": "sociedad", "sociedad": "linea",
             "grupo_articulos": "clase_compras", "pv_region": None, "linea": None, "clase_compras": None}
ENTIDADES = ["proveedor", "material", "imp_ceco", "imp_cuenta", "sp_creador", "creador_oc", "ultimo_liberador", "centro",
             "sociedad", "estrategia_lib", "grupo_compras"]     # se codifican por objetivo y salen como codigo
M = 20


def cargar(ruta):
    df = pd.read_csv(ruta, sep=";", decimal=",", low_memory=False)
    predictoras = [c for c in df.columns if c not in NO_PREDICTORAS]
    categoricas = [c for c in predictoras if df[c].dtype == object]
    numericas = [c for c in predictoras if c not in categoricas]
    df[numericas] = df[numericas].replace([np.inf, -np.inf], np.nan)
    entidades = [c for c in ENTIDADES if c in df.columns]
    pocas = [c for c in categoricas if c not in entidades]
    return df, numericas, pocas, entidades


def _codificar(c, ent, y_ent, apl, glob):
    padre = JERARQUIA.get(c)
    prior = _codificar(padre, ent, y_ent, apl, glob) if padre else np.full(len(apl[c]), glob)
    t = pd.DataFrame({"k": ent[c].values, "y": y_ent.values}).groupby("k").y.agg(["sum", "count"])
    s = apl[c].map(t["sum"]).fillna(0).to_numpy()
    n = apl[c].map(t["count"]).fillna(0).to_numpy()
    return (s + M * prior) / (n + M)


def te(df_ent, y_ent, df_apl, columnas):
    """Codificacion jerarquica de columnas para df_apl, estimada con (df_ent, y_ent)."""
    ent = {c: df_ent[c].astype(str) for c in JERARQUIA if c in df_ent.columns}
    apl = {c: df_apl[c].astype(str) for c in JERARQUIA if c in df_apl.columns}
    return pd.DataFrame({f"te_{c}": _codificar(c, ent, y_ent, apl, y_ent.mean()) for c in columnas}, index=df_apl.index)


def te_fuera_de_pliegue(df, y, grupos, columnas):
    out = pd.DataFrame(index=df.index, columns=[f"te_{c}" for c in columnas], dtype=float)
    for i_e, i_a in GroupKFold(n_splits=5).split(df, y, grupos):
        out.iloc[i_a] = te(df.iloc[i_e], y.iloc[i_e], df.iloc[i_a], columnas).to_numpy()
    return out


def codificar_particion(ent, pru, columnas):
    """Devuelve (ent, pru) con las columnas te_*: fuera de pliegue en ent, con todo ent en pru."""
    y = ent["y_tarde"].astype(int)
    return (pd.concat([ent, te_fuera_de_pliegue(ent, y, ent["oc_id"], columnas)], axis=1),
            pd.concat([pru, te(ent, y, pru, columnas)], axis=1))


def marcar_nuevos(ruta_depurado, ent, pru, entidades):
    """Mascara de filas de pru con alguna entidad que no esta en ent, y la tasa de nuevo por entidad.

    Sin otif_posiciones.csv al lado (el paquete compartido no lo trae) compara los valores del depurado, donde lo raro ya
    va agrupado: cuenta menos filas nuevas que con el seudonimo original."""
    ruta = Path(ruta_depurado).with_name("otif_posiciones.csv")
    if ruta.exists():
        crudo = pd.read_csv(ruta, sep=";", decimal=",", low_memory=False,
                            usecols=["oc_id", "posicion"] + entidades).set_index(["oc_id", "posicion"])
        oe = crudo.loc[pd.MultiIndex.from_frame(ent[["oc_id", "posicion"]])]
        op = crudo.loc[pd.MultiIndex.from_frame(pru[["oc_id", "posicion"]])]
    else:
        print("aviso: sin otif_posiciones.csv, las filas nuevas se marcan con los valores ya agrupados del depurado")
        oe, op = ent, pru
    nuevo, por_campo = np.zeros(len(pru), dtype=bool), {}
    for c in entidades:
        m = ~op[c].astype(str).isin(set(oe[c].astype(str))).to_numpy()
        nuevo |= m
        por_campo[c] = m.mean()
    return nuevo, por_campo


def con_intervalo(y, p, grupos, fn, rng, b=200):
    """Metrica y su intervalo al 95 % por bootstrap de OC."""
    y, p = np.asarray(y), np.asarray(p)
    codigos, inv = np.unique(np.asarray(grupos), return_inverse=True)
    filas = pd.Series(np.arange(len(y))).groupby(inv).apply(np.array).to_list()
    vals = []
    for _ in range(b):
        idx = np.concatenate([filas[k] for k in rng.integers(0, len(codigos), len(codigos))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(fn(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"{fn(y, p):.3f} [{lo:.3f}, {hi:.3f}]"


def categorias_fijas(X_ent, X_apl, pocas):
    """Categorias de LightGBM fijadas con las del entrenamiento: un valor nuevo queda como vacio, nunca con el codigo de otro."""
    X_ent, X_apl = X_ent.copy(), X_apl.copy()
    for c in pocas:
        cats = pd.Index(X_ent[c].dropna().unique())
        X_ent[c] = pd.Categorical(X_ent[c], categories=cats)
        X_apl[c] = pd.Categorical(X_apl[c], categories=cats)
    return X_ent, X_apl
