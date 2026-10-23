"""El modelo que sirve el tablero para la predicción en vivo, sin publicar identificadores.

El *pipeline* entrenado (``artifacts/<modelo>.joblib``) guarda, en su codificador de entidades, una tabla de cada código
de proveedor, material, comprador o liberador con su tasa de retraso. Esas tablas no se publican: el tablero recibe
``artifacts/servido.joblib``, que solo lleva las transformaciones de las numéricas y las categóricas, el modelo y el
umbral. Las entidades llegan ya convertidas en tasas (``te_*``), que es lo que el modelo usa.

Escribe:

- ``artifacts/servido.joblib`` y ``artifacts/servido.json`` (nombre, umbral, entrenado con, métricas);
- ``data/prediccion/ejemplos.parquet``: 300 posiciones reales de la validación 2025 (150 a tiempo y 150 tarde), con
  las predictoras en crudo, las entidades como tasas y el resultado real; sin ``oc_id`` ni códigos de entidad;
- ``data/prediccion/rangos.csv``: el rango de desarrollo de cada campo del formulario (percentiles 1 y 99 y la mediana
  de 2021-2025, o las categorías vistas con su frecuencia).

Uso: ``python -m src.prediccion [--modelo logistica]`` desde la raíz, con ``OTIF_CSV``. Para servir otro modelo basta
con que ``artifacts/<modelo>.joblib`` sea un ``Pipeline([("pre", datos.preprocesador()), ("modelo", ...)])`` y tenga su
``<modelo>.metadata.json`` con el umbral.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from . import datos

CAMPOS_NUM = ["plazo_habil_al_liberar", "plazo_prometido_dias", "dias_habiles_liberacion", "valor_oc_cop",
              "con_liberacion", "prov_tasa_tarde_previa"]
CAMPOS_CAT = ["linea", "moneda", "grupo_articulos", "clase_compras"]
CAMPOS_TE = ["te_proveedor", "te_material"]


def separar(pipe):
    """Del pipeline entrenado, lo que se puede publicar."""
    pre = pipe.named_steps["pre"]
    ent = pre.named_transformers_["ent"]
    if hasattr(pipe.named_steps["modelo"], "modelo_"):
        # XGBoost se entrenó con todos los hilos; en Render (una fracción de CPU) eso son decenas de hilos peleando por
        # ella y una predicción que no termina. Una fila no necesita más de uno.
        pipe.named_steps["modelo"].modelo_.set_params(n_jobs=1)
        pipe.named_steps["modelo"].modelo_.get_booster().set_param({"nthread": 1})
    return {"num": pre.named_transformers_["num"], "cat": pre.named_transformers_["cat"],
            "num_cols": list(datos.NUMERICAS), "cat_cols": list(datos.CATEGORICAS),
            "te_cols": list(ent.get_feature_names_out()), "modelo": pipe.named_steps["modelo"],
            "nombres": list(pre.get_feature_names_out())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="logistica")
    a = ap.parse_args()
    art = datos.RAIZ / "artifacts"
    pipe = joblib.load(art / f"{a.modelo}.joblib")
    meta = json.loads((art / f"{a.modelo}.metadata.json").read_text(encoding="utf-8"))
    serv = separar(pipe)
    joblib.dump(serv, art / "servido.joblib", compress=3)
    (art / "servido.json").write_text(json.dumps({
        "fuente": f"{a.modelo}.joblib", "modelo": meta["modelo"], "entrenado_con": meta["entrenado_con"],
        "umbral": meta["umbral"], "metricas_prueba": meta.get("metricas_prueba", {}),
        "versiones": meta["versiones"]}, ensure_ascii=False, indent=1), encoding="utf-8")

    df = datos.cargar()
    ev = df[df.particion.ne("prueba")]
    val = datos.particion(df, "validacion")
    ej = val.groupby("y_tarde", group_keys=False).apply(lambda d: d.sample(150, random_state=datos.SEMILLA))
    te = pipe.named_steps["pre"].named_transformers_["ent"].transform(ej[pipe.named_steps["pre"].transformers_[2][2]])
    te = pd.DataFrame(np.asarray(te), columns=serv["te_cols"], index=ej.index)
    salida = pd.concat([ej[datos.NUMERICAS + datos.CATEGORICAS].reset_index(drop=True), te.reset_index(drop=True),
                        ej[["y_tarde", "dias_retraso"]].reset_index(drop=True)], axis=1)
    d = datos.RAIZ / "data" / "prediccion"
    d.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(d / "ejemplos.parquet", index=False)

    filas = []
    for c in CAMPOS_NUM:
        q = ev[c].quantile([0.01, 0.5, 0.99])
        filas.append({"campo": c, "tipo": "numérica", "p01": q.iloc[0], "mediana": q.iloc[1], "p99": q.iloc[2]})
    tes = pd.DataFrame(np.asarray(pipe.named_steps["pre"].named_transformers_["ent"].transform(
        ev[pipe.named_steps["pre"].transformers_[2][2]].sample(50_000, random_state=datos.SEMILLA))), columns=serv["te_cols"])
    for c in CAMPOS_TE:
        q = tes[c].quantile([0.01, 0.5, 0.99])
        filas.append({"campo": c, "tipo": "tasa", "p01": q.iloc[0], "mediana": q.iloc[1], "p99": q.iloc[2]})
    for c in CAMPOS_CAT:
        f = ev[c].value_counts(normalize=True)
        filas.append({"campo": c, "tipo": "categórica",
                      "categorias": json.dumps({k: round(v, 4) for k, v in f[f >= 0.002].items()}, ensure_ascii=False)})
    pd.DataFrame(filas).to_csv(d / "rangos.csv", index=False)

    # privacidad: ni la orden ni los códigos de entidad salen en los ejemplos
    assert not ({"oc_id", "posicion"} | set(datos.ENTIDADES)) & set(salida.columns)
    print(f"servido: {a.modelo} ({len(serv['nombres'])} columnas), umbral {meta['umbral']:.3f}; "
          f"{len(salida)} ejemplos; {len(filas)} rangos")


if __name__ == "__main__":
    main()
