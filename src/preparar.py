"""Escribe ``data/`` para el tablero: lo único que lee la aplicación. Se corre después de compilar el capítulo 2, porque
toma de ``resultados/eda/`` las tablas que el libro ya calculó (la cifra del tablero es la del libro).

- ``data/muestra_eda.parquet``: 30.000 posiciones del entrenamiento, estratificadas por año y clase, con la fecha, los
  grupos de los filtros, el objetivo y las doce numéricas que más separan. Anonimizada como el dataset.
- ``data/agregados/serie_mensual.csv``: volumen y tasa por mes, línea y clase, con todo el entrenamiento (las series
  del tablero se filtran y se suman desde aquí, no desde la muestra).
- ``data/agregados/*.csv``: copias de las tablas del libro que el tablero muestra sin recalcular.
- ``data/resumen.json``: las cifras de la portada.
- ``data/modelos/*.csv``: las tablas del modelo base (``resultados/modelo_base``), si ya se corrió ``src.modelo_base``, y las de los 112 (``resultados/experimentos``, de ``src.comparar``).

Uso: ``python -m src.preparar`` desde la raíz del repositorio, con ``OTIF_CSV`` apuntando al dataset completo.
"""
import json
import shutil

import pandas as pd

from . import datos, eda

N_MUESTRA = 30_000
GRUPOS = ["linea", "clase_compras", "sociedad"]
TABLAS = ["tamano", "faltantes", "asociacion", "contrastes", "tasa_linea", "tasa_clase", "deriva", "entidades_nuevas",
          "univariado_numericas", "top_numericas", "estacionariedad"]


def main():
    res = datos.RAIZ / "resultados" / "eda"
    faltan = [t for t in TABLAS if not (res / f"{t}.csv").exists()]
    if faltan:
        raise SystemExit(f"faltan tablas del libro en {res}: {faltan}. Compilar antes el capítulo 2.")
    df = datos.cargar()
    ent = datos.particion(df, "entrenamiento")
    top = pd.read_csv(res / "top_numericas.csv").variable.tolist()

    sal = datos.RAIZ / "data"
    (sal / "agregados").mkdir(parents=True, exist_ok=True)
    estratos = ent.fecha_pedido.dt.year.astype(str) + "_" + ent.y_tarde.astype(str)
    frac = N_MUESTRA / len(ent)
    muestra = ent.groupby(estratos, group_keys=False).apply(lambda d: d.sample(frac=frac, random_state=datos.SEMILLA))
    cols = ["fecha_pedido", "y_tarde", "dias_retraso"] + GRUPOS + top
    muestra[cols].reset_index(drop=True).to_parquet(sal / "muestra_eda.parquet", index=False)

    eda.serie(ent, por=["linea", "clase_compras"]).to_csv(sal / "agregados" / "serie_mensual.csv", index=False)
    for t in TABLAS:
        shutil.copy(res / f"{t}.csv", sal / "agregados" / f"{t}.csv")

    tam = pd.read_csv(res / "tamano.csv").set_index("partición")
    resumen = {"filas": int(tam.loc["total", "filas"]), "ordenes": int(tam.loc["total", "órdenes"]),
               "proveedores": int(tam.loc["total", "proveedores"]), "materiales": int(tam.loc["total", "materiales"]),
               "predictoras": len(datos.PREDICTORAS), "tasa_total": float(tam.loc["total", "tasa de tarde"]),
               "filas_entrenamiento": int(tam.loc["entrenamiento", "filas"]),
               "tasa_entrenamiento": float(tam.loc["entrenamiento", "tasa de tarde"]),
               "periodo": [str(df.fecha_pedido.min().date()), str(df.fecha_pedido.max().date())],
               "particion": datos.PARTICIONES, "muestra": len(muestra), "semilla": datos.SEMILLA}
    (sal / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")

    mb = datos.RAIZ / "resultados" / "modelo_base"
    if (mb / "metricas.csv").exists():
        (sal / "modelos").mkdir(exist_ok=True)
        for f in ["metricas", "curvas", "calibracion", "confusion", "curva_aprendizaje", "cv_C"]:
            shutil.copy(mb / f"{f}.csv", sal / "modelos" / f"{f}.csv")
        pd.read_csv(mb / "coeficientes.csv").head(40).to_csv(sal / "modelos" / "coeficientes.csv", index=False)
        shutil.copy(mb / "resumen.json", sal / "modelos" / "resumen.json")

    ex = datos.RAIZ / "resultados" / "experimentos"
    if (ex / "lideres.csv").exists():
        (sal / "modelos").mkdir(exist_ok=True)
        for f in ["lideres", "rangos", "friedman", "wilcoxon", "auc_por_bloque_modelos", "curvas_optimizador"]:
            shutil.copy(ex / f"{f}.csv", sal / "modelos" / f"{f}.csv")

    # privacidad: la muestra no lleva identificadores ni entidades de personas o terceros
    prohibidas = set(datos.ENTIDADES) - {"sociedad"} | {"oc_id", "posicion"}
    assert not prohibidas & set(cols), prohibidas & set(cols)
    print(f"muestra: {len(muestra):,} filas, {len(cols)} columnas; agregados: {len(TABLAS) + 1}; resumen.json")


if __name__ == "__main__":
    main()
