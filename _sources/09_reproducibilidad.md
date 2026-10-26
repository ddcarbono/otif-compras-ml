# Reproducibilidad

Todo número de este libro sale de un archivo del repositorio que se puede volver a correr. Este capítulo dice con qué
entorno, en qué orden y qué no viaja.

## Entorno

| para qué | Python | dependencias |
|---|---|---|
| libro y cálculos | 3.9 | `book/requirements.txt` (versiones exactas) |
| tablero y pruebas | 3.11.9 (el de Render) | `requirements.txt` y `requirements-dev.txt` |
| las 112 combinaciones, BOREAL y los conjuntos públicos | 3.12, en un servidor de 32 núcleos | las mismas versiones de numpy, scipy, scikit-learn, pandas, XGBoost y Optuna; guiones en `srvgdai/` |

La semilla global es 2026 (`SEMILLA` en `src/datos.py`); todos los guiones la usan, también el boosting propio de
BOREAL (`original/arboles_propios.py`), cuyo resultado es idéntico con la misma semilla (prueba
`test_reproducible_con_la_misma_semilla`).

## Orden de ejecución

| paso | comando | deja |
|---|---|---|
| 1. Datos del tablero y EDA | `python book/compilar.py 02a_datos 02b_eda 02c_tiempo_y_fuga`, `python -m src.preparar` | `resultados/eda/`, `data/` |
| 2. Modelo base | `python -m src.modelo_base` | `resultados/modelo_base/`, `artifacts/logistica.*` |
| 3. Las 112 combinaciones | `bash srvgdai/subir.sh`, `srvgdai/preparar.sh`, `srvgdai/correr.sh`; luego `python -m src.experimentos tabla` y `python -m src.comparar` | `resultados/experimentos/` (un JSON por trabajo) y `resultados/tabla_maestra.csv` |
| 4. Pruebas sobre las predicciones | `srvgdai/puntajes.sh`, `python -m src.comparar_pruebas`, `srvgdai/estabilidad.sh`, `python -m src.diagnostico_validacion` | `resultados/pruebas/`, `resultados/estabilidad/`, `resultados/diagnostico/` |
| 5. Modelo original | `bash srvgdai/subir_boreal.sh`; `srvgdai/boreal.sh validacion`, commit de `original/config_2x2/`, `srvgdai/boreal.sh prueba`; `python original/evaluar_2x2_otif.py <datos> --fase comparar` | `original/config_2x2/`, las predicciones de 2026 y `original/evaluar_2x2_otif_comparar.txt` |
| 6. Prueba 2026 y modelo servido | `python -m src.modelo_final <pred>`, `python -m src.prueba_2026 <pred>`, `python -m src.xgb_vs_logistica <pred>`, `python -m src.lime_xgb`, `python original/servir_original.py <datos>`, `python -m src.prediccion --modelo xgboost`, `python -m src.tabla_completa` | `resultados/prueba_2026/`, `artifacts/`, `resultados/tabla_maestra_completa.csv` |
| 7. Conjuntos públicos | `srvgdai/boreal.sh publicos`, `python original/resumir_publicos_otif.py` | `original/publicos_otif_resumen.*` |
| 8. Libro y pruebas | `python book/compilar.py`, `jupyter-book build book/`, `pytest -q` | `book/_build/html` |

El paso 5 fija Platt y el umbral en la validación 2025 y los guarda en `original/config_2x2/` **antes** de medir 2026:
ese commit es la constancia de que la prueba no eligió nada.

## Qué no viaja

- **El dataset completo** (`otif_depurado.csv`, 342 MB) es de uso académico autorizado y no se redistribuye; se entrega
  aparte al docente. Los guiones lo leen de la variable de entorno `OTIF_CSV`.
- **Las tablas de entidades.** El *pipeline* entrenado de XGBoost (`artifacts/xgboost.joblib`) y las piezas de BOREAL
  guardan la tasa o el efecto de cada código de proveedor, material o comprador; no se publican. Lo publicado
  (`artifacts/servido.joblib`, `artifacts/original_servido.joblib`) recibe esas entidades ya convertidas en números.
- **Los conjuntos públicos** se descargan de OpenML con sus identificadores (capítulo {doc}`06_validacion`).

## Tabla maestra

`resultados/tabla_maestra.csv` tiene una fila por cada una de las 112 combinaciones (modelo, balanceo, optimizador)
con las métricas del bucle externo (media y desviación de los ocho bloques), el tiempo de cómputo, la memoria, las
evaluaciones del optimizador, los hiperparámetros finales (los elegidos en el bloque más reciente, 2025-S2) y la
semilla. `resultados/tabla_maestra_completa.csv` (`python -m src.tabla_completa`) le agrega la fila del modelo
original, medida en la validación 2025 y no en los ocho bloques: la columna `medida` lo dice. El detalle de cada uno de
los 896 trabajos está en `resultados/experimentos/trabajos/`.

## Bibliografía

Las referencias del libro están en `book/references.bib` (BibTeX) y se citan con `{cite}`; la lista completa se arma en
la introducción.
