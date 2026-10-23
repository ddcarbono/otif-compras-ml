# Modelo original (Entregable 3)

Código de la propuesta: **K3 + modelo log-normal del plazo + variables nuevas**. Se explica en el capítulo
«Modelo original» del libro. Los guiones corren con `ml_venv` desde esta carpeta y reciben la ruta del dataset
anonimizado (`otif_depurado.csv`). Los intermedios (variables nuevas, predicciones) se escriben junto al dataset y no
viajan con el repositorio.

| archivo | qué hace |
|---|---|
| `modelo_original_otif.py` | La clase `BoostingMixtoOrdinal`: boosting de árboles (LightGBM) con efectos aleatorios cruzados de proveedor, material, creador de la solped y comprador. Trabaja sobre la reducción ordinal de los días de retraso (cortes 0, 1, 5 y 15) con restricción monótona en el corte y corrección C. |
| `otif_comun.py` | Carga del dataset, codificación por partición, marca de filas con entidad nueva y bootstrap por OC. |
| `variables_nuevas_otif.py` | Las 32 variables nuevas: plazo real previo del proveedor, de la pareja y del material (mediana y percentil 80 de las últimas 20 entregas) y la holgura empírica. Solo usa lo conocido el día del pedido. |
| `mejorar_otif.py` | Validación por cuatro semestres (2024-S1 a 2025-S2) con la regla fijada antes de correr: intervalo sobre 0 y ganar en 3 de 4 semestres. Salida en `mejorar_otif.txt`. |
| `evaluar_k3_otif.py` | K3: los efectos de cada semestre estimados solo con los anteriores, al entrenar. Su fase de validación congela `k3_congelado.json`. |
| `evaluar_2x2_otif.py` | La comparación modelo × información en 2026. Su fase de validación congela `config_2x2/<celda>.json` (Platt y umbral, ajustados en 2025) antes de la prueba. Salida en `evaluar_2x2_otif_comparar.txt`. |
| `publicos_otif.py` | K3 contra el XGBoost de las 112 en conjuntos públicos de OpenML, con el mismo protocolo (Optuna con 20 evaluaciones, una medida en la prueba, DeLong). Recibe la carpeta de los ARFF descargados. Salidas `publicos_otif.*` (primera ronda) y `publicos_otif_ronda2.*`. |
| `resumir_publicos_otif.py` | Junta las dos rondas: Holm sobre los siete conjuntos y Wilcoxon. Salida en `publicos_otif_resumen.txt` y `.csv`. |

Orden de ejecución:

1. `python variables_nuevas_otif.py <ruta>/otif_depurado.csv`
2. `python mejorar_otif.py <ruta>/otif_depurado.csv`
3. `python evaluar_k3_otif.py <ruta>/otif_depurado.csv --fase validacion`
4. `python evaluar_2x2_otif.py <ruta>/otif_depurado.csv --fase validacion --celda <c>`, para cada celda.
5. La fase `prueba` de cada celda.

Lo que esta carpeta no reproduce sola:

- La fase `comparar` del 2x2 y la fase `prueba` de K3 leen también las predicciones de un LightGBM que solo sirve de
  referencia y no viene en esta carpeta. La comparación del libro la hace `python -m src.prueba_2026` desde la raíz,
  con intervalos BCa por órdenes.
- `evaluar_2x2_otif.py` usa el preprocesador y el JSON del trabajo de XGBoost de `src/` y `resultados/` de este
  repositorio (`REPO` apunta a la carpeta de arriba).
