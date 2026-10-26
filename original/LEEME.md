# Modelo original (Entregable 3): BOREAL

Código de la propuesta, **BOREAL** (*Boosting Ordinal de Retraso con Efectos ALeatorios*): un *boosting* de árboles con
efectos aleatorios cruzados por entidad, sobre la reducción ordinal de los días de retraso y con los efectos ordenados
en el tiempo, más un modelo log-normal del plazo y 32 variables nuevas de plazo real. Se explica en el capítulo
«Modelo original» del libro.

**Solo numpy y scipy.** El modelo y sus árboles están escritos a mano; de scikit-learn se usan únicamente
`BaseEstimator` y `ClassifierMixin` (la prueba `test_solo_numpy_y_scipy` de `tests/test_boreal.py` lo verifica sobre
los imports).

| archivo | qué hace |
|---|---|
| `arboles_propios.py` | El aprendiz base: `Discretizador` (hasta 255 tramos por cuantiles; el tramo 0 es el vacío; categóricas por código) y `BoostingHistograma`, un *boosting* de árboles de histograma que crece por hoja, con regularización L1 y L2, vacíos probados a ambos lados, cortes categóricos por grupos ordenados por gradiente, restricción monótona, submuestreo de filas y columnas, objetivo binario o cuadrático, y valores SHAP exactos de árbol (TreeSHAP por camino). |
| `modelo_original_otif.py` | `BoostingMixtoOrdinal` (la pieza central: efectos cruzados de proveedor, material, creador de la solicitud y comprador, más la orden al estimar; pasos de Newton, varianzas por EM, corrección C; cortes 0, 1, 5 y 15 con monotonía en el corte), `efectos_ordenados` (cada semestre con los anteriores, en paralelo) y `Boreal`, la clase de scikit-learn (`fit`, `predict_proba`, `predict`, `decision_function`; valida entradas y se puede clonar). |
| `otif_comun.py` | Carga del dataset, codificación por partición, marca de filas con entidad nueva y bootstrap por orden. |
| `variables_nuevas_otif.py` | Las 32 variables nuevas: plazo real previo del proveedor, de la pareja y del material (mediana y percentil 80 de las últimas 20 entregas) y la holgura empírica. Solo usa lo conocido el día del pedido. |
| `evaluar_2x2_otif.py` | La comparación modelo × información en 2026 (XGBoost o BOREAL, con 136 variables o con las 32 nuevas). La fase `validacion` congela `config_2x2/<celda>.json` (Platt y umbral, ajustados en 2025) antes de la fase `prueba`; la fase `comparar` escribe `evaluar_2x2_otif_comparar.txt`. |
| `servir_original.py` | Toma las piezas entrenadas de la celda BOREAL con las nuevas, comprueba que reproducen la 2x2 y escribe `artifacts/original_servido.joblib` y `.json` y los ejemplos del tablero, sin códigos de entidad. |
| `publicos_otif.py` | BOREAL (sin plazo ni variables nuevas, que son de OTIF) contra el XGBoost de las 112 en siete conjuntos públicos de OpenML, con el mismo protocolo: Optuna con 20 evaluaciones, una medida en la prueba, DeLong. |
| `resumir_publicos_otif.py` | Junta los siete conjuntos: Holm y Wilcoxon. Salida en `publicos_otif_resumen.txt` y `.csv`. |
| `mejorar_otif.py`, `evaluar_k3_otif.py` | La exploración que llevó a BOREAL (variantes, ventana contra peso, efectos ordenados). Usan LightGBM como aprendiz y no forman parte del modelo final; quedan como rastro de cómo se eligió. |

## Cómo se corre

Las celdas de BOREAL se entrenaron en un servidor de 32 núcleos (Python 3.12, mismas versiones de numpy, scipy,
scikit-learn, pandas, XGBoost y Optuna) con los guiones de `srvgdai/`:

1. `python variables_nuevas_otif.py <ruta>/otif_depurado.csv` (deja `variables_nuevas.csv.gz` junto al dataset).
2. `bash srvgdai/subir_boreal.sh`: copia esta carpeta, las variables nuevas y los conjuntos públicos al servidor.
3. `bash srvgdai/boreal.sh validacion`: las cuatro celdas de 2x2 en paralelo; congela `config_2x2/`. Ese JSON se trae
   y se hace commit **antes** de seguir.
4. `bash srvgdai/boreal.sh prueba`: la medida en 2026 de cada celda.
5. `bash srvgdai/boreal.sh publicos`: los siete conjuntos públicos.
6. En local: `python evaluar_2x2_otif.py <ruta> --fase comparar`, `python resumir_publicos_otif.py` y
   `python servir_original.py <ruta>/otif_depurado.csv`.

Sin servidor también corre (`BOREAL_PROCESOS` fija los procesos; `BOREAL_ESCALA_ARBOLES=0.02` reduce los árboles para
una prueba de humo), pero una celda completa tarda unas dos horas con 8 procesos.

## Pruebas

`pytest tests/test_boreal.py` (desde la raíz): aprende un caso sintético con efectos conocidos, es reproducible con la
misma semilla, respeta la monotonía en el corte, se clona, da efecto 0 a un nivel nuevo, rechaza entradas inválidas,
funciona con un solo corte, la regresión cuadrática baja el error, solo importa numpy y scipy, y el TreeSHAP propio
coincide con el valor de Shapley por fuerza bruta.

## Lo que no viaja

Los intermedios (variables nuevas, predicciones por posición, `2x2/prop_nuevas_piezas.joblib`) se escriben junto al
dataset: traen códigos de orden o de entidad y no se publican. Lo publicado (`artifacts/original_servido.joblib`)
recibe las entidades ya convertidas en efectos.
