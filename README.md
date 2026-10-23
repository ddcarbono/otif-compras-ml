# Clasificador binario para la predicción del incumplimiento de órdenes de compra en la agroindustria de la palma de aceite

**Autores:** Daniel Carbonó y Luis Pérez.
Curso de Machine Learning, Maestría en Estadística Aplicada, Universidad del Norte, 2026.

- **Tablero:** https://otif-compras-ml.onrender.com (Render, plan gratuito: la primera visita tras un rato sin uso tarda unos 30 a 60 s).
- **Libro:** <https://ddcarbono.github.io/otif-compras-ml/>

## Qué es

El proyecto predice, el día en que la orden de compra se libera hacia el proveedor, si cada posición llegará más de un
día hábil tarde. Trabaja con 406.126 posiciones de 104.614 órdenes de 2021 a 2026, anonimizadas, con 136 variables
predictoras. La partición es cronológica: entrenamiento 2021-2024, validación 2025 y prueba 2026.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `app.py` | Tablero Dash multipágina (`use_pages=True`); lee solo `data/` y no entrena nada. Producción: `gunicorn app:server`. |
| `pages/` | Una página del tablero por archivo: `contexto.py` (problema, datos y diccionario), `eda.py` (análisis exploratorio con filtros) y `modelos.py` (predicción en vivo con XGBoost, las 112 combinaciones con sus pruebas, el mejor modelo en 2026 con SHAP y la 2 × 2, y el modelo base). |
| `src/` | Código del proyecto, compartido por el libro y el tablero. |
| `src/datos.py` | Ruta del dataset, semilla, partición cronológica, tipos de variable y preprocesamiento. |
| `src/data.py` | Carga y caché de lo que lee el tablero (`data/`); no importa scikit-learn. |
| `src/eda.py` | Cálculos del análisis exploratorio; cada función devuelve una tabla. |
| `src/tiempo.py` | Series de tiempo del EDA: volumen y tasa por periodo, huecos, STL, estacionariedad y autocorrelación. |
| `src/plots.py` | Una función por figura (Plotly) con la plantilla y la paleta del proyecto. |
| `src/modelo_base.py` | Regresión logística L2 frente al Dummy y a la regla del proveedor: C por validación temporal, umbral en 2025, prueba 2026 con bootstrap de órdenes. `python -m src.modelo_base` |
| `src/complejidad.py` | Tiempo de ajuste e inferencia de los 7 modelos contra el número de filas, con hiperparámetros fijos (exponente empírico). |
| `src/similitud.py` | Validación adversaria: qué tan distinta es la evaluación del entrenamiento en la partición cronológica y en una aleatoria. |
| `src/prediccion.py` | Prepara el modelo que sirve el tablero sin tablas de entidades (`artifacts/servido.joblib`), los ejemplos reales y los rangos de desarrollo. |
| `src/vista_prediccion.py` | La pestaña de predicción en vivo de la página Modelos. |
| `srvgdai/` | Guiones para correr los 112 en el servidor: subir el código, piloto, preparar las matrices y correr (30 trabajos a la vez). |
| `src/preparar.py` | Escribe `data/` para el tablero a partir del dataset completo y de las tablas del libro. |
| `artifacts/` | El modelo que sirve el tablero (`servido.joblib` y `servido.json`: transformaciones, modelo, umbral, versiones y métricas) y la ficha del modelo base. El *pipeline* completo no se publica: sus tablas de entidades llevan los códigos. |
| `data/` | Lo liviano y anonimizado que usa el tablero: muestra de 30.000 posiciones, agregados, diccionario y resumen. |
| `resultados/` | Tablas que calculan el libro (`resultados/eda/`) y los scripts de modelos (`resultados/modelo_base/`, `resultados/experimentos/` con un JSON por trabajo y `resultados/tabla_maestra.csv`) y que después leen el tablero y las láminas. |
| `presentacion/` | Diapositivas de la sustentación (PPTX y PDF con la plantilla Uninorte), notas del expositor y guía de estudio; ver `presentacion/LEEME.md`. |
| `assets/` | Estilos del tablero. |
| `book/` | Libro en Jupyter Book. |
| `book/fuentes/` | Capítulos como scripts en formato percent (`# %%`); `compilar.py` los convierte en cuadernos. |
| `book/compilar.py` | Convierte `fuentes/*.py` en `.ipynb` y los ejecuta con el kernel del Python que lo corre. |
| `book/comun.py` | Utilidades compartidas por los cuadernos: carga de datos, figuras y tablas. |
| `requirements.txt` | Dependencias del tablero (versiones exactas). |
| `book/requirements.txt` | Dependencias del libro y de los cálculos (versiones exactas). |
| `render.yaml` | Blueprint de Render para desplegar el tablero. |
| `tests/` | Pruebas con pytest: preprocesamiento y codificación por objetivo, series de tiempo, modelo servido, arranque del tablero y que `data/` no publique identificadores. Usan datos sintéticos y lo que trae el repositorio, no el dataset completo. |
| `requirements-dev.txt` | Dependencias de las pruebas (el tablero más pytest). |
| `.github/workflows/ci.yml` | CI: corre las pruebas en cada push a `main`, con Python 3.11.9 como Render. |

## Código de los experimentos y de la evaluación

Lo que corre una vez (en el servidor o en local) y deja sus tablas en `resultados/`; el libro solo las lee.

| Ruta | Contenido |
|---|---|
| `src/validacion.py` | Validación anidada por semestre (8 bloques externos) y caché de matrices preprocesadas, SMOTE y ADASYN por ventana. |
| `src/optimizadores.py` | Grid, Random, Optuna (TPE) y DEAP (genético) con el mismo presupuesto de evaluaciones. |
| `src/experimentos.py` | Las 112 combinaciones (7 modelos × 4 balanceos × 4 optimizadores) en los 8 bloques; reanudable. |
| `src/comparar.py` | Resume los 896 trabajos: tabla de líderes, curvas de los optimizadores, Friedman, Nemenyi y Wilcoxon con Holm. `python -m src.comparar` |
| `src/puntajes.py` | Reajusta la mejor combinación de cada modelo en cada bloque con los hiperparámetros ya elegidos y guarda el puntaje por posición (`resultados/puntajes/`); comprueba que el AUC coincide con el del trabajo. Corre en srvgdai (`srvgdai/puntajes.sh`). |
| `src/pruebas.py` | DeLong para dos AUC pareados (validado contra `pROC::roc.test` con `src/validar_delong.R`), delta de Cliff e intervalo BCa con remuestreo de órdenes por bloque. |
| `src/comparar_pruebas.py` | DeLong de XGBoost contra cada rival en los 8 bloques (Holm), AUC medio y diferencias con BCa 95 %, delta de Cliff, y la validación en un bloque real contra pROC. `python -m src.comparar_pruebas` |
| `src/estabilidad.py` | La mejor combinación de 4 modelos con 10 semillas por bloque y la curva de convergencia de XGBoost. Corre en srvgdai (`srvgdai/estabilidad.sh`). |
| `src/diagnostico_validacion.py` | Calibración (y Platt de un bloque al siguiente) y errores de XGBoost en los 8 bloques: por línea, proveedor nuevo y perfil de los falsos negativos. `python -m src.diagnostico_validacion` |
| `src/calibrado.py` | XGBoost con el escalado de Platt congelado en 2025: lo que sirve el tablero. |
| `src/modelo_final.py` | Entrena ese modelo con 2021-2025 (`artifacts/xgboost.joblib`), comprueba que reproduce las probabilidades de la comparación 2 × 2 y calcula SHAP en 2026. `python -m src.modelo_final <predicciones_prueba_2x2.csv>` |
| `src/prueba_2026.py` | XGBoost, la logística y la propuesta en la prueba 2026 con BCa por órdenes y DeLong; la tabla de brecha y la 2 × 2. `python -m src.prueba_2026 <predicciones_prueba_2x2.csv>` |
| `src/xgb_vs_logistica.py` | Por qué XGBoost y no la logística: presupuesto de alertas, forma del efecto del plazo (SHAP) y costo. `python -m src.xgb_vs_logistica <predicciones_prueba_2x2.csv>` |
| `original/` | El modelo original del Entregable 3 (K3 + log-normal + variables nuevas); ver `original/LEEME.md`. |

## Datos

El dataset completo (342 MB) **no está en el repositorio**: se entrega aparte al docente. Se lee de la variable de
entorno `OTIF_CSV`. La carpeta `data/` solo trae lo liviano y anonimizado que usa el tablero: una muestra de 30.000
posiciones y los agregados.

## Instalación y uso

### Tablero

```bash
pip install -r requirements.txt
python app.py
```

Abre <http://127.0.0.1:8050>.

### Libro

Requiere Python 3.9 con `ipykernel`; `book/compilar.py` ejecuta los cuadernos con el kernel del mismo Python.

```bat
pip install -r book/requirements.txt
set OTIF_CSV=C:\ruta\al\otif_depurado.csv
python book/compilar.py
jupyter-book build book/
```

### Pruebas

```bash
pip install -r requirements-dev.txt
pytest -q
```

### Datos del tablero

Desde la raíz del repositorio, con `OTIF_CSV` definida y después de compilar el capítulo 2 del libro (el tablero toma de
`resultados/eda/` las tablas que el libro ya calculó):

```bat
python -m src.preparar
```

## Despliegue

**Tablero en Render.** New → Blueprint → elegir este repositorio → Apply. Render toma `render.yaml` (`gunicorn app:server`,
Python 3.11.9). El plan gratuito se duerme tras un rato sin tráfico: conviene abrir la URL unos minutos antes de una
presentación.

**Libro en GitHub Pages.**

```bash
ghp-import -n -p -f book/_build/html
```

Después, en Settings → Pages del repositorio, activar Pages sobre la rama `gh-pages`.

## Reproducibilidad

La semilla global es 2026 (`SEMILLA` en `src/datos.py`). Los dos `requirements.txt` fijan versiones exactas.

## Licencia de los datos

Los datos son de uso académico autorizado y no son redistribuibles.
