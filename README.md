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
| `pages/` | Una página del tablero por archivo: `contexto.py` (problema, datos y diccionario), `eda.py` (análisis exploratorio con filtros) y `modelos.py` (se llena en los entregables siguientes). |
| `src/` | Código del proyecto, compartido por el libro y el tablero. |
| `src/datos.py` | Ruta del dataset, semilla, partición cronológica, tipos de variable y preprocesamiento. |
| `src/data.py` | Carga y caché de lo que lee el tablero (`data/`); no importa scikit-learn. |
| `src/eda.py` | Cálculos del análisis exploratorio; cada función devuelve una tabla. |
| `src/tiempo.py` | Series de tiempo del EDA: volumen y tasa por periodo, huecos, STL, estacionariedad y autocorrelación. |
| `src/plots.py` | Una función por figura (Plotly) con la plantilla y la paleta del proyecto. |
| `src/preparar.py` | Escribe `data/` para el tablero a partir del dataset completo y de las tablas del libro. |
| `artifacts/` | Modelos y objetos serializados; se llena en los entregables siguientes. |
| `data/` | Lo liviano y anonimizado que usa el tablero: muestra de 30.000 posiciones, agregados, diccionario y resumen. |
| `resultados/` | Tablas que calcula el libro (`resultados/eda/`) y que después leen el tablero y las láminas. |
| `assets/` | Estilos del tablero. |
| `book/` | Libro en Jupyter Book. |
| `book/fuentes/` | Capítulos como scripts en formato percent (`# %%`); `compilar.py` los convierte en cuadernos. |
| `book/compilar.py` | Convierte `fuentes/*.py` en `.ipynb` y los ejecuta con el kernel del Python que lo corre. |
| `book/comun.py` | Utilidades compartidas por los cuadernos: carga de datos, figuras y tablas. |
| `requirements.txt` | Dependencias del tablero (versiones exactas). |
| `book/requirements.txt` | Dependencias del libro y de los cálculos (versiones exactas). |
| `render.yaml` | Blueprint de Render para desplegar el tablero. |
| `tests/` | Pruebas; se agregan en los entregables siguientes. |

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
