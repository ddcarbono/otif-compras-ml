# El tablero y su despliegue

**Dirección pública:** <https://otif-compras-ml.onrender.com>. El repositorio es
<https://github.com/ddcarbono/otif-compras-ml>.

El tablero muestra lo que este libro calcula y predice en vivo con los modelos ya entrenados. No entrena nada: lee lo
que dejan los guiones del repositorio en `data/` (tablas livianas y anonimizadas) y en `artifacts/` (los modelos
serializados). Si el plan gratuito de Render está dormido, la primera visita tarda entre 30 y 60 segundos.

## Qué tiene cada página

| página | contenido |
|---|---|
| **Contexto** | el problema y por qué importa, la pregunta de investigación y los objetivos, los datos con el diccionario filtrable, el diagrama de la metodología, las limitaciones, la ética y los autores |
| **EDA** | filtros globales (línea de negocio, clase de compra, periodo) que mueven todas las figuras; la variable objetivo; distribuciones y atípicos con su prueba de normalidad; un explorador bivariado con selectores; la tasa de tarde por tramos; la asociación de cada variable con la clase; componentes principales; y lo que cambia entre particiones (entidades nuevas y validación adversaria). Cada figura trae una frase de lectura |
| **Modelos** | cinco pestañas, en el orden del libro (abajo) |

Las cinco pestañas de Modelos:

1. **Línea base (logística):** métricas en 2026, discriminación, umbral y calibración, qué pesa en el modelo, el
   hiperparámetro y la curva de aprendizaje.
2. **Las 112 combinaciones:** la tabla de líderes (`dash_table`), el efecto del balanceo y del optimizador, Friedman,
   Nemenyi con su diagrama de diferencia crítica, Wilcoxon con Holm, DeLong, intervalos BCa y delta de Cliff, y la
   calibración y la convergencia de XGBoost en la validación.
3. **Modelo óptimo:** XGBoost en la prueba 2026, sus curvas, SHAP global, por qué se prefiere a la logística y cuántas
   órdenes seguir con un presupuesto de alertas.
4. **Modelo original:** BOREAL, su formulación y su flujo, la comparación 2 × 2 y los conjuntos públicos.
5. **Predicción:** un formulario con los campos que más pesan, validados contra el rango de desarrollo de cada uno
   (percentiles 1 y 99 de 2021-2025). Carga una posición real de la prueba 2026 y devuelve la probabilidad calibrada,
   la decisión al umbral fijado en 2025 y la contribución de cada variable (SHAP exacto de árbol). Sirve BOREAL y, para
   comparar, XGBoost. `…/modelos?pestana=t-prediccion` abre directo en esta pestaña.

## Cómo está hecho

- **Dash multipágina** (`use_pages=True`): `app.py` arma la aplicación y cada página vive en `pages/`. Las figuras salen
  de `src/plots.py`, las mismas funciones que usa este libro, así que una cifra del tablero y la del libro vienen del
  mismo cálculo.
- **Un solo productor de datos.** `src/preparar.py` escribe `data/` a partir del dataset completo y de las tablas de
  `resultados/`. El tablero solo lee: una sola carga al arrancar, en caché, y los callbacks filtran en memoria.
- **Modelos serializados.** `artifacts/servido.joblib` (XGBoost con Platt) y `artifacts/original_servido.joblib` (BOREAL)
  llevan cada uno su ficha JSON con las versiones, la semilla, los hiperparámetros, Platt, el umbral y las métricas.
  Lo publicado no trae códigos de entidad: las entidades llegan como tasas o efectos ya calculados, y las tablas por
  código no salen de la máquina donde se entrenó. Las pruebas comprueban que el modelo servido reproduce las
  probabilidades de la evaluación y que un ida y vuelta por `joblib` no cambia ninguna predicción.
- **Diseño:** `dash-bootstrap-components` con el tema FLATLY, una paleta de cinco colores, rejilla que se acomoda a
  escritorio y tableta, `dcc.Store` para el estado de la predicción y `dbc.Alert` para los mensajes. No hay rutas ni
  credenciales en el código.

## Despliegue

Render construye el servicio con `render.yaml`: Python 3.11.9, `pip install -r requirements.txt` (versiones exactas) y
`gunicorn app:server`. Cada cambio en la rama `main` del repositorio redespliega el tablero. Antes, GitHub Actions corre
las pruebas con el mismo Python (`.github/workflows/ci.yml`): preprocesamiento y codificación fuera de pliegue, series
de tiempo, los dos modelos servidos, el arranque de las tres páginas, que `data/` no publique identificadores, y las
pruebas del modelo original (capítulo {doc}`05_modelo_original`).

Para la sustentación hay dos respaldos: el tablero corre igual en local (`python app.py`, en <http://127.0.0.1:8050>) y
`presentacion/img/` guarda capturas de las tres páginas.
