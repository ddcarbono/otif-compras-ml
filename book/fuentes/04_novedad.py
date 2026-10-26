# %% [markdown]
# # Literatura y novedad del modelo original
#
# El capítulo {doc}`literatura` situó el problema: qué se ha publicado sobre predecir el retraso de un proveedor. Este
# responde otra pregunta, la primera de las tres que debe contestar el modelo original (R1): **¿alguien propuso ya un
# método como BOREAL, para este problema o para otro?** Para contestarla se buscó en tres bases, con cadenas en inglés y
# en español, se tamizaron los resultados, se leyeron completos los artículos más cercanos y se armó una matriz que
# compara, rasgo por rasgo, lo que cada método hace con lo que hace BOREAL.

# %% tags=["remove-cell"]
import pandas as pd

from comun import datos, tabla

C = pd.read_csv(datos.RAIZ / "resultados" / "literatura" / "cadenas.csv")

# %% [markdown]
# ## Cómo se buscó
#
# - **Bases.** OpenAlex por su API (los 50 primeros resultados por relevancia de cada cadena); Scopus y Web of Science
#   por el acceso de la biblioteca de la Universidad del Norte (los 200 primeros por relevancia, exportados con resumen).
#   Las cadenas en inglés se corrieron en las tres bases el 22 de octubre de 2026; las cinco en español, en OpenAlex el
#   25 de octubre de 2026 (Scopus y Web of Science indexan sobre todo títulos y resúmenes en inglés).
# - **Dos grupos de cadenas.** Las de *contexto* (D01 a D06, E01, E02 y E05) buscan quién predijo el retraso de compras o
#   de entregas de proveedores y con qué. Las de *métodos* (C01 a C20, E03 y E04) buscan, sin restringir el dominio, los
#   métodos que atacan cada rasgo del problema: entidades de muchos niveles, efectos aleatorios con árboles, categorías
#   nuevas, deriva, la variable continua detrás de la etiqueta, calibración, monotonía, observaciones agrupadas.
# - **Unión y tamizaje.** Los registros se unieron por DOI y por título normalizado (`unir_bases.py`). El tamizaje leyó
#   título y resumen de cada artículo, por lotes, y marcó los candidatos; los seis más cercanos se leyeron completos y
#   tienen ficha técnica (`revision_literatura/fichas_otif/`). Los guiones y las cadenas están en
#   `revision_literatura/` del repositorio de estudio; la tabla de abajo sale de `cadenas_otif.py`.

# %%
n = C.groupby("idioma")[["openalex_traidos", "scopus_traidos", "wos_traidos"]].sum()
n.columns = ["OpenAlex", "Scopus", "Web of Science"]
n["registros"] = n.sum(axis=1)
n["artículos sin duplicados"] = n.index.map({"inglés": 7299, "español": 32})   # union.csv y buscar_openalex_otif_es.txt
tabla(n.reset_index().rename(columns={"idioma": "cadenas en"}), enteros=list(n.columns))

# %% [markdown]
# Las 31 cadenas, con su fecha y lo que trajo cada base («total» es cuántos resultados tenía OpenAlex para esa cadena;
# de ahí se trajeron los 50 primeros):

# %%
t = C.rename(columns={"openalex_total": "OpenAlex total", "openalex_traidos": "OpenAlex", "scopus_traidos": "Scopus",
                      "wos_traidos": "WoS"})
tabla(t[["id", "grupo", "tema", "cadena", "fecha", "OpenAlex total", "OpenAlex", "Scopus", "WoS"]],
      enteros=["OpenAlex total", "OpenAlex", "Scopus", "WoS"])

# %% [markdown]
# ## Qué se encontró
#
# **En el contexto**, unos 90 estudios de retraso de compras, entregas de proveedores, plazos y ERP (los comparables
# están en {doc}`literatura`). Dominan el bosque aleatorio (unos 40), XGBoost (unos 30), la logística (unos 20), SVM,
# vecinos más cercanos y redes; LightGBM y CatBoost aparecen en 7 y 5. Ninguno usa efectos aleatorios por proveedor o
# material, umbrales ordinales sobre los días de retraso, efectos estimados en orden temporal ni validación agrupada por
# orden de compra con datos de empresa. Las 32 referencias de las cadenas en español tratan de gestión de inventarios,
# selección de proveedores por criterios múltiples (AHP, TOPSIS) o auditoría, y las de clasificación ordinal son
# revisiones generales: ninguna predice el retraso de una orden.
#
# **En los métodos**, la familia más cercana a BOREAL es la de efectos aleatorios dentro de un *boosting* o un árbol; apareció en seis de
# los diez lotes de tamizaje. La reducción ordinal y el ordenamiento de los estadísticos por entidad vienen de otras
# dos líneas.
#
# ## Matriz de novedad
#
# Los seis rasgos de BOREAL ({doc}`05_modelo_original`), contra los métodos publicados más cercanos. «Sí» quiere decir
# que el artículo lo propone o lo usa; «no», que no; las celdas de los artículos leídos solo por su resumen lo dicen.
#
# | método | (1) respuesta binaria | (2) parte fija con árboles | (3) efectos aleatorios cruzados de varias entidades | (4) reducción ordinal de los días de retraso | (5) efectos estimados en orden temporal | (6) aplicado al retraso de compras | leído |
# |---|---|---|---|---|---|---|---|
# | Boosting para modelos mixtos lineales {cite}`griesbach2021` | no (gaussiana) | no (aprendices lineales) | no (un factor) | no | no | no | completo |
# | Boosting para modelos mixtos aditivos {cite}`knieper2025` | sí | no (lineales y *splines*) | no (un factor) | no | no | no | completo |
# | GPBoost {cite}`sigrist2022` | sí | sí | sí | no | no | no | documentación |
# | MERF {cite}`hajjem2014` | no (continua) | sí (bosque aleatorio) | no (un factor) | no | no | no | resumen |
# | Árboles GLMM {cite}`fokkema2018` | sí | un solo árbol | admite los de `lme4` | no | no | no | resumen |
# | M-CART {cite}`lin2019` | sí | un solo árbol | no (dos niveles anidados) | no | no | no | resumen |
# | GLMM–XGBoost {cite}`cho2026` | sí | sí | sí (persona × ítem) | no | no | no | resumen |
# | Árboles aditivos con variable latente (IRGB) {cite}`wang2025` | sí | sí | no (latente por observación, no por entidad) | no | no | no | completo |
# | Regresión ordinal por clasificación binaria extendida {cite}`li2006` | sí | árboles C4.5 y tocones | no | sí | no | no | completo |
# | CatBoost {cite}`prokhorenkova2018` | sí | sí | no (codificación por objetivo, sin varianza) | no | sí (estadísticos ordenados) | no | completo |
# | Estudios de retraso de compras ({doc}`literatura`) | sí | sí (bosque, XGBoost) | no | no | no | sí | completo o resumen |
# | **BOREAL** | **sí** | **sí** | **sí (proveedor, material, solicitante y comprador, más la orden al estimar)** | **sí, con monotonía en el corte** | **sí** | **sí** | — |
#
# ## Declaración de novedad
#
# Cada pieza de BOREAL tiene un antecedente publicado, y se cita: los efectos aleatorios dentro del *boosting*
# (Griesbach, Knieper), la reducción ordinal (Li y Lin) y el ordenamiento temporal de los estadísticos (CatBoost). Ningún
# artículo de los 7.331 revisados **combina** las seis cosas, y ninguno de los métodos con efectos aleatorios se ha
# aplicado al retraso de compras. Lo propio es:
#
# 1. pasar el *boosting* mixto de Knieper et al. (respuesta binaria, un factor, aprendices lineales) a **efectos
#    cruzados de cuatro entidades con árboles de histograma**, a 400.000 posiciones;
# 2. aplicar los efectos **sobre la reducción ordinal** de los días de retraso, con un mismo efecto en todos los cortes y
#    una restricción monótona que impide que los umbrales se crucen;
# 3. **estimar los efectos en orden temporal** (cada semestre solo con los anteriores), la idea de CatBoost llevada de la
#    codificación por objetivo a los efectos aleatorios, para que entrenar y predecir una entidad nueva se parezcan;
# 4. un modelo log-normal del plazo y 32 variables de plazo real como información que el clasificador no tenía.
#
# **Lo que no se reclama.** GPBoost y el GLMM–XGBoost de Cho ya combinan árboles con efectos cruzados en respuesta
# binaria; la diferencia de BOREAL con ellos está en los rasgos (4), (5) y (6) del cuadro, no en el (3). GPBoost no se
# comparó con BOREAL: el modelo original se implementa solo con numpy y scipy, sin paquetes de modelos. Una comparación con
# GPBoost en los mismos datos queda como trabajo pendiente ({doc}`08_conclusiones`).
#
# ## Métodos relacionados en una frase
#
# | método | qué hace | por qué no basta aquí |
# |---|---|---|
# | GPBoost {cite}`sigrist2022` | *boosting* de árboles con efectos aleatorios agrupados o procesos gaussianos, estimados juntos por máxima verosimilitud | no usa los días de retraso ni ordena los efectos en el tiempo; es un paquete |
# | MERF {cite}`hajjem2014` | alterna un bosque aleatorio para la parte fija con un modelo lineal mixto para los efectos | respuesta continua y un factor |
# | Árboles GLMM {cite}`fokkema2018` | partición recursiva basada en modelos con efectos aleatorios de un GLMM | un árbol, pensado para hallar subgrupos, no para predecir a 400.000 filas |
# | M-CART {cite}`lin2019` | CART para datos de dos niveles con respuesta binaria | anidado, no cruzado; un árbol |
# | GLMM–XGBoost {cite}`cho2026` | alterna XGBoost y un GLMM para respuestas de ítems | otro dominio; sin reducción ordinal ni orden temporal |
# | Boosting para modelos mixtos {cite}`griesbach2021,knieper2025` | *boosting* por componentes con efectos aleatorios, corrección de los efectos y parada por validación cruzada de grupos | un factor y aprendices lineales; de aquí salen el esquema por bloques y la corrección C |
# | Regresión ordinal por clasificación extendida {cite}`li2006` | copia cada caso una vez por umbral y aprende todos los umbrales con un clasificador binario | sin entidades; de aquí sale la reducción ordinal |
# | CatBoost {cite}`prokhorenkova2018` | codifica cada categoría con los casos anteriores en una permutación, para no usar la propia etiqueta | sin encogimiento con varianza estimada; de aquí sale el ordenamiento |
# | Ponderación por antigüedad {cite}`klinkenberg2004` | pesa los ejemplos con un decaimiento exponencial por edad | se probó como pieza y su propia fuente dice que la ventana rinde más; no quedó en BOREAL |
