# %% [markdown]
# # Tiempo, fuga y preprocesamiento

# %% tags=["remove-cell"]
import numpy as np
import pandas as pd

from comun import datos, guardar, mostrar, tabla
from src import eda, plots

df = datos.cargar()
ent = datos.particion(df, "entrenamiento")
N = datos.NOMBRE

# %% [markdown]
# ## Componente temporal
#
# ### La variable de tiempo
#
# La fecha de referencia es la de creación de la orden (`fecha_pedido`), que define la partición. Es una fecha sin
# hora ni zona horaria, con órdenes en días irregulares.

# %%
tabla(eda.huecos(ent))

# %% [markdown]
# El entrenamiento cubre del 4 de enero de 2021 al 31 de diciembre de 2024 sin semanas vacías. Los días sin órdenes son
# fines de semana y festivos, salvo 15 días hábiles (cierres de fin de año y puentes). No hay fechas duplicadas que
# corregir: varias órdenes por día es lo normal. La frecuencia natural para la serie es el mes (48 puntos).
#
# ### Serie, tendencia y estacionalidad

# %%
mensual = guardar(eda.serie(ent), "serie_mensual")
mostrar(plots.serie_tasa(mensual))
mostrar(plots.serie_tasa(eda.serie(ent, por="linea"), "Tasa de tarde por mes y línea de negocio", color_por="linea"))

# %% [markdown]
# El volumen crece de unas 4.800 posiciones al mes en 2021 a unas 6.100 en 2023-2024, y la tasa de tarde cae en el mismo
# periodo. De 2021 a 2023 bajan todas las líneas de negocio, así que el cambio fue del proceso de compras y no de una
# sola línea. En 2024 se separan: palma (de 30,0 % a 37,7 %) y puerto (de 31,7 % a 44,3 %) vuelven a subir, mientras
# banano, ganadería y transporte siguen bajando, e industrial se queda alrededor del 39 %. Por eso la línea entra como
# predictora y la deriva se vigila por línea.

# %%
d = guardar(eda.descomponer(mensual), "stl")
mostrar(plots.stl(d))

# %%
mes = eda.tasa_por_grupo(ent.assign(mes_n=ent.fecha_pedido.dt.month), "mes_n").sort_values("mes_n")
mostrar(plots.objetivo_por(mes, "mes_n", "Tasa de tarde por mes de creación", "mes"))
dia = eda.tasa_por_grupo(ent.assign(dia=ent.fecha_prometida.dt.dayofweek.map(dict(enumerate(
    ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"])))), "dia")
mostrar(plots.objetivo_por(dia, "dia", "Tasa de tarde por día de la semana de la fecha prometida", "día",
                           orden=["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]))

# %% [markdown]
# La descomposición STL (tendencia con suavizado local, estacionalidad de periodo 12 y residuo, en su versión robusta a
# atípicos) separa dos cosas:
#
# - **Tendencia:** de 45,4 % a comienzos de 2021 a 34,4 % en enero de 2023, y estable desde entonces (33,2 % a final
#   de 2024). Es el cambio de régimen que ya mostraba la tasa por año.
# - **Estacionalidad:** enero está 11 puntos por debajo de la tendencia y octubre, 8 por encima. Por mes de creación,
#   la tasa va de 28,4 % (enero) a 39,5 % (marzo). Por día de la semana de la fecha prometida, los lunes llegan tarde más
#   (37,2 %) que el resto (32 % a 35 %): una entrega prometida para el lunes tiene el fin de semana en contra, aunque el
#   retraso se cuente en días hábiles. El mes y el día de la semana entran como predictoras.
#
# ### Estacionariedad y dependencia
#
# ADF tiene como hipótesis nula una raíz unitaria (la serie no vuelve a su media); KPSS, que la serie es estacionaria.
# Se leen juntas: si ADF no rechaza y KPSS sí, la serie no es estacionaria.

# %%
tabla(guardar(eda.estacionariedad(mensual.tasa), "estacionariedad"), dec=3)

# %%
a = guardar(eda.autocorrelacion(mensual.tasa), "acf")
mostrar(plots.acf_pacf(a))

# %% [markdown]
# ADF no rechaza la raíz unitaria (p = 0,70) y KPSS rechaza la estacionariedad (p = 0,026): **la tasa mensual no es
# estacionaria** en 2021-2024. La ACF decae despacio (0,76 en el primer rezago y todavía 0,52 en el sexto) y la PACF
# tiene sus picos en los rezagos 1, 2 y 4: es la huella de la tendencia más que de un ciclo corto. Consecuencias:
#
# - la partición train/validación/prueba es **cronológica**, y la validación cruzada de los modelos también (bloques de
#   ventana creciente: el modelo nunca ve el futuro de lo que predice);
# - el modelo trabaja por posición, no sobre la serie: no se construyen rezagos de la tasa mensual. El «pasado» entra
#   como el historial de cada proveedor, material y persona, cortado en la fecha de creación.
#
# ### Deriva entre periodos
#
# El índice de estabilidad de la población (PSI) compara la distribución de cada variable del entrenamiento con la de
# 2025 y 2026, por deciles del entrenamiento: $\mathrm{PSI} = \sum_j (a_j - r_j)\ln(a_j / r_j)$. Menos de 0,1 es
# estable; de 0,1 a 0,25, un cambio moderado; más de 0,25, un cambio fuerte. Solo mira las predictoras: no usa el
# objetivo de 2025 ni de 2026.

# %%
top = pd.read_csv(datos.RAIZ / "resultados" / "eda" / "top_numericas.csv").variable.tolist()
tabla(guardar(eda.deriva(df, top), "deriva"), dec=3)

# %% [markdown]
# La mayoría de las variables principales cambia poco o moderadamente (PSI de 0,02 a 0,15). Dos se mueven más:
#
# - la **tasa previa del solicitante** (0,20 en 2025 y 0,25 en 2026), en el límite del cambio fuerte;
# - la **tasa previa del comprador** (1,02 y 0,81), con un cambio fuerte: su distribución en 2025-2026 no se parece a la
#   del entrenamiento. Es un historial acumulado, y un comprador con años de historia arrastra el nivel de retraso alto
#   de 2021; además, el equipo de compras cambia.
#
# Es **deriva de covariables**: un modelo que se apoye mucho en esas dos variables puede perder desempeño fuera del
# entrenamiento. La validación 2025 lo va a mostrar; la prueba 2026 lo confirma una sola vez al final. La deriva del
# concepto (que la misma variable se relacione distinto con el retraso) no se mide aquí, porque exigiría el objetivo
# de 2025 y 2026.
#
# ## Componente espacial
#
# **No aplica.** El dataset no trae coordenadas: los centros y las sociedades van anonimizados y la región del
# proveedor es un código de departamento, que entra como categórica. Sin latitud y longitud no hay autocorrelación
# espacial que medir ni bloques espaciales que respetar.
#
# ## Auditoría de fuga
#
# **Disponibilidad al liberar.** Cada grupo de variables se revisó contra el momento de la predicción:
#
# | grupo | disponible el día de la liberación porque |
# |---|---|
# | orden, posición, plazo, calendario | se fijan al crear la orden; la fecha prometida es la vigente ese día |
# | liberación | ocurre antes o en el momento de la predicción, que es la última liberación |
# | historiales (proveedor, material, comprador, solicitante), pagos, stock | se cortan en la fecha de creación de la orden y cuentan solo posiciones con entrada anterior |
# | maestros | foto actual del ERP; **se retiraron** los que el planificador ajusta después de ver retrasos (plazo del maestro, precio medio móvil, stock de seguridad, punto de pedido, lote mínimo, estado del material) |
# | tasa de cambio | la del día de creación y sus variaciones hacia atrás |
#
# **Retiradas del dataset** por ser el desenlace o posteriores a la predicción: `y_otif_falla` y `dias_retraso` (son
# el objetivo con otra regla), las fechas de entrada y de factura, y los cambios de fecha posteriores a la liberación.
#
# **AUC univariado.** Una variable que separa sola casi a la perfección es sospechosa. Ninguna pasa de 0,75:

# %%
aso = pd.read_csv(datos.RAIZ / "resultados" / "eda" / "asociacion.csv")
mostrar(plots.barras_horizontales(aso.head(15).assign(n=aso.variable.map(N)), "AUC", "n",
                                  "Las 15 variables que más separan solas (AUC univariado)", "AUC", ".3f",
                                  referencia=0.75))

# %% [markdown]
# La que más separa, el plazo hábil al liberar (0,732), se calcula con la fecha prometida vigente el día de la liberación
# y con ese mismo día: las dos se conocen en el momento de la predicción. Separa porque es la causa más directa del
# retraso (sin plazo no hay forma de llegar a tiempo), no porque contenga el desenlace. Las entidades (proveedor,
# solicitante, comprador) se miden con su tasa fuera de pliegue: si se calculara con todo el entrenamiento, cada fila
# se vería a sí misma y el AUC del proveedor subiría de forma artificial.
#
# **Entidades repetidas entre particiones.** Ninguna orden cruza particiones (capítulo {doc}`02a_datos`), pero
# proveedores, materiales y personas sí se repiten, como en el uso real. La tabla dice qué parte de cada partición trae
# un valor que el entrenamiento no vio:

# %%
tabla(guardar(eda.entidades_nuevas(df), "entidades_nuevas"), dec=1)

# %% [markdown]
# Una de cada nueve posiciones de 2026 trae un proveedor que no estaba en 2021-2024 (11,1 %), y una de cada diez, un
# material nuevo (9,7 %). Es el **arranque en frío**: para ellos no hay historial. La codificación jerárquica les da la
# tasa de su nivel superior (región del proveedor, grupo de artículos), y el desempeño en esas filas se reporta aparte
# en la evaluación. Al reentrenar con 2021-2025 antes de medir la prueba, la fracción de valores nuevos baja.
#
# ## Preprocesamiento dentro del pipeline
#
# Todo lo que aprende de los datos (medianas, medias, categorías, tasas por entidad) se ajusta **solo con el
# entrenamiento**, dentro de un `Pipeline` de scikit-learn, para que la validación cruzada lo reajuste en cada pliegue:
#
# - **numéricas** (83): $\mathrm{signo}(x)\log(1+|x|)$ para comprimir las colas largas, mediana en los vacíos y
#   estandarización. La transformación es monótona: a los árboles no les cambia nada y a los modelos lineales les quita
#   el peso de los montos extremos;
# - **categóricas de pocos valores** (42): *one-hot*, con las categorías de menos de 30 filas juntas y las no vistas en
#   ceros;
# - **entidades** (11: proveedor, material, comprador, creador de la solicitud, liberador, centro de costo, cuenta,
#   centro, sociedad, estrategia, grupo de compras): **codificación jerárquica por objetivo**. Cada valor recibe su tasa
#   de retraso suavizada hacia la de su nivel superior,
#
# $$
# \hat p(v) = \frac{\text{tardías}(v) + M\,\hat p(\text{padre}(v))}{n(v) + M}, \qquad M = 20,
# $$ (eq-te)
#
# y un valor nuevo recibe la tasa de su padre (el material, la de su grupo de artículos; el proveedor, la de su
# región). En el ajuste se calcula **fuera de pliegue y agrupada por orden**, para que una posición no vea su propio
# objetivo ni el de las otras posiciones de su orden. Esa es la fuga más fácil de cometer en este problema.

# %%
from sklearn import set_config
set_config(display="text")
pre = datos.preprocesador()
Xt = pre.fit_transform(ent[datos.ENTRADA], ent.y_tarde)
for nombre, trans, cols in pre.transformers_:
    if nombre != "remainder":
        pasos = [n for n, _ in trans.steps] if hasattr(trans, "steps") else [type(trans).__name__]
        print(f"{nombre:4s} {len(cols):3d} columnas de entrada -> {' -> '.join(pasos)}")
print("matriz del entrenamiento después del preprocesamiento:", Xt.shape)

# %% [markdown]
# Las 136 predictoras pasan a 591 columnas, casi todas del *one-hot* de las categóricas. El bloque de entidades recibe
# 16 columnas porque lleva también los niveles superiores (de donde sale la tasa de un valor nuevo) y el número de orden
# (que solo agrupa los pliegues y no entra al modelo); devuelve una columna por entidad. El mismo objeto se ajusta
# con el entrenamiento y solo transforma la validación y la prueba; dentro de la validación cruzada se reajusta en cada
# pliegue. Los modelos de árboles del Entregable 2 usan este mismo preprocesamiento para que la comparación sea justa;
# el balanceo (SMOTE, ADASYN) va después de él y antes del modelo, también dentro del *pipeline*.
