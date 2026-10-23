# %% [markdown]
# # Los datos
#
# ## El problema
#
# Un grupo empresarial colombiano de la agroindustria de la palma de aceite compra cada año decenas de miles de
# materiales y servicios: repuestos, insumos de campo y de planta, fletes, mantenimiento. Cada compra sale como una
# **orden de compra (OC)** hacia un proveedor, con una fecha prometida de entrega por posición. **Una de cada tres
# posiciones llega tarde.** Un repuesto tarde para una extractora o un insumo tarde para una plantación detiene un
# proceso, obliga a comprar de urgencia o a mantener inventario de más. Un estudio de compras en una empresa que usa el
# mismo ERP reporta un 23 % de entregas tardías {cite}`widyaswara2025`, y que bajó al 10 % con seguimiento y
# recordatorios automáticos a los proveedores: el retraso se puede gestionar si se sabe dónde mirar.
#
# **Pregunta de investigación.** ¿Se puede predecir, el día en que la orden se libera hacia el proveedor, si cada
# posición va a llegar tarde, con lo que se sabe ese día?
#
# **Objetivos.**
# 1. Construir un clasificador binario que estime la probabilidad de que una posición llegue tarde, calibrada para que
#    el comprador la pueda usar como riesgo.
# 2. Identificar qué variables, conocidas al liberar, explican el retraso.
#
# **Formulación.** Cada posición $i$ tiene un vector $x_i \in \mathbb{R}^{136}$ conocido el día $t_i$ de su liberación y
# un desenlace
#
# $$
# y_i = \mathbb{1}\{\text{la primera entrada de mercancía llega más de 1 día hábil después de la fecha prometida}\}.
# $$ (eq-objetivo)
#
# La tarea es **clasificación binaria**: estimar $p(x) = P(y = 1 \mid x)$ con lo observado antes de $t_i$ y decidir con
# un umbral $\tau$ que se elige en la validación, nunca en la prueba.

# %% tags=["remove-cell"]
import numpy as np
import pandas as pd

from comun import datos, guardar, mostrar, tabla
from src import eda, plots

df = datos.cargar()
ent = datos.particion(df, "entrenamiento")

# %% [markdown]
# ## Fuente, licencia y estructura
#
# - **Fuente:** las tablas de compras del ERP SAP de la empresa (orden, posición, reparto, imputación, historial de
#   cambios de fecha, entradas de mercancía, solicitudes de pedido, maestros de proveedor y material, pagos y tasa de
#   cambio), extraídas en modo de solo lectura. Corte de la extracción: 21 de octubre de 2026.
# - **Licencia:** datos de la empresa, con autorización de uso académico. **No son redistribuibles en abierto**: el
#   repositorio trae una muestra anonimizada para el tablero, y el conjunto completo se entrega aparte al docente.
# - **Unidad de observación:** la posición de la orden de compra (un material o servicio, una cantidad, una fecha
#   prometida). Sin agregación.
# - **Estructura:** eventos con fecha (órdenes creadas de enero de 2021 a septiembre de 2026, a intervalos irregulares)
#   con entidades que se repiten: varias posiciones por orden, y el mismo proveedor, material o comprador en muchas
#   órdenes. No es una serie de tiempo regular ni un panel balanceado, y no hay coordenadas.
#
# Por eso la partición es **cronológica** (el modelo se usaría sobre órdenes futuras) y **no parte ninguna orden**
# entre conjuntos. Se fijó antes de mirar los datos: entrenamiento 2021-2024, validación 2025 y prueba 2026, por año de
# creación de la orden. Todo este capítulo se calcula solo con el entrenamiento, salvo lo que compara las particiones.

# %%
t = guardar(eda.tamano(df).reset_index(), "tamano")
tabla(t, pct=["tasa de tarde"], dec=1)

# %% [markdown]
# Son 406.126 posiciones, muy por encima del mínimo de 20.000, y $n/p \approx 2.986$ con $p = 136$ predictoras. El
# tamaño efectivo es menor que el número de filas: las posiciones de una misma orden comparten proveedor, fechas y
# aprobación, así que las **unidades independientes son las 104.614 órdenes** (2,8 posiciones por orden en promedio).
# Los intervalos de confianza del proyecto se calculan con *bootstrap* por orden, no por fila. La clase minoritaria
# («tarde») tiene 134.067 casos: no hay escasez de positivos.
#
# **Particiones.** Ninguna orden queda partida entre dos conjuntos:

# %%
print("órdenes con posiciones en más de una partición:", eda.oc_entre_particiones(df))

# %% [markdown]
# ## Población: qué entra y qué no
#
# El extractor parte de 490.310 posiciones y se queda con las **compras con plazo**: las que tienen una fecha por
# delante y pueden llegar tarde.
#
# | filtro | posiciones | por qué |
# |---|---|---|
# | ventana | quedan 474.025 | la fecha prometida es al menos 30 días anterior al corte, para que el desenlace ya se conozca |
# | sin ninguna entrada de mercancía | 15.359 fuera | no hay desenlace que medir |
# | legalizaciones | 40.597 fuera | la entrada o la factura es del día de creación de la orden o anterior: la orden se hizo después de recibir |
# | compras de fruta del mes | 17.928 fuera | orden abierta que acumula las compras del mes, sin fecha real |
# | imputación de títulos valores y de pedido de cliente | 3.819 fuera | no son compras de abastecimiento |
# | rutas cumplidas | 27 fuera | se registran después de prestado el servicio |
#
# Una posición puede caer en más de un filtro. Servicios (33,6 %) y activos fijos (1,9 %) quedan en el mismo dataset y
# se distinguen con `es_servicio` y `clase_compras`.
#
# **El objetivo, con precisión.** El momento de la predicción es la última liberación (aprobación) de la orden; sin
# estrategia de aprobación, su creación. La fecha prometida es la vigente ese día, reconstruida con el historial de
# cambios: un cambio posterior no cuenta, porque el día de la liberación no se conocía. El retraso se cuenta en días
# hábiles con los festivos de Colombia, hasta la primera entrada de mercancía, con un día hábil de tolerancia.
#
# ## Diccionario de variables
#
# Las 136 predictoras van en 13 grupos: orden y posición, plazo y calendario, liberación, personas (con seudónimo),
# solicitud de pedido, imputación, clasificación, historial del proveedor, historial de material, comprador y
# solicitante, pagos al proveedor, maestros, stock y tasa de cambio, y marcas de estructura. **Todo historial se corta
# en la fecha de creación de la orden**: una posición previa cuenta si su primera entrada fue anterior.

# %%
tabla(datos.DIC.groupby(["grupo", "tipo"]).size().unstack(fill_value=0).reset_index())

# %% tags=["hide-output"]
tabla(datos.DIC[["variable", "nombre", "grupo", "tipo", "unidad", "significado"]])

# %% [markdown]
# Los maestros del proveedor y del material (`mc_`, `m_`, `pv_`, `ps_`) son la foto actual del ERP, no la del día de la
# orden. Los que el planificador ajusta después de ver retrasos (plazo de entrega del maestro, precio medio, stock de
# seguridad, punto de pedido, lote mínimo, estado) se retiraron por fuga en la depuración (capítulo {doc}`02c_tiempo_y_fuga`).
#
# ## Calidad de los datos
#
# ### Valores faltantes
#
# Hay vacíos en 33 de las 136 variables. La matriz muestra que no están dispersos al azar: vienen **en bloques**, y
# cada bloque corresponde a algo que la posición no tiene.

# %%
fal = guardar(eda.faltantes(ent), "faltantes")
mostrar(plots.matriz_vacios(ent, fal.variable.tolist(), datos.NOMBRE))

# %%
tabla(eda.vacios_por_bloque(ent), pct=["tasa de tarde"], dec=1)

# %% [markdown]
# **El mecanismo.** Los bloques son **vacíos estructurales**: una posición sin solicitud de pedido no tiene datos de
# solicitud; una sin imputación (gasto que va a inventario) no tiene centro de costo; un servicio no tiene maestro de
# material ni stock. El vacío es la ausencia del objeto, no un dato perdido. Las marcas `tiene_solped`,
# `tiene_imputacion` y `tiene_material` lo explican casi por completo (de 0 % a 18 % de vacíos con la marca en 1).
#
# Como el vacío depende de una variable observada (la marca) y no del valor que faltaría, el mecanismo es **MAR**, y
# no MCAR. La evidencia está en la tabla de abajo: si fuera MCAR, la tasa de tarde sería la misma con y sin dato, y
# difiere hasta en 4 puntos. No hay razón para MNAR: el valor que «falta» no existe. La prueba de Little no se
# aplica: supone normalidad multivariada en variables continuas, y con 136 variables mixtas y 270.000 filas rechazaría
# siempre sin decir por qué.
#
# **Tratamiento:** no se imputa con modelos. En el *pipeline* la mediana rellena el número y la marca `tiene_*` conserva
# la información del vacío; los árboles manejan el vacío directamente.

# %%
tabla(fal.head(15), dec=1, pct=["tasa con dato", "tasa sin dato"])

# %% [markdown]
# ### Duplicados, valores imposibles y atípicos

# %%
tabla(eda.duplicados(ent), dec=2)

# %% [markdown]
# - **Duplicados exactos (2,3 %).** En el 97 % de los casos son posiciones de una misma orden con el mismo material,
#   cantidad y fecha: pedidos partidos en varias líneas, no errores de carga. Se conservan: son compras reales. Lo que importa es que no crucen
#   particiones, y no lo hacen porque la partición es por orden.
# - **Mismas predictoras y distinto objetivo (169 filas, 0,06 %).** Es el ruido irreducible: con la misma información,
#   una llegó a tiempo y otra no.
# - **Casi duplicados (24 %).** El mismo material repetido dentro de una orden (por ejemplo, entregas parciales
#   programadas). Por eso la validación agrupa por orden.

# %%
tabla(eda.imposibles(ent), dec=2)

# %% [markdown]
# - **Valor en cero (7,1 %).** Todas son de la clase «Generales» (el 11 % de esa clase): posiciones que salen sin
#   precio en la orden. Tienen proveedor, fecha prometida y entrada, así que son compras con plazo y se quedan.
# - **Fecha prometida antes del pedido (1,9 %).** La orden se crea con una fecha que ya pasó. Es un comportamiento del
#   proceso, no un error. `nace_vencida` (20,9 %) es más amplia: marca la posición que llega vencida al día de la
#   liberación, porque la aprobación tardó más que el plazo.
# - **Posiciones abiertas del proveedor negativas (0,4 %)** y **antigüedad negativa (56 filas).** Entradas registradas
#   antes de la creación de la orden y proveedores dados de alta después: inconsistencias del ERP. Son pocas y no se
#   corrigen; quedan como límite de esas variables.
# - **Plazos de más de 10 años (12 filas).** Errores de digitación de la fecha. Son 12 de 271.391 filas: no mueven
#   ningún resultado, y se dejan para no tocar a mano los datos de origen.
#
# Los **atípicos univariados** (regla de Tukey: fuera de $[Q_1 - 1{,}5\,\mathrm{IQR},\ Q_3 + 1{,}5\,\mathrm{IQR}]$) están
# en la tabla univariada del capítulo {doc}`02b_eda`. Los montos y las cantidades tienen colas larguísimas (asimetría
# mayor de 100) y concentran los atípicos; no se eliminan, porque una compra grande es justamente la que importa.
#
# ### Sesgos de muestreo
#
# - **Censura al final del periodo.** Las posiciones sin ninguna entrada salen de la población. Cerca del corte, una
#   posición muy atrasada todavía no tiene entrada y queda fuera, mientras que una a tiempo sí entra. La tasa de 2026
#   puede estar algo subestimada. El margen de 30 días lo reduce, no lo elimina.
# - **Solo las órdenes con plazo.** Las legalizaciones y las compras de fruta quedan fuera: el modelo no sirve para
#   ellas, y no se pretende.
# - **Una sola empresa.** Los resultados describen su proceso y sus proveedores; generalizar a otra empresa exige
#   reentrenar.
#
# ## Consideraciones éticas
#
# - **Personas.** Comprador, creador de la solicitud y liberador salen con seudónimo (`CMP021`, `SOL0264`, `LIB007`).
#   El modelo no evalúa personas: su historial entra como contexto del proceso. Una lectura del modelo como ranking de
#   compradores sería un mal uso.
# - **Terceros.** Proveedores, materiales, centros de costo, cuentas, estrategias, clases de orden y almacenes van con
#   código. Las sociedades del Grupo van con una abreviatura legible (`SOC-TEQ`) y los centros, con la de su sociedad.
# - **Reidentificación.** La combinación de fecha, proveedor, material y valor podría identificar una compra para
#   alguien de dentro de la empresa. Por eso el conjunto completo no se publica, las llaves de los seudónimos no salen
#   de la empresa y el tablero solo carga una muestra.
# - **Sin texto libre.** Los textos de las órdenes solo entran como una familia (cotización, corte de pago, semanal,
#   importación, etc.).
