# %% [markdown]
# # Análisis exploratorio
#
# Todo se calcula con el entrenamiento (órdenes de 2021 a 2024, 271.391 posiciones). La validación y la prueba no
# entran en ninguna decisión de este capítulo.

# %% tags=["remove-cell"]
import numpy as np
import pandas as pd

from comun import datos, guardar, mostrar, tabla
from src import eda, plots

df = datos.cargar()
ent = datos.particion(df, "entrenamiento")
N = datos.NOMBRE

# %% [markdown]
# ## La variable objetivo

# %%
mostrar(plots.balance(ent.y_tarde, "Posiciones por clase (entrenamiento)"))

# %% [markdown]
# En el entrenamiento llegan tarde 92.153 de 271.391 posiciones (34,0 %). Es un desbalance moderado: la clase
# minoritaria tiene más de 90.000 casos y no falta información para aprenderla. Dos consecuencias para el modelado:
#
# - **La métrica no puede ser la exactitud.** Un clasificador que diga siempre «a tiempo» acierta el 66 %. Se usan el
#   **AUC** (qué tan bien ordena las posiciones por riesgo, sin depender de un umbral) y el **F1** de la clase «tarde» en
#   el umbral elegido en validación.
# - **El balanceo no es obligatorio**, pero el Entregable 2 lo compara (SMOTE, ADASYN, pesos por clase) y aquí se espera
#   que cambie poco.

# %%
anio = eda.tasa_por_grupo(ent.assign(año=ent.fecha_pedido.dt.year), "año").sort_values("año")
mostrar(plots.objetivo_por(anio, "año", "Tasa de tarde por año de la orden", "año"))
linea = guardar(eda.tasa_por_grupo(ent, "linea"), "tasa_linea")
mostrar(plots.objetivo_por(linea, "linea", "Tasa de tarde por línea de negocio", "línea"))
clase = guardar(eda.tasa_por_grupo(ent, "clase_compras"), "tasa_clase")
mostrar(plots.objetivo_por(clase, "clase_compras", "Tasa de tarde por clase de compra", "clase"))

# %% [markdown]
# - **Por año** la tasa baja de 42,7 % (2021) a 26,9 % (2023) y vuelve a 31,5 % (2024). El proceso cambió: un modelo
#   que mezcle años al azar aprendería un nivel de retraso que ya no existe. Por eso la partición es cronológica.
# - **Por línea de negocio** va de 25,0 % (ganadería) a 41,5 % (industrial); palma, la línea con más posiciones (112.000),
#   está en 36,2 %.
# - **Por clase de compra** las diferencias son mayores: los activos fijos (56,2 %) y la materia prima (64,0 %) llegan
#   tarde más de la mitad de las veces; los servicios (30,1 %), menos que el promedio. Son pocas posiciones (5.043 y
#   1.699), pero el modelo tiene que poder distinguirlas: la clase entra como predictora.
#
# ## Análisis univariado
#
# Una fila por variable, en dos tablas (numéricas y categóricas). La curtosis es **en exceso**: 0 para la normal.

# %% tags=["hide-output"]
un = guardar(eda.univariado_numericas(ent), "univariado_numericas")
tabla(un, dec=2)

# %% tags=["hide-output"]
uc = guardar(eda.univariado_categoricas(ent), "univariado_categoricas")
tabla(uc, dec=1)

# %% [markdown]
# Lo que dicen las dos tablas:
#
# - **Colas larguísimas en montos y cantidades.** Cantidad, valor de la posición, valor de la orden, precio y valor
#   para la liberación tienen asimetría de 30 a 300 y curtosis en exceso de miles: la mediana del valor de una posición
#   es 244.296 COP y la media, 27,8 millones. Entre el 15 % y el 17 % de sus valores cae fuera de las cercas de Tukey.
#   No son errores (una compra grande es una compra grande); para los modelos lineales conviene una transformación
#   logarítmica, y a los árboles no les importa la escala.
# - **Plazos con valores negativos.** El plazo hábil al liberar va del percentil 1 en −30 al 99 en 118 días: una de cada
#   cinco posiciones llega vencida al día de la liberación (`nace_vencida`, 20,9 %).
# - **Normalidad.** No se aplica una prueba formal: con 271.391 filas, Shapiro-Wilk o Kolmogorov-Smirnov rechazan con
#   desviaciones mínimas. La asimetría y la curtosis dicen lo mismo con más información: ninguna variable continua del
#   dataset es normal, y ningún modelo del proyecto lo supone.
# - **Binarias casi constantes.** `importacion` (1 %) y `peso_neto` (1 %) tienen poca variación; se dejan porque no
#   estorban y la importación es un caso de negocio distinto.
# - **Categorías raras.** La depuración ya llevó a su nivel superior las de menos de 30 filas en el entrenamiento;
#   quedan pocas (columna «raras»), y el *one-hot* las vuelve a juntar.
# - **Categorías «(en blanco)».** SAP marca «sin valor» con un espacio (sin imputación, no va a inspección): es una
#   categoría con significado, no un faltante.
#
# Las figuras se limitan a las doce variables numéricas que más separan las dos clases (sección siguiente); el eje va en
# escala logarítmica con signo cuando la asimetría pasa de 2.

# %%
aso = guardar(eda.asociacion(ent), "asociacion")
top = aso[aso.tipo == "numérica"].variable.head(12).tolist()
guardar(pd.DataFrame({"variable": top}), "top_numericas")
for v in top:
    mostrar(plots.histograma_por_clase(ent, v, N[v], datos.UNIDAD[v], log=abs(ent[v].skew()) > 2))

# %% [markdown]
# La que más separa es el **plazo hábil que le queda al proveedor el día de la liberación**: la mediana es de 14 días
# en las a tiempo y de 4 en las tardías. Con el plazo agotado al liberar (cero o menos), llega tarde el 85,9 %; con más
# de 10 días, alrededor del 20 %, y más plazo ya no ayuda. Es una relación **no lineal con un codo**, que un árbol
# capta solo y una logística necesita que se le dé (por tramos o con un *spline*).
#
# Le siguen el **historial del proveedor** (mediana de su tasa previa de retraso: 28 % en las a tiempo, 39 % en las
# tardías), el **plazo prometido** (28 contra 19 días) y los **días que tardó la aprobación interna** (3 contra 5). Ninguna
# separa sola: las distribuciones de las dos clases se solapan en todas.
#
# ## Análisis bivariado
#
# ### Cada predictora frente al objetivo
#
# Una sola tabla para las 136 variables. El **AUC univariado** es la probabilidad de que una posición tardía elegida al
# azar tenga un valor más alto (o más bajo, según la dirección) que una a tiempo: 0,5 es no separar nada. Las
# numéricas llevan además la **información mutua**, que capta relaciones no monótonas; las categóricas, la **V de
# Cramér** ($\sqrt{\chi^2 / (n\,(k-1))}$, de 0 a 1); y las entidades de miles de valores, el AUC de su tasa de retraso
# calculada fuera de pliegue y agrupada por orden (con miles de categorías la V de Cramér se infla). Los p se ajustan por
# Holm sobre las 136 pruebas.

# %%
tabla(aso.head(25), dec=3)

# %% tags=["hide-output"]
tabla(aso, dec=3)

# %% [markdown]
# - **Ninguna variable separa sola con un AUC mayor de 0,75.** La más fuerte es el plazo hábil al liberar (0,732);
#   la siguen el proveedor (0,695) y su historial (0,66). No hay señales de fuga (capítulo {doc}`02c_tiempo_y_fuga`).
# - **Tres familias explican casi todo:** el **plazo** (cuánto tiempo real le queda al proveedor), **quién provee y
#   quién pide** (proveedor, creador de la solicitud, comprador, material, con su historial) y **la aprobación**
#   (días de liberación, estrategia). Las de calendario, pagos, stock y tasa de cambio quedan con AUC por debajo de
#   0,58.
# - **La información mutua** confirma el orden y muestra que el historial del proveedor tiene más información de la que
#   refleja su AUC (0,20, la más alta): su relación con el retraso no es monótona en todo el rango.
# - **Todos los p ajustados son menores de 10⁻³⁰⁰** (el límite de la precisión numérica): con este $n$ la significancia
#   no discrimina. Lo que ordena la tabla es el tamaño del efecto (AUC, V de Cramér), y por eso se reporta.
# - **Selección preliminar:** no se descarta ninguna variable aquí. Las últimas de la tabla tienen AUC cercano a 0,5,
#   pero los árboles pueden usarlas en interacciones; la selección se deja a la regularización de cada modelo.
#
# ### Contrastes por grupo
#
# Con $n$ = 271.391 cualquier diferencia sale significativa; lo que informa es el **tamaño del efecto**. Para el
# retraso como binaria, chi-cuadrado y V de Cramér; para los días de retraso, Kruskal-Wallis (no supone normalidad ni
# varianzas iguales) y $\varepsilon^2 = (H - k + 1)/(n - k)$, la fracción de la variación de los rangos que explica el
# grupo.

# %%
tabla(guardar(eda.contrastes_grupo(ent), "contrastes"), dec=3)

# %% [markdown]
# Las dos asociaciones existen, pero son **pequeñas**: la V de Cramér de la línea de negocio es 0,107 y la de la clase de
# compra, 0,096 (por convención, menos de 0,1 es débil y de 0,1 a 0,3, moderado). En los días de retraso, la línea
# explica el 2,0 % de la variación de rangos y la clase, el 1,0 %. El grupo mueve la tasa, pero dentro de cada grupo
# queda casi toda la variación: el retraso se decide en la posición, no en la línea.
#
# ### Relaciones entre predictoras
#
# Correlación de Spearman (sobre rangos: no exige linealidad y resiste los atípicos) entre las 30 numéricas más
# asociadas al objetivo, y el factor de inflación de la varianza (VIF) de las mismas.

# %%
top30 = aso[aso.tipo == "numérica"].variable.head(30).tolist()
c = eda.spearman(ent, top30)
c.reset_index(names="variable").to_csv(datos.RAIZ / "resultados" / "eda" / "spearman_top30.csv", index=False)
mostrar(plots.mapa_correlacion(c, N))

# %%
vf = guardar(eda.vif(ent, top30), "vif")
tabla(vf.head(10), dec=1)

# %% [markdown]
# - **El bloque de plazo es casi colineal.** El plazo hábil al liberar, el plazo prometido y los días de liberación
#   tienen VIF de 1.059, 1.031 y 322: el primero es, casi por construcción, el segundo menos el tercero (en días
#   hábiles). Para los árboles no importa; para la logística, la regularización L1 o L2 reparte o elige entre ellas, y
#   los coeficientes de ese bloque no se pueden leer uno por uno.
# - **El historial se agrupa:** las tasas previas de proveedor, material, comprador y solicitante se correlacionan
#   entre sí (todas miden «cómo le ha ido a quien participa»), y los conteos previos con la antigüedad del proveedor.
# - Fuera de esos dos bloques, los VIF quedan por debajo de 6: no hay más redundancia que la que la depuración ya
#   quitó (pares con |ρ| ≥ 0,95).
#
# ## Análisis multivariado
#
# Sobre las numéricas, con una muestra de 50.000 posiciones: colas largas comprimidas con $\mathrm{signo}(x)\log(1+|x|)$,
# vacíos con la mediana y estandarización.

# %%
var, cargas = eda.pca(ent)
guardar(var, "pca_varianza")
mostrar(plots.varianza_pca(var.head(40)))
print("componentes para el 80 % de la varianza:", int((var.acumulada < 0.8).sum() + 1), "de", len(var))
tabla(cargas.assign(abs1=cargas.CP1.abs()).sort_values("abs1", ascending=False).drop(columns="abs1").head(8)
      .rename(index=N).reset_index(names="variable"), dec=2)

# %% [markdown]
# La varianza está **muy repartida**: el primer componente explica el 12,3 % y hacen falta 33 de 80 componentes para
# llegar al 80 %. No hay unas pocas direcciones que resuman los datos, así que reducir dimensión antes de modelar
# perdería información; PCA queda como herramienta de exploración. El primer componente opone **materiales con
# historia** (con material, registro info, muchas órdenes previas del par proveedor-material) a **servicios y compras
# de alto valor**: es la diferencia entre comprar un repuesto de catálogo y contratar un servicio.

# %%
tabla(guardar(eda.atipicos_multivariados(ent), "atipicos_if"), dec=3, pct=["tasa de tarde"])

# %% [markdown]
# Isolation Forest marca como atípicas el 2,9 % de las posiciones de la muestra. Llegan tarde el 40,4 %, frente al
# 33,7 % del resto: lo raro se retrasa más (compras inusuales en monto, plazo o proveedor). No se eliminan: son casos
# reales y justamente los que interesa anticipar. El tratamiento de los atípicos queda en la elección de modelos
# robustos (árboles) y en la transformación de las colas para los lineales.

# %%
sil, grupos = eda.grupos_kmedias(ent)
tabla(sil, dec=3)
tabla(grupos, pct=["tasa"])

# %% [markdown]
# La silueta más alta es 0,25, con cinco grupos: una estructura de grupos **débil** (por convención, menos de 0,25 es
# poca o ninguna). Las tasas de tarde por grupo van de 28,6 % a 36,0 %, sin un grupo que concentre el retraso. No hay
# subpoblaciones marcadas que pidan un modelo por grupo; la heterogeneidad que importa está en las entidades (proveedor,
# material, comprador) y en el plazo, y el modelo la toma como predictoras.
