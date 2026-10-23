# %% [markdown]
# # Modelo base: regresión logística
#
# Antes de las 112 combinaciones de modelo, balanceo y optimizador hace falta una referencia: el modelo más simple que
# use todas las predictoras con el preprocesamiento del capítulo anterior. Si un modelo complejo no le gana con claridad,
# su complejidad no se justifica. La regresión logística modela el logaritmo de la razón de odds de llegar tarde como una
# combinación lineal de las predictoras, y la penalización L2 encoge los coeficientes hacia cero para controlar la
# varianza cuando hay cientos de columnas correlacionadas.
#
# El cálculo está en `src/modelo_base.py` (unos 20 minutos); este capítulo lee sus tablas de `resultados/modelo_base/`.

# %% tags=["remove-cell"]
import json

import pandas as pd

from comun import datos, mostrar, tabla
from src import plots

R = datos.RAIZ / "resultados" / "modelo_base"
res = json.loads((R / "resumen.json").read_text(encoding="utf-8"))
met = pd.read_csv(R / "metricas.csv")
NOMBRES = datos.NOMBRE

# %% [markdown]
# ## Protocolo
#
# 1. **Hiperparámetro.** C (el inverso de la fuerza de la penalización) se elige en una rejilla de 0,0001 a 3 con
#    validación cruzada temporal dentro de 2021-2024: se entrena con 2021 y se valida con 2022, con 2021-2022 y se valida
#    con 2023, con 2021-2023 y se valida con 2024. Cada pliegue reajusta todo el pipeline, codificación por objetivo
#    incluida, así que ninguna tasa de un año de validación entra a su entrenamiento.
# 2. **Umbral y calibración.** Con C fijo se ajusta con 2021-2024 y, en la validación 2025, se elige el umbral que
#    maximiza el F1 y se mide la calibración.
# 3. **Prueba.** Se reentrena con 2021-2025 y se mide 2026 una sola vez, con el umbral de la validación. Los intervalos al
#    95 % salen de un bootstrap de órdenes: se remuestrean órdenes completas, porque las posiciones de una misma orden
#    comparten proveedor, fecha y comprador y no son independientes.
#
# Las dos líneas base: `DummyClassifier`, que asigna a todas las posiciones la tasa de tarde del entrenamiento (un AUC de
# 0,5 por construcción), y la regla del proveedor, que ordena las posiciones por la tasa previa de retraso de su proveedor:
# lo que haría un comprador sin modelo.

# %%
cv = pd.read_csv(R / "cv_C.csv")
mostrar(plots.validacion_c(cv))
tabla(cv.pivot(index="C", columns="valida", values="AUC").reset_index(), dec=4)

# %% [markdown]
# El AUC medio sube al encoger C desde 3 hasta 0,003 (de 0,828 a 0,839) y vuelve a bajar por debajo de 0,001: el
# máximo queda dentro de la rejilla, en C = 0,003. Una penalización fuerte ayuda porque hay 591 columnas, muchas de ellas
# indicadoras de categorías raras o casi colineales. Pero la diferencia entre C vecinos (menos de 0,002) es diez veces
# menor que la variación entre años con el mismo C (desviación de 0,022): 2023 es el año más difícil (0,817) y 2024 el más
# fácil (0,869). Elegir bien C importa poco; el año que toque evaluar importa mucho más. Por eso las 112 combinaciones se
# comparan bloque a bloque, con pruebas pareadas.

# %% [markdown]
# ## Resultados en la prueba 2026

# %%
pru = met[met["partición"] == "prueba 2026"]
tabla(pru[["modelo", "ROC-AUC", "ROC-AUC inf", "ROC-AUC sup", "PR-AUC", "F1", "F1 inf", "F1 sup", "precisión",
           "sensibilidad", "exactitud", "Brier", "ECE", "umbral", "marcadas"]], dec=3, pct=["marcadas"])

# %% [markdown]
# Las mismas métricas en la validación 2025, donde se fijó el umbral:

# %%
tabla(met[met["partición"] == "validación 2025"][["modelo", "ROC-AUC", "PR-AUC", "F1", "precisión", "sensibilidad",
                                                  "Brier", "ECE", "umbral"]], dec=3)

# %% [markdown]
# Cómo se leen las métricas:
#
# - **ROC-AUC**: la probabilidad de que una posición tarde reciba más puntaje que una a tiempo elegida al azar. Mide el
#   ordenamiento, sin depender del umbral; 0,5 es azar.
# - **PR-AUC** (precisión promedio): el área bajo la curva de precisión contra sensibilidad. Su referencia es la
#   prevalencia (31,5 % en 2026), no 0,5, y se fija en la clase que importa.
# - **F1**: la media armónica de la precisión (de las marcadas, cuántas llegan tarde) y la sensibilidad (de las que
#   llegan tarde, cuántas se marcan), con el umbral fijado en la validación.
# - **Brier**: el error cuadrático medio de la probabilidad; **ECE**: la distancia media entre la probabilidad predicha y
#   la tasa observada en cada decil. Las dos miden calibración.
#
# La logística ordena las posiciones con un AUC de 0,825 [0,813; 0,838] en 2026, frente a 0,670 [0,654; 0,686] de la
# regla del proveedor: los intervalos no se tocan. Mirar solo el historial del proveedor deja fuera el plazo, el material,
# la cuenta y el resto de lo que se sabe al liberar, y eso vale 0,155 de AUC. La PR-AUC (0,726) más que duplica la
# prevalencia.
#
# Con el umbral de 0,287 la logística marca el 37,7 % de las posiciones; de ellas, el 59,5 % llega tarde, y encuentra el
# 71,3 % de las que llegan tarde (F1 0,649 [0,631; 0,666]). La regla del proveedor, con su mejor umbral, tiene que marcar el
# 67,7 % para encontrar el 80,1 %, con una precisión de 37,3 %: poco más que marcar todas, que es lo que hace el Dummy
# (F1 0,479, precisión igual a la prevalencia).
#
# De la validación 2025 a la prueba 2026 el AUC baja de 0,830 a 0,825 y el F1 se mantiene (0,649): el umbral y el C
# elegidos con años anteriores siguen sirviendo un año después.

# %%
curvas = pd.read_csv(R / "curvas.csv")
mostrar(plots.curva_roc_pr(curvas, "ROC", pru))
mostrar(plots.curva_roc_pr(curvas, "PR", pru))

# %% [markdown]
# La curva ROC de la logística queda por encima de la regla del proveedor en todo el rango. En la curva de precisión y
# sensibilidad la diferencia es mayor en la zona de sensibilidad baja. Para encontrar el 10 % de las posiciones tardías,
# la logística acierta en el 97 % de las que marca y la regla del proveedor en el 67 %; para el 20 %, en el 91 % frente
# al 58 %. Un comprador que solo puede seguir unas pocas órdenes al día trabaja en esa parte de la curva.
#
# ## Umbral y calibración

# %%
conf = pd.read_csv(R / "confusion.csv")
mostrar(plots.matriz_confusion(conf, res["umbral_f1_validacion"]["Regresión logística"]))
cal = pd.read_csv(R / "calibracion.csv")
mostrar(plots.calibracion(cal))

# %% [markdown]
# De las 16.374 posiciones que llegaron tarde en 2026, el modelo marcó 11.672 (71,3 %). Hubo 7.929 falsas alarmas, el
# 22,3 % de las 35.567 que llegaron a tiempo.
#
# La probabilidad se puede leer como riesgo: en cada decil la tasa observada queda a 4,2 puntos o menos de la predicha, y
# el ECE es de 2,1 puntos. El Brier baja de 0,216 (Dummy) a 0,149. La desviación tiene una dirección: en los deciles 3 a 9
# el modelo subestima un poco (decil 8: predice 42,1 %, se observa 46,3 %), y en el decil más alto sobreestima (90,1 %
# contra 87,7 %). La regresión logística ajusta por máxima verosimilitud, que tiende a calibrar bien, y no necesita la
# corrección de Platt que sí suelen necesitar los árboles y el *boosting*.
#
# ## Curva de aprendizaje

# %%
ca = pd.read_csv(R / "curva_aprendizaje.csv")
mostrar(plots.curva_aprendizaje(ca))
tabla(ca, dec=4, pct=["fracción"], enteros=["filas"])

# %% [markdown]
# Con el 10 % de las órdenes (28.366 posiciones) el AUC de validación ya es 0,823. Con todas (271.391) llega a 0,830:
# diez veces más datos suman 0,007. La brecha entre entrenamiento y validación se cierra de 0,066 a 0,037. Es el patrón de
# un modelo con sesgo, no con varianza: le sobran datos y le falta flexibilidad. Una frontera lineal en el espacio de las
# predictoras no capta interacciones como «plazo corto *y* proveedor lento». Ahí tienen margen los árboles, los bosques
# y el *boosting* de las 112 combinaciones.
#
# El AUC de entrenamiento no está inflado por la codificación por objetivo: en el ajuste, las tasas de cada entidad se
# calculan fuera de pliegue, así que ninguna posición ve su propio retraso.
#
# ## Coeficientes

# %%
coef = pd.read_csv(R / "coeficientes.csv")
mostrar(plots.coeficientes(coef, NOMBRES, 20))

# %% [markdown]
# Las numéricas están estandarizadas, así que cada coeficiente es el cambio en el logaritmo de la razón de odds por una
# desviación estándar, con todo lo demás fijo. En las categóricas, es la diferencia frente a las demás categorías.
#
# - **La historia de las entidades pesa más que cualquier otra cosa.** La tasa codificada del proveedor tiene el mayor
#   coeficiente (1,87): una desviación estándar más de retraso histórico multiplica los odds por 6,5. Le siguen la cuenta
#   contable (0,79), el material (0,63) y el centro de costo (0,46).
# - **El plazo**: el plazo hábil al liberar (−0,96) y el plazo prometido en días calendario (−0,66) bajan el riesgo,
#   como en el capítulo anterior. Son colineales (VIF de más de 300 en el bloque): la regularización reparte el efecto
#   entre ellos, y solo la suma (−1,6) se puede leer como el efecto del plazo.
# - **Categorías**: las órdenes en euros (+0,43, importaciones) y los grupos de artículos de dotaciones (+0,45) y
#   construcción (+0,33) suben el riesgo; ganadería (−0,38), las órdenes en pesos (−0,34) y el grupo de transporte (−0,30)
#   lo bajan.
#
# Son asociaciones condicionales encogidas por la penalización, no efectos causales. Con variables correlacionadas, el
# signo de un coeficiente pequeño puede cambiar si se quita otra variable.
#
# ## Qué deja este capítulo
#
# El modelo base deja la vara en **ROC-AUC 0,825 y F1 0,649 en la prueba 2026**, bien calibrado y 0,155 de AUC por
# encima de la regla que usaría un comprador sin modelo. La curva de aprendizaje dice que un modelo lineal ya no mejora con
# más datos. Lo que queda por ganar está en la flexibilidad. Las 112 combinaciones de modelo, balanceo y optimizador se
# miden en los ocho bloques semestrales de la validación anidada y se comparan contra esta referencia.
