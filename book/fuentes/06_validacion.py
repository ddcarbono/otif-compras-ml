# %% [markdown]
# # La propuesta en otros datos
#
# El modelo original le gana al de las 112 en OTIF por 0,012 de AUC. Una mejora en un solo conjunto de datos puede
# deberse a rasgos del caso y no al método. Este capítulo mide la pieza central de la propuesta, K3, contra el XGBoost
# de las 112 en siete conjuntos públicos de OpenML. El código está en `original/publicos_otif.py` y
# `original/resumir_publicos_otif.py`; los datos se descargan aparte y no viajan con el repositorio.

# %% tags=["remove-cell"]
import pandas as pd

from comun import datos, tabla

d = pd.read_csv(datos.RAIZ / "original" / "publicos_otif_resumen.csv")

# %% [markdown]
# ## Protocolo
#
# El mismo para los dos modelos y los siete conjuntos:
#
# - **Preprocesamiento común.** Las columnas con muchos valores van con codificación por objetivo suavizada, fuera de
#   pliegue en el entrenamiento; las numéricas, tal cual. K3 cambia la codificación de sus entidades por efectos
#   aleatorios, como en OTIF.
# - **Mismo presupuesto.** Optuna con 20 evaluaciones sobre el AUC de validación y la misma semilla. XGBoost busca en
#   el espacio de las 112; K3, en el de su *boosting*.
# - **Una sola medida.** Se reajusta con entrenamiento y validación y se mide una vez en la prueba: AUC con DeLong
#   pareado, intervalo de la diferencia por *bootstrap* y F1 al umbral de máximo F1 de la validación.
# - **Partición.** Temporal donde el conjunto trae fecha; si no la trae, estratificada 60/20/20.
#
# El modelo log-normal y las variables nuevas no se pueden llevar a estos conjuntos: necesitan un plazo prometido y la
# historia de entregas, que solo trae OTIF. Lo que se mide aquí es K3 solo.
#
# **Dos rondas.** La primera tomó tres conjuntos conocidos: retrasos de vuelos (Airlines), acceso de empleados (Amazon) y
# clics en publicidad (Click). Sus resultados dieron una hipótesis: K3 necesita rasgos que OTIF tiene. Con ella se fijó,
# antes de correr, el criterio de la segunda ronda: conjuntos con al menos dos de tres rasgos (entidades que se repiten,
# orden en el tiempo, desenlace continuo u ordinal). Entraron los retrasos de buses escolares de Nueva York, los del
# transporte público de Zúrich, los reingresos hospitalarios de Diabetes130US y las citas médicas no asistidas. Se
# reportan todos.
#
# ## Resultados
#
# El p de DeLong se ajusta por Holm sobre los siete conjuntos. «Gana» o «pierde» es con p ajustado menor que 0,05.

# %%
NOMBRE = {"airlines": "Airlines (vuelos)", "amazon": "Amazon (acceso)", "click": "Click (publicidad)",
          "nyc_bus": "Buses de Nueva York", "zurich": "Transporte de Zúrich", "diabetes": "Diabetes130US (reingreso)",
          "noshow": "Citas no asistidas"}
t = d.assign(conjunto=d.conjunto.map(NOMBRE), IC=[f"[{a:.3f}; {b:.3f}]".replace(".", ",") for a, b in zip(d.ic_lo, d.ic_hi)])
tabla(t[["ronda", "conjunto", "n", "tasa", "auc_xgb", "auc_k3", "dif", "IC", "p_holm", "veredicto", "f1_xgb", "f1_k3"]]
      .rename(columns={"tasa": "positivos", "auc_xgb": "AUC XGBoost", "auc_k3": "AUC K3", "dif": "ΔAUC (K3 − XGB)",
                       "IC": "IC 95 %", "p_holm": "p (Holm)", "f1_xgb": "F1 XGBoost", "f1_k3": "F1 K3"}),
      dec=3, pct=["positivos"], enteros=["n", "ronda"])

# %% [markdown]
# K3 gana en dos, empata en tres y pierde en dos. El F1 es mayor con K3 en cinco de los siete. La prueba de Wilcoxon
# sobre las siete diferencias de AUC da p = 0,81: **no hay una superioridad general** de K3 sobre XGBoost.
#
# ## Lectura
#
# - **Gana donde una entidad tiene historia propia repetida en el tiempo.** En Diabetes130US (+0,016 [0,008; 0,024]) y
#   en las citas no asistidas (+0,005 [0,003; 0,008]) el paciente vuelve, como el proveedor y el material en OTIF.
# - **Empata en los retrasos de transporte** (vuelos, buses, Zúrich). Ahí las entidades son pocas y estables
#   (aerolínea, ruta, línea) y la codificación por objetivo de XGBoost ya las aprovecha.
# - **Pierde sin tiempo ni desenlace continuo** (Amazon, Click), donde K3 se reduce a sus efectos aleatorios y XGBoost
#   usa mejor las categorías de muchos valores.
#
# **Cuidado.** Esta separación se observó después de la segunda ronda. Para confirmarla habría que fijar una tercera
# ronda con ese criterio antes de correrla. Por ahora es una hipótesis con apoyo, no un resultado.
#
# Dos limitaciones más. Airlines no trae el año, así que su partición es por mes del año (estacional, no estrictamente
# temporal). Y en OTIF la ganancia de la propuesta viene también del plazo log-normal y de las variables nuevas, que
# aquí no se pueden probar.
#
# ## Qué deja este capítulo
#
# La ventaja de la propuesta en OTIF no es una propiedad general del método. Aparece donde el caso se parece a OTIF:
# entidades con historia individual en el tiempo. En datos sin esa estructura, el XGBoost de las 112 es igual o mejor.
