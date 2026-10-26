# %% [markdown]
# # El mejor modelo en la prueba 2026
#
# XGBoost sin balanceo con Optuna ganó las 112 combinaciones en los ocho bloques de validación. Este capítulo lo mide en
# la prueba 2026, que ningún modelo usó para elegir nada, y lo explica con SHAP. Por qué se prefiere a la regresión
# logística está al final del capítulo anterior.
#
# **Lo que se fijó antes de mirar 2026.** Los hiperparámetros son los que eligió Optuna en el bloque más reciente
# (2025-S2). Con ellos, el modelo se entrenó con 2021-2024 y se ajustaron en la validación 2025 la calibración de Platt y
# el umbral de máximo F1 (0,345). Los dos se congelaron en `resultados/prueba_2026/config_xgb136.json` antes de la prueba.
# Después se reentrenó con 2021-2025 y se midió una vez en 2026. Esa medida es la tercera mirada del proyecto a la
# prueba (las dos anteriores fueron del modelo original del capítulo siguiente): se lee como exploratoria. El cálculo
# está en `src/prueba_2026.py`, `src/modelo_final.py` y `src/xgb_vs_logistica.py`; las probabilidades por posición vienen
# de la comparación del capítulo siguiente (`original/evaluar_2x2_otif.py`).

# %% tags=["remove-cell"]
import json

import pandas as pd

from comun import datos, mostrar, tabla
from src import plots
from src.data import diccionario

P = datos.RAIZ / "resultados" / "prueba_2026"
NOMBRES = dict(zip(diccionario().variable, diccionario().nombre))
NOMBRES.update({"proveedor": "Proveedor (tasa codificada)", "material": "Material (tasa codificada)"})
met = pd.read_csv(P / "metricas.csv")
xgb = met.set_index("modelo").loc["XGBoost (las 112)"]

# %% [markdown]
# El modelo de `artifacts/xgboost.joblib`, el que sirve el tablero, se reentrenó por separado con el mismo código. Sus
# probabilidades en 2026 coinciden con las de la comparación:

# %%
rep = json.loads((P / "reproduccion.json").read_text(encoding="utf-8"))
tabla(pd.DataFrame([{"posiciones": rep["filas"], "ROC-AUC aquí": rep["AUC_aqui"], "ROC-AUC de la comparación": rep["AUC_2x2"],
                     "diferencia máxima": rep["dif_max"], "misma decisión": rep["misma_decision"]}]),
      dec=6, enteros=["posiciones"], pct=["misma decisión"])

# %% [markdown]
# ## De la validación a la prueba
#
# La **tabla de brecha** sigue el AUC del mismo modelo en cada etapa, de la búsqueda de hiperparámetros a la prueba. Si
# el modelo se hubiera sobreajustado a la validación, el AUC caería en cada paso.

# %%
tabla(pd.read_csv(P / "brecha.csv"), dec=4)

# %% [markdown]
# XGBoost pasa de 0,854 en la validación interna a 0,852 en los bloques externos: la búsqueda casi no infló el AUC. En
# la validación 2025, entrenado solo con 2021-2024, da 0,834, y en 2026, con un año más de datos, 0,848. No hay una
# caída de la validación a la prueba: el AUC sube y baja con el semestre (de 0,83 a 0,89 en los bloques), y 2026 queda
# dentro de ese rango. La logística baja un poco (de 0,830 a 0,825).
#
# ## Métricas en 2026

# %%
tabla(met.drop(columns=["partición"]), dec=3)
dif = pd.read_csv(P / "diferencias.csv")
tabla(dif.rename(columns={"diferencia": "ΔROC-AUC"}), dec=4)

# %% [markdown]
# Los intervalos son BCa al 95 % con 1.000 réplicas que remuestrean órdenes completas. XGBoost llega a ROC-AUC 0,848
# [0,835; 0,860], PR-AUC 0,758 y F1 0,675 a su umbral. Le gana a la logística por 0,023 de AUC [0,017; 0,030] y a la
# regla del proveedor por 0,18. La propuesta del Entregable 3, que se presenta en el capítulo siguiente, le gana a su
# vez por 0,011 [0,003; 0,017].

# %%
cur = pd.read_csv(P / "curvas.csv")
mostrar(plots.curva_roc_pr(cur, "ROC", met))
mostrar(plots.curva_roc_pr(cur, "PR", met))
mostrar(plots.matriz_confusion(pd.read_csv(P / "confusion.csv"), xgb.umbral))
mostrar(plots.calibracion(pd.read_csv(P / "calibracion.csv"), "Calibración en la prueba 2026 (deciles)"))

# %% [markdown]
# De las 16.374 posiciones que llegaron tarde en 2026, XGBoost marcó 11.341 (69,3 %), con 5.889 falsas alarmas: el
# 16,6 % de las que llegaron a tiempo. La logística, con su propio umbral, marcó un poco más de tardías (71,3 %) a costa
# de 7.929 falsas alarmas (22,3 %). Con Platt, XGBoost queda calibrado como la logística (ECE 0,022 frente a 0,021). Sin
# Platt sería sobreconfiado, como mostró la curva de convergencia del capítulo anterior.
#
# ## Dónde se equivoca

# %%
seg = pd.read_csv(P / "segmentos.csv")
tabla(seg[seg.modelo == "XGBoost (las 112)"].drop(columns="modelo"), dec=3, enteros=["posiciones"])

# %% [markdown]
# Las posiciones de órdenes con alguna entidad que no estaba en el entrenamiento (proveedor, material, comprador u otra)
# son el 20 % de la prueba. Ahí el AUC baja de 0,856 a 0,821: es el arranque en frío que ya se veía en la validación. Por
# línea, el peor caso es ganadería (AUC 0,712, 908 posiciones), que en la validación también era la peor. Transporte es
# la mejor (0,864).
#
# ## Qué mira el modelo: SHAP
#
# Los valores SHAP reparten el logit de cada predicción entre las variables, de modo que sumen exacto (TreeSHAP, en la
# escala del margen, antes de Platt). La importancia global es la media del valor absoluto en las 51.941 posiciones de
# 2026; las categóricas suman sus columnas *one-hot* y las entidades entran como su tasa codificada.

# %%
sg = pd.read_csv(P / "shap_global.csv")
mostrar(plots.shap_global(sg, NOMBRES))

# %% [markdown]
# El plazo hábil que le queda al proveedor cuando la orden se libera pesa casi el doble que lo siguiente (0,85 de logit
# en promedio), y después viene quién provee (0,46, la tasa codificada del proveedor). El resto queda por debajo de 0,13:
# la promesa en días, la experiencia del proveedor con ese centro, la cuenta contable, los días que tardó la liberación,
# el material y el creador de la solicitud. Las tres familias de la literatura del capítulo {doc}`literatura` aparecen en ese orden: el
# tiempo disponible, el proveedor y el proceso interno.
#
# Tres posiciones de 2026 explicadas, las de probabilidad mediana de cada caso (sin códigos de entidad):

# %%
sl = pd.read_csv(P / "shap_local.csv")
sl["variable"] = sl.variable.map(lambda v: NOMBRES.get(v, v))
tabla(sl.drop(columns="base"), dec=3)

# %% [markdown]
# - **Verdadero positivo** (0,68): el proveedor empuja hacia arriba con fuerza (+1,9) aunque al liberar quedaban 27 días
#   hábiles (−0,66). El historial del proveedor pesa más que la holgura.
# - **Falso negativo** (0,22): el proveedor y el creador de la solicitud empujan hacia abajo, y la liberación lenta (16
#   días hábiles) sube el riesgo, pero no alcanza el umbral. Nada de lo que se sabía al liberar anunciaba el retraso.
# - **Falso positivo** (0,46): el plazo amplio (29 días) baja el riesgo, pero el proveedor, con un retraso medio previo de
#   11 días, lo sube; esa vez cumplió.
#
# ## LIME contra SHAP
#
# LIME {cite}`ribeiro2016` explica una predicción de otra forma: genera vecinos de la fila cambiando algunas variables
# al azar, le pregunta al modelo por cada vecino y ajusta una regresión lineal ponderada por cercanía; sus coeficientes
# son la explicación. No es exacto ni único: depende del muestreo y de cómo se define el vecino. Se calculó en el espacio
# de las variables originales (cada categórica y cada entidad como una variable, cada numérica por cuartiles con un
# tramo propio para el vacío), con 3.000 vecinos, en las tres posiciones de arriba y en 200 de 2026 al azar
# (`src/lime_xgb.py`). SHAP está en la escala del logit y LIME en la de la probabilidad: se comparan el orden y el signo,
# no el tamaño.

# %%
la = pd.read_csv(P / "lime_acuerdo.csv")
tabla(la, dec=2)
le = pd.read_csv(P / "lime_estabilidad.csv")
tabla(le)

# %% [markdown]
# - **Coinciden en lo grueso.** De las cinco variables principales de SHAP, LIME les da el mismo signo a 4,3 en promedio,
#   y las dos explicaciones comparten 2 de sus 5 principales (entre 2 y 3 en la mitad central de las posiciones). En las
#   tres posiciones de arriba, LIME también pone primero el plazo al liberar y el proveedor cuando SHAP lo hace.
# - **Fuera de las primeras, no.** La correlación de rangos sobre todas las variables es 0,16: en las variables que
#   pesan poco, el orden de LIME es casi ruido.
# - **LIME cambia con la semilla.** Con cinco semillas en la misma posición, solo 0 a 2 de sus cinco principales se
#   repiten siempre, y entre 9 y 15 variables distintas pasan por ese grupo. SHAP de árbol es exacto y da siempre lo mismo.
# - **Con entidades de miles de niveles, LIME no distingue.** Un vecino casi nunca cae en el mismo proveedor o material
#   que la fila, así que esas variables valen «igual a la fila» solo en la fila misma y LIME les reparte el mismo peso
#   (en el falso negativo, proveedor, solicitante y material reciben −0,044 los tres).
#
# Por eso el tablero y este libro explican con SHAP: es exacto para árboles, estable y suma el logit. LIME queda como
# contraste, útil para confirmar el signo de lo principal.
#
# ## Qué deja este capítulo
#
# XGBoost, elegido en la validación, da en 2026 ROC-AUC 0,848 [0,835; 0,860], F1 0,675 y ECE 0,022 con Platt. No cae de
# la validación a la prueba. Le gana a la logística por 0,023 de AUC, y lo que la logística ofrece en
# legibilidad lo dan los valores SHAP. Donde más falla es en las órdenes con entidades nuevas (AUC 0,821) y en
# ganadería. El tablero sirve este modelo en la predicción en vivo. La propuesta del capítulo siguiente le gana por
# 0,011.
