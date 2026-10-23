# %% [markdown]
# # El mejor modelo en la prueba 2026
#
# XGBoost sin balanceo con Optuna ganó las 112 combinaciones en los ocho bloques de validación. Este capítulo lo mide en
# la prueba 2026, que ningún modelo usó para elegir nada, lo explica con SHAP y responde por qué se prefiere a la
# regresión logística, que es más simple y queda cerca.
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
# vez por 0,012 [0,004; 0,018].

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
# el material y el creador de la solicitud. Las tres familias de la literatura del capítulo 1 aparecen en ese orden: el
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
# ## ¿Por qué XGBoost y no la logística?
#
# La logística es más simple, se lee coeficiente por coeficiente y queda a 0,023 de AUC. La regla que se aplica es la
# parsimonia: se prefiere el modelo más simple salvo que el complejo gane de forma **significativa** y **material**. Las
# cinco razones, con cifras.
#
# **1. La diferencia es estadística.** En 2026, 0,848 contra 0,825: ΔAUC 0,023 con BCa por órdenes [0,017; 0,030] y
# DeLong pareado z = 22,5. En los ocho bloques (con la logística de las 112), 0,852 contra 0,839: XGBoost gana los ocho
# bloques (Wilcoxon con Holm, p = 0,047), DeLong es significativo en siete, la diferencia media es 0,012 con BCa
# [0,007; 0,016] y el delta de Cliff es 0,34, mediano. Y la ventaja es unas catorce veces lo que mueve la semilla.
#
# **2. La diferencia importa en la operación.** Un comprador no revisa todo: revisa las alertas que le caben. Con el
# mismo presupuesto de alertas para los dos, el de mayor riesgo:

# %%
pr = pd.read_csv(P / "presupuesto.csv")
tabla(pr.drop(columns=["meses"]), dec=3, pct=["presupuesto", "fracción de las tardías"],
      enteros=["alertas", "tardías atrapadas", "órdenes con una tardía atrapada"])

# %% [markdown]
# Revisando el 20 % de mayor riesgo (10.388 posiciones en los diez meses de 2026), XGBoost atrapa 8.242 tardías y la
# logística 7.895: 347 más, unas 35 por mes, con mejor precisión (0,79 frente a 0,76). Con el 10 % atrapa 165 más, con
# precisión 0,91. Al umbral de validación, el F1 es 0,675 contra 0,649. Una precisión: esas tardías de más caen en
# órdenes que ya tenían otra alerta. En órdenes con alguna tardía atrapada, XGBoost cubre 2.362 y la logística 2.407. Si
# la gestión es por orden y no por posición, la ventaja operativa se reduce; si es por posición (lo que se recibe), se
# mantiene.
#
# **3. Por qué la logística se queda corta.** La logística asigna a cada variable una sola pendiente, la misma para
# todas las órdenes. El efecto del plazo al liberar no es así. La figura compara, por tramo de plazo, el valor SHAP medio
# de XGBoost (línea llena) con el término de la logística (punteada), en las cuatro clases de compra más frecuentes:

# %%
fc = pd.read_csv(P / "forma_clase.csv")
mostrar(plots.forma_plazo(fc, "clase de compras", ["Generales", "Servicios", "Produccion Cluster", "Materia Prima"]))
tabla(fc.pivot(index="tramo", columns="clase de compras", values="xgboost").reindex(
    ["≤ 0", "1-2", "3-5", "6-10", "11-20", "> 20"]).reset_index(), dec=2)

# %% [markdown]
# - **Umbral cerca de cero.** Sin plazo (≤ 0 días), XGBoost suma unos 2,9 de logit. Con uno o dos días ya suma solo
#   1,0. La caída del primer tramo al segundo es de 1,9, contra 1,4 de la logística, que la reparte de forma suave con
#   el logaritmo.
# - **Interacción con la clase de compra.** El mismo «sin plazo» vale 3,4 de logit en Generales y 2,4 en Producción
#   Cluster, y el recorrido completo (de ≤ 0 a más de 20 días) es de 4,5 contra 3,1. En la logística las dos clases
#   tienen la misma pendiente; lo que cambia entre ellas es solo cómo se reparten los plazos dentro de cada tramo
#   (2,5 y 2,6). Con el tercio de la tasa previa del proveedor la interacción es más débil: con 1-2 días, 0,90 en los
#   proveedores puntuales y 1,22 en los que se retrasan (`forma_plazo.csv`).
#
# La logística captaría esto solo si alguien escribiera a mano los tramos y los productos plazo × clase; el boosting los
# encuentra solo.
#
# **4. La complejidad no es una restricción aquí.**

# %%
tabla(pd.read_csv(P / "costo.csv"), dec=3, enteros=["filas_un_ajuste"])

# %% [markdown]
# Un ajuste de XGBoost con 311.865 posiciones tarda 3,2 minutos en un hilo, contra 12 segundos de la logística, y se
# reentrena una vez por semestre. Predecir mil posiciones toma 17 milisegundos, contra 0,7. 2026 trae unas 5.200
# posiciones por mes: menos de una décima de segundo al mes. En el tablero, una predicción con sus contribuciones SHAP
# tarda 0,2 segundos. La legibilidad, que es la ventaja real de la logística, se cubre con SHAP: global, para saber qué
# mira el modelo, y local, para explicar cada alerta.
#
# **5. La parsimonia, aplicada con honestidad.** Las dos condiciones se cumplen: la ventaja es significativa (1) y
# material (2), y lo que cuesta es despreciable (4). Por eso se elige XGBoost. La logística se queda como modelo base y
# como referencia legible. Si el negocio pusiera la explicación de cada coeficiente por encima de atrapar tardías, por
# ejemplo para una auditoría o una regla de compras, elegir la logística sería defendible: pierde 0,023 de AUC y unas 35
# tardías atrapadas al mes.
#
# ## Qué deja este capítulo
#
# XGBoost, elegido en la validación, da en 2026 ROC-AUC 0,848 [0,835; 0,860], F1 0,675 y ECE 0,022 con Platt. No cae de
# la validación a la prueba. Le gana a la logística de forma significativa y material, y lo que la logística ofrece en
# legibilidad lo dan los valores SHAP. Donde más falla es en las órdenes con entidades nuevas (AUC 0,821) y en
# ganadería. El tablero sirve este modelo en la predicción en vivo. La propuesta del capítulo siguiente le gana por
# 0,012.
