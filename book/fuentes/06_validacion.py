# %% [markdown]
# # Validación estadística
#
# El capítulo anterior midió BOREAL contra el XGBoost de las 112 en la prueba 2026 con DeLong e intervalos BCa por
# órdenes. Este agrega dos cosas que pide una afirmación de superioridad: **pruebas que respetan el orden en el tiempo**
# dentro de 2026, con ajuste por comparaciones múltiples y tamaño del efecto, y **siete conjuntos públicos** de OpenML
# donde se mide la pieza central de BOREAL, el boosting mixto, con el mismo protocolo. El código está en
# `src/validacion_original.py`, `original/publicos_otif.py` y `original/resumir_publicos_otif.py`; los conjuntos
# públicos se descargan aparte y no viajan con el repositorio.

# %% tags=["remove-cell"]
import pandas as pd

from comun import datos, tabla

d = pd.read_csv(datos.RAIZ / "original" / "publicos_otif_resumen.csv")
P26 = datos.RAIZ / "resultados" / "prueba_2026"

# %% [markdown]
# ## Pruebas en el tiempo dentro de 2026
#
# DeLong compara el AUC de todo 2026 y supone posiciones independientes; las de una misma orden no lo son, y por eso el
# capítulo anterior acompaña cada diferencia con un intervalo BCa que remuestrea órdenes. Además, la prueba 2026 tiene
# orden en el tiempo: una ventaja que solo aparece en unas semanas no es la misma que una que se sostiene. Se agregan:
#
# - **Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold** sobre la serie semanal de la diferencia de
#   pérdida logarítmica media (XGBoost − BOREAL; positiva si BOREAL se equivoca menos), con varianza de Newey-West;
# - **Wilcoxon de rangos con signo** sobre las diferencias de AUC mes a mes, y el **delta de Cliff** entre los AUC
#   mensuales como tamaño del efecto (no despreciable desde |δ| ≥ 0,147).
#
# Las seis pruebas (tres por contraste: la propuesta completa y el modelo con las mismas 136 variables) se ajustan
# juntas por Holm. Como sensibilidad, fuera de Holm, Diebold-Mariano se repite solo con las semanas de 200 posiciones o
# más: en la media semanal una semana de 10 posiciones pesa lo mismo que una de 2.000.

# %%
vo = pd.read_csv(P26 / "validacion_original.csv")
vo["p (Holm)"] = [("fuera de Holm" if pd.isna(v) else f"{v:.4f}".replace(".", ",")) for v in vo["p (Holm)"]]
tabla(vo[["contraste", "prueba", "unidades", "medida del efecto", "efecto", "estadístico", "p", "p (Holm)"]], dec=4,
      enteros=["unidades"])
vm = pd.read_csv(P26 / "validacion_original_mensual.csv")
vm = vm[vm.contraste.eq("propuesta completa contra el ganador de las 112")]
tabla(vm[["mes", "posiciones", "AUC BOREAL 136 + nuevas", "AUC XGBoost 136"]], dec=3, enteros=["posiciones"])

# %% [markdown]
# - **DeLong** confirma la diferencia de AUC de todo 2026 en los dos contrastes (p < 0,001 después de Holm). Su z de 12,6
#   está inflado por suponer posiciones independientes; el intervalo BCa por órdenes del capítulo anterior, [0,003;
#   0,017], es la medida honesta de la incertidumbre, y tampoco toca el cero.
# - **Mes a mes**, BOREAL tiene el AUC más alto en siete de los nueve meses; pierde en mayo y en septiembre, que tiene
#   1.016 posiciones. Wilcoxon no alcanza significancia (p = 0,13; 0,52 con Holm) y el delta de Cliff es 0,21, pequeño.
#   Con nueve meses la prueba tiene poca potencia: para p < 0,05 haría falta ganar en casi todos con diferencias grandes.
# - **En la pérdida logarítmica semanal**, Diebold-Mariano no encuentra diferencia (p = 0,47). La media semanal es
#   negativa por las últimas semanas de la prueba, de 10 a 183 posiciones, donde las pocas órdenes ya resueltas son
#   atípicas y BOREAL, con probabilidades más extremas, pierde mucho en pocas filas. Con las 36 semanas de 200
#   posiciones o más, BOREAL se equivoca menos (0,013 por posición, p = 0,007; con las 136, p = 0,014). Esta
#   sensibilidad se decidió después de ver la serie, así que no entra en Holm ni en el veredicto: muestra de dónde sale
#   el resultado nulo, no lo reemplaza.
#
# En conjunto, la ventaja de BOREAL en el AUC de todo 2026 está bien establecida, pero es del orden de la variación entre
# meses: las dos pruebas que respetan el tiempo no la confirman después de Holm.
#
# ## Siete conjuntos públicos
#
# ### Protocolo
#
# El mismo para los dos modelos y los siete conjuntos:
#
# - **Preprocesamiento común.** Las columnas con muchos valores van con codificación por objetivo suavizada, fuera de
#   pliegue en el entrenamiento; las numéricas, tal cual. El boosting mixto cambia la codificación de sus entidades por efectos
#   aleatorios, como en OTIF.
# - **Mismo presupuesto.** Optuna con 20 evaluaciones sobre el AUC de validación y la misma semilla. XGBoost busca en
#   el espacio de las 112; el boosting mixto, en el suyo.
# - **Una sola medida.** Se reajusta con entrenamiento y validación y se mide una vez en la prueba: AUC con DeLong
#   pareado, intervalo de la diferencia por *bootstrap* y F1 al umbral de máximo F1 de la validación.
# - **Partición.** Temporal donde el conjunto trae fecha; si no la trae, estratificada 60/20/20.
#
# El modelo log-normal y las variables nuevas no se pueden llevar a estos conjuntos: necesitan un plazo prometido y la
# historia de entregas, que solo trae OTIF. Lo que se mide aquí es el boosting mixto solo.
#
# **Dos rondas.** La primera tomó tres conjuntos conocidos: retrasos de vuelos (Airlines), acceso de empleados (Amazon) y
# clics en publicidad (Click). Sus resultados dieron una hipótesis: el boosting mixto necesita rasgos que OTIF tiene. Con ella se fijó,
# antes de correr, el criterio de la segunda ronda: conjuntos con al menos dos de tres rasgos (entidades que se repiten,
# orden en el tiempo, desenlace continuo u ordinal). Entraron los retrasos de buses escolares de Nueva York, los del
# transporte público de Zúrich, los reingresos hospitalarios de Diabetes130US y las citas médicas no asistidas. Se
# reportan todos.
#
# ### Resultados
#
# El p de DeLong se ajusta por Holm sobre los siete conjuntos. «Gana» o «pierde» es con p ajustado menor que 0,05.

# %%
NOMBRE = {"airlines": "Airlines (vuelos)", "amazon": "Amazon (acceso)", "click": "Click (publicidad)",
          "nyc_bus": "Buses de Nueva York", "zurich": "Transporte de Zúrich", "diabetes": "Diabetes130US (reingreso)",
          "noshow": "Citas no asistidas"}
t = d.assign(conjunto=d.conjunto.map(NOMBRE), IC=[f"[{a:.3f}; {b:.3f}]".replace(".", ",") for a, b in zip(d.ic_lo, d.ic_hi)])
tabla(t[["ronda", "conjunto", "n", "tasa", "auc_xgb", "auc_k3", "dif", "IC", "p_holm", "veredicto", "f1_xgb", "f1_k3"]]
      .rename(columns={"tasa": "positivos", "auc_xgb": "AUC XGBoost", "auc_k3": "AUC boosting mixto", "dif": "ΔAUC (mixto − XGB)",
                       "IC": "IC 95 %", "p_holm": "p (Holm)", "f1_xgb": "F1 XGBoost", "f1_k3": "F1 boosting mixto"}),
      dec=3, pct=["positivos"], enteros=["n", "ronda"])

# %% [markdown]
# El boosting mixto gana en tres conjuntos, empata en cuatro y no pierde en ninguno. El F1 es mayor con él en seis de
# los siete. La prueba de Wilcoxon sobre las siete diferencias de AUC da p = 0,69: **no hay una superioridad general**
# del boosting mixto sobre XGBoost, pero tampoco una inferioridad en ninguno.
#
# ### Lectura
#
# - **Gana donde una entidad tiene historia propia repetida en el tiempo.** En Diabetes130US (+0,017 [0,009; 0,025]) y
#   en las citas no asistidas (+0,005 [0,002; 0,008]) el paciente vuelve, como el proveedor y el material en OTIF; en
#   los buses escolares de Nueva York (+0,008 [0,005; 0,010]) se repiten el bus, la ruta y la empresa, con el retraso
#   en minutos como desenlace continuo.
# - **Empata en los otros cuatro.** En vuelos y en Zúrich las entidades son pocas y estables (aerolínea, línea), y la
#   codificación por objetivo de XGBoost ya las aprovecha. En Amazon y Click, sin tiempo ni desenlace continuo, el
#   boosting mixto queda por debajo en AUC (−0,008 y −0,007), pero la diferencia no sobrevive al ajuste por Holm.
#
# **Cuidado.** El criterio de la segunda ronda se fijó antes de correrla, y en ella el boosting mixto gana en tres de
# cuatro; pero los conjuntos se eligieron por parecerse a OTIF, así que esto es evidencia de dónde sirve el método, no
# de que sea mejor en general.
#
# Dos limitaciones más. Airlines no trae el año, así que su partición es por mes del año (estacional, no estrictamente
# temporal). Y en OTIF la ganancia de la propuesta viene también del plazo log-normal y de las variables nuevas, que
# aquí no se pueden probar.
#
# ## Qué deja este capítulo
#
# BOREAL le gana al ganador de las 112 en el AUC de 2026 con un intervalo por órdenes que no toca el cero, y DeLong lo
# confirma después de Holm. Las pruebas que respetan el tiempo, Diebold-Mariano semanal y Wilcoxon mensual, no alcanzan
# significancia: la ganancia de 0,011 es real en el año, pero pequeña frente a lo que varía de un mes a otro. Fuera de
# OTIF, la pieza central de BOREAL gana en tres de siete conjuntos públicos, empata en cuatro y no pierde en ninguno,
# sin superioridad general (Wilcoxon p = 0,69). El veredicto sobre R3 está en {doc}`08_conclusiones`.
