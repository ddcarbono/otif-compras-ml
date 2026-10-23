# %% [markdown]
# # Modelo original: efectos por entidad, plazo log-normal y plazo real
#
# El mejor de las 112 combinaciones pierde donde el problema tiene estructura: órdenes con proveedores o materiales que
# no estaban en el entrenamiento (AUC 0,821 contra 0,856), posiciones de una misma orden que se mueven juntas, y una
# etiqueta que es el corte de una variable continua, los días de retraso. La propuesta ataca esas tres cosas y suma
# información que ningún modelo de las 112 tenía: cuánto tardó de verdad el proveedor en sus entregas anteriores.
#
# **La propuesta** tiene tres piezas:
#
# 1. **K3**, un *boosting* de árboles con efectos aleatorios cruzados por entidad, sobre la reducción ordinal de los días
#    de retraso, con los efectos ordenados en el tiempo al entrenar;
# 2. un **modelo log-normal del plazo**, que da al clasificador la probabilidad de que el plazo prometido no alcance;
# 3. **32 variables nuevas** de plazo real y holgura empírica.
#
# Se eligió con validación, se congeló antes de mirar 2026 y se mide en una comparación 2 × 2 que separa lo que aporta
# el modelo de lo que aporta la información. El código está en `original/` (con su `LEEME.md`); la comparación, en
# `src/prueba_2026.py`.

# %% tags=["remove-cell"]
import pandas as pd

from comun import datos, tabla

P = datos.RAIZ / "resultados" / "prueba_2026"

# %% [markdown]
# ## Formulación
#
# Sea $d_i$ los días de retraso de la posición $i$ y $c_k \in \{0, 1, 5, 15\}$ cuatro cortes. El modelo estima
#
# $$
# P(d_i > c_k \mid x_i) = \sigma\big(f(x_i, c_k) + b_{\text{prov}(i)} + b_{\text{mat}(i)} + b_{\text{sol}(i)}
#                         + b_{\text{comp}(i)}\big),
# $$
#
# con $\sigma$ la logística, $f$ un *boosting* de árboles (LightGBM) y $b_e$ un intercepto aleatorio por nivel de cada
# entidad: proveedor, material, creador de la solicitud y comprador. Se predice en $c = 1$, que es la etiqueta
# `y_tarde`.
#
# - **Reducción ordinal** {cite}`li2006`. Cada posición se repite una vez por corte, con la etiqueta $[d_i > c_k]$ y el
#   corte como una variable más. Así el modelo aprende también de las posiciones que llegaron 3 o 20 días tarde, y no
#   solo de si pasaron el corte. Una restricción monótona decreciente de $f$ en el corte garantiza
#   $P(d > 0) \ge P(d > 1) \ge P(d > 5) \ge P(d > 15)$ en cada fila: 0 % de filas con umbrales cruzados, contra 2,5 a
#   3,3 % sin ella.
# - **Efectos aleatorios cruzados** {cite}`griesbach2021,knieper2025`. Cada $b_e$ tiene una previa
#   $N(0, \sigma^2_e)$ que lo encoge hacia cero cuando la entidad tiene pocas posiciones. Se estiman por bloques
#   alternados: un *boosting* con los efectos como desplazamiento fijo del logit, y luego pasos de Newton por entidad
#   con $\sigma^2_e$ por EM (con la curvatura de Laplace). Al estimar se agrega un efecto por orden de compra, que
#   absorbe la dependencia entre posiciones de la misma orden y se descarta al predecir. La **corrección C** centra cada
#   efecto en cada barrido, para que el nivel medio lo lleve el árbol; por eso una entidad nueva, que recibe 0, recibe
#   el efecto medio.
# - **Efectos ordenados (K3)** {cite}`prokhorenkova2018`. Si los árboles se entrenan con el efecto de cada entidad
#   estimado con todo el periodo, cada fila ve un efecto que incluye su propio futuro, y al predecir una entidad nueva
#   recibe 0: el entrenamiento y la predicción no se parecen. K3 entrena los árboles con efectos estimados solo con los
#   semestres anteriores al de cada fila (0 en el primero y para la entidad que aún no había aparecido). Es la idea del
#   ordenamiento de CatBoost aplicada a efectos aleatorios. Al predecir se usan los efectos con todo el entrenamiento.
#
# **Modelo log-normal del plazo.** Un LightGBM de regresión estima $\mu_i = E[\log(1 + \text{plazo real}_i)]$ sin las
# variables de la promesa, fuera de pliegue por orden al entrenar. El clasificador recibe $\mu_i$ y
#
# $$
# z_i = \frac{\log(1 + \text{plazo prometido}_i + 1{,}5) - \mu_i}{s},
# $$
#
# con $s$ la desviación de los residuos. Si el plazo real fuera log-normal, $P(\text{tarde}) = 1 - \Phi(z_i)$. Solo, sin
# clasificador, ese modelo da un AUC de 0,850 a 0,880 por semestre.
#
# **Variables nuevas.** El plazo real de cada entrega previa en días hábiles (plazo al liberar más días de retraso), con
# su mediana y su percentil 80 en las últimas 20 entregas del proveedor, de la pareja proveedor-material y del material,
# y la **holgura empírica**: la promesa menos esa mediana. Solo cuenta lo que ya se sabía el día del pedido
# (`original/variables_nuevas_otif.py`). Son 32.
#
# **Lo propio.** Los artículos base tratan respuestas continuas o binarias con efectos de un solo tipo. Aquí se combinan
# una respuesta binaria con efectos cruzados de cuatro entidades y árboles, la reducción ordinal con la monotonía en el
# corte y el ordenamiento temporal de los efectos.
#
# ## Algoritmo

# %% [markdown]
# ```text
# entrada: posiciones de entrenamiento (x, d, entidades, OC, semestre), cortes C = {0, 1, 5, 15}
# 1. efectos con todo el periodo:
#      b ← 0 para cada entidad y para la OC;  σ² ← 0,5
#      repetir 3 veces:
#        f ← LightGBM sobre [d > 1] con desplazamiento Σ b            (474 árboles)
#        para cada entidad e, 2 barridos:
#          3 pasos de Newton: b_e ← b_e + (Σ (y − p) − b_e/σ²_e) / (Σ p(1 − p) + 1/σ²_e)
#          b_e ← b_e − media(b_e)                                       (corrección C)
#          σ²_e ← media(b_e² + 1/h_e)                                    (EM)
#      se guardan los b de las cuatro entidades; el de la OC se descarta
# 2. efectos ordenados: para cada semestre t, repetir 1 con las filas anteriores a t y asignar a las filas de t
# 3. ampliar: cada fila × cada corte, con la variable «corte» y la etiqueta [d > c]
# 4. LightGBM con desplazamiento = efectos ordenados, restricción monótona −1 en «corte»   (766 árboles al validar)
# 5. Platt y umbral de máximo F1 en la validación 2025; congelar
# predicción: f(x, c = 1) + Σ b_e (0 para una entidad nueva), luego Platt
# ```
#
# **Complejidad.** Con $n$ posiciones, $p$ columnas, $T$ árboles y $V$ vueltas de estimación de efectos, el
# entrenamiento cuesta $O(V \cdot T_0 \cdot n p)$ para los efectos más $O(T \cdot 4n \cdot p)$ para el *boosting*
# ordinal, que ve cuatro copias de cada fila. Los pasos de Newton son $O(n)$ por entidad. Los efectos ordenados repiten
# la estimación una vez por semestre. Predecir cuesta lo mismo que un LightGBM más una búsqueda por entidad.
#
# ## Cómo se eligió: validación y negativos
#
# Todas las decisiones se tomaron en la validación (2021-2024 para entrenar, 2025 para medir). Las piezas que llegaron a
# la propuesta pasaron además una **validación por cuatro semestres** (`original/mejorar_otif.py`): entrenar con todo lo
# anterior y medir en 2024-S1, 2024-S2, 2025-S1 y 2025-S2, con una regla fijada antes de correr: el intervalo de la
# diferencia por encima de cero y ganar en al menos tres de los cuatro semestres. La regla se puso porque las ganancias
# de una sola validación no se habían sostenido.
#
# | pieza | ΔAUC [IC 95 %] | semestres que gana | regla |
# |---|---|---|---|
# | 32 variables nuevas | +0,0080 [0,0056; 0,0105] | 4 de 4 (+0,005 a +0,015) | pasa |
# | solo plazo real y holgura empírica | +0,0063 [0,0042; 0,0086] | 4 de 4 | pasa |
# | solo lo reciente del proveedor | +0,0009 [−0,0005; 0,0025] | 4 de 4 | no pasa |
# | solo parejas proveedor-material, -grupo, -centro | 0,0000 | 2 de 4 | no pasa |
# | cinco semillas promediadas | +0,0001 | 2 de 4 | no pasa |
# | modelo log-normal del plazo, sobre las nuevas | +0,0028 [0,0012; 0,0043] | 4 de 4 | pasa |
#
# Antes de llegar ahí se probaron unos 34 candidatos en seis rondas. Los **resultados negativos** cuentan tanto como el
# elegido:
#
# - **Combinar** el modelo de efectos con otro *boosting* (promedio de logits, enrutado por entidad nueva, compuerta por
#   experiencia, residual) no sumó: cada combinación quedó donde el modelo de efectos solo, o no le ganó a su control.
# - **Contra el arranque en frío**, cuatro vías: apagar el efecto en la primera fecha de la entidad (K2), ordenar los
#   efectos en el tiempo (K3), un corrector con atributos e interacción proveedor-material y el texto de la posición
#   (TF-IDF, SVD y k-means). Solo K3 corrigió la falla en la validación; la factorización proveedor-material eligió no
#   usar memoria, y el texto no ayudó con los materiales nuevos. El corrector se hizo a partir de un prototipo escrito
#   con asistencia de IA, y así se declara.
# - **Ganancias de validación que no se sostienen.** En 2025 los modelos con efectos por entidad ganaban de 0,003 a
#   0,007; en 2026, el modelo de efectos sin ordenar empató con un LightGBM de referencia, ajustado con Optuna en la
#   misma validación (0,852 los dos). Esa fue la primera mirada a la prueba; K3, elegido después en la validación, fue
#   la segunda (0,854). Por eso se adoptó la validación por cuatro semestres, y por eso las miradas siguientes a 2026 se
#   declaran exploratorias.
#
# ## La comparación 2 × 2 en 2026
#
# Para separar lo que aporta el modelo de lo que aporta la información, se cruzan dos modelos (el XGBoost ganador de
# las 112, con los hiperparámetros de su bloque más reciente, y la propuesta) con dos conjuntos de variables (las 136
# del dataset y 136 + las 32 nuevas). Se agrega la ablación de K3 sin el modelo log-normal. Las seis celdas se
# entrenaron con 2021-2024, se calibraron con Platt y fijaron su umbral en 2025, y se congelaron
# (`original/config_2x2/`) antes de entrenar con 2021-2025 y medir en 2026. Es la tercera mirada a la prueba:
# exploratoria. Los intervalos son BCa al 95 % por órdenes (1.000 réplicas).

# %%
c = pd.read_csv(P / "dos_por_dos.csv")
c = c.assign(intervalo=[f"[{a:.3f}; {b:.3f}]".replace(".", ",") for a, b in zip(c["inf"], c["sup"])])
tabla(c[["celda", "ROC-AUC", "intervalo"]].rename(columns={"intervalo": "BCa 95 %"}), dec=4)
k = pd.read_csv(P / "contrastes.csv")
k = k.assign(intervalo=[f"[{a:.4f}; {b:.4f}]".replace(".", ",") for a, b in zip(k["inf"], k["sup"])])
tabla(k[["contraste", "diferencia", "ΔROC-AUC", "intervalo"]].rename(columns={"intervalo": "BCa 95 %"}), dec=4)

# %% [markdown]
# - **La propuesta completa supera al ganador de las 112** por 0,012 de AUC [0,004; 0,018]: 0,860 contra 0,848. Su F1
#   es 0,685 contra 0,675, y su calibración es mejor (ECE 0,013 contra 0,022; capítulo anterior).
# - **El modelo y la información son dos caminos a la misma ganancia.** Con las mismas 136 variables, la estructura de
#   la propuesta le gana a XGBoost por 0,008 [0,003; 0,014]. Con el plazo real, XGBoost sube 0,010 [0,006; 0,014], llega
#   a 0,858, y la ventaja de la estructura baja a 0,002, sin significancia. Lo que los efectos por entidad le sacan al
#   historial de cada proveedor, las variables de plazo real se lo dan a cualquier *boosting*.
# - **El log-normal suma poco al AUC**: 0,002 a 0,003, con el intervalo tocando el cero. Su papel es dar una
#   probabilidad de que el plazo prometido alcance, que se puede leer por sí sola, no subir el AUC.
# - **Entidades nuevas.** En las órdenes con alguna entidad nueva ninguna diferencia entre la propuesta y XGBoost es
#   significativa. La ganancia está en las órdenes donde todo es conocido. El arranque en frío sigue abierto.
#
# ## Qué deja este capítulo
#
# La propuesta (K3 + log-normal + plazo real) es el mejor modelo medido en 2026: AUC 0,860, F1 0,685, ECE 0,013. Le
# gana al ganador de las 112 por 0,012, con un intervalo lejos de cero. La 2 × 2 muestra de dónde sale esa ganancia: la
# información nueva sola da 0,010 y la estructura sola 0,008, pero juntas dan 0,012, porque se sustituyen entre sí. Para
# el negocio, la consecuencia práctica es que medir el plazo real de cada proveedor vale tanto como el modelo más
# elaborado. Lo que no se logró es mejorar las órdenes con entidades nuevas, aunque se intentó por cuatro vías.
