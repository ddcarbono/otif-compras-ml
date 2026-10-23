# %% [markdown]
# # Fundamentos: XGBoost y el costo de cada modelo
#
# El capítulo anterior dejó a XGBoost primero en los ocho bloques. Aquí se explica cómo funciona: qué función optimiza,
# cómo elige cada corte y qué controla cada hiperparámetro. Después se compara la complejidad teórica de los siete modelos
# con lo que tardan de verdad, medido con `src/complejidad.py` sobre el bloque más grande (311.865 filas y 593 columnas
# después del preprocesamiento, un hilo, los hiperparámetros que eligió Optuna sin balanceo).

# %% tags=["remove-cell"]
import json

import numpy as np
import pandas as pd

from comun import datos, mostrar, tabla
from src import plots

R = datos.RAIZ / "resultados"

# %% [markdown]
# ## El modelo: una suma de árboles
#
# XGBoost predice el logaritmo de la razón de odds de llegar tarde como una suma de $M$ árboles de regresión,
#
# $$
# F(x) = \sum_{m=1}^{M} f_m(x), \qquad \hat p(x) = \sigma\big(F(x)\big) = \frac{1}{1 + e^{-F(x)}},
# $$
#
# donde cada árbol $f_m$ reparte el espacio de las predictoras en $T$ hojas y asigna a cada hoja $j$ un peso $w_j$. Los
# árboles se construyen uno tras otro: el árbol $m$ corrige lo que los $m-1$ anteriores todavía predicen mal. Es
# *boosting*, al contrario del bosque aleatorio, que promedia árboles independientes.
#
# ## La función objetivo
#
# En la iteración $m$ se busca el árbol $f$ que minimiza la pérdida de entrenamiento más una penalización de su
# complejidad:
#
# $$
# \mathcal{L}^{(m)} = \sum_{i=1}^{n} \ell\big(y_i,\; F_{m-1}(x_i) + f(x_i)\big) + \Omega(f), \qquad
# \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 .
# $$
#
# La pérdida es la log-pérdida (la entropía cruzada binaria), $\ell(y, F) = -y \log \sigma(F) - (1-y) \log(1-\sigma(F))$.
# $\gamma$ cobra por cada hoja y $\lambda$ encoge los pesos hacia cero, como la penalización L2 de la regresión logística.
#
# **Aproximación de segundo orden.** XGBoost expande la pérdida en serie de Taylor alrededor de la predicción actual.
# Para la log-pérdida, la primera y la segunda derivada respecto de $F$ son
#
# $$
# g_i = \frac{\partial \ell}{\partial F} = \hat p_i - y_i, \qquad
# h_i = \frac{\partial^2 \ell}{\partial F^2} = \hat p_i\,(1 - \hat p_i),
# $$
#
# así que, sin los términos constantes,
#
# $$
# \mathcal{L}^{(m)} \approx \sum_{i=1}^{n} \Big[g_i f(x_i) + \tfrac{1}{2} h_i f(x_i)^2\Big] + \gamma T +
# \tfrac{1}{2}\lambda \sum_{j} w_j^2 .
# $$
#
# El gradiente $g_i$ es el residuo con signo: positivo si el modelo sobreestima el riesgo de una posición que llegó a
# tiempo. El hessiano $h_i$ es la varianza de Bernoulli: grande cerca de $\hat p = 0{,}5$, donde el modelo duda, y
# pequeño cuando ya está seguro.
#
# **Peso óptimo de cada hoja.** Si $I_j$ son las posiciones que caen en la hoja $j$, con $G_j = \sum_{i \in I_j} g_i$ y
# $H_j = \sum_{i \in I_j} h_i$, la expresión es una parábola en cada $w_j$, y su mínimo es
#
# $$
# w_j^{*} = -\frac{G_j}{H_j + \lambda}, \qquad
# \mathcal{L}^{*} = -\frac{1}{2}\sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T .
# $$
#
# Es un paso de Newton por hoja: el gradiente dividido por la curvatura, encogido por $\lambda$.
#
# **Ganancia de un corte.** Partir una hoja en izquierda (L) y derecha (R) cambia la pérdida en
#
# $$
# \text{ganancia} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} -
# \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma ,
# $$
#
# y el árbol solo corta si la ganancia es positiva. Con el método `hist`, cada predictora se discretiza en a lo sumo 256
# tramos y la ganancia se evalúa solo en los bordes de esos tramos. Así, buscar el mejor corte cuesta un recorrido del
# histograma y no ordenar las filas.
#
# ## Qué controla cada hiperparámetro
#
# | hiperparámetro | en las fórmulas | efecto |
# |---|---|---|
# | `n_estimators` | $M$ | más árboles, más capacidad y más costo |
# | `learning_rate` | $\eta$: se suma $\eta f$ y no $f$ | pasos cortos que necesitan más árboles y generalizan mejor |
# | `max_depth` | limita $T \le 2^{D}$ | profundidad $D$: interacciones de hasta $D$ predictoras |
# | `min_child_weight` | exige $H_j \ge$ ese valor | como $h_i \le 0{,}25$, un valor de 8 obliga a unas 32 posiciones dudosas por hoja como mínimo |
# | `subsample`, `colsample_bytree` | cada árbol ve una fracción de filas y de columnas | azar que decorrelaciona los árboles, como en el bosque |
# | `reg_lambda`, `gamma` | $\lambda$ y $\gamma$ | se dejaron en sus valores por defecto ($\lambda = 1$, $\gamma = 0$): el espacio de búsqueda ya regula con la profundidad y el peso mínimo |
#
# Lo que eligió Optuna, sin balanceo, en cada bloque:

# %%
filas = []
for b in ["2022-S1", "2022-S2", "2023-S1", "2023-S2", "2024-S1", "2024-S2", "2025-S1", "2025-S2"]:
    r = json.loads((R / "experimentos" / "trabajos" / f"xgboost__ninguno__optuna__{b}.json").read_text(encoding="utf-8"))
    filas.append({"bloque": b, **r["params"], "AUC interno": r["AUC_interno"], "AUC del bloque": r["AUC"]})
tabla(pd.DataFrame(filas), dec=3)

# %% [markdown]
# Con un solo semestre de entrenamiento (2022-S1) gana un modelo de muchos árboles poco profundos: 587 de profundidad
# 4, con tasa 0,02. Desde 2022-S2 la profundidad sube a 7-10, porque con más datos caben más interacciones, y en los
# cuatro últimos bloques el submuestreo de filas baja a 0,52-0,73. Los AUC internos de las configuraciones cercanas difieren en milésimas: la superficie es plana cerca del
# óptimo, y por eso los cuatro optimizadores terminaron tan cerca.
#
# ## Complejidad: teoría y medición
#
# Notación: $n$ filas de entrenamiento, $d = 593$ columnas, $k = 30$ componentes principales en KNN, $E$ épocas de SGD,
# $B$ árboles del bosque con $m$ columnas por corte, $M$ árboles de XGBoost de profundidad $D$.
#
# | modelo | ajuste | predicción de una fila |
# |---|---|---|
# | KNN (PCA 30) | $O(n\,d\,k)$ (la PCA); los vecinos no se calculan al ajustar | $O(n\,k)$: compara contra todo el entrenamiento |
# | Naive Bayes | $O(n\,d)$: una media y una varianza por clase y columna | $O(d)$ |
# | Logística y SVM (SGD) | $O(E\,n\,d)$ | $O(d)$ |
# | Árbol | $O(d\,n \log n)$ por nivel de profundidad | $O(D)$ |
# | Random Forest | $O(B\,m\,n \log n)$ por nivel | $O(B\,D)$ |
# | XGBoost (`hist`) | $O(M\,D\,n\,d')$ para los histogramas, con $d' = d\cdot$`colsample` | $O(M\,D)$ |

# %%
c = pd.read_csv(R / "complejidad.csv")
expo = []
for m, g in c.groupby("modelo"):
    expo.append({"modelo": plots.MODELO[m],
                 "exponente del ajuste": np.polyfit(np.log(g.filas), np.log(g.segundos_ajuste), 1)[0],
                 "exponente de la inferencia": np.polyfit(np.log(g.filas), np.log(g.ms_inferencia_por_1000), 1)[0],
                 "s de ajuste con 311.865 filas": g.segundos_ajuste.iloc[-1],
                 "ms por 1.000 predicciones": g.ms_inferencia_por_1000.iloc[-1]})
expo = pd.DataFrame(expo).sort_values("s de ajuste con 311.865 filas", ascending=False)
tabla(expo, dec=2)

# %%
import plotly.graph_objects as go  # noqa: E402

fig = go.Figure()
for m, g in c.groupby("modelo"):
    fig.add_trace(go.Scatter(x=g.filas, y=g.segundos_ajuste, mode="lines+markers", name=plots.MODELO[m],
                             line=dict(color=plots.COLOR_MOD[m]),
                             hovertemplate="%{x:,} filas: %{y:.1f} s<extra>" + plots.MODELO[m] + "</extra>"))
fig.update_layout(title="Tiempo de ajuste contra filas de entrenamiento (escala log-log; la pendiente es el exponente)",
                  xaxis_title="filas de entrenamiento", yaxis_title="segundos (un hilo)", xaxis_type="log",
                  yaxis_type="log")
mostrar(fig)

# %% [markdown]
# El exponente es la pendiente de la recta en escala log-log: 1 quiere decir que el doble de filas cuesta el doble de
# tiempo.
#
# - **El ajuste crece casi linealmente en todos** (exponentes entre 0,7 y 1,2), como predice la teoría para $n$ grande.
#   El árbol es el más alto (1,2): el factor $\log n$ y que, con el mismo mínimo de filas por hoja, más datos permiten
#   árboles más llenos. La logística y el SVM quedan por debajo de 1 porque SGD se detiene antes con más datos (cada
#   época ya da suficientes pasos).
# - **XGBoost y el bosque son los caros** (162 y 129 segundos con todas las filas), uno por los 344 árboles secuenciales
#   de profundidad 10 y el otro por los 100 árboles completos.
# - **En inferencia la teoría se ve clara.** KNN es el único cuyo costo de predecir crece con el entrenamiento: 1,6
#   segundos por cada mil posiciones con todas las filas, porque compara cada posición con todas las del entrenamiento.
#   Los demás predicen en tiempo constante: XGBoost recorre 344 caminos de profundidad 10 (17 ms por mil) y los lineales
#   hacen un producto punto (menos de 1 ms). Para un tablero que predice una orden al vuelo, cualquiera sirve menos KNN.
