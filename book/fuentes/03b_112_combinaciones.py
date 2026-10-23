# %% [markdown]
# # Las 112 combinaciones
#
# Siete modelos, cuatro formas de tratar el desbalance y cuatro optimizadores de hiperparámetros dan 112 combinaciones.
# Cada una se evalúa en los ocho bloques semestrales de la validación anidada: 896 trabajos, que corrieron en un
# servidor de 32 núcleos del 22 al 23 de octubre de 2026 (6,7 horas de reloj, 200,6 horas de CPU, 30 trabajos a la vez,
# ningún error). El código está en `src/validacion.py`, `src/optimizadores.py` y `src/experimentos.py`; el resumen, en
# `src/comparar.py`. La prueba 2026 no se usa en este capítulo.

# %% tags=["remove-cell"]
import pandas as pd

from comun import datos, mostrar, tabla
from src import plots
from src.plots import BALANCEO, MODELO, OPTIMIZADOR

R = datos.RAIZ / "resultados" / "experimentos"
t = pd.read_csv(R / "lideres.csv")
df = pd.read_csv(R / "por_bloque.csv")


def legible(d):
    d = d.copy()
    for c, m in [("modelo", MODELO), ("balanceo", BALANCEO), ("optimizador", OPTIMIZADOR), ("contra", MODELO),
                 ("mejor", MODELO)]:
        if c in d:
            d[c] = d[c].map(m)
    return d


# %% [markdown]
# ## Diseño
#
# **Validación anidada.** El bucle externo tiene ocho bloques de evaluación, de 2022-S1 a 2025-S2; cada uno entrena con
# todos los semestres anteriores (ventana creciente) y evalúa en el suyo. Dentro de la ventana de entrenamiento de cada
# bloque, el bucle interno separa el último semestre como validación: ahí el optimizador elige los hiperparámetros por
# ROC-AUC, y con las puntuaciones de la mejor configuración se fija el umbral de máximo F1. Después se reajusta con toda la
# ventana y se mide en el bloque. Como los cortes son por fecha de la orden, ninguna orden queda partida, y el modelo
# siempre predice un semestre que no ha visto, como pasaría en uso.
#
# **El preprocesamiento** es el del capítulo 3 (logaritmo con signo, imputación, estandarización, *one-hot* y
# codificación jerárquica por objetivo fuera de pliegue), ajustado una vez por ventana con su entrenamiento. No tiene
# hiperparámetros, así que su salida se guarda y los 896 trabajos solo ajustan modelos.
#
# **Modelos y espacios de búsqueda.** Para que el diseño cupiera en una noche se tomaron cuatro recortes, medidos con un
# piloto: la logística y el SVM lineal se ajustan por descenso de gradiente estocástico (`SGDClassifier`, pérdidas
# logística y bisagra), KNN trabaja sobre 30 componentes principales y el bosque tiene 100 árboles.
#
# | modelo | hiperparámetros (rango; escala log donde se indica) |
# |---|---|
# | KNN (PCA 30) | vecinos 5-300 (log), ponderación uniforme o por distancia |
# | Naive Bayes gaussiano | suavizado de la varianza 10⁻¹²-10⁻¹ (log) |
# | Logística (SGD) | penalización α 10⁻⁷-10⁻² (log), L1 o L2 |
# | Árbol de decisión | profundidad 3-30, hojas mínimas 5-2.000 (log), Gini o entropía |
# | Random Forest (100 árboles) | profundidad 5-40, hojas mínimas 1-200 (log), variables por corte √p, log₂ p o 5 % |
# | XGBoost | árboles 100-600, tasa de aprendizaje 0,01-0,3 (log), profundidad 3-10, submuestreo de filas 0,5-1 y de columnas 0,3-1, peso mínimo por hoja 1-100 (log) |
# | SVM lineal (SGD) | penalización α 10⁻⁷-10⁻² (log), L1 o L2 |
#
# **Balanceos.** Ninguno; SMOTE y ADASYN, que crean posiciones tardías sintéticas interpolando entre vecinas (ADASYN
# genera más donde la clase minoritaria está rodeada de la mayoritaria), calculados una vez por ventana y solo con su
# entrenamiento; y pesos de clase, que multiplican la pérdida de cada clase por el inverso de su frecuencia.
#
# **Optimizadores**, todos con el mismo presupuesto de **E = 20 evaluaciones**:
#
# - **Grid**: rejilla regular de a lo sumo 20 puntos. Con seis hiperparámetros (XGBoost) solo caben dos niveles por
#   dimensión (16 puntos): es la maldición de la dimensión de la búsqueda en rejilla.
# - **Random**: 20 puntos al azar en el espacio.
# - **Optuna (TPE)**: después de 10 puntos al azar, modela por separado la densidad de las configuraciones buenas, l(x), y
#   la de las malas, g(x), y propone donde l(x)/g(x) es mayor, que equivale a maximizar la mejora esperada.
# - **DEAP (genético)**: población de 6, selección por torneo de 3, cruce uniforme con probabilidad 0,7, mutación
#   gaussiana (σ = 0,15) con probabilidad 0,3 y elitismo de 1. Un hijo idéntico a un individuo ya evaluado no gasta
#   presupuesto.
#
# ## Tabla de líderes
#
# Media y desviación estándar en los ocho bloques externos. El AUC interno es el de la validación donde se eligieron los
# hiperparámetros; las horas de CPU suman los ocho bloques. La tabla completa, ordenable y filtrable, está en la página
# Modelos del tablero.

# %%
cols = ["puesto", "modelo", "balanceo", "optimizador", "AUC", "AUC_desv", "F1", "F1_desv", "PR_AUC", "AUC_interno",
        "horas_cpu", "ms_inferencia_por_1000"]
tabla(legible(t.head(15))[cols], dec=4)

# %% [markdown]
# Las quince primeras son de XGBoost, y entre la primera (0,8517) y la décima (0,8500) hay menos de dos milésimas,
# cuando la desviación entre bloques de cada una es de unas dos centésimas. Dentro de XGBoost, ni el balanceo ni el
# optimizador separan a las combinaciones. La mejor de cada modelo:

# %%
mejor = t.sort_values("AUC", ascending=False).groupby("modelo").head(1)
tabla(legible(mejor)[cols], dec=4)

# %% [markdown]
# ## Qué pesa: el modelo, no el balanceo ni el optimizador

# %%
r = t.groupby("modelo").AUC.agg(["min", "max"]).sort_values("max", ascending=False)
r["rango"] = r["max"] - r["min"]
tabla(r.rename(index=MODELO).reset_index().rename(columns={"min": "AUC mínimo", "max": "AUC máximo",
                                                          "rango": "rango entre sus 16"}), dec=4)
for m in ["xgboost", "logistica", "random_forest"]:
    fig = plots.mapa_combinaciones(t, m, BALANCEO, OPTIMIZADOR)
    fig.update_layout(title=f"{MODELO[m]}: AUC medio por balanceo y optimizador")
    mostrar(fig)

# %% [markdown]
# Entre modelos el AUC va de 0,72 (Naive Bayes) a 0,85 (XGBoost); dentro de un mismo modelo, sus 16 combinaciones de
# balanceo y optimizador no se separan más de 0,012. Tres lecturas:
#
# - **Balancear no mejora el ordenamiento.** Con una prevalencia del 33 % el desbalance es moderado, y el AUC no depende
#   de la proporción de clases. SMOTE y ADASYN cuestan el doble de cómputo (la matriz crece un tercio) y no ganan nada en
#   XGBoost. En la logística, ADASYN baja el AUC unos 0,006: concentra las posiciones sintéticas en la frontera, donde una
#   frontera lineal se deja arrastrar.
# - **En KNN y Naive Bayes, «pesos de clase» y «ninguno» dan exactamente el mismo resultado.** No es un error. En los dos
#   modelos el peso solo cambia la probabilidad a priori de cada clase, y eso es una transformación monótona del
#   puntaje: el orden de las posiciones no cambia (el AUC tampoco) y el umbral de F1 absorbe el corrimiento.
# - **El F1 sí depende un poco del balanceo** en los modelos lineales (0,662 sin balanceo frente a 0,651 con ADASYN en la
#   logística), porque el remuestreo cambia la forma de la frontera y no solo su posición.
#
# ## Comparación entre modelos
#
# Para comparar modelos se toma la combinación de cada uno con mayor **AUC interno** medio. Elegirla por el AUC externo
# sería escoger con el mismo dato que se quiere comparar.

# %%
m = pd.read_csv(R / "auc_por_bloque_modelos.csv")
mostrar(plots.auc_por_bloque(m, MODELO))

# %% [markdown]
# Todos los modelos suben y bajan juntos: 2024-S2 es el semestre más fácil para los siete, y 2022-S2 el más difícil para
# seis (para el árbol, 2025-S2). La
# variación entre bloques (de 0,82 a 0,89 en XGBoost) es mayor que la diferencia entre los mejores modelos, y por eso la
# comparación tiene que ser pareada por bloque.
#
# La **prueba de Friedman** ordena los modelos dentro de cada bloque (rango 1 al mejor) y contrasta si los rangos medios
# difieren más de lo que daría el azar. No supone normalidad ni varianzas iguales, y compara en el mismo bloque, que es lo
# que se necesita con ocho mediciones pareadas. La corrección de Iman y Davenport convierte el estadístico en una F, menos
# conservadora.

# %%
fr = pd.read_csv(R / "friedman.csv")
tabla(fr.rename(columns={"k": "modelos", "chi2": "χ² de Friedman", "p": "p", "F_iman": "F (Iman-Davenport)",
                         "p_iman": "p de la F", "CD": "diferencia crítica"}), dec=4)
rg = pd.read_csv(R / "rangos.csv")
mostrar(plots.rangos_cd(rg, fr.CD.iloc[0], MODELO))

# %% [markdown]
# Los rangos difieren (χ² = 45,6, p < 0,001). XGBoost queda primero en los ocho bloques (rango medio 1,00). La **prueba de
# Nemenyi** compara todos los pares a la vez: dos modelos difieren si sus rangos medios se separan más que la diferencia
# crítica, 3,19 con siete modelos y ocho bloques. Con ella, XGBoost se separa del árbol, KNN y Naive Bayes, pero no de la
# logística, el bosque ni el SVM (franja sombreada).
#
# Nemenyi corrige por las 21 comparaciones posibles; si la pregunta es solo si el mejor le gana a cada uno, son seis.
# La **prueba de Wilcoxon de rangos con signo** sobre las diferencias pareadas por bloque, con la corrección de Holm por
# las seis comparaciones, es más potente para esa pregunta:

# %%
tabla(legible(pd.read_csv(R / "wilcoxon.csv")).rename(columns={
    "diferencia_media": "diferencia media de AUC", "bloques_gana": "bloques que gana", "p_holm": "p (Holm)"}), dec=4)

# %% [markdown]
# XGBoost le gana a cada modelo en los ocho bloques, y la diferencia frente a los segundos es de 0,011 a 0,014 de AUC. Con
# ocho pares, el p exacto más pequeño que puede dar Wilcoxon es 2/2⁸ = 0,0078; tras Holm queda en 0,047. La diferencia
# es significativa al 5 %, pero en el límite de lo que ocho bloques pueden mostrar. Para decir más no hacen falta más
# bloques: hace falta mirar dentro de cada uno, con las predicciones y no solo con el AUC.
#
# ### DeLong, delta de Cliff e intervalos BCa
#
# Friedman y Wilcoxon ven ocho números por modelo. Cada AUC sale de 33.000 a 42.000 posiciones, y con las
# predicciones se puede preguntar en cada bloque si la diferencia supera lo que daría el azar. Para eso se reajustó la
# mejor combinación de cada modelo de la tabla de líderes en cada bloque, con los hiperparámetros que ya había elegido
# su optimizador (`src/puntajes.py`; 56 ajustes, cada uno con el mismo AUC que su trabajo hasta 1e-5). Las combinaciones
# son las de mayor AUC externo medio, no las de mayor AUC interno que usó Friedman, y por eso XGBoost entra con Optuna
# y no con el genético (0,852 frente a 0,851); en los demás modelos cambia el balanceo o el optimizador, y las medias se
# mueven en la tercera cifra.
#
# - **DeLong** (DeLong, DeLong y Clarke-Pearson, 1988) compara dos AUC medidos sobre las mismas filas. El AUC es la
#   proporción de pares (tardía, puntual) que el modelo ordena bien; DeLong estima su varianza y la covarianza entre los
#   dos modelos a partir de la contribución de cada fila a esa proporción, y da un z para la diferencia. La
#   implementación (`src/pruebas.py`, algoritmo de Sun y Xu, 2014) se validó contra `pROC::roc.test` de R: coincide
#   hasta 1e-10 en seis casos sintéticos con empates y sin ellos (`tests/test_pruebas.py`) y en un bloque real (abajo).
# - **El delta de Cliff** mide el tamaño de una diferencia sin suponer distribución: la proporción de pares (bloque de
#   XGBoost, bloque del rival) en que gana XGBoost menos la proporción en que pierde. Va de −1 a 1; por la escala de
#   Romano y otros (2006), es despreciable bajo 0,147, pequeño bajo 0,33, mediano bajo 0,474 y grande desde ahí. Compara
#   las dos colecciones de ocho AUC sin parearlas, y por eso un 7 de 8 pareado puede quedar en un delta pequeño.
# - **El intervalo BCa** (bootstrap corregido por sesgo y acelerado, Efron, 1987) toma los percentiles del bootstrap y
#   los corrige por dos cosas: el sesgo de la mediana bootstrap frente a la estimación y la asimetría con que cambia el
#   error estándar (la aceleración, estimada con jackknife). Aquí se remuestrean **órdenes** dentro de cada bloque
#   (1.000 réplicas), porque las posiciones de una misma orden no son independientes.
#
# Esa última frase es también el límite de DeLong: supone filas independientes y, con posiciones agrupadas por orden,
# su error estándar queda corto. Por eso el p de DeLong de cada bloque se lee junto con el intervalo BCa por órdenes,
# que sí lo tiene en cuenta.

# %%
P = datos.RAIZ / "resultados" / "pruebas"
v = pd.read_csv(P / "validacion_delong.csv")
tabla(v.rename(columns={"fuente": "implementación", "AUC_1": "AUC XGBoost", "AUC_2": "AUC logística"}), dec=10)

# %% [markdown]
# En el bloque 2025-S2 (XGBoost contra la logística, unas 42.000 posiciones) la implementación propia y la de R dan los
# mismos AUC, el mismo z y el mismo p hasta la décima cifra.
#
# Las 48 pruebas de DeLong (seis rivales por ocho bloques) llevan la corrección de Holm. El resumen por rival:

# %%
rs = pd.read_csv(P / "resumen.csv")
rs = rs.assign(intervalo=[f"[{a:.4f}; {b:.4f}]".replace(".", ",") for a, b in zip(rs.dif_inf, rs.dif_sup)])
tabla(legible(rs.rename(columns={"rival": "modelo"}))[
    ["modelo", "bloques_gana", "bloques_delong_holm", "dif_AUC", "intervalo", "cliff", "magnitud"]].rename(columns={
        "bloques_gana": "bloques que gana XGBoost", "bloques_delong_holm": "bloques con DeLong significativo (Holm, 5 %)",
        "dif_AUC": "diferencia media de AUC", "intervalo": "BCa 95 %", "cliff": "δ de Cliff"}), dec=4)

# %% [markdown]
# - **Frente al árbol, KNN y Naive Bayes** no hay discusión: XGBoost gana en los ocho bloques, DeLong lo confirma en
#   los ocho (z de 19 a 32 en el peor bloque de cada uno), el intervalo BCa de la diferencia queda lejos de cero y el
#   delta de Cliff es grande.
# - **Frente al bosque, la logística y el SVM** la ventaja es real pero chica. La diferencia media es de 0,009 a 0,013,
#   y ningún intervalo BCa toca el cero: el más bajo, el del bosque, empieza en 0,006. DeLong la da por
#   significativa en seis bloques frente al bosque y en siete frente a los otros dos. Donde no, XGBoost empata: con la logística y el SVM en 2022-S1 (−0,001 y +0,001), y con el
#   bosque en 2024-S1 (+0,001) y 2025-S2 (−0,0004). En ningún bloque un rival le gana a XGBoost de forma significativa.
# - **El delta de Cliff** con esos tres es pequeño (bosque, 0,25) o mediano (logística 0,34, SVM 0,38). La diferencia
#   pareada es consistente, pero menor que la variación entre semestres: el peor bloque de XGBoost queda por debajo del
#   mejor de cualquiera de los tres.

# %%
ab = pd.read_csv(P / "auc_bca.csv")
ab = ab.assign(intervalo=[f"[{a:.4f}; {b:.4f}]".replace(".", ",") for a, b in zip(ab["inf"], ab["sup"])])
tabla(legible(ab)[["modelo", "AUC", "intervalo"]].rename(columns={"AUC": "AUC medio (8 bloques)",
                                                                   "intervalo": "BCa 95 %"}), dec=4)

# %% [markdown]
# El intervalo del AUC medio de XGBoost, [0,844; 0,857], apenas se cruza con el del bosque, [0,836; 0,848]. Los
# intervalos de cada modelo por separado se solapan más que el de la diferencia, porque la diferencia es pareada: los
# dos modelos suben y bajan juntos con el bloque, y esa variación se cancela al restar. Qué tan lejos llega esa ventaja
# en un año que ningún modelo vio se mide en la prueba 2026.
#
# ## Los optimizadores
#
# Para comparar optimizadores entre modelos distintos se usa el **arrepentimiento**: cuánto AUC interno le falta a lo
# mejor encontrado hasta la evaluación n, frente a lo mejor que encontró cualquiera de los cuatro con el mismo modelo,
# balanceo y bloque. Cero es encontrar el óptimo conocido.

# %%
mostrar(plots.arrepentimiento(pd.read_csv(R / "curvas_optimizador.csv"), "eval", OPTIMIZADOR,
                              "Arrepentimiento medio contra número de evaluaciones (los 7 modelos)", "evaluación"))
mostrar(plots.arrepentimiento(pd.read_csv(R / "curvas_optimizador_tiempo.csv"), "segundos", OPTIMIZADOR,
                              "Arrepentimiento medio contra tiempo de reloj (XGBoost)", "segundos desde el inicio"))

# %% [markdown]
# - **Optuna** es el más rápido en acercarse: en la evaluación 5 su arrepentimiento medio es 0,006, contra 0,013 del
#   genético, 0,015 de la búsqueda aleatoria y 0,042 de la rejilla. Al final (evaluación 20) queda en 0,0011.
# - **El genético** arranca como la búsqueda aleatoria (su primera generación es al azar) y termina casi igual que Optuna
#   (0,0013). En XGBoost, contra el tiempo de reloj, es el que menos arrepentimiento deja al final. Su población se va
#   hacia submuestreos de columnas bajos (0,48 de media, frente a 0,59-0,67 de los demás), una región que la rejilla de
#   dos niveles y el azar visitan menos.
# - **La rejilla** es la peor hasta el final, porque recorre el espacio en un orden fijo: los mejores puntos pueden caer en
#   la última fila. Con E = 20 termina en 0,0034, como la búsqueda aleatoria (0,0030).
#
# Todo eso es en el AUC interno, que es lo que optimizan. En el AUC externo las diferencias se diluyen, porque son de
# milésimas y la variación entre bloques es de centésimas. Para este problema, con 20 evaluaciones cualquiera de los cuatro
# deja el modelo a menos de 0,004 de su óptimo.
#
# ## Costo

# %%
mostrar(plots.costo(df, MODELO, BALANCEO))

# %% [markdown]
# XGBoost y el bosque se llevan dos tercios de las 200,6 horas de CPU (68 y 67), y SMOTE y ADASYN duplican el costo de
# cada modelo que los usa (en XGBoost, 23-24 horas contra 11 sin balanceo). En inferencia, la diferencia que importa en
# uso es la de KNN: tarda 1,3 segundos por cada mil posiciones, porque tiene que buscar vecinos entre todo el
# entrenamiento. XGBoost tarda 16 milisegundos y los lineales menos de uno.
#
# ## Qué deja este capítulo
#
# El modelo es lo que decide, no el balanceo ni el optimizador. **XGBoost sin balanceo** queda primero, con ROC-AUC
# 0,852 ± 0,020 y F1 0,676 ± 0,048 en los ocho bloques, y le gana a cada uno de los otros seis modelos. Frente a los
# tres más cercanos (bosque, logística, SVM) la ventaja es de 0,009 a 0,013 de AUC, con un intervalo BCa por órdenes que
# no toca el cero y un DeLong significativo en seis o siete de los ocho bloques; en el resto, empate. El
# balanceo no aporta, y con 20 evaluaciones Optuna y el genético llegan más rápido a lo mismo que la rejilla y la búsqueda
# aleatoria. Falta evaluar ese modelo una sola vez en la prueba 2026 (matriz de confusión, ROC, PR, calibración) y
# explicarlo con SHAP.
