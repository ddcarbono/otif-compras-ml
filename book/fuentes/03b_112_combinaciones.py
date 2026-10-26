# %% [markdown]
# # Las 112 combinaciones
#
# Siete modelos, cuatro formas de tratar el desbalance y cuatro optimizadores de hiperparámetros dan 112 combinaciones.
# Cada una se evalúa en los ocho bloques semestrales de la validación anidada: 896 trabajos, que corrieron en un
# servidor de 32 núcleos del 22 al 23 de octubre de 2026 (6,7 horas de reloj, 200,6 horas de CPU, 30 trabajos a la vez,
# ningún error). El código está en `src/validacion.py`, `src/optimizadores.py` y `src/experimentos.py`; el resumen, en
# `src/comparar.py`. La prueba 2026 solo aparece al final, para justificar la elección frente a la logística.

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
# **El preprocesamiento** es el del capítulo {doc}`02c_tiempo_y_fuga` (logaritmo con signo, imputación, estandarización, *one-hot* y
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
# ## La semilla y la convergencia
#
# Todo lo anterior usa una semilla, la del proyecto (2026). La mejor combinación de XGBoost, el bosque, la logística y el
# SVM se reajustó con diez semillas más en cada bloque, con los mismos hiperparámetros (`src/estabilidad.py`, 320
# ajustes en el servidor). Solo cambia la semilla del modelo; el remuestreo SMOTE del bosque es el de la caché.

# %%
E = datos.RAIZ / "resultados" / "estabilidad"
sem = pd.read_csv(E / "semillas.csv")
mostrar(plots.semillas(sem, plots.MODELO))
med = sem.groupby(["modelo", "semilla"]).AUC.mean().unstack(0)
tabla(legible(med.agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={
    "mean": "AUC medio", "std": "desviación entre semillas", "min": "mínimo", "max": "máximo"})), dec=4)

# %% [markdown]
# El AUC medio de cada modelo se mueve 0,0007 entre semillas (desviación estándar), y dentro de un bloque, 0,002. La
# ventaja de XGBoost sobre el bosque va de 0,008 a 0,011 según la semilla, y XGBoost le gana en siete u ocho bloques con
# cada una: la ventaja media, 0,010, es unas catorce veces lo que mueve la semilla. Con las diez, XGBoost queda primero
# y el bosque segundo; la logística y el SVM, separados por 0,001, se alternan el tercer puesto.
#
# La **curva de convergencia** reajusta XGBoost en cada bloque y anota el AUC y la log-loss (la pérdida que minimiza el
# boosting: el promedio de −log de la probabilidad que le dio a lo que pasó) en el entrenamiento y en el semestre externo
# después de cada árbol. El número de árboles es el que eligió Optuna en cada bloque, de 266 a 587.

# %%
conv = pd.read_csv(E / "convergencia.csv")
mostrar(plots.convergencia(conv))
fin = conv[conv.arbol == conv.groupby("bloque").arbol.transform("max")].set_index("bloque")
mejor_auc = conv.loc[conv.groupby("bloque").AUC_externo.idxmax()].set_index("bloque")
mejor_ll = conv.loc[conv.groupby("bloque").logloss_externo.idxmin()].set_index("bloque")
tabla(pd.DataFrame({"árboles": fin.arbol, "AUC entrenamiento": fin.AUC_entrenamiento, "AUC externo": fin.AUC_externo,
                    "brecha": fin.AUC_entrenamiento - fin.AUC_externo, "árbol del mejor AUC externo": mejor_auc.arbol,
                    "AUC que se deja": mejor_auc.AUC_externo - fin.AUC_externo,
                    "árbol de la menor log-loss externa": mejor_ll.arbol}).reset_index(), dec=4,
      enteros=["árboles", "árbol del mejor AUC externo", "árbol de la menor log-loss externa"])

# %% [markdown]
# Al final, el AUC de entrenamiento queda entre 0,92 y 0,99 y el externo entre 0,83 y 0,89: una brecha media de 0,12.
# Hay sobreajuste, pero casi no le cuesta orden al modelo. El mejor AUC externo llega antes del último árbol en siete
# bloques, y lo que se deja es de 0,0014 en promedio (0,004-0,005 en 2023-S1 y 2025-S1). La log-loss externa, en cambio,
# toca su mínimo mucho antes (entre 55 y 501 árboles) y luego sube. Los árboles de más siguen ordenando igual de bien,
# pero empujan las probabilidades hacia 0 y 1. Es la sobreconfianza que se ve en la curva de fiabilidad de la sección
# siguiente, y es lo que corrige Platt. Una parada temprana por log-loss daría mejores probabilidades crudas; por AUC
# no cambiaría casi nada.
#
# ## XGBoost en validación: calibración y errores
#
# Antes de la prueba 2026 se puede mirar cómo se equivoca el ganador en los ocho bloques, con sus predicciones y el
# umbral de máximo F1 que cada trabajo fijó en su validación interna: aquí no se elige ningún umbral
# (`src/diagnostico_validacion.py`).

# %%
D = datos.RAIZ / "resultados" / "diagnostico"
tabla(pd.read_csv(D / "por_bloque.csv").rename(columns={"precision": "precisión"})[
    ["bloque", "filas", "prevalencia", "umbral", "AUC", "F1", "precisión", "sensibilidad", "Brier", "ECE"]],
    dec=3, enteros=["filas"])
fia = pd.read_csv(D / "fiabilidad.csv").rename(columns={"filas": "posiciones"})
mostrar(plots.calibracion(fia, "Calibración de XGBoost en los 8 bloques de validación (deciles)", "XGBoost"))

# %% [markdown]
# El ECE (error de calibración esperado: la distancia media, ponderada por filas, entre la probabilidad predicha y la
# tasa observada en cada decil) va de 0,018 a 0,068 según el bloque, con media 0,040. La logística en la prueba 2026 dio
# 0,021. La curva muestra la forma del error: en los deciles bajos XGBoost predice menos de lo que pasa (1 % contra 4 %
# en el primero) y en los altos un poco más (74 % contra 69 % en el noveno). Exagera en los extremos, que es lo que
# corrige el escalado de Platt: una regresión logística sobre el logit del puntaje, con pendiente menor que 1.
#
# Para decidir si calibrar sin mirar 2026, Platt se ajusta en un bloque y se aplica al siguiente, como se haría con 2025
# y 2026:

# %%
pl = pd.read_csv(D / "platt.csv")
tabla(pl.rename(columns={"ajuste": "Platt ajustado en", "bloque": "aplicado a", "ECE_crudo": "ECE sin calibrar",
                         "ECE_platt": "ECE con Platt", "Brier_crudo": "Brier sin calibrar",
                         "Brier_platt": "Brier con Platt"}), dec=4)

# %% [markdown]
# La evidencia es mixta. La pendiente sale siempre bajo 1 (de 0,61 a 0,95), como se esperaba, pero Platt baja el ECE en
# tres de los siete pares y lo sube en cuatro; el Brier cambia en la tercera cifra. Los dos pares más recientes son los
# que más ganan (2025-S1: de 0,055 a 0,030; 2025-S2: de 0,046 a 0,029), y son los que más se parecen al paso de 2025 a
# 2026. Por eso el modelo que se lleva a la prueba (capítulo siguiente) lleva Platt ajustado con toda la validación
# 2025, congelado antes de mirar 2026.
#
# **Dónde se equivoca.** Por línea de negocio, con los ocho bloques juntos y cada fila al umbral de su bloque:

# %%
tabla(pd.read_csv(D / "por_linea.csv").rename(columns={"linea": "línea", "precision": "precisión"}), dec=3,
      enteros=["filas"])

# %% [markdown]
# Ordena mejor en puerto (AUC 0,897) y peor en ganadería (0,823), que además es la peor calibrada (ECE 0,083) y la de
# menos filas. Palma, con el 43 % de las posiciones de estos semestres, queda en 0,836.
#
# **Proveedores nuevos.** Un proveedor es nuevo para un bloque si no aparece en el entrenamiento de ese bloque: el modelo
# solo sabe de él lo que dicen su región, su grupo de cuentas y su sector.

# %%
nv = pd.read_csv(D / "nuevos.csv")
tabla(nv[nv.bloque == "todos"].drop(columns="bloque").rename(columns={"proveedor_tipo": "proveedor",
                                                                      "precision": "precisión"}), dec=3,
      enteros=["filas"])

# %% [markdown]
# Son el 3,9 % de las posiciones (11.665 de 296.969) y llegan tarde más seguido (40 % contra 31 %). El modelo los
# ordena peor (AUC 0,806 contra 0,852) y acierta menos cuando los marca (precisión 0,594 contra 0,672). En la prueba
# pesará más: una de cada nueve posiciones de 2026 trae un proveedor que no estaba en 2021-2024 (capítulo {doc}`02c_tiempo_y_fuga`), contra
# una de cada 26 aquí.
#
# **El perfil de los errores.** Mediana de algunas variables por tipo de caso:

# %%
pe = pd.read_csv(D / "perfil_errores.csv").set_index("caso").loc[
    ["verdadero positivo", "falso negativo", "falso positivo", "verdadero negativo"]].reset_index()
tabla(pe.rename(columns={"puntaje_mediano": "puntaje", "proveedor_nuevo": "proveedor nuevo",
                         "prov_tasa_tarde_previa_mediana": "tasa previa del proveedor",
                         "prov_n_previas_mediana": "posiciones previas del proveedor",
                         "plazo_habil_al_liberar_mediana": "días hábiles de plazo al liberar",
                         "dias_habiles_liberacion_mediana": "días hábiles hasta liberar"}),
      dec=3, pct=["proveedor nuevo"], enteros=["filas"])

# %% [markdown]
# Un **falso negativo típico** es una posición que parecía segura: le quedaban 13 días hábiles de plazo al liberarse
# (contra 2 en los verdaderos positivos), su proveedor tenía una tasa previa de retraso de 0,31 (contra 0,45) y un
# historial largo, y la orden se liberó rápido. El modelo le dio 0,16. Son retrasos que no anuncian las variables
# disponibles al liberar la orden: se deciden después, en la ejecución del proveedor. Un **falso positivo típico** es lo
# contrario: plazo corto (9 días) y un proveedor con historial de retraso (0,44), que esa vez cumplió.
#
# ## Qué deja este capítulo
#
# El modelo es lo que decide, no el balanceo ni el optimizador. **XGBoost sin balanceo** queda primero, con ROC-AUC
# 0,852 ± 0,020 y F1 0,676 ± 0,048 en los ocho bloques, y le gana a cada uno de los otros seis modelos. Frente a los
# tres más cercanos (bosque, logística, SVM) la ventaja es de 0,009 a 0,013 de AUC, con un intervalo BCa por órdenes que
# no toca el cero y un DeLong significativo en seis o siete de los ocho bloques; en el resto, empate. El
# balanceo no aporta, y con 20 evaluaciones Optuna y el genético llegan más rápido a lo mismo que la rejilla y la búsqueda
# aleatoria. El capítulo siguiente lo mide en la prueba 2026 y lo explica con SHAP.
#
# **Por qué XGBoost y no la logística.** La logística es más simple y se lee coeficiente por coeficiente; la elección se
# decide por lo que pasa en la operación. Un retraso y una falsa alarma no cuestan lo mismo: la alerta de más es una
# llamada al proveedor, la tardía que no se vio puede dejar una planta esperando el material. Con el mismo esfuerzo de
# revisión, el 20 % de las posiciones de mayor riesgo en la prueba 2026 (la medida del capítulo siguiente), XGBoost atrapa
# más tardías, y una parte son de material que el mes anterior tenía el stock en cero:

# %%
cr = pd.read_csv(datos.RAIZ / "resultados" / "prueba_2026" / "criticas.csv").set_index("cifra")
dos = lambda c: f"{cr.loc[c, 'total']:,.0f} [{cr.loc[c, 'BCa inf']:,.0f}; {cr.loc[c, 'BCa sup']:,.0f}]".replace(",", ".")
tabla(pd.DataFrame({
    "": ["Tardías atrapadas", "De ellas, con stock en cero"],
    "XGBoost": [cr.loc["XGBoost: tardías atrapadas", "total"], cr.loc["XGBoost: críticas atrapadas", "total"]],
    "Logística": [cr.loc["logística: tardías atrapadas", "total"], cr.loc["logística: críticas atrapadas", "total"]],
    "Diferencia [BCa 95 %]": [dos("N: tardías de más"), dos("M: de ellas, con stock en cero")],
    "Diferencia por mes": [cr.loc["N: tardías de más", "por mes"], cr.loc["M: de ellas, con stock en cero", "por mes"]]}),
    dec=1, enteros=["XGBoost", "Logística"])

# %% [markdown]
# Son 347 tardías más en los diez meses, unas 35 por mes, y 166 de ellas (unas 17 por mes) de material sin stock. El
# intervalo por órdenes de la primera no toca el cero; el de la segunda sí lo roza (−3), así que esa cifra es
# indicativa. Esas tardías de más caen sobre todo en órdenes que ya tenían otra alerta (2.362 órdenes con una tardía
# atrapada contra 2.407): la ventaja es por posición, que es lo que se recibe. La «crítica» se tomó del stock porque
# las otras marcas posibles no sirven: la pieza crítica es constante en los datos y la razón cantidad/stock está vacía
# en el 73 % de 2026. Ganar esas posiciones va en la dirección que busca el área, que compras responda a la planeación
# y no a la urgencia. No hay datos del costo de una parada por falta de material, así que la ventaja no se lleva a
# pesos. Y entrenar no es una restricción: 3,2 minutos, que permiten reentrenar cada mes.
#
# En los números de la validación: XGBoost gana a la logística en los ocho bloques (Wilcoxon con Holm, p = 0,047), con
# delta de Cliff 0,34 (mediano), y en 2026 por 0,023 de AUC [0,017; 0,030], DeLong z = 22,5.
