# Glosario

Cada entrada dice qué mide o hace el estadístico o el método, cómo se lee y dónde aparece en el proyecto, con la cifra
real cuando la hay. Los términos van en orden alfabético. Las cifras salen de los capítulos del libro y de
`resultados/`; lo que depende de la evaluación final en la prueba 2026 está marcado como pendiente. Notación común:
$y \in \{0, 1\}$ es el desenlace (1 = la posición llega tarde), $x$ el vector de predictoras, $p(x) = P(y = 1 \mid x)$ la
probabilidad estimada y $\tau$ el umbral de decisión.

## A

```{glossary}
ACF y PACF
  Autocorrelación y autocorrelación parcial de una serie: la correlación entre la serie y ella misma desplazada $h$
  rezagos, sin y con descontar los rezagos intermedios.

  *Cómo se lee.* Una ACF que decae despacio indica tendencia o no estacionariedad; picos de la PACF en pocos rezagos
  indican qué rezagos aportan información propia.

  *En el proyecto* ({doc}`02c_tiempo_y_fuga`). Sobre la tasa mensual de retraso, la ACF vale 0,76 en el primer rezago y
  0,52 en el sexto, y la PACF tiene picos en los rezagos 1, 2 y 4: es la huella de la tendencia, no de un ciclo corto.

ADASYN
  Variante de SMOTE que genera más posiciones sintéticas de la clase minoritaria donde esta está rodeada por la
  mayoritaria (las zonas más difíciles). Ver *SMOTE*.

  *En el proyecto* ({doc}`03b_112_combinaciones`). Se calcula una vez por ventana y solo con el entrenamiento. Es el
  balanceo de peor desempeño: en la logística baja el AUC unos 0,006, y duplica el costo de cómputo.

ADF (Dickey-Fuller aumentada)
  Prueba de estacionariedad cuya hipótesis nula es que la serie tiene una raíz unitaria (no vuelve a su media). Un p
  bajo rechaza la raíz unitaria, es decir, sugiere estacionariedad. Se lee junto con *KPSS*, que tiene la hipótesis
  contraria.

  *En el proyecto* ({doc}`02c_tiempo_y_fuga`). Sobre la tasa mensual 2021-2024, p = 0,70: no rechaza la raíz unitaria.

Algoritmo genético
  Optimizador inspirado en la evolución: mantiene una población de configuraciones, elige a las mejores por *torneo*,
  las combina por *cruce*, las perturba por *mutación* y conserva siempre a la mejor (*elitismo*). No usa gradientes ni
  supone que la función sea suave.

  *En el proyecto* ({doc}`03b_112_combinaciones`, implementado con DEAP). Población de 6, torneo de 3, cruce uniforme
  con probabilidad 0,7, mutación gaussiana ($\sigma = 0{,}15$) con probabilidad 0,3, elitismo de 1; un hijo idéntico a
  uno ya evaluado no gasta presupuesto. Su primera generación es al azar, por eso arranca como la búsqueda aleatoria;
  al final de las 20 evaluaciones deja un arrepentimiento de 0,0013, casi igual que Optuna.

Árbol de decisión
  Modelo que parte el espacio de predictoras con preguntas del tipo «¿variable $\le$ valor?», eligiendo en cada nodo
  el corte que más reduce la impureza (Gini $= 1 - \sum_k p_k^2$ o entropía $= -\sum_k p_k \ln p_k$). La predicción de
  una hoja es la proporción de la clase en ella. Varianza alta: cambia mucho con los datos; por eso se limita con la
  profundidad y las hojas mínimas.

  *En el proyecto.* Es el base de Random Forest y XGBoost. Solo, queda en AUC 0,810 (mejor combinación, media de ocho
  bloques) y se separa de XGBoost según Nemenyi.

Arranque en frío
  Situación en que un proveedor, material o persona aparece por primera vez y no tiene historial. El modelo no puede
  usar su tasa propia y recurre a la de su nivel superior.

  *En el proyecto* ({doc}`02c_tiempo_y_fuga`). En 2026, el 11,1 % de las posiciones trae un proveedor que no estaba en
  2021-2024 y el 9,7 %, un material nuevo. El AUC con entidades nuevas frente a conocidas está pendiente: evaluación
  2026.

Arrepentimiento (de un optimizador)
  Cuánto valor le falta a lo mejor encontrado hasta la evaluación $n$ frente a lo mejor que encontró cualquiera de los
  optimizadores con el mismo modelo, balanceo y bloque. Cero es haber encontrado el óptimo conocido. Permite comparar
  optimizadores entre modelos con escalas distintas.

  *En el proyecto* ({doc}`03b_112_combinaciones`). Medido sobre el AUC interno. En la evaluación 5: Optuna 0,006,
  genético 0,013, aleatoria 0,015, rejilla 0,042. En la 20: Optuna 0,0011, genético 0,0013, aleatoria 0,0030, rejilla
  0,0034.

AUC ROC
  Área bajo la curva ROC (sensibilidad contra 1 − especificidad al variar el umbral). Es la probabilidad de que una
  posición tardía elegida al azar reciba un puntaje mayor que una a tiempo elegida al azar (los empates cuentan 0,5):

  $$
  \mathrm{AUC} = P(s(x^{+}) > s(x^{-})) + \tfrac{1}{2}\,P(s(x^{+}) = s(x^{-})).
  $$

  *Cómo se lee.* 0,5 es azar y 1 es orden perfecto. Mide ordenamiento, no depende del umbral ni de la proporción de
  clases.

  *En el proyecto.* Es la métrica con la que los optimizadores eligen hiperparámetros y con la que se comparan los
  modelos. Regla del proveedor 0,670 [0,654; 0,686] y regresión logística 0,825 [0,813; 0,838] en la prueba 2026;
  XGBoost sin balanceo, 0,852 ± 0,020 en los ocho bloques de la validación anidada. El AUC univariado más alto de una
  variable suelta es 0,732 (plazo hábil al liberar). XGBoost en la prueba 2026: pendiente (evaluación 2026).
```

## B a D

```{glossary}
Balanceo de clases
  Conjunto de técnicas para compensar una clase minoritaria: remuestrear (SMOTE, ADASYN) o ponderar la pérdida
  (*pesos de clase*).

  *En el proyecto.* La prevalencia es 34,0 % en el entrenamiento, 30,8 % en la validación 2025 y 31,5 % en 2026: un
  desbalance moderado con más de 90.000 casos de la clase minoritaria. Las cuatro opciones (ninguno, SMOTE, ADASYN,
  pesos) dan AUC medio entre 0,8086 (ADASYN) y 0,8112 (SMOTE) sobre las 112 combinaciones: no mejoran el ordenamiento.

Bootstrap por grupos
  Remuestreo con reposición para estimar la incertidumbre de una métrica sin suponer una distribución. «Por grupos»
  significa remuestrear unidades completas (aquí, órdenes) y no filas, porque las filas de un grupo no son
  independientes. El intervalo al 95 % es el rango entre los percentiles 2,5 y 97,5 de la métrica en las réplicas.

  *En el proyecto* ({doc}`03a_modelo_base`). 200 réplicas por órdenes completas; da los intervalos de AUC, PR-AUC y F1
  de la prueba 2026. Remuestrear por posición daría intervalos demasiado estrechos.

Bootstrap BCa
  Intervalo bootstrap corregido por sesgo y acelerado (Efron, 1987). En lugar de los percentiles 2,5 y 97,5 toma
  $\Phi\big(z_0 + (z_0 + z_\alpha)/(1 - a(z_0 + z_\alpha))\big)$, con $z_0$ el sesgo (qué fracción de réplicas cae bajo la
  estimación) y $a$ la aceleración (cuánto cambia el error estándar con el parámetro, estimada con jackknife). Sirve
  cuando la distribución bootstrap está corrida o es asimétrica, como la de un AUC cerca de 1.

  *En el proyecto* ({doc}`03b_112_combinaciones`). 1.000 réplicas que remuestrean órdenes dentro de cada bloque: la
  diferencia de AUC entre XGBoost y el bosque es 0,009 [0,006; 0,012].

Brier
  Error cuadrático medio de la probabilidad: $\frac{1}{n}\sum_i (p_i - y_i)^2$.

  *Cómo se lee.* Menor es mejor. La referencia es la del clasificador que siempre predice la prevalencia: con
  prevalencia $\pi$, vale $\pi(1-\pi)$. Mide a la vez calibración y capacidad de separar.

  *En el proyecto* ({doc}`03a_modelo_base`). Dummy 0,216; regresión logística 0,149 en la prueba 2026.

Calibración
  Una probabilidad está calibrada si, entre las posiciones a las que se asigna 0,40, llegan tarde alrededor del 40 %.
  Se revisa con el diagrama de calibración por deciles, el *Brier* y el *ECE*.

  *En el proyecto.* La logística queda bien calibrada (ECE 0,021; en cada decil la diferencia es de 4,2 puntos o
  menos), con subestimación leve en los deciles 3 a 9 y sobreestimación en el más alto. Calibración de XGBoost:
  pendiente (evaluación 2026).

Chi-cuadrado de independencia
  Contrasta si dos variables categóricas son independientes comparando las frecuencias observadas con las esperadas
  bajo independencia: $\chi^2 = \sum (O - E)^2 / E$. Con $n$ grande rechaza siempre; por eso se acompaña de un tamaño
  del efecto (*V de Cramér*).

  *En el proyecto* ({doc}`02b_eda`). Retraso (binario) contra línea de negocio y contra clase de compra.

Codificación por objetivo con encogimiento
  Reemplaza una categoría de muchos valores (proveedor, material) por su tasa de retraso, suavizada hacia la de su
  nivel superior para que las categorías con pocos casos no tomen valores extremos:

  $$
  \hat p(v) = \frac{\text{tardías}(v) + M\,\hat p(\text{padre}(v))}{n(v) + M}.
  $$

  *Cómo se lee.* $M$ es el peso del padre: con $n(v) \ll M$ manda el padre y con $n(v) \gg M$ manda la tasa propia.
  Un valor nuevo recibe la tasa del padre.

  *En el proyecto* ({doc}`02c_tiempo_y_fuga`). $M = 20$; once entidades; padre del proveedor = su región, padre del
  material = su grupo de artículos. Es la fuga más fácil de cometer: se calcula fuera de pliegue y agrupada por orden
  (ver *Fuga de información*).

Curtosis y asimetría
  Momentos de la distribución: la asimetría mide el sesgo (0 en una simétrica) y la curtosis en exceso mide el peso de
  las colas respecto de la normal (0 en la normal).

  *En el proyecto* ({doc}`02b_eda`). Cantidades y montos tienen asimetría de 30 a 300 y curtosis de miles, por lo que
  se usa $\mathrm{signo}(x)\log(1+|x|)$ para los modelos lineales. A los árboles no les afecta una transformación
  monótona.

Curva de aprendizaje
  Desempeño en entrenamiento y validación al crecer el tamaño de la muestra de entrenamiento. Una brecha grande que no
  cierra indica varianza (sobreajuste); dos curvas juntas y bajas indican sesgo (falta de flexibilidad).

  *En el proyecto* ({doc}`03a_modelo_base`). Logística: con el 10 % de las órdenes el AUC de validación ya es 0,823 y
  con todas llega a 0,830; la brecha baja de 0,066 a 0,037. Patrón de sesgo, no de varianza.

DeLong
  Prueba para comparar dos AUC calculados sobre las mismas posiciones. Estima la varianza y la covarianza de los AUC
  con los componentes estructurales de la estadística de Mann-Whitney y contrasta
  $z = (\mathrm{AUC}_1 - \mathrm{AUC}_2) / \sqrt{\mathrm{Var}_1 + \mathrm{Var}_2 - 2\,\mathrm{Cov}}$. La covarianza importa
  porque los dos modelos puntúan los mismos casos y sus errores están correlacionados; ignorarla subestima la potencia.

  *En el proyecto* ({doc}`03b_112_combinaciones`). En los 8 bloques, XGBoost contra cada rival (48 pruebas, Holm):
  supera al bosque en 6, a la logística y al SVM en 7 y a los demás en 8; nunca pierde. La implementación coincide con
  `pROC::roc.test` de R hasta 1e-10. Como supone filas independientes y las de una orden no lo son, se lee con un
  intervalo BCa por órdenes. Falta en la prueba 2026 (pendiente: evaluación 2026).
```

## D a F

```{glossary}
Delta de Cliff
  Tamaño del efecto no paramétrico entre dos grupos: $\delta = P(X > Y) - P(X < Y)$, entre −1 y 1. Referencia usual
  (Romano et al., 2006): $|\delta| < 0{,}147$ despreciable, hasta 0,33 pequeño, hasta 0,474 mediano, más grande.

  *En el proyecto* ({doc}`03b_112_combinaciones`). Sobre los 8 AUC por bloque: frente al bosque 0,25 (pequeño), a la
  logística 0,34 y al SVM 0,38 (medianos), a los otros tres de 0,88 a 1 (grandes). Cliff no parea: con el bosque sale
  pequeño aunque XGBoost gane 7 de 8 bloques, porque la variación entre semestres es mayor que la ventaja.

Deriva de covariables
  Cambio de la distribución de las predictoras entre el periodo de entrenamiento y el de uso. Se mide con el *PSI*. La
  *deriva del concepto* (que la relación entre $x$ y $y$ cambie) exige conocer el desenlace y no se mide con el PSI.

  *En el proyecto* ({doc}`02c_tiempo_y_fuga`). Fuerte en la tasa previa del comprador (PSI 1,02 en 2025 y 0,81 en
  2026) y en el límite en la del solicitante (0,20 y 0,25); las demás variables principales, de 0,02 a 0,15.

Diferencia crítica (CD)
  Distancia mínima entre rangos medios para que *Nemenyi* declare distintos a dos modelos:

  $$
  \mathrm{CD} = q_{\alpha} \sqrt{\frac{k(k+1)}{6N}},
  $$

  con $k$ modelos, $N$ bloques y $q_{\alpha}$ el valor crítico del rango studentizado dividido por $\sqrt{2}$.

  *En el proyecto* ({doc}`03b_112_combinaciones`). $k = 7$, $N = 8$, CD = 3,19. Se dibuja en el diagrama de rangos
  (franja sombreada).

ECE (error de calibración esperado)
  Promedio, ponderado por tamaño, de la distancia entre la probabilidad media predicha y la tasa observada en cada
  intervalo (aquí, deciles): $\sum_b \frac{n_b}{n}\,|\bar p_b - \bar y_b|$. Menor es mejor.

  *En el proyecto.* Logística en la prueba 2026: 0,021 (2,1 puntos); Dummy 0,035.

Épsilon cuadrado ($\varepsilon^2$)
  Tamaño del efecto de *Kruskal-Wallis*: $\varepsilon^2 = (H - k + 1)/(n - k)$, la fracción de la variación de los
  rangos que explica el grupo.

  *En el proyecto* ({doc}`02b_eda`). Días de retraso por línea de negocio, 2,0 %; por clase de compra, 1,0 %: el grupo
  mueve poco la variación; el retraso se decide en la posición.

Estacionariedad
  Una serie es (débilmente) estacionaria si su media, su varianza y su autocorrelación no dependen del tiempo. Se
  contrasta con *ADF* y *KPSS*. Una serie con tendencia o cambio de nivel no lo es.

  *En el proyecto.* La tasa mensual 2021-2024 no es estacionaria; por eso la partición y la validación son
  cronológicas.

Exactitud
  Proporción de aciertos: $(TP + TN)/n$. Engañosa con clases desbalanceadas.

  *En el proyecto.* Un clasificador que diga siempre «a tiempo» acierta el 66 % del entrenamiento; por eso las
  métricas son AUC y F1.

F1
  Media armónica de la precisión y la sensibilidad de la clase «tarde»:

  $$
  F_1 = \frac{2\,P\,R}{P + R} = \frac{2\,TP}{2\,TP + FP + FN}.
  $$

  *Cómo se lee.* No usa los verdaderos negativos y depende del umbral. La referencia es la de marcar todo
  (precisión = prevalencia, sensibilidad = 1): con prevalencia 31,5 % el F1 es 0,479 (Dummy).

  *En el proyecto.* El umbral se elige en la validación para maximizar el F1 y se reporta en el bloque o la prueba.
  Logística en 2026: 0,649 [0,631; 0,666]; XGBoost sin balanceo: 0,676 ± 0,048 en los ocho bloques.

Friedman (prueba de)
  Contraste no paramétrico para comparar $k$ modelos en $N$ bloques. Ordena los modelos dentro de cada bloque (rango 1
  al mejor) y contrasta si los rangos medios $R_j$ difieren más de lo que daría el azar:

  $$
  \chi^2_F = \frac{12N}{k(k+1)}\Big[\sum_j R_j^2 - \frac{k(k+1)^2}{4}\Big].
  $$

  No supone normalidad ni varianzas iguales y trabaja con mediciones pareadas por bloque.

  *En el proyecto* ({doc}`03b_112_combinaciones`). $k = 7$, $N = 8$: $\chi^2 = 45{,}6$, $p = 3{,}5 \times 10^{-8}$.
  XGBoost queda primero en los ocho bloques (rango medio 1,00).

Fuga de información
  Que el modelo use, al entrenar o al elegir, información que no tendría al predecir (el desenlace o algo posterior a
  la fecha de predicción). Da resultados optimistas que no se sostienen en uso.

  *En el proyecto* ({doc}`02c_tiempo_y_fuga`). Controles: cada grupo de variables se justifica como disponible el día de
  la liberación; se retiraron el desenlace con otra regla, las fechas de entrada y factura, los cambios de fecha
  posteriores y los maestros que el planificador ajusta tras ver retrasos; los historiales se cortan en la fecha de
  creación de la orden; la codificación por objetivo se calcula fuera de pliegue y agrupada por orden; el
  preprocesamiento se ajusta solo con el entrenamiento; la prueba 2026 no elige nada. Verificación: ninguna variable
  tiene AUC univariado mayor de 0,75 (el máximo es 0,732).
```

## G a M

```{glossary}
Gradiente, hessiano y boosting
  El *boosting* construye el modelo como suma de árboles, cada uno ajustado para reducir la pérdida. Con la
  log-pérdida y $p = \sigma(F)$, el gradiente por posición es $g_i = p_i - y_i$ y el hessiano $h_i = p_i(1 - p_i)$.
  Un paso de Newton da el valor óptimo de una hoja, $w^{*} = -G/(H + \lambda)$, con $G$ y $H$ las sumas de $g_i$ y
  $h_i$ de la hoja. Ver *XGBoost*.

Grid search y búsqueda aleatoria
  Optimizadores simples de hiperparámetros. La rejilla evalúa puntos regulares (recorre el espacio en un orden fijo) y
  sufre la maldición de la dimensión: con muchos hiperparámetros caben pocos niveles por dimensión. La aleatoria
  evalúa puntos al azar y, a igual presupuesto, explora más valores distintos de cada hiperparámetro importante
  (Bergstra y Bengio, 2012).

  *En el proyecto* ({doc}`03b_112_combinaciones`). Con E = 20, la rejilla de XGBoost (seis hiperparámetros) queda en
  dos niveles por dimensión (16 puntos).

Holm (corrección de)
  Corrección de comparaciones múltiples más potente que Bonferroni y que también controla la tasa de error de la
  familia: se ordenan los $m$ p de menor a mayor y el $i$-ésimo se multiplica por $m - i + 1$, con monotonía.

  *En el proyecto.* Sobre las 136 pruebas de asociación del EDA, y sobre las seis comparaciones de Wilcoxon (p exacto
  0,0078 que pasa a 0,047).

Iman-Davenport (corrección de)
  Convierte el estadístico de *Friedman* en una $F$ con $k-1$ y $(k-1)(N-1)$ grados de libertad, menos conservadora:
  $F = (N-1)\chi^2_F / (N(k-1) - \chi^2_F)$.

  *En el proyecto.* $F = 135{,}5$, $p = 7{,}5 \times 10^{-26}$.

Información mutua
  Cuánto reduce el conocimiento de $x$ la incertidumbre sobre $y$: $I(x; y) = \sum p(x,y)\ln\frac{p(x,y)}{p(x)p(y)}$.
  Es 0 si son independientes y capta relaciones no monótonas, que el AUC univariado no ve.

  *En el proyecto* ({doc}`02b_eda`). El historial del proveedor tiene la mayor, 0,20: su relación con el retraso no es
  monótona en todo el rango.

Isolation Forest
  Detector de atípicos multivariado: aísla cada punto con cortes aleatorios en árboles; los puntos raros quedan
  aislados con pocos cortes y obtienen un puntaje de anomalía alto.

  *En el proyecto* ({doc}`02b_eda`). Marca el 2,9 % de las posiciones de una muestra de 50.000; de ellas llega tarde el
  40,4 %, contra 33,7 % del resto. No se eliminan: son los casos que interesa anticipar.

k-medias y silueta
  *k-medias* agrupa las observaciones en $k$ grupos minimizando la suma de distancias cuadradas al centro de su grupo.
  La *silueta* de un punto es $(b - a)/\max(a, b)$, con $a$ la distancia media a su grupo y $b$ la del grupo vecino más
  cercano; su media va de −1 a 1, y menos de 0,25 se toma como poca o ninguna estructura.

  *En el proyecto* ({doc}`02b_eda`). La mejor silueta es 0,25 con cinco grupos; la tasa de retraso por grupo va de
  28,6 % a 36,0 %. No hay subpoblaciones marcadas.

KPSS
  Prueba de estacionariedad con la hipótesis nula contraria a la de *ADF*: la serie es estacionaria. Un p bajo rechaza
  la estacionariedad.

  *En el proyecto.* p = 0,026 sobre la tasa mensual: junto con ADF (p = 0,70) concluye que no es estacionaria.

Kruskal-Wallis
  Extensión de Wilcoxon-Mann-Whitney a $k$ grupos: compara los rangos de una variable entre grupos sin suponer
  normalidad ni varianzas iguales. Su tamaño del efecto es $\varepsilon^2$.

  *En el proyecto.* Días de retraso por línea de negocio y por clase de compra.

k vecinos más cercanos (KNN)
  Clasifica una posición por la proporción de la clase entre sus $k$ vecinas más cercanas en el espacio de
  predictoras. No tiene fase de entrenamiento: cuesta en inferencia y sufre con muchas dimensiones.

  *En el proyecto.* Se ajusta sobre 30 componentes principales, con $k$ de 5 a 300 y ponderación uniforme o por
  distancia. Mejor combinación: AUC 0,781. Inferencia: 1,3 s por cada mil posiciones (XGBoost, 16 ms).

Logaritmo con signo
  Transformación $\mathrm{signo}(x)\log(1 + |x|)$ que comprime colas largas y admite valores negativos y cero.

  *En el proyecto* ({doc}`02c_tiempo_y_fuga`). Se aplica a las 83 predictoras numéricas en el preprocesamiento (junto
  con la mediana para vacíos y la estandarización).

Log-pérdida (entropía cruzada)
  Pérdida de la clasificación probabilística: $-\frac{1}{n}\sum_i [y_i \ln p_i + (1 - y_i)\ln(1 - p_i)]$. Es la
  log-verosimilitud negativa de un modelo Bernoulli; es convexa en los parámetros de una logística y es la pérdida que
  minimizan la regresión logística y XGBoost (`binary:logistic`). Penaliza con fuerza los errores seguros.

Maldición de la dimensión
  Con muchas dimensiones, el volumen del espacio crece tan rápido que los datos quedan dispersos, las distancias se
  vuelven poco informativas y las rejillas no caben en el presupuesto.

  *En el proyecto.* Afecta a KNN (por eso se aplica PCA) y a la rejilla con seis hiperparámetros.
```

## M a P

```{glossary}
MAR, MCAR y MNAR
  Mecanismos de datos faltantes. MCAR: el vacío no depende de nada. MAR: depende de lo observado. MNAR: depende del
  valor que falta. Se distinguen comparando el desenlace entre filas con y sin el dato.

  *En el proyecto* ({doc}`02a_datos`). Los vacíos son estructurales (sin solicitud, sin imputación, sin material): MAR,
  porque dependen de las marcas `tiene_solped`, `tiene_imputacion` y `tiene_material`. La tasa de tarde con y sin dato
  difiere hasta en 4 puntos, lo que descarta MCAR.

Naive Bayes gaussiano
  Clasificador que supone predictoras independientes dadas la clase y normales dentro de cada clase:
  $P(y \mid x) \propto P(y)\prod_j N(x_j; \mu_{jy}, \sigma^2_{jy})$. Rápido y con sesgo alto cuando hay correlación.

  *En el proyecto.* Es el peor modelo (AUC 0,723 a 0,735 según la combinación); el supuesto de independencia choca con
  el bloque de plazo y los historiales, que son casi colineales. Un hiperparámetro: el suavizado de la varianza.

Nemenyi (prueba de)
  Comparaciones por pares tras *Friedman*: dos modelos difieren si sus rangos medios se separan más que la
  *diferencia crítica*. Controla el error de las $k(k-1)/2$ comparaciones a la vez, por lo que es conservadora.

  *En el proyecto.* Con 21 pares, XGBoost se separa del árbol, KNN y Naive Bayes, pero no de la logística, el bosque ni
  el SVM (diferencias de rango de 1,6, 1,6 y 2,8, contra una diferencia crítica de 3,19).

One-hot
  Codifica una categoría de $m$ valores como $m$ columnas binarias. Aumenta la dimensión y las categorías no vistas
  quedan en ceros.

  *En el proyecto.* Las 42 categóricas de pocos valores; con ellas las 136 predictoras pasan a 591 columnas.

Optuna y TPE
  Optimización bayesiana secuencial. *TPE* (estimador de Parzen en árbol) modela la densidad de las configuraciones
  buenas, $l(x)$, y la de las malas, $g(x)$, separadas por un cuantil $\gamma$ de los valores observados, y propone el
  $x$ que maximiza $l(x)/g(x)$, lo que equivale a maximizar la mejora esperada (Bergstra et al., 2011).

  *En el proyecto.* Diez puntos al azar y luego TPE multivariado. Es el que más rápido se acerca al óptimo y la
  combinación líder usa Optuna (XGBoost sin balanceo, AUC 0,8517).

PCA (análisis de componentes principales)
  Rotación a ejes ortogonales ordenados por la varianza que explican, obtenida de los vectores propios de la matriz de
  covarianza (o la SVD) de los datos estandarizados. Sirve para reducir dimensión o explorar.

  *En el proyecto* ({doc}`02b_eda`). La varianza está repartida: el primer componente explica el 12,3 % y hacen falta
  33 de 80 para llegar al 80 %; no se reduce antes de modelar. Solo KNN trabaja sobre 30 componentes.

Pesos de clase
  Multiplican la pérdida de cada clase por el inverso de su frecuencia (`balanced`). En XGBoost el equivalente es
  `scale_pos_weight` = negativos/positivos.

  *En el proyecto* ({doc}`03b_112_combinaciones`). En KNN y Naive Bayes dan resultados idénticos a «ninguno»: solo
  cambian la probabilidad a priori, una transformación monótona del puntaje que no altera el orden ni el AUC, y el
  umbral de F1 absorbe el corrimiento.

Platt (escalamiento de)
  Calibración posterior: se ajusta una logística sobre el puntaje $s$ del modelo, $p = 1/(1 + e^{A s + B})$, con datos
  distintos de los del entrenamiento. Corrige modelos como el boosting o los árboles, cuyas probabilidades suelen
  salir comprimidas o desplazadas.

  *En el proyecto.* La logística no la necesita. Si la calibración de XGBoost en la prueba 2026 lo pide, se ajusta
  con la validación 2025 (pendiente: evaluación 2026).

PR-AUC (precisión promedio)
  Área bajo la curva de precisión contra sensibilidad. A diferencia del AUC ROC, su referencia es la prevalencia y no
  0,5, y se centra en la clase minoritaria.

  *En el proyecto.* Logística en 2026: 0,726, más del doble de la prevalencia (31,5 %). XGBoost sin balanceo: 0,778
  en los ocho bloques.

Precisión (valor predictivo positivo)
  De las posiciones marcadas como tardías, la fracción que de verdad llega tarde: $TP/(TP + FP)$.

  *En el proyecto.* Logística en 2026 con umbral 0,287: marca el 37,7 % y acierta en el 59,5 % de lo marcado. La
  regla del proveedor, con su mejor umbral, tiene precisión de 37,3 %.

PSI (índice de estabilidad de la población)
  Compara la distribución de una variable entre un periodo de referencia y otro, por intervalos de la referencia:

  $$
  \mathrm{PSI} = \sum_j (a_j - r_j)\ln\frac{a_j}{r_j}.
  $$

  *Cómo se lee.* Menos de 0,1, estable; de 0,1 a 0,25, cambio moderado; más de 0,25, cambio fuerte.

  *En el proyecto.* Ver *Deriva de covariables*. Usa solo las predictoras, nunca el desenlace de 2025 ni de 2026.
```

## R a U

```{glossary}
Random Forest
  Promedio de muchos árboles entrenados con muestras bootstrap y, en cada corte, un subconjunto aleatorio de variables
  ($\sqrt{p}$, $\log_2 p$ o 5 %). Reduce la varianza del árbol al decorrelacionar los árboles.

  *En el proyecto.* 100 árboles (recorte de costo acordado). Mejor combinación: AUC 0,8427; con 67 horas de CPU es de
  los más costosos. Segundo en el rango medio junto con la logística (2,625).

Regla del proveedor
  Línea base sin modelo: ordenar las posiciones por la tasa previa de retraso de su proveedor, que es lo que haría un
  comprador.

  *En el proyecto.* AUC 0,670 [0,654; 0,686] en 2026; la logística le saca 0,155.

Regresión logística
  Modela el logaritmo de la razón de odds como combinación lineal: $\ln\frac{p}{1 - p} = w^{\top}x + b$. Se ajusta
  minimizando la log-pérdida (convexa: un solo óptimo) con una penalización L1 o L2.

  *En el proyecto.* Modelo base: AUC 0,825 y F1 0,649 en 2026. El coeficiente más grande es el del proveedor codificado
  (1,87): una desviación estándar más de retraso histórico multiplica los odds por 6,5. En las 112 se ajusta con SGD.

Regularización L1 y L2
  Penalizar la magnitud de los coeficientes para controlar la varianza. L2 (cresta) suma $\alpha\sum_j w_j^2$ y los
  encoge sin anularlos; reparte el efecto entre variables correlacionadas. L1 (lazo) suma $\alpha\sum_j |w_j|$ y lleva
  algunos a cero: selecciona. En XGBoost, el análogo es $\lambda\|w\|^2/2$ sobre los pesos de las hojas más $\gamma$ por
  hoja.

  *En el proyecto.* En la logística base, $C = 0{,}003$ (el inverso de la fuerza de L2; se eligió en una rejilla de
  0,0001 a 3 con validación temporal); en las 112, $\alpha$ de $10^{-7}$ a $10^{-2}$ y penalización L1 o L2 como
  hiperparámetro. En XGBoost, la regularización viene de la tasa de aprendizaje, la profundidad, el peso mínimo por
  hoja y los submuestreos; $\lambda$ y $\gamma$ quedan en los valores por defecto de la librería (no se optimizan).

SGD (descenso de gradiente estocástico)
  Actualiza los parámetros con el gradiente de la pérdida en una observación (o un lote) a la vez:
  $w \leftarrow w - \eta\,\nabla \ell_i(w)$. Escala a muchos datos y converge a un entorno del óptimo.

  *En el proyecto* ({doc}`03b_112_combinaciones`). `SGDClassifier` con pérdida logística (logística) o bisagra (SVM
  lineal), `max_iter = 30`. Es un recorte de costo para que las 896 tareas cupieran en una noche: la «logística» de
  las 112 no es el óptimo exacto del modelo base.

SHAP
  Reparte la predicción de una posición entre sus variables con valores de Shapley (teoría de juegos): la
  contribución de cada variable es su aporte promedio sobre todos los órdenes posibles de incorporarlas. Los valores
  suman la diferencia entre la predicción y la media. TreeSHAP lo calcula de forma eficiente para árboles.

  *En el proyecto.* Explicación de XGBoost, global y local (pendiente: evaluación 2026).

Silueta
  Ver *k-medias y silueta*.

SMOTE
  Genera posiciones tardías sintéticas interpolando entre una existente y una de sus vecinas cercanas de la misma clase.

  *En el proyecto* ({doc}`03b_112_combinaciones`). 5 vecinos, hasta igualar la clase mayoritaria, una vez por ventana y
  solo con el entrenamiento. Las columnas one-hot sintéticas quedan entre 0 y 1 (aproximación, porque SMOTENC
  costaba demasiado). Duplica el costo y no gana nada en XGBoost.

Spearman
  Correlación de Pearson sobre los rangos: mide asociación monótona, sin exigir linealidad y con resistencia a los
  atípicos.

  *En el proyecto.* Mapa de calor de las 30 predictoras numéricas más asociadas al objetivo.

STL
  Descomposición de una serie en tendencia, estacionalidad y residuo con suavizado local (Loess); la versión robusta
  resiste atípicos.

  *En el proyecto* ({doc}`02c_tiempo_y_fuga`). Sobre la tasa mensual (periodo 12): la tendencia baja de 45,4 % a
  comienzos de 2021 a 34,4 % en enero de 2023 y se estabiliza (33,2 %); enero está 11 puntos por debajo de la
  tendencia y octubre 8 por encima.

SVM lineal
  Busca el hiperplano que separa las clases con el mayor margen, minimizando la pérdida bisagra
  $\max(0,\ 1 - y\,f(x))$ más una penalización. Da un puntaje, no una probabilidad.

  *En el proyecto.* Se ajusta con SGD y pérdida bisagra. Mejor combinación: AUC 0,8385. No se separa de XGBoost
  según Nemenyi.

Tukey (cercas de)
  Regla de atípicos univariados: fuera de $[Q_1 - 1{,}5\,\mathrm{IQR},\ Q_3 + 1{,}5\,\mathrm{IQR}]$.

  *En el proyecto.* Entre el 15 % y el 17 % de los valores de monto y cantidad queda fuera; no se eliminan.

Umbral de decisión
  Valor $\tau$ con el que la probabilidad se vuelve una alerta: marcar si $p(x) \ge \tau$. Subirlo gana precisión y
  pierde sensibilidad.

  *En el proyecto.* Se elige en la validación para maximizar el F1 (en la validación interna de cada bloque para las
  112; con la validación 2025 en la prueba). Logística: 0,287.
```

## V a X

```{glossary}
Validación anidada y ventana creciente
  Dos bucles. El externo estima el desempeño con bloques que el modelo no vio; el interno, dentro de cada ventana de
  entrenamiento, elige los hiperparámetros. Así el optimizador nunca ve la fila que se evalúa. «Ventana creciente»
  significa que cada bloque entrena con todo lo anterior, sin mirar el futuro.

  *En el proyecto* ({doc}`03b_112_combinaciones`). Ocho bloques semestrales externos (2022-S1 a 2025-S2); el último
  semestre de cada ventana valida en el bucle interno; 20 evaluaciones por optimizador; 896 trabajos. La prueba 2026
  no entra.

V de Cramér
  Tamaño del efecto de una tabla de contingencia: $V = \sqrt{\chi^2 / (n\,(\min(r,c) - 1))}$, de 0 a 1. Por
  convención, menos de 0,1 es débil y de 0,1 a 0,3, moderado.

  *En el proyecto.* Línea de negocio 0,107 y clase de compra 0,096 frente al retraso. Con entidades de miles de valores
  se infla, y por eso se usa el AUC de su tasa fuera de pliegue.

VIF (factor de inflación de la varianza)
  Mide la colinealidad de una predictora con las demás: $\mathrm{VIF}_j = 1/(1 - R_j^2)$, donde $R_j^2$ es el de
  regresar la variable $j$ sobre las otras. Más de 5 a 10 indica redundancia que vuelve inestables los coeficientes
  de un modelo lineal.

  *En el proyecto* ({doc}`02b_eda`). Plazo hábil al liberar, plazo prometido y días de liberación: 1.059, 1.031 y 322
  (casi por construcción uno es la resta de los otros). Fuera de esos bloques, menos de 6.

Wilcoxon (rangos con signo)
  Prueba pareada no paramétrica: ordena las diferencias absolutas entre dos modelos en cada bloque y suma los rangos
  de las positivas. Usa la magnitud además del signo, y no supone normalidad.

  *En el proyecto.* XGBoost contra cada modelo (seis comparaciones) en los ocho bloques: gana en los ocho; con $N = 8$
  el p exacto mínimo es $2/2^8 = 0{,}0078$, que con Holm queda en 0,047. La diferencia media de AUC es de 0,011 a
  0,014 frente a los tres modelos más cercanos.

XGBoost
  *Boosting* de árboles con regularización. Minimiza

  $$
  \mathcal{L} = \sum_i \ell(y_i, \hat y_i) + \sum_t \Omega(f_t), \qquad
  \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2,
  $$

  con una aproximación de segundo orden de la pérdida (gradiente $g_i$ y hessiano $h_i$). El valor de cada hoja es
  $w_j^{*} = -G_j/(H_j + \lambda)$ y la ganancia de un corte es la mejora de la pérdida regularizada de dividir. Usa
  histogramas para encontrar cortes (`tree_method = hist`), por lo que el costo de cada nivel crece con el número de filas y de columnas.

  *En el proyecto.* Es el mejor de los siete modelos: AUC 0,852 ± 0,020 y F1 0,676 ± 0,048 sin balanceo, con Optuna;
  gana en los ocho bloques. Espacio: árboles 100-600, tasa de aprendizaje 0,01-0,3, profundidad 3-10, submuestreo de
  filas 0,5-1 y de columnas 0,3-1, peso mínimo por hoja 1-100. Inferencia, 16 milisegundos por mil posiciones; 68
  horas de CPU en total. Evaluación final en la prueba 2026: pendiente.
```
