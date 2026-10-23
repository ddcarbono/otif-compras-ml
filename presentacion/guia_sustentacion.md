# Guía de estudio para la sustentación

Guía para que cada autor defienda el proyecto completo: lo que hay que saber decir en cada criterio de la rúbrica, las
cifras clave y las preguntas difíciles del jurado con su respuesta corta.

**Fuentes.** La rúbrica de sustentación (siete criterios, 20 minutos; `entregables/diapositivas.md`), los capítulos del
libro (`book/fuentes/02a` a `05`), `resultados/` y `src/`. Toda cifra sale de esos archivos. La evaluación de XGBoost en
2026 (`resultados/prueba_2026/`) es la tercera mirada del proyecto a la prueba: se dice que es exploratoria. El glosario
(`book/glosario.md`) define cada estadístico.

**Cómo usarla.** El criterio 7 evalúa el dominio individual: los dos autores responden y cada uno ensaya también las
partes del otro. Primero las cifras, luego los criterios, y al final las preguntas en voz alta con la respuesta tapada.

## Las cifras que hay que saberse

| tema | cifra |
|---|---|
| población | 406.126 posiciones de 104.614 órdenes (2,8 por orden), 2021 a 2026; las unidades independientes son las órdenes |
| objetivo | $y = 1$ si la primera entrada de mercancía llega más de 1 día hábil después de la fecha prometida vigente al liberar |
| predictoras | 136 (83 numéricas, 42 categóricas de pocos valores, 11 entidades); 591 columnas tras el preprocesamiento |
| partición | entrenamiento 2021-2024: 271.391; validación 2025: 82.794; prueba 2026: 51.941 |
| prevalencia | 34,0 % (entrenamiento), 30,8 % (2025), 31,5 % (2026) |
| por año | 42,7 % (2021), 26,9 % (2023), 31,5 % (2024) |
| señal por variable | ningún AUC univariado pasa de 0,75; el máximo es 0,732 (plazo hábil al liberar); proveedor 0,695; su historial 0,66 |
| plazo | con el plazo agotado al liberar llega tarde el 85,9 %; con más de 10 días, cerca del 20 % |
| tiempo | ADF p = 0,70 y KPSS p = 0,026: la tasa mensual no es estacionaria |
| arranque en frío | en 2026, 11,1 % de las posiciones trae un proveedor nuevo y 9,7 %, un material nuevo |
| línea base | Dummy: AUC 0,5, F1 0,479. Regla del proveedor: AUC 0,670 [0,654; 0,686] |
| logística (prueba 2026) | AUC 0,825 [0,813; 0,838]; F1 0,649 [0,631; 0,666]; PR-AUC 0,726; Brier 0,149; ECE 0,021; umbral 0,287 |
| diseño | 7 modelos × 4 balanceos × 4 optimizadores = 112; 8 bloques externos; E = 20; 896 trabajos; 6,7 h de reloj y 200,6 h de CPU |
| mejor | XGBoost sin balanceo con Optuna: AUC 0,852 ± 0,020, F1 0,676 ± 0,048, PR-AUC 0,778 (media de los 8 bloques) |
| modelos | AUC por mejor combinación: XGBoost 0,8517, Random Forest 0,8427, logística 0,8394, SVM 0,8385, árbol 0,8100, KNN 0,7814, Naive Bayes 0,7347 |
| comparación | Friedman χ² = 45,6 (p = 3,5·10⁻⁸); diferencia crítica de Nemenyi 3,19; Wilcoxon con Holm: p = 0,047 contra cada uno de los seis |
| dentro de cada bloque | DeLong (validado contra pROC hasta 1e-10), Holm sobre 48: XGBoost supera al bosque en 6 de 8 bloques, a la logística y al SVM en 7, y a los otros tres en 8; ningún rival le gana en ninguno. Diferencia media con BCa 95 % por órdenes: bosque 0,009 [0,006; 0,012], logística 0,012 [0,007; 0,016], SVM 0,013 [0,008; 0,017]. δ de Cliff: 0,25 (pequeño), 0,34 y 0,38 (medianos), 0,88-1,00 (grandes) |
| XGBoost en 2026 | AUC 0,848 [0,835; 0,860], PR-AUC 0,758, F1 0,675 al umbral 0,345, ECE 0,022 con Platt; Δ sobre la logística 0,023 [0,017; 0,030]; con el 20 % de alertas, 8.242 tardías contra 7.895. Propuesta del E3: 0,860, Δ 0,012 [0,004; 0,018] |
| balanceo | AUC medio de las 112: ninguno 0,8111, pesos 0,8110, SMOTE 0,8112, ADASYN 0,8086 |
| optimizadores | arrepentimiento en la evaluación 5: Optuna 0,006, genético 0,013, aleatoria 0,015, rejilla 0,042 |

Una trampa de lectura: el 0,825 de la logística es de la **prueba 2026** y el 0,852 de XGBoost es la **media de los ocho
bloques** de la validación anidada. No se comparan entre sí. Con el mismo protocolo, la mejor logística (SGD) queda en
0,8394 y XGBoost en 0,8517: una diferencia de 0,012.

## Criterio 1: problema y formulación (2 min, 10 %)

**Qué hay que decir.**

- El problema: una de cada tres posiciones llega tarde, y un repuesto tarde detiene una planta o obliga a comprar de
  urgencia. Un estudio en una empresa con el mismo ERP reporta 23 % de entregas tardías y que bajó al 10 % con
  seguimiento automático: el retraso se gestiona si se sabe dónde mirar.
- La pregunta: ¿se puede predecir, el día en que la orden sale hacia el proveedor, si cada posición llegará tarde?
- Objetivos: (1) un clasificador binario cuya probabilidad sirva como riesgo (calibrada); (2) identificar qué variables,
  conocidas al liberar, explican el retraso.
- La formulación, completa y con cada símbolo: la unidad es la posición $i$; $x_i \in \mathbb{R}^{136}$ se conoce el día
  $t_i$ de la liberación; $y_i = 1\{\text{primera entrada} > \text{fecha prometida} + 1 \text{ día hábil}\}$. La tarea es
  clasificación binaria: estimar $p(x) = P(y = 1 \mid x)$ y decidir con un umbral $\tau$ elegido en validación.
- Por qué binaria: la decisión del comprador es binaria (seguir o no una orden). La variable `dias_retraso` se retiró
  del dataset por ser el desenlace con otra regla.

**No olvidar.** La tolerancia es de un día hábil, con festivos de Colombia. La fecha prometida es la vigente el día de la
liberación, reconstruida con el historial de cambios (un cambio posterior no cuenta: ese día no se conocía).

## Criterio 2: EDA, datos y preprocesamiento (3 min, 15 %)

**Qué hay que decir.**

- Fuente: tablas de compras del ERP SAP, solo lectura, uso académico autorizado, anonimizadas (seudónimos para
  personas, códigos para terceros). No son redistribuibles.
- Población: de 490.310 posiciones se quedan las compras con plazo; se excluyen las que no tienen entrada, las
  legalizaciones (la orden se hizo después de recibir), la fruta del mes, los títulos valores y las rutas cumplidas.
- Descriptiva: montos y cantidades con asimetría de 30 a 300, curtosis de miles (mediana del valor de una posición
  244.296 COP y media 27,8 millones) → logaritmo con signo para los lineales; los árboles no lo necesitan.
- Faltantes: en 33 de las 136 variables, **estructurales** (sin solicitud, sin imputación, sin material): mecanismo
  MAR, porque dependen de las marcas `tiene_*`; la tasa de tarde con y sin dato difiere hasta 4 puntos (descarta MCAR).
  Se imputa con la mediana y la marca conserva el vacío. No se usa la prueba de Little: con 136 variables mixtas y
  270.000 filas rechaza siempre.
- Atípicos: Tukey (15 a 17 % de los montos) e Isolation Forest (2,9 %, con 40,4 % de tarde frente a 33,7 %). No se
  eliminan: son los casos que importan.
- Relaciones: AUC univariado, información mutua y V de Cramér, con Holm sobre 136 pruebas (todos los p < 10⁻³⁰⁰: el
  tamaño del efecto es lo que ordena). Línea y clase de compra: V de Cramér 0,107 y 0,096; $\varepsilon^2$ de Kruskal-Wallis
  2,0 % y 1,0 %. El bloque de plazo es casi colineal (VIF 1.059, 1.031, 322).
- Multivariado: PCA no concentra (12,3 % el primer componente, 33 de 80 para el 80 %); k-medias con silueta 0,25.
- Tiempo: tasa 45,4 % → 34,4 % → 33,2 % (STL), no estacionaria; PSI fuerte en la tasa previa del comprador (1,02 y 0,81).
- **Fuga**, que es lo que más se pregunta: ver la sección de preguntas.

**No olvidar.** El EDA se calcula solo con 2021-2024. Las entidades repiten entre particiones; las órdenes, nunca.

## Criterio 3: fundamentos teóricos del modelo (5 min, 25 %)

El modelo es XGBoost (el mejor de los siete); la regresión logística es el contraste y el modelo base. Este criterio
pide fórmulas, no resultados.

**Formulación.** El modelo es una suma de $K$ árboles: $F(x) = \sum_{k=1}^{K} \eta\, f_k(x)$, con $\eta$ la tasa de
aprendizaje, y la probabilidad es $p(x) = \sigma(F(x)) = 1/(1 + e^{-F(x)})$.

**Función de pérdida.** La log-pérdida (log-verosimilitud negativa de un Bernoulli):
$\ell(y, F) = -y \ln p - (1 - y)\ln(1 - p)$. Con $p = \sigma(F)$ las derivadas respecto de $F$ son limpias:

```
g = ∂ℓ/∂F = p − y          h = ∂²ℓ/∂F² = p (1 − p)
```

**Objetivo regularizado.** $\mathcal{L} = \sum_i \ell(y_i, F(x_i)) + \sum_k \Omega(f_k)$, con
$\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2$ ($T$ hojas, $w_j$ el valor de la hoja $j$).

**Principio de optimización.** Descenso en el espacio de funciones con paso de Newton: se agrega un árbol por vez. Con la
aproximación de segundo orden, el árbol $k$ minimiza $\sum_i [g_i f(x_i) + \tfrac{1}{2} h_i f(x_i)^2] + \Omega(f)$. Para
una estructura fija, el valor óptimo de la hoja $j$ y la mejora de un corte son:

```
w*_j = − G_j / (H_j + λ)                      G_j = Σ g_i,  H_j = Σ h_i  de las posiciones de la hoja
ganancia = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ
```

El corte se hace si la ganancia es positiva. Con $\lambda = 0$ y pérdida cuadrática es el boosting clásico sobre los
residuos; con la log-pérdida, $g = p - y$ es el residuo de probabilidad y $h$ pondera cuánto se confía en él.

**Regularización.** Cinco frenos: $\eta$ (cada árbol aporta poco), profundidad máxima, peso mínimo por hoja (suma de
hessianos $H_j$, no número de filas), submuestreo de filas y de columnas, y el número de árboles. Los dos términos
explícitos, $\lambda$ y $\gamma$, quedan en los valores por defecto de la librería: lo que se optimiza son los seis
hiperparámetros del espacio (árboles 100-600, $\eta$ 0,01-0,3 en log, profundidad 3-10, submuestreo de filas 0,5-1 y de
columnas 0,3-1, peso mínimo por hoja 1-100 en log).

**Supuestos.** XGBoost no supone distribución ni linealidad: es no paramétrico y capta interacciones y no linealidades
(el codo del plazo). Supone observaciones intercambiables dadas las predictoras; aquí se violan en dos sentidos (las
posiciones de una orden se parecen y el proceso cambia en el tiempo), y por eso la validación es agrupada y cronológica.

**Complejidad.** Los cortes se buscan sobre histogramas de las columnas: preparar los histogramas cuesta del orden de
$n \cdot p$ una vez y cada nivel de cada árbol, del orden de $n \cdot p$ con tamaño de histograma fijo (resultado
estándar de la librería, no medido aquí). Predecir cuesta $K \times \text{profundidad}$ por posición: 16 ms por cada mil
posiciones. La capacidad crece con $K$, la profundidad y el número de hojas.

**Generalización.** La pérdida de entrenamiento baja siempre al añadir árboles; la de validación no. Se controla con los
frenos de arriba y con la validación cronológica, y se mide con la diferencia entre el AUC interno (donde se eligen los
hiperparámetros) y el externo (el bloque): 0,8542 contra 0,8517 en la combinación líder.

**Contraste con la logística.** Pérdida convexa en $w$: un solo óptimo; frontera lineal en el logit; regularización L2 o L1
explícita. La curva de aprendizaje mostró sesgo, no varianza (con el 10 % de los datos, 0,823; con todos, 0,830): le falta
flexibilidad, que es lo que el boosting aporta.

## Criterio 4: implementación y diseño experimental (3 min, 15 %)

**Qué hay que decir.**

- Partición cronológica fijada antes de mirar los datos: 2021-2024 entrena, 2025 valida, 2026 prueba, usada una sola
  vez y sin elegir nada con ella. Ninguna orden queda partida entre conjuntos.
- Validación anidada con ventana creciente. **Externo:** ocho bloques semestrales (2022-S1 a 2025-S2); cada uno entrena
  con todo lo anterior y evalúa en el suyo. **Interno:** dentro de cada ventana, el último semestre valida; ahí el
  optimizador elige los hiperparámetros por ROC-AUC y con la mejor configuración se fija el umbral de máximo F1.
  Después se reajusta con toda la ventana y se mide en el bloque.
- 112 combinaciones: 7 modelos × 4 balanceos × 4 optimizadores, con el mismo presupuesto de **E = 20 evaluaciones**
  para que la comparación entre optimizadores sea justa.
- Preprocesamiento sin hiperparámetros, ajustado una vez por ventana con su entrenamiento (logaritmo con signo,
  mediana, estandarización, one-hot y codificación por objetivo fuera de pliegue). SMOTE y ADASYN, también una vez por
  ventana y solo con su entrenamiento.
- Hiperparámetros: rangos amplios y en escala logarítmica donde el efecto es multiplicativo (tasa de aprendizaje,
  hojas mínimas, penalización $\alpha$, vecinos).
- Reproducibilidad: semilla 2026; cada trabajo escribe su JSON y no se repite si ya existe (reanudable); las matrices
  preprocesadas se guardan en disco; el código está en `src/validacion.py`, `src/optimizadores.py` y
  `src/experimentos.py`; los resultados, en `resultados/`.
- Decisiones computacionales: 896 trabajos en un servidor de 32 núcleos, 30 a la vez, un hilo por trabajo; 6,7 h de reloj
  y 200,6 h de CPU. Cuatro recortes de costo, medidos con un piloto: logística y SVM lineal con SGD (`max_iter = 30`),
  KNN sobre 30 componentes principales y bosque de 100 árboles.

**Honestidad sobre los recortes.** La «logística» de las 112 es un `SGDClassifier`, no la logística exacta del modelo
base; por eso no se compara su cifra con la del base. E = 20 es un presupuesto de costo, no un óptimo.

## Criterio 5: resultados y evaluación crítica (3 min, 15 %)

**Qué hay que decir.**

- Métricas, cada una con su razón. AUC: ordena por riesgo sin depender del umbral ni de la prevalencia; es lo que el
  optimizador maximiza. F1 de la clase tarde, al umbral fijado en validación: es la decisión. PR-AUC (su referencia es la
  prevalencia). Brier y ECE: calibración, porque el objetivo declarado es que la probabilidad sirva como riesgo.
- Línea base y alternativos: Dummy (AUC 0,5; F1 0,479), regla del proveedor (0,670), logística (0,825 en 2026) y los
  siete modelos de las 112. La logística le saca 0,155 de AUC a la regla.
- XGBoost sin balanceo queda primero en los ocho bloques (rango medio 1,00). Con Nemenyi se separa del árbol, KNN y Naive
  Bayes, pero no de la logística, el bosque ni el SVM; con Wilcoxon y Holm le gana a los seis (p = 0,047). Dentro de
  cada bloque, con DeLong, supera a los tres más cercanos en seis o siete de los ocho y empata en el resto; la
  diferencia media (0,009 a 0,013) tiene un intervalo BCa por órdenes que no toca el cero.
- Lo que pesa es el modelo, no el balanceo ni el optimizador: dentro de un modelo, sus 16 combinaciones no se separan
  más de 0,012 de AUC.
- Error y sobreajuste. Lo que ya hay: con la logística en 2026, de 16.374 tardías marcó 11.672 (71,3 %) y dio 7.929 falsas
  alarmas (22,3 % de las 35.567 a tiempo); en cada decil la tasa observada queda a 4,2 puntos o menos de la predicha. La
  curva de aprendizaje de la logística muestra sesgo, no varianza (brecha de 0,066 a 0,037). En XGBoost, el AUC interno
  (0,8542) y el externo (0,8517) difieren en 0,003, lo que indica poco sobreajuste de la selección de hiperparámetros.
  XGBoost en 2026, con Platt y umbral congelados en 2025: AUC 0,848 [0,835; 0,860] (BCa por órdenes), PR-AUC 0,758,
  F1 0,675, ECE 0,022. De 16.374 tardías marcó 11.341 (69,3 %) con 5.889 falsas alarmas (16,6 %). Yerra más con alguna
  entidad nueva (AUC 0,821 contra 0,856) y en ganadería (0,712). Brecha: interno 0,854, bloques 0,852, 2026 0,848.
- Coherencia con los objetivos: el objetivo 1 (probabilidad como riesgo) se cumple con Platt (ECE 0,022, como la
  logística); el objetivo 2 (qué variables) lo responde SHAP: plazo hábil al liberar (0,85 de |SHAP| medio), proveedor
  (0,46) y, muy por debajo, promesa, experiencia con el centro, cuenta, liberación y material.
- Dato útil para el comprador: con la logística, para encontrar el 10 % de las tardías acierta el 97 % de lo que marca.

## Criterio 6: conclusiones y limitaciones (2 min, 10 %)

**Conclusiones sustentadas.**

1. Sí se puede anticipar el retraso al liberar la orden: la logística, con solo lo conocido ese día, ordena con AUC 0,825
   en 2026 frente a 0,670 de la regla del proveedor.
2. El modelo decide la comparación: XGBoost sin balanceo (AUC 0,852 ± 0,020, F1 0,676 ± 0,048 en ocho bloques) y gana en
   los ocho. El balanceo no mejora el ordenamiento y con 20 evaluaciones Optuna y el genético llegan antes a lo mismo.
3. La señal está repartida: ninguna variable pasa de 0,75 de AUC; plazo, historial de quien participa y la aprobación.
4. En 2026 XGBoost da 0,848 y le gana a la logística por 0,023 [0,017; 0,030]: la ventaja se sostiene en un año que
   ningún modelo vio. La propuesta del Entregable 3 le gana a su vez por 0,012 [0,004; 0,018].

**Limitaciones.**

- *Teóricas.* Observaciones no intercambiables (órdenes y tiempo); XGBoost no da incertidumbre por sí mismo; el F1 supone
  que un falso positivo y uno negativo cuestan parecido, y no se conoce el costo real.
- *Empíricas.* Censura al final del periodo (la tasa de 2026 puede estar subestimada); maestros en foto actual (se
  retiraron los que el planificador ajusta tras ver retrasos, pero otros pueden filtrar algo); una sola empresa; arranque
  en frío (11,1 % de proveedores nuevos en 2026); ocho bloques no independientes (las ventanas se solapan) y p de Wilcoxon
  en el mínimo posible; E = 20; recortes de costo (SGD, PCA en KNN, 100 árboles); $\lambda$ y $\gamma$ sin optimizar;
  deriva de covariables en la tasa previa del comprador.
- *Uso.* El modelo no evalúa personas: el historial del comprador entra como contexto del proceso. Las asociaciones no
  son causales.
- *Mejoras.* Más bloques o repeticiones, mayor presupuesto, ajustar $\lambda$ y $\gamma$, calibración posterior si hace
  falta (Platt), umbral por costo de negocio, ventana adaptativa para la deriva, un tratamiento específico del arranque
  en frío.

## Criterio 7: dominio y preguntas (2 min, 10 %)

- Los dos exponen y los dos responden; cada uno debe poder decir las fórmulas del criterio 3 sin mirar.
- Derivaciones cortas a mano: $g = p - y$ y $h = p(1-p)$; el valor óptimo de la hoja; por qué la codificación por
  objetivo va fuera de pliegue; por qué el empate cuenta 0,5 en el AUC; la diferencia crítica de Nemenyi; el p mínimo de
  Wilcoxon con $N = 8$ ($2/2^8$).
- Si no se sabe: decir qué se mediría para saberlo (más bloques, una prueba sobre la prueba 2026, una ablación) en vez
  de adivinar.
- Pedir que repitan la pregunta si es ambigua, y separar lo que se midió de lo que se argumenta.

## Preguntas difíciles: formulación, datos y fuga

**¿Por qué partición cronológica y no al azar?** Porque el modelo se usaría sobre órdenes futuras, y el proceso cambia: la
tasa de retraso baja de 42,7 % (2021) a 26,9 % (2023) y sube a 31,5 % (2024), y la tasa mensual no es estacionaria (ADF
p = 0,70, KPSS p = 0,026). Una partición al azar mezclaría años y daría un desempeño optimista, porque aprendería un nivel
de retraso que ya no existe y vería el futuro de lo que predice. La validación cruzada también es cronológica
(ventana creciente).

**¿Cómo evitaron la fuga en la codificación por objetivo?** La tasa de retraso de un proveedor, calculada con todo el
entrenamiento, incluye el desenlace de la propia fila: la fila se vería a sí misma. Se calcula **fuera de pliegue** (cada
fila recibe la tasa de los demás pliegues) y **agrupada por orden** (para que no vea el desenlace de las otras posiciones de
su orden, que comparten proveedor y fechas). Se ajusta dentro del `Pipeline` con solo el entrenamiento de cada ventana;
validación y prueba solo transforman. Se suaviza hacia el nivel superior con $M = 20$, y un valor nuevo recibe la tasa de su
padre. Comprobación: el AUC univariado del proveedor es 0,695; con fuga sería cercano a 1.

**¿Cómo saben que no hay otras fugas?** Cada grupo de variables se justificó contra el momento de la predicción (la
última liberación de la orden). Se retiraron el desenlace con otra regla, las fechas de entrada y factura, los cambios
de fecha posteriores y los maestros que el planificador ajusta después de ver retrasos. Los historiales se cortan en la
fecha de creación y cuentan solo posiciones con entrada anterior. Ninguna variable pasa de 0,75 de AUC. El límite: los
maestros restantes son la foto actual del ERP, no la del día de la orden.

**El plazo hábil al liberar tiene AUC 0,732: ¿no será fuga?** Se calcula con la fecha prometida vigente y con el día de la
liberación: ambos se conocen al predecir. Separa porque es la causa más directa del retraso (sin plazo no hay forma de
llegar a tiempo), no porque contenga el desenlace.

**¿Por qué un día hábil de tolerancia?** Es la regla con la que se mide el retraso; se cuenta en días hábiles con los
festivos de Colombia para que un fin de semana no cuente como atraso.

**¿Por qué clasificación y no regresión sobre los días de retraso?** Porque la decisión del comprador es binaria y porque
la probabilidad calibrada se usa como riesgo. Los días de retraso son otra formulación: se retiraron del dataset.

**¿Cómo trataron los faltantes y por qué no hicieron la prueba de Little?** Son estructurales: la posición no tiene el
objeto (solicitud, imputación, material). Mecanismo MAR por las marcas `tiene_*`; si fuera MCAR la tasa de tarde sería igual
con y sin dato y difiere hasta en 4 puntos. Se imputa con la mediana y la marca conserva el vacío (el mismo
preprocesamiento sirve a los siete modelos). Little supone normalidad multivariada y con 136 variables mixtas y 270.000
filas rechazaría siempre.

**¿Por qué no eliminaron los atípicos?** Una compra grande es una compra grande. Isolation Forest marca el 2,9 %, y llegan
tarde el 40,4 % frente a 33,7 %: son los casos que interesa anticipar. Para los lineales se comprimen las colas con el
logaritmo con signo; los árboles son invariantes a transformaciones monótonas.

**¿Qué pasa con los proveedores nuevos?** Es el arranque en frío: en 2026, 11,1 % de las posiciones trae un proveedor que
no estaba en 2021-2024 y 9,7 %, un material nuevo. Sin historial, la codificación les da la tasa de su nivel superior
(la región del proveedor, el grupo de artículos), así que el modelo los trata como «un proveedor promedio de su región».
Con XGBoost en 2026, las órdenes con alguna entidad nueva (20 % de la prueba) bajan a AUC 0,821, contra 0,856 con todo
conocido. Ni la propuesta del Entregable 3 lo mejora de forma significativa: el arranque en frío sigue abierto.

**¿Las posiciones de una misma orden no son dependientes?** Sí, comparten proveedor, fechas y comprador. Por eso las
unidades independientes son las 104.614 órdenes, ninguna se parte entre particiones, la validación agrupa por orden y los
intervalos se calculan con bootstrap de órdenes.

## Preguntas difíciles: modelo y métricas

**¿Por qué no balancear?** La prevalencia es 31 a 34 % y la clase minoritaria tiene más de 90.000 casos: el desbalance es
moderado. El AUC no depende de la proporción de clases, y el umbral se elige en validación, lo que ya corrige el
corrimiento de la prior. Lo medido: el AUC medio de las 112 va de 0,8086 (ADASYN) a 0,8112 (SMOTE), y SMOTE y ADASYN
duplican el cómputo sin ganar nada en XGBoost. ADASYN baja el AUC de la logística unos 0,006: concentra las sintéticas en
la frontera, donde una frontera lineal se deja arrastrar. En KNN y Naive Bayes, «pesos» y «ninguno» dan lo mismo porque
solo cambian la prior, una transformación monótona del puntaje.

**¿Por qué AUC y F1, y no exactitud?** La exactitud engaña: decir siempre «a tiempo» acierta el 66 %. El AUC mide qué tan
bien se ordenan las posiciones por riesgo, sin umbral, y es lo que optimiza cada búsqueda. El F1 mide la decisión al
umbral elegido (media armónica de precisión y sensibilidad de la clase que importa). Se agregan PR-AUC (su referencia es
la prevalencia) y Brier y ECE (calibración). Límite del F1: ignora los verdaderos negativos y supone costos simétricos.

**¿Por qué el umbral se elige por F1 y en validación?** Si se eligiera en la prueba, la prueba dejaría de ser independiente.
Se usa F1 porque el costo real de un falso positivo y uno negativo no se conoce; con ese costo, el umbral se elegiría
minimizando el costo esperado.

**¿Por qué XGBoost y no una red neuronal u otro?** Los datos son tabulares con mezcla de numéricas, categóricas y
relaciones no lineales con codo, donde el boosting de árboles es el estándar. Es un argumento, no un resultado: la red no
fue una de las siete alternativas, y la comparación solo vale dentro de ellas.

**¿Qué significa AUC 0,85 para un comprador?** Que si toma una posición que llegará tarde y una que llegará a tiempo, el
modelo le asigna más riesgo a la tardía el 85 % de las veces. Para una decisión concreta se usa el umbral: con la
logística, para encontrar el 10 % de las tardías acierta el 97 % de lo que marca.

**¿Es bueno un F1 de 0,676?** Se lee contra la referencia de marcar todo, que da 0,479 en 2026 (precisión igual a la
prevalencia), y contra la regla del proveedor, 0,509. Y varía entre bloques: desviación de 0,048.

**¿Por qué la logística es el modelo base y cuánto le gana XGBoost?** Es el modelo más simple con todas las predictoras:
si uno complejo no le gana con claridad, no se justifica. Con el mismo protocolo (ocho bloques), la mejor logística queda en
0,8394 y XGBoost en 0,8517, 0,012 más; el 0,825 de la logística en 2026 es otro protocolo y no se resta del 0,852.

**¿Por qué XGBoost supera a una logística si el problema parece lineal en el logit?** Porque la relación con el plazo no es
lineal (codo: con plazo agotado llega tarde el 85,9 %, con más de 10 días cerca del 20 %) y hay interacciones como plazo
corto con proveedor lento. La curva de aprendizaje de la logística dice sesgo: le falta flexibilidad.

**Los coeficientes de la logística: ¿se pueden interpretar?** Condicionales y encogidos, no causales. El bloque de plazo
tiene VIF de más de 300, así que solo la suma de los coeficientes de esas variables se puede leer (−1,6).

**¿La probabilidad es confiable?** La logística: ECE 0,021 y desvíos de a lo sumo 4,2 puntos por decil. XGBoost crudo es
sobreconfiado (en validación, ECE medio 0,040 y pendiente de Platt de 0,61 a 0,95): la log-loss externa toca su mínimo
mucho antes que el AUC, entre 55 y 501 árboles. Con Platt ajustado en 2025, en 2026 da ECE 0,022.

**¿Qué hacen $\lambda$ y $\gamma$ en XGBoost?** Penalizan los pesos de las hojas ($\lambda$) y el número de hojas ($\gamma$). En
este proyecto quedaron en el valor por defecto; la regularización viene de $\eta$, la profundidad, el peso mínimo por hoja y
los submuestreos. Es una limitación declarada.

## Preguntas difíciles: diseño experimental y comparación estadística

**¿Qué significa que Nemenyi no separe a XGBoost de la logística pero Wilcoxon sí?** Son dos preguntas distintas.
Nemenyi compara los 21 pares a la vez con una sola diferencia crítica (3,19 con 7 modelos y 8 bloques): corrige por muchas
comparaciones y solo mira los rangos medios, así que es conservadora. XGBoost tiene rango medio 1,00 y la logística 2,625:
la diferencia, 1,6, no llega a 3,19. Wilcoxon responde solo si el mejor le gana a cada uno (seis comparaciones, con Holm) y
usa las diferencias pareadas de AUC por bloque. No hay contradicción de dirección: XGBoost gana en los ocho bloques; lo que
cambia es la potencia. Y hay que decir que el resultado de Wilcoxon es el mínimo que ocho bloques pueden dar: p exacto
$2/2^8 = 0{,}0078$, que con Holm por seis comparaciones queda en 0,047, apenas bajo 0,05.

**¿Ocho bloques son suficientes?** Para decir que XGBoost lidera, alcanzan; para decir más, no. Además los bloques no son
independientes: las ventanas crecientes comparten entrenamiento, y las pruebas suponen independencia. Es una limitación.
Por eso se miró también dentro de cada bloque, con las predicciones: DeLong en cada uno y un intervalo BCa que remuestrea
órdenes. La ventaja se sostiene ahí (seis o siete bloques significativos frente a los más cercanos, ninguno en contra).
En 2026 se sostiene: 0,848 contra 0,825 de la logística, Δ 0,023 [0,017; 0,030]. Y la semilla no la explica: la ventaja
sobre el bosque va de 0,008 a 0,011 con diez semillas, catorce veces lo que mueve la semilla.

**¿Por qué comparar los modelos con la combinación de mayor AUC interno y no con la de mayor AUC externo?** Elegir por el
AUC externo sería escoger con el mismo dato que se quiere comparar; el AUC interno es donde se eligió y no sesga la
comparación.

**¿Por qué la comparación es pareada por bloque?** Todos los modelos suben y bajan juntos: 2024-S2 es el más fácil para
los siete. La variación entre bloques (de 0,82 a 0,89 en XGBoost) es mayor que la diferencia entre los mejores, y una
comparación no pareada se ahogaría en esa variación.

**¿Qué es la prueba de Friedman y por qué la corrección de Iman-Davenport?** Ordena los modelos dentro de cada bloque y
contrasta si los rangos medios difieren más de lo esperable por azar, sin suponer normalidad. Iman y Davenport la convierten
en una $F$, menos conservadora: aquí $\chi^2 = 45{,}6$ ($p = 3{,}5 \cdot 10^{-8}$) y $F = 135{,}5$.

## Preguntas difíciles: pruebas sobre las predicciones (DeLong, Cliff, BCa)

**¿Por qué DeLong y por qué entra la covarianza?** Compara dos AUC sobre las mismas posiciones. Los dos modelos puntúan los
mismos casos y sus AUC están correlacionados: la varianza de la diferencia es $\mathrm{Var}_1 + \mathrm{Var}_2 - 2\,\mathrm{Cov}$,
y ignorar la covarianza subestima la potencia. El empate cuenta 0,5 porque el AUC es $P(s^{+} > s^{-}) + \tfrac{1}{2} P(s^{+} = s^{-})$.
Se aplicó en cada uno de los 8 bloques, XGBoost contra los otros seis (48 pruebas, Holm): supera al bosque en 6, a la
logística y al SVM en 7, y al árbol, KNN y Naive Bayes en 8 (z de 19 a 32 en el peor bloque). Donde no es significativo
es empate (diferencias de ±0,001), nunca un rival por encima. La implementación es propia (Sun y Xu, 2014) y coincide con
`pROC::roc.test` de R hasta la décima cifra, en seis casos sintéticos y en el bloque 2025-S2. En 2026, XGBoost contra la
logística: z = 22,5; la propuesta contra XGBoost: z = 12,9.

**¿DeLong no supone filas independientes?** Sí, y las posiciones de una orden no lo son: su error estándar queda corto y
sus p, demasiado pequeños. Por eso no se lee solo: va junto al intervalo BCa, que remuestrea órdenes completas y sí
recoge esa dependencia. Los dos dicen lo mismo aquí.

**¿Para qué el delta de Cliff si ya hay p?** Con muchas filas todo p sale significativo; el delta da el tamaño del efecto
(va de −1 a 1; 0,147, 0,33 y 0,474 separan despreciable, pequeño, mediano y grande). Sobre los ocho AUC por bloque:
frente al bosque 0,25 (pequeño), a la logística 0,34 y al SVM 0,38 (medianos), al árbol, KNN y Naive Bayes de 0,88 a 1
(grandes). Es pequeño con el bosque aunque XGBoost gane 7 de 8 pareados, porque Cliff no parea: compara las dos
colecciones, y la variación entre semestres es mayor que la ventaja.

**¿Por qué el bootstrap es por órdenes?** Remuestrear filas trataría como independientes posiciones que comparten
proveedor, fechas y comprador, y los intervalos saldrían demasiado estrechos. Se remuestrean órdenes completas: 200
réplicas en la prueba 2026 de la logística, 1.000 (dentro de cada bloque) para las 112.

**¿Por qué BCa y no el intervalo de percentiles?** El de percentiles supone que la distribución bootstrap está centrada
en la estimación y que su forma no cambia con el valor del parámetro. BCa corrige las dos cosas: el sesgo $z_0$ (qué
fracción de réplicas cae bajo la estimación) y la aceleración $a$ (estimada con jackknife, cuánto cambia el error
estándar con el parámetro). Con un AUC cerca de 0,85, acotado por 1, la distribución es asimétrica y BCa importa.

## Preguntas difíciles: optimizadores y presupuesto

**¿Por qué el mismo presupuesto de 20 evaluaciones y por qué 20?** Para que la comparación entre optimizadores sea justa:
cuentan evaluaciones, no horas. El 20 fue una decisión de costo (con un piloto, para que cupiera en una noche), no un
óptimo. Con 20, cualquiera de los cuatro deja el modelo a menos de 0,004 de su óptimo interno.

**¿Cuál optimizador es mejor?** Optuna se acerca más rápido: en la evaluación 5 su arrepentimiento es 0,006 contra 0,013
(genético), 0,015 (aleatoria) y 0,042 (rejilla); en la 20, 0,0011 y 0,0013 (genético). En el AUC externo las diferencias se
diluyen (milésimas contra centésimas entre bloques). La rejilla es la peor porque recorre el espacio en orden fijo y, con
seis hiperparámetros y 20 puntos, solo cabe en dos niveles por dimensión (maldición de la dimensión).

**Describe TPE y el algoritmo genético.** TPE modela la densidad de configuraciones buenas $l(x)$ y la de malas $g(x)$ con
un corte por cuantil, y propone el $x$ que maximiza $l(x)/g(x)$, equivalente a maximizar la mejora esperada; arranca con 10
puntos al azar. El genético: población de 6, torneo de 3, cruce uniforme (0,7), mutación gaussiana ($\sigma$ = 0,15, con
probabilidad 0,3), elitismo de 1; un hijo repetido no gasta presupuesto.

**¿El optimizador no se sobreajusta a la validación interna?** El umbral y los hiperparámetros salen del mismo semestre
interno, y eso puede inflar algo el interno. La diferencia entre el AUC interno y el externo de la combinación líder es de
0,003 (0,8542 contra 0,8517): pequeña.

**¿Por qué la logística y el SVM con SGD, el KNN sobre PCA y el bosque de 100 árboles?** Recortes de costo medidos con un
piloto para que las 896 tareas cupieran en una noche. Implican que la «logística» de las 112 no es la del modelo base, y
que KNN y el bosque pueden estar algo subestimados. Se declaran como limitación.

## Preguntas difíciles: resultados, errores y límites

**¿Dónde se equivoca el modelo?** Con la logística: de las 16.374 tardías de 2026 marcó 11.672 y dio 7.929 falsas alarmas.
Los activos fijos (56,2 % de retraso) y la materia prima (64,0 %) son clases con pocas posiciones (5.043 y 1.699) y retrasan
más. XGBoost en 2026 marcó 11.341 de las 16.374 tardías con 5.889 falsas alarmas; AUC 0,821 con entidad nueva y 0,856
sin ella; peor en ganadería (0,712) y mejor en transporte (0,864). El falso negativo típico tenía 13 días hábiles de
plazo al liberar, un proveedor con tasa previa de 0,31 y una liberación rápida: nada anunciaba el retraso.

**¿Hay sobreajuste?** En la logística, la curva de aprendizaje dice que no: la brecha entre entrenamiento y validación se
cierra de 0,066 a 0,037 y es un modelo con sesgo. En XGBoost, el AUC interno y el externo difieren en 0,003, pero esa no es
la brecha entre entrenamiento y validación. La curva de convergencia sí lo es: al final, AUC de entrenamiento 0,92-0,99
contra 0,83-0,89 externo (brecha media 0,12), pero el mejor AUC externo solo está 0,0014 por encima del final. Hay
sobreajuste de probabilidades (la log-loss externa sube), no de orden; lo corrige Platt.

**¿Qué variables explican el retraso?** Tres familias: el plazo real que le queda al proveedor, quién provee y quién pide
(con su historial) y la aprobación interna (días de liberación, estrategia). Con la logística, el coeficiente mayor es el del
proveedor codificado (1,87). Con SHAP en XGBoost (2026): plazo hábil al liberar 0,85 de |SHAP| medio, proveedor 0,46, y
el resto bajo 0,13 (promesa, experiencia con el centro, cuenta, días de liberación, material, creador de la solped).

**¿Por qué XGBoost y no la logística, si es más simple y queda cerca?** Por parsimonia aplicada con cifras: se acepta lo
complejo solo si gana de forma significativa y material, y aquí pasan las dos. (1) Significativa: en 2026, 0,848 contra
0,825, Δ 0,023 [0,017; 0,030], DeLong z = 22,5; en los ocho bloques gana los ocho (Wilcoxon-Holm 0,047), Cliff 0,34. (2)
Material: con el mismo 20 % de alertas atrapa 8.242 tardías contra 7.895, unas 35 más por mes, con precisión 0,79 contra
0,76; F1 0,675 contra 0,649. Ojo: esas tardías de más caen en órdenes que ya tenían alerta (2.362 órdenes contra 2.407).
(3) Forma: sin plazo, el riesgo sube 3,4 de logit en Generales y 2,4 en Producción Cluster; la logística da la misma
pendiente a todas y una caída suave cerca de cero. (4) Costo: un ajuste tarda 3,2 min contra 12 s y mil predicciones 17
ms contra 0,7; nada para unas 5.200 posiciones al mes. SHAP da la lectura. (5) Si el negocio pusiera explicar cada
coeficiente por encima de atrapar tardías, la logística sería defendible: pierde 0,023 de AUC y unas 35 tardías al mes.

**¿La diferencia entre XGBoost y el bosque es real?** Sí, y pequeña. Con las combinaciones de mayor AUC interno gana en
los ocho bloques (Wilcoxon, p Holm 0,047); con las de la tabla de líderes, en siete, y DeLong la da por significativa en
seis (en 2024-S1 y 2025-S2 empatan). La diferencia media es 0,009 con BCa 95 % [0,006; 0,012]: no toca el cero. El delta
de Cliff, 0,25, es pequeño, porque la ventaja es menor que la variación entre bloques (0,02). Nemenyi no los separa. Se
dice «lidera de forma consistente, con una ventaja pequeña».

**¿Se puede usar para decidir hoy?** Con límites. La tasa de 2026 puede estar subestimada por censura; hay deriva de
covariables en la tasa previa del comprador; los maestros son la foto actual; los resultados valen para una sola empresa; y
una asociación no es causa: usarlo como ranking de personas sería un mal uso.

**¿Qué haría con más tiempo?** Más bloques o repeticiones; $\lambda$ y $\gamma$ y un presupuesto mayor; calibración con Platt;
umbral por costo; ventana adaptativa; y tratamiento del arranque en frío.

**¿Qué aporta el estudio?** Un punto de comparación honesto y reproducible: las 112 combinaciones con el mismo presupuesto y
validación anidada, que muestran que decide el modelo y no el balanceo ni el optimizador, y un clasificador que ordena
mejor que la regla del comprador.

## La evaluación en 2026, de dónde sale

`src/prueba_2026.py`, `src/modelo_final.py` y `src/xgb_vs_logistica.py`, sobre las probabilidades de la comparación 2 × 2
(Platt y umbral congelados en 2025). El modelo servido en el tablero las reproduce (diferencia máxima 3·10⁻⁸, misma
decisión en las 51.941 posiciones). Capítulos 6 (mejor modelo) y 9 (modelo original) del libro.
