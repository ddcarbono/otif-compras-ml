# Literatura

La revisión sirve para dos cosas: ver qué se ha hecho con el mismo problema (predecir si una entrega de un proveedor
llega tarde) y decidir qué cifras de otros trabajos se pueden poner al lado de las de este. Se buscó en OpenAlex, Scopus y
Web of Science y se revisaron unos 90 estudios de retraso de compras y entregas de proveedores. La tabla recoge los que
tienen datos reales de empresa y una definición de retraso cercana; los demás (casi todos sobre el conjunto sintético
DataCo, entregas a cliente final o comercio electrónico) quedan fuera. Donde la fuente es solo el resumen del artículo,
se indica y lo que el resumen no dice figura como «no informa».

| estudio | datos | objetivo y tarea | modelos | validación | métrica y resultado | por qué no es comparable con este trabajo |
|---|---|---|---|---|---|---|
| {cite}`brintrup2020` (texto completo) | fabricante de equipos complejos; 232.912 pedidos, 351 proveedores; periodo no informa | pedido que llega después de lo contratado (13 %, más del 90 % con menos de un día); clasificación | KNN, bosque aleatorio, regresión logística, SVM; submuestreo con *bagging* | 80 % entrenamiento y 20 % prueba, sin decir que sea por tiempo | exactitud de 80 % con la variable de agilidad del proveedor (gana el bosque aleatorio) | la exactitud con clases desbalanceadas no es el AUC; la partición no es temporal y la agilidad pudo calcularse con pedidos posteriores (fuga); excluye proveedores con menos de cinco pedidos, justo los más riesgosos |
| {cite}`steinberg2023` (texto completo) | fabricante alemán de maquinaria a pedido; 119.610 pedidos con 17 campos; periodo no informa | días de desfase contra la fecha de necesidad (14 % tarde); regresión | siete, entre ellos AdaBoost, *gradient boosting*, bosque aleatorio y red neuronal | partición aleatoria 80/20 | error absoluto medio de 8,2 días con la solicitud y 3,2 con el pedido puesto; R² de 92 % y 98 % | otra tarea (regresión) y otra unidad (componentes de producción a pedido); validación aleatoria, con pedidos del mismo proyecto a ambos lados; las variables son diferencias contra la misma fecha que define el objetivo |
| {cite}`gabellini2024` (solo resumen) | fabricante del sector automotor; tamaño y periodo no informa | retraso en días de las entregas, como series; regresión | red profunda con autocorrelación e indicadores macroeconómicos, contra modelos estadísticos | no informa | no informa; dice superar a los modelos de comparación en la mayoría de las predicciones | tarea de pronóstico de series, no clasificación por posición; usa variables macroeconómicas externas; sin cifras comparables |
| {cite}`billah2026` (solo resumen) | compras de materiales de construcción; n no informa | ocurrencia del retraso y duración en días; clasificación y regresión | modelos de conjunto (CatBoost, XGBoost y otros) | 80/20 estratificado al azar, validación cruzada de cinco pliegues con cinco semillas | AUC de 0,958 (CatBoost); en días, R² de 0,72 y MAE de 17,5 días (XGBoost) | validación aleatoria sin orden en el tiempo; otro sector, con proyectos y fabricantes como predictores principales; no es comparable con un AUC medido con partición cronológica |
| {cite}`faulkner2026` (solo resumen) | más de 10 millones de envíos de un socio industrial, cuatro orígenes; periodo no informa | días de retraso de los envíos tardíos; clasificación y regresión juntas | red profunda multitarea, contra árboles en uno y dos pasos | no informa | MAE de 0,67 a 0,91 días en los envíos tardíos | unidad envío (transporte), no posición de compra; mide los días, no si llega tarde; no informa la validación |
| {cite}`raval2026` (solo resumen) | revisión de 26 estudios sobre DataCo (2024 a 2026); reproduce 9 | evaluar la fuga en la predicción de entregas tardías | no aplica | reevalúa con solo datos al pedir, por pedidos agrupados y ordenada en el tiempo | exactitud de 0,67 a 0,70 contra 0,55 de la clase mayoritaria; en un estudio el AUC baja de 0,96 a 0,73 | DataCo es sintético y de entrega a cliente final; no es un modelo sino una auditoría. Sirve como advertencia: 16 de los 26 parten filas al azar |
| {cite}`liu2018` (solo resumen) | datos de compras de un fabricante de equipos originales; tamaño no informa | tiempo de entrega de la orden; regresión | bosque aleatorio y bosque de cuantiles, con reducción de niveles de variables categóricas | no informa | error menor que el de los plazos que da el proveedor; cifra no informa | tarea de regresión; sin AUC ni definición de tarde; no informa validación |
| {cite}`kumar2016` (solo resumen) | caso de una empresa; tamaño no informa | probabilidad de entrega a tiempo de un proveedor | regresión logística, supervivencia en tiempo discreto, Bayes ingenuo | no informa | no informa | evalúa proveedores para seleccionarlos, no posiciones al liberar la orden; sin cifras |
| {cite}`muller2025` (revisión, texto completo) | 29 estudios de 1970 a 2024; 93 % con datos reales de empresa | revisar métodos de predicción de retrasos de entrega | el bosque aleatorio es el más usado (41 % de los clasificadores) y el que más veces gana (48 %) | no aplica | no aplica | es la referencia para situar este trabajo, no un resultado con el que comparar; su crítica central es que la literatura busca décimas de exactitud y descuida el uso real |

Dos reglas guían la lectura. Un AUC solo se compara con otro de la misma definición de retraso, de la misma unidad
(posición de compra, no envío ni pedido a cliente) y de una partición temporal. Y un resultado con partición aleatoria
es una cota superior: las posiciones de una misma orden y de un mismo proveedor caen a ambos lados y el modelo las
reconoce. Por eso ningún número de la tabla se usa como meta; la referencia de este trabajo es la regla del proveedor
(AUC de 0,670 en 2026) y la logística (0,825), medidas con la partición cronológica del capítulo {doc}`03a_modelo_base`.

## Qué aporta este trabajo

Lo que los estudios anteriores no hacen a la vez, y este sí:

- **Validación temporal por semestres.** Ocho bloques de evaluación, de 2022-S1 a 2025-S2, cada uno con ventana
  creciente y un semestre de validación interno para elegir los hiperparámetros; la prueba de 2026 queda aparte.
- **Solo variables conocidas al liberar la orden**, con el objetivo definido sobre la fecha prometida original y no sobre
  la vigente, que ya recoge el retraso.
- **Codificación jerárquica por objetivo fuera de pliegue**, de modo que ninguna posición ve su propio retraso al
  codificar proveedor, material o comprador.
- **Una comparación de 112 combinaciones** (siete modelos, cuatro tratamientos del desbalance y cuatro optimizadores) con
  el mismo presupuesto de 20 evaluaciones cada una, en lugar de un modelo ajustado con más esfuerzo que sus rivales.

No es un método nuevo y no pretende superar las cifras de la tabla: su aporte es un protocolo de evaluación que no
infla el resultado y una comparación controlada, en un sector (palma de aceite) del que esta revisión no encontró estudios.
