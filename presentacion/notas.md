# Notas del expositor: Entregable 2 (10 minutos)

## 1. Clasificador binario para la predicción del incumplimiento de órdenes de compra en la agroindustria de la palma de aceite (0:30)

Presentarse y dar las tres direcciones: repositorio, libro y tablero (este último aún sin publicar). Una frase: predecir, el día en que sale la orden, si la posición llegará tarde.

## 2. Estimamos la probabilidad de que cada posición llegue tarde y decidimos con un umbral elegido en validación (0:45)

x_i son 136 predictoras conocidas el día de liberación; y_i es la etiqueta de retraso (tolerancia 1 día hábil). Se estima p(x) y se decide con un umbral τ. Se pone en validación para no mirar la prueba.

## 3. Ninguna variable ordena sola por encima de 0,75 de AUC: la señal está repartida entre plazo, historial y entidad (1:30)

ETL: de las tablas de SAP a una fila por posición, con la etiqueta y las 136 predictoras. Ver las diferencias por línea y clase, las mejores variables sueltas (ninguna pasa de 0,75) y que los vacíos son estructurales. Fuga: se descartaron variables registradas después del hecho; la codificación por objetivo va fuera de pliegue y agrupada por orden.

## 4. La tasa de retraso se mueve en el tiempo, por eso se parte cronológicamente y nunca al azar (0:45)

La serie mensual no es estacionaria (ADF y KPSS) y cambia de nivel entre 2021 y 2026. Por eso 2021-2024 entrena, 2025 valida y 2026 prueba. También hay entidades nuevas en validación y prueba que el modelo no vio. La etiqueta de lo más reciente madura tarde: es una limitación.

## 5. El boosting minimiza la log-pérdida paso a paso con árboles; la logística hace lo mismo con una sola recta y un óptimo (2:00)

Dos minutos. Boosting: cada árbol es un paso de descenso por gradiente sobre la log-pérdida en el espacio de funciones, usando gradiente y hessiano; el peso de hoja sale cerrado. Regularización: η, hojas, λ, α, submuestreo. Complejidad: histograma. Contraste: logística, convexa, un óptimo, L2/L1. El nombre exacto del mejor benchmark depende de los 112.

## 6. Cada una de las 112 combinaciones se evalúa con el mismo presupuesto, hacia adelante en el tiempo (1:30)

La prueba 2026 se usa una sola vez y no elige nada. La validación anidada evita que el optimizador vea la fila que se evalúa. Mismo presupuesto para comparar optimizadores de forma justa.

## 7. LightGBM supera a la logística en 0,06 de AUC con las cifras de hoy (provisionales) (1:15)

Hoy solo hay el ajuste de LightGBM, la logística y la regla del proveedor. Se muestran como provisionales. Los 112 darán las curvas de optimización, la tabla de líderes y el diagrama de diferencia crítica.

## 8. Con entidades nuevas el AUC baja de 0,860 a 0,823: el costo del arranque en frío (provisional) (0:45)

Calibración: lo predicho coincide con lo observado por decil. Con entidades nuevas el AUC baja. Pendiente: errores por línea, sobreajuste (brecha entrenamiento-validación) y SHAP para el mejor de los 112.

## 9. El retraso se anticipa mejor que con la regla del proveedor, con límites claros de periodo, maestros y entidades nuevas (1:00)

Cerrar con la respuesta y sus cifras (provisionales), y con las cuatro limitaciones: censura, maestros en foto actual, una sola empresa y entidades nuevas. Los dos autores responden preguntas.

