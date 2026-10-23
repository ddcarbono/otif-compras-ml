# Notas del expositor

Las escribe `construir.py` (también van en las notas de cada diapositiva). A = Daniel Carbonó, B = Luis Pérez; cada uno ensaya también lo del otro.

| # | min | criterio | expone | qué decir |
|---|---|---|---|---|
| 1 | 0:30 | criterio 1 | A | Presentarse los dos. Una frase: el día en que la orden se libera, predecir si cada posición llegará tarde. Las tres direcciones quedan en pantalla; el tablero se abre al final si hay tiempo. |
| 2 | 0:50 | criterio 1 | A | Problema, unidad, x, y, p(x) y el umbral. Insistir: el umbral sale de 2025; la prueba 2026 se mira una vez. Por qué binaria: la decisión es binaria. |
| 3 | 0:40 | criterio 2 | A | Fuente y estructura. La unidad independiente es la orden: por eso los pliegues se agrupan por orden. La partición es cronológica. |
| 4 | 0:40 | criterio 2 | A | Lo más importante: ninguna variable sola es sospechosa (máximo 0,732) y los vacíos son estructurales, con su mecanismo. Los atípicos llegan más tarde: son información. |
| 5 | 0:40 | criterio 2 | A | Tres pruebas de que el tiempo importa: la serie no es estacionaria, hay entidades nuevas y la validación adversaria reconoce el año. Fuga: la codificación por objetivo va fuera de pliegue y agrupada por orden, y se descartaron las variables registradas después de la liberación. |
| 6 | 1:00 | criterio 3 | B | Boosting: descenso por gradiente en el espacio de funciones. Derivar en voz alta g y h de la log-pérdida: dℓ/dF = p − y; d²ℓ/dF² = p(1 − p). Es lo que más preguntan. |
| 7 | 0:45 | criterio 3 | B | w* = −G/(H + λ) sale de derivar la parábola e igualar a cero. La ganancia compara la pérdida óptima de las dos hojas hijas con la de la madre; γ es el costo de una hoja más. |
| 8 | 0:45 | criterio 3 | B | Cada hiperparámetro y dónde vive en la fórmula. Complejidad: teoría y medición coinciden; el exponente 0,91 en log-log es casi lineal. Supuesto clave: independencia dado x. |
| 9 | 0:45 | criterio 4 | B | Validación anidada: el externo mide, el interno elige. La prueba 2026 no entra. Mismo presupuesto para comparar optimizadores de forma justa. Reproducibilidad: pruebas y CI. |
| 10 | 0:45 | criterio 4 | B | Lectura: con poco presupuesto conviene Optuna; con 20 evaluaciones la superficie es plana cerca del óptimo y los cuatro terminan a milésimas. El balanceo tampoco cambia el AUC. |
| 11 | 1:00 | criterio 5 | A | Media y desviación del bucle externo, 8 semestres. Ser precisos con las pruebas: Friedman rechaza la igualdad; Nemenyi es conservador y no separa a los tres primeros; el Wilcoxon pareado con Holm sí, porque XGBoost gana en cada bloque. La ganancia es de una centésima de AUC. |
| 12 | 1:00 | criterio 5 | B | Sobreajuste en dos vistas: interno contra externo por bloque y la curva de aprendizaje. Lo que falta del análisis de errores depende de la evaluación final en 2026; no se adelanta. |
| 13 | 1:00 | criterio 6 | A | Responder la pregunta con cifras y decir los límites sin rodeos. Mejoras: el modelo original del Entregable 3 ataca el arranque en frío y usa los días de retraso. |
| 14 | 2:00 | criterio 7 | A y B | Preparar: derivar g, h y w*; por qué la codificación va fuera de pliegue; por qué la prueba no elige nada; por qué Nemenyi y Wilcoxon dicen cosas distintas; qué pasa con un proveedor nuevo. |
