# Notas del expositor

Las escribe `construir.py` (también van en las notas de cada diapositiva). A = Daniel Carbonó, B = Luis Pérez; cada uno ensaya también lo del otro. Rúbrica de 20 minutos: 18 de exposición y 2 de preguntas.

| # | min | criterio | expone | qué decir |
|---|---|---|---|---|
| 1 | 0:15 | criterio 1 | A | Presentarse los dos y dejar las tres direcciones en pantalla. Render ya abierto desde antes: el plan gratuito se duerme. |
| 2 | 0:45 | criterio 1 | A | El problema en una frase y su magnitud. La tercera cifra es del modelo base en 2026 y ya está en el libro: marcando poco más de un tercio, se ven venir siete de cada diez tardías. |
| 3 | 1:00 | criterio 1 | A | x, y, el modelo Bernoulli, la log-pérdida como menos log-verosimilitud y el umbral. Insistir: τ sale de 2025 y 2026 no elige nada. |
| 4 | 0:45 | criterio 2 | A | Pantalla en vivo. Si el tablero no carga, la captura queda en la diapositiva. Mostrar el filtro por línea: industrial y palma llegan tarde más que banano y ganadería. |
| 5 | 0:45 | criterio 2 | A | Fuente y estructura. La unidad independiente es la orden: por eso los pliegues se agrupan por orden. La partición es cronológica. |
| 6 | 0:30 | criterio 2 | A | Distribuciones: colas largas, transformación justificada. Faltantes: el mecanismo es estructural, no aleatorio. Atípicos: tienen más retraso, se quedan. |
| 7 | 0:30 | criterio 2 | A | Relaciones: la señal está repartida. Fuga: lo que se retiró y por qué, y la codificación por objetivo fuera de pliegue, que es la fuga más fácil de cometer aquí. |
| 8 | 0:30 | criterio 2 | A | Tres pruebas de que el tiempo importa: la serie no es estacionaria, hay entidades nuevas y la validación adversaria reconoce el año aun sin las variables de reloj. |
| 9 | 1:15 | criterio 3 | B | Qué es el modelo, qué pérdida minimiza y por qué esa pérdida (Bernoulli). Principio: gradiente en el espacio de funciones. Contraste con el bosque: sesgo contra varianza. |
| 10 | 1:15 | criterio 3 | B | Derivar en voz alta: σ′ = σ(1 − σ); dℓ/dF = σ − y; d²ℓ/dF² = σ(1 − σ). γ cobra por hoja y λ encoge los pesos. Es la derivación que más preguntan. |
| 11 | 1:15 | criterio 3 | B | w* = −G/(H + λ) sale de derivar la parábola e igualar a cero. La ganancia compara la pérdida óptima de las dos hojas hijas con la de la madre; γ es el costo de una hoja más. |
| 12 | 1:15 | criterio 3 | B | Cada hiperparámetro y dónde vive en la fórmula. Complejidad: teoría y medición coinciden; el exponente 0,91 en log-log es casi lineal. Supuesto clave: independencia dado x, que la agrupación por orden protege. |
| 13 | 1:00 | criterio 4 | B | Validación anidada: el externo mide, el interno elige. Mismo presupuesto para comparar optimizadores de forma justa. La prueba 2026 se mira una vez, al final. |
| 14 | 1:00 | criterio 4 | B | Por qué ese espacio: rangos usuales de la literatura, escala logarítmica donde el efecto es multiplicativo (tasa, peso mínimo). Decisiones computacionales: matrices preprocesadas en disco, SMOTE calculado una vez por ventana. |
| 15 | 1:00 | criterio 4 | A | Optuna (TPE) aprende de los ensayos y propone donde la mejora esperada es mayor; DEAP llega tarde pero llega. Con poco presupuesto conviene Optuna; con 20 la superficie es plana. |
| 16 | 1:00 | criterio 5 | B | Por qué cada métrica. La ganancia de XGBoost es de una centésima de AUC frente al bosque y la logística: real pero pequeña. La comparación en 2026 llega con la evaluación final. |
| 17 | 1:00 | criterio 5 | A | En vivo: la predicción la sirve hoy el modelo base (logística), con sus contribuciones. Pruebas: Friedman rechaza la igualdad; Nemenyi es conservador con 8 bloques; el Wilcoxon pareado con Holm separa porque XGBoost gana en cada bloque. |
| 18 | 1:00 | criterio 5 | B | Sobreajuste en dos vistas: interno contra externo por bloque y la curva de aprendizaje. Lo que falta del análisis de errores depende de la evaluación final en 2026; no se adelanta. |
| 19 | 1:00 | criterio 6 | A | Responder la pregunta con cifras y el veredicto de cada objetivo, sin adelantar lo que depende de 2026. |
| 20 | 1:00 | criterio 6 | B | Separar lo teórico (supuestos del modelo) de lo empírico (los datos). Las mejoras apuntan al modelo original. |
| 21 | 2:00 | criterio 7 | A y B | Preparar: derivar g, h y w*; por qué la codificación va fuera de pliegue; por qué la prueba no elige nada; por qué Nemenyi y Wilcoxon dicen cosas distintas; qué pasa con un proveedor nuevo. |
