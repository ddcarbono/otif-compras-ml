# Clasificador binario para la predicción del incumplimiento de órdenes de compra en la agroindustria de la palma de aceite

**Autores:** Daniel Carbonó y Luis Pérez. Maestría en Estadística Aplicada, curso de Machine Learning, Universidad del
Norte, 2026.

**Tablero:** https://otif-compras-ml.onrender.com · **Repositorio:** https://github.com/ddcarbono/otif-compras-ml

## Resumen

Una de cada tres posiciones de las órdenes de compra de un grupo agroindustrial de la palma de aceite llega más de un
día hábil después de la fecha prometida. Este libro construye un clasificador binario que estima, el día en que la
orden se libera hacia el proveedor, la probabilidad de que cada posición llegue tarde. Los datos son 406.126 posiciones
de 104.614 órdenes creadas entre 2021 y 2026, extraídas del ERP de la empresa y anonimizadas, con 136 predictoras
conocidas al liberar. La partición es cronológica: entrenamiento 2021-2024, validación 2025 y prueba 2026.

## Mapa del libro

| capítulo | qué responde |
|---|---|
| {doc}`literatura` | qué se ha publicado sobre predecir retrasos de proveedores y por qué esas cifras no se comparan con estas |
| {doc}`02a_datos` | el problema, la fuente, la población, el diccionario, la calidad de los datos y la ética |
| {doc}`02b_eda` | cómo se distribuye el objetivo y qué variables se asocian con él |
| {doc}`02c_tiempo_y_fuga` | cómo cambia el retraso en el tiempo, qué tan distinta es la evaluación del entrenamiento, qué no puede entrar al modelo y cómo se preprocesa |
| {doc}`03a_modelo_base` | cuánto logra una regresión logística frente a no tener modelo y frente a la regla del proveedor |
| {doc}`03b_112_combinaciones` | qué modelo, balanceo y optimizador funcionan mejor en la validación anidada, y por qué XGBoost y no la logística |
| {doc}`03d_mejor_modelo` | cuánto logra el ganador en la prueba 2026 y qué mira (SHAP) |
| {doc}`03c_fundamentos` | cómo funciona XGBoost por dentro y cuánto cuesta cada modelo, en teoría y medido |
| {doc}`05_modelo_original` | la propuesta del Entregable 3 (efectos por entidad, plazo log-normal, plazo real), cómo se eligió y la comparación 2 × 2 en 2026 |
| {doc}`06_validacion` | si la propuesta gana también en siete conjuntos públicos, y dónde no |
| {doc}`glosario` | qué mide y cómo se lee cada estadístico del libro |

## Cómo reproducirlo

El código está en el repositorio. Los cuadernos se generan desde `book/fuentes/*.py` con `python book/compilar.py` y el
libro con `jupyter-book build book/`. El conjunto de datos completo no está en el repositorio (ver el capítulo
{doc}`02a_datos`); se lee de la variable de entorno `OTIF_CSV`. Semilla global: 2026.

## Referencias

```{bibliography}
```
