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
| {doc}`02a_datos` | el problema, la fuente, la población, el diccionario, la calidad de los datos y la ética |
| {doc}`02b_eda` | cómo se distribuye el objetivo y qué variables se asocian con él |
| {doc}`02c_tiempo_y_fuga` | cómo cambia el retraso en el tiempo, qué no puede entrar al modelo y cómo se preprocesa |

## Cómo reproducirlo

El código está en el repositorio. Los cuadernos se generan desde `book/fuentes/*.py` con `python book/compilar.py` y el
libro con `jupyter-book build book/`. El conjunto de datos completo no está en el repositorio (ver el capítulo
{doc}`02a_datos`); se lee de la variable de entorno `OTIF_CSV`. Semilla global: 2026.

## Referencias

```{bibliography}
```
