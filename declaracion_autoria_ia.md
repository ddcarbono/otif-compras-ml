# Declaración de autoría y de uso de inteligencia artificial

**Trabajo:** Clasificador binario para la predicción del incumplimiento de órdenes de compra en la agroindustria de la
palma de aceite (Entregables 1, 2 y 3). **Curso:** Machine Learning, Maestría en Estadística Aplicada, Universidad del
Norte, 2026. **Tablero:** <https://otif-compras-ml.onrender.com>. **Repositorio:** <https://github.com/ddcarbono/otif-compras-ml>.

## Autoría

Declaramos que este trabajo es nuestro: definimos el problema y la pregunta de investigación, obtuvimos y depuramos los
datos de la empresa con su autorización para uso académico, decidimos el diseño de la evaluación (partición
cronológica, validación anidada por semestres, métricas y pruebas), elegimos y formulamos el modelo original y
revisamos cada resultado antes de incluirlo. Respondemos por todo lo que dicen el libro, el tablero y el código.

Las ideas, métodos y cifras de otros autores se citan en el libro con su referencia (`book/references.bib`). Ninguna
parte del código se copió de un repositorio ajeno; las bibliotecas usadas son las que lista `requirements.txt`, y el
modelo original (`original/`) usa solo numpy y scipy, más `BaseEstimator` y `ClassifierMixin` de scikit-learn.

## Uso de inteligencia artificial

Usamos un asistente de IA generativa, **Claude (Anthropic), a través de Claude Code**, entre septiembre y octubre de
2026, en estas tareas:

| tarea | qué hizo el asistente | qué hicimos nosotros |
|---|---|---|
| Código | escribió borradores de los guiones de extracción, preprocesamiento, experimentos, pruebas, tablero y del modelo original, y los ejecutó a pedido | definimos qué debía hacer cada guion, revisamos el código, corrimos y verificamos los resultados contra los datos |
| Revisión de literatura | armó las cadenas de búsqueda a partir de los temas que fijamos, las corrió en OpenAlex y, con nuestra sesión de la biblioteca, en Scopus y Web of Science; tamizó títulos y resúmenes por lotes, resumió en fichas los artículos leídos completos y recomendó familias de métodos | fijamos los temas y los criterios del modelo original, revisamos las fichas y decidimos qué método proponer |
| Redacción | redactó borradores de capítulos, textos del tablero y diapositivas | revisamos, corregimos y aprobamos cada texto; las conclusiones son nuestras |
| Análisis | propuso pruebas estadísticas y diagnósticos, y señaló errores (por ejemplo, fuga de información y comparaciones injustas) | decidimos qué pruebas usar y cómo leerlas |

El asistente no tuvo acceso a datos fuera de los que le dimos y no aparece como autor. Cada configuración se congeló
en la validación 2025, con un commit, antes de medirla en 2026; las veces que se miró la prueba 2026 están declaradas
en el libro. Los datos de entidades (proveedores, materiales, personas) no se publican.

Nos comprometemos a explicar y a modificar en vivo, en la sustentación, cualquier parte del código, en particular el
algoritmo del modelo original.

## Firmas

Daniel Carbonó Domínguez y Luis Pérez, 25 de octubre de 2026.
