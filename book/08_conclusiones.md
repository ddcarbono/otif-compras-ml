# Conclusiones

## Hallazgos

- **El problema se puede predecir el día en que la orden se libera.** Con 136 variables conocidas ese día, el XGBoost
  elegido entre 112 combinaciones en una validación anidada por semestres da en la prueba 2026 un ROC-AUC de 0,848
  [0,835; 0,860] y F1 0,675, contra 0,825 de la regresión logística y 0,670 de la regla del proveedor
  ({doc}`03d_mejor_modelo`). No cae de la validación a la prueba.
- **Lo que pesa** es el plazo que le queda al proveedor al liberar y quién provee; después, la promesa en días, la
  experiencia del proveedor con el centro y el proceso interno de liberación (SHAP; LIME coincide en el signo de lo
  principal, pero cambia con la semilla).
- **El modelo original, BOREAL, es el mejor medido en 2026**: ROC-AUC 0,860 [0,847; 0,871], F1 0,690 y ECE 0,014,
  contra 0,848, 0,675 y 0,022 del XGBoost de las 112. La diferencia de AUC es 0,011 [0,003; 0,017] con intervalos que
  remuestrean órdenes ({doc}`05_modelo_original`).
- **Modelo e información son dos caminos a la misma ganancia.** Con las mismas 136 variables, la estructura de BOREAL
  le suma 0,009 de AUC a XGBoost; las 32 variables de plazo real le suman 0,010 a XGBoost; juntas, 0,011, porque se
  sustituyen entre sí. El plazo log-normal suma 0,003. En las órdenes con alguna entidad nueva ningún modelo le gana a
  otro: el arranque en frío sigue abierto.
- **Fuera de OTIF**, la pieza central de BOREAL, el boosting mixto, gana en tres de siete conjuntos públicos (diabetes,
  citas no asistidas y buses de Nueva York), empata en cuatro y no pierde en ninguno. Gana donde las entidades tienen
  historia propia en el tiempo ({doc}`06_validacion`).

## Veredicto sobre R1, R2 y R3

| requisito | veredicto | evidencia |
|---|---|---|
| **R1, novedad en la aplicación** | **sí** | Búsqueda en OpenAlex, Scopus y Web of Science con 26 cadenas en inglés y 5 en español, con fecha (7.331 artículos); seis leídos completos. Ningún estudio de retraso de compras usa efectos aleatorios por entidad, la reducción ordinal de los días de retraso ni efectos ordenados en el tiempo, y ningún método publicado combina los seis rasgos de BOREAL. Cada pieza tiene antecedente y se cita; lo nuevo es la combinación y su aplicación ({doc}`04_novedad`). |
| **R2, implementación algorítmica propia** | **sí** | `Boreal` hereda de `BaseEstimator` y `ClassifierMixin`, y todo el algoritmo (árboles de histograma, efectos cruzados por Newton y EM, reducción ordinal con monotonía, efectos ordenados, TreeSHAP) está escrito con numpy y scipy, sin paquetes de modelos. Tiene formulación, pseudocódigo, complejidad teórica contrastada con la medida, 11 pruebas `pytest` (aprendizaje sintético, reproducibilidad, entradas inválidas, `clone`, `Pipeline` con `cross_val_score`, solo numpy y scipy, TreeSHAP igual a Shapley), ablación (la 2 × 2 y el modelo sin el log-normal) y SHAP comparado con el de XGBoost ({doc}`05_modelo_original`). |
| **R3, superioridad estadística** | **parcial** | En OTIF 2026, BOREAL le gana al ganador de las 112 en AUC por 0,011 [0,003; 0,017] (BCa por órdenes) y DeLong lo confirma después de Holm (p < 0,001); el F1 sube 0,015 [0,004; 0,026]. Las dos pruebas que respetan el tiempo no alcanzan significancia después de Holm sobre las seis pruebas: Diebold-Mariano semanal sobre la pérdida log (p = 0,47) y Wilcoxon sobre el AUC mensual (p = 0,13; delta de Cliff 0,21, pequeño; gana en 7 de 9 meses). En siete conjuntos públicos: gana 3, empata 4, pierde 0; Wilcoxon sobre los siete, p = 0,69 ({doc}`05_modelo_original`, {doc}`06_validacion`). |

La superioridad de BOREAL en OTIF es estadísticamente sólida en el AUC del año, que es la medida que decide el
proyecto, y su intervalo no toca el cero aun remuestreando órdenes completas. No lo es cuando la prueba se parte en
semanas o meses: la ganancia de 0,011 es del orden de lo que el AUC varía de un mes a otro, y con nueve meses Wilcoxon
tiene poca potencia. Fuera de OTIF no hay superioridad general, sino un dominio donde sirve. Además, 2026 se miró
cuatro veces a lo largo del proyecto, así que estas cifras se leen como exploratorias. Por eso el veredicto es parcial:
BOREAL es mejor que el ganador de las 112 en esta prueba, pero no hay evidencia de que lo sea en cada tramo del tiempo
ni en cualquier conjunto de datos.

## Limitaciones

- **Una sola empresa y un solo periodo de prueba.** Los datos son de un grupo agroindustrial; la prueba 2026 va de
  enero a septiembre, con septiembre incompleto. Las cifras no se trasladan sin más a otra empresa ni a otro ERP.
- **La prueba 2026 se miró más de una vez.** La primera mirada fue la del modelo de efectos sin ordenar; después se
  eligió el ordenamiento y se volvió a medir. Cada configuración se congeló en la validación 2025 antes de medirla,
  pero la secuencia de miradas hace que las cifras de 2026 del modelo original se lean como exploratorias
  ({doc}`05_modelo_original`).
- **Los hiperparámetros del aprendiz de BOREAL** salen de una búsqueda con Optuna en la validación, hecha durante la
  exploración con LightGBM como aprendiz. Con la implementación propia no se repitió la búsqueda dentro de la validación
  anidada de ocho bloques: cada celda cuesta horas en un servidor de 32 núcleos. Los efectos y el número de árboles sí
  se fijaron en la validación 2025 con el aprendiz propio.
- **Las entidades nuevas siguen sin resolverse.** En las órdenes con un proveedor o material que no estaba en el
  entrenamiento, ninguna de las cuatro vías probadas mejoró a XGBoost.
- **Fuera de OTIF solo se probó la pieza central.** El modelo log-normal y las variables de plazo real necesitan una
  promesa de entrega y la historia de entregas, que los conjuntos públicos no traen.

## Trabajo futuro

- Comparar BOREAL con GPBoost en los mismos datos, la alternativa publicada más cercana ({doc}`04_novedad`).
- Una tercera ronda de conjuntos públicos con el criterio fijado de antemano (entidades con historia propia en el
  tiempo), para confirmar o descartar la hipótesis de {doc}`06_validacion`.
- Atacar el arranque en frío con atributos del proveedor y del material que existan antes de su primera orden.
- Medir el valor operativo: cuántos retrasos se evitan siguiendo las alertas con el presupuesto real de seguimiento.

## Declaración de autoría y de uso de herramientas de inteligencia artificial

```{include} ../declaracion_autoria_ia.md
:start-line: 6
```
