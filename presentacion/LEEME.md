# Diapositivas de la sustentación

`sustentacion_otif.pptx` (y su PDF): 23 diapositivas de 25,4 × 14,29 cm con la plantilla Uninorte de las
sustentaciones de GLM (`plantilla_uninorte.py`, la misma de la regresión Beta). Siguen la rúbrica de 20 minutos, con la
tabla de `entregables/diapositivas.md` y el mejor benchmark (XGBoost) en la teoría: 18 minutos de exposición y 2 de
preguntas.

| # | criterio | min | contenido | expone |
|---|---|---|---|---|
| 1-3 | 1 | 2:00 | portada; el problema con su magnitud y la literatura; formulación (Bernoulli, log-pérdida, umbral) | A |
| 4-8 | 2 | 3:00 | tablero en vivo (Contexto y EDA); datos y partición; descriptiva y calidad; relaciones y fuga; tiempo y validación adversaria | A |
| 9-12 | 3 | 5:00 | XGBoost: modelo y pérdida; Taylor con la derivación de g y h; peso de hoja y ganancia; regularización, supuestos y complejidad medida | B |
| 13-15 | 4 | 3:00 | diseño anidado; implementación y espacio de búsqueda; tablero en vivo (las 112 y los optimizadores) | B, A |
| 16-18 | 5 | 3:00 | métricas y línea base; tablero en vivo (predicción) con Friedman, Nemenyi y Wilcoxon; errores y sobreajuste | B, A |
| 19-21 | 6 | 3:00 | modelo original (fórmula y la 2 × 2 de 2026); conjuntos públicos; conclusiones con el veredicto de cada objetivo; limitaciones y mejoras | A, B |
| 22 | 7 | 2:00 | preguntas | A y B |

A = Daniel Carbonó, B = Luis Pérez. Guion y tiempos en `notas.md` y en las notas de cada diapositiva. Las pantallas del
tablero llevan su captura (`capturar_tablero.py`) por si Render no carga.

**Pendiente** (caja gris en las diapositivas 16 y 18, espera la evaluación final 2026): XGBoost en la prueba 2026,
tabla de brecha, matriz de confusión por línea, errores con entidades nuevas, un falso negativo típico y SHAP.

## Regenerar

```
"C:\Users\Daniel Carbono\.conda\envs\ml_venv\python.exe" presentacion/figuras.py   # img/*.png: figuras y fórmulas (matplotlib)
python presentacion/capturar_tablero.py              # img/tablero_*.png desde Render
python presentacion/construir.py                     # sustentacion_otif.pptx y notas.md (python-pptx, Python base)
python presentacion/revisar_pptx.py presentacion/sustentacion_otif.pptx            # cajas fuera del lienzo o desbordadas
python presentacion/exportar.py presentacion/sustentacion_otif.pptx presentacion/render 1600
python presentacion/mosaico.py presentacion/render presentacion/hoja.png 3         # revisión visual de todas juntas
python presentacion/generar_pdf.py                   # sustentacion_otif.pdf con PowerPoint
```

Ninguna cifra se teclea: `construir.py` las lee de `resultados/` y `data/modelos/` cada vez. `render/` y `hoja.png` son
de revisión y no se versionan.

`guia_sustentacion.md` es la guía de estudio para las preguntas (criterio 7).
