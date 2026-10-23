# Diapositivas de la sustentación

`sustentacion_otif.pptx` (y su PDF): 14 diapositivas de 25,4 × 14,29 cm con la plantilla Uninorte de las
sustentaciones de GLM (`plantilla_uninorte.py`, la misma de la regresión Beta: barra verde, filete oro, logo, tarjetas
de color, Calibri). Siguen los siete criterios de la rúbrica de sustentación, con la teoría (criterio 3, 25 % de la
nota) en tres diapositivas con fórmulas y el análisis de errores y sobreajuste en una propia.

| # | criterio | contenido | expone |
|---|---|---|---|
| 1 | — | portada con las tres direcciones | A |
| 2 | 1 | problema, x, y, p(x), umbral, objetivos | A |
| 3-5 | 2 | datos y partición; EDA (AUC univariado, faltantes, atípicos); tiempo, fuga y validación adversaria | A |
| 6-8 | 3 | XGBoost: objetivo y Taylor; peso de hoja y ganancia; hiperparámetros, supuestos y complejidad medida | B |
| 9-10 | 4 | diseño anidado de las 112, reproducibilidad; optimizadores | B |
| 11-12 | 5 | tabla de líderes, Friedman, Nemenyi y Wilcoxon, línea base 2026; sobreajuste y calibración | A, B |
| 13 | 6 | conclusiones y limitaciones | A |
| 14 | 7 | preguntas | A y B |

A = Daniel Carbonó, B = Luis Pérez (reparto de `entregables/diapositivas.md`). Tiempos y guion en `notas.md` y en las
notas de cada diapositiva: 10 minutos de exposición más preguntas.

**Pendiente** (caja gris en las diapositivas 11 y 12, espera la evaluación final 2026): XGBoost en la prueba 2026,
tabla de brecha, matriz de confusión por línea, errores con entidades nuevas, un falso negativo típico y SHAP.

## Regenerar

```
"C:\Users\Daniel Carbono\.conda\envs\ml_venv\python.exe" presentacion/figuras.py   # img/*.png: figuras y fórmulas (matplotlib)
python presentacion/construir.py                     # sustentacion_otif.pptx y notas.md (python-pptx, Python base)
python presentacion/revisar_pptx.py presentacion/sustentacion_otif.pptx            # cajas fuera del lienzo o desbordadas
python presentacion/exportar.py presentacion/sustentacion_otif.pptx presentacion/render 1600
python presentacion/mosaico.py presentacion/render presentacion/hoja.png 3         # revisión visual de todas juntas
python presentacion/generar_pdf.py                   # sustentacion_otif.pdf con PowerPoint
```

Ninguna cifra se teclea: `construir.py` las lee de `resultados/` y `data/modelos/` cada vez. `render/` y `hoja.png` son
de revisión y no se versionan.

`guia_sustentacion.md` es la guía de estudio para las preguntas (criterio 7).
