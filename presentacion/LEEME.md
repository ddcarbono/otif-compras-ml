# Láminas de la sustentación (Entregable 2, 10 minutos)

Nueve láminas HTML de 1920 × 1080, impresas a PDF con Edge. Las figuras son las de `src/plots.py`; las cifras salen de
`resultados/eda/*.csv` y, las del modelo, de `../../otif/modelar_otif.txt` (marcadas «cifra provisional»).

## Regenerar

```
python presentacion/armar.py        # escribe laminas/*.html, mazo.html y notas.md
python presentacion/generar_pdf.py  # imprime mazo.html a sustentacion_e2.pdf (Edge sin ventana; Plotly viene de cdn.plot.ly)
```

Python: `C:\Users\Daniel Carbono\.conda\envs\ml_venv\python.exe`. Si falta una tabla de `resultados/`, la lámina sale con
un recuadro gris «pendiente» y el script no falla. `notas.md` trae lo que se dice en cada lámina y su tiempo.

## Pendiente

- Láminas 7 y 8: curvas de optimización, tabla de líderes, diagrama de diferencia crítica, errores por línea,
  sobreajuste y SHAP, que dependen de los 112 modelos. Las cifras de LightGBM, la logística y la regla del proveedor son
  provisionales.
- Lámina 5: el nombre del mejor benchmark (hoy LightGBM) se confirma con los 112.
- Lámina 1: URL del tablero.
- Cifras fijas en el texto (490.310 posiciones antes del filtro, 136 predictoras, 7 modelos × 4 × 4) vienen de
  `otif/dataset_otif.md` y del plan, no de `resultados/`.
