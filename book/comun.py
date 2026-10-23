"""Lo que comparten los cuadernos del libro: la ruta al código del proyecto, la carga de los datos, cómo se muestran las
figuras de Plotly y las tablas, y dónde se guardan las tablas que después leen el tablero y las láminas."""
import sys
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import datos  # noqa: E402

RESULTADOS = RAIZ / "resultados" / "eda"
RESULTADOS.mkdir(parents=True, exist_ok=True)


def mostrar(fig):
    """La figura como HTML; plotly.js lo carga el libro una vez (``html_js_files`` en _config.yml)."""
    display(HTML(fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})))


def tabla(df, dec=2, pct=(), enteros=(), indice=False):
    """Tabla con coma decimal y punto de miles. ``pct``: columnas en proporción que se muestran como porcentaje."""
    fmt = {}
    for c in df.columns:
        if c in pct:
            fmt[c] = lambda v: "" if pd.isna(v) else f"{100 * v:,.1f} %".replace(",", "X").replace(".", ",").replace("X", ".")
        elif c in enteros:
            fmt[c] = lambda v: "" if pd.isna(v) else f"{v:,.0f}".replace(",", ".")
        elif pd.api.types.is_float_dtype(df[c]):
            fmt[c] = (lambda d: lambda v: "" if pd.isna(v) else
                      (f"{v:,.{d}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}").replace(",", "X").replace(".", ",").replace("X", "."))(dec)
        elif pd.api.types.is_integer_dtype(df[c]):
            fmt[c] = lambda v: f"{v:,.0f}".replace(",", ".")
    s = df.style.format(fmt)
    return s if indice else s.hide(axis="index")


def guardar(df, nombre):
    """Guarda una tabla del capítulo en resultados/eda/ (la leen preparar.py y las láminas)."""
    df.to_csv(RESULTADOS / f"{nombre}.csv", index=False)
    return df
