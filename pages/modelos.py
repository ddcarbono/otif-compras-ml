"""Sección 3: modelos. Se llena con el Entregable 2 (tabla de líderes de las 112 combinaciones y diagnóstico del
modelo base) y el 3 (predicción en vivo, SHAP, comparación estadística y modelo original)."""
import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path="/modelos", name="Modelos", order=2, title="Modelos · órdenes de compra")

layout = html.Div([
    html.H2("Modelos"),
    dbc.Alert("Esta sección se publica con el Entregable 2: la tabla de las mejores combinaciones de modelo, balanceo "
              "y optimizador, con su métrica del bucle externo, y el diagnóstico del modelo base.", color="info"),
])
