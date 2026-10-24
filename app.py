"""Tablero del proyecto: predicción del incumplimiento de órdenes de compra.

Multipágina (``use_pages=True``): Contexto, EDA y Modelos, una página por archivo en ``pages/``. Lee solo ``data/``
(lo escribe ``src/preparar.py``); no entrena nada. Local: ``python app.py`` (puerto 8050 o la variable PORT).
Producción: ``gunicorn app:server``.
"""
import os

import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY],
                title="Incumplimiento de órdenes de compra", suppress_callback_exceptions=True)
server = app.server

navbar = dbc.NavbarSimple(
    children=[dbc.NavItem(dbc.NavLink(p["name"], href=p["relative_path"], active="exact"))
              for p in dash.page_registry.values()],
    # portada de la app Compras de Qlik Sense (ejemplos/2026-10-23_portada_compras en el repo de estudio)
    brand=html.Span([html.Img(src=dash.get_asset_url("carrito_compras.png"), className="marca-carrito", alt=""),
                     "Incumplimiento de órdenes de compra"]),
    color="primary", dark=True, fixed="top", className="navbar-otif")

app.layout = html.Div([navbar, dbc.Container(dash.page_container, fluid="lg", className="pt-5 mt-4 pb-5")])

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 8050)))
