"""Encabezado de cada página: el título (y su entradilla) a la izquierda y la portada de la app Compras de Qlik Sense a
la derecha (``assets/portada_compras.jpg``; se bajó con ``ejemplos/2026-10-23_portada_compras`` del repo de estudio).
En celular la imagen va arriba, centrada."""
import dash
import dash_bootstrap_components as dbc
from dash import html


def encabezado(contenido, clase_imagen="portada"):
    """``contenido``: el título y lo que va bajo él. ``clase_imagen``: ``portada`` (Contexto) o ``portada portada-pagina``
    (las demás, más pequeña)."""
    return dbc.Row([
        dbc.Col(contenido, md=9, className="order-2 order-md-1"),
        dbc.Col(html.Img(src=dash.get_asset_url("portada_compras.jpg"), className=clase_imagen, alt=""),
                md=3, className="order-1 order-md-2 text-center text-md-end")], className="align-items-center")
