"""La pestaña de predicción en vivo de la página Modelos: un formulario con los campos que más pesan, el rango de
desarrollo bajo cada uno, un botón que carga una posición real de la validación 2025 y el resultado con sus
contribuciones. Las demás predictoras se toman de la posición cargada (o de la primera, al abrir)."""
import json

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dcc, html, no_update

from . import plots
from .data import diccionario, ejemplos, predecir, rangos, servido

NOMBRE = dict(zip(diccionario().variable, diccionario().nombre))
NOMBRE.update({"te_proveedor": "Historial de retraso del proveedor", "te_material": "Historial de retraso del material"})
AYUDA = {"te_proveedor": "tasa de retraso del proveedor en el entrenamiento, encogida hacia la de su región",
         "te_material": "tasa de retraso del material en el entrenamiento, encogida hacia la de su grupo de artículos",
         "plazo_habil_al_liberar": "días hábiles entre la liberación y la fecha prometida (negativo: ya vencida)",
         "con_liberacion": "1 si la orden pasó por estrategia de liberación"}


def _n(v, d=0):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def campo(c):
    r = rangos().loc[c]
    if r.tipo == "categórica":
        cats = json.loads(r.categorias)
        top = max(cats, key=cats.get)
        ctrl = dcc.Dropdown(id={"tipo": "campo", "c": c}, options=[{"label": k, "value": k} for k in cats],
                            clearable=False)
        nota = f"más frecuente: {top} ({_n(100 * cats[top], 1)} %)"
    else:
        d = 3 if r.tipo == "tasa" else 0
        paso = 0.01 if r.tipo == "tasa" else 1
        ctrl = dbc.Input(id={"tipo": "campo", "c": c}, type="number", step=paso, debounce=True)
        nota = f"rango de desarrollo: {_n(r.p01, d)} a {_n(r.p99, d)} (mediana {_n(r.mediana, d)})"
    return dbc.Col([dbc.Label(NOMBRE.get(c, c), className="mb-0 small fw-bold"), ctrl,
                    html.Small(nota, className="text-muted d-block"),
                    html.Small(AYUDA.get(c, ""), className="text-muted fst-italic")], md=6, lg=4, className="mb-3")


CAMPOS = list(rangos().index)


def vista():
    _, ficha = servido()
    return html.Div([
        html.P(["Estime el riesgo de una posición el día en que la orden se libera. El botón carga una posición real de "
                "la validación 2025 con todas sus variables; los campos de abajo, que son los que más pesan, se pueden "
                "cambiar para ver cómo se mueve el riesgo. Modelo servido: ", html.Strong(ficha["modelo"]),
                f", entrenado con {ficha['entrenado_con']}; umbral de alerta {_n(ficha['umbral'], 3)}."]),
        dbc.Row([dbc.Col(dbc.Button("Cargar una posición real", id="p-ejemplo", color="secondary"), width="auto"),
                 dbc.Col(dbc.Button("Predecir", id="p-predecir", color="primary"), width="auto"),
                 dbc.Col(html.Div(id="p-origen", className="small text-muted pt-2"))], className="mb-3 g-2"),
        dcc.Store(id="p-indice", data=0),
        dbc.Row([campo(c) for c in CAMPOS]),
        dcc.Loading(html.Div(id="p-resultado"), type="default",
                    custom_spinner=html.Div([dbc.Spinner(size="sm"), html.Span(" Calculando…", className="ms-2")])),
        html.P("Cómo leer las contribuciones: cada barra es cuánto sube (oro) o baja (verde) esa variable el logaritmo de "
               "los odds de llegar tarde, frente a una posición promedio. Suman, con el intercepto, el riesgo estimado. "
               "Si un valor queda fuera del rango de desarrollo, el modelo está extrapolando y la cifra es menos confiable.",
               className="small text-muted mt-2"),
    ])


@callback(Output("p-indice", "data"), Input("p-ejemplo", "n_clicks"), prevent_initial_call=True)
def _sortear(n):
    return int(np.random.default_rng(n).integers(len(ejemplos())))


@callback(Output({"tipo": "campo", "c": ALL}, "value"), Output("p-origen", "children"), Input("p-indice", "data"),
          State({"tipo": "campo", "c": ALL}, "id"))
def _cargar(i, ids):
    e = ejemplos().iloc[i]
    vals = []
    for d in ids:
        v = e[d["c"]]
        vals.append(None if pd.isna(v) else (v if isinstance(v, str) else round(float(v), 4)))
    real = "llegó tarde" if e.y_tarde == 1 else "llegó a tiempo"
    return vals, f"Posición real n.º {i + 1} de {len(ejemplos())} (validación 2025): {real}."


@callback(Output("p-resultado", "children"), Input("p-predecir", "n_clicks"), Input({"tipo": "campo", "c": ALL}, "value"),
          State({"tipo": "campo", "c": ALL}, "id"), State("p-indice", "data"))
def _predecir(_, vals, ids, i):
    if any(v is None for v, d in zip(vals, ids) if rangos().loc[d["c"]].tipo != "categórica"):
        return no_update
    fila = ejemplos().iloc[[i]].copy()
    fuera = []
    for v, d in zip(vals, ids):
        c, r = d["c"], rangos().loc[d["c"]]
        fila[c] = v
        if r.tipo != "categórica" and not (r.p01 <= float(v) <= r.p99):
            fuera.append(NOMBRE.get(c, c))
        if r.tipo == "categórica" and v not in json.loads(r.categorias):
            fuera.append(NOMBRE.get(c, c))
    p, u, contrib = predecir(fila)
    alerta = p >= u
    top = contrib.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.contribucion, y=[NOMBRE.get(v, v) for v in top.variable], orientation="h",
                           marker_color=[plots.ORO if x > 0 else plots.VERDE for x in top.contribucion],
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=plots.GRIS))
    fig.update_layout(title="Las 10 variables que más mueven esta predicción", xaxis_title="contribución al logit",
                      height=380, margin=dict(l=260, t=50))
    tarjeta = dbc.Alert([html.H4(f"{_n(100 * p, 1)} % de riesgo de llegar tarde", className="mb-1"),
                         html.Div("Alerta: supera el umbral; conviene hacerle seguimiento." if alerta else
                                  "Sin alerta: por debajo del umbral.")],
                        color="warning" if alerta else "success", className="mb-2")
    aviso = (dbc.Alert("Fuera del rango de desarrollo: " + ", ".join(fuera) + ". El modelo extrapola.",
                       color="light", className="small py-2") if fuera else None)
    return [tarjeta, aviso, dcc.Graph(figure=fig, config={"displaylogo": False})]
