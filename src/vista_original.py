"""La predicción en vivo con el modelo original, en la pestaña Predicción: el mismo formulario que ``vista_prediccion``
(campos que más pesan, rango de desarrollo, una posición real que se carga), con posiciones de la prueba 2026 que el
modelo no vio, la probabilidad del plazo log-normal sola y las contribuciones de los árboles y de los efectos por entidad."""
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dcc, html, no_update

from . import plots
from .data import diccionario, ejemplos_original, original_servido, predecir_original, rangos_original

NOMBRE = dict(zip(diccionario().variable, diccionario().nombre))
NOMBRE.update({
    "efecto_proveedor": "Efecto del proveedor", "efecto_material": "Efecto del material",
    "efecto_sp_creador": "Efecto del creador de la solicitud", "efecto_creador_oc": "Efecto del comprador",
    "dist_z": "Holgura log-normal (z)", "dist_mu": "Plazo real esperado (log)", "corte": "Corte c = 1",
    "prov_lt_med": "Plazo real del proveedor (mediana, últimas 20)", "prov_lt_q80": "Plazo real del proveedor (p80)",
    "prov_holgura_med": "Holgura frente al proveedor (mediana)", "prov_holgura_q80": "Holgura frente al proveedor (p80)",
    "provmat_lt_med": "Plazo real proveedor-material (mediana)", "provmat_holgura_med": "Holgura proveedor-material",
    "provga_holgura_med": "Holgura proveedor-grupo de artículos", "mat_lt_med": "Plazo real del material (mediana)",
    "mat_holgura_med": "Holgura frente al material", "prov_tasa_ult10": "Tasa de retraso, últimas 10 del proveedor",
    "prov_tasa_ult30": "Tasa de retraso, últimas 30 del proveedor", "prov_retraso_ult10": "Retraso medio, últimas 10",
    "prov_tasa_90d": "Tasa de retraso del proveedor, últimos 90 días", "prov_n_90d": "Entregas del proveedor, 90 días",
    "provmat_tasa": "Tasa de retraso proveedor-material", "provga_tasa": "Tasa de retraso proveedor-grupo de artículos",
    "provcen_tasa": "Tasa de retraso proveedor-centro", "prov_carga_rel": "Carga actual del proveedor (relativa)"})
QUIEN = {"prov": "proveedor", "provmat": "proveedor-material", "provga": "proveedor-grupo de artículos",
         "mat": "material"}
for _k, _q in QUIEN.items():
    for _s, _t in (("med", "mediana"), ("q80", "p80")):
        NOMBRE.setdefault(f"{_k}_lt_{_s}", f"Plazo real del {_q} ({_t}, últimas 20)")
        NOMBRE.setdefault(f"{_k}_holgura_{_s}", f"Holgura frente al {_q} ({_t})")
NOMBRE.update({f"te_{k}":f"Historial de retraso: {v.lower()}" for k, v in NOMBRE.items() if not k.startswith("te_")})
AYUDA = {"efecto_proveedor": "efecto aleatorio en el logit; 0 = proveedor medio o nuevo",
         "efecto_material": "efecto aleatorio en el logit; 0 = material medio o nuevo",
         "prov_lt_med": "días hábiles que tardaron de verdad sus últimas 20 entregas",
         "plazo_habil_al_liberar": "al cambiarlo se recalculan las holguras y z"}


def _n(v, d=0):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def campo(c):
    r = rangos_original().loc[c]
    d = 3 if c.startswith("efecto") or "tasa" in c else 0
    return dbc.Col([dbc.Label(NOMBRE.get(c, c), className="mb-0 small fw-bold"),
                    dbc.Input(id={"tipo": "o-campo", "c": c}, type="number", step=0.01 if d else 1, debounce=True),
                    html.Small(f"rango de desarrollo: {_n(r.p01, d)} a {_n(r.p99, d)} (mediana {_n(r.mediana, d)})",
                               className="text-muted d-block"),
                    html.Small(AYUDA.get(c, ""), className="text-muted fst-italic")], md=6, lg=4, className="mb-3")


def vista():
    _, ficha = original_servido()
    return html.Div([
        html.P(["Estime el riesgo con el modelo original. El botón carga una posición real de la prueba 2026 (que el "
                "modelo no vio); los campos se pueden cambiar. Modelo servido: ", html.Strong(ficha["modelo"]),
                f", entrenado con {ficha['entrenado_con']}; umbral de alerta {_n(ficha['umbral'], 3)}. Las entidades "
                "entran como números (efectos y tasas), sin códigos."]),
        dbc.Row([dbc.Col(dbc.Button("Cargar una posición real", id="o-ejemplo", color="secondary"), width="auto"),
                 dbc.Col(dbc.Button("Predecir", id="o-predecir", color="primary"), width="auto"),
                 dbc.Col(html.Div(id="o-origen", className="small text-muted pt-2"))], className="mb-3 g-2"),
        dcc.Store(id="o-indice", data=0),
        dbc.Row([campo(c) for c in rangos_original().index]),
        dcc.Loading(html.Div(id="o-resultado"), type="default"),
        html.P("Cada barra sube (oro) o baja (verde) el logit de llegar tarde. Las variables son valores SHAP exactos del "
               "boosting propio en el corte c = 1; los efectos por entidad se suman tal cual. Todo, con el valor base, da el "
               "logit que Platt convierte en riesgo.",
               className="small text-muted mt-2"),
    ])


@callback(Output("o-indice", "data"), Input("o-ejemplo", "n_clicks"), prevent_initial_call=True)
def _sortear(n):
    return int(np.random.default_rng(n).integers(len(ejemplos_original())))


@callback(Output({"tipo": "o-campo", "c": ALL}, "value"), Output("o-origen", "children"), Input("o-indice", "data"),
          State({"tipo": "o-campo", "c": ALL}, "id"))
def _cargar(i, ids):
    e = ejemplos_original().iloc[i]
    vals = [None if pd.isna(e[d["c"]]) else round(float(e[d["c"]]), 4) for d in ids]
    real = (f"llegó tarde ({_n(e.dias_retraso)} días hábiles de retraso)" if e.y_tarde == 1 else "llegó a tiempo")
    return vals, f"Posición real n.º {i + 1} de {len(ejemplos_original())} (prueba 2026): {real}."


@callback(Output("o-resultado", "children"), Input("o-predecir", "n_clicks"),
          Input({"tipo": "o-campo", "c": ALL}, "value"), State({"tipo": "o-campo", "c": ALL}, "id"),
          State("o-indice", "data"))
def _predecir(_, vals, ids, i):
    fila = ejemplos_original().iloc[[i]].copy()
    fuera = []
    for v, d in zip(vals, ids):
        c, r = d["c"], rangos_original().loc[d["c"]]
        if v is None:
            if not pd.isna(fila[c].iloc[0]):
                return no_update
            continue
        fila[c] = float(v)
        if not (r.p01 <= float(v) <= r.p99):
            fuera.append(NOMBRE.get(c, c))
    p, u, p_ln, contrib = predecir_original(fila)
    alerta = p >= u
    top = contrib.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.contribucion, y=[NOMBRE.get(v, v) for v in top.variable], orientation="h",
                           marker_color=[plots.ORO if x > 0 else plots.VERDE for x in top.contribucion],
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=plots.GRIS))
    fig.update_layout(title="Las 10 piezas que más mueven esta predicción", xaxis_title="contribución al logit",
                      height=380, margin=dict(l=300, t=50))
    tarjeta = dbc.Alert([html.H4(f"{_n(100 * p, 1)} % de riesgo de llegar tarde", className="mb-1"),
                         html.Div("Alerta: supera el umbral; conviene hacerle seguimiento." if alerta else
                                  "Sin alerta: por debajo del umbral."),
                         html.Small(f"Solo con el plazo log-normal: {_n(100 * p_ln, 1)} %.",
                                    className="d-block mt-1")],
                        color="warning" if alerta else "success", className="mb-2")
    aviso = (dbc.Alert("Fuera del rango de desarrollo: " + ", ".join(fuera) + ". El modelo extrapola.",
                       color="light", className="small py-2") if fuera else None)
    return [tarjeta, aviso, dcc.Graph(figure=fig, config={"displaylogo": False})]
