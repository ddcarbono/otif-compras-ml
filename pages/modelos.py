"""Sección 3: modelos. Con el Entregable 2 lleva el modelo base (regresión logística frente a dos líneas base) medido en
la prueba 2026; con el 3 se suman la tabla de las combinaciones, la predicción en vivo, SHAP y el modelo original.
Todo se lee de ``data/modelos`` (lo escribe ``src.preparar`` a partir de ``src.modelo_base``): aquí no se entrena."""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dash_table, dcc, html

from src import plots
from src.data import diccionario, modelo

dash.register_page(__name__, path="/modelos", name="Modelos", order=2, title="Modelos · órdenes de compra")

MET = modelo("metricas")
RES = modelo("resumen")
PRU = MET[MET["partición"] == "prueba 2026"].set_index("modelo")
LOG = PRU.loc["Regresión logística"]
NOMBRES = dict(zip(diccionario().variable, diccionario().nombre))
CONFIG = {"displaylogo": False}


def _n(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def _ic(fila, k):
    return f"{_n(fila[k])} [{_n(fila[k + ' inf'])}; {_n(fila[k + ' sup'])}]"


def tabla_metricas():
    filas = []
    for nombre, f in PRU.iterrows():
        filas.append({"modelo": nombre, "ROC-AUC [IC 95 %]": _ic(f, "ROC-AUC"), "PR-AUC [IC 95 %]": _ic(f, "PR-AUC"),
                      "F1 [IC 95 %]": _ic(f, "F1"), "precisión": _n(f["precisión"]), "sensibilidad": _n(f.sensibilidad),
                      "Brier": _n(f.Brier), "ECE": _n(f.ECE), "umbral": _n(f.umbral)})
    cols = list(filas[0])
    return dash_table.DataTable(data=filas, columns=[{"name": c, "id": c} for c in cols],
                                style_table={"overflowX": "auto"}, style_header={"fontWeight": "bold"},
                                style_cell={"textAlign": "left", "fontSize": 13},
                                style_data_conditional=[{"if": {"filter_query": '{modelo} = "Regresión logística"'},
                                                         "fontWeight": "bold"}])


def cifra(valor, texto):
    return dbc.Col(dbc.Card(dbc.CardBody([html.Div(valor, className="cifra"), html.Div(texto, className="cifra-texto")]),
                            className="h-100"), md=3, xs=6, className="mb-3")


layout = html.Div([
    html.H2("Modelos"),
    html.P("El modelo base es una regresión logística con penalización L2, con el mismo preprocesamiento que usarán los "
           "demás modelos: logaritmo con signo, imputación y estandarización en las numéricas, codificación one-hot en "
           "las categóricas y codificación jerárquica por objetivo en las entidades, todo dentro del pipeline. Se compara "
           "con dos líneas base: un clasificador que predice la tasa del entrenamiento y la regla de negocio de mirar la "
           "tasa previa de retraso del proveedor."),
    dbc.Row([cifra(_n(LOG["ROC-AUC"]), "ROC-AUC de la logística en la prueba 2026"),
             cifra(_n(PRU.loc["Regla del proveedor", "ROC-AUC"]), "ROC-AUC de la regla del proveedor"),
             cifra(_n(LOG.F1), f"F1 al umbral {_n(LOG.umbral)}, fijado en la validación 2025"),
             cifra(_n(LOG.ECE), "error de calibración esperado (ECE)")]),
    html.H4("Métricas en la prueba 2026"),
    html.P(f"Intervalos al 95 % por bootstrap de órdenes ({RES['bootstrap']} remuestras): las posiciones de una misma "
           "orden no son independientes, así que se remuestrean órdenes completas. El umbral de cada modelo es el que "
           "maximiza el F1 en la validación 2025; la prueba se usó una sola vez.", className="small text-muted"),
    tabla_metricas(),
    html.H4("Discriminación", className="mt-4"),
    dbc.Row([dbc.Col(dcc.Graph(figure=plots.curva_roc_pr(modelo("curvas"), "ROC", PRU.reset_index()), config=CONFIG),
                     lg=6),
             dbc.Col(dcc.Graph(figure=plots.curva_roc_pr(modelo("curvas"), "PR", PRU.reset_index()), config=CONFIG),
                     lg=6)]),
    html.H4("Umbral y calibración"),
    dbc.Row([dbc.Col(dcc.Graph(figure=plots.matriz_confusion(modelo("confusion"), LOG.umbral), config=CONFIG), lg=6),
             dbc.Col(dcc.Graph(figure=plots.calibracion(modelo("calibracion")), config=CONFIG), lg=6)]),
    html.H4("Qué pesa en el modelo"),
    html.P("Coeficientes del modelo final (entrenado con 2021-2025). Las numéricas están estandarizadas, así que el "
           "coeficiente es el cambio en log-odds por una desviación estándar; en las categóricas, frente a la media de "
           "las demás. El bloque de plazo es colineal (VIF de más de 300): sus coeficientes por separado no se leen como "
           "efectos, solo su suma.", className="small text-muted"),
    html.Div([dbc.Label("Coeficientes a mostrar"),
              dcc.Slider(id="m-n-coef", min=10, max=40, step=10, value=20, marks={k: str(k) for k in (10, 20, 30, 40)})],
             style={"maxWidth": 360}),
    dcc.Graph(id="m-coef", config=CONFIG),
    html.H4("Hiperparámetro y tamaño de muestra"),
    dbc.Row([dbc.Col(dcc.Graph(figure=plots.validacion_c(modelo("cv_C")), config=CONFIG), lg=6),
             dbc.Col(dcc.Graph(figure=plots.curva_aprendizaje(modelo("curva_aprendizaje")), config=CONFIG), lg=6)]),
    html.P(f"C = {RES['C']:g}, elegido con validación cruzada temporal dentro de 2021-2024 (entrena con los años "
           "anteriores y valida con el siguiente). La curva de aprendizaje mide el AUC en la validación 2025 con 10 %, "
           "25 %, 50 % y 100 % de las órdenes de entrenamiento.", className="small text-muted"),
    dbc.Alert("Con el Entregable 3 se suman aquí las 112 combinaciones de modelo, balanceo y optimizador, la "
              "comparación estadística entre ellas, la explicación con SHAP y la predicción de una orden nueva.",
              color="light", className="mt-3"),
])


@callback(Output("m-coef", "figure"), Input("m-n-coef", "value"))
def coef(n):
    return plots.coeficientes(modelo("coeficientes"), NOMBRES, n)
