"""Sección 3: modelos. Con el Entregable 2 lleva la tabla de líderes de las 112 combinaciones (validación anidada en 8
bloques semestrales, con Friedman, Nemenyi y Wilcoxon) y el modelo base (regresión logística frente a dos líneas base)
medido en la prueba 2026; con el 3 se suman la predicción en vivo, SHAP y el modelo original.
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
LID = modelo("lideres")
FR = modelo("friedman").iloc[0]
W = modelo("wilcoxon")
G = LID.iloc[0]
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


def tabla_lideres():
    t = LID.assign(modelo=LID.modelo.map(plots.MODELO), balanceo=LID.balanceo.map(plots.BALANCEO),
                   optimizador=LID.optimizador.map(plots.OPTIMIZADOR))
    cols = [("puesto", "puesto", None), ("modelo", "modelo", None), ("balanceo", "balanceo", None),
            ("optimizador", "optimizador", None), ("AUC", "AUC", 4), ("AUC_desv", "± AUC", 4), ("F1", "F1", 4),
            ("F1_desv", "± F1", 4), ("PR_AUC", "PR-AUC", 4), ("AUC_interno", "AUC interno", 4),
            ("horas_cpu", "horas de CPU", 1), ("ms_inferencia_por_1000", "ms por 1.000 posiciones", 1)]
    columnas = [{"name": n, "id": c, "type": "numeric" if d is not None else "text",
                 **({"format": {"specifier": f".{d}f", "locale": {"decimal": ",", "group": "."}}} if d is not None else {})}
                for c, n, d in cols]
    return dash_table.DataTable(data=t[[c for c, _, _ in cols]].to_dict("records"), columns=columnas,
                                filter_action="native", sort_action="native", page_size=15,
                                style_table={"overflowX": "auto"}, style_header={"fontWeight": "bold"},
                                style_cell={"textAlign": "left", "fontSize": 13},
                                style_data_conditional=[{"if": {"filter_query": "{puesto} = 1"}, "fontWeight": "bold"}])


def cifra(valor, texto):
    return dbc.Col(dbc.Card(dbc.CardBody([html.Div(valor, className="cifra"), html.Div(texto, className="cifra-texto")]),
                            className="h-100"), md=3, xs=6, className="mb-3")


layout = html.Div([
    html.H2("Modelos"),
    html.H3("Las 112 combinaciones"),
    html.P("Siete modelos × cuatro balanceos (ninguno, SMOTE, ADASYN, pesos de clase) × cuatro optimizadores (Grid, "
           "Random, Optuna, DEAP) con 20 evaluaciones cada uno, medidos en 8 bloques semestrales de 2022-S1 a 2025-S2: "
           "cada bloque entrena con todo lo anterior, elige hiperparámetros en el último semestre de su ventana y evalúa en "
           "el suyo. Son 896 ajustes; la prueba 2026 no se usa aquí."),
    dbc.Row([cifra(f"{plots.MODELO[G.modelo]}", f"primero: {G.balanceo} · {plots.OPTIMIZADOR[G.optimizador]}"),
             cifra(f"{_n(G.AUC)} ± {_n(G.AUC_desv)}", "ROC-AUC medio en los 8 bloques"),
             cifra(f"{_n(G.F1)} ± {_n(G.F1_desv)}", "F1 medio en los 8 bloques"),
             cifra(f"{int(W.bloques_gana.min())} de 8", "bloques en que XGBoost le gana a cada uno de los otros modelos")]),
    html.P("Escriba en la fila de filtro para buscar (por ejemplo, «SMOTE» en balanceo) y pulse los encabezados para "
           "ordenar.", className="small text-muted"),
    tabla_lideres(),
    dbc.Row([dbc.Col(dcc.Graph(figure=plots.rangos_cd(modelo("rangos"), FR.CD, plots.MODELO), config=CONFIG), lg=6),
             dbc.Col(dcc.Graph(figure=plots.auc_por_bloque(modelo("auc_por_bloque_modelos"), plots.MODELO),
                               config=CONFIG), lg=6)], className="mt-3"),
    html.P(f"Friedman sobre los 8 bloques con la mejor combinación de cada modelo (elegida por AUC interno): "
           f"χ² = {_n(FR.chi2, 1)}, p < 0,001. Nemenyi (diferencia crítica {_n(FR.CD, 2)}) solo separa a XGBoost del "
           f"árbol, KNN y Naive Bayes; la prueba de Wilcoxon pareada del mejor contra cada uno, con Holm, da p = "
           f"{_n(W.p_holm.max(), 3)} en las seis comparaciones: el mínimo posible con 8 bloques.", className="small text-muted"),
    dbc.Row([dbc.Col(dcc.Graph(figure=plots.arrepentimiento(modelo("curvas_optimizador"), "eval", plots.OPTIMIZADOR,
                                                            "Optimizadores: AUC perdido contra evaluaciones", "evaluación"),
                               config=CONFIG), lg=6),
             dbc.Col([dbc.Label("Mapa de un modelo"),
                      dcc.Dropdown(id="m-mapa", options=[{"label": v, "value": k} for k, v in plots.MODELO.items()],
                                   value="xgboost", clearable=False),
                      dcc.Graph(id="m-mapa-fig", config=CONFIG)], lg=6)]),
    html.P("El modelo decide: dentro de cada modelo, sus 16 combinaciones de balanceo y optimizador no se separan más de "
           "0,012 de AUC, y entre modelos la diferencia llega a 0,12. Optuna y el genético se acercan antes al mejor "
           "valor conocido; con 20 evaluaciones los cuatro quedan a menos de 0,004.", className="small text-muted"),
    html.Hr(),
    html.H3("Modelo base: regresión logística"),
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
    dbc.Alert("Con el Entregable 3 se suman aquí la evaluación de XGBoost en la prueba 2026, la explicación con SHAP, "
              "la predicción de una orden nueva y el modelo original.", color="light", className="mt-3"),
])


@callback(Output("m-mapa-fig", "figure"), Input("m-mapa", "value"))
def mapa(m):
    fig = plots.mapa_combinaciones(LID, m, plots.BALANCEO, plots.OPTIMIZADOR)
    fig.update_layout(title=f"{plots.MODELO[m]}: AUC medio por balanceo y optimizador")
    return fig


@callback(Output("m-coef", "figure"), Input("m-n-coef", "value"))
def coef(n):
    return plots.coeficientes(modelo("coeficientes"), NOMBRES, n)
