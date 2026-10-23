"""Sección 2: análisis exploratorio con filtros globales (línea, clase de compra y rango de meses) que actualizan todos
los gráficos. Las distribuciones salen de la muestra de 30.000 posiciones del entrenamiento; las series, de los
agregados mensuales de todo el entrenamiento. Cada gráfico lleva una frase de lectura."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from src import plots
from src.data import agregado, es, muestra, nombre, pct, serie_mensual, unidad
from src.tiempo import autocorrelacion, descomponer, estacionariedad

dash.register_page(__name__, path="/eda", name="EDA", order=1, title="EDA · órdenes de compra")

M = muestra()
S = serie_mensual()
TOP = agregado("top_numericas").variable.tolist()
LINEAS = sorted(M.linea.dropna().unique())
CLASES = sorted(M.clase_compras.dropna().unique())
MESES = sorted(S.fecha_pedido.unique())
MARCAS = {i: pd.Timestamp(m).strftime("%Y") for i, m in enumerate(MESES) if pd.Timestamp(m).month == 1}
TASA_GLOBAL = M.y_tarde.mean()
MES = dict(enumerate(["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
                      "noviembre", "diciembre"], 1))
OPC_VAR = [{"label": nombre(v), "value": v} for v in TOP]
OPC_COLOR = [{"label": "Clase (a tiempo / tarde)", "value": "y_tarde"}, {"label": "Línea de negocio", "value": "linea"},
             {"label": "Clase de compra", "value": "clase_compras"}]


def tarjeta(id_fig, id_texto, controles=None):
    return dbc.Card(dbc.CardBody([controles or html.Div(), dcc.Loading(dcc.Graph(id=id_fig, config={"displaylogo": False})),
                                  html.P(id=id_texto, className="lectura")]), className="mb-4")


filtros = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([dbc.Label("Línea de negocio"),
             dcc.Dropdown(id="f-linea", options=LINEAS, multi=True, placeholder="Todas")], md=4),
    dbc.Col([dbc.Label("Clase de compra"),
             dcc.Dropdown(id="f-clase", options=CLASES, multi=True, placeholder="Todas")], md=4),
    dbc.Col([dbc.Label("Mes de creación de la orden"),
             dcc.RangeSlider(id="f-meses", min=0, max=len(MESES) - 1, step=1, value=[0, len(MESES) - 1],
                             marks=MARCAS, allowCross=False)], md=4),
])), className="mb-4 filtros sticky-top")

layout = html.Div([
    html.H2("Análisis exploratorio"),
    html.P("Entrenamiento: órdenes de 2021 a 2024. Los filtros cambian todos los gráficos de esta página.",
           className="text-muted"),
    filtros,
    html.Div(id="eda-alerta"),
    html.H4("La variable objetivo"),
    dbc.Row([dbc.Col(tarjeta("g-balance", "t-balance"), lg=5), dbc.Col(tarjeta("g-serie", "t-serie"), lg=7)]),
    html.H4("Distribuciones y atípicos"),
    dbc.Row([
        dbc.Col(tarjeta("g-hist", "t-hist", dbc.Row([dbc.Col(dcc.Dropdown(id="v-hist", options=OPC_VAR, value=TOP[0],
                                                                         clearable=False), md=8),
                                                    dbc.Col(dbc.Checklist(id="v-log", options=[{"label": " escala log",
                                                                                                "value": "log"}],
                                                                          value=["log"], switch=True), md=4)])), lg=7),
        dbc.Col(tarjeta("g-caja", "t-caja"), lg=5)]),
    html.H4("Relaciones entre variables"),
    tarjeta("g-bi", "t-bi", dbc.Row([
        dbc.Col([dbc.Label("Eje x"), dcc.Dropdown(id="v-x", options=OPC_VAR, value=TOP[0], clearable=False)], md=4),
        dbc.Col([dbc.Label("Eje y"), dcc.Dropdown(id="v-y", options=OPC_VAR, value=TOP[1], clearable=False)], md=4),
        dbc.Col([dbc.Label("Color"), dcc.Dropdown(id="v-color", options=OPC_COLOR, value="y_tarde", clearable=False)],
                md=4)])),
    dbc.Row([dbc.Col(tarjeta("g-corr", "t-corr"), lg=7), dbc.Col(tarjeta("g-vacios", "t-vacios"), lg=5)]),
    html.H4("Tiempo"),
    tarjeta("g-stl", "t-stl"),
    tarjeta("g-acf", "t-acf"),
])


def filtrar(lineas, clases, meses):
    m, s = M, S
    if lineas:
        m, s = m[m.linea.isin(lineas)], s[s.linea.isin(lineas)]
    if clases:
        m, s = m[m.clase_compras.isin(clases)], s[s.clase_compras.isin(clases)]
    desde, hasta = pd.Timestamp(MESES[meses[0]]), pd.Timestamp(MESES[meses[1]]) + pd.offsets.MonthEnd(0)
    m = m[m.fecha_pedido.between(desde, hasta)]
    s = s[s.fecha_pedido.between(desde, hasta)].groupby("fecha_pedido", as_index=False)[["filas", "tardias"]].sum()
    s["tasa"] = s.tardias / s.filas
    return m, s


ENTRADAS = [Input("f-linea", "value"), Input("f-clase", "value"), Input("f-meses", "value")]


@callback(Output("g-balance", "figure"), Output("t-balance", "children"), Output("g-serie", "figure"),
          Output("t-serie", "children"), Output("eda-alerta", "children"), *ENTRADAS)
def objetivo(lineas, clases, meses):
    m, s = filtrar(lineas, clases, meses)
    if len(m) < 50:
        alerta = dbc.Alert("La selección deja menos de 50 posiciones en la muestra: amplíe los filtros.", color="warning")
        return dash.no_update, "", dash.no_update, "", alerta
    t = m.y_tarde.mean()
    txt_b = (f"En la selección llega tarde el {pct(t)} de las posiciones ({es(len(m))} en la muestra), frente al "
             f"{pct(TASA_GLOBAL)} de todo el entrenamiento. Con una clase de un tercio no hay un desbalance fuerte: "
             "se mide con AUC y F1, no con exactitud.")
    pico = s.loc[s.tasa.idxmax()]
    txt_s = (f"La tasa mensual va de {pct(s.tasa.min())} a {pct(s.tasa.max())}; el mes más alto es "
             f"{pico.fecha_pedido:%m/%Y}. Baja de 2021 a 2023 y se estabiliza: el modelo se valida en el tiempo, "
             "nunca mezclando años.")
    return (plots.balance(m.y_tarde, "Posiciones por clase"), txt_b,
            plots.serie_tasa(s, "Tasa de tarde y volumen por mes (todo el entrenamiento)"), txt_s, None)


@callback(Output("g-hist", "figure"), Output("t-hist", "children"), Output("g-caja", "figure"),
          Output("t-caja", "children"), Input("v-hist", "value"), Input("v-log", "value"), *ENTRADAS)
def distribucion(v, log, lineas, clases, meses):
    m, _ = filtrar(lineas, clases, meses)
    if len(m) < 50:
        return dash.no_update, "", dash.no_update, ""
    med = m.groupby("y_tarde")[v].median()
    txt_h = (f"Mediana de {nombre(v).lower()}: {es(med.get(1, np.nan), 2)} en las tardías y {es(med.get(0, np.nan), 2)} "
             f"en las a tiempo. Asimetría {es(m[v].skew(), 1)}: "
             + ("cola larga, por eso la escala logarítmica." if abs(m[v].skew()) > 2 else "sin cola larga."))
    x = m[v].dropna()
    q1, q3 = x.quantile([.25, .75])
    fuera = ((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).mean() if q3 > q1 else 0
    txt_c = (f"El {pct(fuera)} de los valores cae fuera de las cercas de Tukey (1,5 veces el rango intercuartílico). "
             "No se eliminan: una compra grande o un plazo largo son casos reales, no errores.")
    return (plots.histograma_por_clase(m, v, nombre(v), unidad(v), log=bool(log)), txt_h,
            plots.caja_por_clase(m, v, nombre(v)), txt_c)


@callback(Output("g-bi", "figure"), Output("t-bi", "children"), Input("v-x", "value"), Input("v-y", "value"),
          Input("v-color", "value"), *ENTRADAS)
def bivariado(x, y, color, lineas, clases, meses):
    m, _ = filtrar(lineas, clases, meses)
    if len(m) < 50:
        return dash.no_update, ""
    rho = m[[x, y]].corr(method="spearman").iloc[0, 1]
    txt = (f"Correlación de Spearman entre las dos: {es(rho, 2)}. Si los colores se separan en alguna zona del plano, "
           "esa combinación ayuda a distinguir las tardías aunque cada variable sola no lo haga.")
    return plots.dispersion(m.sample(min(len(m), 6000), random_state=2026), x, y, color,
                            {v: nombre(v) for v in TOP}), txt


@callback(Output("g-corr", "figure"), Output("t-corr", "children"), Output("g-vacios", "figure"),
          Output("t-vacios", "children"), *ENTRADAS)
def correlacion_y_vacios(lineas, clases, meses):
    m, _ = filtrar(lineas, clases, meses)
    if len(m) < 50:
        return dash.no_update, "", dash.no_update, ""
    c = m[TOP].corr(method="spearman")
    par = c.where(~np.eye(len(c), dtype=bool)).abs().stack().idxmax()
    txt_c = (f"El par más correlacionado es {nombre(par[0]).lower()} con {nombre(par[1]).lower()} "
             f"(ρ = {es(c.loc[par], 2)}). Las variables de plazo se mueven juntas: en los modelos lineales hay que "
             "vigilar la colinealidad.")
    vac = (m[TOP].isna().mean().sort_values(ascending=False) * 100).rename("vacios").reset_index()
    vac["n"] = vac["index"].map(nombre)
    txt_v = ("Los vacíos son estructurales: faltan los datos de solicitud cuando la orden no viene de una solicitud, "
             "y el historial cuando el proveedor o el material son nuevos. No se imputan con modelos: una marca "
             "conserva la información del vacío.")
    return (plots.mapa_correlacion(c, {v: nombre(v) for v in TOP}, "Correlación de Spearman (12 principales)"), txt_c,
            plots.barras_horizontales(vac, "vacios", "n", "Vacíos en las 12 principales", "% de posiciones", ".1f"),
            txt_v)


@callback(Output("g-stl", "figure"), Output("t-stl", "children"), Output("g-acf", "figure"),
          Output("t-acf", "children"), *ENTRADAS)
def tiempo(lineas, clases, meses):
    _, s = filtrar(lineas, clases, meses)
    if len(s) < 24:
        aviso = "La descomposición necesita al menos 24 meses: amplíe el rango de fechas."
        return plots.go.Figure(), aviso, plots.go.Figure(), aviso
    d = descomponer(s)
    e = estacionariedad(s.tasa).set_index("prueba").p
    a = autocorrelacion(s.tasa)
    est = d.groupby(d.fecha.dt.month).estacionalidad.mean()     # perfil medio de cada mes
    amp = est.max() - est.min()
    txt_s = (f"La tendencia pasa de {pct(d.tendencia.iloc[0])} a {pct(d.tendencia.iloc[-1])}; la estacionalidad "
             f"mueve la tasa en {es(100 * amp, 1)} puntos entre el mes más bajo ({MES[est.idxmin()]}) y el más alto ({MES[est.idxmax()]}), "
             "en promedio.")
    estacionaria = e["ADF"] < 0.05 and e["KPSS"] >= 0.05
    txt_a = (f"ADF: p = {es(e['ADF'], 3)}; KPSS: p = {es(e['KPSS'], 3)}. "
             + ("La serie se comporta como estacionaria." if estacionaria else
                "La serie no es estacionaria (cambia de nivel), así que la partición cronológica es obligatoria y un "
                "modelo entrenado con años viejos puede sobrestimar el retraso.")
             + f" La autocorrelación del primer rezago es {es(a.ACF.iloc[1], 2)}: un mes se parece al anterior.")
    return plots.stl(d), txt_s, plots.acf_pacf(a), txt_a
