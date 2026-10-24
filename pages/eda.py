"""Sección 2: análisis exploratorio con filtros globales (línea, clase de compra y rango de meses) que actualizan todos
los gráficos. Las distribuciones salen de la muestra de 30.000 posiciones del entrenamiento; las series, de los
agregados mensuales de todo el entrenamiento. Cada gráfico lleva una frase de lectura. Cierra con la exploración
multivariada: PCA sobre las 12 numéricas principales (``src/multivariado.py``)."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from src import plots
from src.data import agregado, es, muestra, nombre, pct, serie_mensual, unidad
from src import multivariado

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


ASO = agregado("asociacion")
NUEVAS = agregado("entidades_nuevas")
ADV = agregado("similitud_auc")


def fija(fig, texto):
    """Tarjeta de un gráfico que no depende de los filtros (se calcula con todo el entrenamiento)."""
    return dbc.Card(dbc.CardBody([dcc.Graph(figure=fig, config={"displaylogo": False}), html.P(texto, className="lectura")]),
                    className="mb-4")


def asociacion():
    auc = ASO.sort_values("AUC", ascending=False).head(15)
    cram = ASO.dropna(subset=["V de Cramér"]).sort_values("V de Cramér", ascending=False).head(12)
    t_auc = (f"La que más separa sola: {auc.nombre.iloc[0].lower()} ({es(auc.AUC.iloc[0], 3)}). Ninguna pasa de 0,75, "
             "el corte de sospecha de fuga.")
    t_cram = (f"V de Cramér = √(χ² / n), de 0 a 1. La más alta: {cram.nombre.iloc[0].lower()} "
              f"({es(cram['V de Cramér'].iloc[0], 2)}). Débiles una a una.")
    return dbc.Row([
        dbc.Col(fija(plots.barras_horizontales(auc, "AUC", "nombre", "AUC univariado (las 15 que más separan)", "AUC",
                                               ".3f", referencia=0.75), t_auc), lg=6),
        dbc.Col(fija(plots.barras_horizontales(cram, "V de Cramér", "nombre", "V de Cramér de las categóricas (12)",
                                               "V de Cramér", ".2f", color=plots.VERDE), t_cram), lg=6)])


def particiones():
    nv = NUEVAS.set_index("variable")
    t_nv = (f"En 2026, proveedor nuevo en el {es(nv.loc['proveedor', 'nuevo en prueba (%)'], 1)} % y material nuevo en el "
            f"{es(nv.loc['material', 'nuevo en prueba (%)'], 1)} %: reciben la tasa de su nivel superior.")
    adv = ADV.assign(nombre=ADV.particion.str.replace("cronológica: ", "", regex=False))
    t_adv = (f"Un XGBoost adivina la partición: al azar no puede ({es(ADV.AUC.iloc[2], 3)}), por año sí "
             f"({es(ADV.AUC.iloc[0], 3)}). Los años son poblaciones distintas: partición cronológica.")
    return dbc.Row([
        dbc.Col(fija(plots.entidades_nuevas(NUEVAS), t_nv), lg=6),
        dbc.Col(fija(plots.barras_horizontales(adv, "AUC", "nombre", "Validación adversaria", "AUC", ".3f", referencia=0.5),
                     t_adv), lg=6)])


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
    html.H4("Tasa de tarde por tramos"),
    tarjeta("g-tramo", "t-tramo", dbc.Row(dbc.Col(dcc.Dropdown(id="v-tramo", options=OPC_VAR, value=TOP[0],
                                                               clearable=False), md=6))),
    html.H4("Asociación de cada variable con la clase (todo el entrenamiento)"),
    asociacion(),
    html.H4("Componentes principales (PCA)"),
    dbc.Row([dbc.Col(tarjeta("g-pca-var", "t-pca-var"), lg=5), dbc.Col(tarjeta("g-pca", "t-pca"), lg=7)]),
    html.H4("Lo que cambia entre particiones"),
    particiones(),
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
             + ("cola larga, por eso la escala logarítmica." if abs(m[v].skew()) > 2 else "sin cola larga.")
             + " Con miles de posiciones la prueba de la caja rechaza siempre: se leen la asimetría y la curtosis.")
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


@callback(Output("g-pca-var", "figure"), Output("t-pca-var", "children"), Output("g-pca", "figure"),
          Output("t-pca", "children"), *ENTRADAS)
def componentes(lineas, clases, meses):
    m, _ = filtrar(lineas, clases, meses)
    if len(m) < 50:
        return dash.no_update, "", dash.no_update, ""
    var, z, cargas = multivariado.pca(m, TOP)
    k80 = int(np.searchsorted(np.cumsum(var), 0.8)) + 1
    txt_v = (f"Dos componentes resumen el {pct(var[:2].sum(), 0)}; hacen falta {k80} para el 80 %. La información está "
             "repartida.")
    c1 = cargas.PC1.abs().idxmax()
    txt_p = (f"PC1 la lleva {nombre(c1).lower()}. Las clases se solapan: no las separa una dirección lineal.")
    i = np.random.default_rng(2026).choice(len(m), min(len(m), 6000), replace=False)
    return (plots.pca_varianza(var), txt_v,
            plots.pca_plano(z[i], m.y_tarde.to_numpy()[i], cargas, {v: nombre(v) for v in TOP}, var), txt_p)


@callback(Output("g-tramo", "figure"), Output("t-tramo", "children"), Input("v-tramo", "value"), *ENTRADAS)
def tramos(v, lineas, clases, meses):
    m, _ = filtrar(lineas, clases, meses)
    if len(m) < 50:
        return dash.no_update, ""
    x = m[v]
    q = pd.qcut(x, 10, duplicates="drop")
    t = m.groupby(q, observed=True).y_tarde.agg(tasa="mean", posiciones="size").reset_index()
    t["tramo"] = [f"{es(i.left, 0)} a {es(i.right, 0)}" if abs(i.right - i.left) >= 2 else
                  f"{es(i.left, 2)} a {es(i.right, 2)}" for i in t[v]]
    vac = m[x.isna()].y_tarde
    alto, bajo = t.loc[t.tasa.idxmax()], t.loc[t.tasa.idxmin()]
    txt = (f"De {pct(bajo.tasa, 0)} (tramo {bajo.tramo}) a {pct(alto.tasa, 0)} (tramo {alto.tramo})."
           + (f" Sin dato: {pct(vac.mean(), 0)}." if len(vac) else ""))
    return plots.tasa_por_tramo(t, nombre(v), m.y_tarde.mean()), txt
