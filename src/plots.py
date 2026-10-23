"""Una función por figura, en Plotly. Las usan el libro y el tablero: reciben las tablas de ``eda.py`` (o la muestra
filtrada del tablero) y devuelven un ``go.Figure`` con la plantilla del proyecto.

Paleta de cinco colores: verde (a tiempo), oro (tarde), azul (series y barras neutras), gris (referencias) y marrón
(atípicos y alertas)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

VERDE, ORO, AZUL, GRIS, MARRON = "#405731", "#B08A2D", "#3F6F8F", "#6B6B6B", "#7A5526"
PALETA = [VERDE, ORO, AZUL, GRIS, MARRON]
FUENTE = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

pio.templates["otif"] = go.layout.Template(layout=dict(
    font=dict(family=FUENTE, size=13, color="#333333"), colorway=PALETA, paper_bgcolor="white", plot_bgcolor="white",
    title=dict(x=0, xanchor="left", font=dict(size=15)), margin=dict(l=60, r=20, t=60, b=50),
    xaxis=dict(showgrid=False, linecolor="#BBBBBB", ticks="outside"),
    yaxis=dict(gridcolor="#EEEEEE", linecolor="#BBBBBB", zeroline=False),
    legend=dict(orientation="h", y=-0.18, x=0), hoverlabel=dict(font_family=FUENTE),
    separators=",."))
pio.templates.default = "otif"
CLASE = {0: "a tiempo", 1: "tarde"}
COLOR_CLASE = {"a tiempo": VERDE, "tarde": ORO}


def _pct(v):
    return f"{100 * v:.1f} %".replace(".", ",")


def objetivo_por(t, g, titulo, eje="", orden=None):
    """Barras de la tasa de tarde por grupo, con el número de filas en el hover y la tasa global como referencia."""
    t = t.copy()
    if orden is not None:
        t = t.set_index(g).loc[orden].reset_index()
    glob = (t.tasa * t.filas).sum() / t.filas.sum()
    fig = go.Figure(go.Bar(x=t[g].astype(str), y=t.tasa, marker_color=ORO, text=[_pct(v) for v in t.tasa],
                           textposition="outside", customdata=t.filas,
                           hovertemplate="%{x}<br>tasa de tarde %{y:.1%}<br>%{customdata:,} posiciones<extra></extra>"))
    fig.add_hline(y=glob, line=dict(color=GRIS, dash="dot"), annotation_text=f"global {_pct(glob)}",
                  annotation_position="top left")
    fig.update_layout(title=titulo, xaxis_title=eje, yaxis_title="tasa de tarde", yaxis_tickformat=".0%",
                      yaxis_range=[0, max(t.tasa) * 1.2])
    return fig


def balance(y, titulo="Distribución del objetivo"):
    c = pd.Series(y).map(CLASE).value_counts().reindex(["a tiempo", "tarde"])
    fig = go.Figure(go.Bar(x=c.index, y=c.values, marker_color=[VERDE, ORO],
                           text=[f"{v:,}<br>{_pct(v / c.sum())}" for v in c.values], textposition="outside",
                           hovertemplate="%{x}: %{y:,} posiciones<extra></extra>"))
    fig.update_layout(title=titulo, yaxis_title="posiciones", yaxis_range=[0, c.max() * 1.25])
    return fig


def histograma_por_clase(df, v, nombre, unidad="", log=False, bins=40):
    """Distribución de una numérica separada por clase (densidad), con el eje en log si la cola es larga."""
    x = df[v].astype(float)
    if log:
        x = np.sign(x) * np.log10(1 + x.abs())
    # se agrupa aquí y se dibujan barras: así la figura pesa lo mismo con 300 que con 300.000 filas
    lim = np.nanpercentile(x, [0.5, 99.5])
    if lim[0] == lim[1]:
        lim = [np.nanmin(x), np.nanmax(x) + 1]
    cortes = np.linspace(lim[0], lim[1], bins + 1)
    centro, ancho = (cortes[:-1] + cortes[1:]) / 2, np.diff(cortes)
    fig = go.Figure()
    for k, et in CLASE.items():
        xs = x[(df.y_tarde == k).to_numpy()].dropna().clip(lim[0], lim[1])
        dens = np.histogram(xs, cortes)[0] / max(len(xs), 1) / ancho
        fig.add_trace(go.Bar(x=centro, y=dens, width=ancho, name=et, marker_color=COLOR_CLASE[et], opacity=0.6,
                             hovertemplate="%{x:.2f}: densidad %{y:.3f}<extra>" + et + "</extra>"))
    eje = f"{nombre} ({unidad})" if unidad and unidad != "—" else nombre
    fig.update_layout(barmode="overlay", title=nombre, xaxis_title=("signo · log10(1 + |x|) de " if log else "") + eje,
                      yaxis_title="densidad")
    return fig


def caja_por_clase(df, v, nombre):
    fig = go.Figure()
    for k, et in CLASE.items():
        fig.add_trace(go.Box(y=df.loc[df.y_tarde == k, v], name=et, marker_color=COLOR_CLASE[et], boxpoints="outliers",
                             marker=dict(size=3, opacity=0.4)))
    fig.update_layout(title=f"{nombre}: atípicos por clase", yaxis_title=nombre, showlegend=False)
    return fig


def barras_horizontales(t, x, y, titulo, eje_x, formato=".2f", color=AZUL, referencia=None):
    t = t.iloc[::-1]
    fig = go.Figure(go.Bar(x=t[x], y=t[y], orientation="h", marker_color=color,
                           hovertemplate="%{y}: %{x:" + formato + "}<extra></extra>"))
    if referencia is not None:
        fig.add_vline(x=referencia, line=dict(color=GRIS, dash="dot"))
    fig.update_layout(title=titulo, xaxis_title=eje_x, height=max(320, 26 * len(t) + 120), margin=dict(l=220))
    return fig


def mapa_correlacion(c, nombres, titulo="Correlación de Spearman"):
    et = [nombres.get(v, v) for v in c.columns]
    fig = go.Figure(go.Heatmap(z=c.values, x=et, y=et, zmin=-1, zmax=1,
                               colorscale=[[0, AZUL], [0.5, "#FFFFFF"], [1, ORO]], colorbar=dict(title="ρ"),
                               hovertemplate="%{y} · %{x}<br>ρ = %{z:.2f}<extra></extra>"))
    fig.update_layout(title=titulo, height=max(450, 18 * len(et) + 200), xaxis_tickangle=-45,
                      yaxis_autorange="reversed", margin=dict(l=200, b=180))
    return fig


def matriz_vacios(df, columnas, nombres, filas=1500, semilla=2026):
    """Matriz de nulos: una muestra de filas ordenadas por fecha y las columnas con vacíos agrupadas por bloque."""
    m = df.sample(min(filas, len(df)), random_state=semilla).sort_values("fecha_pedido")[columnas].isna().astype(int)
    fig = go.Figure(go.Heatmap(z=m.T.values, y=[nombres.get(c, c) for c in columnas], x=np.arange(len(m)),
                               colorscale=[[0, "#F2F2F2"], [1, MARRON]], showscale=False,
                               hovertemplate="%{y}<extra></extra>"))
    fig.update_layout(title="Patrón de vacíos (marrón = vacío; filas ordenadas por fecha)", xaxis_title="posición (muestra)",
                      height=max(400, 14 * len(columnas) + 150), margin=dict(l=230))
    return fig


def serie_tasa(t, titulo="Tasa de tarde y volumen por mes", color_por=None):
    """Volumen en barras (eje derecho) y tasa en línea (eje izquierdo)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if color_por:
        for i, (g, d) in enumerate(t.groupby(color_por)):
            fig.add_trace(go.Scatter(x=d.fecha_pedido, y=d.tasa, name=str(g), mode="lines",
                                     line=dict(color=PALETA[i % 5])), secondary_y=False)
    else:
        fig.add_trace(go.Bar(x=t.fecha_pedido, y=t.filas, name="posiciones", marker_color="#DCE3EA"), secondary_y=True)
        fig.add_trace(go.Scatter(x=t.fecha_pedido, y=t.tasa, name="tasa de tarde", mode="lines+markers",
                                 line=dict(color=ORO), marker=dict(size=4)), secondary_y=False)
    fig.update_yaxes(title_text="tasa de tarde", tickformat=".0%", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="posiciones", secondary_y=True, showgrid=False)
    fig.update_layout(title=titulo, xaxis_title="mes de creación de la orden")
    return fig


def stl(d):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=["observada", "tendencia", "estacionalidad", "residuo"])
    for i, c in enumerate(["observada", "tendencia", "estacionalidad", "residuo"], 1):
        fig.add_trace(go.Scatter(x=d.fecha, y=d[c], mode="lines", line=dict(color=[ORO, AZUL, VERDE, GRIS][i - 1]),
                                 showlegend=False), row=i, col=1)
    fig.update_layout(title="Descomposición STL de la tasa mensual de tarde", height=620)
    return fig


def acf_pacf(a):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
    for j, c in enumerate(["ACF", "PACF"], 1):
        fig.add_trace(go.Bar(x=a.rezago, y=a[c], marker_color=AZUL, showlegend=False), row=1, col=j)
        for s in (1, -1):
            fig.add_hline(y=s * a.banda.iloc[0], line=dict(color=GRIS, dash="dot"), row=1, col=j)
    fig.update_xaxes(title_text="rezago (meses)")
    fig.update_layout(title="Autocorrelación de la tasa mensual (bandas: ±1,96/√n)")
    return fig


def varianza_pca(v):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=v.componente, y=v.varianza, name="por componente", marker_color=AZUL))
    fig.add_trace(go.Scatter(x=v.componente, y=v.acumulada, name="acumulada", mode="lines", line=dict(color=ORO)))
    fig.add_hline(y=0.8, line=dict(color=GRIS, dash="dot"), annotation_text="80 %")
    fig.update_layout(title="PCA de las numéricas: varianza explicada", xaxis_title="componente",
                      yaxis_title="proporción de la varianza", yaxis_tickformat=".0%")
    return fig


def dispersion(df, x, y, color, nombres, log_x=False, log_y=False):
    """Explorador bivariado: dos variables y un color (la clase o una categórica)."""
    fig = go.Figure()
    grupos = df[color].map(CLASE) if color == "y_tarde" else df[color].astype(str)
    for i, (g, d) in enumerate(df.groupby(grupos)):
        fig.add_trace(go.Scattergl(x=d[x], y=d[y], mode="markers", name=str(g),
                                   marker=dict(size=4, opacity=0.35,
                                               color=COLOR_CLASE.get(g, PALETA[i % 5]))))
    fig.update_layout(title=f"{nombres.get(y, y)} frente a {nombres.get(x, x)}", xaxis_title=nombres.get(x, x),
                      yaxis_title=nombres.get(y, y), xaxis_type="log" if log_x else None,
                      yaxis_type="log" if log_y else None)
    return fig
