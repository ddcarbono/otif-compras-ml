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
    caja = ["<b>Normalidad</b> (D'Agostino-Pearson K²)"]
    for k, et in CLASE.items():
        crudo = x[(df.y_tarde == k).to_numpy()].dropna()
        xs = crudo.clip(lim[0], lim[1])
        dens = np.histogram(xs, cortes)[0] / max(len(xs), 1) / ancho
        fig.add_trace(go.Bar(x=centro, y=dens, width=ancho, name=et, marker_color=COLOR_CLASE[et], opacity=0.6,
                             hovertemplate="%{x:.2f}: densidad %{y:.3f}<extra>" + et + "</extra>"))
        caja.append(f"{et}: " + _normalidad(crudo))
    eje = f"{nombre} ({unidad})" if unidad and unidad != "—" else nombre
    fig.add_annotation(text="<br>".join(caja), xref="paper", yref="paper", x=1, y=1.01, xanchor="right", yanchor="bottom",
                       showarrow=False, align="left", bgcolor="rgba(255,255,255,0.9)", bordercolor="#BBBBBB",
                       borderwidth=1, borderpad=6, font=dict(size=11))
    fig.update_layout(barmode="overlay", title=dict(text=nombre, y=0.97, yanchor="top"), margin=dict(t=120, b=95), legend_y=-0.3,
                      xaxis_title=("signo · log10(1 + |x|) de " if log else "") + eje, yaxis_title="densidad")
    return fig


def _normalidad(x):
    """K² de D'Agostino y Pearson (asimetría y curtosis juntas), con n, asimetría y curtosis en exceso. Sirve con miles de
    filas, donde Shapiro-Wilk ya no aplica; con tantas, cualquier desvío pequeño rechaza."""
    from scipy import stats
    x = np.asarray(x, float)
    if len(x) < 20 or np.ptp(x) == 0:
        return "sin datos suficientes"
    k2, p = stats.normaltest(x)
    es = lambda v, d=0: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    pt = "p < 0,001" if p < 0.001 else f"p = {es(p, 3)}"
    return (f"K² = {es(k2)}, {pt} (n = {es(len(x))}; asimetría {es(stats.skew(x), 2)}, "
            f"curtosis {es(stats.kurtosis(x), 2)})")


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
        fig.add_trace(go.Bar(x=t.fecha_pedido, y=t.filas, name="posiciones", marker_color=GRIS, opacity=0.2), secondary_y=True)
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


# --- modelos ---------------------------------------------------------------------------------------------------------
COLOR_MODELO = {"Dummy (tasa del entrenamiento)": GRIS, "Regla del proveedor": MARRON, "Regresión logística": AZUL,
                "XGBoost (las 112)": ORO, "Propuesta (K3 + log-normal + nuevas)": VERDE}


def curva_roc_pr(c, curva, metricas=None):
    """ROC o PR de cada modelo en la prueba (``curvas.csv``). La referencia es la diagonal (ROC) o la prevalencia (PR)."""
    fig = go.Figure()
    for nombre, d in c[c.curva == curva].groupby("modelo", sort=False):
        et = nombre
        if metricas is not None:
            k = "ROC-AUC" if curva == "ROC" else "PR-AUC"
            et += f" ({k} {metricas.set_index('modelo').loc[nombre, k]:.3f})".replace(".", ",")
        fig.add_trace(go.Scatter(x=d.x, y=d.y, mode="lines", name=et, line=dict(color=COLOR_MODELO.get(nombre, AZUL))))
    if curva == "ROC":
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=GRIS, dash="dot"))
        fig.update_layout(title="Curva ROC en la prueba 2026", xaxis_title="tasa de falsos positivos",
                          yaxis_title="sensibilidad")
    else:
        fig.update_layout(title="Curva de precisión y sensibilidad en la prueba 2026", xaxis_title="sensibilidad",
                          yaxis_title="precisión", yaxis_range=[0, 1])
    fig.update_layout(legend=dict(orientation="v", y=0.02, x=0.98, xanchor="right", yanchor="bottom"))
    return fig


def calibracion(t, titulo="Calibración en la prueba 2026 (deciles de probabilidad)", nombre="logística"):
    """Diagrama de confiabilidad por deciles de probabilidad predicha (``calibracion.csv``)."""
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=GRIS, dash="dot"))
    grupos = t.groupby("modelo", sort=False) if "modelo" in t else [(nombre, t)]
    for n, g in grupos:
        fig.add_trace(go.Scatter(x=g.predicha, y=g.observada, mode="lines+markers",
                                 line=dict(color=COLOR_MODELO.get(n, AZUL)), customdata=np.c_[g.decil, g.posiciones],
                                 name=n, hovertemplate="decil %{customdata[0]}<br>predicha %{x:.1%}<br>observada "
                                                       "%{y:.1%}<br>%{customdata[1]:,} posiciones<extra></extra>"))
    fig.update_layout(title=titulo, xaxis_title="probabilidad predicha",
                      yaxis_title="tasa de tarde observada", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], showlegend="modelo" in t,
                      legend=dict(y=0.02, x=0.98, xanchor="right", yanchor="bottom"))
    return fig


def matriz_confusion(t, umbral=None):
    """Matriz de confusión en conteos y en proporción de cada fila real (``confusion.csv``)."""
    orden = ["tarde", "a tiempo"]
    m = t.pivot(index="real", columns="predicho", values="posiciones").loc[orden, orden]
    prop = m.div(m.sum(axis=1), axis=0)
    texto = [[f"{m.iloc[i, j]:,}".replace(",", ".") + f"<br>{_pct(prop.iloc[i, j])}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=prop.values, x=[f"predicho {o}" for o in orden], y=[f"real {o}" for o in orden],
                               text=texto, texttemplate="%{text}", colorscale=[[0, "#FFFFFF"], [1, AZUL]], zmin=0, zmax=1,
                               showscale=False, hoverinfo="skip"))
    tit = "Matriz de confusión en la prueba 2026"
    if umbral is not None:
        tit += f" (umbral {umbral:.3f})".replace(".", ",")
    fig.update_layout(title=tit, yaxis_autorange="reversed", height=360)
    return fig


def coeficientes(t, nombres, n=20):
    """Los ``n`` coeficientes de mayor valor absoluto, con el nombre legible de la variable de origen."""
    t = t.head(n).copy()
    def etiqueta(v):  # numérica o entidad: su nombre; one-hot "linea_palma": "Línea de negocio: palma"
        v = v[3:] if v.startswith("te_") else v  # entidad codificada por objetivo
        if v in nombres:
            return nombres[v]
        k = next((k for k in sorted(nombres, key=len, reverse=True) if v.startswith(k + "_")), None)
        return f"{nombres[k]}: {v[len(k) + 1:]}" if k else v
    t["etiqueta"] = t.variable.map(etiqueta)
    t = t.iloc[::-1]
    fig = go.Figure(go.Bar(x=t.coeficiente, y=t.etiqueta, orientation="h",
                           marker_color=[ORO if v > 0 else VERDE for v in t.coeficiente],
                           customdata=t["razón de odds por 1 desv."],
                           hovertemplate="%{y}<br>coeficiente %{x:.3f}<br>razón de odds %{customdata:.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=GRIS))
    fig.update_layout(title=f"Los {n} coeficientes de mayor magnitud (oro: sube el riesgo; verde: lo baja)",
                      xaxis_title="coeficiente (log-odds por desviación estándar o por categoría)",
                      height=max(360, 24 * len(t) + 140), margin=dict(l=320))
    return fig


def curva_aprendizaje(t):
    fig = go.Figure()
    for c, color in [("AUC entrenamiento", ORO), ("AUC validación", AZUL)]:
        fig.add_trace(go.Scatter(x=t.filas, y=t[c], mode="lines+markers", name=c.replace("AUC ", ""),
                                 line=dict(color=color), hovertemplate="%{x:,} filas: AUC %{y:.4f}<extra></extra>"))
    fig.update_layout(title="Curva de aprendizaje de la regresión logística", xaxis_title="filas de entrenamiento",
                      yaxis_title="ROC-AUC", xaxis_type="log")
    return fig


def validacion_c(t):
    """AUC medio ± desviación de cada C en los tres pliegues temporales (``cv_C.csv``)."""
    g = t.groupby("C").AUC.agg(["mean", "std"]).reset_index()
    fig = go.Figure(go.Scatter(x=g.C, y=g["mean"], error_y=dict(type="data", array=g["std"], color=GRIS),
                               mode="lines+markers", line=dict(color=AZUL),
                               hovertemplate="C = %{x}<br>AUC medio %{y:.4f}<extra></extra>"))
    fig.update_layout(title="Elección de C: AUC medio en los tres pliegues temporales (± desviación)",
                      xaxis_title="C (inverso de la penalización L2)", yaxis_title="ROC-AUC", xaxis_type="log")
    return fig


# --- los 112 -----------------------------------------------------------------------------------------------------------
MODELO = {"xgboost": "XGBoost", "random_forest": "Random Forest", "logistica": "Logística (SGD)",
          "svm": "SVM lineal (SGD)", "arbol": "Árbol", "knn": "KNN (PCA 30)", "naive_bayes": "Naive Bayes"}
BALANCEO = {"ninguno": "ninguno", "smote": "SMOTE", "adasyn": "ADASYN", "pesos": "pesos de clase"}
OPTIMIZADOR = {"grid": "Grid", "random": "Random", "optuna": "Optuna (TPE)", "deap": "DEAP (genético)"}
COLOR_OPT = {"grid": GRIS, "random": AZUL, "optuna": ORO, "deap": VERDE}
COLOR_MOD = {"xgboost": ORO, "random_forest": VERDE, "logistica": AZUL, "svm": AZUL, "arbol": MARRON,
             "knn": GRIS, "naive_bayes": GRIS}
# siete modelos y cinco colores: los dos que repiten color van punteados
DASH_MOD = {"svm": "dot", "naive_bayes": "dot"}


def auc_por_bloque(m, nombres):
    """AUC externo por bloque semestral de la mejor combinación de cada modelo (``auc_por_bloque_modelos.csv``)."""
    fig = go.Figure()
    orden = m.drop(columns="bloque").mean().sort_values(ascending=False).index
    for c in orden:
        fig.add_trace(go.Scatter(x=m.bloque, y=m[c], mode="lines+markers", name=nombres.get(c, c),
                                 line=dict(color=COLOR_MOD.get(c, AZUL), dash=DASH_MOD.get(c, "solid")),
                                 hovertemplate="%{x}: AUC %{y:.3f}<extra>" + nombres.get(c, c) + "</extra>"))
    fig.update_layout(title="ROC-AUC por bloque externo (mejor combinación de cada modelo)", xaxis_title="bloque evaluado",
                      yaxis_title="ROC-AUC")
    return fig


def mapa_combinaciones(t, modelo, nombres_bal, nombres_opt):
    """AUC medio de un modelo en la rejilla balanceo × optimizador (``lideres.csv``)."""
    d = t[t.modelo == modelo].pivot(index="balanceo", columns="optimizador", values="AUC")
    d = d.loc[[b for b in nombres_bal if b in d.index], [o for o in nombres_opt if o in d.columns]]
    fig = go.Figure(go.Heatmap(z=d.values, x=[nombres_opt[o] for o in d.columns], y=[nombres_bal[b] for b in d.index],
                               text=[[f"{v:.4f}".replace(".", ",") for v in fila] for fila in d.values],
                               texttemplate="%{text}", colorscale=[[0, "#FFFFFF"], [1, AZUL]], showscale=False,
                               hovertemplate="%{y} · %{x}: AUC %{z:.4f}<extra></extra>"))
    fig.update_layout(title="AUC medio por balanceo y optimizador", height=320, yaxis_autorange="reversed")
    return fig


def arrepentimiento(c, x, nombres_opt, titulo, eje):
    """Distancia al mejor AUC interno conocido contra evaluaciones o contra tiempo (``curvas_optimizador*.csv``)."""
    fig = go.Figure()
    y = "mean" if "mean" in c.columns else "arrepentimiento"
    for o in nombres_opt:
        d = c[c.optimizador == o]
        fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="lines", name=nombres_opt[o], line=dict(color=COLOR_OPT[o]),
                                 hovertemplate="%{x:.0f}: %{y:.4f}<extra>" + nombres_opt[o] + "</extra>"))
    fig.update_layout(title=titulo, xaxis_title=eje, yaxis_title="AUC perdido frente al mejor conocido",
                      yaxis_type="log", xaxis_type="log" if x == "segundos" else None)
    return fig


def rangos_cd(r, cd, nombres):
    """Rango medio de cada modelo en los 8 bloques (1 = mejor) y la diferencia crítica de Nemenyi desde el mejor."""
    r = r.sort_values("rango_medio")
    et = [nombres.get(m, m) for m in r.modelo]
    fig = go.Figure(go.Scatter(x=r.rango_medio, y=et, mode="markers+text", text=[f"{v:.2f}".replace(".", ",") for v in r.rango_medio],
                               textposition="middle right", marker=dict(size=12, color=[COLOR_MOD.get(m, AZUL) for m in r.modelo]),
                               hovertemplate="%{y}: rango medio %{x:.3f}<extra></extra>"))
    x0 = r.rango_medio.iloc[0]
    fig.add_shape(type="rect", x0=x0, x1=x0 + cd, y0=-0.5, y1=len(r) - 0.5, fillcolor=ORO, opacity=0.12, line_width=0)
    fig.add_annotation(x=x0 + cd, y=len(r) - 0.5, text=f"diferencia crítica {cd:.2f}".replace(".", ","), showarrow=False,
                       yanchor="bottom", xanchor="right")
    fig.update_layout(title="Rango medio en los 8 bloques (Friedman) y diferencia crítica de Nemenyi (α = 0,05)",
                      xaxis_title="rango medio (1 = mejor)", xaxis_range=[0.5, 7.5], yaxis_autorange="reversed",
                      showlegend=False, height=380, margin=dict(l=150))
    return fig


def costo(df, nombres, nombres_bal):
    """Horas de CPU por modelo, apiladas por balanceo (``por_bloque.csv``)."""
    h = df.groupby(["modelo", "balanceo"]).segundos_total.sum().div(3600).unstack()
    h = h.loc[h.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for b, color in zip(nombres_bal, [AZUL, ORO, MARRON, VERDE]):
        fig.add_trace(go.Bar(y=[nombres.get(m, m) for m in h.index], x=h[b], name=nombres_bal[b], orientation="h",
                             marker_color=color, hovertemplate="%{y}: %{x:.1f} h<extra>" + nombres_bal[b] + "</extra>"))
    fig.update_layout(barmode="stack", title="Horas de CPU de los 112 por modelo y balanceo",
                      xaxis_title="horas de CPU (un núcleo)", height=380, margin=dict(l=150))
    return fig


def convergencia(c):
    """AUC de XGBoost en entrenamiento y en el semestre externo después de cada árbol (media de los 8 bloques, con el
    rango entre bloques del externo)."""
    m = c.groupby("arbol").agg(ent=("AUC_entrenamiento", "mean"), ext=("AUC_externo", "mean"),
                               lo=("AUC_externo", "min"), hi=("AUC_externo", "max")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[m.arbol, m.arbol[::-1]], y=np.r_[m.hi, m.lo[::-1]], fill="toself",
                             fillcolor="rgba(63,111,143,0.15)", line=dict(width=0), hoverinfo="skip",
                             name="rango entre bloques"))
    for col, nombre, color in [("ent", "entrenamiento", ORO), ("ext", "semestre externo", AZUL)]:
        fig.add_trace(go.Scatter(x=m.arbol, y=m[col], mode="lines", name=nombre, line=dict(color=color),
                                 hovertemplate="árbol %{x}: AUC %{y:.4f}<extra></extra>"))
    fig.update_layout(title="Convergencia de XGBoost: AUC por número de árboles (media de los 8 bloques)",
                      xaxis_title="árboles", yaxis_title="ROC-AUC")
    return fig


def semillas(s, nombres):
    """AUC de cada modelo en cada bloque con 10 semillas: un punto por semilla."""
    fig = go.Figure()
    for i, (m, g) in enumerate(s.groupby("modelo", sort=False)):
        fig.add_trace(go.Box(x=g.bloque, y=g.AUC, name=nombres.get(m, m), boxpoints="all", jitter=0.3,
                             marker=dict(size=4), hovertemplate="%{x}: AUC %{y:.4f}<extra></extra>"))
    fig.update_layout(title="AUC por bloque con 10 semillas por modelo", boxmode="group", xaxis_title="bloque",
                      yaxis_title="ROC-AUC")
    return fig


def shap_global(g, nombres, n=15):
    """Media del valor SHAP absoluto por variable (``shap_global.csv``), las n mayores."""
    g = g.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(x=g.shap_abs_medio, y=[nombres.get(v, v) for v in g.variable], orientation="h",
                           marker_color=ORO, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title="Qué pesa en XGBoost en 2026: media del valor SHAP absoluto",
                      xaxis_title="media de |SHAP| (logit)", height=480)
    return fig


def forma_plazo(t, col, grupos):
    """Efecto de los días hábiles de plazo al liberar sobre el logit, por tramo y por grupo (``forma_*.csv``): SHAP medio
    de XGBoost (línea llena) contra el término lineal de la logística (punteada)."""
    fig = go.Figure()
    colores = [ORO, VERDE, AZUL, MARRON, GRIS]
    for i, g in enumerate(grupos):
        d = t[t[col] == g]
        c = colores[i % len(colores)]
        fig.add_trace(go.Scatter(x=d.tramo, y=d.xgboost, mode="lines+markers", name=f"{g} · XGBoost",
                                 line=dict(color=c), hovertemplate="%{x} días: %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=d.tramo, y=d.logistica, mode="lines", name=f"{g} · logística",
                                 line=dict(color=c, dash="dot"), hovertemplate="%{x} días: %{y:.2f}<extra></extra>"))
    fig.update_layout(title="Efecto del plazo al liberar sobre el logit: XGBoost (SHAP) contra la logística",
                      xaxis_title="días hábiles de plazo al liberar", yaxis_title="efecto sobre el logit", height=560,
                      legend=dict(orientation="h", y=-0.22, x=0, xanchor="left", yanchor="top"))
    return fig


def log_perdida(p):
    """Log-pérdida de una posición según la probabilidad predicha: −ln p si llegó tarde y −ln(1 − p) si llegó a tiempo,
    con el punto de ``p`` marcado en las dos curvas."""
    x = np.linspace(0.005, 0.995, 199)
    fig = go.Figure()
    for nombre, y, yp, color, linea in [("llegó tarde: −ln p", -np.log(x), -np.log(p), ORO, "solid"),
                                        ("llegó a tiempo: −ln(1 − p)", -np.log(1 - x), -np.log(1 - p), VERDE, "dash")]:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=nombre, line=dict(color=color, dash=linea),
                                 hovertemplate="p %{x:.2f}<br>pérdida %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[p], y=[yp], mode="markers+text", text=[f"{yp:.2f}".replace(".", ",")],
                                 textposition="top right", marker=dict(color=color, size=10), showlegend=False,
                                 hoverinfo="skip"))
    fig.update_layout(title=f"Log-pérdida con p = {p:.2f}".replace(".", ","), xaxis_title="probabilidad predicha de tarde, p",
                      yaxis_title="pérdida", yaxis_range=[0, 5.3], xaxis_range=[0, 1], height=400,
                      legend=dict(y=-0.3), margin=dict(b=90))
    return fig


# --- exploración multivariada (página EDA) ------------------------------------------------------------------------------
def pca_varianza(var):
    """Varianza explicada por componente (barras) y acumulada (línea)."""
    k = np.arange(1, len(var) + 1)
    fig = go.Figure(go.Bar(x=k, y=var, marker_color=AZUL, name="por componente",
                           hovertemplate="PC%{x}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=k, y=np.cumsum(var), mode="lines+markers", line=dict(color=ORO), name="acumulada",
                             hovertemplate="hasta PC%{x}: %{y:.1%}<extra></extra>"))
    fig.update_layout(title="Varianza explicada", xaxis_title="componente", yaxis_title="varianza explicada",
                      yaxis_tickformat=".0%", xaxis_dtick=1)
    return fig


def pca_plano(puntajes, y, cargas, nombres, var):
    """Las posiciones en el plano PC1-PC2 por clase, con las cargas de cada variable como flechas."""
    fig = go.Figure()
    for k, et in CLASE.items():
        s = np.asarray(y) == k
        fig.add_trace(go.Scattergl(x=puntajes[s, 0], y=puntajes[s, 1], mode="markers", name=et,
                                   marker=dict(size=4, opacity=0.3, color=COLOR_CLASE[et])))
    esc = 0.8 * np.abs(puntajes[:, :2]).max(axis=0) / np.abs(cargas.to_numpy()).max()
    for v, (a, b) in cargas.iterrows():
        x, y_ = a * esc.min(), b * esc.min()
        fig.add_annotation(x=x, y=y_, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2,
                           arrowcolor=MARRON, text="")
        fig.add_annotation(x=x, y=y_, xref="x", yref="y", showarrow=False, text=nombres.get(v, v), font=dict(size=10),
                           xanchor="left" if x >= 0 else "right", yanchor="bottom" if y_ >= 0 else "top",
                           bgcolor="rgba(255,255,255,0.7)")
    fig.update_layout(title="Plano de las dos primeras componentes", xaxis_title=f"PC1 ({var[0]:.0%})",
                      yaxis_title=f"PC2 ({var[1]:.0%})", height=520)
    return fig



def tasa_por_tramo(t, nombre, tasa_global):
    """Tasa de tarde por tramo (cuantil) de una numérica, con el número de posiciones de cada tramo en el hover y la tasa
    global como referencia (``t``: columnas tramo, tasa, posiciones)."""
    fig = go.Figure(go.Scatter(x=t.tramo, y=t.tasa, mode="lines+markers", line=dict(color=ORO), marker=dict(size=8),
                               customdata=t.posiciones, hovertemplate="%{x}<br>tasa de tarde %{y:.1%}<br>"
                                                                      "%{customdata:,} posiciones<extra></extra>"))
    fig.add_hline(y=tasa_global, line=dict(color=GRIS, dash="dot"), annotation_text="tasa de la selección",
                  annotation_position="top left")
    fig.update_layout(title=f"Tasa de tarde por tramo de {nombre.lower()}", xaxis_title=f"{nombre} (tramo por cuantiles)",
                      yaxis_title="tasa de tarde", yaxis_tickformat=".0%", yaxis_rangemode="tozero")
    return fig


def entidades_nuevas(t):
    """Porcentaje de posiciones con un valor de cada entidad que el entrenamiento no vio, en la validación y la prueba."""
    t = t[(t["nuevo en validacion (%)"] > 0) | (t["nuevo en prueba (%)"] > 0)].iloc[::-1]
    fig = go.Figure()
    for col, et, color in [("nuevo en validacion (%)", "validación 2025", GRIS), ("nuevo en prueba (%)", "prueba 2026", AZUL)]:
        fig.add_trace(go.Bar(y=t.nombre, x=t[col], orientation="h", name=et, marker_color=color,
                             hovertemplate="%{y}: %{x:.1f} %<extra>" + et + "</extra>"))
    fig.update_layout(title="Posiciones con una entidad que el entrenamiento no vio", xaxis_title="% de posiciones",
                      barmode="group", height=max(320, 40 * len(t) + 120), margin=dict(l=250))
    return fig
