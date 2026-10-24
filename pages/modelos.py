"""Sección 3: modelos. La predicción en vivo (el modelo original y XGBoost, con Platt), la tabla de líderes de las 112 combinaciones
(validación anidada en 8 bloques semestrales, con Friedman, Nemenyi, Wilcoxon, DeLong y BCa), el mejor modelo en la
prueba 2026 con SHAP y la comparación 2 × 2 con la propuesta original, y el modelo base (regresión logística).
Todo se lee de ``data/modelos`` (lo escribe ``src.preparar`` a partir de ``src.modelo_base``): aquí no se entrena."""
import json

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dash_table, dcc, html

from src import plots
from src.data import DATA, diccionario, modelo
from src.vista_prediccion import vista
from src.vista_original import vista as vista_original

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


def tabla_pruebas():
    t = modelo("pruebas_resumen")
    filas = [{"modelo": plots.MODELO[f.rival], "bloques que gana XGBoost": f"{f.bloques_gana} de 8",
              "DeLong significativo (Holm)": f"{f.bloques_delong_holm} de 8",
              "diferencia de AUC [BCa 95 %]": f"{_n(f.dif_AUC)} [{_n(f.dif_inf)}; {_n(f.dif_sup)}]",
              "δ de Cliff": f"{_n(f.cliff, 2)} ({f.magnitud})"} for f in t.itertuples()]
    return dash_table.DataTable(data=filas, columns=[{"name": c, "id": c} for c in filas[0]],
                                style_table={"overflowX": "auto"}, style_header={"fontWeight": "bold"},
                                style_cell={"textAlign": "left", "fontSize": 13})


def calibracion_xgb():
    f = modelo("fiabilidad_xgboost").rename(columns={"filas": "posiciones"})
    return plots.calibracion(f, "XGBoost en los 8 bloques de validación (deciles de probabilidad)", "XGBoost")


P26 = modelo("p26_metricas").set_index("modelo")
XG = P26.loc["XGBoost (las 112)"]
DIF = modelo("p26_diferencias").set_index("comparación")


def tabla_simple(df, dec=3):
    filas = [{c: (_n(v, dec) if isinstance(v, float) else v) for c, v in r.items()} for r in df.to_dict("records")]
    return dash_table.DataTable(data=filas, columns=[{"name": c, "id": c} for c in df.columns],
                                style_table={"overflowX": "auto"}, style_header={"fontWeight": "bold"},
                                style_cell={"textAlign": "left", "fontSize": 13})


def pestana_2026():
    d = DIF.iloc[0]
    m = P26.reset_index()[["modelo", "ROC-AUC", "ROC-AUC inf", "ROC-AUC sup", "PR-AUC", "F1", "precisión",
                           "sensibilidad", "ECE", "umbral"]]
    c = modelo("p26_criticas").set_index("cifra")
    od = modelo("p26_ordenes")
    ent = lambda v: f"{v:,.0f}".replace(",", ".")
    dif = lambda f: f"{ent(c.loc[f, 'total'])} [{ent(c.loc[f, 'BCa inf'])}; {ent(c.loc[f, 'BCa sup'])}]"
    cr = pd.DataFrame({
        "": ["Tardías atrapadas", "De ellas, con stock en cero"],
        "XGBoost": [ent(c.loc["XGBoost: tardías atrapadas", "total"]), ent(c.loc["XGBoost: críticas atrapadas", "total"])],
        "Logística": [ent(c.loc["logística: tardías atrapadas", "total"]),
                      ent(c.loc["logística: críticas atrapadas", "total"])],
        "Diferencia [BCa 95 %]": [dif("N: tardías de más"), dif("M: de ellas, con stock en cero")],
        "Diferencia por mes": [ent(c.loc["N: tardías de más", "por mes"]),
                               ent(c.loc["M: de ellas, con stock en cero", "por mes"])]})
    return html.Div([
        html.H3("XGBoost en la prueba 2026"),
        html.P("El ganador de las 112 (sin balanceo, Optuna), con Platt y el umbral de máximo F1 fijados en la validación "
               "2025 y congelados antes de mirar 2026. Es la tercera mirada del proyecto a la prueba: se lee como "
               "exploratoria. Intervalos BCa al 95 % que remuestrean órdenes completas."),
        dbc.Row([cifra(f"{_n(XG['ROC-AUC'])}", f"ROC-AUC [{_n(XG['ROC-AUC inf'])}; {_n(XG['ROC-AUC sup'])}]"),
                 cifra(_n(XG.F1), f"F1 al umbral {_n(XG.umbral)}"),
                 cifra(f"+{_n(d.diferencia)}", f"de AUC sobre la logística [{_n(d['BCa inf'])}; {_n(d['BCa sup'])}]"),
                 cifra(_n(XG.ECE), "error de calibración esperado (ECE), con Platt")]),
        tarjeta_formula(1),
        flujo("Pipeline final", PIPE_XGB),
        tabla_simple(m),
        dbc.Row([dbc.Col(dcc.Graph(figure=plots.curva_roc_pr(modelo("p26_curvas"), "ROC", P26.reset_index()),
                                   config=CONFIG), lg=6),
                 dbc.Col(dcc.Graph(figure=plots.calibracion(modelo("p26_calibracion"),
                                                            "Calibración en la prueba 2026 (deciles)"), config=CONFIG),
                         lg=6)], className="mt-3"),
        html.H4("Qué mira el modelo (SHAP)"),
        dbc.Row([dbc.Col(dcc.Graph(figure=plots.shap_global(modelo("p26_shap_global"), NOMBRES), config=CONFIG), lg=6),
                 dbc.Col(dcc.Graph(figure=plots.matriz_confusion(modelo("p26_confusion"), XG.umbral), config=CONFIG),
                         lg=6)]),
        html.P("Cómo leer: a la izquierda, cuánto mueve en promedio cada variable el logit de XGBoost en 2026 (las "
               "entidades entran como su tasa codificada). El plazo hábil que queda al liberar pesa casi el doble que el "
               "proveedor. A la derecha, las posiciones de 2026 según lo que pasó y lo que dijo el modelo.",
               className="small text-muted"),
        html.H4("¿Por qué XGBoost y no la logística?"),
        html.P("Un retraso no visto cuesta más que una falsa alarma. Con el mismo esfuerzo de revisión, el 20 % de "
               "posiciones de mayor riesgo en 2026, XGBoost atrapa más tardías; «con stock en cero» es material que el "
               "mes anterior no tenía existencias. Intervalos BCa por órdenes; el de las críticas roza el cero. Entrenar "
               "tarda minutos: se puede reentrenar cada mes. En los 8 bloques gana los 8 (Wilcoxon-Holm 0,047, Cliff "
               "0,34).", className="small text-muted"),
        tabla_simple(cr),
        html.H4("Llevado a órdenes: cuántas seguir"),
        html.P(f"El comprador sigue órdenes, no posiciones. El riesgo de una orden es el de su posición más riesgosa, y "
               f"una orden es tardía si tiene al menos una posición tardía ({ent(od.tardías.iloc[0])} de "
               f"{ent(od.órdenes.iloc[0])} en 2026). Siguiendo el {od.loc[2, 'fracción de las órdenes']:.0%} de las "
               f"órdenes se ve venir el 69 % de las órdenes con retraso. Es menos que por posiciones (33 % para el "
               f"69 %): una orden tardía suele traer varias posiciones tardías juntas.", className="small text-muted"),
        tabla_simple(pd.DataFrame({
            "Encontrar": [f"{v:.0%}" for v in od.meta],
            "Órdenes tardías": [ent(v) for v in od["órdenes tardías"]],
            "Órdenes a seguir": [ent(v) for v in od["órdenes a seguir"]],
            "% de las órdenes": [f"{v:.1%}".replace(".", ",") for v in od["fracción de las órdenes"]],
            "Seguidas por cada tardía": [f"{v:.2f}".replace(".", ",") for v in od["seguidas por tardía"]]})),
    ])


PUB = {"airlines": "Vuelos (Airlines)", "amazon": "Acceso de empleados (Amazon)", "click": "Publicidad (Click)",
       "nyc_bus": "Buses escolares de Nueva York", "zurich": "Transporte de Zúrich",
       "diabetes": "Reingreso hospitalario (Diabetes130US)", "noshow": "Citas médicas no asistidas"}


def pestana_original():
    pr = P26.loc["Propuesta (K3 + log-normal + nuevas)"]
    k0 = modelo("p26_contrastes").iloc[0]
    m = P26.loc[["XGBoost (las 112)", "Propuesta (K3 + log-normal + nuevas)"]].reset_index()[
        ["modelo", "ROC-AUC", "ROC-AUC inf", "ROC-AUC sup", "PR-AUC", "F1", "precisión", "sensibilidad", "ECE"]]
    dd = modelo("p26_dos_por_dos")
    dd = dd.assign(**{"ROC-AUC [IC 95 %]": [f"{_n(a)} [{_n(b)}; {_n(c)}]" for a, b, c in zip(dd["ROC-AUC"], dd.inf, dd.sup)]})[
        ["celda", "ROC-AUC [IC 95 %]"]]
    k = modelo("p26_contrastes")
    k = k.assign(**{"ΔROC-AUC [IC 95 %]": [f"{_n(a, 4)} [{_n(b, 4)}; {_n(c, 4)}]"
                                           for a, b, c in zip(k["ΔROC-AUC"], k.inf, k.sup)]})[["contraste", "ΔROC-AUC [IC 95 %]"]]
    p = modelo("publicos")
    gana, emp, pierde = ((p.veredicto == v).sum() for v in ("K3 gana", "empate", "K3 pierde"))
    pub = pd.DataFrame({
        "ronda": p.ronda, "conjunto": p.conjunto.map(PUB), "posiciones": [f"{v:,}".replace(",", ".") for v in p.n],
        "AUC XGBoost": [_n(v) for v in p.auc_xgb], "AUC K3": [_n(v) for v in p.auc_k3],
        "ΔAUC [IC 95 %]": [f"{_n(a)} [{_n(b)}; {_n(c)}]" for a, b, c in zip(p.dif, p.ic_lo, p.ic_hi)],
        "p (Holm)": ["< 0,001" if v < 0.001 else _n(v) for v in p.p_holm], "veredicto": p.veredicto,
        "F1 XGBoost": [_n(v) for v in p.f1_xgb], "F1 K3": [_n(v) for v in p.f1_k3]})
    return html.Div([
        html.H3("Modelo original: boosting mixto ordinal"),
        html.P("LightGBM con efectos aleatorios por entidad sobre los días de retraso en cortes, con el plazo "
               "log-normal y 32 variables de plazo real. Congelado (Platt y umbral de 2025) antes de medir 2026; "
               "intervalos por remuestreo de órdenes."),
        flujo("El modelo en cuatro fases", FASES_ORIGINAL),
        dbc.Row([cifra(_n(pr["ROC-AUC"]), f"ROC-AUC en 2026 [{_n(pr['ROC-AUC inf'])}; {_n(pr['ROC-AUC sup'])}]"),
                 cifra(_n(pr.F1), f"F1 al umbral {_n(pr.umbral)}"),
                 cifra(f"+{_n(k0['ΔROC-AUC'])}", f"de AUC sobre el ganador de las 112 [{_n(k0.inf)}; {_n(k0.sup)}]"),
                 cifra(_n(pr.ECE), "error de calibración esperado (ECE)")]),
        tarjeta_formula(2),
        tarjeta_formula(3, None),
        tarjeta_formula(4, None),
        flujo("Pipeline final", PIPE_ORIGINAL),
        tabla_simple(m),
        html.H4("Modelo × información: la comparación 2 × 2", className="mt-4"),
        html.P("Separa lo que aporta el modelo de lo que aporta la información: cada modelo con las 136 variables del "
               "proyecto y con las 136 más las 32 nuevas. La ablación quita a K3 el plazo log-normal.",
               className="small text-muted"),
        dbc.Row([dbc.Col(tabla_simple(dd), lg=5), dbc.Col(tabla_simple(k), lg=7)]),
        html.P("Cómo leer: la información nueva le suma a XGBoost casi una centésima de AUC; el modelo, con las mismas 136 "
               "variables, otro tanto. Con las dos juntas, lo que agrega cada pieza por separado ya no es significativo: "
               "se solapan, porque las dos capturan la historia de cada proveedor y material.",
               className="small text-muted"),
        html.H4("La propuesta en otros datos", className="mt-4"),
        html.P("K3 contra el XGBoost de las 112 en siete conjuntos públicos de OpenML, con el mismo protocolo: Optuna con "
               "20 evaluaciones sobre la validación, una sola medida en la prueba y DeLong pareado con Holm sobre los "
               "siete. El criterio de la segunda ronda (al menos dos de tres rasgos: entidades que se repiten, orden en el "
               "tiempo, desenlace continuo) se fijó antes de correrla. El plazo log-normal y las variables nuevas no "
               "aplican fuera de OTIF.", className="small text-muted"),
        tabla_simple(pub),
        html.P(f"K3 gana en {gana}, empata en {emp} y pierde en {pierde}; F1 mayor en {(p.f1_k3 > p.f1_xgb).sum()} de "
               f"{len(p)}. Wilcoxon sobre las siete diferencias: p = 0,81, sin superioridad general. Gana donde una entidad "
               "tiene historia propia en el tiempo (el paciente que vuelve), como el proveedor en OTIF; empata en "
               "transporte, con entidades pocas y estables; pierde sin tiempo ni desenlace continuo. Cuidado: esa "
               "separación se vio después de la segunda ronda; confirmarla pide una tercera fijada antes de correr.",
               className="small text-muted mt-2"),
        html.P("La pestaña Predicción sirve este modelo con posiciones reales de 2026: cada una trae sus entidades ya "
               "convertidas en números (efectos y tasas), sin sus códigos.", className="small text-muted"),
    ])


def cifra(valor, texto):
    return dbc.Col(dbc.Card(dbc.CardBody([html.Div(valor, className="cifra"), html.Div(texto, className="cifra-texto")]),
                            className="h-100"), md=3, xs=6, className="mb-3")


FORMULAS = [
    ("Modelo base: regresión logística",
     r"$$\hat p(x) = \sigma(\beta_0 + x^\top\beta)$$"
     r"$$\min_\beta \; \sum_i \ell(y_i, \hat p_i) + \frac{1}{2C}\lVert\beta\rVert^2$$",
     "Una recta en el logit: cada variable empuja siempre igual. La penalización L2 (C por validación cruzada) encoge "
     "los coeficientes. Bien calibrada sin corrección."),
    ("XGBoost: suma de árboles",
     r"$$F(x) = \sum_{m=1}^{M} \eta\, f_m(x)$$"
     r"$$\hat p(x) = \sigma\big(a\,F(x) + b\big)$$"
     r"$$\min_{f_m} \sum_i \ell(y_i, \hat p_i) + \gamma T + \tfrac{1}{2}\lambda \textstyle\sum_j w_j^2$$"
     r"$$g_i = \hat p_i - y_i, \qquad h_i = \hat p_i\,(1 - \hat p_i)$$"
     r"$$w_j^* = -\frac{G_j}{H_j + \lambda}$$"
     r"$$\text{ganancia} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - "
     r"\frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$",
     "Cada árbol corrige lo que los anteriores predicen mal. g es el residuo con signo (positivo si el modelo "
     "sobreestima el riesgo de una posición que llegó a tiempo) y h la varianza de Bernoulli: grande donde el modelo duda "
     "(p̂ cerca de 0,5), pequeña donde ya está seguro. Con G y H, sus sumas en la hoja, el peso óptimo es un paso de "
     "Newton, el gradiente sobre la curvatura, encogido por λ. Una hoja se parte en izquierda (L) y derecha (R) solo si "
     "la ganancia es positiva: γ es el costo de cada hoja nueva. a y b son la calibración de Platt, fijada en 2025."),
    ("Modelo original: efectos por entidad y cortes ordenados",
     r"$$P(d_i > c \mid x_i) = \sigma\Big(f(x_i, c) + \sum_{e} b_{e(i)}\Big)$$"
     r"$$c \in \{0, 1, 5, 15\}, \qquad b_e \sim N(0, \sigma_e^2)$$"
     r"$$f(x, c) = \sum_{m} \eta\, f_m(x, c), \qquad w_j^* = -\frac{G_j}{H_j + \lambda}, \qquad \frac{\partial f}{\partial c} \le 0$$",
     "f es un LightGBM: la misma suma de árboles, la misma log-pérdida y el mismo peso de hoja que XGBoost. Difiere en "
     "que crece por hoja (parte la de mayor ganancia, hasta 154 hojas) y no por niveles (XGBoost: profundidad 10), y "
     "aquí recibe los efectos como desplazamiento fijo del logit y es monótono decreciente en el corte c. "
     "d son los días de retraso; se predice en c = 1. Cada b es un intercepto aleatorio por "
     "proveedor, material, solicitante y comprador, "
     "encogido hacia 0 si tiene pocas posiciones. Recibe además el plazo log-normal y las 32 variables de plazo real."),
    ("Entidades: tasa encogida hacia su nivel superior",
     r"$$\hat p(v) = \frac{\text{retrasos}(v) + M\,\hat p(\text{padre}(v))}{n(v) + M}, \qquad M = 20$$",
     "Cada proveedor, material o persona entra como su tasa de retraso, encogida hacia la de su nivel superior (el "
     "proveedor, la de su región): con pocas posiciones pesa la del padre, y un valor nuevo recibe la del padre. Se "
     "calcula fuera de pliegue y agrupada por orden, para que una posición no vea su propio objetivo ni el de su orden. "
     "La usan la logística y XGBoost para las 11 entidades; el original, para las 7 que no llevan efecto aleatorio."),
    ("Platt: del puntaje al riesgo",
     r"$$\hat p = \sigma(a\,z + b), \qquad (a, b) = \arg\min_{a, b} \sum_{i \in 2025} \ell\big(y_i, \sigma(a\,z_i + b)\big)$$",
     "z es el logit crudo del modelo (aquí, el del LightGBM más los efectos, en c = 1). Platt es una regresión logística "
     "de y sobre z, ajustada en la validación 2025 y congelada antes de 2026: a < 1 encoge una confianza excesiva y b "
     "corrige el nivel. Como es creciente en z, no cambia el orden ni el AUC; solo hace que el número se lea como "
     "riesgo. La usan XGBoost y el original; la logística ya sale calibrada."),
]


def flujo(titulo, pasos):
    """El pipeline final de un modelo: una fila de pasos (título y detalle) unidos por flechas."""
    cajas = []
    for k, (t, d) in enumerate(pasos):
        if k:
            cajas.append(html.Div("→", className="align-self-center text-muted fs-4 px-1"))
        cajas.append(dbc.Card(dbc.CardBody([html.H6(f"{k + 1}. {t}", className="mb-1"), html.Small(d)]),
                              className="paso", style={"flex": "1 1 160px", "minWidth": "150px"}))
    return html.Div([html.H4(titulo, className="mt-4"), html.Div(cajas, className="d-flex flex-wrap gap-1 mb-2")])


ART = DATA.parent / "artifacts"
FX = json.loads((ART / "xgboost.metadata.json").read_text(encoding="utf-8"))
FO = json.loads((ART / "original_servido.json").read_text(encoding="utf-8"))
CO = json.loads((DATA.parent / "original" / "original_congelado.json").read_text(encoding="utf-8"))
_e = lambda v, d=3: f"{v:.{d}f}".replace(".", ",")
HX = FX["hiperparametros"]
PIPE_XGB = [
    ("Entrada", "136 variables conocidas el día de la liberación."),
    ("Preprocesamiento", "Numéricas (83): signo · log, mediana, estandarización. Categóricas (42): one-hot, las de menos "
                         "de 30 filas juntas. Entidades (11): tasa encogida (M = 20), fuera de pliegue. Todo se ajusta "
                         "solo con el entrenamiento."),
    ("XGBoost", f"{HX['n_estimators']} árboles, profundidad {HX['max_depth']}, η = {_e(HX['learning_rate'])}, "
                f"{_e(HX['subsample'], 2)} de las filas y {_e(HX['colsample_bytree'], 2)} de las columnas por árbol; "
                "sin balanceo, elegido con Optuna. Reentrenado con 2021-2025."),
    ("Platt", f"σ({_e(FX['platt'][0])} · F + {_e(FX['platt'][1])}), ajustado en la validación 2025 y congelado."),
    ("Umbral", f"Alerta si el riesgo ≥ {_e(FX['umbral'])} (máximo F1 en 2025)."),
]
# las cuatro fases del original, en el orden en que corren: cuáles son modelos y para qué
FASES_ORIGINAL = [
    ("Plazo log-normal", "Modelo (LightGBM de regresión): estima el plazo real esperado. Entrega dist_mu y dist_z."),
    ("Cortes ordinales", "Repite cada posición en c = 0, 1, 5 y 15 días de retraso."),
    ("Boosting mixto", "Modelo (LightGBM + efectos aleatorios por entidad): da el riesgo. Efectos por semestre, "
                       "con el pasado (K3)."),
    ("Platt", "Modelo (logística de dos parámetros): vuelve el puntaje un riesgo calibrado."),
]

PIPE_ORIGINAL = [
    ("Entrada", "136 variables más 32 de plazo real previo (mediana y p80 de las últimas 20 entregas, holguras)."),
    ("Entidades y categóricas", "Proveedor, material, solicitante y comprador: efecto aleatorio b, ordenado por "
                                "semestre (K3). Las otras 7: tasa encogida (M = 20). Categóricas: nativas de LightGBM."),
    ("Plazo log-normal", "Un LightGBM de regresión sin las variables de la promesa estima μ, el plazo real esperado; "
                         "z mide cuánto alcanza la promesa."),
    ("Cortes ordinales", "Cada posición se repite en c ∈ {0, 1, 5, 15} con la etiqueta [días de retraso > c]."),
    ("LightGBM", f"{FO['arboles']} árboles, {CO['prm']['num_leaves']} hojas, η = {_e(CO['prm']['learning_rate'])}, "
                 "monótono decreciente en c; Σ b como desplazamiento del logit. Reentrenado con 2021-2025."),
    ("Platt", f"σ({_e(FO['platt'][0])} · logit + {_e(FO['platt'][1])}) en c = 1, ajustado en 2025 y congelado."),
    ("Umbral", f"Alerta si el riesgo ≥ {_e(FO['umbral'])} (máximo F1 en 2025)."),
]


def tarjeta_formula(i, titulo="Formulación"):
    """La formulación de un modelo (``FORMULAS[i]``): la fórmula a la izquierda y su lectura a la derecha."""
    t, f, d = FORMULAS[i]
    return html.Div([html.H4(titulo, className="mt-4") if titulo else None, dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col(dcc.Markdown(f.replace("$$$$", "$$\n\n$$"), mathjax=True, style={"overflowX": "auto"}), md=7),
        dbc.Col([html.H6(t), html.Small(d, className="text-muted")], md=5)])), className="mb-2")])


def formulacion():
    """La de la logística, con la log-pérdida que comparten los tres modelos y su gráfica interactiva."""
    return html.Div([
        tarjeta_formula(0),
        html.P("Los tres modelos (base, óptimo y original) estiman la misma probabilidad con la logística σ y minimizan "
               "la misma pérdida, la log-pérdida (menos la log-verosimilitud de Bernoulli); cambia la forma de la "
               "función.", className="mb-0"),
        dcc.Markdown(r"$$\ell(y, \hat p) = -\,y \ln \hat p - (1 - y)\ln(1 - \hat p), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$",
                     mathjax=True),
        html.P("Mueva p: equivocarse con seguridad cuesta sin límite; dudar (p = 0,5) cuesta ln 2 = 0,69 en los dos casos.",
               className="small text-muted mb-1"),
        dcc.Slider(id="m-p", min=0.01, max=0.99, step=0.01, value=0.3, marks={v: f"{v:.1f}".replace(".", ",")
                                                                              for v in (0.1, 0.3, 0.5, 0.7, 0.9)}),
        dcc.Graph(id="m-perdida", config=CONFIG),
    ])


def pestana_prediccion():
    """Las dos predicciones en vivo: el modelo original (el foco del proyecto) y el XGBoost de las 112, para comparar."""
    return html.Div([
        dbc.RadioItems(id="p-modelo", value="original", inline=True, className="mb-3",
                       options=[{"label": "Modelo original (propuesta)", "value": "original"},
                                {"label": "XGBoost (las 112)", "value": "xgboost"}]),
        html.Div(vista_original(), id="p-vista-original"),
        html.Div(vista(), id="p-vista-xgboost", style={"display": "none"}),
    ])


@callback(Output("p-vista-original", "style"), Output("p-vista-xgboost", "style"), Input("p-modelo", "value"))
def elegir_modelo(m):
    return ({}, {"display": "none"}) if m == "original" else ({"display": "none"}, {})


PESTANAS = ("t-base", "t-112", "t-2026", "t-original", "t-prediccion")


def layout(pestana="t-base", **_):
    """La página; ``?pestana=t-prediccion`` abre otra pestaña (la usa ``presentacion/capturar_tablero.py``)."""
    pestana = pestana if pestana in PESTANAS else "t-base"
    return html.Div([
        html.H2("Modelos"),
        dbc.Tabs([
            dbc.Tab(html.Div([
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
                formulacion(),
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
                html.P("Cómo leer: la curva ROC sube más rápido cuanto mejor ordena el modelo (la diagonal es el azar). La curva de precisión y sensibilidad dice, para cada fracción de las tardías que se quiera encontrar, qué fracción de las alertas acierta; su referencia es la prevalencia, no 0,5.", className="small text-muted"),
                html.H4("Umbral y calibración"),
                dbc.Row([dbc.Col(dcc.Graph(figure=plots.matriz_confusion(modelo("confusion"), LOG.umbral), config=CONFIG), lg=6),
                         dbc.Col(dcc.Graph(figure=plots.calibracion(modelo("calibracion")), config=CONFIG), lg=6)]),
                html.P("Cómo leer: la matriz cuenta las posiciones de 2026 según lo que pasó (filas) y lo que dijo el modelo (columnas), con el porcentaje de cada fila. En la calibración, cada punto es un decil de probabilidad: si cae sobre la diagonal, «30 % de riesgo» quiere decir que llega tarde el 30 % de esas posiciones.", className="small text-muted"),
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
            ]), label="Línea base (logística)", tab_id="t-base", className="pt-3"),
            dbc.Tab(html.Div([
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
                html.P("Cómo leer: a la izquierda, cada punto es el puesto medio de un modelo en los 8 bloques (1 = el mejor); los modelos dentro de la franja sombreada no se distinguen del primero según Nemenyi. A la derecha, el AUC de cada modelo en cada semestre: las líneas que suben y bajan juntas indican que la dificultad es del semestre, no del modelo.", className="small text-muted"),
                html.P(f"Friedman sobre los 8 bloques con la mejor combinación de cada modelo (elegida por AUC interno): "
                       f"χ² = {_n(FR.chi2, 1)}, p < 0,001. Nemenyi (diferencia crítica {_n(FR.CD, 2)}) solo separa a XGBoost del "
                       f"árbol, KNN y Naive Bayes; la prueba de Wilcoxon pareada del mejor contra cada uno, con Holm, da p = "
                       f"{_n(W.p_holm.max(), 3)} en las seis comparaciones: el mínimo posible con 8 bloques.", className="small text-muted"),
                html.H4("Dentro de cada bloque: DeLong, intervalos BCa y delta de Cliff", className="mt-3"),
                html.P("Con las predicciones de cada posición, la prueba de DeLong compara el AUC de XGBoost con el de cada rival en "
                       "cada bloque (48 pruebas, con Holm). El intervalo BCa de la diferencia media remuestrea órdenes completas, "
                       "porque las posiciones de una orden no son independientes (DeLong supone que sí). El delta de Cliff mide el "
                       "tamaño: qué tan seguido el AUC de un bloque de XGBoost supera al de un bloque del rival. Combinaciones de "
                       "la tabla de líderes.", className="small text-muted"),
                tabla_pruebas(),
                html.H4("XGBoost en validación: calibración y convergencia", className="mt-4"),
                dbc.Row([dbc.Col(dcc.Graph(figure=calibracion_xgb(), config=CONFIG), lg=6),
                         dbc.Col(dcc.Graph(figure=plots.convergencia(modelo("convergencia_xgboost")), config=CONFIG), lg=6)]),
                html.P("Cómo leer: a la izquierda, los puntos bajo la diagonal en los deciles altos y sobre ella en los bajos dicen "
                       "que XGBoost exagera un poco en los extremos. A la derecha, el AUC de entrenamiento sigue subiendo con cada "
                       "árbol mientras el del semestre externo se aplana: esa distancia es la brecha de sobreajuste.",
                       className="small text-muted"),
                dbc.Row([dbc.Col(dcc.Graph(figure=plots.arrepentimiento(modelo("curvas_optimizador"), "eval", plots.OPTIMIZADOR,
                                                                        "Optimizadores: AUC perdido contra evaluaciones", "evaluación"),
                                           config=CONFIG), lg=6),
                         dbc.Col([dbc.Label("Mapa de un modelo"),
                                  dcc.Dropdown(id="m-mapa", options=[{"label": v, "value": k} for k, v in plots.MODELO.items()],
                                               value="xgboost", clearable=False),
                                  dcc.Graph(id="m-mapa-fig", config=CONFIG)], lg=6)]),
                html.P("Cómo leer: a la izquierda, cuánto AUC le falta a lo mejor encontrado hasta cada evaluación frente al mejor valor conocido (eje logarítmico: más abajo es mejor; la curva que baja antes llega antes). A la derecha, el AUC de cada combinación de balanceo y optimizador del modelo elegido: colores parejos quieren decir que esas decisiones no importan.", className="small text-muted"),
                html.P("El modelo decide: dentro de cada modelo, sus 16 combinaciones de balanceo y optimizador no se separan más de "
                       "0,012 de AUC, y entre modelos la diferencia llega a 0,12. Optuna y el genético se acercan antes al mejor "
                       "valor conocido; con 20 evaluaciones los cuatro quedan a menos de 0,004.", className="small text-muted"),
            ]), label="Las 112 combinaciones", tab_id="t-112", className="pt-3"),
            dbc.Tab(pestana_2026(), label="Modelo óptimo", tab_id="t-2026", className="pt-3"),
            dbc.Tab(pestana_original(), label="Modelo original", tab_id="t-original", className="pt-3"),
            dbc.Tab(pestana_prediccion(), label="Predicción", tab_id="t-prediccion", className="pt-3"),
        ], active_tab=pestana),
        dbc.Alert("La predicción en vivo sirve el modelo original, que le gana al óptimo de las 112 por 0,012 de AUC en 2026, "
                  "y deja comparar con XGBoost. Las entidades llegan como números, sin sus códigos.", color="light",
                  className="mt-3"),
    ])


@callback(Output("m-mapa-fig", "figure"), Input("m-mapa", "value"))
def mapa(m):
    fig = plots.mapa_combinaciones(LID, m, plots.BALANCEO, plots.OPTIMIZADOR)
    fig.update_layout(title=f"{plots.MODELO[m]}: AUC medio por balanceo y optimizador")
    return fig


@callback(Output("m-perdida", "figure"), Input("m-p", "value"))
def perdida(p):
    return plots.log_perdida(p)


@callback(Output("m-coef", "figure"), Input("m-n-coef", "value"))
def coef(n):
    return plots.coeficientes(modelo("coeficientes"), NOMBRES, n)
