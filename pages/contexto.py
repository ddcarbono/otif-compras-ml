"""Sección 1: el problema, la pregunta, los datos, la metodología y los límites. Narrativa estática más el diccionario
filtrable."""
import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

from src.data import diccionario, es, pct, resumen

dash.register_page(__name__, path="/", name="Contexto", order=0, title="Contexto · órdenes de compra")

R = resumen()
VERSION, ACTUALIZADO = "0.3", "2026-10-23"


def cifra(valor, texto, nota=None):
    return dbc.Card(dbc.CardBody([html.Div(valor, className="cifra"), html.Div(texto, className="cifra-texto"),
                                  html.Small(nota, className="text-muted") if nota else None]), className="h-100")


def paso(titulo, texto):
    return dbc.Col(dbc.Card(dbc.CardBody([html.H6(titulo, className="mb-1"), html.Small(texto)]),
                            className="h-100 paso"), xs=6, md=3, lg=True, className="mb-2")


DIC = diccionario()
tabla_dic = dash_table.DataTable(
    data=DIC[["variable", "nombre", "grupo", "tipo", "unidad", "significado"]].to_dict("records"),
    columns=[{"name": c, "id": c} for c in ["variable", "nombre", "grupo", "tipo", "unidad", "significado"]],
    filter_action="native", sort_action="native", page_size=12,
    style_table={"overflowX": "auto"}, style_cell={"textAlign": "left", "fontSize": 13, "whiteSpace": "normal",
                                                   "height": "auto", "maxWidth": 420},
    style_header={"fontWeight": "bold"})

layout = html.Div([
    html.Section([
        dbc.Row([
            dbc.Col([
                html.H1("¿Va a llegar tarde esta orden de compra?", className="titulo"),
                html.P("Un clasificador binario que estima, el día en que una orden de compra sale hacia el proveedor, "
                       "la probabilidad de que cada posición llegue más de un día hábil después de la fecha prometida. "
                       "Datos de un grupo agroindustrial colombiano de la palma de aceite.", className="lead")],
                md=9, className="order-2 order-md-1"),
            # portada de la app Compras de Qlik Sense (ejemplos/2026-10-23_portada_compras en el repo de estudio)
            dbc.Col(html.Img(src=dash.get_asset_url("portada_compras.jpg"), className="portada", alt=""),
                    md=3, className="order-1 order-md-2 text-center text-md-end")], className="align-items-center"),
        dbc.Row([
            dbc.Col(cifra(pct(R["tasa_total"], 0), "de las posiciones llega tarde",
                          f"{es(R['filas'])} posiciones de {es(R['ordenes'])} órdenes, 2021-2026"), md=4, className="mb-3"),
            dbc.Col(cifra("23 %", "de entregas tardías en las compras de otra empresa con el mismo ERP, que bajó a 10 % "
                                  "con seguimiento a proveedores",
                          "Widyaswara y Bustommy (2025), Journal of Innovative and Creativity 5(2), "
                          "doi:10.31004/joecy.v5i2.3441"), md=4, className="mb-3"),
            dbc.Col(cifra(es(R["predictoras"]), "variables conocidas el día de la liberación",
                          "plazo, proveedor, material, aprobación, historial, pagos, stock y tasa de cambio"), md=4,
                    className="mb-3"),
        ]),
    ]),
    html.H3("Por qué importa"),
    html.P("Un repuesto que llega tarde a una extractora o un insumo que llega tarde a una plantación detiene un "
           "proceso, obliga a comprar de urgencia o a mantener inventario de más. Saber el día de la liberación qué "
           "posiciones tienen más riesgo permite hacerles seguimiento a esas, y no a todas."),
    html.H3("Pregunta de investigación y objetivos"),
    html.P(html.Strong("¿Se puede predecir, el día en que la orden se libera hacia el proveedor, si cada posición va a "
                       "llegar tarde, con lo que se sabe ese día?")),
    html.Ul([html.Li("Estimar la probabilidad de llegar tarde de cada posición, calibrada para leerse como riesgo."),
             html.Li("Identificar qué variables conocidas al liberar explican el retraso.")]),
    html.H3("Los datos"),
    dbc.Row([
        dbc.Col(html.Ul([
            html.Li(["Fuente: tablas de compras del ERP SAP de la empresa, en modo de solo lectura; uso académico "
                     "autorizado. Corte: 21 de octubre de 2026."]),
            html.Li(f"Periodo: órdenes creadas del {R['periodo'][0]} al {R['periodo'][1]}."),
            html.Li(f"{es(R['filas'])} posiciones (unidad de observación), {es(R['ordenes'])} órdenes, "
                    f"{es(R['proveedores'])} proveedores y {es(R['materiales'])} materiales."),
            html.Li(f"{R['predictoras']} predictoras: 83 numéricas, 42 categóricas y 11 entidades de muchos valores."),
        ]), md=6),
        dbc.Col(html.Ul([
            html.Li(["Objetivo: ", html.Code("y_tarde"), " = 1 si la primera entrada de mercancía llega más de un día "
                     "hábil (con festivos de Colombia) después de la fecha prometida vigente al liberar."]),
            html.Li("Tarea: clasificación binaria."),
            html.Li("Partición cronológica por año de la orden: entrenamiento 2021-2024, validación 2025, prueba "
                    "2026. Ninguna orden queda partida."),
        ]), md=6),
    ]),
    html.H5("Diccionario de variables", className="mt-2"),
    html.P("Escriba en la fila de filtro para buscar; por ejemplo, «proveedor» en la columna nombre.",
           className="text-muted small"),
    tabla_dic,
    html.H3("Metodología", className="mt-4"),
    dbc.Row([paso("1. ETL", "Extracción del ERP, población con plazo, etiqueta en días hábiles, anonimización"),
             paso("2. EDA", "Objetivo, uni, bi y multivariado, tiempo, fuga; solo con 2021-2024"),
             paso("3. Modelos", "7 modelos × 4 balanceos × 4 optimizadores, con preprocesamiento en el pipeline"),
             paso("4. Validación anidada", "Bloques semestrales de ventana creciente; la prueba 2026 se usa una vez"),
             paso("5. Comparación", "Friedman, Nemenyi, DeLong y delta de Cliff"),
             paso("6. Modelo original", "Propuesta propia, con las mismas particiones y pruebas")],
            className="g-2"),
    html.H3("Limitaciones, ética y sesgos", className="mt-4"),
    html.Ul([
        html.Li("Censura al final del periodo: las posiciones sin ninguna entrada no entran, y cerca del corte una "
                "posición muy atrasada todavía no tiene entrada."),
        html.Li("Los maestros de proveedor y material son la foto actual del ERP, no la del día de la orden; los que se "
                "ajustan después de ver retrasos se retiraron."),
        html.Li("Una sola empresa: los resultados describen su proceso y sus proveedores."),
        html.Li("Personas con seudónimo y terceros con código. El modelo no evalúa compradores: su historial es contexto "
                "del proceso. El conjunto completo no se publica; este tablero solo carga una muestra de "
                f"{es(R['muestra'])} posiciones."),
    ]),
    html.Hr(),
    html.P([html.Strong("Autores: "), "Daniel Carbonó y Luis Pérez. Maestría en Estadística Aplicada, Universidad del "
            "Norte. ", html.Strong("Versión: "), VERSION, ". ", html.Strong("Actualizado: "), ACTUALIZADO, ". ",
            html.A("Repositorio", href="https://github.com/ddcarbono/otif-compras-ml", target="_blank"), " · ",
            html.A("Informe (Jupyter Book)", href="https://ddcarbono.github.io/otif-compras-ml/", target="_blank")],
           className="small"),
])
