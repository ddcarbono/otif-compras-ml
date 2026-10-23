"""Arma las laminas de la sustentacion del Entregable 2 (10 minutos, 9 laminas de 1920 x 1080).

Entrada: resultados/eda/*.csv (las escribe el libro) y, para las cifras provisionales del modelo,
         ../../otif/modelar_otif.txt. Las figuras salen de src/plots.py (las mismas del libro y del tablero).
Salida:  presentacion/laminas/NN_tema.html (una por lamina), presentacion/mazo.html (todas, en orden) y
         presentacion/notas.md (lo que se dice en cada lamina y su tiempo).
Uso:     python presentacion/armar.py
Si una tabla de resultados/ no existe, la lamina sale con un recuadro gris «pendiente» y el script no falla.
"""
import sys, os, re, json, html
import pandas as pd
import plotly.graph_objects as go

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, os.path.join(RAIZ, 'src'))
import plots as P  # noqa: E402

EDA = os.path.join(RAIZ, 'resultados', 'eda')
MODELAR = os.path.normpath(os.path.join(RAIZ, '..', '..', 'otif', 'modelar_otif.txt'))
PLOTLY = 'https://cdn.plot.ly/plotly-2.32.0.min.js'
N = 9


# ---------- formato en espanol ----------
def num(v, d=0):
    s = f'{v:,.{d}f}'
    return s.replace(',', '#').replace('.', ',').replace('#', '.')


def pct(v, d=1):
    return num(100 * v, d) + ' %'


def tabla(nombre):
    f = os.path.join(EDA, nombre + '.csv')
    if not os.path.exists(f):
        return None
    try:
        return pd.read_csv(f)
    except Exception:
        return None


# ---------- cifras provisionales de modelar_otif.txt ----------
def leer_modelar():
    """Devuelve {bloque: {modelo: (auc, pr, f1)}} para 'toda', 'nuevo' y 'conocido', y los deciles de confiabilidad."""
    if not os.path.exists(MODELAR):
        return None
    t = open(MODELAR, encoding='utf-8', errors='replace').read()
    i = t.find('FASE B')
    if i < 0:
        return None
    t = t[i:]
    k1, k2, k3 = t.find('con algún valor nuevo'), t.find('todo conocido'), t.find('confiabilidad de')
    trozos = {'toda': t[:k1 if k1 > 0 else len(t)], 'nuevo': t[k1:k2] if k1 > 0 and k2 > 0 else '',
              'conocido': t[k2:k3 if k3 > 0 else len(t)] if k2 > 0 else ''}
    pat = re.compile(r'^\s*(regla \(tasa previa del proveedor\)|logística ajustada|LightGBM ajustado|LightGBM \+ Platt)'
                     r'\s+([\d.]+) \[[^\]]*\] \| ([\d.]+) \[[^\]]*\] \| ([\d.]+) \[', re.M)
    r = {}
    for b, s in trozos.items():
        r[b] = {m.group(1): tuple(float(m.group(j)) for j in (2, 3, 4)) for m in pat.finditer(s)}
    dec = re.findall(r'decil\s+(\d+): ([\d.]+) \| ([\d.]+) \| ([\d,]+)', t[k3:]) if k3 > 0 else []
    r['deciles'] = [(int(a), float(b), float(c)) for a, b, c, _ in dec]
    return r


MOD = leer_modelar()


# ---------- piezas HTML ----------
def pend(que, h=300):
    return f'<div class="pend" style="height:{h}px"><div><b>pendiente</b><br>{que}</div></div>'


PROV = '<span class="prov">cifra provisional</span>'
_g = [0]


def barras_tasa(t, g, w, h, titulo):
    """objetivo_por sin la anotacion de la tasa global (choca con las etiquetas) y con las categorias inclinadas."""
    fig = P.objetivo_por(t, g, '')
    fig.layout.annotations = ()
    fig.update_layout(yaxis_title='', margin=dict(l=60, r=10, t=85, b=125))
    fig.update_yaxes(range=[0, max(t.tasa) * 1.4])
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=18))
    fig.update_traces(textfont=dict(size=18))
    return fig_html(fig, w, h, titulo + ' (punteada: global)')


def fig_html(fig, w, h, titulo=None):
    """Figura Plotly con fuente grande, como div con script (JSON de fig.to_json())."""
    fig.update_layout(width=w, height=h, autosize=False, font=dict(size=22, color='#333333'),
                      legend=dict(orientation='h', y=-0.2, x=0, font=dict(size=20)))
    if titulo:
        fig.update_layout(title=dict(text=titulo, font=dict(size=26), x=0, xanchor='left'))
    fig.update_xaxes(tickfont=dict(size=20), title_font=dict(size=20))
    fig.update_yaxes(tickfont=dict(size=20), title_font=dict(size=20))
    _g[0] += 1
    i = f'g{_g[0]}'
    d = json.loads(fig.to_json())
    return (f'<div id="{i}" style="width:{w}px;height:{h}px"></div><script>Plotly.newPlot("{i}",'
            f'{json.dumps(d["data"], ensure_ascii=False)},{json.dumps(d["layout"], ensure_ascii=False)},'
            '{staticPlot:true,displayModeBar:false});</script>')


def lamina(n, titulo, cuerpo, fuente, notas, minutos):
    return dict(n=n, titulo=titulo, cuerpo=cuerpo, fuente=fuente, notas=notas, minutos=minutos)


def seccion(l):
    return (f'<section><!-- notas del expositor ({l["minutos"]}): {html.escape(l["notas"]).replace("--", "- -")} -->'
            f'<h1>{l["titulo"]}</h1><div class="cuerpo">{l["cuerpo"]}</div>'
            f'<div class="pie"><span>Fuente: {l["fuente"]}</span><span>{l["n"]} / {N}</span></div></section>')


CSS = """@page { size: 1920px 1080px; margin: 0 }
* { margin: 0; box-sizing: border-box }
:root { --k: 1 }
html, body { overflow-x: hidden; background: #E9E9E4; -webkit-print-color-adjust: exact; print-color-adjust: exact;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #2B2B2B }
section { width: 1920px; height: 1080px; position: relative; overflow: hidden; break-after: page; background: #FFFFFF;
  padding: 56px 90px 0 90px }
section:last-child { break-after: auto }
@media screen { section { zoom: var(--k); margin: 0 auto 24px auto; box-shadow: 0 2px 10px #0003 } }
h1 { font-size: 46px; line-height: 1.15; color: #405731; font-weight: 700; height: 118px; letter-spacing: -0.5px }
.cuerpo { height: 820px; font-size: 30px; line-height: 1.4 }
.pie { position: absolute; left: 90px; right: 90px; bottom: 26px; display: flex; justify-content: space-between;
  font-size: 22px; color: #6B6B6B; border-top: 2px solid #E4E4DE; padding-top: 12px }
.cols { display: flex; gap: 44px } .cols > div { flex: 1 }
ul { padding-left: 1.1em } li { margin: 0 0 12px }
.fm { font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif; font-size: 34px; background: #F6F5EE;
  border-left: 8px solid #B08A2D; padding: 12px 22px; margin: 10px 0 4px 0; line-height: 1.5 }
.fm i { font-style: italic }
.def { font-size: 24px; color: #4A4A4A; margin: 4px 0 16px 4px; line-height: 1.3 }
.pend { background: #E6E6E6; border: 3px dashed #9A9A9A; color: #555; display: flex; align-items: center;
  justify-content: center; text-align: center; font-size: 28px; padding: 24px; border-radius: 6px }
.prov { display: inline-block; background: #B08A2D; color: #fff; font-size: 20px; padding: 2px 12px; border-radius: 14px;
  vertical-align: middle; font-weight: 600 }
table { border-collapse: collapse; font-size: 30px } th, td { padding: 12px 22px; border-bottom: 2px solid #E4E4DE; text-align: right }
th:first-child, td:first-child { text-align: left } th { color: #405731; font-weight: 700 }
.paso { background: #F1F4EE; border: 2px solid #405731; border-radius: 8px; padding: 12px 14px; font-size: 26px; text-align: center; flex: 1 }
.flecha { align-self: center; font-size: 40px; color: #B08A2D }
.caja { background: #F6F5EE; border-radius: 8px; padding: 18px 24px }
.peq { font-size: 24px; color: #555 }
"""


def documento(cuerpo_secciones, titulo, solo_una=False):
    return ('<!doctype html><html lang="es"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{titulo}</title><script src="{PLOTLY}"></script><style>{CSS}</style></head><body>'
            f'{cuerpo_secciones}'
            + ('' if solo_una else '<script>function fit(){document.documentElement.style.setProperty("--k",'
               'Math.min(1,(innerWidth-30)/1920))}fit();addEventListener("resize",fit)</script>')
            + '</body></html>')


# ---------- las 9 laminas ----------
L = []
tam = tabla('tamano')

# 1 titulo
L.append(lamina(1, 'Clasificador binario para la predicción del incumplimiento de órdenes de compra en la agroindustria de la palma de aceite',
    '<div style="margin-top:20px"><p style="font-size:40px"><b>Daniel Carbonó</b> y <b>Luis Pérez</b></p>'
    '<p style="font-size:32px;color:#4A4A4A;margin-top:10px">Maestría en Estadística Aplicada, Universidad del Norte, 2026</p>'
    '<p style="font-size:30px;margin-top:70px">Un grupo agroindustrial colombiano de la palma de aceite: 406.126 posiciones de orden de compra, 2021-2026</p>'
    '<table style="margin-top:60px;font-size:30px"><tr><th>Repositorio</th><td style="text-align:left">https://github.com/ddcarbono/otif-compras-ml</td></tr>'
    '<tr><th>Libro</th><td style="text-align:left">https://ddcarbono.github.io/otif-compras-ml/</td></tr>'
    '<tr><th>Tablero</th><td style="text-align:left">https://otif-compras-ml.onrender.com</td></tr></table></div>',
    'repositorio del proyecto', 'Presentarse y dar las tres direcciones: repositorio, libro y tablero (este último aún sin publicar). Una frase: predecir, el día en que sale la orden, si la posición llegará tarde.',
    '0:30'))

# 2 formulacion
tasa = pct(tam.loc[tam['partición'] == 'total', 'tasa de tarde'].iloc[0]) if tam is not None else None
f2 = ('<div class="cols"><div>'
      '<p>Cada fila es una <b>posición de orden de compra</b>, observada el día en que la orden sale hacia el proveedor.</p>'
      '<div class="fm"><i>x<sub>i</sub></i> ∈ ℝ<sup>136</sup></div><div class="def">predictoras conocidas en la fecha de liberación <i>t<sub>i</sub></i></div>'
      '<div class="fm"><i>y<sub>i</sub></i> ∈ {0, 1}</div><div class="def"><i>y<sub>i</sub></i> = 1 si la primera entrada de mercancía llega más de 1 día hábil después de la fecha prometida</div>'
      '<div class="fm"><i>p</i>(<i>x</i>) = P(<i>y</i> = 1 | <i>x</i>)</div><div class="def">probabilidad de llegar tarde, que se estima</div>'
      '<div class="fm"><i>ŷ</i> = 1 si <i>p̂</i>(<i>x</i>) ≥ <i>τ</i></div><div class="def">el umbral <i>τ</i> se elige en validación (2025), no en prueba</div></div>'
      '<div><div class="caja"><p><b>Tarea:</b> clasificación binaria supervisada.</p>'
      '<p style="margin-top:14px"><b>Por qué binaria:</b> la decisión del comprador es una: dar seguimiento a la orden o no hacerlo.</p>'
      + (f'<p style="margin-top:14px"><b>Magnitud:</b> {tasa} de las posiciones llega tarde.</p>' if tasa else '')
      + '</div><p style="margin-top:30px"><b>Objetivo general:</b> ordenar las posiciones por riesgo de retraso y señalar cuáles atender primero.</p>'
        '<p style="margin-top:14px"><b>Objetivos específicos:</b></p><ul><li>comparar 7 modelos con 4 formas de balancear y 4 optimizadores de hiperparámetros;</li>'
        '<li>medir el costo de decidir con entidades nuevas (proveedor o material sin historia).</li></ul></div></div>')
L.append(lamina(2, 'Estimamos la probabilidad de que cada posición llegue tarde y decidimos con un umbral elegido en validación',
    f2, 'libro, capítulo 02a_datos (definición de la etiqueta y de la población)',
    'x_i son 136 predictoras conocidas el día de liberación; y_i es la etiqueta de retraso (tolerancia 1 día hábil). Se estima p(x) y se decide con un umbral τ. Se pone en validación para no mirar la prueba.',
    '0:45'))

# 3 ETL + EDA
lin, cla, asoc, falt = tabla('tasa_linea'), tabla('tasa_clase'), tabla('asociacion'), tabla('faltantes')
ati, uni = tabla('atipicos_if'), tabla('univariado_numericas')
total = num(tam.loc[tam['partición'] == 'total', 'filas'].iloc[0]) if tam is not None else '406.126'
etl = ('<div style="display:flex;gap:10px;margin-bottom:18px"><div class="paso">SAP<br><span class="peq">compras, 2021-2026</span></div><div class="flecha">›</div>'
       '<div class="paso">490.310 posiciones<br><span class="peq">filtro: compras con plazo</span></div><div class="flecha">›</div>'
       f'<div class="paso">{total} posiciones<br><span class="peq">población</span></div><div class="flecha">›</div>'
       '<div class="paso">etiqueta tarde / a tiempo<br><span class="peq">entrada &gt; 1 día hábil</span></div><div class="flecha">›</div>'
       '<div class="paso">136 predictoras<br><span class="peq">solo lo conocido al liberar</span></div></div>')
c1 = ''
if lin is not None:
    c1 += barras_tasa(lin, 'linea', 520, 312, 'Tasa de tarde por línea')
else:
    c1 += pend('tasa_linea.csv', 320)
c1 += '<div style="height:10px"></div>'
if cla is not None:
    c1 += barras_tasa(cla, 'clase_compras', 520, 312, 'Tasa por clase de compra')
else:
    c1 += pend('tasa_clase.csv', 320)
if asoc is not None:
    top = asoc.sort_values('AUC', ascending=False).head(12)[['nombre', 'AUC']]
    c2 = fig_html(P.barras_horizontales(top, 'AUC', 'nombre', '', 'AUC univariado', '.3f', P.AZUL, 0.5)
                  .update_layout(margin=dict(l=340, r=20, t=70, b=60)).update_yaxes(tickfont=dict(size=18)), 690, 590, 'AUC univariado, 12 mayores')
else:
    c2 = pend('asociacion.csv (AUC univariado)', 650)
if falt is not None:
    fv = falt.sort_values('vacíos (%)', ascending=False).head(8)[['variable', 'vacíos (%)']]
    dic = os.path.join(RAIZ, 'data', 'diccionario.csv')
    if os.path.exists(dic):
        nom = pd.read_csv(dic).set_index('variable')['nombre']
        fv['variable'] = fv['variable'].map(lambda v: nom.get(v, v))
    c3 = fig_html(P.barras_horizontales(fv, 'vacíos (%)', 'variable', '', 'vacíos (%)', '.1f', P.MARRON).update_xaxes(nticks=3)
                  .update_layout(margin=dict(l=330, r=20, t=70, b=60)).update_yaxes(tickfont=dict(size=17)), 470, 590, 'Vacíos estructurales (%)')
else:
    c3 = pend('faltantes.csv', 650)
nota3 = ''
if falt is not None:
    nota3 += f'{int((falt["vacíos (%)"] > 50).sum())} variables con más de 50 % de vacíos; son estructurales (sin solicitud de pedido o sin imputación), no azar. '
if ati is not None:
    a = ati.set_index('grupo')['tasa de tarde']
    nota3 += f'Atípicos por Isolation Forest: {pct(a["atípicas"])} de tarde contra {pct(a["resto"])} en el resto. '
if uni is not None and (uni.variable == 'cantidad').any():
    nota3 += f'La cantidad tiene asimetría {num(uni.loc[uni.variable == "cantidad", "asimetría"].iloc[0])}: para los modelos lineales conviene transformar con logaritmo. '
if asoc is not None:
    nota3 += f'Máximo AUC univariado: {num(asoc.AUC.max(), 3)}.'
L.append(lamina(3, 'Ninguna variable ordena sola por encima de 0,75 de AUC: la señal está repartida entre plazo, historial y entidad',
    etl + f'<div class="cols" style="gap:30px"><div style="flex:0 0 520px">{c1}</div><div style="flex:0 0 690px">{c2}</div><div style="flex:0 0 470px">{c3}</div></div>'
    f'<p class="peq" style="margin-top:4px;font-size:22px;line-height:1.3">{nota3}</p>',
    'libro, capítulos 02a_datos y 02b_eda',
    'ETL: de las tablas de SAP a una fila por posición, con la etiqueta y las 136 predictoras. Ver las diferencias por línea y clase, las mejores variables sueltas (ninguna pasa de 0,75) y que los vacíos son estructurales. Fuga: se descartaron variables registradas después del hecho; la codificación por objetivo va fuera de pliegue y agrupada por orden.',
    '1:30'))

# 4 tiempo
sm, adf, ent = tabla('serie_mensual'), tabla('estacionariedad'), tabla('entidades_nuevas')
if sm is not None:
    sm = sm.copy(); sm['fecha_pedido'] = pd.to_datetime(sm['fecha_pedido'])
    s4 = fig_html(P.serie_tasa(sm, '').update_xaxes(tickformat='%m/%Y').update_yaxes(title_standoff=20), 1100, 640, 'Tasa de tarde y volumen por mes')
else:
    s4 = pend('serie_mensual.csv', 560)
der = ''
if tam is not None:
    r = tam.set_index('partición')['tasa de tarde']
    der = (f'<div class="caja"><p><b>Tasa de tarde por partición</b></p><p>2021-2024: {pct(r["entrenamiento"])}<br>2025: {pct(r["validacion"])}<br>'
           f'2026: {pct(r["prueba"])}</p></div>')
if adf is not None:
    a = adf.set_index('prueba')['p']
    der += (f'<div class="caja" style="margin-top:20px"><p><b>Estacionariedad de la serie</b></p><p>ADF (raíz unitaria): p = {num(a["ADF"], 3)}<br>'
            f'KPSS (estacionaria): p = {num(a["KPSS"], 3)}</p><p class="peq">las dos pruebas apuntan a que la tasa no es estable en el tiempo</p></div>')
if ent is not None:
    prv = ent.set_index('variable').loc['proveedor']
    der += (f'<div class="caja" style="margin-top:20px"><p><b>Entidades nuevas</b></p><p>proveedores nuevos: {num(prv["nuevo en validacion (%)"], 1)} % en 2025 y '
            f'{num(prv["nuevo en prueba (%)"], 1)} % en 2026</p></div>')
if not der:
    der = pend('tamano.csv, estacionariedad.csv y entidades_nuevas.csv', 400)
L.append(lamina(4, 'La tasa de retraso se mueve en el tiempo, por eso se parte cronológicamente y nunca al azar',
    f'<div class="cols"><div style="flex:0 0 1130px">{s4}</div><div>{der}</div></div>'
    '<p style="margin-top:18px">Una partición aleatoria mezclaría el futuro con el pasado y daría un AUC optimista. Además, las posiciones recientes tienen la etiqueta sin madurar (censura al final del periodo).</p>',
    'libro, capítulos 02b_eda y 02c_tiempo_y_fuga',
    'La serie mensual no es estacionaria (ADF y KPSS) y cambia de nivel entre 2021 y 2026. Por eso 2021-2024 entrena, 2025 valida y 2026 prueba. También hay entidades nuevas en validación y prueba que el modelo no vio. La etiqueta de lo más reciente madura tarde: es una limitación.',
    '0:45'))

# 5 teoria
f5 = ('<div class="cols"><div>'
      '<p><b>Mejor benchmark: boosting de árboles</b> <span class="peq">(hoy LightGBM; puede cambiar con los 112)</span></p>'
      '<div class="fm"><i>ℓ</i>(<i>y</i>, <i>F</i>) = −[ <i>y</i> log <i>p</i> + (1 − <i>y</i>) log(1 − <i>p</i>) ]</div>'
      '<div class="def">log-pérdida sobre el puntaje <i>F</i>(<i>x</i>), con <i>p</i> = 1 / (1 + e<sup>−<i>F</i></sup>); es convexa en <i>F</i></div>'
      '<div class="fm"><i>F<sub>m</sub></i> = <i>F</i><sub><i>m</i>−1</sub> + <i>η</i> <i>f<sub>m</sub></i>,  <i>g<sub>i</sub></i> = <i>p<sub>i</sub></i> − <i>y<sub>i</sub></i>,  <i>h<sub>i</sub></i> = <i>p<sub>i</sub></i>(1 − <i>p<sub>i</sub></i>)</div>'
      '<div class="def">descenso por gradiente en el espacio de funciones: cada árbol <i>f<sub>m</sub></i> ajusta gradiente <i>g</i> y curvatura <i>h</i>; <i>η</i> = tasa de aprendizaje</div>'
      '<div class="fm" style="font-size:30px">Obj = Σ<sub><i>i</i></sub> <i>ℓ</i><sub><i>i</i></sub> + Σ<sub><i>m</i></sub> [ <i>γ</i> <i>T<sub>m</sub></i> + ½ <i>λ</i> ‖<i>w<sub>m</sub></i>‖² + <i>α</i> ‖<i>w<sub>m</sub></i>‖<sub>1</sub> ]</div>'
      '<div class="def"><i>T</i> = hojas, <i>w</i> = valores de hoja; peso óptimo de una hoja (sin L1): <i>w</i>* = − <i>G</i> / (<i>H</i> + <i>λ</i>), con <i>G</i> = Σ<i>g</i> y <i>H</i> = Σ<i>h</i> de la hoja</div>'
      '</div><div>'
      '<div class="caja"><p><b>Regularización:</b> tasa de aprendizaje <i>η</i>, número de hojas, mínimo de filas por hoja, <i>λ</i> (L2), <i>α</i> (L1), submuestreo de filas y columnas, parada temprana.</p>'
      '<p style="margin-top:12px"><b>Complejidad:</b> árbol por histograma (columnas en bins): construir un árbol cuesta O(<i>n</i>·<i>p</i>) más la búsqueda de cortes O(<i>bins</i>·<i>p</i>); predecir cuesta O(<i>M</i>·profundidad).</p></div>'
      '<p style="margin-top:22px"><b>Contraste: regresión logística</b></p>'
      '<div class="fm" style="font-size:30px">mín<sub><i>β</i></sub> Σ<sub><i>i</i></sub> <i>ℓ</i>(<i>y<sub>i</sub></i>, <i>x<sub>i</sub></i>ᵀ<i>β</i>) + (1 / 2<i>C</i>) ‖<i>β</i>‖²</div>'
      '<div class="def">pérdida convexa: un solo óptimo; log-odds lineal en <i>x</i>; <i>C</i> controla la penalización L2 (o L1). No captura interacciones ni no linealidades sin construirlas.</div>'
      '<p class="peq">El boosting corrige lo que la recta no ve (interacciones, umbrales) y paga con más hiperparámetros y riesgo de sobreajuste.</p>'
      '</div></div>')
L.append(lamina(5, 'El boosting minimiza la log-pérdida paso a paso con árboles; la logística hace lo mismo con una sola recta y un óptimo',
    f5, 'libro, capítulo de modelos (en preparación); Friedman (2001), Chen y Guestrin (2016)',
    'Dos minutos. Boosting: cada árbol es un paso de descenso por gradiente sobre la log-pérdida en el espacio de funciones, usando gradiente y hessiano; el peso de hoja sale cerrado. Regularización: η, hojas, λ, α, submuestreo. Complejidad: histograma. Contraste: logística, convexa, un óptimo, L2/L1. El nombre exacto del mejor benchmark depende de los 112.',
    '2:00'))

# 6 diseno
if tam is not None:
    tt = tam.set_index('partición')

    def fila(nombre, k, etiqueta):
        r = tt.loc[k]
        return f'<tr><td>{nombre}</td><td>{etiqueta}</td><td>{num(r["filas"])}</td><td>{num(r["órdenes"])}</td><td>{pct(r["tasa de tarde"])}</td></tr>'
    tabp = ('<table><tr><th>partición</th><th>uso</th><th>posiciones</th><th>órdenes</th><th>tarde</th></tr>'
            + fila('2021-2024', 'entrenamiento', 'entrenar') + fila('2025', 'validacion', 'validar, umbral τ')
            + fila('2026', 'prueba', 'probar (una vez)') + '</table>')
else:
    tabp = pend('tamano.csv', 260)
f6 = (f'<div class="cols"><div style="flex:0 0 960px">{tabp}'
      '<div class="caja" style="margin-top:26px"><p><b>Validación anidada</b></p><ul><li>bucle externo: 8 bloques semestrales, cada uno se predice con lo anterior;</li>'
      '<li>bucle interno: el último semestre de cada ventana elige hiperparámetros;</li><li>mismo presupuesto de optimización para todos; una sola semilla.</li></ul></div></div>'
      '<div><div class="caja"><p><b>112 combinaciones</b></p><div class="fm" style="font-size:36px">7 modelos × 4 balanceos × 4 optimizadores = 112</div></div>'
      '<div class="caja" style="margin-top:22px"><p><b>Optimizadores</b></p><ul><li>Optuna: TPE con mejora esperada, aprende de los ensayos previos;</li>'
      '<li>genética: selección, cruce, mutación y elitismo.</li></ul></div></div></div>')
L.append(lamina(6, 'Cada una de las 112 combinaciones se evalúa con el mismo presupuesto, hacia adelante en el tiempo',
    f6, 'libro, capítulos 02a_datos y 02c_tiempo_y_fuga; plan de entregables',
    'La prueba 2026 se usa una sola vez y no elige nada. La validación anidada evita que el optimizador vea la fila que se evalúa. Mismo presupuesto para comparar optimizadores de forma justa.',
    '1:30'))

# 7 resultados (112 pendientes + provisional)
if MOD and MOD['toda']:
    t = MOD['toda']
    nombres = [('regla (tasa previa del proveedor)', 'regla del proveedor'), ('logística ajustada', 'logística'), ('LightGBM + Platt', 'LightGBM + Platt')]
    filas = ''.join(f'<tr><td>{n2}</td><td>{num(t[n1][0], 3)}</td><td>{num(t[n1][2], 3)}</td></tr>' for n1, n2 in nombres if n1 in t)
    dif = t['LightGBM + Platt'][0] - t['logística ajustada'][0] if 'LightGBM + Platt' in t and 'logística ajustada' in t else None
    tit7 = (f'LightGBM supera a la logística en {num(dif, 2)} de AUC con las cifras de hoy (provisionales)' if dif is not None else 'Resultados de las 112 combinaciones')
    izq = (f'<p><b>Prueba 2026</b> {PROV}</p><table style="font-size:36px;margin-top:10px"><tr><th>modelo</th><th>AUC</th><th>F1 (τ de 2025)</th></tr>{filas}</table>'
           '<p class="peq" style="margin-top:12px">Un solo ajuste por modelo; cambia con los 112 y su validación anidada.</p>')
else:
    izq = pend('modelar_otif.txt (cifras provisionales)', 300)
    tit7 = 'Resultados de las 112 combinaciones'
der7 = (pend('curvas de optimización: AUC contra evaluaciones y contra tiempo, por optimizador (resultados de los 112)', 330)
        + '<div style="height:20px"></div>' + pend('tabla de líderes y diagrama de diferencia crítica (DeLong, Holm, delta de Cliff)', 330))
L.append(lamina(7, tit7,
    f'<div class="cols"><div style="flex:0 0 800px">{izq}</div><div>{der7}</div></div>',
    'libro, capítulo de resultados (pendiente); modelar_otif.txt para las cifras provisionales',
    'Hoy solo hay el ajuste de LightGBM, la logística y la regla del proveedor. Se muestran como provisionales. Los 112 darán las curvas de optimización, la tabla de líderes y el diagrama de diferencia crítica.',
    '1:15'))

# 8 errores, sobreajuste, calibracion
if MOD and MOD.get('nuevo') and MOD.get('conocido') and 'LightGBM ajustado' in MOD['nuevo'] and 'LightGBM ajustado' in MOD['conocido']:
    n, c = MOD['nuevo']['LightGBM ajustado'][0], MOD['conocido']['LightGBM ajustado'][0]
    tit8 = f'Con entidades nuevas el AUC baja de {num(c, 3)} a {num(n, 3)}: el costo del arranque en frío (provisional)'
    cuadro = (f'<div class="caja"><p><b>AUC de LightGBM en 2026</b> {PROV}</p><table><tr><td>todo conocido</td><td>{num(c, 3)}</td></tr>'
              f'<tr><td>con algún valor nuevo</td><td>{num(n, 3)}</td></tr></table></div>')
else:
    tit8 = 'Dónde se equivoca el mejor modelo y cuánto sobreajusta'
    cuadro = pend('AUC con entidades nuevas frente a conocidas', 200)
if MOD and MOD.get('deciles'):
    d = pd.DataFrame(MOD['deciles'], columns=['decil', 'predicha', 'observada'])
    fg = go.Figure([go.Scatter(x=[0, 0.6], y=[0, 0.6], mode='lines', line=dict(color=P.GRIS, dash='dot'), name='ideal'),
                    go.Scatter(x=d.predicha, y=d.observada, mode='lines+markers', line=dict(color=P.AZUL), marker=dict(size=10), name='LightGBM + Platt')])
    fg.update_layout(xaxis_title='probabilidad predicha media', yaxis_title='tasa observada', xaxis_tickformat='.0%', yaxis_tickformat='.0%')
    fg.update_xaxes(title_standoff=15)
    fg.update_yaxes(title_standoff=20)
    fg.update_layout(margin=dict(l=100, r=20, t=70, b=120))
    calib = fig_html(fg, 760, 620, 'Calibración por decil (provisional)')
else:
    calib = pend('calibración por decil', 480)
L.append(lamina(8, tit8,
    f'<div class="cols"><div style="flex:0 0 780px">{calib}</div><div>{cuadro}'
    '<div style="height:20px"></div>' + pend('matriz de confusión por línea; brecha entre entrenamiento y validación por iteración (sobreajuste); SHAP del mejor modelo', 380) + '</div></div>',
    'libro, capítulo de resultados (pendiente); modelar_otif.txt para las cifras provisionales',
    'Calibración: lo predicho coincide con lo observado por decil. Con entidades nuevas el AUC baja. Pendiente: errores por línea, sobreajuste (brecha entrenamiento-validación) y SHAP para el mejor de los 112.',
    '0:45'))

# 9 conclusiones
if MOD and MOD['toda'] and 'LightGBM + Platt' in MOD['toda'] and 'regla (tasa previa del proveedor)' in MOD['toda']:
    cifra = (f'{num(MOD["toda"]["LightGBM + Platt"][0], 3)} de AUC y {num(MOD["toda"]["LightGBM + Platt"][2], 3)} de F1 en 2026 '
             f'(regla del proveedor: {num(MOD["toda"]["regla (tasa previa del proveedor)"][0], 3)} de AUC) {PROV}')
else:
    cifra = 'cifras de prueba pendientes'
nuevos = (num(ent.set_index('variable').loc['proveedor', 'nuevo en prueba (%)'], 1) + ' % de proveedores nuevos en 2026') if ent is not None else 'proveedores y materiales'
f9 = ('<div class="cols"><div><p><b>Conclusiones</b></p><ul>'
      f'<li>El incumplimiento se puede anticipar el día de la liberación: {cifra}.</li>'
      '<li>El plazo, el historial del proveedor y las entidades concentran la señal; ninguna variable sola pasa de 0,75 de AUC.</li>'
      '<li>La partición cronológica y la validación anidada son lo que hace creíble el número.</li>'
      '<li>Veredicto sobre los 112 modelos: pendiente.</li></ul></div>'
      '<div><p><b>Limitaciones</b></p><ul>'
      '<li>Censura al final del periodo: las posiciones recientes tienen la etiqueta sin madurar.</li>'
      '<li>Los maestros (proveedor, material) están en foto actual, no con vigencia histórica.</li>'
      '<li>Una sola empresa: la generalización a otros grupos no está probada.</li>'
      f'<li>Entidades nuevas ({nuevos}): el modelo no tiene historia que usar.</li>'
      '</ul></div></div>')
L.append(lamina(9, 'El retraso se anticipa mejor que con la regla del proveedor, con límites claros de periodo, maestros y entidades nuevas',
    f9, 'libro, capítulos 02a_datos, 02b_eda y 02c_tiempo_y_fuga; modelar_otif.txt (cifras provisionales)',
    'Cerrar con la respuesta y sus cifras (provisionales), y con las cuatro limitaciones: censura, maestros en foto actual, una sola empresa y entidades nuevas. Los dos autores responden preguntas.',
    '1:00'))

# ---------- escritura ----------
nombres_f = ['01_titulo', '02_formulacion', '03_etl_eda', '04_tiempo', '05_teoria', '06_diseno', '07_resultados', '08_errores', '09_conclusiones']
os.makedirs(os.path.join(AQUI, 'laminas'), exist_ok=True)
secs = []
for l, nf in zip(L, nombres_f):
    s = seccion(l)
    secs.append(s)
    open(os.path.join(AQUI, 'laminas', nf + '.html'), 'w', encoding='utf-8').write(documento(s, f'Lámina {l["n"]}', True))
open(os.path.join(AQUI, 'mazo.html'), 'w', encoding='utf-8').write(documento('\n'.join(secs), 'Sustentación, Entregable 2'))
with open(os.path.join(AQUI, 'notas.md'), 'w', encoding='utf-8') as f:
    f.write('# Notas del expositor: Entregable 2 (10 minutos)\n\n')
    for l in L:
        f.write(f'## {l["n"]}. {re.sub("<[^>]+>", "", l["titulo"])} ({l["minutos"]})\n\n{l["notas"]}\n\n')
pend_n = sum(s.count('class="pend"') for s in secs)
print(f'{len(L)} láminas escritas; recuadros pendientes: {pend_n}; cifras del modelo: {"sí" if MOD else "no"}')
