# -*- coding: utf-8 -*-
"""Sustentacion del proyecto OTIF (Entregable 2) con la plantilla Uninorte de las sustentaciones de GLM.

Ordenada por los siete criterios de la rubrica (entregables/diapositivas.md): problema y formulacion, EDA y fuga,
fundamentos teoricos del mejor benchmark (XGBoost), diseno experimental, resultados y errores, conclusiones, preguntas.
Las cifras se leen de resultados/ y data/modelos/ al construir; las figuras y formulas son PNG de figuras.py.
Lo que depende de la evaluacion final en 2026 (la 2x2) sale en una caja gris "pendiente".

Uso (Python base, que tiene python-pptx):
    python presentacion/construir.py [salida.pptx]      # por defecto presentacion/sustentacion_otif.pptx
Revision:
    python presentacion/revisar_pptx.py presentacion/sustentacion_otif.pptx
    python presentacion/exportar.py presentacion/sustentacion_otif.pptx presentacion/render 1600
"""
import json
import os
import sys

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('IMG_DIR', os.path.join(AQUI, 'img'))
sys.path.insert(0, AQUI)
from plantilla_uninorte import *  # noqa: E402,F401,F403

RAIZ = os.path.dirname(AQUI)
R, MOD = os.path.join(RAIZ, 'resultados'), os.path.join(RAIZ, 'data', 'modelos')
SAL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, 'sustentacion_otif.pptx')
PIE_OTIF = 'Daniel Carbonó y Luis Pérez   •   Maestría en Estadística Aplicada   •   Universidad del Norte'
URL = {'repo': 'github.com/ddcarbono/otif-compras-ml', 'libro': 'ddcarbono.github.io/otif-compras-ml',
       'tablero': 'otif-compras-ml.onrender.com'}


def leer(*ruta):
    return pd.read_csv(os.path.join(*ruta))


def n(v, d=0):
    return f'{v:,.{d}f}'.replace(',', '#').replace('.', ',').replace('#', '.')


def pc(v, d=1):
    return n(100 * v, d) + ' %'


# ---------------------------------------------------------------- cifras
tam = leer(R, 'eda', 'tamano.csv').set_index('partición')
aso = leer(R, 'eda', 'asociacion.csv')
falt = leer(R, 'eda', 'faltantes.csv')
ati = leer(R, 'eda', 'atipicos_if.csv').set_index('grupo')['tasa de tarde']
est = leer(R, 'eda', 'estacionariedad.csv').set_index('prueba')['p']
nue = leer(R, 'eda', 'entidades_nuevas.csv').set_index('variable')
adv = leer(R, 'similitud', 'auc.csv').AUC.tolist()
lid = leer(MOD, 'lideres.csv')
mejor = lid.sort_values('AUC', ascending=False).groupby('modelo').head(1)
top = mejor.iloc[0]
fri = leer(MOD, 'friedman.csv').iloc[0]
wil = leer(MOD, 'wilcoxon.csv')
cur = leer(MOD, 'curvas_optimizador.csv').pivot(index='eval', columns='optimizador', values='mean')
met = leer(MOD, 'metricas.csv').set_index(['modelo', 'partición'])
lb = met.loc[('Regresión logística', 'prueba 2026')]
rp = met.loc[('Regla del proveedor', 'prueba 2026')]
ca = leer(MOD, 'curva_aprendizaje.csv').iloc[-1]
pb = leer(R, 'experimentos', 'por_bloque.csv')
xb = pb[(pb.modelo == top.modelo) & (pb.balanceo == top.balanceo) & (pb.optimizador == top.optimizador)]
cx = leer(R, 'complejidad.csv')
cxx = cx[cx.modelo == 'xgboost'].sort_values('filas')
import numpy as np  # noqa: E402
expo = np.polyfit(np.log(cxx.filas), np.log(cxx.segundos_ajuste), 1)[0]
horas = pb.segundos_total.sum() / 3600
NOMBRE = {'xgboost': 'XGBoost', 'random_forest': 'Random Forest', 'logistica': 'Logística (SGD)',
          'svm': 'SVM lineal (SGD)', 'arbol': 'Árbol', 'knn': 'KNN (PCA 30)', 'naive_bayes': 'Naive Bayes'}
BAL = {'ninguno': 'sin balanceo', 'smote': 'SMOTE', 'adasyn': 'ADASYN', 'pesos': 'pesos'}
OPT = {'grid': 'Grid', 'random': 'Random', 'optuna': 'Optuna', 'deap': 'DEAP'}

N_TOTAL = 13
NOTAS = []


def pie(s, k, fuente):
    filete(s, 1.3, 13.45, 22.9, LINEA, alto=0.04)
    texto(s, 1.3, 13.6, 16.5, 0.5, f'Fuente: {fuente}', tam=8.5, color=GRIS_CL)
    texto(s, 17.8, 13.6, 6.4, 0.5, f'{k} / {N_TOTAL}', tam=8.5, color=GRIS_CL, alineacion=PP_ALIGN.RIGHT)


def notas(s, minutos, expone, criterio, txt):
    s.notes_slide.notes_text_frame.text = f'[{minutos} · {criterio} · expone {expone}] {txt}'
    NOTAS.append((minutos, expone, criterio, txt))


def pendiente(s, x, y, w, h, txt):
    caja(s, x, y, w, h, RGBColor(0xEC, 0xEC, 0xEC))
    texto(s, x + 0.4, y + 0.25, w - 0.8, h - 0.4, [('Pendiente: evaluación final 2026', {'negrita': True, 'color': GRIS}),
                                                    (txt, {'tam': 10.5, 'color': GRIS})], tam=11.5, espacio=3)


def lista(s, x, y, w, items, tam=11.5, paso=1.15, alto=1.1):
    for t in items:
        vineta(s, x, y, w - 0.6, t, tam=tam, alto=alto, color=TINTA, negrita=False)
        y += paso
    return y


# ============================================================== 1 PORTADA
s = nueva()
caja(s, 0, 0, 8.8, 14.29, VERDE)
logo(s, 1.3, 1.4, 3.9, 3.0)
texto(s, 1.3, 5.2, 6.8, 0.8, 'UNIVERSIDAD DEL NORTE', tam=12.5, color=BLANCO, negrita=True)
texto(s, 1.3, 6.0, 6.8, 1.6, ['Maestría en Estadística Aplicada', 'Machine Learning'], tam=11, color=PALIDO, espacio=4)
filete(s, 1.3, 10.6, 1.8)
texto(s, 1.3, 11.1, 6.8, 1.6, ['Daniel Carbonó Domínguez', 'Luis Pérez'], tam=11.5, color=BLANCO, negrita=True, espacio=4)
texto(s, 1.3, 12.6, 6.8, 0.8, 'Octubre de 2026', tam=10.5, color=PALIDO)
texto(s, 10.0, 2.6, 13.5, 0.8, 'ENTREGABLE 2  ·  CLASIFICACIÓN BINARIA SUPERVISADA', tam=13, color=VERDE, negrita=True)
texto(s, 10.0, 3.7, 14.4, 4.2, ['Predicción del incumplimiento de órdenes de compra en la agroindustria de la palma '
                                'de aceite'], tam=24, color=TINTA, negrita=True)
filete(s, 10.0, 8.1, 2.3)
texto(s, 10.0, 8.6, 14.0, 1.2, f'{n(tam.loc["total", "filas"])} posiciones de {n(tam.loc["total", "órdenes"])} '
      'órdenes de compra, 2021-2026', tam=15, color=GRIS)
texto(s, 10.0, 10.4, 14.0, 2.4, [f'Repositorio   {URL["repo"]}', f'Libro            {URL["libro"]}',
                                 f'Tablero         {URL["tablero"]}'], tam=12, color=GRIS_CL, espacio=4)
notas(s, '0:30', 'A', 'criterio 1', 'Presentarse los dos. Una frase: el día en que la orden se libera, predecir si cada '
      'posición llegará tarde. Las tres direcciones quedan en pantalla; el tablero se abre al final si hay tiempo.')

# ============================================================== 2 PROBLEMA Y FORMULACION
s = nueva()
cabecera(s, 'Uno de cada tres pedidos llega tarde, y se sabe cuando ya pasó', 'CRITERIO 1  ·  PROBLEMA Y FORMULACIÓN')
cifra(s, 1.3, 3.4, 5.4, 2.7, pc(tam.loc['total', 'tasa de tarde']), 'de las posiciones llega tarde')
cifra(s, 1.3, 6.4, 5.4, 2.7, '1 día', 'hábil de tolerancia sobre la fecha prometida', CREMA)
tarjeta(s, 1.3, 9.4, 5.4, 3.7, 'Objetivo', 'Ordenar las posiciones por riesgo el día de la liberación y señalar cuáles '
        'seguir primero.', NIEBLA, tam_c=10.5)
texto(s, 7.6, 3.3, 9.0, 0.6, 'Unidad: la posición i de una orden', tam=12, color=VERDE, negrita=True)
figura(s, 'f_x.png', 7.7, 4.1, h=0.75)
texto(s, 7.6, 4.95, 9.0, 1.8, 'x: 136 predictoras conocidas el día t en que la orden sale al proveedor. '
      'y = 1 si la primera entrada llega más de 1 día hábil después de la fecha prometida.', tam=10.5, color=GRIS)
figura(s, 'f_p.png', 7.7, 6.9, h=0.75)
texto(s, 7.6, 7.8, 9.0, 1.4, 'Se estima la probabilidad p(x) y se decide con un umbral τ, elegido en la '
      'validación 2025 maximizando F1; la prueba 2026 no elige nada.', tam=10.5, color=GRIS)
texto(s, 7.6, 9.5, 9.0, 0.6, 'Por qué clasificación binaria', tam=12, color=VERDE, negrita=True)
texto(s, 7.6, 10.2, 9.0, 2.9, 'La decisión del comprador es una sola: hacerle seguimiento a la orden o no. Los días '
      'de retraso tienen colas largas y se usan como contexto, no como objetivo.', tam=10.5)
tarjeta(s, 17.4, 3.4, 6.8, 9.7, 'Objetivos específicos',
        ['1. Comparar 7 modelos × 4 formas de balancear × 4 optimizadores con el mismo presupuesto.',
         '2. Medir cuánto cuesta decidir con entidades nuevas: proveedor o material sin historia.',
         '3. Dejar el modelo servido en un tablero con la predicción en vivo.'], VERDECL, tam_c=11)
pie(s, 2, 'libro, capítulos 1 y 2a (definición de la etiqueta y de la población)')
notas(s, '0:50', 'A', 'criterio 1', 'Problema, unidad, x, y, p(x) y el umbral. Insistir: el umbral sale de 2025; la '
      'prueba 2026 se mira una vez. Por qué binaria: la decisión es binaria.')

# ============================================================== 3 DATOS
s = nueva()
cabecera(s, f'De SAP a {n(tam.loc["total", "filas"])} posiciones con una etiqueta y 136 predictoras',
         'CRITERIO 2  ·  FUENTE, ESTRUCTURA Y PARTICIÓN')
pasos = [('SAP', 'compras 2021-2026'), ('490.310', 'posiciones con plazo'), (n(tam.loc['total', 'filas']), 'población'),
         ('etiqueta', '> 1 día hábil tarde'), ('136', 'predictoras al liberar')]
for k, (a, b) in enumerate(pasos):
    x = 1.3 + k * 4.65
    caja(s, x, 3.3, 4.2, 1.9, VERDECL if k % 2 == 0 else CREMA)
    texto(s, x, 3.5, 4.2, 0.8, a, tam=17, color=VERDE, negrita=True, alineacion=PP_ALIGN.CENTER)
    texto(s, x, 4.35, 4.2, 0.7, b, tam=10, color=GRIS, alineacion=PP_ALIGN.CENTER)
filas_t = [['partición', 'uso', 'posiciones', 'órdenes', 'tarde']]
for k, nom, uso in [('entrenamiento', '2021-2024', 'entrenar'), ('validacion', '2025', 'elegir umbral'),
                    ('prueba', '2026', 'probar una vez')]:
    r = tam.loc[k]
    filas_t.append([nom, uso, n(r['filas']), n(r['órdenes']), pc(r['tasa de tarde'])])
tabla(s, 1.3, 5.8, 13.0, 3.6, filas_t, anchos=[2.2, 2.8, 2.4, 2.2, 1.7], tam=11)
tarjeta(s, 15.0, 5.8, 9.2, 7.2, 'Las 136 predictoras',
        ['83 numéricas: plazo, valor, liberación, historial del proveedor, stock, tasa de cambio.',
         '42 categóricas de pocos valores: línea, clase, moneda, tipo de material.',
         '11 entidades: proveedor, material, comprador, liberador, centro de costo y otras.',
         'Solo lo que se conoce el día de la liberación.'], VERDECL, tam_c=10.5)
texto(s, 1.3, 9.8, 13.0, 3.4, [('Las órdenes son la unidad independiente: '
                                f'{n(tam.loc["total", "filas"] / tam.loc["total", "órdenes"], 1)} posiciones por orden. '
                                'Ninguna orden cruza particiones y toda validación cruzada agrupa por orden.', {}),
                               ('La partición es por la fecha de la orden, no al azar (lámina 5).', {'color': GRIS})],
      tam=11, espacio=6)
pie(s, 3, 'libro, capítulo 2a (datos); diccionario de variables en data/diccionario.csv')
notas(s, '0:40', 'A', 'criterio 2', 'Fuente y estructura. La unidad independiente es la orden: por eso los pliegues '
      'se agrupan por orden. La partición es cronológica.')

# ============================================================== 4 EDA
s = nueva()
cabecera(s, 'Ninguna variable sola pasa de 0,75 de AUC', 'CRITERIO 2  ·  DESCRIPTIVA, '
         'FALTANTES, ATÍPICOS Y RELACIONES')
figura(s, 'auc_univariado.png', 1.2, 3.2, w=12.2)
y = 3.3
for tit, cuerpo, rell in [
        ('Faltantes estructurales', f'{int((falt["vacíos (%)"] > 50).sum())} variables con más de 50 % de vacíos, y no '
         'son azar: faltan porque la orden no tiene solicitud o no tiene imputación. Se marcan con indicadores tiene_* y '
         'se imputa la mediana.', VERDECL),
        ('Atípicos (Isolation Forest)', f'{pc(ati["atípicas"])} de tarde en las atípicas contra {pc(ati["resto"])} en el '
         'resto: son señal, no ruido, y no se borran.', CREMA),
        ('Colas largas', 'Valor, cantidad y plazos son muy asimétricos: signo(x)·log(1 + |x|) antes de los modelos '
         'lineales; a los árboles no les cambia nada.', NIEBLA)]:
    tarjeta(s, 14.3, y, 9.9, 3.1, tit, cuerpo, rell, tam_t=12, tam_c=10)
    y += 3.3
texto(s, 1.3, 11.0, 12.7, 1.2, f'El plazo hábil al liberar separa más ({n(aso.AUC.max(), 3)}): es la causa más directa '
      'del retraso y se conoce al liberar; no es fuga.', tam=10.5, color=GRIS)
pie(s, 4, 'libro, capítulo 2b (EDA): resultados/eda/asociacion.csv, faltantes.csv, atipicos_if.csv')
notas(s, '0:40', 'A', 'criterio 2', 'Lo más importante: ninguna variable sola es sospechosa (máximo 0,732) y los '
      'vacíos son estructurales, con su mecanismo. Los atípicos llegan más tarde: son información.')

# ============================================================== 5 TIEMPO Y FUGA
s = nueva()
cabecera(s, 'El tiempo importa: partir al azar mediría con datos ya vistos',
         'CRITERIO 2  ·  TIEMPO Y FUGA')
figura(s, 'serie.png', 1.2, 3.2, w=11.6)
texto(s, 1.3, 9.0, 11.6, 1.6, f'No es estacionaria: ADF no rechaza la raíz unitaria (p = {n(est["ADF"], 2)}) y KPSS '
      f'rechaza la estacionariedad (p = {n(est["KPSS"], 3)}). Proveedores nuevos: {n(nue.loc["proveedor", "nuevo en validacion (%)"], 1)} '
      f'% en 2025 y {n(nue.loc["proveedor", "nuevo en prueba (%)"], 1)} % en 2026.', tam=10.5, color=GRIS)
texto(s, 1.3, 10.8, 11.6, 0.6, 'Entidades: codificación fuera de pliegue, por orden', tam=11,
      color=VERDE, negrita=True)
figura(s, 'f_te.png', 1.4, 11.5, h=1.5)
figura(s, 'adversaria.png', 13.4, 3.2, w=10.8)
texto(s, 13.4, 8.2, 10.8, 3.6, [
    (f'Validación adversaria: un XGBoost intenta adivinar de qué conjunto viene cada posición.', {'negrita': True}),
    (f'Al azar no puede ({n(adv[2], 3)}). Por fecha acierta siempre ({n(adv[0], 3)}), y sigue en {n(adv[1], 3)} sin '
     'las variables que fechan la fila: muchas derivas pequeñas (IVA, región del centro, mezcla de líneas) bastan para '
     'reconocer el año.', {}),
    ('Por eso: partición cronológica, validación por ventanas crecientes y reentrenar con 2021-2025 antes de la prueba.',
     {'color': VERDE, 'negrita': True})], tam=10.5, espacio=5)
pie(s, 5, 'libro, capítulo 2c (tiempo y fuga): resultados/eda/ y resultados/similitud/auc.csv')
notas(s, '0:40', 'A', 'criterio 2', 'Tres pruebas de que el tiempo importa: la serie no es estacionaria, hay '
      'entidades nuevas y la validación adversaria reconoce el año. Fuga: la codificación por objetivo va fuera de pliegue '
      'y agrupada por orden, y se descartaron las variables registradas después de la liberación.')

# ============================================================== 6 TEORIA: OBJETIVO
s = nueva()
cabecera(s, 'XGBoost suma árboles; cada uno es un paso de Newton', 'CRITERIO 3  ·  FORMULACIÓN, '
         'PÉRDIDA Y OPTIMIZACIÓN')
texto(s, 1.3, 3.2, 14.5, 0.6, 'El modelo: el log-odds es una suma de M árboles', tam=11.5, color=VERDE, negrita=True)
figura(s, 'f_modelo.png', 1.4, 3.9, h=1.15)
texto(s, 1.3, 5.35, 14.5, 0.6, 'La pérdida: log-pérdida (entropía cruzada binaria), convexa en F', tam=11.5,
      color=VERDE, negrita=True)
figura(s, 'f_perdida.png', 1.4, 6.0, h=0.62)
texto(s, 1.3, 7.0, 14.5, 0.6, 'El objetivo del árbol m: pérdida más penalización de su complejidad', tam=11.5,
      color=VERDE, negrita=True)
figura(s, 'f_obj.png', 1.4, 7.7, h=1.15)
texto(s, 1.3, 9.2, 14.5, 0.6, 'Taylor de segundo orden alrededor de la predicción actual', tam=11.5, color=VERDE,
      negrita=True)
figura(s, 'f_taylor.png', 1.4, 9.9, h=1.0)
figura(s, 'f_gh.png', 1.4, 11.3, h=0.62)
caja(s, 16.6, 3.3, 7.6, 9.8, VERDECL)
texto(s, 17.1, 3.6, 6.6, 0.7, 'Qué es cada término', tam=12.5, color=VERDE, negrita=True)
y = 4.6
for t, c in [('f_m', 'un árbol de regresión: reparte x en T hojas con pesos w_j.'),
             ('σ', 'la logística: pasa el log-odds F a probabilidad.'),
             ('γ T', 'cobra por cada hoja; λ encoge los pesos hacia 0 (L2).'),
             ('g_i', 'gradiente: el residuo con signo, p̂ − y.'),
             ('h_i', 'hessiano: varianza de Bernoulli, alta donde el modelo duda.')]:
    caja(s, 17.1, y + 0.14, 0.2, 0.2, VERDE)
    texto(s, 17.7, y, 6.2, 1.5, [(t, {'negrita': True, 'color': VERDE}), (c, {'tam': 10})], tam=11)
    y += 1.65
pie(s, 6, 'libro, capítulo 7 (fundamentos de XGBoost); Chen y Guestrin (2016), Friedman (2001)')
notas(s, '1:00', 'B', 'criterio 3', 'Boosting: descenso por gradiente en el espacio de funciones. Derivar en voz alta '
      'g y h de la log-pérdida: dℓ/dF = p − y; d²ℓ/dF² = p(1 − p). Es lo que más preguntan.')

# ============================================================== 7 TEORIA: HOJAS Y CORTES
s = nueva()
cabecera(s, 'Peso de hoja y ganancia de corte en forma cerrada', 'CRITERIO 3  ·  PRINCIPIO DE '
         'OPTIMIZACIÓN')
texto(s, 1.3, 3.2, 11.0, 0.6, 'Peso óptimo de la hoja j', tam=11.5, color=VERDE, negrita=True)
figura(s, 'f_w.png', 1.4, 3.95, h=1.35)
texto(s, 1.3, 5.6, 11.0, 1.8, 'Con G_j y H_j la suma de g y h en la hoja, la aproximación es una parábola en w_j: su '
      'mínimo es un paso de Newton, el gradiente sobre la curvatura, encogido por λ.', tam=10.5, color=GRIS)
texto(s, 1.3, 7.6, 22.9, 0.6, 'Ganancia de partir una hoja en izquierda (L) y derecha (R): se corta solo si es positiva',
      tam=11.5, color=VERDE, negrita=True)
figura(s, 'f_ganancia.png', 1.4, 8.35, h=1.35)
texto(s, 1.3, 10.1, 13.0, 0.6, 'Paso corto: cada árbol entra encogido por la tasa de aprendizaje', tam=11.5,
      color=VERDE, negrita=True)
figura(s, 'f_eta.png', 1.4, 10.8, h=0.62)
tarjeta(s, 13.2, 3.3, 11.0, 4.0, 'Búsqueda por histograma (hist)', 'Cada predictora se discretiza en a lo sumo 256 '
        'tramos y la ganancia se evalúa solo en sus bordes: buscar el mejor corte es recorrer un histograma, no ordenar '
        'las filas.', CREMA, tam_c=10.5)
banda(s, 13.2, 10.2, 11.0, 2.8, [('Contraste: la logística  ', True, VERDE),
                                 ('minimiza la misma log-pérdida con una sola recta, convexa y con un único óptimo. No ve '
                                  'interacciones ni umbrales sin construirlos; XGBoost los encuentra con los cortes.',
                                  False, TINTA)], NIEBLA, tam=10.5)
pie(s, 7, 'libro, capítulo 7; Chen y Guestrin (2016)')
notas(s, '0:45', 'B', 'criterio 3', 'w* = −G/(H + λ) sale de derivar la parábola e igualar a cero. La ganancia compara '
      'la pérdida óptima de las dos hojas hijas con la de la madre; γ es el costo de una hoja más.')

# ============================================================== 8 TEORIA: REGULARIZACION Y COMPLEJIDAD
s = nueva()
cabecera(s, f'El ajuste crece casi lineal en n (exponente {n(expo, 2)})', 'CRITERIO 3  ·  '
         'SUPUESTOS, REGULARIZACIÓN, COMPLEJIDAD Y GENERALIZACIÓN')
filas_h = [['hiperparámetro', 'en la fórmula', 'qué controla'],
           ['n_estimators', 'M', 'capacidad y costo'],
           ['learning_rate', 'η', 'pasos cortos, mejor generalización'],
           ['max_depth', 'T ≤ 2^D', 'interacciones de hasta D variables'],
           ['min_child_weight', 'H_j ≥ c', 'hojas con suficiente evidencia'],
           ['subsample, colsample', 'filas y columnas', 'azar que decorrelaciona árboles'],
           ['reg_lambda, gamma', 'λ, γ', 'encoge pesos, cobra hojas']]
tabla(s, 1.3, 3.3, 12.2, 5.6, filas_h, anchos=[3.4, 2.4, 4.4], tam=10)
figura(s, 'complejidad.png', 14.0, 3.2, w=10.2)
tarjeta(s, 1.3, 9.3, 7.4, 3.8, 'Supuestos', 'Filas independientes dado x (por eso los pliegues agrupan por orden); la '
        'relación de 2021-2025 sigue valiendo en 2026; no supone linealidad ni normalidad.', VERDECL, tam_t=12, tam_c=10)
tarjeta(s, 9.0, 9.3, 7.4, 3.8, 'Complejidad', f'Ajuste O(M·D·n·d′) por histograma; predicción O(M·D). Medido: '
        f'{n(cxx.segundos_ajuste.iloc[-1])} s con {n(cxx.filas.iloc[-1])} filas; {n(cxx.ms_inferencia_por_1000.iloc[-1], 1)} ms '
        'por cada 1.000 predicciones.', CREMA, tam_t=12, tam_c=10)
tarjeta(s, 16.7, 9.3, 7.5, 3.8, 'Generalización', 'Parada por presupuesto y validación por ventanas en el tiempo; una '
        'entidad nueva recibe la tasa de su nivel superior.', NIEBLA, tam_t=12, tam_c=10)
pie(s, 8, 'libro, capítulo 7: resultados/complejidad.csv (tiempos medidos en srvgdai)')
notas(s, '0:45', 'B', 'criterio 3', 'Cada hiperparámetro y dónde vive en la fórmula. Complejidad: teoría y medición '
      f'coinciden; el exponente {n(expo, 2)} en log-log es casi lineal. Supuesto clave: independencia dado x.')

# ============================================================== 9 DISENO
s = nueva()
cabecera(s, 'Hacia adelante en el tiempo, con el mismo presupuesto', 'CRITERIO 4  ·  '
         'DISEÑO EXPERIMENTAL E IMPLEMENTACIÓN')
figura(s, 'f_112.png', 1.4, 3.4, w=12.8)
bl = ['22-S1', '22-S2', '23-S1', '23-S2', '24-S1', '24-S2', '25-S1', '25-S2']
for k, b in enumerate(bl):          # semestres desde 2021-S1: entrena k + 1, valida 1 adentro, evalúa el siguiente
    y, u = 4.65 + k * 0.62, 1.05
    caja(s, 1.3, y, u * (k + 1) - 0.05, 0.48, VERDE)
    caja(s, 1.3 + u * (k + 1), y, u - 0.05, 0.48, PALIDO)
    caja(s, 1.3 + u * (k + 2), y, u - 0.05, 0.48, ORO)
    texto(s, 1.3 + u * (k + 3) + 0.1, y + 0.03, 2.0, 0.45, b, tam=9.5, color=GRIS)
rico(s, 1.3, 9.85, 13.0, 0.6, [('verde: entrena   ', True, VERDE), ('gris: valida adentro (elige hiperparámetros)   ',
                                False, GRIS), ('oro: evalúa (bucle externo)', True, ORO)], tam=9.5)
texto(s, 1.3, 10.9, 13.0, 2.2, f'4 balanceos: ninguno, SMOTE, ADASYN y pesos de clase, siempre dentro del pipeline y '
      f'solo sobre el entrenamiento. 20 evaluaciones por optimizador y bloque. Total: {n(horas)} horas de CPU en un '
      'servidor del grupo (srvgdai), con 30 trabajos a la vez.', tam=10.5)
tarjeta(s, 15.0, 3.3, 9.2, 5.2, 'Reproducibilidad', ['Un script por paso en src/; semilla 2026 en todo.',
                                                    'Preprocesador sin hiperparámetros, ajustado una vez por ventana.',
                                                    '57 pruebas con pytest e integración continua en GitHub.',
                                                    'Cada trabajo deja su JSON con parámetros y métricas.'],
        VERDECL, tam_c=10.5)
tarjeta(s, 15.0, 8.8, 9.2, 4.3, 'Decisiones computacionales', 'Matrices preprocesadas en float32 leídas con mmap, '
        'para correr 30 trabajos a la vez; SMOTE y ADASYN calculados una vez por ventana; KNN sobre 30 componentes '
        'principales.', CREMA, tam_c=10.5)
pie(s, 9, 'libro, capítulo 5 (las 112): src/validacion.py, src/experimentos.py, tests/')
notas(s, '0:45', 'B', 'criterio 4', 'Validación anidada: el externo mide, el interno elige. La prueba 2026 no entra. '
      'Mismo presupuesto para comparar optimizadores de forma justa. Reproducibilidad: pruebas y CI.')

# ============================================================== 10 OPTIMIZADORES
s = nueva()
cabecera(s, 'Optuna llega antes; con 20 evaluaciones empatan los cuatro', 'CRITERIO 4  ·  '
         'HIPERPARÁMETROS Y OPTIMIZADORES')
figura(s, 'optimizadores.png', 1.2, 3.2, w=12.6)
filas_o = [['optimizador', 'AUC perdido a 5 evaluaciones', 'a 20']]
for o in ['optuna', 'deap', 'random', 'grid']:
    filas_o.append([OPT[o], n(cur.loc[5, o], 4), n(cur.loc[20, o], 4)])
tabla(s, 14.3, 3.4, 9.9, 3.3, filas_o, anchos=[2.6, 3.8, 1.8], tam=10.5)
tarjeta(s, 14.3, 7.1, 9.9, 3.0, 'Optuna (TPE)', 'Modela qué zonas del espacio dieron buen AUC y propone donde la '
        'mejora esperada es mayor: aprende de los ensayos anteriores.', VERDECL, tam_t=12, tam_c=10)
tarjeta(s, 14.3, 10.3, 9.9, 2.8, 'DEAP (genético)', 'Selección, cruce, mutación y elitismo sobre una población de '
        'configuraciones; llega tarde pero llega.', CREMA, tam_t=12, tam_c=10)
pie(s, 10, 'libro, capítulo 5: data/modelos/curvas_optimizador.csv')
notas(s, '0:45', 'B', 'criterio 4', 'Lectura: con poco presupuesto conviene Optuna; con 20 evaluaciones la superficie '
      'es plana cerca del óptimo y los cuatro terminan a milésimas. El balanceo tampoco cambia el AUC.')

# ============================================================== 11 RESULTADOS
s = nueva()
cabecera(s, f'XGBoost gana los 8 bloques: AUC {n(top.AUC, 3)} contra {n(mejor.iloc[1].AUC, 3)}',
         'CRITERIO 5  ·  MÉTRICAS, LÍNEA BASE Y COMPARACIÓN')
filas_r = [['modelo (mejor combinación)', 'AUC ± desv.', 'PR-AUC', 'F1', 'ms / 1.000']]
for _, r in mejor.iterrows():
    filas_r.append([f'{NOMBRE[r.modelo]} · {BAL[r.balanceo]} · {OPT[r.optimizador]}',
                    f'{n(r.AUC, 3)} ± {n(r.AUC_desv, 3)}', n(r.PR_AUC, 3), n(r.F1, 3), n(r.ms_inferencia_por_1000, 1)])
tabla(s, 1.3, 3.3, 13.6, 5.8, filas_r, anchos=[5.6, 2.6, 1.6, 1.4, 1.6], tam=10)
figura(s, 'rangos_cd.png', 15.3, 3.2, w=8.9)
wl = wil[wil.contra == 'logistica'].iloc[0]
texto(s, 15.3, 9.35, 8.9, 1.6, [(f'Friedman: χ² = {n(fri.chi2, 1)}, p < 0,001.', {'negrita': True}),
                               (f'Nemenyi no separa a XGBoost de la logística ni del bosque (diferencia crítica '
                                f'{n(fri.CD, 2)}); Wilcoxon por pares sí: gana los 8 bloques, p de Holm = '
                                f'{n(wl.p_holm, 3)}.', {})], tam=10, espacio=3)
texto(s, 1.3, 9.4, 13.6, 1.1, 'Métricas: AUC ordena por riesgo (independiente del umbral), PR-AUC atiende a la '
      'clase de interés, F1 mide la decisión al umbral; Brier y ECE, si la probabilidad es creíble.', tam=10,
      color=GRIS)
tabla(s, 1.3, 10.6, 13.6, 2.5, [['prueba 2026 (línea base)', 'AUC', 'F1', 'Brier'],
                                ['regla del proveedor', n(rp['ROC-AUC'], 3), n(rp.F1, 3), n(rp.Brier, 3)],
                                ['regresión logística', n(lb['ROC-AUC'], 3), n(lb.F1, 3), n(lb.Brier, 3)]],
      anchos=[5.6, 2.0, 2.0, 2.0], tam=10)
pendiente(s, 15.3, 11.15, 8.9, 1.95, 'XGBoost en la prueba 2026 y la tabla de brecha validación-prueba de cada modelo.')
pie(s, 11, 'libro, capítulos 4 y 5: data/modelos/lideres.csv, friedman.csv, wilcoxon.csv, metricas.csv')
notas(s, '1:00', 'A', 'criterio 5', 'Media y desviación del bucle externo, 8 semestres. Ser precisos con las pruebas: '
      'Friedman rechaza la igualdad; Nemenyi es conservador y no separa a los tres primeros; el Wilcoxon pareado con Holm '
      'sí, porque XGBoost gana en cada bloque. La ganancia es de una centésima de AUC.')

# ============================================================== 12 ERRORES Y SOBREAJUSTE
s = nueva()
gap = (xb.AUC_interno - xb.AUC).mean()
cabecera(s, f'Sobreajuste bajo: {n(gap, 3)} de AUC entre interno y externo',
         'CRITERIO 5  ·  ANÁLISIS DE ERRORES Y SOBREAJUSTE')
figura(s, 'interno_externo.png', 1.2, 3.2, w=11.0)
figura(s, 'aprendizaje.png', 12.6, 3.2, w=6.2)
figura(s, 'calibracion.png', 19.0, 3.2, w=5.3)
b25 = xb.set_index('bloque').loc['2025-S1']
texto(s, 1.3, 9.3, 11.0, 1.6, f'El bloque 2025-S1 baja de {n(b25.AUC_interno, 3)} adentro a {n(b25.AUC, 3)} afuera: es el '
      'semestre en que cambia la mezcla de 2025 (lámina 5), no un modelo que memoriza.', tam=10.5, color=GRIS)
texto(s, 12.6, 9.3, 11.6, 1.6, f'Logística (modelo base): {n(ca["AUC entrenamiento"], 3)} en entrenamiento y '
      f'{n(ca["AUC validación"], 3)} en validación con todas las filas; bien calibrada en 2026 (ECE {n(lb.ECE, 3)}).',
      tam=10.5, color=GRIS)
pendiente(s, 1.3, 11.1, 22.9, 2.0, 'para el mejor modelo en 2026: matriz de confusión por línea de negocio, errores con '
          'entidades nuevas frente a conocidas, un falso negativo típico y SHAP (qué variables mueven cada predicción).')
pie(s, 12, 'libro, capítulos 4 y 5: resultados/experimentos/por_bloque.csv, data/modelos/curva_aprendizaje.csv')
notas(s, '1:00', 'B', 'criterio 5', 'Sobreajuste en dos vistas: interno contra externo por bloque y la curva de '
      'aprendizaje. Lo que falta del análisis de errores depende de la evaluación final en 2026; no se adelanta.')

# ============================================================== 13 CONCLUSIONES
s = nueva()
cabecera(s, 'El retraso se anticipa al liberar, con límites claros', 'CRITERIO 6  ·  CONCLUSIONES Y '
         'LIMITACIONES')
texto(s, 1.3, 3.2, 11.2, 0.6, 'Conclusiones', tam=13, color=VERDE, negrita=True)
lista(s, 1.3, 4.0, 11.2, [
    f'Se puede ordenar por riesgo al liberar: AUC {n(top.AUC, 3)} hacia adelante en el tiempo, contra '
    f'{n(rp["ROC-AUC"], 3)} de la regla del proveedor en 2026.',
    'La señal está en el plazo, el historial del proveedor y las entidades; ninguna variable sola basta.',
    'XGBoost gana, pero por una centésima: la logística, más simple y calibrada, queda como alternativa defendible.',
    'El optimizador importa con poco presupuesto; el balanceo casi no mueve el AUC.'], tam=11, paso=2.05, alto=1.9)
texto(s, 13.2, 3.2, 11.0, 0.6, 'Limitaciones y mejoras', tam=13, color=TEJA, negrita=True)
lista(s, 13.2, 4.0, 11.0, [
    'Deriva: 2025 se distingue de 2021-2024 casi por completo; hay que reentrenar y vigilar.',
    'La etiqueta de lo más reciente madura tarde (censura al final del periodo).',
    'Maestros en foto actual, no con su vigencia histórica.',
    'Una sola empresa: la generalización a otros grupos no está probada.',
    f'Arranque en frío: {n(nue.loc["proveedor", "nuevo en prueba (%)"], 1)} % de proveedores nuevos en 2026.'],
      tam=11, paso=1.75, alto=1.6)
pie(s, 13, 'libro, capítulos 2 a 7')
notas(s, '1:00', 'A', 'criterio 6', 'Responder la pregunta con cifras y decir los límites sin rodeos. Mejoras: el '
      'modelo original del Entregable 3 ataca el arranque en frío y usa los días de retraso.')

# ============================================================== 14 PREGUNTAS
s = nueva(VERDE)
logo(s, 1.3, 1.4, 3.9, 3.0)
texto(s, 1.3, 5.6, 22.0, 1.6, 'Preguntas', tam=40, color=BLANCO, negrita=True)
filete(s, 1.3, 7.4, 2.3)
texto(s, 1.3, 8.0, 22.0, 3.0, [f'Repositorio   {URL["repo"]}', f'Libro            {URL["libro"]}',
                               f'Tablero         {URL["tablero"]}'], tam=14, color=PALIDO, espacio=6)
texto(s, 1.3, 12.2, 22.0, 0.8, 'Responden los dos autores (criterio 7: dominio individual).', tam=11, color=ORO_CL)
notas(s, '2:00', 'A y B', 'criterio 7', 'Preparar: derivar g, h y w*; por qué la codificación va fuera de pliegue; por '
      'qué la prueba no elige nada; por qué Nemenyi y Wilcoxon dicen cosas distintas; qué pasa con un proveedor nuevo.')

prs.save(SAL)
with open(os.path.join(AQUI, 'notas.md'), 'w', encoding='utf-8') as f:
    f.write('# Notas del expositor\n\nLas escribe `construir.py` (también van en las notas de cada diapositiva). '
            'A = Daniel Carbonó, B = Luis Pérez; cada uno ensaya también lo del otro.\n\n'
            '| # | min | criterio | expone | qué decir |\n|---|---|---|---|---|\n')
    for k, (m, e, c, t) in enumerate(NOTAS, 1):
        f.write(f'| {k} | {m} | {c} | {e} | {t} |\n')
print(f'{len(prs.slides)} diapositivas en {SAL}')
