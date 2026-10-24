# -*- coding: utf-8 -*-
"""Sustentacion del proyecto OTIF con la plantilla Uninorte de las sustentaciones de GLM, para la rubrica de 20 minutos.

Sigue la tabla de 18 minutos de entregables/diapositivas.md con el mejor benchmark (XGBoost) en la teoria: criterio 1
(2 min), 2 (3), 3 (5), 4 (3), 5 (3), 6 (2) y 2 de preguntas. Tres diapositivas son pantallas del tablero en vivo, con
su captura como respaldo (capturar_tablero.py). Las cifras se leen de resultados/ y data/modelos/ al construir; las
figuras y formulas son PNG de figuras.py. Lo que depende de la evaluacion final en 2026 sale en una caja "pendiente".

Uso (Python base, que tiene python-pptx):
    python presentacion/construir.py [salida.pptx]      # por defecto presentacion/sustentacion_otif.pptx
Revision:
    python presentacion/revisar_pptx.py presentacion/sustentacion_otif.pptx
    python presentacion/exportar.py presentacion/sustentacion_otif.pptx presentacion/render 1600
"""
import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('IMG_DIR', os.path.join(AQUI, 'img'))
sys.path.insert(0, AQUI)
from plantilla_uninorte import *  # noqa: E402,F401,F403

RAIZ = os.path.dirname(AQUI)
R, MOD = os.path.join(RAIZ, 'resultados'), os.path.join(RAIZ, 'data', 'modelos')
SAL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, 'sustentacion_otif.pptx')
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
uni = leer(R, 'eda', 'univariado_numericas.csv')
ati = leer(R, 'eda', 'atipicos_if.csv').set_index('grupo')['tasa de tarde']
est = leer(R, 'eda', 'estacionariedad.csv').set_index('prueba')['p']
nue = leer(R, 'eda', 'entidades_nuevas.csv').set_index('variable')
lin = leer(R, 'eda', 'tasa_linea.csv').sort_values('tasa')
adv = leer(R, 'similitud', 'auc.csv').AUC.tolist()
lid = leer(MOD, 'lideres.csv')
mejor = lid.sort_values('AUC', ascending=False).groupby('modelo').head(1)
top = mejor.iloc[0]
fri = leer(MOD, 'friedman.csv').iloc[0]
wil = leer(MOD, 'wilcoxon.csv')
pru = leer(R, 'pruebas', 'resumen.csv')
m26 = leer(R, 'prueba_2026', 'metricas.csv').set_index('modelo')
x26 = m26.loc['XGBoost (las 112)']
d26 = leer(R, 'prueba_2026', 'diferencias.csv').iloc[0]
s26 = leer(R, 'prueba_2026', 'segmentos.csv')
s26 = s26[s26.modelo == 'XGBoost (las 112)'].set_index('valor')
cr26 = leer(R, 'prueba_2026', 'criticas.csv').set_index('cifra')
k26 = leer(R, 'prueba_2026', 'contrastes.csv').iloc[0]
pub = pd.read_csv(os.path.join(AQUI, '..', 'original', 'publicos_otif_resumen.csv'))
b26 = leer(R, 'prueba_2026', 'brecha.csv').set_index('modelo').loc['XGBoost (las 112)']
f26 = leer(R, 'prueba_2026', 'confusion.csv').set_index(['real', 'predicho']).posiciones
od26 = leer(MOD, 'p26_ordenes.csv').set_index('meta')
cur = leer(MOD, 'curvas_optimizador.csv').pivot(index='eval', columns='optimizador', values='mean')
met = leer(MOD, 'metricas.csv').set_index(['modelo', 'partición'])
lb = met.loc[('Regresión logística', 'prueba 2026')]
rp = met.loc[('Regla del proveedor', 'prueba 2026')]
ca = leer(MOD, 'curva_aprendizaje.csv').iloc[-1]
pb = leer(R, 'experimentos', 'por_bloque.csv')
xb = pb[(pb.modelo == top.modelo) & (pb.balanceo == top.balanceo) & (pb.optimizador == top.optimizador)]
cx = leer(R, 'complejidad.csv')
cxx = cx[cx.modelo == 'xgboost'].sort_values('filas')
expo = np.polyfit(np.log(cxx.filas), np.log(cxx.segundos_ajuste), 1)[0]
horas = pb.segundos_total.sum() / 3600
elegidos = pd.DataFrame([json.load(open(os.path.join(R, 'experimentos', 'trabajos',
                                                     f'{top.modelo}__{top.balanceo}__{top.optimizador}__{b}.json'),
                                        encoding='utf-8'))['params'] for b in xb.bloque])
NOMBRE = {'xgboost': 'XGBoost', 'random_forest': 'Random Forest', 'logistica': 'Logística (SGD)',
          'svm': 'SVM lineal (SGD)', 'arbol': 'Árbol', 'knn': 'KNN (PCA 30)', 'naive_bayes': 'Naive Bayes'}
BAL = {'ninguno': 'sin balanceo', 'smote': 'SMOTE', 'adasyn': 'ADASYN', 'pesos': 'pesos'}
OPT = {'grid': 'Grid', 'random': 'Random', 'optuna': 'Optuna', 'deap': 'DEAP'}

N_TOTAL = 22
NOTAS = []
_k = [0]


def pie(s, fuente):
    _k[0] += 1
    filete(s, 1.3, 13.45, 22.9, LINEA, alto=0.04)
    texto(s, 1.3, 13.6, 16.5, 0.5, f'Fuente: {fuente}', tam=8.5, color=GRIS_CL)
    texto(s, 17.8, 13.6, 6.4, 0.5, f'{_k[0] + 1} / {N_TOTAL}', tam=8.5, color=GRIS_CL, alineacion=PP_ALIGN.RIGHT)


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


def en_vivo(s, x, y, w, h, pasos):
    """Recuadro de una pantalla del tablero: lo que se hace en vivo, paso a paso."""
    caja(s, x, y, w, h, VERDE)
    texto(s, x + 0.45, y + 0.3, w - 0.9, 0.7, 'EN VIVO', tam=11, color=ORO_CL, negrita=True)
    texto(s, x + 0.45, y + 1.0, w - 0.9, 0.7, URL['tablero'], tam=10, color=PALIDO)
    texto(s, x + 0.45, y + 1.8, w - 0.9, h - 2.0, [f'{k}. {p}' for k, p in enumerate(pasos, 1)], tam=10.5,
          color=BLANCO, espacio=6)


def captura(s, nombre, x, y, w):
    figura(s, nombre, x, y, w=w)
    texto(s, x, y - 0.5, w, 0.45, 'captura de respaldo, por si el tablero no carga', tam=8.5, color=GRIS_CL)


# ============================================================== 1 PORTADA
s = nueva()
caja(s, 0, 0, 8.8, 14.29, VERDE)
logo(s, 1.3, 1.4, 3.9, 3.0)
texto(s, 1.3, 5.2, 6.8, 0.8, 'UNIVERSIDAD DEL NORTE', tam=12.5, color=BLANCO, negrita=True)
texto(s, 1.3, 6.0, 6.8, 1.6, ['Maestría en Estadística Aplicada', 'Machine Learning'], tam=11, color=PALIDO, espacio=4)
filete(s, 1.3, 10.6, 1.8)
texto(s, 1.3, 11.1, 6.8, 1.6, ['Daniel Carbonó Domínguez', 'Luis Pérez'], tam=11.5, color=BLANCO, negrita=True, espacio=4)
texto(s, 1.3, 12.6, 6.8, 0.8, 'Octubre de 2026', tam=10.5, color=PALIDO)
texto(s, 10.0, 2.6, 13.5, 0.8, 'CLASIFICACIÓN BINARIA SUPERVISADA', tam=13, color=VERDE, negrita=True)
texto(s, 10.0, 3.7, 14.4, 4.2, ['Predicción del incumplimiento de órdenes de compra en la agroindustria de la palma '
                                'de aceite'], tam=24, color=TINTA, negrita=True)
filete(s, 10.0, 8.1, 2.3)
texto(s, 10.0, 8.6, 14.0, 1.2, f'{n(tam.loc["total", "filas"])} posiciones de {n(tam.loc["total", "órdenes"])} '
      'órdenes de compra, 2021-2026', tam=15, color=GRIS)
texto(s, 10.0, 10.4, 14.0, 2.4, [f'Repositorio   {URL["repo"]}', f'Libro            {URL["libro"]}',
                                 f'Tablero         {URL["tablero"]}'], tam=12, color=GRIS_CL, espacio=4)
notas(s, '0:15', 'A', 'criterio 1', 'Presentarse los dos y dejar las tres direcciones en pantalla. Render ya abierto '
      'desde antes: el plan gratuito se duerme.')

# ============================================================== 2 PROBLEMA
s = nueva()
cabecera(s, 'Uno de cada tres pedidos llega tarde, y se sabe cuando ya pasó', 'CRITERIO 1  ·  EL PROBLEMA')
cifra(s, 1.3, 3.4, 5.4, 2.7, pc(tam.loc['total', 'tasa de tarde']), 'de las posiciones llega tarde')
cifra(s, 1.3, 6.4, 5.4, 2.7, pc(lin.tasa.max(), 0), f'en {lin.linea.iloc[-1]}; {pc(lin.tasa.min(), 0)} en {lin.linea.iloc[0]}', CREMA)
cifra(s, 1.3, 9.4, 5.4, 2.7, pc(lb.sensibilidad, 0), f'de las tardías de 2026, marcando el {pc(lb.marcadas, 0)}', NIEBLA)
texto(s, 7.6, 3.3, 9.2, 0.6, 'Por qué importa', tam=12.5, color=VERDE, negrita=True)
texto(s, 7.6, 4.0, 9.2, 3.6, 'El comprador hoy se entera del retraso cuando la mercancía no llega. Con miles de '
      'posiciones abiertas, no puede seguirlas todas: necesita saber, el día en que libera la orden, cuáles vigilar.',
      tam=11)
texto(s, 7.6, 7.3, 9.2, 0.6, 'Qué dice la literatura', tam=12.5, color=VERDE, negrita=True)
texto(s, 7.6, 8.0, 9.2, 5.0, ['En fabricantes de maquinaria y equipos llega tarde del 13 % al 14 % de los pedidos '
                              '(Brintrup y otros, 2020; Steinberg y otros, 2023): aquí es más del doble.',
                              ('Una revisión de 26 estudios encuentra que 16 parten las filas al azar; reevaluado sin '
                               'fuga y en el tiempo, un AUC de 0,96 baja a 0,73 (Raval, 2026). Este trabajo evalúa '
                               'siempre hacia adelante.', {'antes': 6})], tam=10.5, color=GRIS)
tarjeta(s, 17.4, 3.4, 6.8, 9.7, 'Objetivos',
        ['General: ordenar las posiciones por riesgo de retraso el día de la liberación y señalar cuáles seguir.',
         'Específico 1: comparar 7 modelos × 4 balanceos × 4 optimizadores con el mismo presupuesto.',
         'Específico 2: medir cuánto cuesta decidir con entidades nuevas (proveedor o material sin historia).',
         'Específico 3: dejar el modelo servido en un tablero con predicción en vivo.'], VERDECL, tam_c=10.5)
pie(s, 'libro, introducción y literatura; capítulo 5 (modelo base): data/modelos/metricas.csv')
notas(s, '0:45', 'A', 'criterio 1', 'El problema en una frase y su magnitud. La tercera cifra es del modelo base en '
      '2026 y ya está en el libro: marcando poco más de un tercio, se ven venir siete de cada diez tardías.')

# ============================================================== 3 FORMULACION
s = nueva()
cabecera(s, 'Estimar la probabilidad de llegar tarde y decidir con un umbral', 'CRITERIO 1  ·  FORMULACIÓN '
         'MATEMÁTICA')
texto(s, 1.3, 3.2, 14.0, 0.6, 'Unidad: la posición i, el día t en que se libera su orden',
      tam=12, color=VERDE, negrita=True)
figura(s, 'f_x.png', 1.4, 4.0, h=0.75)
texto(s, 1.3, 4.95, 14.0, 1.2, 'x: 136 predictoras conocidas en t. y = 1 si la primera entrada de mercancía llega más '
      'de 1 día hábil después de la fecha prometida vigente al liberar.', tam=10.5, color=GRIS)
texto(s, 1.3, 6.4, 14.0, 0.6, 'El modelo estadístico: y | x es Bernoulli con probabilidad p(x)', tam=12, color=VERDE,
      negrita=True)
figura(s, 'f_p.png', 1.4, 7.15, h=0.75)
texto(s, 1.3, 8.1, 14.0, 1.4, 'Se estima p(x) minimizando la log-pérdida, que es menos la log-verosimilitud de '
      'Bernoulli; la decisión ŷ compara p̂ con un umbral τ.', tam=10.5, color=GRIS)
figura(s, 'f_bern.png', 1.4, 9.6, h=0.75)
texto(s, 1.3, 10.7, 14.0, 2.4, [('El umbral τ se elige en la validación 2025, maximizando F1; la prueba 2026 se usa una '
                                 'sola vez y no elige nada.', {'negrita': True}),
                                ('Métricas: AUC para ordenar por riesgo (no depende de τ) y F1 para la decisión.', {})],
      tam=10.5, espacio=4)
tarjeta(s, 16.0, 3.3, 8.2, 4.6, 'Tarea', 'Clasificación binaria supervisada: la etiqueta se observa después, cuando '
        'entra la mercancía, y el modelo aprende de las órdenes ya cerradas.', VERDECL, tam_c=10.5)
tarjeta(s, 16.0, 8.2, 8.2, 4.9, 'Por qué no regresión', 'La decisión del comprador es binaria: seguir la orden o no. '
        'Los días de retraso tienen colas muy largas y se usan como contexto; el modelo original del Entregable 3 los '
        'aprovecha con cortes ordenados.', CREMA, tam_c=10.5)
pie(s, 'libro, capítulos 1 y 2 (definición de la etiqueta y de la población)')
notas(s, '1:00', 'A', 'criterio 1', 'x, y, el modelo Bernoulli, la log-pérdida como menos log-verosimilitud y el '
      'umbral. Insistir: τ sale de 2025 y 2026 no elige nada.')

# ============================================================== 4 TABLERO: CONTEXTO Y EDA
s = nueva()
cabecera(s, 'Tablero: el contexto y la exploración, con filtros', 'CRITERIO 2  ·  PANTALLA EN VIVO')
captura(s, 'tablero_eda.png', 1.3, 3.6, 15.6)
en_vivo(s, 17.6, 3.3, 6.6, 9.8, ['Contexto: la fuente (SAP, 2021-2026), la población y la etiqueta.',
                                 'EDA: el objetivo por clase y la tasa por mes.',
                                 'Filtrar una línea de negocio: todo cambia con ella (de '
                                 f'{pc(lin.tasa.min(), 0)} a {pc(lin.tasa.max(), 0)} de tarde).',
                                 'Bajar a distribuciones y atípicos.'])
pie(s, f'tablero, páginas Contexto y EDA ({URL["tablero"]})')
notas(s, '0:45', 'A', 'criterio 2', 'Pantalla en vivo. Si el tablero no carga, la captura queda en la diapositiva. '
      'Mostrar el filtro por línea: industrial y palma llegan tarde más que banano y ganadería.')

# ============================================================== 5 DATOS
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
                               ('La partición es por la fecha de la orden, no al azar (diapositiva 8).', {'color': GRIS})],
      tam=11, espacio=6)
pie(s, 'libro, capítulo 2 (datos); diccionario de variables en data/diccionario.csv')
notas(s, '0:45', 'A', 'criterio 2', 'Fuente y estructura. La unidad independiente es la orden: por eso los pliegues '
      'se agrupan por orden. La partición es cronológica.')

# ============================================================== 6 DESCRIPTIVA Y CALIDAD
s = nueva()
cabecera(s, 'Colas largas, vacíos con causa y atípicos que llegan más tarde', 'CRITERIO 2  ·  DESCRIPTIVA, FALTANTES '
         'Y ATÍPICOS')
asim = uni.sort_values('asimetría', ascending=False).head(6)
filas_a = [['variable', 'mediana', 'p99', 'asimetría']]
for _, r in asim.iterrows():
    filas_a.append([r.nombre, n(r.mediana, 1), n(r.p99, 0), n(r['asimetría'], 1)])
tabla(s, 1.3, 3.3, 12.8, 4.6, filas_a, anchos=[5.4, 2.0, 2.4, 2.0], tam=10)
texto(s, 1.3, 8.2, 12.8, 1.6, 'Por eso signo(x)·log(1 + |x|) antes de los modelos lineales: comprime las colas sin '
      'cambiar el orden. A los árboles no les cambia nada, porque solo usan el orden.', tam=10.5, color=GRIS)
tarjeta(s, 1.3, 10.0, 12.8, 3.1, 'Atípicos multivariados (Isolation Forest)', f'{pc(ati["atípicas"])} de tarde en las '
        f'atípicas contra {pc(ati["resto"])} en el resto: son señal, no ruido, y no se borran.', CREMA, tam_t=12,
        tam_c=10.5)
fv = falt.sort_values('vacíos (%)', ascending=False).head(6)
filas_f = [['variable', 'vacíos', 'Δ tasa (pp)']] + [[r.variable, n(r['vacíos (%)'], 1) + ' %',
                                                      n(r['diferencia (pp)'], 1)] for _, r in fv.iterrows()]
tabla(s, 14.8, 3.3, 9.4, 4.6, filas_f, anchos=[4.6, 1.8, 1.8], tam=9.5)
tarjeta(s, 14.8, 8.2, 9.4, 4.9, 'Faltantes estructurales', f'{int((falt["vacíos (%)"] > 50).sum())} variables con más '
        'de 50 % de vacíos, y no son azar: faltan porque la orden no tiene solicitud o no tiene imputación. Se marcan '
        'con indicadores tiene_* y se imputa la mediana; Δ tasa es la diferencia de retraso entre filas con y sin dato.',
        VERDECL, tam_t=12, tam_c=10)
pie(s, 'libro, capítulo 3 (EDA): resultados/eda/univariado_numericas.csv, faltantes.csv, atipicos_if.csv')
notas(s, '0:30', 'A', 'criterio 2', 'Distribuciones: colas largas, transformación justificada. Faltantes: el mecanismo '
      'es estructural, no aleatorio. Atípicos: tienen más retraso, se quedan.')

# ============================================================== 7 RELACIONES Y FUGA
s = nueva()
cabecera(s, 'Ninguna variable sola pasa de 0,75 de AUC, y ninguna es fuga', 'CRITERIO 2  ·  RELACIONES Y FUGA')
figura(s, 'auc_univariado.png', 1.2, 3.2, w=12.2)
texto(s, 1.3, 11.9, 12.2, 1.3, f'El plazo hábil al liberar separa más ({n(aso.AUC.max(), 3)}): es la causa más directa '
      'del retraso y se conoce al liberar.', tam=10.5, color=GRIS)
tarjeta(s, 14.0, 3.3, 10.2, 4.6, 'Auditoría de fuga', ['Cada grupo de variables, revisado contra el día de la '
                                                       'liberación.',
                                                       'Retiradas: los días de retraso, las fechas de entrada y de '
                                                       'factura, los cambios de fecha posteriores y los maestros que el '
                                                       'planificador ajusta después de ver retrasos.'], CREMA, tam_t=12,
        tam_c=10)
texto(s, 14.0, 8.2, 10.2, 0.6, 'Entidades: tasa encogida hacia su nivel superior', tam=11, color=VERDE, negrita=True)
figura(s, 'f_te.png', 14.1, 8.9, w=9.8)
texto(s, 14.0, 10.9, 10.2, 2.2, 'Se calcula fuera de pliegue y agrupada por orden: una posición no ve su propio '
      'objetivo ni el de su orden. Un valor nuevo recibe la tasa de su padre (el proveedor, la de su región).',
      tam=10, color=GRIS)
pie(s, 'libro, capítulos 3 y 4: resultados/eda/asociacion.csv; src/datos.py')
notas(s, '0:30', 'A', 'criterio 2', 'Relaciones: la señal está repartida. Fuga: lo que se retiró y por qué, y la '
      'codificación por objetivo fuera de pliegue, que es la fuga más fácil de cometer aquí.')

# ============================================================== 8 TIEMPO
s = nueva()
cabecera(s, 'El tiempo importa: partir al azar mediría con datos ya vistos', 'CRITERIO 2  ·  DERIVA Y PARTICIÓN')
figura(s, 'serie.png', 1.2, 3.2, w=11.6)
texto(s, 1.3, 9.0, 11.6, 2.4, f'No es estacionaria: ADF no rechaza la raíz unitaria (p = {n(est["ADF"], 2)}) y KPSS '
      f'rechaza la estacionariedad (p = {n(est["KPSS"], 3)}). Proveedores nuevos: '
      f'{n(nue.loc["proveedor", "nuevo en validacion (%)"], 1)} % en 2025 y '
      f'{n(nue.loc["proveedor", "nuevo en prueba (%)"], 1)} % en 2026.', tam=10.5, color=GRIS)
figura(s, 'adversaria.png', 13.4, 3.2, w=10.8)
texto(s, 13.4, 8.2, 10.8, 4.9, [
    ('Validación adversaria: un XGBoost intenta adivinar de qué conjunto viene cada posición.', {'negrita': True}),
    (f'Al azar no puede ({n(adv[2], 3)}). Por fecha acierta siempre ({n(adv[0], 3)}), y sigue en {n(adv[1], 3)} sin '
     'las variables que fechan la fila: muchas derivas pequeñas (IVA, región del centro, mezcla de líneas) bastan para '
     'reconocer el año.', {}),
    ('Por eso: partición cronológica, validación por ventanas crecientes y reentrenar con 2021-2025 antes de la prueba.',
     {'color': VERDE, 'negrita': True})], tam=10.5, espacio=5)
pie(s, 'libro, capítulo 4 (tiempo y fuga): resultados/eda/ y resultados/similitud/auc.csv')
notas(s, '0:30', 'A', 'criterio 2', 'Tres pruebas de que el tiempo importa: la serie no es estacionaria, hay entidades '
      'nuevas y la validación adversaria reconoce el año aun sin las variables de reloj.')

# ============================================================== 9 TEORIA I: MODELO Y PERDIDA
s = nueva()
cabecera(s, 'XGBoost: el log-odds como suma de árboles', 'CRITERIO 3  ·  FORMULACIÓN DEL ALGORITMO')
texto(s, 1.3, 3.2, 14.5, 0.6, 'El modelo', tam=12, color=VERDE, negrita=True)
figura(s, 'f_modelo.png', 1.4, 3.9, h=1.15)
texto(s, 1.3, 5.3, 14.5, 1.6, 'Cada árbol f_m reparte el espacio de las predictoras en T hojas y da a cada hoja un peso '
      'w_j. Los árboles se construyen en secuencia: el m corrige lo que los m − 1 anteriores todavía predicen mal.',
      tam=10.5, color=GRIS)
texto(s, 1.3, 7.1, 14.5, 0.6, 'La pérdida: menos la log-verosimilitud de Bernoulli', tam=12,
      color=VERDE, negrita=True)
figura(s, 'f_perdida.png', 1.4, 7.8, h=0.62)
texto(s, 1.3, 8.7, 14.5, 1.4, 'Es convexa en F y penaliza sin límite una probabilidad segura y equivocada: por eso '
      'produce probabilidades, no solo un orden.', tam=10.5, color=GRIS)
texto(s, 1.3, 10.3, 14.5, 0.6, 'El principio: descenso por gradiente en el espacio de funciones', tam=12, color=VERDE,
      negrita=True)
texto(s, 1.3, 11.0, 14.5, 2.1, 'En vez de mover parámetros, cada iteración agrega una función (un árbol) en la '
      'dirección que más baja la pérdida (Friedman, 2001). XGBoost usa además la curvatura: un paso de Newton '
      '(Chen y Guestrin, 2016).', tam=10.5)
tarjeta(s, 16.6, 3.3, 7.6, 4.7, 'Boosting', 'Árboles en secuencia, cada uno sobre los errores del anterior: baja el '
        'sesgo. Pide regularización para no sobreajustar.', VERDECL, tam_t=12, tam_c=10.5)
tarjeta(s, 16.6, 8.3, 7.6, 4.8, 'Bosque aleatorio', 'Árboles independientes y promediados: baja la varianza. Fue el '
        'segundo de los 112; la diferencia está en corregir el error en vez de promediarlo.', CREMA, tam_t=12,
        tam_c=10.5)
pie(s, 'libro, capítulo 8 (fundamentos); Friedman (2001), Chen y Guestrin (2016)')
notas(s, '1:15', 'B', 'criterio 3', 'Qué es el modelo, qué pérdida minimiza y por qué esa pérdida (Bernoulli). '
      'Principio: gradiente en el espacio de funciones. Contraste con el bosque: sesgo contra varianza.')

# ============================================================== 10 TEORIA II: TAYLOR
s = nueva()
cabecera(s, 'Cada árbol minimiza una aproximación de segundo orden', 'CRITERIO 3  ·  PRINCIPIO DE OPTIMIZACIÓN')
texto(s, 1.3, 3.2, 22.9, 0.6, 'El objetivo del árbol m: la pérdida más una penalización de su complejidad', tam=12,
      color=VERDE, negrita=True)
figura(s, 'f_obj.png', 1.4, 3.95, h=1.15)
texto(s, 1.3, 5.4, 22.9, 0.6, 'Taylor de segundo orden alrededor de la predicción actual F_{m−1}', tam=12,
      color=VERDE, negrita=True)
figura(s, 'f_taylor.png', 1.4, 6.1, h=1.0)
texto(s, 1.3, 7.5, 22.9, 0.6, 'Gradiente y hessiano de la log-pérdida: la derivación', tam=12, color=VERDE,
      negrita=True)
figura(s, 'f_deriv.png', 1.4, 8.25, w=21.5)
figura(s, 'f_gh.png', 1.4, 9.75, h=0.62)
tarjeta(s, 1.3, 10.8, 11.2, 2.3, 'g_i, el gradiente', 'El residuo con signo: positivo si el modelo sobreestima el '
        'riesgo de una posición que llegó a tiempo.', VERDECL, tam_t=12, tam_c=10)
tarjeta(s, 13.0, 10.8, 11.2, 2.3, 'h_i, el hessiano', 'La varianza de Bernoulli: grande donde el modelo duda (p̂ cerca '
        'de 0,5) y pequeño donde ya está seguro.', CREMA, tam_t=12, tam_c=10)
pie(s, 'libro, capítulo 8; Chen y Guestrin (2016)')
notas(s, '1:15', 'B', 'criterio 3', 'Derivar en voz alta: σ′ = σ(1 − σ); dℓ/dF = σ − y; d²ℓ/dF² = σ(1 − σ). γ cobra por '
      'hoja y λ encoge los pesos. Es la derivación que más preguntan.')

# ============================================================== 11 TEORIA III: HOJAS Y CORTES
s = nueva()
cabecera(s, 'Peso de hoja y ganancia de corte en forma cerrada', 'CRITERIO 3  ·  CÓMO SE CONSTRUYE CADA ÁRBOL')
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
pie(s, 'libro, capítulo 8; Chen y Guestrin (2016)')
notas(s, '1:15', 'B', 'criterio 3', 'w* = −G/(H + λ) sale de derivar la parábola e igualar a cero. La ganancia compara '
      'la pérdida óptima de las dos hojas hijas con la de la madre; γ es el costo de una hoja más.')

# ============================================================== 12 TEORIA IV: REGULARIZACION Y COMPLEJIDAD
s = nueva()
cabecera(s, f'El ajuste crece casi lineal en n (exponente {n(expo, 2)})', 'CRITERIO 3  ·  SUPUESTOS, REGULARIZACIÓN, '
         'COMPLEJIDAD Y GENERALIZACIÓN')
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
tarjeta(s, 16.7, 9.3, 7.5, 3.8, 'Generalización', 'Regularización más validación por ventanas en el tiempo; una '
        'entidad nueva recibe la tasa de su nivel superior.', NIEBLA, tam_t=12, tam_c=10)
pie(s, 'libro, capítulo 8: resultados/complejidad.csv (tiempos medidos en srvgdai)')
notas(s, '1:15', 'B', 'criterio 3', 'Cada hiperparámetro y dónde vive en la fórmula. Complejidad: teoría y medición '
      f'coinciden; el exponente {n(expo, 2)} en log-log es casi lineal. Supuesto clave: independencia dado x, que la '
      'agrupación por orden protege.')

# ============================================================== 13 DISENO
s = nueva()
cabecera(s, 'Hacia adelante en el tiempo, con el mismo presupuesto', 'CRITERIO 4  ·  DISEÑO EXPERIMENTAL')
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
texto(s, 1.3, 10.6, 13.0, 2.5, 'Bucle externo: 8 semestres, cada uno predicho con todo lo anterior. Bucle interno: el '
      'último semestre de cada ventana elige los hiperparámetros. La prueba 2026 no entra en nada de esto.', tam=10.5)
tarjeta(s, 15.0, 3.3, 9.2, 4.6, '4 balanceos', 'Ninguno, SMOTE, ADASYN y pesos de clase, siempre dentro del pipeline y '
        'solo sobre el entrenamiento de cada ventana: lo sintético nunca llega a la evaluación.', VERDECL, tam_c=10.5)
tarjeta(s, 15.0, 8.2, 9.2, 4.9, '4 optimizadores, mismo presupuesto', '20 evaluaciones por combinación y bloque para '
        'Grid, Random, Optuna (TPE) y DEAP (genético). Medias y desviaciones salen de los 8 bloques externos.', CREMA,
        tam_c=10.5)
pie(s, 'libro, capítulo 6 (las 112): src/validacion.py, src/experimentos.py')
notas(s, '0:45', 'B', 'criterio 4', 'Validación anidada: el externo mide, el interno elige. Mismo presupuesto para '
      'comparar optimizadores de forma justa. La prueba 2026 se mira una vez, al final.')

# ============================================================== 14 IMPLEMENTACION
s = nueva()
cabecera(s, 'Implementación: un pipeline, un espacio de búsqueda y pruebas', 'CRITERIO 4  ·  IMPLEMENTACIÓN E '
         'HIPERPARÁMETROS')
rango = {'n_estimators': '100 a 600', 'learning_rate': '0,01 a 0,3 (log)', 'max_depth': '3 a 10',
         'subsample': '0,5 a 1', 'colsample_bytree': '0,3 a 1', 'min_child_weight': '1 a 100 (log)'}
filas_e = [['hiperparámetro', 'espacio', 'elegido (8 bloques)']]
for c, rg in rango.items():
    d = 0 if c in ('n_estimators', 'max_depth') else 2
    filas_e.append([c, rg, f'{n(elegidos[c].min(), d)} a {n(elegidos[c].max(), d)}'])
tabla(s, 1.3, 3.3, 12.6, 4.9, filas_e, anchos=[3.6, 3.2, 3.4], tam=10)
texto(s, 1.3, 8.5, 12.6, 2.3, 'Con un semestre de entrenamiento gana un modelo de muchos árboles poco profundos; con '
      'más datos la profundidad sube a 7-10 y el submuestreo baja. Cerca del óptimo la superficie es plana: '
      'configuraciones distintas dan AUC a milésimas.', tam=10.5, color=GRIS)
banda(s, 1.3, 11.0, 12.6, 2.1, [('Reproducible:  ', True, VERDE),
                                (f'semilla 2026; un JSON por trabajo con parámetros y métricas; {n(horas)} horas de CPU '
                                 'en el servidor del grupo, 30 trabajos a la vez.', False, TINTA)], NIEBLA, tam=10.5)
tarjeta(s, 14.6, 3.3, 9.6, 4.8, 'Un solo pipeline de scikit-learn', ['preprocesador (log, imputación, one-hot, '
                                                                    'codificación por objetivo) → balanceo → modelo;',
                                                                    'el preprocesador se ajusta solo con el '
                                                                    'entrenamiento de cada ventana.'], VERDECL,
        tam_c=10.5)
tarjeta(s, 14.6, 8.4, 9.6, 4.7, 'Pruebas e integración continua', '57 pruebas con pytest (codificación fuera de '
        'pliegue, series, modelo servido, tablero, que no se publiquen identificadores) que corren en GitHub en cada '
        'cambio.', CREMA, tam_c=10.5)
pie(s, 'src/experimentos.py (ESPACIOS), resultados/experimentos/trabajos/*.json, tests/')
notas(s, '0:45', 'B', 'criterio 4', 'Por qué ese espacio: rangos usuales de la literatura, escala logarítmica donde el '
      'efecto es multiplicativo (tasa, peso mínimo). Decisiones computacionales: matrices preprocesadas en disco, SMOTE '
      'calculado una vez por ventana.')

# ============================================================== 15 TABLERO: LIDERES Y OPTIMIZADORES
s = nueva()
cabecera(s, 'Optuna llega antes; con 20 evaluaciones empatan los cuatro', 'CRITERIO 4  ·  PANTALLA EN VIVO: LAS 112')
figura(s, 'optimizadores.png', 1.2, 3.2, w=12.6)
filas_o = [['optimizador', 'AUC perdido a 5 evaluaciones', 'a 20']]
for o in ['optuna', 'deap', 'random', 'grid']:
    filas_o.append([OPT[o], n(cur.loc[5, o], 4), n(cur.loc[20, o], 4)])
tabla(s, 14.3, 3.4, 9.9, 3.3, filas_o, anchos=[2.6, 3.8, 1.8], tam=10.5)
en_vivo(s, 14.3, 7.1, 9.9, 6.0, ['Modelos → Las 112 combinaciones.', 'Tabla de líderes: las mejores combinaciones, con '
                                 'media ± desviación del bucle externo.',
                                 'Mapa balanceo × optimizador de XGBoost: el balanceo casi no mueve el AUC.'])
pie(s, f'tablero, página Modelos ({URL["tablero"]}); data/modelos/curvas_optimizador.csv')
notas(s, '0:45', 'A', 'criterio 4', 'Optuna (TPE) aprende de los ensayos y propone donde la mejora esperada es mayor; '
      'DEAP llega tarde pero llega. Con poco presupuesto conviene Optuna; con 20 la superficie es plana.')

# ============================================================== 16 METRICAS Y COMPARACION
s = nueva()
cabecera(s, f'XGBoost gana los 8 bloques: AUC {n(top.AUC, 3)} contra {n(mejor.iloc[1].AUC, 3)}',
         'CRITERIO 5  ·  MÉTRICAS, LÍNEA BASE Y COMPARACIÓN')
filas_r = [['modelo (mejor combinación)', 'AUC ± desv.', 'PR-AUC', 'F1', 'ms / 1.000']]
for _, r in mejor.iterrows():
    filas_r.append([f'{NOMBRE[r.modelo]} · {BAL[r.balanceo]} · {OPT[r.optimizador]}',
                    f'{n(r.AUC, 3)} ± {n(r.AUC_desv, 3)}', n(r.PR_AUC, 3), n(r.F1, 3), n(r.ms_inferencia_por_1000, 1)])
tabla(s, 1.3, 3.3, 13.6, 5.8, filas_r, anchos=[5.6, 2.6, 1.6, 1.4, 1.6], tam=10)
texto(s, 1.3, 9.2, 13.6, 1.0, 'AUC ordena por riesgo sin depender del umbral; PR-AUC atiende a la clase de interés; F1 '
      'mide la decisión al umbral; Brier y ECE dicen si la probabilidad es creíble.', tam=9.5, color=GRIS)
tabla(s, 1.3, 10.2, 13.6, 3.0, [['prueba 2026', 'AUC [BCa 95 %]', 'F1', 'ECE'],
                                ['regla del proveedor', n(rp['ROC-AUC'], 3), n(rp.F1, 3), n(rp.ECE, 3)],
                                ['regresión logística', n(lb['ROC-AUC'], 3), n(lb.F1, 3), n(lb.ECE, 3)],
                                ['XGBoost, con Platt', f"{n(x26['ROC-AUC'], 3)} [{n(x26['ROC-AUC inf'], 3)}; "
                                 f"{n(x26['ROC-AUC sup'], 3)}]", n(x26.F1, 3), n(x26.ECE, 3)]],
      anchos=[5.0, 3.6, 1.8, 1.8], tam=9.5)
tarjeta(s, 15.6, 3.3, 8.6, 6.8, 'Cómo leer la tabla', ['Media y desviación de los 8 bloques externos: cada cifra se '
                                                      'midió en un semestre que el modelo no vio.',
                                                      'La línea base son la regla del proveedor (su tasa previa de '
                                                      'retraso) y la logística, medidas en 2026.',
                                                      f'Frente a la logística, con el mismo 20 % de alertas: '
                                                      f'{n(cr26.loc["N: tardías de más", "por mes"])} tardías más por '
                                                      f'mes, {n(cr26.loc["M: de ellas, con stock en cero", "por mes"])} '
                                                      'de material sin stock.'], VERDECL, tam_c=10)
tarjeta(s, 15.6, 10.4, 8.6, 2.7, 'De la validación a 2026', f'AUC interno {n(b26.iloc[0], 3)}, bloques externos '
        f'{n(b26.iloc[1], 3)}, prueba 2026 {n(b26.iloc[3], 3)}: no cae.', VERDECL, tam_c=10)
pie(s, 'libro, capítulos 6 y 7: data/modelos/lideres.csv; resultados/prueba_2026/metricas.csv, brecha.csv, criticas.csv')
notas(s, '1:00', 'B', 'criterio 5', 'Por qué cada métrica. En los bloques la ganancia es de una centésima frente al '
      f'bosque y la logística. En 2026, medido una vez con lo congelado en 2025, XGBoost da {n(x26["ROC-AUC"], 3)} y le '
      f'saca {n(d26.diferencia, 3)} a la logística. La brecha no muestra caída de la validación a la prueba. Por qué '
      'XGBoost y no la logística: un retraso cuesta más que una falsa alarma, y con el mismo esfuerzo de revisión '
      f'atrapa {n(cr26.loc["N: tardías de más", "total"])} tardías más en 2026 (BCa por órdenes '
      f'[{n(cr26.loc["N: tardías de más", "BCa inf"])}; {n(cr26.loc["N: tardías de más", "BCa sup"])}]); las de material '
      'sin stock tienen un intervalo que roza el cero. Entrenar tarda minutos: se puede reentrenar cada mes.')

# ============================================================== 17 TABLERO: PREDICCION + CD
s = nueva()
wl = wil[wil.contra == 'logistica'].iloc[0]
rb, rl = (pru.set_index('rival').loc[m] for m in ('random_forest', 'logistica'))
cabecera(s, 'Tablero: una predicción en vivo, y la prueba de diferencias', 'CRITERIO 5  ·  PANTALLA EN VIVO')
captura(s, 'tablero_modelos.png', 1.3, 3.6, 10.8)
banda(s, 1.3, 11.0, 11.4, 2.1, [('En vivo (Modelos → Predicción):  ', True, ORO_CL), ('cargar una posición real, cambiar el plazo hábil al liberar '
                                 'y ver cómo se mueven el riesgo y sus contribuciones.', False, BLANCO)], VERDE, tam=10.5)
figura(s, 'rangos_cd.png', 13.6, 2.7, w=10.4)
texto(s, 13.4, 9.9, 10.8, 3.4, [(f'Friedman: χ² = {n(fri.chi2, 1)}, p < 0,001: los 7 modelos no rinden igual.',
                                  {'negrita': True}),
                                 (f'Nemenyi no separa a XGBoost de la logística ni del bosque (diferencia crítica '
                                  f'{n(fri.CD, 2)}); Wilcoxon por pares sí: gana los 8 bloques, p de Holm = '
                                  f'{n(wl.p_holm, 3)}.', {}),
                                 (f'Dentro de cada bloque, DeLong con Holm: supera al bosque en {rb.bloques_delong_holm} '
                                  f'de 8 y a la logística en {rl.bloques_delong_holm}, sin perder en ninguno. Ventaja '
                                  f'{n(rb.dif_AUC, 3)}-{n(rl.dif_AUC, 3)} de AUC, con BCa por órdenes lejos de cero.',
                                  {})], tam=10, espacio=3)
pie(s, f'tablero, página Modelos; data/modelos/friedman.csv, wilcoxon.csv; resultados/pruebas/resumen.csv')
notas(s, '1:00', 'A', 'criterio 5', 'En vivo: la predicción sirve el modelo original y, para comparar, XGBoost, los dos con '
      'Platt y contribuciones SHAP exactas. Pruebas: Friedman rechaza la igualdad; Nemenyi es conservador con 8 bloques; el Wilcoxon pareado '
      'con Holm separa porque XGBoost gana en cada bloque. DeLong mira dentro de cada bloque (validado contra pROC); como '
      'supone filas independientes, se lee con el BCa que remuestrea órdenes. Cliff con el bosque: pequeño (0,25).')

# ============================================================== 18 ERRORES Y SOBREAJUSTE
s = nueva()
gap = (xb.AUC_interno - xb.AUC).mean()
cabecera(s, f'Sobreajuste bajo: {n(gap, 3)} de AUC entre interno y externo',
         'CRITERIO 5  ·  ANÁLISIS DE ERRORES Y SOBREAJUSTE')
figura(s, 'interno_externo.png', 1.2, 3.2, w=11.0)
figura(s, 'aprendizaje.png', 12.6, 3.2, w=6.2)
figura(s, 'calibracion.png', 19.0, 3.2, w=5.3)
b25 = xb.set_index('bloque').loc['2025-S1']
texto(s, 1.3, 9.3, 11.0, 1.6, f'El bloque 2025-S1 baja de {n(b25.AUC_interno, 3)} adentro a {n(b25.AUC, 3)} afuera: es el '
      'semestre en que cambia la mezcla de 2025 (diapositiva 8), no un modelo que memoriza.', tam=10.5, color=GRIS)
texto(s, 12.6, 9.3, 11.6, 1.6, f'Logística (modelo base): {n(ca["AUC entrenamiento"], 3)} en entrenamiento y '
      f'{n(ca["AUC validación"], 3)} en validación con todas las filas; bien calibrada en 2026 (ECE {n(lb.ECE, 3)}).',
      tam=10.5, color=GRIS)
tardias = f26[('tarde', 'tarde')] + f26[('tarde', 'a tiempo')]
banda(s, 1.3, 11.1, 22.9, 1.7, [('Errores de XGBoost en 2026:  ', True, ORO_CL),
                                (f'de {n(tardias)} tardías marcó {n(f26[("tarde", "tarde")])} '
                                 f'({pc(f26[("tarde", "tarde")] / tardias, 0)}), con {n(f26[("a tiempo", "tarde")])} falsas '
                                 f'alarmas. AUC {n(s26.loc["todo conocido", "ROC-AUC"], 3)} con todo conocido y '
                                 f'{n(s26.loc["alguna nueva", "ROC-AUC"], 3)} con alguna entidad nueva; ganadería, '
                                 f'{n(s26.loc["ganaderia", "ROC-AUC"], 3)}. El falso negativo típico tenía plazo amplio y un '
                                 'proveedor puntual: el retraso se decide en la ejecución.', False, BLANCO)], VERDE, tam=10.5)
pie(s, 'libro, capítulos 6 y 7: resultados/experimentos/por_bloque.csv, resultados/prueba_2026/segmentos.csv')
notas(s, '0:45', 'B', 'criterio 5', 'Sobreajuste en dos vistas: interno contra externo por bloque y la curva de '
      'aprendizaje. En 2026, el error se concentra en las órdenes con entidades nuevas y en ganadería; el falso negativo '
      'típico no se podía ver al liberar.')

# ============================================================== 19 EL MODELO ORIGINAL
s = nueva()
cabecera(s, f'El modelo original le gana al óptimo de las 112 por {n(k26["ΔROC-AUC"], 3)} de AUC',
         'CRITERIO 6  ·  MODELO ORIGINAL (ENTREGABLE 3)')
texto(s, 1.3, 3.2, 14.5, 0.6, 'BOREAL: efectos por entidad sobre un boosting, con cortes ordenados', tam=12, color=VERDE,
      negrita=True)
figura(s, 'f_original.png', 1.4, 3.9, w=14.3)
texto(s, 1.3, 4.7, 14.5, 1.5, 'd_i: días de retraso; se predice en c = 1 (y_tarde) y aprende también de cuánto se atrasó. '
      'b_e: intercepto aleatorio por proveedor, material, solicitante y comprador, encogido hacia 0 (σ²_e por EM). Al entrenar, '
      'los árboles ven efectos estimados solo con los semestres anteriores, como al predecir.', tam=9.5, color=GRIS)
texto(s, 1.3, 6.2, 14.5, 0.6, 'f es un LightGBM, no un XGBoost', tam=12, color=VERDE, negrita=True)
texto(s, 1.3, 6.8, 14.5, 1.5, 'Misma suma de árboles, misma log-pérdida y mismo peso de hoja −G/(H + λ) (diapositiva 11). '
      'Difiere en cómo crece el árbol: por hoja, partiendo la de mayor ganancia hasta 154 hojas, y no por niveles '
      '(XGBoost: profundidad 10). Aquí toma Σ b_e como desplazamiento fijo del logit y es monótono decreciente en c.',
      tam=9.5, color=GRIS)
texto(s, 1.3, 8.4, 14.5, 0.6, 'Plazo log-normal y plazo real', tam=12, color=VERDE, negrita=True)
figura(s, 'f_lognormal.png', 1.4, 9.0, w=11.5)
texto(s, 1.3, 9.85, 14.5, 1.3, 'z: cuánto alcanza la promesa frente al plazo real esperado (un LightGBM sin la promesa). '
      'Más 32 variables de plazo real: mediana y percentil 80 de las últimas 20 entregas, y la holgura empírica.',
      tam=9.5, color=GRIS)
c26 = leer(R, 'prueba_2026', 'dos_por_dos.csv').set_index('celda')
q26 = leer(R, 'prueba_2026', 'contrastes.csv').set_index('contraste')
filas_o = [['2026, AUC [BCa 95 %]', '136 del dataset', '136 + 32 nuevas']]
for et, a, b in [('XGBoost', 'XGBoost 136', 'XGBoost 136 + nuevas'),
                 ('BOREAL sin log-normal', 'Boosting mixto 136 (ablación)', 'Boosting mixto 136 + nuevas (ablación)'),
                 ('Propuesta', 'BOREAL 136', 'BOREAL 136 + nuevas')]:
    filas_o.append([et] + [f'{n(c26.loc[k, "ROC-AUC"], 3)} [{n(c26.loc[k, "inf"], 3)}; {n(c26.loc[k, "sup"], 3)}]'
                           for k in (a, b)])
tabla(s, 16.2, 3.3, 8.0, 3.6, filas_o, anchos=[2.6, 2.7, 2.7], tam=9)


def _q(k):
    r = q26.loc[k]
    return f'{n(r["ΔROC-AUC"], 3)} [{n(r.inf, 3)}; {n(r.sup, 3)}]'


tarjeta(s, 16.2, 7.1, 8.0, 3.9, 'Lectura de la 2 × 2', [
    f'Con las 136, la estructura suma {_q("modelo, con las 136")}.',
    f'El plazo real le da a XGBoost {_q("información, en XGBoost")}; con él, la ventaja de la estructura baja a '
    f'{n(q26.loc["modelo, con 136 + nuevas", "ΔROC-AUC"], 3)}, sin significancia.',
    f'El log-normal suma {n(q26.loc["log-normal, con las 136", "ΔROC-AUC"], 3)}; su intervalo toca el cero.'],
        VERDECL, tam_c=9)
banda(s, 1.3, 11.2, 22.9, 1.9, [('En el tablero:  ', True, ORO_CL),
                                ('la pestaña Predicción lo sirve con posiciones reales de 2026, sin códigos de entidad. Es la '
                                 'tercera mirada a 2026: exploratoria. En las órdenes con alguna entidad nueva no gana: '
                                 'el arranque en frío sigue abierto.', False, BLANCO)], VERDE, tam=10.5)
pie(s, 'libro, capítulo 9: resultados/prueba_2026/dos_por_dos.csv y contrastes.csv; tablero: Modelos → Modelo original')
notas(s, '1:00', 'A', 'criterio 6', 'Qué es la propuesta y qué aporta cada parte. La 2 × 2 separa modelo de información: '
      'los efectos por entidad y el plazo real son dos caminos a la misma ganancia. Decir que es la tercera mirada a '
      '2026 y que por eso se lee como exploratoria.')

# ============================================================== 20 LA PROPUESTA EN OTROS DATOS
s = nueva()
cabecera(s, 'En otros datos, la propuesta gana donde hay historia por entidad', 'CRITERIO 6  ·  VALIDACIÓN EN CONJUNTOS PÚBLICOS')
NPUB = {'airlines': 'Vuelos (Airlines)', 'amazon': 'Acceso (Amazon)', 'click': 'Publicidad (Click)',
        'nyc_bus': 'Buses de Nueva York', 'zurich': 'Transporte de Zúrich', 'diabetes': 'Reingreso (Diabetes130US)',
        'noshow': 'Citas no asistidas'}
filas_p = [['conjunto', 'AUC XGBoost', 'AUC boosting mixto', 'Δ [IC 95 %]', 'p Holm', 'veredicto']]
for _, r in pub.iterrows():
    filas_p.append([NPUB[r.conjunto], n(r.auc_xgb, 3), n(r.auc_k3, 3), f'{n(r.dif, 3)} [{n(r.ic_lo, 3)}; {n(r.ic_hi, 3)}]',
                    '< 0,001' if r.p_holm < 0.001 else n(r.p_holm, 3), r.veredicto])
tabla(s, 1.3, 3.3, 14.4, 7.2, filas_p, anchos=[4.4, 1.8, 1.6, 3.6, 1.4, 1.8], tam=9.5)
texto(s, 1.3, 10.7, 14.4, 2.4, 'Mismo protocolo para los dos: Optuna con 20 evaluaciones en validación, una medida en la '
      'prueba, DeLong pareado con Holm sobre los siete. La segunda ronda (filas 4 a 7) se eligió con un criterio fijado '
      'antes de correr. El plazo log-normal y las variables nuevas no aplican fuera de OTIF.', tam=10, color=GRIS)
tarjeta(s, 16.3, 3.3, 7.9, 6.4, 'Lectura', [f'Gana {(pub.veredicto == "boosting mixto gana").sum()}, empata '
                                           f'{(pub.veredicto == "empate").sum()}, pierde '
                                           f'{(pub.veredicto == "boosting mixto pierde").sum()}; Wilcoxon p = 0,81: no hay '
                                           'superioridad general.',
                                           'Gana donde la entidad vuelve (el paciente), como el proveedor en OTIF.',
                                           'Empata en transporte; pierde sin tiempo ni desenlace continuo.'],
        VERDECL, tam_c=10)
banda(s, 16.3, 10.0, 7.9, 3.1, [('Cuidado:  ', True, ORO_CL), ('esa separación se vio después de la segunda ronda; '
                                'confirmarla pide una tercera fijada antes de correr.', False, BLANCO)], VERDE, tam=10.5)
pie(s, 'libro, capítulo 10: original/publicos_otif_resumen.csv (OpenML); tablero: Modelos → Modelo original')
notas(s, '1:00', 'A', 'criterio 6', 'Responde «¿y en otros datos?». La ganancia de la propuesta en OTIF (0,012) depende de '
      'rasgos del caso: entidades con historia en el tiempo, desenlace continuo y plazo prometido. Airlines no trae el '
      'año: su partición es por mes. No decir que generaliza: es una hipótesis con apoyo.')

# ============================================================== 20 CONCLUSIONES
s = nueva()
cabecera(s, 'El retraso se puede anticipar el día de la liberación', 'CRITERIO 6  ·  CONCLUSIONES')
lista(s, 1.3, 3.4, 13.4, [
    f'Se puede ordenar por riesgo al liberar: AUC {n(top.AUC, 3)} medido hacia adelante en el tiempo, contra '
    f'{n(rp["ROC-AUC"], 3)} de la regla del proveedor en 2026.',
    'La señal está en el plazo, el historial del proveedor y las entidades; ninguna variable sola basta.',
    f'XGBoost gana de forma significativa y material: en 2026, AUC {n(x26["ROC-AUC"], 3)} contra '
    f'{n(lb["ROC-AUC"], 3)} de la logística, y más tardías atrapadas con las mismas alertas.',
    'El optimizador importa con poco presupuesto; el balanceo casi no mueve el AUC.',
    'La partición cronológica no es un detalle: al azar, el problema parecería más fácil de lo que es.'],
      tam=11.5, paso=1.9, alto=1.8)
filas_v = [['objetivo', 'cumplido'], ['comparar las 112 con el mismo presupuesto', 'sí'],
           ['costo de las entidades nuevas', f'AUC {n(s26.loc["alguna nueva", "ROC-AUC"], 2)} contra '
            f'{n(s26.loc["todo conocido", "ROC-AUC"], 2)}'], ['modelo servido en el tablero', 'sí, el original (y XGBoost)'],
           ['modelo original contra el óptimo', f'+{n(k26["ΔROC-AUC"], 3)} de AUC [{n(k26.inf, 3)}; {n(k26.sup, 3)}]']]
tabla(s, 15.4, 3.4, 8.8, 4.8, filas_v, anchos=[4.4, 3.6], tam=10.5)
banda(s, 15.4, 8.6, 8.8, 4.5, [('Para el negocio:  ', True, ORO_CL),
                               (f'marcando el {pc(x26.marcadas, 0)} de las posiciones, XGBoost ve venir el '
                                f'{pc(x26.sensibilidad, 0)} de las tardías de 2026. Por órdenes: siguiendo el '
                                f'{pc(od26.loc[0.69, "fracción de las órdenes"], 0)} se ve venir el 69 % de las órdenes '
                                'con retraso.', False, BLANCO)], VERDE, tam=11.5)
pie(s, 'libro, capítulos 2 a 10')
notas(s, '1:00', 'A', 'criterio 6', 'Responder la pregunta con cifras y el veredicto de cada objetivo. El modelo original '
      'del Entregable 3 le gana al óptimo de las 112 por 0,012 en 2026 (pestaña Modelo original del tablero); la pestaña '
      'Predicción lo sirve en vivo, con XGBoost al lado para comparar.')

# ============================================================== 21 LIMITACIONES Y MEJORAS
s = nueva()
cabecera(s, 'Límites: deriva, etiqueta que madura y una sola empresa', 'CRITERIO 6  ·  LIMITACIONES Y MEJORAS')
tarjeta(s, 1.3, 3.3, 7.4, 9.8, 'Teóricas', ['Independencia dado x: las posiciones de una orden se parecen; se protege '
                                           'agrupando, no se modela.',
                                           'Estabilidad: el modelo supone que la relación de 2021-2025 vale en 2026.',
                                           'Una sola etiqueta binaria desperdicia los días de retraso.'], VERDECL,
        tam_t=12.5, tam_c=10.5)
tarjeta(s, 9.0, 3.3, 7.4, 9.8, 'Empíricas', ['Deriva: 2025 se distingue de 2021-2024 casi por completo.',
                                            'La etiqueta de lo más reciente madura tarde (censura al final del '
                                            'periodo).',
                                            'Maestros en foto actual, no con su vigencia histórica.',
                                            f'Arranque en frío: {n(nue.loc["proveedor", "nuevo en prueba (%)"], 1)} % '
                                            'de proveedores nuevos en 2026.',
                                            'Una sola empresa: la generalización a otros grupos no está probada.'],
        CREMA, tam_t=12.5, tam_c=10.5)
tarjeta(s, 16.7, 3.3, 7.5, 9.8, 'Mejoras', ['Reentrenar por semestre y vigilar la deriva con la validación '
                                           'adversaria.',
                                           'Efectos aleatorios por proveedor y material para el arranque en frío.',
                                           'Usar los días de retraso con cortes ordenados (el modelo original del '
                                           'Entregable 3).',
                                           'Maestros con vigencia por fecha.'], NIEBLA, tam_t=12.5, tam_c=10.5)
pie(s, 'libro, capítulos 4, 6 y 9')
notas(s, '1:00', 'B', 'criterio 6', 'Separar lo teórico (supuestos del modelo) de lo empírico (los datos). Las mejoras '
      'apuntan al modelo original.')

# ============================================================== 22 PREGUNTAS
s = nueva(VERDE)
logo(s, 1.3, 1.4, 3.9, 3.0)
texto(s, 1.3, 5.6, 22.0, 1.6, 'Preguntas', tam=40, color=BLANCO, negrita=True)
filete(s, 1.3, 7.4, 2.3)
texto(s, 1.3, 8.0, 22.0, 3.0, [f'Repositorio   {URL["repo"]}', f'Libro            {URL["libro"]}',
                               f'Tablero         {URL["tablero"]}'], tam=14, color=PALIDO, espacio=6)
texto(s, 1.3, 12.2, 22.0, 0.8, 'Responden los dos autores (criterio 7: dominio individual).', tam=11, color=ORO_CL)
notas(s, '2:00', 'A y B', 'criterio 7', 'Preparar: derivar g, h y w*; por qué la codificación va fuera de pliegue; por '
      'qué la prueba no elige nada; por qué Nemenyi y Wilcoxon dicen cosas distintas; qué pasa con un proveedor nuevo.')

assert _k[0] == N_TOTAL - 1, _k[0]
prs.save(SAL)
with open(os.path.join(AQUI, 'notas.md'), 'w', encoding='utf-8') as f:
    f.write('# Notas del expositor\n\nLas escribe `construir.py` (también van en las notas de cada diapositiva). '
            'A = Daniel Carbonó, B = Luis Pérez; cada uno ensaya también lo del otro. Rúbrica de 20 minutos: 18 de '
            'exposición y 2 de preguntas.\n\n| # | min | criterio | expone | qué decir |\n|---|---|---|---|---|\n')
    for k, (m, e, c, t) in enumerate(NOTAS, 1):
        f.write(f'| {k} | {m} | {c} | {e} | {t} |\n')
seg = sum(int(m.split(':')[0]) * 60 + int(m.split(':')[1]) for m, *_ in NOTAS[:-1])
print(f'{len(prs.slides)} diapositivas en {SAL}; exposición {seg // 60}:{seg % 60:02d} más 2:00 de preguntas')
