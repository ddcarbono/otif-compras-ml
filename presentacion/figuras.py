# -*- coding: utf-8 -*-
"""Figuras y formulas de la sustentacion, en PNG, con los colores de la plantilla Uninorte.

Entrada: resultados/eda/*.csv, resultados/similitud/auc.csv, resultados/experimentos/*.csv y data/modelos/*.csv (las
escriben el libro y los scripts de src/; aqui no se calcula ningun modelo).
Salida:  presentacion/img/*.png (figuras) y presentacion/img/f_*.png (formulas, con mathtext de matplotlib).
Uso:     python presentacion/figuras.py      (con ml_venv: el Python base no tiene matplotlib)
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
IMG = os.path.join(AQUI, 'img')
R = os.path.join(RAIZ, 'resultados')
M = os.path.join(RAIZ, 'data', 'modelos')
sys.path.insert(0, RAIZ)
from src.plots import MODELO, OPTIMIZADOR  # noqa: E402

VERDE, ORO, GRIS, TINTA, VERDECL, TEJA = '#405731', '#B08A2D', '#8C8C8C', '#1A1A1A', '#E8F0EB', '#8C4A2D'
AZUL = '#4A6FA5'
plt.rcParams.update({'font.family': 'Calibri', 'font.size': 15, 'axes.edgecolor': '#BBBBBB', 'axes.labelcolor': TINTA,
                     'xtick.color': '#444444', 'ytick.color': '#444444', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 16, 'axes.titleweight': 'bold',
                     'axes.titlecolor': VERDE, 'axes.titlelocation': 'left', 'mathtext.fontset': 'cm'})
os.makedirs(IMG, exist_ok=True)


def coma(v, d=3):
    return f'{v:.{d}f}'.replace('.', ',')


def guardar(fig, nombre):
    fig.savefig(os.path.join(IMG, nombre), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def formula(nombre, tex, tam=26):
    """Una formula en PNG transparente; la altura sale del contenido (bbox ajustado)."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f'${tex}$', fontsize=tam, color=TINTA)
    fig.savefig(os.path.join(IMG, nombre), dpi=250, bbox_inches='tight', pad_inches=0.04, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------- formulas
FORMULAS = {
    'f_x': r'x_i \in \mathbb{R}^{136},\quad y_i \in \{0,1\}',
    'f_p': r'p(x) = P(y = 1 \mid x),\qquad \hat y = \mathbb{1}\{\hat p(x) \geq \tau\}',
    'f_te': r'\hat p(v) = \frac{\mathrm{retrasos}(v) + M\,\hat p(\mathrm{padre}(v))}{n(v) + M},\quad M = 20',
    'f_modelo': r'F(x) = \sum_{m=1}^{M} f_m(x),\qquad \hat p(x) = \sigma(F(x)) = \frac{1}{1+e^{-F(x)}}',
    'f_obj': r'\mathcal{L}^{(m)} = \sum_{i=1}^{n} \ell\left(y_i,\, F_{m-1}(x_i) + f(x_i)\right) + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2',
    'f_perdida': r'\ell(y, F) = -y \log \sigma(F) - (1-y)\log(1 - \sigma(F))',
    'f_gh': r'g_i = \hat p_i - y_i,\qquad h_i = \hat p_i\,(1-\hat p_i)',
    'f_taylor': r'\mathcal{L}^{(m)} \approx \sum_i \left[g_i f(x_i) + \frac{1}{2} h_i f(x_i)^2\right] + \gamma T + \frac{1}{2}\lambda\sum_j w_j^2',
    'f_w': r'w_j^{*} = -\frac{G_j}{H_j + \lambda}',
    'f_ganancia': r'\mathrm{ganancia} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma',
    'f_eta': r'F_m = F_{m-1} + \eta\, f_m',
    'f_logit': r'\min_{\beta}\ \sum_i \ell(y_i,\, x_i^{\top}\beta) + \frac{1}{2C}\,\Vert\beta\Vert^2',
    'f_112': r'7 \times 4 \times 4 = 112\ \mathrm{combinaciones} \times 8\ \mathrm{bloques} = 896\ \mathrm{trabajos}',
}
for k, v in FORMULAS.items():
    formula(k + '.png', v)

# ---------------------------------------------------------------- datos y EDA
sm = pd.read_csv(os.path.join(R, 'eda', 'serie_mensual.csv'), parse_dates=['fecha_pedido'])
fig, ax = plt.subplots(figsize=(8.4, 3.9))
ax2 = ax.twinx()
ax2.bar(sm.fecha_pedido, sm.filas, width=25, color=VERDECL, zorder=0)
ax2.set_ylabel('posiciones por mes', color=GRIS)
ax2.spines['top'].set_visible(False)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
ax.plot(sm.fecha_pedido, 100 * sm.tasa, color=VERDE, lw=2.4, marker='o', ms=3)
ax.set_ylabel('% de posiciones tarde')
ax.set_title('Tasa de tarde por mes de creación, 2021-2024')
guardar(fig, 'serie.png')

aso = pd.read_csv(os.path.join(R, 'eda', 'asociacion.csv')).sort_values('AUC', ascending=False).head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.barh(aso.nombre, aso.AUC, color=[ORO if v >= 0.7 else VERDE for v in aso.AUC])
ax.axvline(0.75, color=TEJA, ls='--', lw=1.4)
ax.text(0.751, -0.9, '0,75', color=TEJA, fontsize=13)
ax.set_xlim(0.5, 0.78)
for i, v in enumerate(aso.AUC):
    ax.text(v + 0.003, i, coma(v), va='center', fontsize=12)
ax.set_xlabel('AUC univariado (la variable sola)')
ax.set_title('Las 12 variables que más separan solas')
guardar(fig, 'auc_univariado.png')

adv = pd.read_csv(os.path.join(R, 'similitud', 'auc.csv'))
etq = ['2021-2024 contra 2025', '…sin variables de reloj', 'órdenes al azar (80/20)']
fig, ax = plt.subplots(figsize=(6.6, 2.9))
ax.barh(etq[::-1], adv.AUC[::-1], color=[GRIS, ORO, VERDE])
ax.axvline(0.5, color=TINTA, lw=1, ls=':')
for i, v in enumerate(adv.AUC[::-1]):
    ax.text(v + 0.01, i, coma(v), va='center', fontsize=14, fontweight='bold')
ax.set_xlim(0.4, 1.12)
ax.set_xlabel('AUC del clasificador que adivina el conjunto')
ax.set_title('Validación adversaria')
guardar(fig, 'adversaria.png')

# ---------------------------------------------------------------- los 112
cur = pd.read_csv(os.path.join(M, 'curvas_optimizador.csv'))
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for o, c in zip(['grid', 'random', 'optuna', 'deap'], [GRIS, AZUL, ORO, VERDE]):
    d = cur[cur.optimizador == o]
    ax.plot(d['eval'], d['mean'], color=c, lw=2.6, label=OPTIMIZADOR[o])
ax.set_yscale('log')
ax.set_xlabel('evaluaciones del bucle interno')
ax.set_ylabel('AUC perdido frente al mejor conocido')
ax.set_title('Cuánto le falta a cada optimizador (media de los 112 × 8)')
ax.legend(frameon=False, fontsize=13)
guardar(fig, 'optimizadores.png')

ran = pd.read_csv(os.path.join(M, 'rangos.csv')).sort_values('rango_medio')
cd = float(pd.read_csv(os.path.join(M, 'friedman.csv')).CD.iloc[0])
fig, ax = plt.subplots(figsize=(5.8, 3.9))
y = np.arange(len(ran))[::-1]
x0 = ran.rango_medio.iloc[0]
ax.axvspan(x0, x0 + cd, color=ORO, alpha=0.15, lw=0)
ax.scatter(ran.rango_medio, y, s=110, color=[ORO if m == 'xgboost' else VERDE for m in ran.modelo], zorder=3)
for xi, yi in zip(ran.rango_medio, y):
    ax.text(xi + 0.12, yi, coma(xi, 2), va='center', fontsize=13)
ax.set_yticks(y)
ax.set_yticklabels([MODELO[m] for m in ran.modelo])
ax.set_xlim(0.5, 7.9)
ax.set_xlabel('rango medio en los 8 bloques (1 = mejor)')
ax.text(x0 + cd + 0.08, 4.6, 'diferencia\ncrítica ' + coma(cd, 2), ha='left', va='center', color=ORO, fontsize=13)
ax.set_title('Friedman y Nemenyi (α = 0,05)')
guardar(fig, 'rangos_cd.png')

pb = pd.read_csv(os.path.join(R, 'experimentos', 'por_bloque.csv'))
x = pb[(pb.modelo == 'xgboost') & (pb.balanceo == 'ninguno') & (pb.optimizador == 'optuna')]
fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.plot(x.bloque, x.AUC_interno, color=GRIS, lw=2.2, marker='o', label='interno (eligió los hiperparámetros)')
ax.plot(x.bloque, x.AUC, color=ORO, lw=2.6, marker='o', label='externo (semestre siguiente, no visto)')
ax.set_ylabel('ROC-AUC')
ax.tick_params(axis='x', labelsize=12)
ax.set_title('XGBoost: AUC interno contra externo por bloque')
ax.legend(frameon=False, fontsize=12, loc='lower right')
guardar(fig, 'interno_externo.png')

ca = pd.read_csv(os.path.join(M, 'curva_aprendizaje.csv'))
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(ca.filas, ca['AUC entrenamiento'], color=ORO, lw=2.4, marker='o', label='entrenamiento')
ax.plot(ca.filas, ca['AUC validación'], color=VERDE, lw=2.4, marker='o', label='validación 2025')
ax.set_xscale('log')
ax.set_xlabel('filas de entrenamiento')
ax.set_ylabel('ROC-AUC')
ax.set_title('Curva de aprendizaje (logística)')
ax.legend(frameon=False, fontsize=12)
guardar(fig, 'aprendizaje.png')

cal = pd.read_csv(os.path.join(M, 'calibracion.csv'))
fig, ax = plt.subplots(figsize=(4.4, 4.0))
ax.plot([0, 1], [0, 1], color=GRIS, ls=':')
ax.plot(cal.predicha, cal.observada, color=VERDE, lw=2.4, marker='o')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('probabilidad predicha')
ax.set_ylabel('tasa observada')
ax.set_title('Calibración, prueba 2026')
guardar(fig, 'calibracion.png')

cx = pd.read_csv(os.path.join(RAIZ, 'resultados', 'complejidad.csv'))
fig, ax = plt.subplots(figsize=(5.8, 3.8))
for m, c in [('xgboost', ORO), ('random_forest', VERDE), ('logistica', AZUL), ('knn', GRIS)]:
    d = cx[cx.modelo == m]
    if len(d):
        ax.plot(d.filas, d.segundos_ajuste, color=c, lw=2.4, marker='o', label=MODELO[m])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('filas de entrenamiento')
ax.set_ylabel('segundos de ajuste')
ax.set_title('Costo de ajuste medido (log-log)')
ax.legend(frameon=False, fontsize=11)
guardar(fig, 'complejidad.png')

print('figuras en', IMG, ':', len([f for f in os.listdir(IMG) if f.endswith('.png')]))
