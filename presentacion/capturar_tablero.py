# -*- coding: utf-8 -*-
"""Captura las tres paginas del tablero publicado en Render (Contexto, EDA y Modelos con la prediccion en vivo) con
Edge sin ventana, para las diapositivas de tablero y como respaldo si Render falla el dia de la sustentacion.

Uso: python presentacion/capturar_tablero.py      (abre antes la URL en un navegador si Render esta dormido)
Salida: presentacion/img/tablero_{contexto,eda,modelos}.png
"""
import os
import subprocess

AQUI = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(AQUI, 'img')
URL = 'https://otif-compras-ml.onrender.com'
EDGE = next(p for p in (r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe') if os.path.exists(p))
PERFIL = os.path.join(os.environ.get('TEMP', '.'), 'edge_captura_otif')   # perfil aparte: funciona con Edge abierto

for nombre, ruta in [('contexto', '/'), ('eda', '/eda'), ('modelos', '/modelos')]:
    salida = os.path.join(IMG, f'tablero_{nombre}.png')
    subprocess.run([EDGE, '--headless', '--disable-gpu', f'--user-data-dir={PERFIL}', '--hide-scrollbars',
                    '--window-size=1600,1000', '--virtual-time-budget=25000', f'--screenshot={salida}', URL + ruta],
                   check=True, timeout=240, capture_output=True)
    print(salida, os.path.getsize(salida))
