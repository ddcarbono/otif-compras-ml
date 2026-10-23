# -*- coding: utf-8 -*-
"""Guarda presentacion/sustentacion_otif.pptx como PDF con PowerPoint (COM, sin ventana).

Uso (Python base, que tiene pywin32): python presentacion/generar_pdf.py
Salida: presentacion/sustentacion_otif.pdf
"""
import os

import win32com.client

AQUI = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(AQUI, 'sustentacion_otif.pptx')
PDF = os.path.join(AQUI, 'sustentacion_otif.pdf')

app = win32com.client.Dispatch('PowerPoint.Application')
pres = app.Presentations.Open(PPTX, ReadOnly=True, WithWindow=False)
try:
    pres.SaveAs(PDF, 32)   # 32 = ppSaveAsPDF
finally:
    pres.Close()
    app.Quit()
print(PDF, os.path.getsize(PDF), 'bytes')
