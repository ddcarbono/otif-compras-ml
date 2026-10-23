"""Imprime presentacion/mazo.html a presentacion/sustentacion_e2.pdf con Edge sin ventana a 1920 x 1080.

Entrada: presentacion/mazo.html (lo escribe armar.py; Plotly se baja de cdn.plot.ly, hace falta internet).
Salida:  presentacion/sustentacion_e2.pdf
Uso:     python presentacion/generar_pdf.py
"""
import os, subprocess, time, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(AQUI, 'mazo.html')
PDF = os.path.join(AQUI, 'sustentacion_e2.pdf')
EDGE = next((p for p in (r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                         r'C:\Program Files\Microsoft\Edge\Application\msedge.exe') if os.path.exists(p)), None)
if not EDGE:
    sys.exit('No se encontro Edge.')
# perfil aparte: con Edge abierto, sin el no escribe el PDF
perfil = os.path.join(os.environ.get('TEMP', '.'), 'edge_pdf_otif')
if os.path.exists(PDF):
    os.remove(PDF)
subprocess.run([EDGE, '--headless', '--disable-gpu', f'--user-data-dir={perfil}', '--no-pdf-header-footer',
                '--window-size=1920,1080', '--virtual-time-budget=20000', f'--print-to-pdf={PDF}',
                'file:///' + HTML.replace('\\', '/')], check=True, timeout=180, capture_output=True)
t0, ult = time.time(), -1
while time.time() - t0 < 60:  # Edge puede devolver antes de terminar de escribir
    tam = os.path.getsize(PDF) if os.path.exists(PDF) else -1
    if tam > 0 and tam == ult:
        break
    ult = tam
    time.sleep(1)
if not os.path.exists(PDF):
    sys.exit('Edge no escribio el PDF.')
print('PDF:', PDF, os.path.getsize(PDF), 'bytes')
