# -*- coding: utf-8 -*-
# Exporta cada diapositiva de un .pptx a PNG usando PowerPoint por COM.
#   python exportar.py <archivo.pptx> <carpeta_destino> [ancho_px]
import os, sys, glob
import win32com.client

src = os.path.abspath(sys.argv[1])
dst = os.path.abspath(sys.argv[2])
w   = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
os.makedirs(dst, exist_ok=True)
for f in glob.glob(os.path.join(dst, '*.png')):
    os.remove(f)

app = win32com.client.Dispatch('PowerPoint.Application')
pres = app.Presentations.Open(src, ReadOnly=True, WithWindow=False)
try:
    h = int(round(w * pres.PageSetup.SlideHeight / pres.PageSetup.SlideWidth))
    pres.SaveCopyAs(os.path.join(dst, 'd.png'), 18)   # 18 = ppSaveAsPNG
finally:
    pres.Close()
    app.Quit()

hechos = sorted(glob.glob(os.path.join(dst, '**', '*.PNG'), recursive=True) +
                glob.glob(os.path.join(dst, '**', '*.png'), recursive=True))
print(len(hechos), 'imagenes en', dst)
for f in hechos[:40]:
    print('  ', os.path.relpath(f, dst), os.path.getsize(f))
