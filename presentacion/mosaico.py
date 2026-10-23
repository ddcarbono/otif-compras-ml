# -*- coding: utf-8 -*-
# Junta las diapositivas exportadas en una hoja de contactos para revisarlas de
# una sola vez.  python mosaico.py <carpeta_con_PNG> <salida.png> [columnas]
import os, re, sys, glob
from PIL import Image

src = sys.argv[1]
out = sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 4

fs = glob.glob(os.path.join(src, '**', '*.PNG'), recursive=True)
vistos, unicos = set(), []
for f in sorted(fs, key=lambda x: int(re.search(r'(\d+)\.PNG$', x, re.I).group(1))):
    n = int(re.search(r'(\d+)\.PNG$', f, re.I).group(1))
    if n in vistos:
        continue
    vistos.add(n); unicos.append((n, f))

im0 = Image.open(unicos[0][1])
w = 620
h = int(round(w * im0.height / im0.width))
filas = -(-len(unicos) // cols)
hoja = Image.new('RGB', (cols * w + (cols + 1) * 6, filas * h + (filas + 1) * 6),
                 (210, 210, 210))
for k, (n, f) in enumerate(unicos):
    im = Image.open(f).convert('RGB').resize((w, h), Image.LANCZOS)
    x = 6 + (k % cols) * (w + 6)
    y = 6 + (k // cols) * (h + 6)
    hoja.paste(im, (x, y))
hoja.save(out)
print(len(unicos), 'diapositivas ->', out, hoja.size)
