#!/usr/bin/env bash
# Los 896 trabajos (112 combinaciones x 8 bloques) con E = 20 evaluaciones por optimizador (decision de Daniel,
# 2026-10-22), 30 a la vez y con nice -n 19 (en srvgdai corre el monitor de DAABON AI). Reanudable: si se corta, se
# vuelve a lanzar y salta los trabajos que ya tienen su JSON. Al final escribe resultados/tabla_maestra.csv.
#   nohup srvgdai/correr.sh > correr.log 2>&1 &
set -e
cd ~/estudio_otif/otif-compras-ml
export OTIF_CSV=~/estudio_otif/otif_depurado.csv PYTHONWARNINGS=ignore
P=~/estudio_otif/venv/bin/python
while pgrep -f preparar.sh > /dev/null; do sleep 60; done   # los trabajos leen las matrices que deja preparar.sh
nice -n 19 $P -m src.experimentos correr --E 20 --trabajos 30
$P -m src.experimentos tabla
