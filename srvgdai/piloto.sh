#!/usr/bin/env bash
# Piloto de los 112 en srvgdai: prepara el bloque 2025-S2 (el mas grande) y corre un trabajo por modelo (SMOTE, busqueda
# aleatoria, E=2) para medir segundos por evaluacion y pico de memoria. En el servidor:
#   nohup srvgdai/piloto.sh > piloto.log 2>&1 &
set -e
cd ~/estudio_otif/otif-compras-ml
export OTIF_CSV=~/estudio_otif/otif_depurado.csv PYTHONWARNINGS=ignore
P=~/estudio_otif/venv/bin/python
nice -n 19 $P -m src.experimentos preparar --bloques 2025-S2 --trabajos 4
nice -n 19 $P -m src.experimentos piloto --E 2 --trabajos 7
