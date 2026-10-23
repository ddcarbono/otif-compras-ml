#!/usr/bin/env bash
# Matrices preprocesadas, SMOTE y ADASYN de los 7 bloques que no son el del piloto (2025-S2 lo prepara piloto.sh).
#   nohup srvgdai/preparar.sh > preparar.log 2>&1 &
set -e
cd ~/estudio_otif/otif-compras-ml
export OTIF_CSV=~/estudio_otif/otif_depurado.csv PYTHONWARNINGS=ignore
nice -n 19 ~/estudio_otif/venv/bin/python -m src.experimentos preparar --trabajos 8 --bloques 2022-S1 2022-S2 2023-S1 2023-S2 2024-S1 2024-S2 2025-S1
