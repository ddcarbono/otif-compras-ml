#!/usr/bin/env bash
# Puntajes por posicion de la mejor combinacion de cada modelo en los 8 bloques (para DeLong y BCa). Reajusta 56 modelos
# con los hiperparametros ya elegidos; unos 45 min de CPU. Usa la cache de preparar.sh y los JSON de correr.sh.
#   nohup srvgdai/puntajes.sh > puntajes.log 2>&1 &
# Despues, desde la maquina local: scp -r srvgdai:~/estudio_otif/otif-compras-ml/resultados/puntajes resultados/
set -e
cd ~/estudio_otif/otif-compras-ml
export OTIF_CSV=~/estudio_otif/otif_depurado.csv PYTHONWARNINGS=ignore
nice -n 19 ~/estudio_otif/venv/bin/python -m src.puntajes --trabajos 8
