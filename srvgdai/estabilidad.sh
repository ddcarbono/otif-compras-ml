#!/usr/bin/env bash
# Estabilidad ante la semilla (4 modelos x 8 bloques x 10 semillas) y curva de convergencia de XGBoost en los 8 bloques.
# Usa la cache de preparar.sh y los JSON de correr.sh; no toca la prueba 2026.
#   nohup srvgdai/estabilidad.sh > estabilidad.log 2>&1 &
# Despues, desde la maquina local:
#   scp -i "$HOME/.ssh/id_ed25519_srvgdai" "dcarbono@10.0.0.91:~/estudio_otif/otif-compras-ml/resultados/estabilidad/*.csv" resultados/estabilidad/
set -e
cd ~/estudio_otif/otif-compras-ml
export OTIF_CSV=~/estudio_otif/otif_depurado.csv PYTHONWARNINGS=ignore
nice -n 19 ~/estudio_otif/venv/bin/python -m src.estabilidad --trabajos 8
