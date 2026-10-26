#!/usr/bin/env bash
# Trae de srvgdai lo que dejaron las corridas de BOREAL (boreal.sh y la fase prueba de evaluar_2x2_otif.py).
#   bash srvgdai/traer_boreal.sh config     las cuatro configuraciones congeladas en la validacion 2025 -> original/config_2x2/
#                                           (y la de prop_nuevas a resultados/prueba_2026/config_prop_nuevas.json):
#                                           se hace commit de esto ANTES de aplicar Platt a 2026
#   bash srvgdai/traer_boreal.sh prueba     corre la fase platt de cada celda en el servidor y trae <celda>.csv y las
#                                           piezas de prop_nuevas a <datos>/2x2/ (no viajan: traen codigos)
#   bash srvgdai/traer_boreal.sh publicos   trae publicos_otif_*boreal_* a original/
# Uso, desde la raiz del repositorio.
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=(ssh -i "$HOME/.ssh/id_ed25519_srvgdai" -o BatchMode=yes dcarbono@10.0.0.91)
DATOS="/c/Proyectos/estudio/datos/otif"
CELDAS="prop136 prop_nuevas k3_136 k3_nuevas"
case "$1" in
  config)
    for c in $CELDAS; do
      "${SSH[@]}" "cat ~/estudio_otif/boreal/original/config_2x2/$c.json" > "original/config_2x2/$c.json"
    done
    cp original/config_2x2/prop_nuevas.json resultados/prueba_2026/config_prop_nuevas.json
    grep -h '"auc_validacion"' original/config_2x2/{prop136,prop_nuevas,k3_136,k3_nuevas}.json ;;
  prueba)
    "${SSH[@]}" "cd ~/estudio_otif/boreal/original && for c in $CELDAS; do ~/estudio_otif/venv/bin/python evaluar_2x2_otif.py ~/estudio_otif/otif_depurado.csv --fase platt --celda \$c; done"
    mkdir -p "$DATOS/2x2"
    "${SSH[@]}" "cd ~/estudio_otif/2x2 && tar -cf - prop136.csv prop_nuevas.csv k3_136.csv k3_nuevas.csv prop_nuevas_piezas.joblib" \
      | tar -C "$DATOS/2x2" -xf -
    ls -la "$DATOS/2x2" ;;
  publicos)
    "${SSH[@]}" "cd ~/estudio_otif/boreal/original && tar -cf - publicos_otif_*boreal_*" | tar -C original -xf -
    ls original/publicos_otif_*boreal_* ;;
esac
