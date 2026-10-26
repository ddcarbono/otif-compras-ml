#!/usr/bin/env bash
# Corre BOREAL (boosting propio) en srvgdai, cada celda en su proceso y los efectos ordenados repartidos entre procesos.
#   bash srvgdai/boreal.sh validacion      celdas prop136, prop_nuevas, k3_136, k3_nuevas: Platt y umbral en 2025
#   bash srvgdai/boreal.sh prueba          las mismas en 2026, con lo congelado en config_2x2/ (commitear antes)
#   bash srvgdai/boreal.sh publicos        los siete conjuntos publicos, uno por proceso
# BOREAL_ESCALA_ARBOLES=0.02 hace una prueba de humo. Los registros quedan en ~/estudio_otif/boreal/registros/.
set -uo pipefail
cd ~/estudio_otif/boreal/original
PY=~/estudio_otif/venv/bin/python
RUTA=~/estudio_otif/otif_depurado.csv
mkdir -p ../registros
# memoria usada cada minuto (la reducción ordinal de 2021-2025 pasa de 1,4 millones de filas): el pico queda en el registro
( while true; do echo "$(date +%H:%M) $(free -m | awk '/Mem:/ {print $3}') MB"; sleep 60; done ) > "../registros/memoria_$1.log" &
VIGIA=$!
trap 'kill $VIGIA 2>/dev/null' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BOREAL_PROCESOS=${BOREAL_PROCESOS:-8}
case "$1" in
  validacion|prueba)
    for c in prop136 prop_nuevas k3_136 k3_nuevas; do
      $PY evaluar_2x2_otif.py "$RUTA" --fase "$1" --celda "$c" > "../registros/$1_$c.log" 2>&1 &
    done
    wait ;;
  encadenar)
    # la prueba de cada celda arranca apenas su validacion escribe config_2x2/<celda>.json (registro "AUC validación")
    for c in prop136 prop_nuevas k3_136 k3_nuevas; do
      ( until grep -q "AUC validación .* Platt" "../registros/validacion_$c.log" 2>/dev/null; do
          grep -q "Traceback" "../registros/validacion_$c.log" 2>/dev/null && exit 1; sleep 30; done
        $PY evaluar_2x2_otif.py "$RUTA" --fase prueba --celda "$c" > "../registros/prueba_$c.log" 2>&1 ) &
    done
    wait ;;
  publicos)
    for c in airlines amazon click nyc_bus zurich diabetes noshow; do
      r=1; case "$c" in nyc_bus|zurich|diabetes|noshow) r=2;; esac
      suf=$([ $r = 1 ] && echo "boreal_$c" || echo "ronda2_boreal_$c")
      OMP_NUM_THREADS=4 BOREAL_PROCESOS=4 $PY -I publicos_otif.py ~/estudio_otif/publicos --solo "$c" --sufijo "$suf" > "../registros/publicos_$c.log" 2>&1 &
    done
    wait ;;
esac
echo "fin $1"
