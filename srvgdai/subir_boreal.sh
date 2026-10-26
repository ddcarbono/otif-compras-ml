#!/usr/bin/env bash
# Sube a srvgdai (~/estudio_otif/boreal) lo que necesita la corrida de BOREAL con el boosting propio: los guiones de
# original/, sus configuraciones congeladas, las variables nuevas en csv.gz (el venv del servidor no tiene pyarrow) y los
# conjuntos publicos. El dataset ya esta en ~/estudio_otif/otif_depurado.csv (mismo md5 que el local).
# Uso, desde la raiz del repositorio: bash srvgdai/subir_boreal.sh [--solo-codigo]
# config_2x2/ no se sube: la escribe la fase de validacion en el servidor y de ahi se trae (es la constancia de lo
# congelado). --solo-codigo sube solo los guiones, sin datos (para no tocar lo que lee una corrida en curso).
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=(ssh -i "$HOME/.ssh/id_ed25519_srvgdai" -o BatchMode=yes dcarbono@10.0.0.91)
DATOS="/c/Proyectos/estudio/datos"
tar --exclude=__pycache__ -cf - original/arboles_propios.py original/modelo_original_otif.py original/otif_comun.py \
    original/evaluar_2x2_otif.py original/publicos_otif.py original/original_congelado.json original/k3_congelado.json \
    srvgdai/boreal.sh \
  | "${SSH[@]}" 'mkdir -p ~/estudio_otif/boreal && cd ~/estudio_otif/boreal && tar -xf - && sed -i "s/\r$//" srvgdai/*.sh && chmod -R go-rwx . && chmod 700 srvgdai/*.sh'
if [ "${1:-}" = "--solo-codigo" ]; then echo "subido el codigo a srvgdai:~/estudio_otif/boreal"; exit 0; fi
tar -C "$DATOS/otif" -cf - variables_nuevas.csv.gz | "${SSH[@]}" 'cd ~/estudio_otif && tar -xf -'
tar -C "$DATOS" -cf - publicos | "${SSH[@]}" 'cd ~/estudio_otif && tar -xf - && chmod -R go-rwx publicos'
echo "subido a srvgdai:~/estudio_otif/boreal"
