#!/usr/bin/env bash
# Copia el codigo de los experimentos a srvgdai (~/estudio_otif/otif-compras-ml) junto con los guiones de esta carpeta.
# El dataset ya esta en ~/estudio_otif/otif_depurado.csv (mismo md5 que el local, 2026-10-22). Uso, desde la raiz del
# repositorio: bash srvgdai/subir.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=(ssh -i "$HOME/.ssh/id_ed25519_srvgdai" -o BatchMode=yes dcarbono@10.0.0.91)   # arreglo: $HOME lleva un espacio
tar --exclude=__pycache__ -cf - src/__init__.py src/datos.py src/validacion.py src/optimizadores.py src/experimentos.py src/complejidad.py \
    data/diccionario.csv srvgdai/piloto.sh srvgdai/preparar.sh srvgdai/correr.sh \
  | "${SSH[@]}" 'mkdir -p ~/estudio_otif/otif-compras-ml && cd ~/estudio_otif/otif-compras-ml && tar -xf - && chmod -R go-rwx . && chmod 700 srvgdai/*.sh'
echo "subido a srvgdai:~/estudio_otif/otif-compras-ml"
