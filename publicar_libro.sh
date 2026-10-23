#!/usr/bin/env bash
# Publica el libro compilado (book/_build/html) en la rama gh-pages de ddcarbono/otif-compras-ml, que sirve GitHub Pages
# en https://ddcarbono.github.io/otif-compras-ml/. Hace lo mismo que ghp-import, sin instalarlo.
# Uso: bash publicar_libro.sh "mensaje del commit"   (antes: python book/compilar.py y jupyter-book build book/)
# La rama gh-pages se reescribe entera en cada corrida: solo lleva el HTML, no historia.
set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
URL="https://github.com/ddcarbono/otif-compras-ml.git"
MSG="${1:-}"
HTML="$AQUI/book/_build/html"
if [ -z "$MSG" ]; then echo "Falta el mensaje del commit: bash publicar_libro.sh \"mensaje\"" >&2; exit 1; fi
if [ ! -f "$HTML/index.html" ]; then echo "No hay libro compilado en $HTML: correr jupyter-book build book/" >&2; exit 1; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp -r "$HTML"/. "$TMP"/
touch "$TMP/.nojekyll"           # sin esto GitHub Pages ignora las carpetas que empiezan con guion bajo (_static)
cd "$TMP"
git init -q
git checkout -q -b gh-pages
for e in $(ls -A | grep -v '^\.git$'); do git add -- "$e"; done
git commit -q -m "$MSG"
git push -q -f "$URL" gh-pages
echo "publicado en la rama gh-pages: https://ddcarbono.github.io/otif-compras-ml/"
