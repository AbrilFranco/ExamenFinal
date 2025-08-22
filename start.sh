#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p data

# Descargar dataset si no existe
if [ ! -s "data/spam_dataset_with_content.pkl" ]; then
  echo "Descargando dataset..."
  curl -L --fail --retry 3 --retry-delay 2 \
    -o data/spam_dataset_with_content.pkl \
    https://github.com/AbrilFranco/ExamenFinal/releases/download/v1.0.0/spam_dataset_with_content.pkl
fi

# Ejecutar con Gunicorn (Render provee $PORT)
exec gunicorn --bind 0.0.0.0:$PORT app:app

