#!/bin/bash
# Levanta el stack de Wallet (backend + frontend) sin reconstruir la imagen.
# Solo usa "--build" a mano cuando cambies código o dependencias.

cd "$(dirname "$0")" || exit 1

echo "== Levantando Wallet (docker compose up -d) =="
docker compose up -d

echo
echo "== Estado de los contenedores =="
docker compose ps

echo
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"

# Abre el frontend en el navegador si está disponible
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:5173" >/dev/null 2>&1 &
fi

echo
read -p "Pulsa Enter para cerrar esta ventana..."
