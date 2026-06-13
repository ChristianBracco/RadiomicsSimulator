#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "============================================================"
echo "THERADIOMICS - Cross-platform launcher"
echo "Sistema rilevato: macOS/Linux"
echo "============================================================"
echo ""

if ! command -v bun >/dev/null 2>&1; then
  echo "[ERRORE] Bun non trovato nel PATH."
  echo "Installa Bun oppure aggiungilo al PATH."
  echo "https://bun.sh"
  exit 1
fi

if [ -z "${PYTHON_EXE:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    export PYTHON_EXE="python3"
  elif command -v python >/dev/null 2>&1; then
    export PYTHON_EXE="python"
  else
    echo "[ERRORE] Python 3 non trovato."
    echo "Installa Python 3 oppure imposta PYTHON_EXE."
    echo "Esempio:"
    echo "  export PYTHON_EXE=python3"
    exit 1
  fi
fi

export PORT="${PORT:-8765}"

echo "Cartella progetto:"
echo "$(pwd)"
echo ""
echo "Python usato:"
echo "$PYTHON_EXE"
echo ""
echo "Porta:"
echo "$PORT"
echo ""
echo "Avvio dashboard:"
echo "http://127.0.0.1:$PORT"
echo ""

# Apri il browser in automatico, senza bloccare il server.
if command -v open >/dev/null 2>&1; then
  (sleep 1; open "http://127.0.0.1:$PORT") >/dev/null 2>&1 &
elif command -v xdg-open >/dev/null 2>&1; then
  (sleep 1; xdg-open "http://127.0.0.1:$PORT") >/dev/null 2>&1 &
fi

bun run server.ts
