#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "THERADIOMICS - stable launcher"
echo "============================================================"
echo ""

if ! command -v bun >/dev/null 2>&1; then
  echo "[ERRORE] Bun non trovato nel PATH."
  exit 1
fi

if [ -x ".venv/bin/python" ]; then
  export PYTHON_EXE="$PWD/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  export PYTHON_EXE="python3"
else
  export PYTHON_EXE="python"
fi

export PORT="${PORT:-8765}"

echo "Python: $PYTHON_EXE"
echo "Dashboard: http://127.0.0.1:$PORT"
echo ""

if command -v open >/dev/null 2>&1; then
  (sleep 1; open "http://127.0.0.1:$PORT") >/dev/null 2>&1 &
fi

bun run server.ts
