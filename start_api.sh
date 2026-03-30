#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/georgezhu/Documents/wty"
ENV_FILE="$PROJECT_DIR/.env"
APP_MODULE="api_server:app"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env at: $ENV_FILE"
  echo "Create it from .env.example first:"
  echo "  cp $PROJECT_DIR/.env.example $ENV_FILE"
  exit 1
fi

if ! python3 -c "import fastapi, uvicorn, openai, dotenv" >/dev/null 2>&1; then
  echo "Installing required packages: fastapi uvicorn openai python-dotenv"
  python3 -m pip install fastapi uvicorn openai python-dotenv
fi

echo "Starting API on ${HOST}:${PORT}"
cd "$PROJECT_DIR"
python3 -m uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT"
