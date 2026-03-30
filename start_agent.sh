#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/georgezhu/Documents/wty"
ENV_FILE="$PROJECT_DIR/.env"
RUNTIME_SCRIPT="$PROJECT_DIR/chat_runtime_openrouter.py"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env at: $ENV_FILE"
  echo "Create it from .env.example first:"
  echo "  cp $PROJECT_DIR/.env.example $ENV_FILE"
  exit 1
fi

if ! python3 -c "import openai, dotenv" >/dev/null 2>&1; then
  echo "Installing required packages: openai python-dotenv"
  python3 -m pip install openai python-dotenv
fi

echo "Starting Wang Tianyu agent..."
python3 "$RUNTIME_SCRIPT"
