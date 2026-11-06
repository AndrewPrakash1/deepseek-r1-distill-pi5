#!/usr/bin/env bash
# Self-contained local runtime: OpenAI-compatible REST server on the Pi 5.
# No cloud APIs, no network dependency at inference time.
#
#   ./run_server.sh              # bind 127.0.0.1:8080 (this machine only)
#   HOST=0.0.0.0 ./run_server.sh # expose to the LAN
#
# Then, from anywhere:
#   curl http://<pi>:8080/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":512}'
set -euo pipefail

HOME_DIR="${HOME}"
MODEL="${MODEL:-$HOME_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf}"
BIN="${BIN:-$HOME_DIR/llama.cpp/build/bin/llama-server}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"
THREADS="${THREADS:-4}"

[ -x "$BIN" ] || { echo "llama-server not built at $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found at $MODEL" >&2; exit 1; }

echo "[server] model=$MODEL ctx=$CTX threads=$THREADS  http://$HOST:$PORT" >&2
exec "$BIN" -m "$MODEL" -c "$CTX" -t "$THREADS" -ngl 0 --host "$HOST" --port "$PORT"
