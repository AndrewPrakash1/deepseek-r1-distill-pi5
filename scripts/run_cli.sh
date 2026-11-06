#!/usr/bin/env bash
# Self-contained offline terminal client.
#
#   ./run_cli.sh                       # interactive chat
#   ./run_cli.sh "What is 17 * 23?"    # single-shot, prints timings then exits
set -euo pipefail

HOME_DIR="${HOME}"
MODEL="${MODEL:-$HOME_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf}"
BIN="${BIN:-$HOME_DIR/llama.cpp/build/bin/llama-cli}"
CTX="${CTX:-4096}"
THREADS="${THREADS:-4}"
MAXTOK="${MAXTOK:-1024}"

[ -x "$BIN" ] || { echo "llama-cli not built at $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found at $MODEL" >&2; exit 1; }

COMMON=(-m "$MODEL" -c "$CTX" -t "$THREADS" -ngl 0 --temp 0.6 --top-p 0.95)

if [ "$#" -gt 0 ]; then
  exec "$BIN" "${COMMON[@]}" -n "$MAXTOK" -st -p "$*"
else
  exec "$BIN" "${COMMON[@]}" -n "$MAXTOK"
fi
