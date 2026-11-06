#!/usr/bin/env bash
# Raw throughput sweep: prompt-processing (pp / TTFT proxy) vs token-generation (tg)
# across context depths and thread counts.
#
# llama-bench's -o csv emits a 42-column header + two row kinds per config:
#   pp rows: n_prompt=<P>, n_gen=0      -> prompt processing speed
#   tg rows: n_prompt=0,   n_gen=<N>    -> token generation speed
# We keep the raw csv and derive a tidy table with parse_llamabench.py.
set -uo pipefail

MODEL="${MODEL:-$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf}"
BIN="${BIN:-$HOME/llama.cpp/build/bin/llama-bench}"
OUTDIR="${OUTDIR:-$HOME/r1bench/results}"
RAW="$OUTDIR/llamabench_raw.csv"
TIDY="$OUTDIR/llamabench.csv"

mkdir -p "$OUTDIR"

echo "[bench] running matrix p=128,512,1024,2048,4096  t=2,4  r=3" >&2
"$BIN" -m "$MODEL" \
  -p 128,512,1024,2048,4096 -n 128 \
  -t 2,4 -r 3 -ngl 0 -o csv > "$RAW" 2> "$OUTDIR/llamabench.stderr"

echo "[bench] raw rows: $(wc -l < "$RAW")" >&2
python3 "$(dirname "$0")/parse_llamabench.py" "$RAW" "$TIDY"
echo "[bench] wrote $TIDY" >&2
cat "$TIDY" >&2
