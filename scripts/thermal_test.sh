#!/usr/bin/env bash
# Controlled thermal experiment: measure tg/pp with the SoC cool, then heat-soak
# with sustained generation and measure again at the throttle plateau.
# Isolates the thermal effect from prompt-length effects.
set -uo pipefail

MODEL="${MODEL:-$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf}"
BENCH="${BENCH:-$HOME/llama.cpp/build/bin/llama-bench}"
CLI="${CLI:-$HOME/llama.cpp/build/bin/llama-cli}"
OUT="${OUT:-$HOME/r1bench/results/thermal.csv}"
HEAT_TARGET="${HEAT_TARGET:-84}"
THREADS="${THREADS:-4}"

temp_now() { vcgencmd measure_temp | grep -oE '[0-9.]+'; }

measure() {
  local label="$1"
  local t_before; t_before=$(temp_now)
  local csv
  csv=$("$BENCH" -m "$MODEL" -p 32 -n 256 -t "$THREADS" -r 3 -ngl 0 -o csv 2>/dev/null)
  local t_after; t_after=$(temp_now)
  python3 - "$label" "$t_before" "$t_after" "$csv" >> "$OUT" <<'PY'
import sys, csv, io
label, t_before, t_after, data = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pp = tg = ""
for row in csv.DictReader(io.StringIO(data)):
    if int(float(row["n_prompt"])) > 0:
        pp = round(float(row["avg_ts"]), 2)
    elif int(float(row["n_gen"])) > 0:
        tg = round(float(row["avg_ts"]), 2)
print(f"{label},{t_before},{t_after},{pp},{tg}")
PY
  tail -1 "$OUT" >&2
}

echo "phase,temp_before_c,temp_after_c,pp_32tok_tps,tg_256tok_tps" > "$OUT"

echo "[thermal] cooling down..." >&2
for _ in $(seq 1 90); do
  t=$(temp_now); if [ "${t%.*}" -lt 58 ]; then break; fi; sleep 2
done
echo "[thermal] cool baseline (temp $(temp_now)C)" >&2
measure cool

echo "[thermal] heat soak: sustained generation until ${HEAT_TARGET}C..." >&2
for i in $(seq 1 20); do
  "$CLI" -m "$MODEL" -c 2048 -n 512 -t "$THREADS" -ngl 0 --temp 0.6 -st \
    -p "Prove that the sum of the first n odd numbers is n squared, and give the first 10 partial sums." \
    >/dev/null 2>&1 || true
  t=$(temp_now)
  echo "   soak $i: temp=${t}C" >&2
  if [ "${t%.*}" -ge "$HEAT_TARGET" ]; then break; fi
done

echo "[thermal] hot measurement (temp $(temp_now)C)" >&2
measure hot

echo "[thermal] done" >&2
cat "$OUT" >&2
