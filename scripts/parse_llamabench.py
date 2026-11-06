#!/usr/bin/env python3
"""Turn llama-bench's raw 42-column CSV into the project's benchmark table.

Input rows are of two kinds (identified by n_prompt / n_gen):
  pp rows  -> n_prompt=P, n_gen=0   (prompt processing, tokens/sec)
  tg rows  -> n_prompt=0, n_gen=N   (token generation,  tokens/sec)

Output columns:
  threads, prompt_tokens, pp_tokens_per_sec, pp_stddev,
  ttft_ms (prompt_tokens / pp speed),
  tg_tokens_per_sec, tg_stddev,
  est_think_512tok_s (512 generated tokens / tg speed)
"""
import csv
import sys


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    raw, out = sys.argv[1], sys.argv[2]
    pp = {}   # (threads, prompt) -> (avg_ts, stddev_ts)
    tg = {}   # threads -> (avg_ts, stddev_ts, n_gen)

    with open(raw, newline="") as f:
        for row in csv.DictReader(f):
            threads = int(float(row["n_threads"]))
            n_prompt = int(float(row["n_prompt"]))
            n_gen = int(float(row["n_gen"]))
            ts = num(row["avg_ts"])
            sd = num(row["stddev_ts"])
            if ts is None:
                continue
            if n_prompt > 0 and n_gen == 0:
                pp[(threads, n_prompt)] = (ts, sd)
            elif n_gen > 0 and n_prompt == 0:
                tg[threads] = (ts, sd, n_gen)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "threads", "prompt_tokens", "pp_tokens_per_sec", "pp_stddev",
            "ttft_ms", "tg_tokens_per_sec", "tg_stddev", "est_think_512tok_s",
        ])
        for threads in sorted({t for t, _ in pp} | set(tg)):
            tg_ts, tg_sd, _ = tg.get(threads, (None, None, None))
            for (th, prompt) in sorted(k for k in pp if k[0] == threads):
                pp_ts, pp_sd = pp[(th, prompt)]
                ttft_ms = round(prompt / pp_ts * 1000, 1) if pp_ts else None
                est = round(512 / tg_ts, 2) if tg_ts else None
                w.writerow([
                    threads, prompt,
                    round(pp_ts, 2), round(pp_sd, 3) if pp_sd is not None else None,
                    ttft_ms,
                    round(tg_ts, 2) if tg_ts else None,
                    round(tg_sd, 3) if tg_sd is not None else None,
                    est,
                ])
            # tg-only row when no pp data at this thread count
            if tg_ts and not any(k[0] == threads for k in pp):
                w.writerow([threads, "", None, None, None,
                            round(tg_ts, 2), round(tg_sd, 3) if tg_sd is not None else None,
                            round(512 / tg_ts, 2)])


if __name__ == "__main__":
    main()
