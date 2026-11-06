#!/usr/bin/env python3
"""
Reasoning eval harness for DeepSeek-R1-Distill-Qwen-1.5B on Raspberry Pi 5.

Drives the running llama-server's OpenAI-compatible endpoint and records, per prompt:
  - TTFT                      -> timings.prompt_ms (prompt processing)
  - prompt processing speed   -> timings.prompt_per_second
  - generation speed          -> timings.predicted_per_second  (the <think> phase cost)
  - generated / prompt tokens -> timings.predicted_n / prompt_n
  - peak RSS                  -> peak VmHWM of the llama-server process
  - peak SoC temperature      -> vcgencmd sampling during the request
  - qualitative markers       -> reasoning_content: think tag, backtracking cue words

Requires llama-server already listening (default 127.0.0.1:8080).
Appends one row per prompt to results/reasoning_runs.csv; keeps full raw JSON per run.
"""
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

HOME = os.path.expanduser("~")
API = os.environ.get("API", "http://127.0.0.1:8080")
OUTDIR = os.environ.get("OUTDIR", f"{HOME}/r1bench/results")
MAXTOK = int(os.environ.get("MAXTOK", "1024"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))

os.makedirs(os.path.join(OUTDIR, "raw"), exist_ok=True)


def server_pid():
    try:
        out = subprocess.check_output(["pgrep", "-f", "llama-server"], text=True)
        return int(out.split()[0])
    except Exception:
        return None


def cpu_temp():
    try:
        return float(re.search(r"([\d.]+)",
                    subprocess.check_output(["vcgencmd", "measure_temp"], text=True)).group(1))
    except Exception:
        return None


class Monitor(threading.Thread):
    """Samples SoC temperature and the server's peak RSS while the request runs."""

    def __init__(self, pid, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.pid = pid
        self.temps = []
        self.rss_kb = 0
        self.done = threading.Event()

    def run(self):
        while not self.done.is_set():
            t = cpu_temp()
            if t is not None:
                self.temps.append(t)
            if self.pid:
                try:
                    with open(f"/proc/{self.pid}/status") as f:
                        for ln in f:
                            if ln.startswith("VmHWM:"):
                                self.rss_kb = max(self.rss_kb, int(ln.split()[1]))
                                break
                except Exception:
                    pass
            self.done.wait(self.interval)

    def stop(self):
        self.done.set()
        self.join(timeout=3)

    @property
    def peak_temp(self):
        return max(self.temps) if self.temps else None

    @property
    def peak_rss_mb(self):
        return round(self.rss_kb / 1024, 1) if self.rss_kb else None


BACKTRACK_CUES = [
    "wait", "actually", "hmm", "on second thought", "i made an error",
    "let me reconsider", "hold on", "that's wrong", "that is wrong",
    "recheck", "re-check", "double-check", "let me redo", "but wait",
    "let me try", "correcting", "mistake", "however",
]
SOFT_CUES = ["let me", "verify", "check", "alternatively", "explore", "consider"]


def markers(text):
    low = text.lower()
    return {
        "has_think_tag": ("<think>" in low or "[start thinking]" in low
                          or "reasoning_content" in low or bool(text.strip())),
        "has_backtrack": any(c in low for c in BACKTRACK_CUES),
        "n_backtrack_cues": sum(low.count(c) for c in BACKTRACK_CUES),
        "n_soft_cues": sum(low.count(c) for c in SOFT_CUES),
        "reasoning_chars": len(text),
    }


def ask(prompt):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": False,
        "cache_prompt": False,
    }).encode()

    req = urllib.request.Request(
        f"{API}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")

    pid = server_pid()
    mon = Monitor(pid)
    mon.start()
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read().decode())
    finally:
        wall = time.time() - t0
        mon.stop()
    return data, wall, mon


def run_one(pid, category, prompt, expected):
    raw_path = os.path.join(OUTDIR, "raw", f"{pid}.json")
    data, wall, mon = ask(prompt)

    tim = data.get("timings", {}) or {}
    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
    reasoning = msg.get("reasoning_content") or ""
    answer = msg.get("content") or ""

    row = {
        "prompt_id": pid,
        "category": category,
        "expected": expected,
        "wall_s": round(wall, 2),
        "ttft_s": round(tim["prompt_ms"] / 1000, 3) if tim.get("prompt_ms") else None,
        "prompt_tokens": tim.get("prompt_n"),
        "pp_tokens_per_sec": round(tim["prompt_per_second"], 2) if tim.get("prompt_per_second") else None,
        "gen_tokens": tim.get("predicted_n"),
        "gen_ms": round(tim["predicted_ms"], 1) if tim.get("predicted_ms") else None,
        "gen_tokens_per_sec": round(tim["predicted_per_second"], 2) if tim.get("predicted_per_second") else None,
        "gen_seconds": round(tim["predicted_ms"] / 1000, 2) if tim.get("predicted_ms") else None,
        "peak_rss_mb": mon.peak_rss_mb,
        "peak_temp_c": mon.peak_temp,
        "temp_samples": len(mon.temps),
        "answer_chars": len(answer),
        **markers(reasoning),
    }

    with open(raw_path, "w") as f:
        json.dump({"prompt": prompt, "timings": tim,
                   "reasoning_content": reasoning, "content": answer,
                   "wall_s": wall, "peak_rss_mb": mon.peak_rss_mb,
                   "peak_temp_c": mon.peak_temp}, f, indent=1)
    return row


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    prompts = json.load(open(os.path.join(here, "prompts.json")))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = os.path.join(OUTDIR, "reasoning_runs.csv")

    ok, _ = True, None
    try:
        with urllib.request.urlopen(f"{API}/health", timeout=5) as r:
            ok = "ok" in r.read().decode()
    except Exception:
        ok = False
    if not ok:
        print(f"[eval] llama-server not reachable at {API}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for item in prompts:
        if only and item["id"] != only:
            continue
        print(f"--- {item['id']} ({item['category']}) ---", file=sys.stderr, flush=True)
        try:
            row = run_one(item["id"], item["category"], item["prompt"], item.get("expected", ""))
        except Exception as e:
            print(f"    FAILED: {e}", file=sys.stderr, flush=True)
            continue
        rows.append(row)
        print(f"    ttft={row['ttft_s']}s  pp={row['pp_tokens_per_sec']}t/s  "
              f"gen={row['gen_tokens_per_sec']}t/s  gen_tokens={row['gen_tokens']}  "
              f"rss={row['peak_rss_mb']}MB  temp={row['peak_temp_c']}C  "
              f"backtrack={row['has_backtrack']}({row['n_backtrack_cues']})",
              file=sys.stderr, flush=True)

    if not rows:
        return
    fields = list(rows[0].keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"[eval] appended {len(rows)} rows -> {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
