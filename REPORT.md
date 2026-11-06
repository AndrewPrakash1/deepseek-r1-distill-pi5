# Deploying & Benchmarking DeepSeek-R1-Distill-Qwen-1.5B on a Raspberry Pi 5

**Status:** runtime built and serving · benchmarks + thermal experiment complete · qualitative reasoning analysis done
**Date:** 2026-11-06 · **Host:** Raspberry Pi 5 Model B Rev 1.1 (8 GB) · accessed from macOS over SSH

---

## 1. Deliverable 1 — Working local runtime

| Component | Detail |
|---|---|
| Engine | llama.cpp `0.9.0-dev`, commit `18ab62a`, built from source **on the Pi** |
| Toolchain | gcc 14.2.0 (`GNU 14.2.0 for Linux aarch64`), cmake 3.31.6, `GGML_NATIVE=ON` (NEON auto-enabled via AArch64) |
| Backend | **CPU only** — no GPU/tensor backend present (`--list-devices` → none), integer matmul on NEON |
| Model | `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` (unsloth), **1,117,321,312 B = 1.04 GiB** |
| Artifact 1 | `llama-cli` — offline single-shot + interactive terminal client |
| Artifact 2 | `llama-server` — OpenAI-compatible REST API on `127.0.0.1:8080`, verified `/health` → `{"status":"ok"}` |

Entry points (on the Pi, `~/r1bench/scripts/`): `run_cli.sh`, `run_server.sh`.

```
$ ./run_cli.sh "What is 17 * 23? Think step by step."
[Start thinking]
First, I need to multiply 17 by 23. I'll break down 23 into 20 and 3 ...
Finally, I'll add 340 and 51 together to get the final result of 391.
[End thinking]
\boxed{391}
[ Prompt: 57.1 t/s | Generation: 11.6 t/s ]
```

No external API, no network at inference time.

---

## 2. Deliverable 2 — Performance benchmark table

### 2a. Raw throughput vs. context depth (`llama-bench`, 3 repeats, `-ngl 0`)

| threads | prompt tok | pp (t/s) | TTFT | tg (t/s) | 512-tok think |
|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 33.51 | 3.82 s | 12.31 | 41.6 s |
| 2 | 512 | 32.29 | 15.86 s | 12.31 | 41.6 s |
| 2 | 1024 | 31.00 | 33.04 s | 12.31 | 41.6 s |
| 2 | 2048 | 28.63 | 71.53 s | 12.31 | 41.6 s |
| 2 | 4096 | 24.75 | 165.49 s | 12.31 | 41.6 s |
| **4** | 128 | **56.61** | 2.26 s | 10.90 | 47.0 s |
| **4** | 512 | **51.10** | 10.02 s | 10.90 | 47.0 s |
| **4** | 1024 | **47.63** | 21.50 s | 10.90 | 47.0 s |
| **4** | 2048 | **43.14** | 47.47 s | 10.90 | 47.0 s |
| **4** | 4096 | **37.14** | **110.29 s** | 10.90 | 47.0 s |

**Readings**

1. **Prompt processing degrades with context depth** — 56.61 → 37.14 t/s at 4 threads (**−34.4%**) going 128 → 4096 tokens, and −26.1% at 2 threads. Every token already in the KV cache makes the next one cheaper to reach but the batch more expensive to touch. TTFT grows super-linearly in wall-clock: 2.3 s → **110 s**.
2. **Threads help prompt processing, hurt generation.** pp nearly doubles with 4 threads (56.61 vs 33.51), but token generation is **12.9% faster on 2 threads** (12.31 vs 10.90 t/s). Decode is memory-bandwidth-bound, not compute-bound — extra cores contend for the same memory bus and add heat for nothing. **This is the single biggest free optimisation: `-t 2` for chat, `-t 4` only if prompts are long.**
3. **The reasoning tax is real.** DeepSeek-R1-Distill writes hundreds of think tokens before the answer. At 4 threads a 512-token trace costs **47 s** of pure generation *before* the answer starts; the observed `logic_selfcheck` run took **100 s** and still produced no answer.

### 2b. End-to-end reasoning runs (`llama-server`, `-c 4096 -t 4`, temp 0.6, cap 1024 tokens)

| prompt | cat | TTFT | pp t/s | gen t/s | gen tok | peak RSS | peak °C | backtrack cues |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arith_multistep | arithmetic | 0.857 s | 60.67 | 11.37 | 421 | 2065 MB | 82.3 | 0 |
| arith_backtrack | arithmetic | 0.597 s | 56.92 | 10.61 | 407 | 2065 MB | 83.4 | 0 |
| logic_puzzle | logic | 0.913 s | 54.76 | 10.79 | 476 | 2065 MB | 84.5 | 0 |
| logic_false_premise | logic | 0.610 s | 47.56 | 9.28 | **1024** | 2065 MB | 85.1 | 5 |
| code_algorithm | programming | 0.754 s | 49.06 | 10.30 | **1024** | 2070 MB | 85.1 | 6 |
| code_bugfix | programming | 0.766 s | 49.64 | 9.02 | 761 | 2099 MB | 85.1 | 5 |
| logic_selfcheck | logic | 0.651 s | 47.64 | 10.27 | **1024** | 2121 MB | 85.1 | 8 |

- Generation settles at **9.02–11.37 t/s** (median 10.30) and *decays down the column* as the die heats.
- **RSS grows 2065 → 2121 MB** as the 4096-token KV window fills — the context-pressure effect the project brief warns about, visible at ~56 MB of live KV pages.

### 2c. Memory model — the brief's numbers are wrong in both directions

| Component | Project brief assumed | Measured |
|---|---|---|
| Weights (Q4_K_M) | ~1.1 GB | 1.04 GiB file, mmap'd |
| KV cache @ 4096 ctx | ~0.5 GB | **112 MiB** |
| OS + overhead | ~1.0 GB | ~0.3 GB host |
| **ARM repack buffer** | *not mentioned* | **934 MiB** |
| **Observed peak RSS** | ~2.6 GB implied | **2.07–2.12 GB** |

Two corrections:

- **KV cache is 4.5× smaller than assumed.** Qwen2.5-1.5B uses GQA with only **2 KV heads**, so at `-c 4096` with f16 K/V it is `2 × 28 layers × 4096 × 256 dim × 2 B = 112 MiB`, not ~0.5 GB.
- **But there is a hidden ~934 MiB cost the brief misses:** llama.cpp's `CPU_REPACK` buffer, which re-lays-out quantised weights into ARM-friendly block formats. Total resident is ~2.07 GB, not ~1.1 GB.

Net: the model fits 8 GB with ~6 GB to spare, and would even fit a 4 GB Pi 5 — but the project's memory budget should be restated as *weights 1.04 GiB + repack 0.93 GiB + KV 0.11 GiB*, because **the repack buffer, not the KV cache, is the second-largest allocation.**

---

## 3. Thermal behaviour (controlled experiment)

Cool baseline measured at 53.2 °C, then a heat soak of sustained 512-token generations until the die plateaued, then re-measured:

| phase | SoC before → after | pp (32 tok) | tg (256 tok) |
|---|---|---|---|
| cool | 53.2 → 81.2 °C | 60.98 t/s | **11.53 t/s** |
| hot | 82.9 → 82.9 °C | 56.15 t/s | **10.69 t/s** |
| **delta** | | **−7.9%** | **−7.3%** |

- The die **plateaus at ~82–83 °C** under sustained load and does not run away (active cooling present), but it **does** cross the soft limit: `vcgencmd get_throttled` latched **`0xe0000`** — bits 17/18/19 = *ARM frequency capped*, *throttling occurred*, *soft temperature limit occurred*.
- Cost of heat is a real **~7% throughput tax on both phases**, plus the drift seen live in §2b (11.37 → 9.02 t/s over the run).
- The brief's "80–85 °C" throttle window is confirmed: the Pi self-limits just inside it.
- Practical consequence: **the first answer after boot is the fastest.** A Pi 5 in a closed case without a heatsink will throttle earlier and lose more than 7%.

---

## 4. Deliverable 3 — Qualitative reasoning analysis

7 verifiable problems. Scoring below is by hand against ground truth.

| prompt | verdict | notes |
|---|---|---|
| arith_backtrack | ✅ **correct** | `3x+7=2x+19 → x=12`, then `5x−4 = 56`. Clean, no digression. |
| code_bugfix | ✅ **correct** | Correctly diagnosed `s[::-2]` and fixed to `s[::-1]`. |
| logic_false_premise | ✅ **correct** | "Not valid" — reasoned about the undistributed middle term. |
| arith_multistep | ⚠️ **arithmetic slip** | See below. |
| logic_puzzle | ❌ **wrong** | See below — self-contradictory trace. |
| code_algorithm | ⏹️ **truncated** | Sound approach, ran out of budget mid-code. |
| logic_selfcheck | ⏹️ **correct but never delivered** | See below. |

### 4a. Emergent behaviour 1 — self-verification (the "aha" pattern)

`logic_selfcheck` ("sum of the first 20 even numbers, two independent ways") is the clearest demonstration that the distilled student retains R1's behaviours. The model solved it via the arithmetic-series formula, then **independently re-derived it by pairing terms**, and explicitly cross-checked:

> "…each pair adds up to 42 … there are 20 terms, we can pair them into 10 pairs … the total sum should be 10\*42 = **420**. That's the same result as before. **So that seems consistent.**"
>
> "Wait, but is that always the case? Let me think. If I have an even number of terms, then yes…"

It then began generalising to odd-length sequences — and **burned the remaining budget doing so, never emitting the final answer.** The run ended at the 1024-token cap with `content = ""`. The reasoning was right; the user got nothing.

### 4b. Emergent behaviour 2 — backtracking cues track difficulty, not correctness

| | short/clean answers | long/struggling answers |
|---|---|---|
| backtrack cue count | **0, 0, 0** | 5, 6, 5, 8 |

All four runs that showed self-correction markers (`wait`, `actually`, `hmm`, `on second thought`, `that's wrong`) also ran long, and **three of the four hit the token cap without converging**. So the "aha moment" markers are real and reproduce on a 1.5 B student on a CPU — but **on this hardware they correlate with non-termination.** Verbose self-doubt is what the latency budget is spent on.

### 4c. Failure mode 1 — internally contradictory reasoning (`logic_puzzle`)

The model derived the correct deduction and then violated it in the very next sentence:

> "…the one with the dog is not C, **so the dog must be with A**.
> Therefore, the pets are: **A has the cat**, **C has the dog**, B has the bird."

It asserted dog→A and then assigned dog→C. Ground truth is **A=dog, C=cat, B=bird**; the model returned A=cat, C=dog. Note this trace was **clean** — zero backtracking cues. The model was confidently wrong, which is exactly the failure the R1 self-correction training is meant to prevent, and the small student did not catch it.

### 4d. Failure mode 2 — arithmetic slippage (`arith_multistep`)

Prompt: notebooks "3 for ₹47", buy 14, pay with ₹500. Exact pro-rata total is `14 × 47/3 = ₹219.33`, so change = **₹280.67**.

The model computed `47/3 ≈ 15.67`, then wrote:

> "14 times 15.67 rupees, which equals **220.38** rupees" → change **279.62**

`14 × 15.67 = 219.38`, not 220.38 — a raw multiplication error of ₹1, which then propagated. Final answer 279.62 vs 280.67. **No backtracking was triggered** (0 cues); the error shipped silently.

### 4e. Verdict on knowledge distillation

The 1.5 B student **does** reproduce R1's zero-shot reasoning *form*: visible think traces, multi-step decomposition, independent re-derivation, and genuine self-correction language. It does **not** reliably reproduce its *reliability*. On this small sample the model was correct on 3/7 outright, wrong on 2/7 (one silently, one self-contradictorily), and 2/7 never finished. The emergent behaviours are present; the error-catching that makes them useful is not yet dependable at this scale — which is the expected outcome of distilling a 671 B teacher into 1.5 B, and the honest answer to the project's central question.

---

## 5. Reproduce

```bash
# on the Pi
~/r1bench/scripts/run_server.sh &                     # or run_cli.sh
python3 ~/r1bench/scripts/reasoning_eval.py            # 7-prompt reasoning suite
bash ~/r1bench/scripts/bench_llamabench.sh             # pp/tg sweep -> llamabench.csv
bash ~/r1bench/scripts/thermal_test.sh                 # cool-vs-hot controlled test
```

Raw rows: `results/llamabench_raw.csv` (42-column llama-bench dump), `results/reasoning_runs.csv`,
`results/thermal.csv`, per-prompt traces in `results/raw/*.json`.

**Note on methodology:** the first `arith_multistep` pass ran before `--perf`-style timing output was
located, so its timings were re-captured through `llama-server`'s `timings` block; all seven rows in
§2b come from that API path for consistency.
