# DeepSeek-R1-Distill-Qwen-1.5B on a Raspberry Pi 5

Deploying and benchmarking a distilled DeepSeek-R1 reasoning model on ARM64 edge hardware with
[llama.cpp](https://github.com/ggml-org/llama.cpp) — no cloud APIs, no network at inference time.

This repo accompanies a study of whether the reasoning behaviours discovered in the DeepSeek-R1
research (chain-of-thought, backtracking, self-correction) survive knowledge distillation into a
1.5 B student **and** post-training 4-bit quantisation, when run under the memory, thermal and
latency limits of a single-board computer.

**Full write-up: [`REPORT.md`](REPORT.md)** · **Prompting guide: [`PROMPTING.md`](PROMPTING.md)**

---

## Headline results

Measured on a Raspberry Pi 5 Model B (8 GB) running Debian 13 (trixie), llama.cpp `0.9.0-dev`
(commit `18ab62a`) built on-device, CPU-only, Q4_K_M.

| Metric | Result |
|---|---|
| Model footprint | 1,117,321,312 B (1.04 GiB) GGUF |
| Peak resident RAM | **2.07 – 2.12 GB** |
| Prompt processing (4 threads) | 56.6 t/s @ 128 tok → 37.1 t/s @ 4096 tok (**−34%**) |
| Time to first token | 2.3 s @ 128 tok → **110 s** @ 4096 tok |
| Token generation | **10.9 t/s** (4 threads) / **12.3 t/s** (2 threads) |
| Thermal penalty at 83 °C vs 53 °C | **−7.3%** generation, −7.9% prompt processing |
| Reasoning accuracy (7 prompts) | 3 correct · 2 wrong · 2 never finished |

### Three findings worth highlighting

1. **Generation is memory-bandwidth-bound, not compute-bound.** Token generation is **12.9 %
   *faster* on 2 threads than on 4** (12.31 vs 10.90 t/s), while prompt processing nearly doubles
   with 4 threads. Decode contends for the memory bus, so extra cores only add heat.
   **Use `-t 2` for chat, `-t 4` only for long prompts.**

2. **The "reasoning tax" is the dominant latency cost.** The model emits hundreds of thinking
   tokens before answering. A 512-token trace costs ~47 s of pure generation *on top of* TTFT.
   In one run the model reasoned correctly to the answer **two independent ways, verified them
   against each other — and then exhausted its 1024-token budget generalising, never emitting an
   answer at all.** 3 of 7 prompts hit the cap.

3. **The project brief's memory model is wrong in both directions.** The KV cache at 4096 ctx is
   **112 MiB, not ~0.5 GB** (Qwen2.5-1.5B uses GQA with only 2 KV heads). But llama.cpp also
   allocates a **~934 MiB `CPU_REPACK` buffer** on ARM that the brief never mentions — making it
   the second-largest allocation. Real budget:
   *weights 1.04 GiB + repack 0.93 GiB + KV 0.11 GiB*.

---

## Hardware and software

| | |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1, 8 GB |
| CPU | Broadcom BCM2712, 4 × Cortex-A76 @ 2.4 GHz, NEON (no tensor cores) |
| OS | Debian GNU/Linux 13 (trixie), kernel `6.18.34+rpt-rpi-2712` |
| Toolchain | gcc 14.2.0, cmake 3.31.6 |
| Engine | llama.cpp `0.9.0-dev`, commit `18ab62a`, `GGML_NATIVE=ON`, CPU backend only |
| Model | [`unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF`](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) → `Q4_K_M` |

---

## Quick start

### 1. Build llama.cpp (on the Pi, no sudo required)

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build build --config Release -j"$(nproc)"
```

`-DLLAMA_CURL=OFF` drops the libcurl-dev dependency; `GGML_NATIVE=ON` enables NEON/i8mm on aarch64.
Takes ~20–30 min on a Pi 5.

### 2. Fetch the model

```bash
mkdir -p ~/models && cd ~/models
curl -fL -C - -o DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
```

### 3. Run it

```bash
./scripts/run_cli.sh                       # interactive chat
./scripts/run_cli.sh "What is 17 * 23?"    # single-shot
./scripts/run_server.sh                    # REST API on 127.0.0.1:8080
```

---

## Benchmarking

```bash
bash scripts/bench_llamabench.sh    # pp/tg sweep across ctx depth × thread count
python3 scripts/reasoning_eval.py   # 7-prompt reasoning suite (needs run_server.sh up)
bash scripts/thermal_test.sh        # controlled cool-vs-hot experiment
```

| Script | Purpose |
|---|---|
| `scripts/run_cli.sh` | Offline terminal client (interactive or single-shot) |
| `scripts/run_server.sh` | OpenAI-compatible REST server |
| `scripts/bench_llamabench.sh` | `llama-bench` matrix → `results/llamabench.csv` |
| `scripts/parse_llamabench.py` | Parses llama-bench's 42-column CSV into a tidy table |
| `scripts/reasoning_eval.py` | Drives `/v1/chat/completions`, records TTFT/tg/RSS/temp/markers |
| `scripts/prompts.json` | The 7 verifiable reasoning prompts |
| `scripts/thermal_test.sh` | Heat-soak and re-measure to isolate the thermal effect |

## Results

| File | Contents |
|---|---|
| `results/llamabench.csv` | Tidy pp/tg/TTFT table across context depth |
| `results/llamabench_raw.csv` | Raw 42-column llama-bench dump |
| `results/reasoning_runs.csv` | Per-prompt TTFT, throughput, peak RSS, peak temp, markers |
| `results/raw/*.json` | Full reasoning traces + final answers for each prompt |
| `results/thermal.csv` | Cool vs heat-soaked throughput |

---

## Caveats

- 7 prompts is a small sample — the 3/7 accuracy figure is **indicative, not a rate**.
- The notebook word problem is ambiguous (bulk vs pro-rata pricing); the exact pro-rata answer is
  ₹280.67, and the model's ₹279.62 stems from a genuine multiplication slip it never caught.
- Timings come from llama.cpp's own `timings` block via the REST API, not from external wall-clock
  instrumentation; a single 3-repeat `llama-bench` run is long enough to heat the die, so
  "cool" readings record temperature before *and* after.

## License

MIT — see [LICENSE](LICENSE). Model weights are subject to their own license on Hugging Face.
