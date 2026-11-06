# How to prompt DeepSeek-R1 (Distill) on this setup

R1 is a **reasoning** model, not a chat model. It writes a long internal thinking pass before
answering. Prompting it like ChatGPT — short, chatty, expecting an instant reply — produces worse
answers and confusing output. This guide covers what actually works here.

---

## 1. Recommended sampling settings

| Parameter | Value | Why |
|---|---|---|
| `temperature` | **0.6** | DeepSeek's recommended setting for R1. Higher → incoherent loops. |
| `top_p` | **0.95** | |
| `min_p` | `0.05` | llama.cpp's default; fine to leave |
| `repeat_penalty` | **1.0 (off)** | Do *not* penalise repetition — reasoning legitimately restates things. It makes R1 worse. |

```bash
# these are already the defaults in scripts/run_cli.sh
llama-cli -m model.gguf --temp 0.6 --top-p 0.95
```

---

## 2. Give it room to think — `-n` is the setting people get wrong

The thinking pass is **long**. On this Pi a reasoning answer routinely consumes **400–1000+ tokens
before the final answer starts**. If you cap generation low, the model just thinks and stops —
you get reasoning and no answer.

```bash
# BAD: truncated before it finishes thinking
./scripts/run_cli.sh "..."        # defaults to -n 1024

# BETTER for hard questions
MAXTOK=2048 ./scripts/run_cli.sh "Prove that sqrt(2) is irrational."
```

In this study, **3 of 7 prompts hit the 1024-token cap without producing an answer.** If a reply
looks truncated, raise `-n` before assuming the model failed.

---

## 3. Zero-shot, direct, no few-shot examples

R1 was trained to reason from a plain instruction. Few-shot examples and heavy role-play prompts
*degrade* its output. Just state the problem.

```
What is 1234 * 5678? Show your reasoning step by step.
```

Do not write "You are a helpful assistant..." — R1 has no system-prompt persona in its training
template, and one mostly confuses the reasoning trace. Go straight to the question.

---

## 4. Ask for the reasoning, then the answer

Explicitly requesting steps reliably engages the thinking pass and makes the output checkable:

| Prompt | Effect |
|---|---|
| `What is 17 * 23?` | Works, but thin reasoning |
| `What is 17 * 23? Think step by step.` | ✅ Decomposes, checks, reliable |
| `Solve 3x+7=2x+19, then compute 5x-4.` | ✅ Multi-stage chains work well |
| `...but I think the answer is 42.` | ⚠️ Sycophancy risk — it may agree with a wrong premise |

**Do not ask it to skip reasoning** ("answer in one word", "no explanation"). You are disabling
the part that makes it good. Ask for a short answer *after* reasoning instead:

```
What is the capital of Australia? Reason briefly, then give the final answer on its own line.
```

---

## 5. Watch out: the model is confidently wrong sometimes

In this study a logic puzzle produced a trace that asserted *"the dog must be with A"* and then,
in the next sentence, assigned the dog to C — with **zero hesitation markers**. Clean reasoning
does not mean correct reasoning.

Practical guardrails:
- **Make it verify**: `Work it out two independent ways and confirm the answers agree.` This is
  the pattern that worked best — the model genuinely cross-checked its arithmetic.
- **Ask for a check**: `Double-check your arithmetic before answering.`
- **For anything that matters, verify externally.** The 1.5 B student makes silent arithmetic
  slips (it computed 14 × 15.67 = 220.38 instead of 219.38 and shipped the wrong answer).

---

## 6. Using it from the CLI

```bash
# single-shot — the reliable default
./scripts/run_cli.sh "A shop sells 3 notebooks for Rs 47. I buy 14 and pay with Rs 500. Change?"

# interactive chat
./scripts/run_cli.sh
# (type, then Ctrl-D / Ctrl-C to exit)

# tune per question
THREADS=4 MAXTOK=2048 ./scripts/run_cli.sh "Derive the quadratic formula."
```

Reasoning arrives between `[Start thinking]` and `[End thinking]`; the answer follows.

---

## 7. Using it from the REST API

```bash
./scripts/run_server.sh &
```

Then the reasoning and the answer come back **separately** — `reasoning_content` vs `content`:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is 17 * 23? Think step by step."}],
    "max_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.95
  }'
```

The response also carries a `timings` block — `prompt_ms` is your time-to-first-token and
`predicted_per_second` is generation speed:

```json
"timings": {
  "prompt_n": 18, "prompt_ms": 346.0, "prompt_per_second": 52.0,
  "predicted_n": 512, "predicted_ms": 47000.0, "predicted_per_second": 10.9
}
```

From Python:

```python
import json, urllib.request

def ask(prompt, max_tokens=2048):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]

m = ask("What is 17 * 23? Think step by step.")
print("THINKING:", m["reasoning_content"][:300])
print("ANSWER:  ", m["content"])
```

> `urllib` here has no default timeout behaviour you'd want — always pass a generous `timeout`,
> because a long reasoning answer on this hardware can take **100+ seconds**.

---

## 8. Performance tips specific to the Pi 5

| Tip | Why |
|---|---|
| **`-t 2` for chat** | Generation is memory-bandwidth-bound; 2 threads beat 4 by ~13%. |
| `-t 4` when the prompt is long | Prompt processing *does* scale with cores (56 vs 33 t/s). |
| Lower `-c` if you don't need 4k | Smaller KV window → less memory touched per token. |
| Expect thermally-slowed replies | Sustained generation settles ~82–83 °C and costs ~7% throughput. The **first answer after boot is the fastest.** |
| Raise `-n`, not `temperature` | If the answer is cut off or rambling, it's a token budget problem first. |

---

## 9. Prompt patterns that work well

```
# Verify-by-two-methods (the strongest pattern observed)
What is the sum of the first 20 positive even numbers? Work it out two independent
ways and confirm the answers agree.

# Decompose a word problem
A shop sells notebooks at 3 for Rs 47. I buy 14 and pay with Rs 500.
How much change do I get? Show your reasoning step by step.

# Debug/explain code
This Python function should reverse a string but is wrong:
def rev(s): return s[::-2]
Explain the bug and give the corrected one-liner.

# Force a clean final answer
<question>  ...then state the final answer on its own line as "Answer: <value>".
```

---

## TL;DR

`temperature 0.6`, `top_p 0.95`, `-n 2048`, `-t 2`, no few-shot examples, no system persona,
ask for step-by-step reasoning, and ask it to verify by a second method. Then check the answer
yourself — the small model is fluent but not always right.
