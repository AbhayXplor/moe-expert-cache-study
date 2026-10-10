# The expert cache: a 3x win on one model, a 3x loss on another

*Measured 10 Oct 2026 on the RTX 4060 Laptop (8188 MiB), llama.cpp **b11537**
(official prebuilt, CUDA 12.4), 32K context, thinking off, temperature 0.*

## The one-paragraph version

llama.cpp mainline now ships `--moe-cache-mib N`: a pool of VRAM that holds the
*hot* MoE experts so they do not have to be re-read from system RAM every token.
I tested it on both models. On the **full 35B** it is a **disaster — 22.2 → 8.0
tok/s (−64%)** — and on the **REAP-48 model it is a breakthrough — 12.6 → 39.6
tok/s (+216%, i.e. 3.2x, best single run 45.6)**. The difference is not luck and not thermal: it is
**how much of the expert weights the pool can actually hold**. On the 35B the
pool is 11–22% of the expert bytes, far too little to hit anything, so every token
pays the lookup cost and gains nothing. On REAP the same VRAM holds 43–71%, most
lookups hit, and the CPU stops doing the work that was the bottleneck. **Rule of
thumb: only turn this flag on when the pool can hold about a third or more of the
expert weights.** Below that it is strictly slower than doing nothing.

## Setup — why these numbers are trustworthy

Everything in the tables below was measured with the **same binary, same model
file, same flags and the same session**, with the configurations **alternated
(A/B/A/B)** wherever they are being compared. That matters because this laptop
drifts by 10–20% between sessions (a config measured at 12:10 gave 22.0 tok/s and
the same config measured at 12:40 gave 26.3 tok/s). Within-session interleaved
comparisons do not have that problem; absolute numbers quoted across sessions do,
and are marked as such.

Two things had to be fixed first:

* The staged Ollama-derived `llama-server.exe` **does not have the flag at all**
  (`error: invalid argument: --moe-cache-mib`). The official b11537 prebuilt does.
  The harness (`scripts/moe_bench_35b.py`) now takes `--server` and records both
  the server path and the model path in every result file.
* b11537 loads this model fine (`qwen35moe` is supported) and exposes **no
  cache/hit-rate metrics** — `/metrics` has prompt-cache counters only. So the
  evidence that the pool engaged is the VRAM arithmetic (requested 2048 MiB →
  measured +2050 MiB) plus the speed and the output hashes.

## Result 1 — the full 35B: the cache makes it 2.8x slower

Interleaved A/B/A/B, one full model load per arm, 32K ctx, ~3.3k-token prompt:

| config | VRAM used | free | decode (3 prompts) | vs baseline |
|---|---|---|---|---|
| `cpu-moe` (all experts in RAM) | 3187 MiB | 5001 | 22.02 | — |
| `--moe-cache-mib 2048` | 5237 MiB | 2951 | **8.03** | **−64%** |
| `cpu-moe` (repeat) | 3187 MiB | 5001 | 22.16 | — (control) |
| `--moe-cache-mib 2048` (repeat) | 5237 MiB | 2951 | **7.97** | **−64%** |
| `cpu-moe` (repeat) | 3187 MiB | 5001 | 22.29 | — (control) |
| `--moe-cache-mib 4096` | 7283 MiB | 905 | 11.91 | −46% |

The two cache arms reproduce to 0.7% and the three controls to 1.2%, so this is
not drift. Note it is **not** a VRAM shortage: at 5237 MiB there are still
2.9 GB free, and the pool that is *closer* to the wall (4096, only 905 MiB free)
is the *faster* of the two cache arms. More pool = more hits = less bad.

## Result 2 — REAP-48: the same flag is worth 3x

REAP-48 is the router-pruned variant of the same model: **19.17 B parameters,
9.43 GB, 133 experts instead of 256, 3.93 bpw average (mostly IQ3_S)**. Its expert
tensors are only **~180 MB per layer instead of ~467 MB**, which is the whole
reason the pool works here.

| config | VRAM used | free | decode mean | range |
|---|---|---|---|---|
| `cpu-moe` (baseline, no pool) | 2517 MiB | 5671 | 12.6 | 10.9–18.1 |
| `--moe-cache-mib 3072` | 5593 MiB | 2595 | 25.5 / 25.8 / 30.4 | 22.7–34.7 |
| `--moe-cache-mib 4096` | 6615 MiB | 1573 | 33.6 / 32.8 / 34.1 / 34.3 | 28.8–36.7 |
| `--moe-cache-mib 4608` | 7129 MiB | 1059 | 36.1 / 36.7 / 41.4 / **45.6** | 28.9–46.9 |
| `--moe-cache-mib 4864` | 7383 MiB | 805 | 37.6 / 36.8 / 37.8 / 38.2 | 29.9–41.5 |
| `--moe-cache-mib 5120` | 7639 MiB | 549 | **39.3 / 39.4 / 40.3** | 31.3–46.8 |
| `--n-cpu-moe 14` (static placement) | 7303 MiB | 885 | 31.5 / 27.5 / 29.1 | 24.6–38.0 |
| `--n-cpu-moe 16` | 6959 MiB | 1229 | 25.1 / 26.0 / 28.8 | 22.7–36.6 |
| `--n-cpu-moe 20` | 6275 MiB | 1913 | 18.2 / 21.0 | 16.8–21.5 |
| `--n-cpu-moe 20 --moe-cache-mib 1024` | 7299 MiB | 889 | 30.3 | 28.2–34.2 |

Three separate things to read out of that table:

1. **The pool curve has a knee around 4096–4608 MiB and then plateaus at ~38–40
   tok/s.** 3072 → 25.7, 4096 → 33.7, 4608+ → 38–40. Below the knee you are paying
   for a cache that misses; above it you are just buying hit rate.
2. **The adaptive pool beats static placement while using less VRAM.** Directly
   interleaved, `cache-4096` (6615 MiB used) beat `ncmoe-14` (7303 MiB used)
   **33.6 / 32.8 vs 27.5 / 29.1**. Putting the *hot* experts on the card is
   better than putting the *first N layers* of experts there, and it costs
   **1.4 GB less VRAM** because the pool does not have to be a whole number of
   layers.
3. **Stacking the two wins is worse than either alone**: `ncmoe-20` +
   `cache-1024` = 30.3, below `cache-4096` alone. They compete for the same VRAM
   and a 1 GB pool is under the knee.

### Combining with the prompt-processing win

`-ub 4096` (the setting that fixed prompt processing, see MOE_35B_RESULT.md)
inside the cache config, REAP-48:

| config | decode | prefill (~3.3k tokens) | VRAM |
|---|---|---|---|
| `cache-4096` + `-ub 512` | 33.6 | 470 tok/s | 6615 MiB |
| `cache-4096` + `-ub 4096` | 34.3 | **1090–1235 tok/s** | 7573 MiB |

So the two wins compose cleanly on REAP: **~34 tok/s decode with ~1.1k tok/s
prefill**, at 7573 of 8188 MiB. (Pushing the pool to 4608 *and* `-ub 4096` does
not fit: it would need ~8.1 GiB.)

## Result 3 — the rule that explains both results

Express the pool as a fraction of the model's total expert bytes:

| model | expert bytes | pool | % of expert bytes | outcome |
|---|---|---|---|---|
| full 35B | 19.57 GB | 2048 MiB (2.15 GB) | **11%** | −64% |
| full 35B | 19.57 GB | 4096 MiB (4.30 GB) | **22%** | −46% |
| REAP-48 | 7.552 GB | 3072 MiB (3.22 GB) | **42.7%** | +104% |
| REAP-48 | 7.552 GB | 4096 MiB (4.30 GB) | **56.9%** | +167% |
| REAP-48 | 7.552 GB | 5120 MiB (5.37 GB) | **71.1%** | +216% |

The switch-over sits between 21.9% and 42.7% of expert bytes. (Expert bytes are
the sum of tensors named `*exps*`, read from the GGUF index by
`scripts/paper_data.py`; the router and norms are excluded.) That lines up with two
independent community measurements from the research pass: kisasexypantera94's
offline LFU simulation needed **88 of 256 experts per layer (34%)** resident to
reach 98.8% hit rate, and the `--moe-expert-cache` RFC explicitly warns that a
pool the size of top-k is the *worst* case — every lookup misses, so you pay the
per-layer id readback sync and gain nothing, which is measurably slower than not
having a cache. Our 35B result is that worst case with a big pool: 11–22% resident
means nearly every expert lookup still has to go to RAM, and the lookup machinery
becomes the cost.

**Practical rule: compute `pool / expert_bytes` before enabling this flag. Below
~⅓, leave it off.**

## Result 4 — quality: the cache changes answers

The cache is **not bit-exact**, for the same reason MTP is not: a cache hit
computes the expert on the GPU (different kernel, different accumulation order)
while a miss computes it on the CPU. Same prompt, same seed, temperature 0:

| prompt | pool vs no pool | REAP vs full 35B |
|---|---|---|
| prose | different | different |
| code | different | different |
| security | different | different |

That is expected and fine for near-ties, but it means **anyone who needs
byte-reproducible output should leave the pool off** (`--moe-cache-mib 0`), exactly
as they would leave `--spec-type draft-mtp` off.

### And REAP-48 itself is weaker than the full 35B

Same three prompts, 400 tokens each, greedy, same session
(`experiments/quality_reap_vs_full.txt`):

| prompt | full 35B (34.66 B, 5.11 bpw) | REAP-48 (19.17 B, 3.93 bpw) |
|---|---|---|
| prose (explain the 80x RAM-vs-VRAM claim) | challenges the premise, separates latency from bandwidth, tabulates real numbers | accepts the premise, then states "NVIDIA A100 (Volta 2)" and "eCPU" — **factual errors** |
| code (percentiles + rolling 60 s average) | writes a working interpolation-based `percentile()` helper | defines the rolling window *better* than the full model, then computes p95 as `statistics.quantiles(x, 3)[1]` — **that is the median, the answer is wrong** |
| security (fingerprint a TLS service on 8443) | well-ordered: `nc`/`nmap`, `openssl s_client`, cert chain, cipher enumeration | also good: `ss -tlnp`, `/proc/<pid>/exe`, `lsof` — slightly less structured |

Both are competent; the full 35B is consistently better, and on the code prompt
REAP's bug is the kind that matters. **REAP-48 is a speed option, not a
quality-equivalent replacement** — and at 3.93 bpw it is also below the Q4 floor
this project set for a daily-driver model.

## What this means for the frontier

Same binary, same session, 32K context, RAM ceiling respected:

| model | weights | best config | decode | prefill | VRAM | peak RAM |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (full) | 22.13 GB, 5.11 bpw | `cpu-moe` (or +MTP, +ub4096) | 26–31 tok/s | 587 tok/s @ub4096 | 3203 MiB | 46 GiB |
| Qwen3.6-35B-A3B-REAP-48 | 9.43 GB, 3.93 bpw | `cache-4096` + `ub4096` | **34 tok/s** | **1090 tok/s** | 7573 MiB | 33 GiB |
| Qwen3.6-35B-A3B-REAP-48 | 9.43 GB, 3.93 bpw | `cache-4608`/`5120` | **39–40 tok/s** (45.6 best) | 540–640 tok/s | 7129–7639 MiB | 33 GiB |

REAP-48 with the pool is **~1.3–1.5× faster than the full 35B while being 57%
smaller**, uses 13 GB less RAM, and still leaves the 8 GB card with headroom. It
gets there by spending the VRAM on the *right* experts. On the full 35B the same
idea is impossible on this card — the pool would need ~6.5 GB *of expert bytes
alone* (⅓ of 19.57 GB) on top of the 3.2 GB the model already needs, i.e. ~9.7 GB
on an 8 GB card. So:

* **No cache size makes the full 35B faster on this GPU.** It is the wrong shape
  of model for this technique: 256 experts is too many for the pool to cover.
* The cache's benefit **needs a pruned or narrower-expert model**. That is a
  genuinely new insight from this session, and it points somewhere: models
  designed so that ~⅓ of their experts fit in the card's spare VRAM (REAP-style
  pruning, fewer/larger experts, or expert-level mixed precision that shrinks the
  cold half) get a 3× decode speedup for free from stock llama.cpp today.

## Caveats and open questions

* **Absolute numbers drift 10–20% between sessions** on this laptop. Every
  comparison in this document that matters was interleaved; the cumulative table
  mixes sessions and should be read as ±15%.
* `cache-5120` reported **49.1 tok/s** once and **39.3–40.3** three times. The 49
  is an outlier and is reported but not relied on.
* An earlier session suggested the official b11537 build was ~40% slower than the
  staged Ollama-derived build on this model (22.0 vs 31.3 tok/s). A **back-to-back
  control disproved it**: minutes apart, same config, the old build gives 28.57 /
  26.38 and b11537 gives 26.30 / 26.38. The two builds are **equivalent on this
  model (27.5 vs 26.3, well inside drift)**; the 40% was the time of day. Use
  whichever build has the flag you need.
* **`-fa off` is not a reliable win.** On b11537 it measured 28.9 vs 26.3 for
  `-fa on` (+10%), but on the staged older build the same change measured 24.0 vs
  27.5 (**−13%**). One sample each, opposite signs ⇒ **inconclusive**, and it is
  not free anyway: `-fa off` costs ~850 MiB of extra VRAM (4049 vs 3203 MiB).
  Do not act on it without an interleaved test of your own.
* `--perf` produced no extra backend breakdown in this build (25.2 vs 26.3
  control — noise). The "CPU-side expert math is the bottleneck" conclusion is
  still **inferred**, though the REAP result strengthens it: making the pool big
  enough to remove CPU expert work is exactly what buys the 3×, and making it too
  small to remove any is exactly what costs 2.8×.
* With a read-mostly pool, REAP is still fast at **549 MiB free VRAM**, where
  static layer placement on the full model fell off a cliff at 287 MiB free. The
  WDDM rail is **access-pattern dependent**, not a single number. Not explained;
  do not generalise it.
* The pool's own hit rate is **not observable** — b11537 exposes no metric for it.
  The hit-rate claims here are inferred from the speed curve and the byte
  arithmetic, not read off a counter. That is the single biggest gap to close.

## Reproduce it

```
# the sweep tables
.venv/Scripts/python.exe scripts/moe_cache_summary.py

# check every number in THIS document against the raw JSON it came from
.venv/Scripts/python.exe scripts/verify_doc_numbers.py

# one interleaved A/B, full model
.venv/Scripts/python.exe scripts/moe_bench_35b.py \
  --server .runtime/official/cuda/llama-server.exe \
  --model models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --configs cpu-moe cache-2048 cpu-moe cache-2048 --reps 2 \
  --n-predict 192 --tag cache_ab2 --num-ctx 32768 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'

# the winning REAP-48 configuration, as a daily driver
.runtime/official/cuda/llama-server.exe \
  -m models/Qwen3.6-35B-A3B-REAP-48-v2-GGUF/Qwen3.6-35B-A3B-REAP-48-v2.gguf \
  -ngl 999 -ot exps=CPU --moe-cache-mib 4608 -b 4096 -ub 4096 \
  -c 32768 -fa on -np 1 --metrics
```

Artifacts: `runs/moe_bench_cache_ab.json` (old binary, proves the flag is absent
there), `cache_ab2` (35B A/B), `reap`, `reap2`–`reap5` (REAP sweeps),
`qualreap`/`qualfull` (quality), `probe` (`-fa off`, `--perf`), `oldbin`
(cross-build control). Logs in `logs/moe_*`.
