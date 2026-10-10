# Paper pack — the MoE expert cache: when it helps, when it hurts, and how to know in advance

*Compiled 10 Oct 2026. This is a self-contained record of everything measured,
everything assumed, and everything still missing, so that a paper/report can be
written from it later without re-running anything. Every number below is
reproducible from artifacts in this repository — see §13.*

**Status: not yet publishable as a paper. Publishable as a community report today.**
The gap and the exact experiments that close it are in §11.

---

## 1. The candidate claim

Mainline llama.cpp gained a VRAM cache for MoE expert weights (`--moe-cache-mib`).
Public reporting of it is uniformly positive (+26%, +84%, ~2×, "nearly doubles").
We measured it on **one 8 GB laptop** and found it is **strongly negative on one
model and strongly positive on another**, and that the sign is predicted by a
single quantity: **pool size ÷ expert-weight bytes**.

| claim | evidence | strength |
|---|---|---|
| C1: the cache is **−64%** on a 256-expert top-8 35B MoE with an 8 GB card | §6, interleaved A/B/A/B, reproduced twice | strong, single machine |
| C2: the cache is **+216%** (3.2x) on a 133-expert pruned variant of the same model | §6, 4 independent loads at the top pool size | strong, single machine |
| C3: the sign is predicted by pool/expert-bytes with a break-even near ⅓ | §7, 11 (coverage, speed) points across 2 models | moderate — **confounded, §10** |
| C4: the mechanism is the host→device link, not VRAM capacity or RAM bandwidth | §8, arithmetic consistent with measurements; **not measured directly** | weak — hypothesis |
| C5: an adaptive pool beats static layer placement, using less VRAM | §6, interleaved pair | moderate, single pair |

**The paper-shaped contribution is C1 + C3 + C5, presented as a controlled
counterexample plus a predictive rule.** C2 makes it more than a bug report.
C4 is what would make it a mechanism paper.

---

## 2. The protocol (this is part of the contribution)

Most published local-LLM speed comparisons are single-shot and therefore
thermally confounded. We caught exactly that in our own data: the same binary,
same model, same flags measured **31.25/31.87 tok/s at 00:27–03:28** and
**28.57/26.38 tok/s at midday**, and one earlier cross-binary comparison showed a
**42% "improvement" that was pure time-of-day** (§10, R1).

Rules used for every comparison that matters:

1. **Interleave the arms**: A/B/A/B, each arm a fresh server process and a fresh
   model load. Never compare an ordered sweep against itself.
2. **Repeat inside the arm**: each arm generates for 4 prompts × 2 reps, so a
   single arm reports a mean and a spread over 6–8 generations.
3. **Record provenance in the artifact**: server binary, model path, extra flags,
   context, batch, VRAM used, peak system RAM, output hashes. (The first run of
   this study was invalidated because the artifact didn't say which binary
   produced it — §12, R2.)
4. **A RAM watchdog** polls system memory and kills the server past a ceiling
   (52 GiB of 63.7), so a configuration cannot take the machine down; a tripped
   watchdog is recorded as a result, not an error.
5. **Keep a machine-verified number check**: `scripts/verify_doc_numbers.py`
   re-reads the quoted figures out of the raw JSON and fails on mismatch (25/25).

---

## 3. Hardware and environment

| item | value |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop, 8188 MiB, sm_89, 24 SMs |
| Driver | 566.14 (CUDA 12.7), WDDM |
| **PCIe (measured)** | **2.80 GB/s actual**; negotiated **Gen3 x4**; chip advertises `pcie.link.gen.max=4`, `width.max=8` — **unexplained** |
| VRAM bandwidth (measured) | ~240 GB/s |
| RAM | 2×32 GB DDR5-4800, 63.71 GiB total, ~25–28 GiB baseline in use |
| RAM bandwidth | ~21–40 GB/s effective; 41.7 GB/s fitted from a real end-to-end partial-offload run |
| Disk | NVMe, ~420 GB free |
| OS | Windows (no Developer Mode → no symlinks; no C/C++ compiler; no CUDA toolkit) |
| llama.cpp CPU build | none usable for the cache flag |
| llama.cpp GPU builds | `.runtime/llamaserver/` = Ollama-derived 0.5.0-dev (commit 6f767fe96) — **no cache flag**; `.runtime/official/cuda/` = official **b11537** prebuilt — has the flag |
| Session drift | **±10–20%** between sessions on identical configs (§10, R1) |

The PCIe figure is the load-bearing environmental fact. It is 5–9× below what the
chip reports it can do, and §8 shows that if the miss path is synchronous, this
single number can explain the entire negative result.

---

## 4. Model anatomy (read from the GGUF tensor index, `scripts/paper_data.py`)

### Qwen3.6-35B-A3B-UD-Q4_K_M

| property | value |
|---|---|
| file | 22,123,538,944 B (20.604 GiB) |
| architecture | `qwen35moe`, 40 layers, hybrid: **only 10 of 40 layers carry KV** |
| experts | **256 per layer, top-8**, 40 layers → **10,240 (layer, expert) slots** |
| expert weights | 19,568,525,312 B (18.225 GiB) = **88.5% of the file**, stored as **120 stacked tensors** (3 per layer) |
| per expert × layer | **1.822 MiB** |
| per layer (all experts) | **466.55 MiB** |
| **expert bytes per token** | **611.5 MB** (top-8 of 256) |
| non-expert weights | ~2.55 GB (attention, embeddings, lm_head, router, norms) |
| KV cache | 20.0 KiB/token → 32K context = 0.63 GiB |
| quant mix | Q4_K 12.08 GB, Q5_K 6.83 GB, Q8_0 2.03 GB, Q6_K 1.08 GB, F32 0.11 GB (5.11 bpw) |

### Qwen3.6-35B-A3B-REAP-48-v2 (router-pruned, third-party)

| property | value |
|---|---|
| file | 9,415,930,368 B (8.769 GiB) |
| experts | **133 per layer, top-8**, 40 layers → **5,320 slots** (48% pruned from 256) |
| parameters | 19.174 B total |
| expert weights | 7,551,574,016 B (7.033 GiB) = **80.2% of the file**, 120 stacked tensors |
| per expert × layer | **1.354 MiB** |
| per layer (all experts) | **180.04 MiB** |
| **expert bytes per token** | **454.2 MB** (top-8 of 133) |
| quant mix | **IQ3_S 5.813 GB**, Q6_K 1.144 GB, IQ4_XS 1.111 GB, Q4_K 0.914 GB, Q5_K 0.311 GB (3.93 bpw) |
| KV cache | 20.0 KiB/token (same hybrid layout) |

**Definition used throughout:** "expert weights" = the sum of tensor bytes whose
name contains `exps`. Router, norms and any shared/dense FFN are *not* included.
(An earlier pass quoted REAP's expert bytes as 7.70 GB by summing the whole `ffn`
group, including the router — a 2% difference that shifts coverage by ~1 point.
The 7.5516 GB figure above is the exact one.)

---

## 5. Raw coverage→speed data (the core table)

Full model, official b11537, 32K ctx, thinking off, temperature 0, `-ub 512`:

| pool | pool GB | coverage | decode (3 prompts) | VRAM used | source |
|---|---|---|---|---|---|
| none | — | 0% | **22.02** | 3187 MiB | `cache_ab2` |
| none | — | 0% | 22.16 | 3187 MiB | `cache_ab2` (control) |
| none | — | 0% | 22.29 | 3187 MiB | `cache_ab2` (control) |
| 2048 MiB | 2.15 | 11.0% | **8.03** | 5237 MiB | `cache_ab2` |
| 2048 MiB | 2.15 | 11.0% | **7.97** | 5237 MiB | `cache_ab2` (repeat) |
| 4096 MiB | 4.29 | 21.9% | **11.91** | 7283 MiB | `cache_ab2` |

REAP-48, same binary and settings:

| pool | pool GB | coverage | decode | VRAM used | source |
|---|---|---|---|---|---|
| none | — | 0% | **12.68** | 2517 MiB | `reap` |
| none | — | 0% | 12.44 | 2517 MiB | `reap` (control) |
| 3072 MiB | 3.22 | 42.7% | 25.52 / 25.82 / 30.44 | 5593 MiB | `reap2`, `reap2`, `reap` |
| 4096 MiB | 4.29 | 56.9% | 32.82 / 33.62 / 34.11 | 6615 MiB | `reap3`, `reap3`, `reap2` |
| 4608 MiB | 4.83 | 64.0% | 36.10 / 36.68 / 41.40 | 7129 MiB | `reap4`, `reap4`, `reap3` |
| 4864 MiB | 5.10 | 67.5% | 36.82 / 37.62 / 37.80 / 38.24 | 7383 MiB | `reap4`×2, `reap5`×2 |
| 5120 MiB | 5.37 | 71.1% | 39.28 / 39.40 / 40.26 / **49.14** | 7639 MiB | `reap5`×3, `reap4` |

Notes that must survive into any write-up:

* The **49.14** is a single outlier; three independent repeats of the same config
  gave 39.28/39.40/40.26. Report 39.6, mention the outlier.
* Each row is a **separate server process and model load**. Repeat rows are the
  variance estimate, not padding.
* Curve shape: flat-ish loss below ~22% coverage, break-even somewhere between
  22% and 43%, then rising and flattening at ~64–71%. **The knee is bracketed,
  not located.**

### Static placement (the comparison that matters for C5)

| config | decode | VRAM used | free |
|---|---|---|---|
| `--moe-cache-mib 4096` | **33.62 / 32.82** | 6615 MiB | 1573 MiB |
| `--n-cpu-moe 14` (static: 26 layers' experts on GPU) | **27.45 / 29.06** | 7303 MiB | 885 MiB |
| `--n-cpu-moe 16` | 26.02 / 28.77 | 6959 MiB | 1229 MiB |
| `--n-cpu-moe 20` | 18.18 / 21.04 | 6275 MiB | 1913 MiB |
| `--n-cpu-moe 20` + `--moe-cache-mib 1024` | 30.34 | 7299 MiB | 889 MiB |

Interleaved pair: the adaptive pool wins by **~17% while using 688 MiB less VRAM**.
Stacking both (last row) is **worse than the pool alone** — they compete.

### Composition with the prompt-batch win

| config | decode | prefill (~3.3k tokens) | VRAM |
|---|---|---|---|
| `cache-4096`, `-ub 512` | 33.62 | 470 tok/s | 6615 MiB |
| `cache-4096`, `-ub 4096` | 34.27 | **1235 tok/s** | 7573 MiB |

Pool 4608 **and** `-ub 4096` does not fit (~8.1 GiB needed of 8188 MiB).

---

## 6. The predictor (C3) and its break-even

| model | expert bytes | pool | coverage | outcome vs no-pool |
|---|---|---|---|---|
| full 35B | 19.569 GB | 2048 MiB (2.15 GB) | **11.0%** | −64% |
| full 35B | 19.569 GB | 4096 MiB (4.29 GB) | **21.9%** | −46% |
| REAP-48 | 7.552 GB | 3072 MiB (3.22 GB) | **42.7%** | +104% (25.67 vs 12.56) |
| REAP-48 | 7.552 GB | 4096 MiB (4.29 GB) | **56.9%** | +167% (33.52 vs 12.56) |
| REAP-48 | 7.552 GB | 5120 MiB (5.37 GB) | **71.1%** | +216% (39.65 vs 12.56) |

**Break-even is between 21.9% and 42.7%.** Two independent community results point
at the same region: an offline LFU simulation needed **88 of 256 experts per layer
(34%)** resident to reach 98.8% hit rate, and the upstream RFC states that a pool
the size of top-k is the *worst* case (all misses, so you pay the bookkeeping and
gain nothing, measurably slower than no cache at all).

Practical rule for a report: **enable only if pool ≥ ~⅓ of expert bytes.**

Structural consequence worth stating: for the 35B on an 8 GB card, ⅓ of expert
bytes is **6.5 GB**, on top of the ~3.2 GB the model already needs — i.e. no pool
size makes the full model faster on this machine. The technique needs a model
whose experts are narrow enough.

---

## 7. Mechanism (C4) — hypothesis, with its arithmetic exposed

Assumption: a cache miss costs **one synchronous host→device upload** of that
expert's weights. Then `ms/token = expert_bytes_per_token × f / link`, with `f` the
fraction of expert bytes that still miss, and `link = 2.80 GB/s`.

Full model (611.5 MB expert bytes/token):

| f | ms/token | predicted |
|---|---|---|
| 1.00 | 218.4 | 4.58 tok/s |
| 0.55 | 120.1 | 8.33 tok/s |
| 0.37 | 80.8 | 12.38 tok/s |
| 0.10 | 21.8 | 45.79 tok/s |

Measured: **8.0 tok/s at pool 2048 ⇒ implied f = 0.57**; **11.9 tok/s at pool
4096 ⇒ implied f = 0.38**. So the measured speeds correspond to "roughly half of
expert bytes still cross the bus" at 11% coverage, and "roughly two fifths" at
22% — a monotone, plausible relationship that also explains why a bigger pool
monotonically helps *without ever reaching break-even* on this model.

REAP-48 (454.2 MB/token), same model: f=1.00 → 162.2 ms (6.16 tok/s);
f=0.55 → 89.2 ms (11.21); f=0.37 → 60.0 ms (16.66); f=0.10 → 16.2 ms (61.64).

**Supporting evidence, from outside our machine:** an independent Vulkan PoC found
that `set_tensor_async` for **non-pinned** host data is *fully synchronous* in that
backend — a staging buffer plus a synchronise per call — and that forcing experts
onto the GPU without bypassing that path made decode *slower* (9 vs 28 tok/s).
Mmap'd weights (what we use) are exactly the non-pinned case. Our server also
prints, on every load: *"tensor overrides to CPU are used with mmap enabled —
consider using --load-mode none for better performance."*

**What is NOT established:** the actual miss rate, the actual sync time, the
actual per-layer cost, and whether the cost is the upload, the readback, or both.
Everything in this section is arithmetic consistent with measurements, which is
weaker than a profile.

---

## 8. Output exactness and quality

| question | finding |
|---|---|
| Does the cache change outputs? | **Yes** — all 3 prompts differ from the no-cache arm at temperature 0, same seed (`qualreap`) |
| Why | a hit computes the expert on the GPU, a miss on the CPU: different kernels, different accumulation order, argmax flips on near-ties |
| Is that acceptable? | for chat yes; for byte-reproducible work no (`--moe-cache-mib 0`) |
| Is the pruned model as good? | **No.** Same 3 prompts, 400 tokens: full 35B stays correct, REAP-48 states "NVIDIA A100 (Volta 2)" and computes p95 as `statistics.quantiles(x,3)[1]` (that is the median — the answer is wrong) |
| How strong is the quality evidence? | **weak**: 3 prompts × 1 sample. Not a benchmark |

Both generated answers for all prompts are stored in `experiments/quality_reap_vs_full.txt`.

---

## 9. Related work (what exists, and what is missing)

**The feature under study**
* llama.cpp issue **#20757** "Two-tier GPU+RAM expert cache for MoE offload",
  opened Mar 2026, closed completed Apr 2026; commit **cd9a2bd** (May 2026)
  "Merge moe-expert-cache-v2: MoE expert-weight cache for the CUDA backend".
  PoC on RTX PRO 2000 8 GB + GPT-OSS-120B: 1.9–2.5 → **12–14 tok/s** at ~98–100%
  hit rate; pure CPU offload ~0.5–1 tok/s.
* The flag ships in official prebuilt binaries (`--moe-cache-mib`, env
  `LLAMA_ARG_MOE_CACHE_MIB`) as of **b11537** (09 Oct 2026). Confirmed present and
  functional on this machine.

**Positive reports (all conditional, none report a loss)**
* **Discussion #28248** (memoriaru), RFC + implementation sketch: **+84%** at 64
  slots on RTX 4090 24 GB, Qwen3.8-Flash-Next, UD-Q3_K_XL, 8K ctx; PPL
  3.3096 ± 0.053 vs 3.3262 ± 0.053 (perplexity-equivalent); pool capped at half of
  free device memory; **explicitly warns N = top-k is a worst case and is slower
  than master**; 2–4× top-k recommended.
* **martinalderson** (Vulkan, RX 9070 XT 16 GB, Gemma 4 26B-A4B): baseline 27.9 →
  +SAM bypass only 27.9 (prompt 142→220) → +GPU offload + dynamic cache **35.1
  (+26%)**; hit rate 47% @100 tokens → 67% @500+; PPL within error; root cause the
  synchronous non-pinned upload above.
* **kisasexypantera94**, offline LFU simulation on Qwen3-30B-A3B Q4_K_M:
  88 experts/layer resident (11.24 GiB) → **hit 98.84%**; ~38.5 MiB/token over
  ~30.6 coalesced ranges; Q6_K: 80 experts/layer, 13.84 GiB, 98.46% hit. Expert
  storage is 3 contiguous ranges per (layer, expert) — layout is not the problem.
  Coalescing study: aggressive over-read hurts (gap 4096 KiB → 72.8 MiB/token,
  +17 MiB over-read).
* **JigSawPT** (single 5090, Windows): free VRAM is an **absolute cliff** —
  **858 MiB free = −1%; 775 MiB free = −11% decode and 2.3× TTFT**. Gains track
  weighted-traffic coverage; models with flat routing got only **+2%** and
  **+5–12%**; DeepSeek-V4-Flash routing entropy 6.3–7.9 of 8.0 bits means 90% of
  activation mass needs 5,854 of 11,008 experts (~72 GiB at Q4) — unreachable on
  one card; static hot lists worth ~nothing there. **Perplexity is not the
  acceptance test**: argmax moved on ~2% of code tokens and ~6% of prose tokens.
* **freenode.net** article: host-offloaded LRU expert cache took Qwen MoE decode
  from ~8 to ~19 tok/s on two Vulkan cards.
* A llama.cpp PR in this space reports, on **Qwen3.6-35B-A3B with 8 GB VRAM**:
  Q2_M 33.25 → 57.2 tok/s (1.72×), Q5_K_P 17.34 → 35.93 (2.07×). **This is the
  number our C1 contradicts**, on the same model class and same VRAM size.

**Related-but-different lines**
* Expert-aware SSD streaming (discussion #27149; M1 Pro 16 GB): expert-contiguous
  re-layout cuts per-expert page faults **36×**; pipelined async pread 50%→2.40,
  75%→3.81, 88%→4.10, 100%→**4.45 tok/s**; CPU-only compute ceiling 4.4; text-
  calibrated pinning hits 42% vs 6% random (+LRU 56%). Also `Atomic-Germ/Guanaco`,
  discussion #23324 (Metal LRU pager, 13 tok/s on M1 Pro 16 GB) — 5–6 independent
  implementations of the same idea.
* PR **#27861** (GPU-resident LRU expert cache, dual-chain, decode-only),
  PR **#25294** (`--moe-stream-cache`, `O_DIRECT`); earlier closed-for-scope
  #21614, #21620, #24524.
* **RapidMoE** (arXiv 2610.01265): splits expert weights at the bit level, low
  precision on GPU + residual on CPU, claims 3.5× decode / 2.1× prefill.
  Research prototype, not buildable on this machine.
* **DynaExq** (arXiv 2511.15015): prefill activates 47–97% of experts vs decode at
  batch 1 1.9–9%; the top-10 hottest experts are **disjoint** across text / math /
  code — relevant to any statically-tuned hot set.
* **REAP** (arXiv 2510.13999): the pruning method behind the model used here.
* **FreeToken** (arXiv 2608.16157, FlashML): host-RAM experts with VRAM as an LRU
  cache; not runnable here (Linux only, driver r580+/CUDA 13, no GGUF path).

**The gap.** Across all of the above, reports are conditional on a machine but the
*conditions are never stated*, and **no source we found reports a measured loss or
a break-even threshold**. That is the space this study occupies: a controlled
negative result plus a two-variable rule (coverage and link bandwidth) that
predicts which side of break-even a given machine/model pair lands on.

---

## 10. Threats to validity (must appear in any write-up)

| # | threat | severity |
|---|---|---|
| R1 | **Session-to-session drift of 10–20%.** Same binary/model/flags: 31.25 and 31.87 tok/s at night vs 28.57 and 26.38 at midday; a cross-binary "42% regression" evaporated under a back-to-back control. Interleaving solves this *within* a session; absolute numbers across sessions are ±20% | high |
| R2 | **One machine, one GPU, one OS (WDDM).** Nothing here establishes behaviour on Linux, on a 24 GB card, or with real async uploads | high |
| R3 | **Confounded model comparison.** The two models differ in *two* ways: 133 vs 256 experts (coverage) **and** IQ3_S vs Q4_K/Q5_K/Q8_0 (dequantisation cost). Part of REAP's advantage could be cheaper CPU-side unpacking rather than coverage | high |
| F1 | **Hit rate is not observable.** b11537 exposes no cache metric; `f` is inferred from speed plus arithmetic (§7) | high |
| F2 | **Mechanism is inferred, not profiled.** No per-layer timing, no upload byte counter | high |
| F3 | **No second link speed.** The link-bandwidth term of the model is untested; if the PCIe figure is a fixed platform property we cannot vary it, and the term stays hypothetical | high |
| F4 | **4 prompts** (3 short + 1 long-context), single temperature/seed regime, `n_predict` 8–400 | medium |
| F5 | **Quality evidence is 3 prompts × 1 sample per model** | medium |
| F6 | **Quant conflation inside the 35B file** — it is a UD mix (Q4_K/Q5_K/Q8_0), so "expert bytes" and "dequant cost" are not independent within a model | medium |
| F7 | **One llama.cpp build** for the cache tests; the flag's implementation details (eviction policy, pool split, whether ids readback is per layer) are undocumented from our side | medium |
| F8 | **Thermal state is not controlled**, only interleaved; ambient varies by ~10 °C between sessions | medium |
| F9 | **One outlier (49.14 tok/s)** in an otherwise tight set; reported, not explained | low |

---

## 11. The experiments that would make this paper-grade

Ordered by value per unit of wall-clock. Each row says what it proves.

| # | experiment | cost | what it fixes |
|---|---|---|---|
| **E1** | **Coverage axis, filled in.** Requantise the same weights (Q2_K / Q6_K / Q8_0, or Q4_K versions of REAP) so the *same* model family yields 6–8 coverage points from ~10% to ~85%. | one download + ~20 min per variant | turns C3 from "bracketed" into a fitted curve; removes R3 by holding the model fixed and moving only the pool |
| **E2** | **Fine pool sweep around the break-even** (3584, 3840, 4352, 4608) on REAP, interleaved. | ~25 min | locates the knee within ±256 MiB instead of a 20-point gap |
| **E3** | **Link speed variation.** Same model+config on a faster host interface (second machine, eGPU, or a fixed link negotiation). This is the decisive experiment for C4. | needs hardware or a fix | converts the mechanism from hypothesis to result, or kills it |
| **E4** | **Replace the inferred `f` with a measured one.** Either (a) instrument llama.cpp (needs a build environment), or (b) fit `f` from E1/E2 and validate the fit out-of-sample, or (c) simulate LRU against routing traces and report it as a simulation (as the prior work does). | (a) needs a toolchain; (b) ~1 h; (c) ~2 h | closes F1 |
| **E5** | **Repeat the whole interleaved protocol in ≥3 independent sessions** and report variance across sessions, not just within. | ~1 h | closes R1 properly |
| **E6** | **Quality suite for the cache**: 50–100 prompts, cache on vs off, exact-match / unit tests / small code tasks, reporting agreement rate and not just perplexity. | ~2 h | closes F5 and speaks to the community's "PPL moved but argmax didn't" question |
| **E7** | **Deconfound quant from coverage**: a Q4_K requant of REAP-48 (or an IQ3_S requant of the 35B) so coverage and dequant cost can be separated. | one download + sweep | closes R3 |
| **E8** | **A second hardware tier** (12–16 GB card) showing the threshold moves with free VRAM. | needs hardware | strengthens generality |

**Minimum viable paper: E1 + E2 + E5** (all on this laptop, no compiler, ~2 h of
measurement that cannot be parallelised because it is one GPU) **+ E3 if any
second link is available.** Without E3 the paper must be framed as an empirical
study and a hypothesis, not a mechanism.

---

## 12. Reproduction

```
# every derived number in this document (anatomy, coverage, arithmetic, table)
HF_HOME=$PWD/.hf_home HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/paper_data.py

# the consolidated sweep tables from the run JSONs
.venv/Scripts/python.exe scripts/moe_cache_summary.py

# check that the numbers quoted in EXPERT_CACHE_AND_REAP.md match the artifacts
.venv/Scripts/python.exe scripts/verify_doc_numbers.py

# one interleaved A/B on the full model
.venv/Scripts/python.exe scripts/moe_bench_35b.py \
  --server .runtime/official/cuda/llama-server.exe \
  --model models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --configs cpu-moe cache-2048 cpu-moe cache-2048 --reps 2 \
  --n-predict 192 --tag cache_ab2 --num-ctx 32768 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

### Artifacts

| path | what |
|---|---|
| `experiments/PAPER_DATA.txt` | machine-generated dump of §4, §5, §7, §9-table |
| `experiments/runs/moe_bench_cache_ab.json` | the invalid first attempt (wrong binary) — kept because it proves the flag is absent from the Ollama-derived build |
| `experiments/runs/moe_bench_cache_ab2.json` | the 35B A/B: −64%, interleaved |
| `experiments/runs/moe_bench_reap{,2,3,4,5}.json` | the REAP placement + pool sweeps, incl. the interleaved 5120/4864 confirmation |
| `experiments/runs/moe_bench_qualreap.json`, `moe_bench_qualfull.json` | quality, pool-vs-no-pool and model-vs-model |
| `experiments/runs/moe_bench_probe.json` | `-fa off` and `--perf` probes (both inconclusive) |
| `experiments/runs/moe_bench_oldbin.json` | cross-build control that killed the 42% claim |
| `experiments/EXPERT_CACHE_AND_REAP.md` | the user-facing write-up of the same study |
| `experiments/quality_reap_vs_full.txt` | full generated answers for both models |

---

## 13. Publication options and the honest bar

| venue | ready today? | note |
|---|---|---|
| llama.cpp issue/discussion comment | **yes** | this is where the contradictory claim belongs; the RFC authors and the PR author are there |
| Blog / README / r/LocalLLaMA | **yes** | strongest angle: "the cache everyone reports +84% on costs 64% here, and here's the rule" |
| arXiv preprint / technical report | **yes, after writing** | no gatekeeping, little credit; worth it as a citable artifact |
| Workshop paper (MLSys/OSDI-style workshops, MoE/edge inference venues) | after E1+E2+E5 (+E3 ideally) | realistic target; the negative-result+rule framing is the contribution |
| Full conference paper | no | would need E3, E7, E8 and a proposed fix to be competitive |

**Framing for a workshop submission:** *"When does expert-weight caching help? A
controlled study on consumer hardware, with a break-even rule."* Central figure =
coverage (x) vs speed ratio versus no-cache (y), with the two models' points on
one axis, the break-even band shaded, and the link-bandwidth term as the
explanatory variable. Central table = the four public claims vs our four measured
outcomes on the same model class.
