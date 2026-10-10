# Findings so far — everything measured, in plain English

One-page summary of the whole investigation, plus what to look at next.
Written so it can be read without knowing the jargon.

* Detail on the deep technical work: [RESULTS.md](RESULTS.md) (lab notebook).
* Detail on the Ollama tests and the "3 wastes": [REPORT_ollama_and_wastes.md](REPORT_ollama_and_wastes.md).
* Literature we read: [../research/litreview_pass1.md](../research/litreview_pass1.md).

Everything below was **measured on this laptop** unless it says **[quoted]**.
Nothing here is copied from a blog post as if we had measured it.

---

## 1. The question we started with

> Run a *bigger* model on this 8 GB card, without losing speed and without losing quality.

Your own correction, which is the right framing: **fitting was never the problem.**
A 100B model already "fits" in 8 GB if you're willing to run it at 1 token/second.
The real problem is fitting + fast + same quality, all at once.

## 2. The machine's real limits (measured, not spec sheets)

| | |
|---|---|
| GPU | RTX 4060 Laptop, 8188 MiB, 68.5 W cap, **no throttling** (all throttle flags inactive) |
| Achievable bandwidth | **~240 GB/s** read, ~250 GB/s write (spec is 256) |
| Usable VRAM for a model | **7.70 GiB** — proven by a real run that peaked at 7878 of 8188 MiB |
| L2 cache | 32 MiB |
| Fixed overhead | ~0.35 GiB (context, scratch, driver) |

So the budget is: **7.70 GiB total, minus ~0.35 overhead, minus the KV cache, = weights.**

## 3. What we now know, in order of how much it matters

### 3.1 The single biggest win was free: Ollama was leaving 24% on the table

Ollama picks how many layers go on the GPU **by policy, not by free memory.**
At 32K context it chose 76% GPU and ran 23.4 tok/s. Force all 33 layers and the
same model runs **40.6 tok/s**.

| setting (32K ctx) | layers on GPU | tok/s |
|---|---|---|
| Ollama default | 76% | 23.43 |
| forced `num_gpu 33` | 100% | **40.63 (+74%)** |

Saved as a model: `ollama run qwen3.5:9b-fast` (num_gpu 33, num_ctx 65536).
Re-verified on clean VRAM: **39.70 tok/s @ 32K, 100% GPU, peak 7267 MiB.**
At 64K context: 40.35 tok/s, peak 7878 MiB — long context is nearly free here.

### 3.2 This model is a hybrid, and that is why long context is cheap

Only every 4th layer does attention (8 of 32 layers); the other 24 are Mamba-style
state that **does not grow with context**.

| | KV per token | @32K | @64K |
|---|---|---|---|
| qwen3.5:9b (hybrid, f16) | 32 KiB | 1.0 GiB | 2.0 GiB |
| a classic dense 9B (f16) | 128 KiB | 4.0 GiB | 8.0 GiB |

A classic dense 9B at 32K would need 9.06 GiB and **not fit at all**.
Architecture, not parameter count, is where the headroom is on this card.

### 3.3 Bandwidth is not the problem — that lever is already closed

Decode time splits cleanly: the weight term is **flat at 24.5–24.9 ms/token =
~227 GB/s ≈ 95% of what this card can do**, and the KV term is exactly linear
(32 KiB/token ÷ 240 GB/s). An earlier claim of ours that 165 GB/s was being
achieved, leaving a 1.4× opportunity, **was wrong** — that was our test harness,
not the runtime. Negative result; this lever is off the list.

### 3.4 The trap that makes the same config report 5 tok/s instead of 40

Ollama reads free VRAM **at load time** and caches it. If anything else holds VRAM
(leftover `llama-server.exe`, a game launcher), the stale reading pushes layers to
the CPU: measured **5.09 tok/s at 93% CPU** with 6051 MiB still resident.
Three orphaned `llama-server.exe` (2.2 GB) turned a 40.9 tok/s run into "16 tok/s".
`ollama /api/ps` shows nothing; use `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.
**Any benchmark made with dirty VRAM is worthless.**

### 3.5 Lossless compression works, is bit-exact, and is *slower* — here is exactly why

We built the DF11 codec ourselves on Qwen3-1.7B (not just downloaded it):

| | bf16 | DF11 (lossless) |
|---|---|---|
| size | 3.441 GB | **2.538 GB (1.356×)** |
| exactness | — | **28/28 blocks bit-exact, max logit difference 0.0** |
| GPU time per token | 28.72 ms | **66.68 ms (2.32× slower)** |

Why slower: it writes a bf16 copy of the weights to scratch memory, then reads it
back. Per token it moves **2.20× the bytes** (1.92 GB compressed read + 2.82 GB
scratch write + 2.82 GB scratch read, vs 3.44 GB plain). The entropy decode itself
is only **~19–20%** over a pure copy — the loss is all in the extra passes.

**So the fix is not a better codec. It is removing the scratch round-trip.**

### 3.6 There is almost nothing left to squeeze out of raw bf16 (0.9%)

Measured on our own weights with an entropy probe:

| form | stored bits/weight | real content (entropy) | removable |
|---|---|---|---|
| bf16 | 16 | 10.57 | **0.9%** — lossless is capped at ~1.36× |
| int8 quantized | 8 | 7.195 | 10.1% |
| int6 quantized | 6 | 5.180 | 13.7% |
| **int4 quantized** | **4** | **3.124** | **21.9%** |

This kills a whole direction ("add a smarter entropy layer to bf16") before any
work is spent on it. It also shows where the real target is: **after** you quantize,
22% of a 4-bit file is pure redundancy.

### 3.7 The lossy comparison point (4-bit NF4)

| format | VRAM | GPU ms/token | matches bf16 output |
|---|---|---|---|
| bf16 | 3.20 GiB | 28.72 | — |
| DF11 (lossless) | 2.36 GiB | 66.68 | **64/64 tokens, logits bit-exact** |
| NF4 (4-bit) | **1.24 GiB** | **25.65** | **9/64 tokens**, diverges at token 8 |

4-bit is smaller *and* faster on GPU — and it visibly damages the output.
The two axes (lossless vs lossy) trade different things; they are not competitors.

### 3.8 The main new result: fusing the decode into the math kernel flips the sign

Built from scratch in Triton (`scripts/fused_decode_gemv.py`), on real weights,
whole-token (28 layers × 7 projections = 196 kernels in one CUDA graph, 2.82 GB of
distinct weights so it cannot hide in cache):

| variant | ms/token | bytes read per weight |
|---|---|---|
| bf16 (plain, cuBLAS) | 13.32 | 2.00 |
| unfused (dequant → bf16 scratch → cuBLAS) | 15.99 | 6.52 |
| **fused (one kernel: unpack in registers, then multiply)** | **5.91** | **0.52** |
| floor (read the payload, discard it) | 3.15 | 0.52 |

* **2.25× faster than bf16, 2.71× faster than unfused** — compression goes from
  *slower* to *faster*, purely by removing the scratch round-trip.
* Correctness: worst relative error **2.2e-07** vs an fp32 reference — that is
  just the order the additions happen in, i.e. the unpacking is exact.
* It captures ~54% of the theoretical win (floor 3.15 ms). ~2.8 ms/token is still
  in-kernel decode/dispatch overhead — that is the tuning target.
* **Important caveat:** this is a *whole-token* number. Measured one kernel at a
  time, the fused kernel looks *worse* (14.3 vs 12.5 ms). A single projection
  cannot fill this GPU; pipelining 196 of them can. Never quote the 2.25× at a
  single layer.

### 3.9 The third lever (read fewer weights) is real but narrow

Speculative decoding via n-gram lookup, greedy, 100% GPU (tok/s):

| config | free-form prose | copy-heavy text | repetitive JSON |
|---|---|---|---|
| off | 38.84 | 39.74 | 40.00 |
| best n-gram setting | 37.51 | **47.17 (+18.7%)** | 42.69 |

Pays only where text repeats itself, and is **negative on prose** (rejected drafts
are not free). Ceiling here is **+13–19%, not 2×**. A real draft model would do
better, but this model's tokenizer has 248,320 tokens and no off-the-shelf small
partner shares it.

Honest note on outputs: each config is deterministic, but spec decode does **not**
always reproduce the baseline bytes. Two different configs produced the *same*
alternative output, diverging at the same character on a low-information filler
field (`"category": "Storage"` → `"Cache"`). Batched verification sums in a
different order, and a near-tie flips. **Same quality, different bytes.**

### 3.10 Why your 16B run went from 28 to 4 tok/s (arithmetic, not guesswork)

A 16B dense model needs **8.38 GiB of weights alone** at 4.5 bits — over the
7.70 GiB budget before a single token of context. Stacking every lever we have:

| 16B dense, 16K ctx | weights | KV | total | verdict |
|---|---|---|---|---|
| today: 4.5 bpw, f16 KV | 8.38 | 2.25 | 10.98 | NO (+3.28) |
| + q8_0 KV | 8.38 | 1.12 | 9.86 | NO (+2.16) |
| + entropy-coded weights (−21.9%) | 6.55 | 2.25 | 9.15 | NO (+1.45) |
| + both | 6.55 | 1.12 | 8.02 | **NO (+0.32)** |

You'd need **2.55 bits/weight (~3-bit)** to fit it — the quality loss you're
avoiding. **The 22% waste budget is not big enough to close the 16B gap, even if
the kernel existed.**

Your 4 tok/s, predicted exactly: 16B at 3-bit ≈ 7.03 GiB + 0.35 overhead leaves
~0.32 GiB for KV = **~2,200 tokens**. The model loads and looks healthy, then your
first real message walks past 2K tokens, the cache spills to RAM, and decode
collapses. That is the mechanism.

---

## 4. Honest ledger: what is new here, and what is not

You were right to call this out, so it is written down plainly.

**Reproduced known techniques (not discoveries):**

* Fused 4-bit dequant-inside-GEMM is *already shipped*: llama.cpp Q4 kernels,
  vLLM Marlin/Machete, ExLlama, TensorRT-LLM all do it. Our E6 result confirms it
  works on this card and gives us our own clean number — it is not news.
* The 21.9% post-quantization redundancy reproduces a published Shannon-bound
  result ("Approaching Shannon Bound with Lossless LLM Weight Compression"),
  5 results for 5.
* DF11's ~2× slowdown at batch size 1 is what the DF11 authors themselves state.

**Actually measured here, that we could not find already published:**

* Ollama offloads by **policy, not necessity** — +74% from one setting, on this model.
* Ollama's **stale free-VRAM cache** silently moves layers to CPU (5.09 vs 40.6 tok/s).
* The hybrid `qwen3.5:9b` KV cost is **32 KiB/token** (not the 128 KiB a dense model
  of that size implies), and its 64K context is nearly free at 100% GPU.
* The weight term is **flat at ~227 GB/s = 95% of achievable** — a negative result
  that removes a whole lever from the plan.
* Lossless bf16 headroom is **0.9%** on our weights — kills the entropy-layer branch.
* DF11's slowdown is **entirely the scratch round-trip, not the codec** (codec
  overhead is only ~19–20%).
* The fused/unfused sign flip measured end to end, with the **per-kernel vs
  whole-token** trap documented.
* The 16B arithmetic that **predicts your 4 tok/s** from first principles.

**Not new, but useful:** three measurement hazards we hit and fixed, each of which
would have produced a confident wrong answer — L2 residency (measured 486–730 GB/s
on a ~240 GB/s card), dirty write-backs during the timed kernel, and VRAM
oversubscription (a tidy, completely wrong 401 ms/token). All are written up in
[RESULTS.md](RESULTS.md) so they can't bite us again.

---

## 5. Dead ends — do not re-walk these

| direction | verdict | why |
|---|---|---|
| better lossless entropy coding of bf16 | **dead** | 0.9% headroom, measured on our own weights |
| unfused decompress-then-multiply, any codec | **dead** | always pays 2× bytes; measured 2.32× slower |
| getting more bandwidth from the runtime | **dead** | already at 95% of the card's limit |
| making a 16B dense model fit at good quality | **dead (arithmetic)** | needs 2.55 bpw even with every lever stacked |
| DF11 as a speed path | **dead** | it is a size path; its own authors say ~2× slower at batch 1 |
| pairing a small draft model for spec decode | **blocked** | no model shares this 248,320-token vocabulary |
| building ZipServ's kernels as-is | **blocked** | no C++ compiler or CUDA toolkit on this machine |

---

## 6. Next directions — the options on the table

Six real options, ordered by "how much do we learn per hour spent".
Effort and honest status are stated for each — none of these is sold as a
certainty.

### Option 1 — Measure the one number that decides if the 22% is real *(~half a day)*

The 21.9% saving on a 4-bit file comes from **variable-length** coding: common
values get short codes, rare values long ones. That is exactly what a fast kernel
cannot do — inside a kernel, every weight must start at a **known offset**, so the
code must be **fixed-length**.

So: if you force the codec to be fixed-length, how much of the 21.9% survives?
The answer is somewhere between 0% and 21.9%, and **nobody we found has published
that specific number** — the published figure is the unconstrained upper bound.

* Tools already on disk: Triton, the weights, the entropy probes.
* Either answer is a real result: ~18% → the direction is alive and worth building.
  ~3% → the direction is honestly dead and we stop guessing.
* This is the cheapest way to buy information, and it is the one genuinely open
  question left from this whole investigation.

### Option 2 — Fight a real 4-bit kernel, not cuBLAS *(~1 day)*

Our 2.25× was measured against cuBLAS **reading plain bf16** — a fair comparison
for "does fusion help", but a *strawman* for "is our kernel good": llama.cpp's
Q4_K kernels already do fused dequant and are hand-tuned.

* `llama-server.exe` with the CUDA backend is already staged on disk — the
  head-to-head is cheap.
* If we win or match → the technique is worth continuing to build on.
* If we lose → the E6 work is engineering practice, not a path to new speed, and
  we should stop treating it as one. **Better to know this now.**
* This is a reality check, not a discovery. Its value is that it protects weeks.

### Option 3 — Build a lossless codec that can be fused *(weeks)*

The only design that wins on **both** axes at **zero quality loss** is lossless
compression decoded *inside* the kernel. Currently impossible: the compressed
format is variable-rate and its values are not addressable, so it can only be
materialised into scratch memory first — which is the whole 2.32× penalty.

* Fixing that means redesigning the format to be fixed-rate and branch-free
  (the ZipServ / TCA-TBE direction). Payoff if it works: **1.36–1.5× smaller and
  faster than bf16, bit-exact.**
* Honest label: a paper already describes how. Nobody *ships* it. You would be the
  first to have it working locally — not the first to think of it. That is the
  honest version of "new".
* Cost: weeks, and it needs a real kernel. Triton makes it possible without a
  C++ compiler.

### Option 4 — Buy architecture instead of compression *(~half a day, highest certainty)*

The measured structural win on this card is **architecture**: the hybrid 9B runs
100% GPU at 64K context and 40 tok/s, while a classic dense 9B of the same size
needs 9.06 GiB and does not fit at all. That is free speed and free context, already
proven, not theory.

* Concrete next step: test a **12B-class hybrid** at 4.5–5 bpw, 100% GPU, and find
  its real max context (a 12B *hybrid* lands at ~7.2–7.6 GiB and should fit; the
  same 12B *dense* does not).
* That answers "can I run a bigger model here" with **+3B more parameters than you
  run today** — which is your original question, answered without any new code.
* This is procurement + measurement, not research. Highest certainty of the six.

### Option 5 — Shrink the cache instead of the weights *(~half a day)*

Your 16B run died because the **KV cache** spilled to RAM, not because the weights
were wrong. We only measured one KV setting: f16 (today's default) and q8_0 (costs
~10% speed, only pays at 128K).

* Next: test whether the staged `llama-server.exe` accepts **q4_0/q4_1 KV**
  (`--cache-type-k/-v`), which Ollama itself refuses to expose. If it works, long
  context stops being the thing that breaks. Needs flash attention, which the
  bundled build already enables.
* Honest label: this is mostly known upstream work, but our own numbers for it are
  missing, and it targets the exact failure you hit.

### Option 6 — Read fewer weights per token *(1 day measured / open-ended)*

Everything above makes the model smaller. This option makes the model **not read
itself** — the only family of techniques that gives "bigger model, same speed" **by
design** rather than by squeezing.

* **(a) Speculative decoding** — measured already: real, but narrow (+13–19% on
  repetitive text, negative on prose). The blocker for the good version (a real
  draft model) is the 248,320-token vocabulary. A version that dodges the blocker:
  **self-speculation** — use the model's own early layers as the draft, so no second
  model and no tokenizer match is needed.
* **(b) Mixture-of-Experts** — the plain-English explanation you never got: MoE
  splits each layer into many *experts*, and only 1–2 are used per token. So the
  file can be 4× bigger than the per-token work — the architecture where "big model,
  small fast path" is true by construction. The catch on 8 GB is *where the unused
  experts live*: in VRAM they eat the budget you are already short of; in system RAM
  they are slow, which is exactly what you saw. So the honest version of this option
  is a MoE whose **active** footprint plus its hot experts fit inside 7.7 GiB.

---

## 7. Recommendation, if you want one

Do **Option 1 first** (half a day, decides whether the 22% is usable at all) and
**Option 2 second** (one day, decides whether our kernel is a path or a curiosity).
Both are cheap, both are falsifiable, and together they stop us from building for
a week on top of an assumption that might be wrong.

**Option 4 is the one with a guaranteed payoff** and can run in parallel — it needs
no new code and immediately answers "how much model can this card actually carry".

---

## 8. House rules for any future measurement

Learned the hard way; each of these produced a wrong answer first time round.

1. **Drain VRAM before benchmarking.** Kill orphan `llama-server.exe`, check
   `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.
   Models load fine and report 8× slower with a contaminated cache. `ollama /api/ps`
   is not enough.
2. **Never time a single small kernel repeatedly** — you will measure L2, not the
   card (we saw 486–730 GB/s on a ~240 GB/s GPU). Evict L2 before every timed call,
   and make the headline a whole-model pass.
3. **Evict with a read, not a write.** Zeroing a buffer leaves dirty cache lines
   that write back *during* the kernel you are timing.
4. **Never hold the fp32 copy and the quantized copy at once.** Windows silently
   pages to system RAM and you get a tidy, meaningless result (401 ms/token).
5. **Quote whole-token numbers, not one-projection numbers.** A single GEMV cannot
   saturate this GPU; the fused win only appears when kernels are pipelined.
6. **A reproduction is not a discovery.** Label it. Every claim above says whether
   it was measured here, quoted from a paper, or copied from a shipped runtime.
7. **Clean up after each run** — Triton cache, `__pycache__`, temp files, stray
   processes; leave nothing global changed on the machine.

## 9. Where everything lives

| file | what it is |
|---|---|
| [FINDINGS.md](FINDINGS.md) | this file — the plain-English summary + options |
| [RESULTS.md](RESULTS.md) | the lab notebook: every experiment, method, raw number, caveat |
| [REPORT_ollama_and_wastes.md](REPORT_ollama_and_wastes.md) | Ollama tests, the 3 wastes, the fit ladder, the fix list |
| [../research/litreview_pass1.md](../research/litreview_pass1.md) | literature catalogue (ZipServ, Shannon-bound, Marlin, etc.) |
| `runs/*.json` | raw result files for every run quoted here |
| `scripts/` | the instruments: `fused_decode_gemv.py` (E6 kernel), `ollama_bench.py`, `llamaserver_bench.py`, `entropy_probe.py`, `quant_entropy_probe.py`, `spec_divergence.py`, `fit_ladder.py`, `stage_llamaserver.py` |

### Machine-setup changes we made (all reversible, nothing global)

* `.runtime/llamaserver/` — 41 files hardlinked from Ollama's own install so the
  bundled `llama-server.exe` can see its CUDA DLLs. **Zero extra disk**, removable
  by deleting the folder; recreated by `python scripts/stage_llamaserver.py`.
* `qwen3.5:9b-fast` — an Ollama model entry from `scripts/Modelfile.fullgpu`
  (num_gpu 33, num_ctx 65536). Remove with `ollama rm qwen3.5:9b-fast` if unwanted.
* Nothing was installed globally, no drivers or system settings touched, no
  environment variables left permanently set.

## 10. One-line summary

On this card, the speed comes from the **runtime** (+74%, free) and the
**architecture** (hybrid, long context nearly free); the memory comes from
**quantization** (4-bit works, quality visibly drops); the one genuinely open
technical question left is whether a **fixed-length** codec keeps enough of the
measured 22% redundancy to be worth fusing into the math kernel.
