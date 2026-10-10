# 35B MoE on an 8 GB laptop card — measured, not estimated

Model: **Qwen3.6-35B-A3B-UD-Q4_K_M** (unsloth), 22.13 GB on disk, arch `qwen35moe`,
34.661 B params @ 5.106 bpw.
Hardware: RTX 4060 Laptop, **8188 MiB**, i7-13620H, 63.7 GB system RAM.
Runtime: Ollama's bundled `llama-server.exe` staged into `.runtime/llamaserver/`.
All numbers below are stopwatch measurements from `scripts/moe_bench_35b.py`,
raw JSON in `experiments/runs/moe_bench_*.json`.

## The headline

| | 9B (today's driver) | **35B MoE (this)** |
|---|---|---|
| Weights | 5.62 GB | **22.13 GB** |
| Reads per token | ~5.6 GB | ~3.17 GB |
| VRAM used | 5595 MiB @16K ctx | **3203 MiB @32K ctx** |
| Decode (fresh machine) | 40.6 tok/s | **34.6 tok/s** |
| Decode (warm/hot machine) | ~40 tok/s | **23–26 tok/s** |
| Prefill, short prompt | 231–579 tok/s | 20–42 tok/s |
| Prefill, ~3.3k-token prompt | not measured | **211 → 474 tok/s** (tuned) |
| Answer quality | good, one factual error | **denser, no error found** |

**A 4× larger model at 0.6–0.85× the speed, using 39% of the card.**

## The model's anatomy (why it works)

256 experts per layer, top-8 selected, 40 layers. Only the selected experts are
read per token — 8 of 256 is 3.125%, so 19.57 GB of expert weight costs only
~0.61 GB of RAM traffic per token. The remaining 2.56 GB (attention, SSM,
router, embeddings, head) is read every token and lives in VRAM.

It is also a **hybrid**: 30 of 40 layers are SSM with constant state, so only 10
carry a KV cache. That makes context nearly free — 20 KiB/token, so 32K context
costs 0.63 GiB instead of the ~2.5 GiB a normal 40-layer transformer would need.

## What actually made it faster

**Micro-batch size: `-ub 2048` instead of `-ub 512`.** Prefill on a ~3,300-token
prompt: **211 tok/s → 474 tok/s (+2.24×)**, +222 MiB VRAM (3203 → 3425 MiB),
byte-identical output.

Related correction: short-prompt "prefill" numbers are almost entirely fixed
per-request overhead and do not describe long-context throughput. The 35B looks
like it prefills at 20–42 tok/s on a 50-token prompt, but the same configuration
reaches 211 tok/s on a 3.3k-token prompt. The 9B was only measured on short
prompts, so no apples-to-apples long-prompt comparison exists between the two.
Measure prompt processing with a prompt that is actually long.

## What did not help (all measured, all null)

**Where expert weights live makes no difference.** Moving 4, 6, 8 or 10 layers'
worth of expert weights into VRAM (3.2 → 7.85 GB VRAM) left decode where it was:

| config | VRAM | decode |
|---|---|---|
| all experts in RAM | 3203 MiB | 25.4 tok/s |
| 10 layers of experts in VRAM | 7849 MiB | 25.7 |
| all experts in RAM | 3203 MiB | 23.3 |
| 4 layers of experts in VRAM | 5127 MiB | 26.2 |
| all experts in RAM | 3203 MiB | 21.6 |

The identical baseline config drifts 25.4 → 23.3 → 21.6 over minutes of sustained
load, and every VRAM-heavy config lands inside that drift band. **The apparent
"GPU experts are faster/slower" effect in an ordered sweep was thermal drift.**

**Threads:** default (10 physical cores) is best. `-t 4` = 23.3 tok/s,
default = 26.3 / 24.3, `-t 16` (hyperthreads) = **16.1 tok/s (−38%)**.

**There is a hard VRAM cliff.** At 7.9 GB used, decode collapses from ~24 tok/s to
**11.9 tok/s** — Windows spilling VRAM into shared system memory. Keep well under
the card's nominal size.

## CORRECTION (added after a later research pass): the cache is NOT dead

**An earlier version of this document said the hot-expert cache idea was dead.
That was too strong and is corrected here.**

What is actually measured, and still true:

* A **static** per-layer split (`--n-cpu-moe N`, permanently parking N layers'
experts in RAM) gives **no measurable benefit** at any N between 0 and 10 layers
in VRAM. Verified by an interleaved A/B/A/B control.
* The routing skew is weak and does not sharpen with expert count.

What was NOT tested and should not have been declared dead:

* A **dynamic, runtime** cache that promotes the experts actually being used and
  evicts the ones that are not. Two independent sources report roughly **2x** on
  *this exact model and this exact 8 GB VRAM size*:
  * a llama.cpp pull request, "llama : add a GPU cache for MoE experts kept in
    host [RAM]" (experimental, CUDA only, **not in mainline**): reported
    Q2_M 33.25 -> **57.2** tok/s (1.72x) and Q5_K_P 17.34 -> **35.93** tok/s
    (2.07x) on Qwen3.6-35B-A3B with 8 GB VRAM. Caveat printed with it: output
    can vary slightly as experts move between CPU and GPU.
  * **FreeToken** (FlashML, Apache-2.0, arXiv 2608.16157), a shipped engine whose
    entire design is "experts in host RAM, VRAM as an LRU cache". Independently
    benchmarked at **1.8x decode and 4.7x prefill** over llama.cpp `-cmoe`
    (41.6 vs 22.8 tok/s) on a 122B MoE. Its own README claims 35B on 8 GB laptops.

**Why the static split could measure as zero while the dynamic cache claims 2x**
— my inference, and it is *not verified*: a static layer split merely moves which
device owns a whole layer's expert work; it does not create a second lane. A
dynamic cache can leave some experts of the same layer computing on the GPU while
others compute on the CPU, i.e. it can overlap the two. Byte savings were proven
irrelevant; **concurrency** is the only mechanism left that explains the reports.

**Why neither could be tested on this machine** (all verified):

| requirement | this machine |
|---|---|
| llama.cpp PR: needs compiling | no C++ compiler, no CUDA toolkit, `nvcc` absent |
| FreeToken: Linux x86_64, or the Windows desktop installer | Windows; the installer is a GUI app and was not run |
| FreeToken: driver r580+ / CUDA 13 | **driver 566.14 / CUDA 12.7** |
| FreeToken: NVFP4 or block-FP8 weights | we have GGUF Q4_K_M; FreeToken has no GGUF path |

**And the mechanism would likely lose here anyway.** FreeToken's own framing: on
their benchmark box host RAM streams at 63.6 GB/s and PCIe at 56.0 GB/s — "when
the bus is as fast as the memory, shipping bytes to the fast compute beats
computing where the bytes already are." On this laptop PCIe measures **2.8 GB/s**
against ~21-40 GB/s of effective RAM bandwidth. The bus is roughly **15x slower
than the memory**, so streaming experts to the GPU is the wrong trade here, while
a *cache* (one-time promotion, amortised over many tokens) is not obviously
ruled out.

## ANSWERED (later): the dynamic cache is now in mainline, and it was tested

The upstream llama.cpp expert cache **shipped**: `--moe-cache-mib N` is in the
official prebuilt binaries as of build **b11537** (09 Oct 2026), so no compiler
was needed after all. Measured on this laptop, same binary, same model, same
session, interleaved A/B/A/B at 32K context:

| model | no cache | with cache | outcome |
|---|---|---|---|
| full 35B (22.13 GB, 256 experts) | 22.2 tok/s | 8.0 tok/s @ 2048 MiB | **−64%** |
| REAP-48 (9.43 GB, 133 experts) | 12.6 tok/s | **39.6 tok/s** @ 5120 MiB | **+216%** (3.2x) |

So the community's ~2x claim is real, but it is **not a property of the flag — it
is a property of the model's shape**. The pool has to hold roughly a third or more
of the expert weights to win; 8 GB holds 11–22% of the 35B's 19.6 GB of experts
(pure overhead, hence the loss) but 43–71% of REAP-48's 7.55 GB (hence the win).
The "concurrency" inference above was **wrong** — what buys the speedup is simply
removing CPU-side expert math for a large fraction of experts, not overlapping
lanes. The full detail, the byte-fraction rule, the quality verdict and the
caveats are in [EXPERT_CACHE_AND_REAP.md](EXPERT_CACHE_AND_REAP.md).

**That promotes the parked PCIe anomaly to the critical path.** The chip
advertises `pcie.link.gen.max = 4`, `width.max = 8`; it negotiates Gen3 x4. If
only x4 lanes are physically wired, that is a hardware fact and nothing can fix
it; if it is power management, a fifth of the theoretical link is being left on
the table, and that fifth is exactly what the 2x-class engines need.

## The hot-expert cache idea: what was measured (static split only)

**1. The routing skew does not grow with expert count.** From the model's own
imatrix (per-expert selection counts over a large calibration corpus):

| | OLMoE (64 experts) | Qwen3.6 (256 experts) |
|---|---|---|
| Gini | 0.380–0.427 | **0.366** |
| top 5% capture | — | 16.1% |
| top 10% capture | — | 26.7% |
| top 20% capture | 44–49% | **42.6%** |
| top 50% capture | 75–92% | 74.8% |
| experts needed for 80% | 53–56% | **56.9%** |
| efficiency vs random | ~2.2× | **2.32×** |

4× more experts produced no sharper concentration. The "20% of experts do 80% of
the work" claim is false at both scales; the real effect is about **2.3× better
than random**.

**2. Even eliminating expert RAM reads buys nothing.** The A/B above moved expert
weights out of RAM and into VRAM — exactly what a cache does, just at coarser
granularity — and decode did not move. Expert RAM traffic is not the bottleneck.

## What the bottleneck actually is

The 35B MoE reads ~3.17 GB/token and runs at 34.6 tok/s ⇒ ~110 GB/s of aggregate
traffic, against ~240 GB/s available from the card. **It is not bandwidth-bound.**

The 9B at 40.6 tok/s reads ~5.6 GB/token ⇒ ~227 GB/s. **That one is saturated.**

So the 9B's speed comes from saturating VRAM bandwidth, while the 35B MoE's speed
is set by the CPU-side expert math and the per-layer GPU↔CPU hand-offs — which is
exactly why moving weights around does nothing.

## Quality

Same three prompts, thinking disabled, temperature 0, both models.

* **35B** correctly explained the CPU memory hierarchy and DDR5 channel topology;
  its security methodology named ALPN, cipher-suite fingerprints and specific
  `openssl s_client` flags; its code answer honoured the stated requirement.
* **9B** was well-structured and safe-first (it opens with an authorization
  check, which is good practice), but asserted that *"System RAM is connected via
  a PCIe bus"* — false — and silently redefined the requested rolling-60s average
  as non-overlapping windows.

Verdict: a genuine upgrade, most visible on hard technical prompts. Not a
night-and-day difference on easy ones.

## Operational discovery: it is a reasoning model by default

With thinking left on, **all 192 generated tokens went into the thinking block and
`content` came back empty.** Practical consequences: budget tokens for thinking,
expect the visible answer to arrive late, or disable it per request with
`chat_template_kwargs: {"enable_thinking": false}`.

## RAM safety

Only ever memory-mapped; `--no-mmap` and `--mlock` were never passed, so the 22 GB
of weights stay in evictable page cache rather than committed process memory.
Peak system RAM observed: **46 GiB of 63.7**. The watchdog in
`scripts/moe_bench_35b.py` samples RAM every 2 s and hard-kills the server above a
ceiling; it never tripped.

## Levers considered and their disposition

Speculative decoding via **MTP** — the model family ships an MTP variant
(`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) and this llama-server build supports
`--spec-type draft-mtp`. **This was subsequently tested and it worked: +18 to +20%, measured interleaved
against the same model without it — see "Two practical wins" below.** Earlier n-gram
speculation on the 9B gave +13–19% on repetitive text only and nothing on prose,
so the prior is weak but not zero for a trained multi-token head.

## Every configuration measured (one table)

All figures are stopwatch results from `scripts/moe_bench_35b.py`. Decode is the
mean of the prose / code / security prompts. "Prompt reads" is the ~3.3k-token
prompt unless stated otherwise. **Decode ranges are wide on purpose**: the same
configuration measured 34.6 tok/s on a cool machine and 21.6 on a hot one, so a
single number would be misleading.

| configuration | VRAM MiB | decode tok/s | prompt reads tok/s | notes |
|---|---|---|---|---|
| all experts in RAM (`-ot exps=CPU`) | 3203 | **21.6 – 34.6** | 211 | baseline; the spread is thermal, not noise |
| `--n-cpu-moe 40` | 3203 | 31.0 – 32.1 | — | control: identical to the baseline |
| `--n-cpu-moe 36` | 5127 | 25.8 – 26.3 | — | |
| `--n-cpu-moe 34` | 6055 | 24.4 – 26.9 | — | |
| `--n-cpu-moe 32` | 6985 | 27.1 – 30.2 | — | |
| `--n-cpu-moe 30` | 7849 | 28.1 – 31.6 | — | |
| `--n-cpu-moe 28` | 7901 | **11.9 – 13.4** | — | collapse: Windows spilling VRAM |
| `-t 4` | 3203 | 23.3 | — | |
| default threads | 3203 | 24.3 – 26.3 | — | **best** |
| `-t 16` | 3203 | **16.1** | — | hyperthreads actively hurt |
| `-ub 2048` | 3425 | 23.0 – 30.7 | 490 | |
| `-ub 4096` | — | — | **590** | batch-sweep JSON lost, see note below |
| `-ub 8192` | — | — | killed | RAM hit 53.4 GB, watchdog fired |
| MTP (`-ub 2048`) | 4037 | **23.9 / 37.0** | 471 – 509 | 73–75% draft acceptance |
| **MTP + `-ub 4096`** | 5071 | **29.25** | **587** | the recommended configuration |
| 9B dense, for reference | 5595 | 39.93 | 231–579 (short prompts) | saturates VRAM bandwidth |

The `-ub 4096` / `-ub 8192` rows have no JSON artifact: that run used an earlier
version of the harness which aborted when the `8192` server was killed, so the
numbers survive only in `.runtime/bench_35b_batch.log`. The harness now records a
dead configuration and continues.

## Reproduce

```bash
# baseline, all experts in RAM (recommended default)
python scripts/moe_bench_35b.py --model models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --configs cpu-moe --reps 2 --n-predict 192

# the prefill win
python scripts/moe_bench_35b.py --model ... --configs ub2048 --n-predict 96

# routing skew from imatrix
python scripts/moe_imatrix_skew.py models/Qwen3.6-35B-A3B-GGUF/imatrix_unsloth.gguf_file
```

## Two practical wins, found after the first write-up

### 1. Raise the prompt batch: 211 -> 590 tok/s (2.8x)

llama.cpp refuses to hand a prompt batch to the GPU until it reaches
`GGML_OP_OFFLOAD_MIN_BATCH`, which is `32 * total_experts / active_experts`.
For a 256-expert top-8 MoE that is **1024 tokens**. Below it, prompt processing
runs on the CPU, which is exactly the wrong device for it. So the micro-batch
has to clear 1024 before GPU prompt processing engages at all.

| `-b` / `-ub` | prompt processing (~3.3k-token prompt) |
|---|---|
| 512 (llama.cpp default) | 211 tok/s |
| 2048 | 490 tok/s |
| **4096** | **590 tok/s** |
| 8192 | killed — see below |

**2.8x, with byte-identical output.** A community report for the same class of
setup claims 5.5x (380 -> 2091 tok/s) using even larger batches, which this
machine cannot afford.

At `-b 8192 -ub 8192` system RAM reached **53.4 GB** and the RAM watchdog in the
harness killed the server by design. So on a 64 GB machine that already has
~25 GB in use, **4096 is the practical ceiling**.

### 2. MTP (multi-token prediction): +18-20% generation

Speculative decoding with the model's own built-in draft head. Interleaved
A/B/B/A so thermal drift cannot fake the result:

| run | config | prose | code | security | mean |
|---|---|---|---|---|---|
| 1 | no MTP | 28.47 | 31.16 | 32.44 | 30.69 |
| 2 | **MTP** | **34.30** | **41.12** | **35.51** | **36.97** |
| 3 | **MTP** | 21.18 | 25.81 | 24.65 | 23.88 |
| 4 | no MTP | 19.15 | 21.49 | 20.09 | 20.24 |

Warm pair: **+20.5%**. Hot pair: **+18.0%**. Draft acceptance was **464/616 =
75%**, which is what proves the drafting is real rather than noise.

MTP does not change prompt processing (it is a decode-time technique).

**Caveat, stated honestly: MTP is not bit-exact.** The `code` answer was
identical in all four runs, but the `prose` and `security` answers differ between
MTP and non-MTP (each config was identical *to itself* across runs, so this is a
deterministic consequence of the speculative path, not nondeterminism). So use
MTP when you want speed, and turn it off when you need a reproducible output.

### Research pointers (checked October 2026)

* **RapidMoE** (EuroSys '27, arXiv 2610.01265) — splits expert weights at the
  *bit* level: a low-precision component stays on the GPU, the high-precision
  residual goes to the CPU. Non-critical experts then compute entirely on the
  GPU with no transfer. Claims 3.5x decode / 2.1x prefill over prior offloading
  systems. This directly attacks the coupling our measurements identified — an
  expert must be computed where its weights live. Research prototype; not
  buildable here.
* **DynaExq** (arXiv 2511.15015) — two findings that back up our results:
  (a) **prefill activates 47-97% of experts** while decode at batch 1 activates
  1.9-9%, which is why prompt processing is the expensive half; and (b) the
  **top-10 hottest experts are completely disjoint across text / math / code**,
  so any statically-tuned expert cache is tuned for the wrong workload.
* **GEMQ / EAQuant** — expert-level mixed-precision quantization, i.e. spend
  more bits on hot experts and fewer on cold ones at the same total size.
* **MTP in llama.cpp** — merged May 2026; the argument was renamed from
  `--spec-type mtp` to `--spec-type draft-mtp`, which is the form this
  machine's bundled server supports.
* **llama.cpp hot-expert cache PR** and **FreeToken** (FlashML, Apache-2.0,
  arXiv 2608.16157) — the two dynamic-cache systems described in the CORRECTION
  above. Both claim roughly 2x on 8 GB class hardware; neither could be run here.
* **REAP** (Router-weighted Expert Activation Pruning, Cerebras, arXiv 2510.13999)
  — one-shot structural pruning of experts, scoring them by router-weighted
  activation rather than raw selection count, reported to beat expert merging
  especially at 50% compression. A pre-pruned GGUF exists for this exact model
  (`crucible-labs/Qwen3.6-35B-A3B-REAP-48-v2-GGUF`), which makes it a
  **directly testable** compression option: smaller model, some quality cost,
  no runtime support needed. Untested here.

---

**Recommended daily configuration:**

```
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -ot exps=CPU -b 4096 -ub 4096 -c 32768 -fa on -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

using the **MTP** GGUF from `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (the MTP head must
be present in the file for `--spec-type draft-mtp` to do anything). Drop
`--spec-type draft-mtp` when a byte-reproducible answer matters.

### That exact combination, verified running

| metric | value |
|---|---|
| prompt processing (~3.3k tokens) | **586.8 tok/s** |
| decode (prose / code / security) | 32.94 / 26.14 / 28.68 tok/s |
| draft acceptance | 225/307 = **73%** |
| VRAM | 5243 MiB of 8188 |
| peak system RAM | 47.9 GiB of 63.7 |

The two wins do not interfere: prompt processing matched the batch-only result
(587 vs 590 tok/s), and decode kept the MTP gain.

**One side effect worth knowing:** MTP costs about **1.9 GB of extra VRAM**
(3203 -> 5071 MiB), because the draft head has to be resident too. That is fine
here — it still leaves ~3 GB of headroom — but on a card any smaller, or with a
longer context, do not assume MTP is free.
