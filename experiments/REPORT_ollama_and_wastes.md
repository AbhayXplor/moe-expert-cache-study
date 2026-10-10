# Ollama tests + the "3 wastes" — what is real, what is not

Measured on: RTX 4060 Laptop 8188 MiB (sm_89, 68.5 W cap), i7-13620H, 68 GB RAM,
Windows, Ollama 0.35.1, model `qwen3.5:9b` (8.954 B params, 5.020 bpw, 5.618 GB).
Everything here is measured on this box unless marked **[quoted]**.

---

## Part 1 — Ollama tests

### 1.1 The biggest win was not a "waste" at all: Ollama was leaving 24% on the table

Ollama decides GPU offload by policy, not by free memory. At 32K it chose 76% GPU
and ran **23.4 tok/s**. Force it to 33/33 layers and the same model, same context,
same quality runs **40.6 tok/s**.

| setting, 32K context | layers on GPU | tok/s | size | VRAM peak |
|---|---|---|---|---|
| default | 76% | 23.43 | 6.05 GB | — |
| `num_gpu 33` | 100% | **40.63** | 6.558 GB | 7264 MiB |
| `num_gpu 33`, 64K ctx | 100% | 40.35–40.49 | 7.666 GB | 7878 MiB |

Persisted in [scripts/Modelfile.fullgpu](../scripts/Modelfile.fullgpu):

```
FROM qwen3.5:9b
PARAMETER num_gpu 33
PARAMETER num_ctx 65536
```

`ollama create qwen3.5:9b-fast -f scripts/Modelfile.fullgpu` then just
`ollama run qwen3.5:9b-fast`. Re-verified today on drained VRAM: **39.70 tok/s at
32K, 100% GPU, 6.558 GB, peak 7267 MiB, no flags.**

### 1.2 This model is a hybrid — that is why long context is nearly free

`qwen3.5.full_attention_interval = 4`: only every 4th layer does attention, so
**8 of 32 layers carry a KV cache**; the other 24 are Mamba-style SSM with
*constant* state (50 MiB, does not grow with context).

| | KV per token | at 16K | at 32K | at 64K |
|---|---|---|---|---|
| qwen3.5:9b (8/32 layers, f16) | 32 KiB | 0.5 GiB | 1.0 GiB | 2.0 GiB |
| a classic 32-layer dense 9B (f16) | 128 KiB | 2.0 GiB | 4.0 GiB | 8.0 GiB |

The KV cache is **f16, not quantized** (an earlier claim of a 4-bit default was
wrong). Ollama refuses `cache_type_k` in the API; it is env-only
(`OLLAMA_KV_CACHE_TYPE=q8_0`) and costs ~10% decode speed:

| 64K context | f16 KV | q8_0 KV |
|---|---|---|
| empty prompt | 40.49 | 36.40 |
| 32K tokens in context | 34.34 | 31.48 |

q8_0 only pays for itself at 128K, where it is the difference between fitting
(7.70 GiB) and not (9.58 GiB).

### 1.3 Where the decode time goes (weights vs context)

| tokens in context | tok/s | ms/token | ~weights | ~KV read |
|---|---|---|---|---|
| 0 | 40.83 | 24.49 | 24.49 | 0.01 |
| 8 000 | 38.82 | 25.76 | 24.80 | 0.96 |
| 16 000 | 37.57 | 26.61 | 24.67 | 1.95 |
| 32 000 | 34.81 | 28.73 | 24.84 | 3.88 |
| 60 000 | 31.07 | 32.19 | 24.92 | 7.27 |

The weights term is **flat at 24.5–24.9 ms/token = ~227 GB/s**, and the KV term is
exactly linear in context and equals 32 KiB/token ÷ ~240 GB/s. Nothing is leaking.

### 1.4 The trap that made the same config report 5 or 16 tok/s

Ollama caches its free-VRAM reading at load time. If anything else holds VRAM
(leftover `llama-server.exe`, a game launcher) it keeps the stale value and pushes
layers to the CPU:

* with 6051 MiB still resident → `qwen3.5:9b @32K` = **5.09 tok/s, 93% on CPU**
* same config on clean VRAM → 23.43 tok/s (default policy), 40.63 forced

Three orphaned `llama-server.exe` (2.2 GB) also turned a 40.9 tok/s run into
"16 tok/s". `ollama /api/ps` shows nothing; use
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.
Drain VRAM before benchmarking ([scripts/ollama_clean.py](../scripts/ollama_clean.py)).
**Any comparison made with dirty VRAM is worthless.**

---

## Part 2 — The 3 wastes, corrected

Waste 1 = information in the file that need not be there.
Waste 2 = not using the bandwidth you already have.
Waste 3 = reading every weight for every token.

### 2.1 Waste 1 — real, but the win needs kernels, not just entropy coding

| form | stored bits/weight | real content | removable, bit-exact |
|---|---|---|---|
| bf16 | 16 | 10.57 | — (only **0.9%**, lossless floor) |
| int8 quantized | 8 | 7.195 | **10.1%** |
| int6 quantized | 6 | 5.180 | **13.7%** |
| int4 quantized | 4 | 3.124 | **21.9%** |

So: lossless compression of the *original* weights is capped at ~1.36x, but **after
quantization, 21.9% of a 4-bit file is pure redundancy** (this reproduces the
published Shannon-bound result 5-for-5 on our own data).

The trap is that decompression costs bandwidth. Lossless DF11 on Qwen3-1.7B:
3.441 GB → 2.538 GB (1.356x, 28/28 blocks bit-exact, max logit diff 0.0) but
**2.32x slower**, because the unfused version moves 2.20x the bytes per token
(1.92 GB compressed read + 2.82 GB scratch write + 2.82 GB scratch read vs 3.44 GB
plain). The entropy codec itself is only ~19–20% over a pure copy — the loss is all
in the extra passes.

Fix: decode **directly into the tensor-core operand registers** so the scratch
round-trip disappears (ZipServ's "ZipGEMM" idea). **[quoted]** ZipServ claims 2.21x
kernel speedup over cuBLAS, 1.22x end-to-end over vLLM, up to 5.53x over DF11,
requires compute capability ≥ 8.0 (we have 8.9). Not reproduced here: no host
compiler and no CUDA toolkit on this machine.

### 2.2 Waste 2 — mostly already closed (correction to my earlier claim)

I previously measured ~165 GB/s in my own test harness and called it a 1.4x
opportunity. **That was the harness, not the problem.** Ollama with CUDA graphs +
flash attention moves the weights at **~227 GB/s ≈ 95% of what this card can
actually do** (~240 GB/s measured achievable). Remaining headroom: **~5%**, not 1.4x.
This is a negative result and it removes waste 2 from the big-lever list.

### 2.3 Waste 3 — measured, and it is real but narrow

The bundled `llama-server.exe` can do speculative decoding; Ollama exposes nothing.
Fix for the harness: the CUDA backend DLLs live in `lib/ollama\cuda_v12` while the
binary lives in `lib\ollama`, and ggml only scans the executable's own directory —
hardlink both into one directory and CUDA0 appears, with no more
`warning: no usable GPU found, --gpu-layers option will be ignored` in the log
(`python scripts/stage_llamaserver.py`, then `llama-server.exe --list-devices`
reports `CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU`).

Greedy, 256 tokens, 32K context, 100% GPU (tok/s):

| config | free-form | copy-heavy | repetitive JSON | drafts accepted |
|---|---|---|---|---|
| no spec decode | 38.84 | 39.74 | 40.00 | — |
| ngram-simple, lookup 12 (default) | 39.76 | 41.27 | 39.79 | 30/186 |
| ngram-simple, lookup 6 | 39.44 | **45.05 (+13.4%)** | 42.69 | 75/239 |
| ngram-simple, lookup 4 | 37.51 | **47.17 (+18.7%)** | 36.39 | 94/216 |
| ngram-map-k, lookup 4 | 37.93 | 42.66 | 37.48 | 66/171 |
| ngram-mod (match 8) | 38.90 | 42.43 | 41.18 | 56/148 |

* It pays **only** where text repeats itself (copy-heavy: +13 to +19%).
* On free-form prose it is neutral to **negative** (lookup 4: −3.4%): rejected
  drafts are not free.
* Acceptance rate is the whole story: 16% at lookup 12 → 44% at lookup 4.
* The ceiling is not draft quality but the fact that verification is a real
  forward pass. +20% is roughly what this lever is worth here — **not 2x**.

A real draft model would do better, but a draft must share the target's vocabulary
and `qwen3.5:9b` uses a 248 320-token `qwen35` tokenizer, so no off-the-shelf
Qwen3-0.6B draft can be paired with it. `draft-mtp` is impossible too: this model
ships no MTP tensors.

**Bit-exactness — the honest version.** Each configuration is perfectly
deterministic (identical output hash across repeats, identical draft counts), but
speculative decoding does *not* always reproduce the baseline token stream:

* copy-heavy: byte-identical to baseline in **all** configurations;
* free-form: identical except at lookup 4;
* repetitive: **differs in every configuration that drafted.**

The divergence is not degradation — it is a near-tie. Two completely different
configs (lookup 12 and lookup 4) produced the *same* alternative output, diverging
at the same character 340, on a low-information filler field:

```
base: ... "name": "Gamma-3", "category": "Storage",  "status": "inactive", ...
spec: ... "name": "Gamma-3", "category": "Cache",    "status": "degraded", ...
```

Verification runs as a batch, a batched GEMM sums in a different order than a
single-token GEMM, and a greedy pick sitting on a tie flips. Same distribution,
same quality, different bytes. So: **spec decoding costs no quality, but do not
promise byte-identical output.**

### 2.4 Revised multiplier

| lever | effect | quality cost | status |
|---|---|---|---|
| 1. entropy-coded 4-bit + fused decode | 21.9% less VRAM/bandwidth | none | needs a kernel; not built |
| 2. runtime / kernel efficiency | ~5% | none | already captured by Ollama |
| 3. speculative decoding | +13 to +19% on repetitive text | none | works today, narrow |

Stacked realistically: **~1.2–1.3x, not the 2.5–4x I estimated earlier.**

---

## Part 3 — What this means for the 16B plan

Budget: **7.70 GiB usable** (proven: a 100%-GPU run peaked at 7878 MiB of 8188),
of which ~0.35 GiB is fixed overhead. KV for a classic dense GQA model is
≈ 144 KiB/token; for the hybrid it is 32 KiB/token.

| model, all at 4.5 bpw | weights | KV @32K (f16) | total | fits? |
|---|---|---|---|---|
| qwen3.5:9b hybrid (measured) | 4.69 GiB | 1.00 | 6.04 | **YES** |
| 9B classic dense | 4.71 | 4.00 | 9.06 | NO (needs q8_0 KV → 7.06, then yes) |
| 12B classic dense | 6.29 | 4.50 | 11.14 | NO |
| **16B classic dense (your target)** | **8.38** | 4.50 | 13.23 | **NO** |
| 32B classic dense | 16.76 | 8.00 | 25.11 | NO |

A 16B dense model is over budget on **weights alone** (8.38 GiB > 7.70 GiB) before
a single token of context. Stacking every lever on it (16K context):

| 16B dense, 16K ctx | weights | KV | total | verdict |
|---|---|---|---|---|
| today: 4.5 bpw, f16 KV | 8.38 | 2.25 | 10.98 | NO (+3.28) |
| + q8_0 KV | 8.38 | 1.12 | 9.86 | NO (+2.16) |
| + entropy-coded weights (−21.9%) | 6.55 | 2.25 | 9.15 | NO (+1.45) |
| + both | 6.55 | 1.12 | 8.02 | **NO (+0.32)** |

To fit a 16B dense at 16K you would need ≈ **2.55 bits/weight**, i.e. 3-bit — the
quality loss you are trying to avoid. **The waste budget (22%) is not big enough to
close the 16B gap, even if the kernel existed.**

And this explains your 4 tok/s precisely. A 16B at 3-bit is ~7.55 GB = 7.03 GiB.
Add 0.35 GiB fixed and there is ~0.32 GiB left for KV = **~2 200 tokens** of f16 KV.
The model loads and looks healthy, then the first real message walks past 2K tokens,
the KV cache spills to RAM and decode collapses from ~28 to 4 tok/s. The arithmetic
predicts exactly what you saw.

Where the real headroom is on this card: **architecture, not parameter count.**
A 12B *hybrid* at 4.5 bpw ≈ 6.29 GiB weights + ~0.5–1.0 GiB KV + 0.35 ≈ 7.2–7.6 GiB
fits at 16–32K context. The same 12B as a classic dense transformer does not.

---

## Part 4 — Fix list for this machine, ranked

1. **Use `qwen3.5:9b-fast`** (num_gpu 33, num_ctx 65536): +74% speed, same quality,
   free. Done — [scripts/Modelfile.fullgpu](../scripts/Modelfile.fullgpu).
2. **Drain VRAM before every run**, and check for orphan `llama-server.exe`.
   Otherwise you lose 8x and conclude the model is bad.
3. **Prefer hybrid/SSM models over dense** at the same size. Context is nearly free;
   it is the single biggest structural advantage available on 8 GB.
4. **`OLLAMA_KV_CACHE_TYPE=q8_0` only for 128K**, where it is the difference between
   fitting and not. Costs ~10%.
5. **Keep 64K context** — it is free at 100% GPU (7.666 GB, peak 7878 MiB).
6. **N-gram speculative decoding** for repetitive/copy-heavy work only
   ([scripts/llamaserver_bench.py](../scripts/llamaserver_bench.py)); it needs the
   hardlinked runtime directory, since Ollama cannot enable it.
7. **Do not buy a 16B dense model for this card.** At 4-bit it is over budget before
   context; at 3-bit it has ~2K tokens of usable context.

---

## Part 5 — Step 1

The wall is fine. Past the waste, bits/weight vs quality is a genuine trade, and the
levers we can still pull are exactly three: better representation, change the model,
or read fewer weights per token (MoE/sparsity — ruled out).

Step 1 = **the smallest real piece of the waste-1 fix: a fused entropy-decode GEMM,
in Triton, on Qwen3-1.7B.** We already have every ingredient and every number:

* DF11 data: 1.356x smaller, bit-exact, but 2.32x slower — with a byte-accounting
  breakdown that says exactly where the loss is (scratch round-trip, not the codec);
* the int4 target: 21.9% removable, bit-exact, with the entropy measured on our own
  weights;
* Triton 3.8.0 verified working on this box (no C++ compiler needed);
* a 1.7B testbed whose decode time we have already measured end-to-end.

Milestone, and it is falsifiable: **DF11-encoded weights must decode to bit-identical
logits within 1.15x of bf16 decode time** (today: 2.32x). If that works, the same
kernel is the prerequisite for the int4+entropy version, which is the version that
actually frees 22% of VRAM at 4-bit quality — the first honest step toward "same
quality, less memory, same speed". If it does not work, we will know in a day, with
numbers, and not after building a whole runtime.
