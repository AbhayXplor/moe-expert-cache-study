# Compressed-resident LLM inference — measured findings

Living lab notebook. Everything below was measured on this machine unless marked
*(prediction)*. Anything not measured is labelled as such; the prior workspace's
`research-agent/` write-ups are treated as unverified context and are explicitly
contradicted where noted.

## Machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8188 MiB, sm_89, 24 SMs, 33.6 MB L2 |
| Driver / CUDA | 566.14 / 12.7 (torch 2.14.1+cu126 in `.venv`) |
| CPU / RAM | i7-13620H, 16 threads, 68.4 GB |
| Python | 3.13.12, Windows, `transformers==4.57.6`, `dfloat11==0.5.0` |

### Bandwidth calibration (`scripts/bench_bw.py`, `experiments/logs/bench_bw.log`)

| pattern | bytes | ms | GB/s |
|---|---|---|---|
| pure read, 3.441 GB bf16 | 3.441 | 14.75 | **233** |
| pure write, 3.441 GB bf16 | 3.441 | 13.75 | **250** |
| copy r+w, 6.882 GB | 6.882 | 30.34 | 227 |
| read 1.409 GB u8 | 1.409 | 5.66 | 249 |
| **expand u8→bf16 (read 1.409, write 2.818)** | 4.227 | 23.6 | **177** |
| **decode-traffic floor (read 1.878, write 2.818)** | 4.696 | 25.9 | **181** |
| fused-input read only (both streams, 1.878 GB) | 1.878 | 7.55 | 249 |

Two lessons from building this table:
* `sum()` on a `uint8` tensor promotes to `int64` and becomes ALU-bound at
  **0.5 GB/s** — it measures nothing. `max()` keeps the dtype and is memory-bound.
* The achievable bandwidth is ~250 GB/s read / ~250 GB/s write, close to the
  256 GB/s spec. An earlier "190 GB/s" figure measured through a different
  pattern and was pessimistic.

## E0 — baseline, Qwen3-1.7B bf16 (`experiments/runs/e0_bf16_qwen3_1.7b.json`)

* resident **3.2048 GiB** (all `bf16:matrix`), peak alloc 3.243 GiB,
  nvidia-smi peak 3471 MiB, `bytes_offloaded_to_cpu = 0`
* load 5.7 s, TTFT 0.812 s
* decode **7.5 wall tok/s**; **GPU 28.72 ms/token** (`e0b_..._gpuprof.json`),
  top kernel `gemvx` (cutlass bf16 GEMV) = **20.809 ms over 197 launches**
* bf16 GEMV therefore reads its 3.44 GB of weights at **165 GB/s effective**
  (66% of the 249 GB/s achievable read rate). Batch-1 GEMV on this card leaves a
  third of the bandwidth unused, which sets the bar a fused kernel has to beat.

## E2a — DFloat11 Qwen3-1.7B, built locally (`experiments/runs/e2a_df11_qwen3_1.7b.json`)

I built the compressed model myself with `scripts/compress_df11_par.py`
(8 processes, 28 layers in 469 s wall) so the pipeline is reproducible rather
than trusted.

* **bit-exactness verified**: `scripts/finish_df11.py` decoded all 28 blocks with
  the DF11 CUDA kernel and compared to the original bf16 tensors —
  **28/28 blocks bit-exact** (`experiments/logs/finish_df11_verify.log`).
* compression: 3.441 GB bf16 → 2.538 GB DF11 = **1.356× overall**,
  1.472× on the compressed (7 projections/layer) part.
* resident **2.3633 GiB** — and it *is* resident: `bytes_offloaded_to_cpu = 0`,
  and the VRAM contents by class are

  | class | GiB |
  |---|---|
  | `df11:sign_mantissa` (8 raw bits/weight) | 1.3125 |
  | `bf16:matrix` (norms, embed, lm_head) | 0.5796 |
  | `df11:encoded_exponent` (Huffman bitstream) | 0.4364 |
  | `df11:gaps` | 0.0341 |
  | `df11:output_positions`, `df11:luts`, misc | ~0.0006 |

  So the weight-carrying buffers in VRAM are the *compressed* streams plus a
  small bf16 remainder. No full-size weight matrix is resident.
* peak alloc 2.502 GiB, nvidia-smi peak **2724 MiB** (vs 3471 for bf16).
* **output identity: 64/64 greedy tokens identical, max top-1 logit diff 0.0** —
  bit-exact reconstruction ⇒ bit-exact logits (`scripts/compare_runs.py`).
* **speed: 5.7 wall tok/s vs 7.5**, and on the honest metric
  **66.68 ms/token GPU time vs 28.72 = 2.32× slower**. The DF11 README's own
  "≈2× slower at batch size 1" is therefore correct, and the prior workspace's
  "decode neutral-to-faster" claim is **wrong**.

## E2b — official `DFloat11/Qwen3-4B-DF11` (`experiments/runs/e2b_official_qwen3_4b_df11.json`)

Tests the capability claim "a 4B model runs in 8 GB of VRAM", using the
authors' own published artifact (36 layers, 5.6 GB on disk).

* resident **5.332 GiB**, peak alloc 5.569 GiB, nvidia-smi peak
  **5948 MiB of 8188** ⇒ **it fits, with ~2.2 GB to spare**, no CPU offload.
* load 10.7 s, TTFT 0.890 s, decode 5.17 wall tok/s.
* **GPU 108.93 ms/token**, of which the `decode` kernel = **74.6 ms**.
* compressed streams 4.851 GB standing in for 7.267 GB of bf16 weights
  (1.498× on those weights).

## Why DF11 is slower — the actual mechanism

DF11 is compressed-resident but **not fused**. Reading
`repos/DFloat11/dfloat11/dfloat11.py`:

* the hook is a **forward pre-hook** on the *layer* module (`pattern_dict` maps
  `model\.layers\.\d+` → the 7 projection names, so all 7 projections of a layer
  share one concatenated compressed buffer set). It launches `decode.cu` into a
  single reused bf16 scratch (`TensorManager.allocate_bfloat16`), then does
  `module.weight = reconstructed.view(...)` and calls stock `F.linear`. The
  weights are split back into the projection submodules via `split_positions`.
* consequence: **one decode launch per layer** (28 for 1.7B, 36 for 4B — matches
  the profiler's launch counts), and per token the weight data moves

  | | per token |
  |---|---|
  | bf16 | read 3.44 GB |
  | DF11 (1.7B) | read 1.92 GB streams + **write 2.82 GB scratch** + **read 2.82 GB scratch** = 7.55 GB |

  i.e. **2.2× the bytes**. Measured time ratio is 2.32×. The byte count explains
  essentially all of the slowdown.

Per-kernel, per-token breakdown (`gpu_kernel_ms_top`, in-model profiler):

| | bf16 | DF11 (1.7B) | DF11 (4B) | NF4 (1.7B) |
|---|---|---|---|---|
| total | 28.72 | 66.68 | 108.93 | 25.65 |
| dequant/decode kernel | — | `decode` 49.53 (×28) | `decode` 74.61 (×36) | `gemm_4bit_simt` 14.33 (×196) |
| GEMV | `gemvx` 20.81 (×197) | `gemvx` 13.50 (×197) | `gemvx` 25.64 (×217) + 5.49 (×36) | `gemvx` 3.50 (×1) |
| weight-read rate | 165 GB/s | decode moves 4.73 GB at **95.6 GB/s** | ~168 GB/s | 0.66 GiB in 14.3 ms = 49 GB/s |

Note the asymmetry: DF11's *rebuild* step runs at 96 GB/s effective while the
GEMV that follows reads the same 2.82 GB scratch at **209 GB/s** (13.5 ms). The
decode kernel is the slow one, and it is slow relative to *its own* traffic
floor (181 GB/s), not just relative to peak.

### Is the *entropy decode itself* the bottleneck? No.

`scripts/bench_decode_kernel.py` runs the real kernel standalone on real layer
buffers, back to back, and compares against a pure-traffic copy of the same bytes
under identical conditions:

| model | weights/layer | decode | traffic floor | entropy excess |
|---|---|---|---|---|
| Qwen3-1.7B (mine) | 50.3 M | 1.20–1.32 ms @ 128–141 GB/s | 0.86–0.92 ms @ 164–175 GB/s | **0.23–0.28 ms (≈20%)** |
| Qwen3-4B (official) | 100.9 M | 2.47–2.54 ms @ 133–140 GB/s | 1.78–1.85 ms @ 164–170 GB/s | **0.43–0.49 ms (≈19%)** |

Two independently-built models land within 20% of the pure-traffic ceiling for
the identical byte pattern, and the Huffman work is only ~19–20% of the decode
kernel. **The variable-length decode is not the wall; the scratch round-trip is.**
(The naive in-model profiler figure of 49.5 ms/token for 28 decode launches is
1.38× the standalone per-launch figure — kernels measured inside a mostly-idle
launch-heavy loop run slower, so I trust the standalone number for the split and
the in-model number for the A/B ratio.)

## E3 — is there more lossless headroom? **No. (0.9%)**

`scripts/entropy_probe.py` (CPU-only, 201 M weights from 4 layers of Qwen3-1.7B)
reconstructs DF11's two byte streams directly from the original bf16 weights by
bit-field extraction, so no 1e9-symbol Python Huffman decode is needed.

| quantity | bits/weight |
|---|---|
| H(exponent byte) | **2.6436** |
| …stored by DF11 (Huffman, per layer, memoryless) | 2.663 |
| H(exponent \| previous exponent) | 2.6386 (saves 0.005) |
| H(sign+mantissa byte) | **7.9728** |
| …stored by DF11 (raw) | 8.000 |
| H(sign+mantissa \| exponent) | 7.9326 (saves 0.040) |
| H(exp, sign+mantissa) jointly | 10.5761 |
| **best order-1 static lossless** | **10.5711** |
| current standing cost | 10.6630 |

* The Huffman code is **99.3% of the marginal entropy**, and the exponent stream
  is essentially **i.i.d.** (conditioning on the previous exponent buys 0.2%).
* The sign+mantissa byte is **0.34% below 8 bits** — the mantissa of a trained
  transformer is incompressible noise, and zstd-19 confirms it: it stores the
  `sign_mantissa` stream at exactly **8.000 bits/weight** (no gain at all), and
  matches the Huffman result on the exponent stream (2.642 bits).
* Total remaining lossless headroom: **0.9%**, i.e. 16 MB on a 1.88 GB
  compressed part. Modelling order-2 context would not change this materially.

**Consequence: lossless compression of bf16 weights tops out at ~1.36× overall
(1.50× on the projections). Any further VRAM reduction must be lossy** —
quantization, sharing/factorization, or both. This kills the "add a better
entropy layer" branch of the plan before spending effort on it, and reframes
DF11 correctly: it is not a lossy-quantization competitor on size (1.36× vs
~3.5–4×) but it is the only option at *exactly zero* quality cost.

### Consequence: fusion is the whole game *(prediction, from measured constants)*

| design | weight bytes moved per token (1.7B) | vs bf16 |
|---|---|---|
| bf16 (today) | 3.44 GB read | 1.00× |
| DF11 (today) | 7.55 GB (1.92 read + 2.82 write + 2.82 read) | 2.20× |
| **DF11 fused** (decode inside the GEMV, no scratch) | **1.92 GB read** | **0.56×** |

A fused decode-in-GEMV kernel that never materialises a bf16 copy should move
**~1.8× fewer bytes than bf16** and pay only the ~19% entropy surcharge on top of
that, so the speed sign flips. Per-layer roofline from the standalone components
(1.7B, 50.3 M weights/layer):

| term | now (unfused) | fused *(prediction)* |
|---|---|---|
| read compressed streams | 0.90 ms of mixed traffic | 0.05 GB @ 249 GB/s = **0.20 ms** |
| write bf16 scratch | included above (0.101 GB) | **0** |
| entropy-decode ALU | 0.23–0.28 ms | 0.23–0.28 ms |
| read scratch back in GEMV | (separate `gemvx`, 13.5 ms/token) | **0** |
| per layer | ~1.28 ms decode + ~0.48 ms gemv | **~0.45–0.50 ms** |
| **per token (28 layers + attention/norms)** | **52–67 ms** | **~14–18 ms** |

against bf16's measured 28.72 ms/token. So the prediction is **fused ≈ 1.5–2×
faster than bf16, at 1.36× less VRAM, bit-exact** — the only design that wins on
both axes. **Not implemented and not measured**; it is the clear next step, and
it needs a real CUDA kernel (decode inside the GEMV tile), not a wiring change.
The same measurement disagrees with itself by ±40% between in-model profiling
and standalone timing, so treat the ~14–18 ms as an order-of-magnitude target.

### Other verified notes

* My generated `dfloat11_config` (`bytes_per_thread=8`, `threads_per_block=[512]`)
  matches the official Qwen3-4B-DF11 config byte for byte, so the operating point
  measured above is the authors' own tuning, not a mistuning of mine.
* `dfloat11.dfloat11`'s loader accepted my independently produced directory
  unchanged — independent evidence the compressor output format is faithful.
* The DF11 decode kernel is **not grid-size agnostic**: over-launching it
  (grid ×2 or ×4, to probe occupancy) writes out of bounds and raises
  `CUDA_ERROR_ILLEGAL_ADDRESS`, because `output_positions`/`gaps` are sized for
  exactly `ceil(n_bytes/(512*8))` blocks. Relevant to anyone trying to retune it.
* Windows-specific traps hit while building this, all reproducible:
  a new `safe_open` handle per tensor silently kills the process after ~40 reads
  (use `load_file` per shard); `dahuffman`'s `print_code_table` and `get_luts`'
  per-prefix printing dominate compressor runtime unless suppressed;
  their success print contains `\u2705` and crashes under the default `charmap`
  codec (`PYTHONIOENCODING=utf-8` fixes it).

## E4a — the lossy reference point: NF4 4-bit (`experiments/runs/e4a_nf4_qwen3_1.7b.json`)

Same model, same prompt, same harness, `bitsandbytes` NF4 with double
quantization and bf16 compute (`--kind nf4`):

* resident **1.2361 GiB** (`int4:packed` 0.6562 + `bf16:matrix` 0.5796 — the
  embedding/lm_head/norms are still bf16, which is why 4-bit only buys 2.59×
  overall and not 4×). nvidia-smi peak 2474 MiB.
* **GPU 25.65 ms/token** — *faster* than bf16's 28.72, because it reads 0.66 GiB
  instead of 3.20 GiB. But wall clock is **worse**: 4.86 vs 6.49 tok/s, because
  the `gemm_4bit_simt` kernel is 73 µs/launch over 196 launches vs bf16's 68 µs.
* **quality is measurably damaged**: only **9/64 greedy tokens match bf16**,
  first divergence at index 8, max top-1 logit difference **7.5**. The generated
  text degrades from a fluent technical answer into a repetition loop
  ("What is the impact on the inference latency? What is the impact on the
  memory bandwidth utilization? …").

So the two compression axes are not competitors, they trade different things:

| format | resident | vs bf16 | GPU ms/tok | wall tok/s | greedy tokens matching bf16 |
|---|---|---|---|---|---|
| bf16 | 3.2048 GiB | 1.00× | 28.72 | 6.49 | 64/64 |
| **DF11 (lossless)** | **2.3633 GiB** | **1.36×** | 66.68 | 5.70 | **64/64, logits bit-exact** |
| **NF4 (lossy 4-bit)** | **1.2361 GiB** | **2.59×** | 25.65 | 4.86 | 9/64, diverges at token 8 |

## E4b — does entropy coding pay *after* quantization? **Yes: 22%.**

The E3 result (nothing to gain on raw bf16) is specific to bf16. Quantization
changes the picture, because a uniform quantizer emits a *peaked* code
histogram. `scripts/quant_entropy_probe.py`, per-output-channel affine
quantization of 151 M projection weights:

| scheme | stored bits | H(codes) | saving | GB for a 1.74 B model |
|---|---|---|---|---|
| bf16 (unchanged) | 16.0000 | — | — | 3.440 |
| **DF11 lossless (measured)** | **10.663** | (its floor 10.58) | — | 2.293 on the projections |
| uniform int8 | 8 | 7.1954 | **10.1%** | 1.547 |
| uniform int6 | 6 | 5.1804 | **13.7%** | 1.114 |
| **uniform int4** | **4** | **3.1244** | **21.9%** | **0.672** |

Reading: **entropy-coding the bf16 mantissa saves 0.0%; entropy-coding a 4-bit
quantized code saves 21.9%** — 4 bits of storage holding 3.12 bits of content.
So the productive combination is *quantize first, then entropy-code the codes,
then fuse the decode into the GEMM*, which is 2.5× smaller than the lossless
floor once the codes are coded at their entropy.

*(A `sqrt`-scaled variant was also probed and its numbers are deliberately not
reported: it scales the codebook span rather than the weights, so it only uses
~11 of 16 codes per row and its low entropy is information thrown away, not
information removed.)*

## What this establishes, and what is left

Answering the original question: **yes, LLM weights can be genuinely resident in
GPU VRAM in compressed form and run — verified bit-exact, with zero CPU
offload, on a real model.** But three things follow from the measurements that
change what "worth doing" means:

1. **Lossless is capped at ~1.5× on the projections (1.36× overall).** E3 shows
   the representation is within 0.9% of its entropy floor. No entropy-coding
   cleverness changes this. If 1.36× VRAM is the goal, DF11 already delivers it;
   if 3–5× is the goal, it must be lossy.
2. **Speed only comes from fusion.** Paying 2.20× the bytes for a 1.36× memory
   win (measured 2.32× slower) is the unfused design's tax: read streams → write
   bf16 scratch → read scratch. Removing that scratch round-trip is 5.65 GB/token
   on the 1.7B and is the only change that flips the speed sign.
3. **The lossy axis is where the memory win is**, and it costs visible quality
   (9/64 tokens identical). The strongest design on this evidence is a
   *quantized* payload (4–6 bits) whose codes are then entropy-coded (21.9%
   saving measured) and decoded inside the GEMM — not a lossless bf16 codec.

Not yet attempted, in the order the evidence now suggests:

* **E6: a fused decode-in-GEMM kernel.** The roofline says ~14–18 ms/token vs
  bf16's 28.72 at 1.36× less VRAM. Highest value, real CUDA work.
* **E5: sharing/factorization.** Untested. Its ceiling can be bounded the same
  way E3 bounded entropy: measure the rank structure of the projection matrices.
* A 4B DF11 vs 4B NF4 comparison at equal quality, which needs a perplexity or
  task metric this harness does not yet have — right now quality is only
  "bit-exact vs bf16" or "first divergence index", not a score.
* A bf16 Qwen3-4B run to convert "bf16 4B cannot fit in 8 GB" from an
  arithmetic observation into a measured OOM.

## Caveats that limit what can be claimed

* **Wall tok/s on this box does not measure the weight format.** HF-transformers
  decode here is launch-bound: ≈160 ms/step of CPU enqueue vs ≈21 ms of GPU work,
  1644 kernel launches/step at 20–47 µs each, and batching 1→16 barely changes
  step time while aggregate tok/s scales linearly. Hence every speed claim above
  rests on GPU kernel time, not wall clock.
* The in-model vs standalone decode measurements disagree by up to ±40% in
  opposite directions between the two models; the split between traffic and
  entropy work should be read as ~20–30% entropy, not as an exact figure.
* Quality is only established as *bit-exact logits* for the 1.7B build. No
  perplexity or task benchmark has been run on the 4B, and no lossy method
  (quantization, sharing/factorization) has been tested yet.
* The 4B bf16 baseline was not run, so "DF11 4B fits where bf16 4B would not" is
  an arithmetic observation (8.04 GB of bf16 weights alone against 8188 MiB of
  total VRAM), not a measured OOM.

---

# E6 — fused decode-in-GEMM kernel (ATTEMPTED, first result)

Question: the unfused codec was smaller but 2.32x slower; is that because the
decode is *materialised* (read compressed -> write bf16 scratch -> read scratch),
and does removing the scratch round-trip flip the sign?

Instrument: `scripts/fused_decode_gemv.py`. Three variants on real Qwen3-1.7B
weights, all computing the same `y = x @ W^T`, with a 4-bit codec whose decode is
fixed-rate and branch-free (the property that makes in-kernel decode possible at
all — Huffman is not addressable and cannot be fused without redesign):

| variant | bytes/elem read | what it does |
|---|---|---|
| bf16 | 2.00 | cuBLAS matvec, original weights |
| unfused | 0.52 + 2 + 2 | dequant -> bf16 scratch -> cuBLAS matvec |
| fused | 0.52 | one Triton kernel: unpack in registers, then MAC |
| floor | 0.52 | read the payload only, discard (traffic floor) |

Whole-token measurement: 28 layers x 7 projections (196 kernels) captured as one
CUDA graph — the real workload, 2.82 GB of distinct weights, so the result cannot
be L2-resident. `block_n=8, block_k=1024`:

| variant | ms/token | achieved |
|---|---|---|
| bf16 | 13.32 | 211.6 GB/s (card's practical ceiling ~240) |
| unfused | 15.99 | slower than bf16 |
| **fused** | **5.91** | **2.25x faster than bf16, 2.71x faster than unfused** |
| floor | 3.15 | 223.9 GB/s |

Correctness: fused vs an fp32 reference, worst relative error **2.2e-07** over 14
projections — pure accumulation-order error, i.e. the in-kernel unpacking is exact.
Peak VRAM 3.50 GiB (guard added: above 7 GiB the run is rejected, see below).

Conclusions:

1. **Fusion flips the sign, as predicted.** Compression goes from 1.20x *slower*
   than bf16 (unfused) to 2.25x *faster* (fused). The scratch round-trip was the
   whole penalty, not the codec.
2. **It captures ~54% of the theoretical win.** Reading 0.70 GB instead of 2.82 GB
   at the floor rate would be 3.15 ms; we get 5.91 ms. So ~2.8 ms/token is still
   in-kernel decode/dispatch overhead — the tuning target, not the traffic.
3. Tuning matters a lot: fused went 8.19 ms (32/128) -> 6.63 (16/512) -> 5.80
   (8/1024) -> 5.71 (4/2048) -> 7.20 (8/2048), while bf16 stayed flat at 13.3 and
   the floor flat at 3.1-3.2. Narrow row blocks with a wide K block win.
4. The unfused figure here is *optimistic*: its scratch is one reused 25 MB buffer,
   partly L2-resident (implied >390 GB/s). Real DF11 materialises the whole model
   (5.6 GB, never L2-resident) which is why it measured 2.32x there and 1.20x here.

Measurement hazards hit and fixed during this work (all three would have produced
a wrong conclusion):

* **L2 residency.** Re-running one 8-25 MB matrix 200x measures L2 and reports
  486-730 GB/s on a ~240 GB/s card. Fixed by evicting L2 before every timed call
  and by making the headline number a 2.82 GB whole-token pass.
* **Dirty-writeback flush.** Zeroing 128 MB to evict L2 leaves dirty lines that
  write back *during* the timed kernel. Switched to a read-based eviction.
* **VRAM oversubscription.** Keeping fp32 `w_deq` for all 196 projections (~8.5 GB)
  made Windows page to system RAM and produced a confident, meaningless 7 GB/s /
  401 ms-token. Now: minimal residency + an explicit peak-VRAM guard.

Not yet done: this codec is 4-bit *lossy*. The 21.9% entropy layer on top of it is
variable-rate and therefore not addressable, so fusing it needs a fixed-length
redesign (the ZipServ/TCA-TBE direction) — that is the next real question.

## E6 addendum — the fused win is a *throughput* win, not a per-kernel win

The two measurements in this section disagree, and both are correct:

| measured how | bf16 | fused | verdict |
|---|---|---|---|
| one kernel at a time, L2 flushed | 12.5 ms/token(28L) | 14.3 | fused looks *worse* |
| 196 kernels in one CUDA graph | 13.29 | **6.04** | fused 2.2x better |

Reason: with BLOCK_N=8 the fused kernel's grid is small (128-1536 programs) and a
single isolated GEMV cannot saturate the GPU — it is latency-bound, not
bandwidth-bound. Pipelining 196 of them, which is what a real token does, keeps
enough work in flight to hide that, and then the 4x lower byte count dominates.
cuBLAS GEMV does not need this help, so its number barely moves between the two
methods (212 vs 226 GB/s).

Consequence for how to read E6: the 2.2x is the **whole-token** figure, and it is
the one that matches deployment (Ollama replays a graph of all kernels). The
per-kernel table above should not be used to claim a speedup at a single
projection. If this were ported to a real runtime, the same caveat applies:
batch-1 decoding of one projection at a time will not show the win.

Reproduced twice at block_n=8/block_k=1024: fused 5.91 and 6.04 ms/token.
