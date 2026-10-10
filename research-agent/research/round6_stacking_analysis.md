# Round 6 — Can Compression Methods Actually Stack? Evidence-Based Analysis
### Weight sharing × quantization × lossless entropy coding · verified 2026-10-07 · GPU-memory-first

Builds directly on Rounds 4–5 (no re-derivation). Every claim below was checked against a repo, issue thread, recipe doc, or paper text this round. Legend: ✅ verified · ◐ secondhand-consistent · ⚠ unverified claim.

---

## 0. The VRAM-during-inference taxonomy (the question that decides everything)

A compression method only reduces *inference* VRAM if the compressed form is what **stays resident** while running. Three classes exist:

| Class | Behavior | Inference VRAM | Examples (verified) |
|---|---|---|---|
| **R1: Compressed-resident** | compressed bytes live in VRAM; decode fused into GEMM/GEMV per tile | = compressed size | DFloat11 (Huffman streams + fused CUDA kernel) ✅; GPTQ/AWQ int4 packed + Marlin ✅; GGUF quant blocks ✅; EXL3 trellis ✅; NeuZip (layer-wise on-the-fly) ✅; EOQ `QuantizedLinear` (INT4-packed resident, 2.8× vs FP16) ⚠ solo-dev; ISCA'26 ANS tiles (SGLang) ⚠ no code |
| **R2: Decompress-at-load** | compressed on disk, expands to full working format in VRAM before/at load | = **uncompressed working size** | ZipNN ✅ (storage/transfer only); QStore ✅ (multi-precision storage); **EOQ's rANS layer in llama.cpp** (C rANS decodes at load, ~114 MB/s → resident form is the INT4-packed quant, not the rANS stream) ⚠; SqueezeLLM's `.pt` downloads (packed, but kernels consumed packed form — actually R1 in its day) |
| **R3: Streamed/offloaded** | part of the model never fully resident | < model size, but host-RAM traffic | NeuZip-cpu_offload, DFloat11 `cpu_offload=True`, BitStack (dial), llama.cpp layer-split (the Round-1 disease) |

**"5 GB on disk is useless if it expands to 8 GB in VRAM" — exactly right, and it's the most common failure in this space.** ZipNN/QStore are R2 (they win disk/network, not VRAM). DFloat11 and int4-Marlin are R1 (they win VRAM). Any stack must be R1 at every layer that claims VRAM savings. Entropy layers are the hardest to keep R1 (sequential bitstreams resist random access — the ISCA'26 tile-ANS design exists precisely to solve this; SqueezeLLM's LUT kernels solved it in 2023; EOQ ships 16 CUDA kernel variants + a C rANS decoder ⚠ solo-dev quality).

---

## 1. Layer-by-layer status (what actually runs today)

### Layer 1 — Weight sharing / redundancy reduction
| Method | Code? | Stacks with quantization? | VRAM class | Evidence |
|---|---|---|---|---|
| ShortGPT layer removal | ✅ repo + PrunaAI integration track | ✅ trivially (pruned model is just a smaller model — quantize afterwards; Minitron-style prune→distill→PTQ is industry-proven: NVIDIA prunes Llama-3.1-8B→Minitron-4B, community AWQ/GGUF quantizes it) | R1 after requant | ✅ |
| SliceGPT / LaCo | ✅ microsoft/SliceGPT ◐ | same as above (needs light healing for quality) | R1 after requant | ◐ |
| **DeltaLLM** (share + low-rank deltas) | ❌ models claimed released are NOT on HF (API-verified R5); no repo | **unknown, never demonstrated.** Technical tension: a shared base weight is read by every sharing block ⇒ its quantization error is *reused k times*; deltas are small and could stay BF16 while base goes low-bit, but nobody has measured this interaction | would be R1 if kernels fused `W_shared + A·B` (no engine supports it today) | ⚠ paper-only |
| Basis Sharing / Cross-layer Attention Sharing / MoBE | ❌ no public code | unknown | — | ⚠ |
| **LegoLM** (global codebook = sharing AND quantization fused in one step) | ❌ no official code; **fully specified, data-free, reimplementable** (→ our `prototype/stageC_legolm.py`) | it *is* a quantizer (VQ-style) — composition question becomes "codebook indices + entropy coding" (Layer 3) | R1 achievable with LUT-GEMV (i-quant pattern) | ⚠ claims unreplicated: +0.03% PPL @ 4.41× |

**Verdict Layer 1:** only *pruning-style* redundancy removal has a demonstrated path into quantized production formats. True weight-sharing methods are paper-only; the sharing→quantization interaction is **unmeasured by anyone** (our prototype opportunity, and our flagged risk: shared-weight error amplification).

### Layer 2 — Quality-preserving quantization (production tier)
- W4A16 AWQ/GPTQ/AutoRound + Marlin/Machete kernels, FP8, **2:4-sparsity × int4/FP8 combos**: all ✅ via `llm-compressor`/`compressed-tensors` with official vLLM recipes — including the exact stacked recipe **"int4 Weight Quantization of a 2:4 Sparse Model"** (docs.vllm.ai example, Llama2-7B) ✅. This is R1: sparse-compressed format (50% values + 2-bit metadata) resident, kernels read it directly.
- VPTQ/AQLM codebook VQ: code real, engine support second-tier (R5 finding).
- **Order matters when stacking lossy stages:** *"Prune-then-Quantize or Quantize-then-Prune?"* (ICLR 2026, arXiv:2603.18426) ✅ — introduces the **Progressive Intensity Hypothesis** (weaker perturbation before stronger), with theory + experiments showing order changes outcomes; prior work had "sidestepped the issue by assuming orthogonality." SparseGPT's classical result (prune then quantize per-layer) is the special case. **Lossless layers (Layer 3) perturb nothing ⇒ order-free.**
- Quality caveat for small models: "Revisiting Pruning vs Quantization for SLMs" ◐ — quantization beats pruning on fidelity at small scale; pruning-first stacking costs more quality the smaller the model.

### Layer 3 — Lossless compression of the (already quantized) model
The reconciliation of contradictory claims — **it's the quantization recipe that sets the entropy headroom:**

| Source | Format measured | Residual lossless headroom | Status |
|---|---|---|---|
| ISCA'26 ANS paper §II-B ✅ | deployed **AWQ-int4 / SmoothQuant-int8** (group-wise scales) | **1.1–1.3×** | measured, credible |
| ISCA'26 ✅ | raw INT4/FP4 tensors (no group metadata) | up to 6–10× (entropy ratio) | measured; not a deployed format |
| ISCA'26 ✅ | BF16 | ~1.5× (4–5 redundant bits of 16) | matches DFloat11 |
| DFloat11 maintainer (issue #15) ✅ | BF16 ≈ 10.4 effective bits; **INT8 "already >7 bits entropy — gain minimal; not prioritizing quantized formats"** | — | honest negative evidence |
| Community measurement in same issue (zji996, Dec 2025) ✅ | **FP8 E5M2, per-channel PTQ from BF16** (1.17B elements, Wan2.1-14B): full-byte entropy 5.53/8 → ~**1.43×**; E4M3: 6.52/8 → ~1.2× | recipe-dependent | real numbers, script offered in-thread |
| **EntroLLM** (ICASSP 2026, arXiv:2505.02380) ✅ | own mixed unsigned/asymmetric quantization, **tensor-level** scales ("entropy-reducing effect"): up to **65% storage cut vs uint4** (≈2.9×!), 30% vs uint8; **31.9–146.6% faster inference on Jetson P3450**; parallel Huffman decode; no retraining; PTQ-pipeline-compatible | huge headroom — because tensor-level (not group-wise) quantization maximizes skew | peer-reviewed, edge hardware; code not located ⚠ |
| **drxddy/ecq** ✅ (repo audit) | naive 4-bit of Qwen2.5-0.5B/1.5B, GPT-2, SmolLM: nibble entropy **1.12–1.54 bits** → rANS **3.27–3.58×** beyond 4-bit | huge headroom — same cause: global/simple scaling ⇒ extreme skew | 5 commits, research prototype; **tok/s table (42.9→140) is a projection — the Metal kernel is in "Next Steps", not built** ⚠ |
| **EOQ** (`caiovicentino/eoq-quantization`) ✅ repo + HF models verified | absmax quant + rANS (`.eoq` format): "Q5 11–41% smaller than GGUF Q4_K_M, PPL delta <1.0"; Qwen3.5-35B-A3B 69→20 GB, 30 tok/s on RTX PRO 6000; PolarQuant+AWQ v3 PPL 6.43 vs FP16 6.37 | rANS at rest; INT4-packed resident | **real code (30 commits, 200+ tests, 16 CUDA kernels, C rANS, llama.cpp patches, 10 HF models — API-verified `caiovicentino1/*`)**; solo dev + LLM-co-authored; adoption ≈ 0 (8–21 downloads); "zero inference overhead / identical tok/s to FP16" phrasing is suspect — the honest reading of their own logs ("GPU dequant 3.5s, 45.8 tok/s") is R2-for-rANS + R1-for-int4 |
| **QStore** (arXiv:2505.04081 v2, Oct 2025) ✅ | stores low-precision model + residual to reconstruct high precision, losslessly; 2.2× storage, 1.7–1.8× faster save/load | storage-layer stacking (two precisions, not two codecs) | R2; different goal (multi-precision serving) |
| SqueezeLLM (2023) ✅ R5 | k-means non-uniform 3/4-bit + sparse sidecar + packed format + LUT CUDA kernels; was in vLLM, removed | quant+entropy-ish packing+sidecar fused — the original full stack | dormant |

**Synthesis (the key insight of Round 6):** entropy headroom after quantization is **not a property of the bit-width — it's a property of the scaling recipe.** Group-wise-scaled AWQ/GPTQ (the production quality leaders) flatten the symbol distribution → only 1.1–1.3× left. Tensor-level/global-scale quantizers (EntroLLM, ECQ, EOQ-absmax) keep extreme skew → 2.9–3.6× left, *but* those recipes are exactly the ones that lose more quality at equal bits. So the stack "quantize → entropy-code" trades against itself: **the more quality-preserving the quantizer, the less the entropy layer can win.** Exceptions with real headroom AND production quality: **FP8-E5M2 per-channel (≈1.4× measured)** and **scales/zeros metadata** of GPTQ/AWQ containers (fp16 scales — our Stage-B probe measures this separately; EOQ found nibble-level GGUF compression 3.34% vs byte-level 1.49% ⚠).

---

## 2. Combination matrix — does it actually work?

| Combo | Demonstrated? | Where | Total size effect | Quality | Speed | VRAM class |
|---|---|---|---|---|---|---|
| 2:4 sparsity + W4A16 int4 | ✅ **official recipe** | llm-compressor docs (vLLM), Marlin 2:4 kernels | ~2×(sparse) × ~4×(int4) on weight bytes vs BF16 | small additive cost | ≈ FP16 decode | R1 |
| 2:4 + FP8 | ✅ official recipe | llm-compressor/vLLM docs | ~4× vs BF16 | small | ≈FP16 | R1 |
| Prune(+distill) → PTQ | ✅ industry | Minitron → community AWQ/GGUF | 2–40× (prune) × 4× (quant) | distillation-healed | faster (smaller) | R1 |
| Quant → lossless entropy (group-wise int4) | ✅ measured, modest | ISCA'26 (1.1–1.3×); DFloat11 maintainer concurs | ≤1.3× extra | **zero** (lossless) | neutral IF tile-fused kernel (else load-time cost) | R1 only with fused kernel (ISCA/EOQ-int4); else R2 |
| Quant → lossless entropy (tensor-scale/absmax int4, FP8-E5M2) | ✅ working code exists | EntroLLM (Jetson, +32–147% speed); EOQ (RTX PRO 6000); ECQ (measurements + rANS codec, kernel unbuilt) | 2.9–3.6× extra (int4-absmax), ~1.4× (FP8 E5M2) | zero from codec; **quantizer itself is the quality risk** | faster on bandwidth-bound decode (EntroLLM measured; ECQ projected) | R1 (EntroLLM parallel Huffman; EOQ kernels ⚠) |
| BF16 → lossless (DFloat11) | ✅ shipping | pip + HF models + fused kernels | 1.45× | **bit-identical** | decode ≥ baseline; prefill −slight | R1 |
| Weight-share → quantize | ❌ **nobody** | — (DeltaLLM BF16-only; LegoLM fuses both in one step but unreplicated) | theoretical: share 1.15–1.3× × quant 4× × entropy 1.1–1.5× | **unknown risk: shared-base error amplification** | needs fused `base+delta` kernels (none exist) | would be R1 only with new kernels |
| Prune → quantize ORDER | ✅ studied | ICLR 2026 order paper (PIH: weaker perturbation first) + SparseGPT classic | — | order-dependent, theoretically characterized | — | — |
| ZipNN/QStore anywhere in stack | ✅ but storage-only | — | disk/download wins | zero | load-time cost | **R2 — no VRAM win** |

**Why nobody built sharing→quantize→entropy:** (1) kernel multiplication — each fused stage needs its own GEMM, and stacks need ONE kernel doing all stages (SqueezeLLM did 2 stages in 2023 and still got dropped from vLLM for maintenance cost); (2) the quality-preserving quantizers (group-wise) leave little entropy headroom, so stage 3 pays kernel complexity for ≤1.3×; (3) sharing methods never shipped artifacts to stack *on*. The pieces that DO stack today are exactly the ones with matching kernel economics: sparsity×quant (both Marlin-native) and skew-quant×entropy (both bandwidth-bound-decode wins).

---

## 3. Answers

**1. Best combination already demonstrated:** `2:4 sparsity + W4A16 (or FP8) via llm-compressor → vLLM Marlin` ✅ — official, documented, R1, ≈4–8× vs BF16 with kernel-speed ≈ FP16 decode. Runner-up (demonstrated on real hardware, weaker support): `tensor-scale quant + Huffman/rANS` (EntroLLM on Jetson: +32–147% inference speed; EOQ on RTX PRO 6000) and `BF16 + DFloat11` (bit-identical, 1.45×, shipping).

**2. Best combination nobody has built:** **weight-sharing (DeltaLLM/LegoLM-class) × group-wise quantization × tile-fused entropy coding** — i.e., a single resident format `{shared bases (low-bit, entropy-coded) + per-layer low-rank deltas (BF16/FP8) + outlier sidecar}` decoded inside one fused GEMM. No repo, no paper measures the interaction. The LegoLM↔entropy sub-question (are global-codebook indices compressible?) is answerable in a day with our Stage-B probe; the sharing↔quantization error-amplification question is answerable in a week with stageC_legolm + lm-eval.

**3. Expected memory savings (honest, R1-only):**
- Demonstrated stack: BF16 → 2:4+W4A16 ≈ **6–8×** (resident); quality cost small at ≥7B.
- + entropy layer on the *scales/metadata and skewed tensors*: +**5–20%** (measured range 1.05–1.43× depending on recipe; 1.1–1.3× typical for group-wise).
- Sharing layer (if it replicates): +**12–25%** param cut (DeltaLLM-class) at ~90% perf retention, or LegoLM-class 4.4× *as the quantizer itself* (unreplicated).
- Full speculative stack: **~8–12× vs BF16 resident** — with the caveat that the last 20–30% requires writing the fused kernel nobody has published.
- Disk-only tricks (ZipNN/QStore/zstd): additional ~1.3–2.2× on *storage*, **0× on VRAM**.

**4. Expected speed impact:** every R1 stage that reduces resident bytes *helps* bandwidth-bound decode (Marlin ≥FP16; EntroLLM +32–147% on edge; DFloat11 neutral-to-faster decode). Costs appear in: prefill/compute-bound phases (dequant ALU), load time (R2 stages: EOQ's rANS ~114 MB/s C decoder adds seconds), and any non-fused decompress-store-compute pattern (ZipNN/NeuZip layer-sync — the ISCA critique). Rule: **fuse into GEMM or don't bother.**

**5. Quality risk:** (a) lossless stages: zero by construction (verify with greedy-identity like Stage A); (b) 2:4+int4: small, characterized, recipe-supported; (c) skew-quant+entropy stacks: the *quantizer* carries the risk (absmax/tensor-scale loses more at equal bits than AWQ — that's WHY the entropy headroom exists); (d) sharing stages: unquantified — shared-base error reuse is the specific new risk; layer-pruning at small scale costs more (SLM study); (e) order effects are real (ICLR'26) — apply lossy stages weakest-perturbation-first, lossless anytime.

**6. Exact repos to start from:**
- `vllm-project/llm-compressor` + recipe `quantization_2of4_sparse_w4a16` (demonstrated stack) ✅
- `LeanModels/DFloat11` (lossless R1 baseline + kernel patterns) ✅
- `caiovicentino/eoq-quantization` (closest existing full attempt: rANS codec, .eoq format, CUDA kernels, llama.cpp patches, HF models `caiovicentino1/*-EOQ-*` — audit before trusting) ✅⚠
- `drxddy/ecq` (entropy measurements + rANS codec + `entropy_analysis.py` — measurement code worth borrowing; kernel is a design doc) ✅⚠
- `SqueezeAILab/SqueezeLLM` (LUT-kernel blueprint for packed-format inference) ✅ dormant
- `BorealisAI/neuzip`, `xinghaow99/BitStack` (compressed-resident plumbing references) ✅
- `microsoft/VPTQ`, `intel/auto-round` (Layer-2 alternatives) ✅
- Papers: 2603.18426 (order/PIH, ICLR'26), 2505.02380 (EntroLLM, ICASSP'26), 2505.04081 (QStore), 2606.15789 (ISCA ANS), 2608.08652 (LegoLM), 2501.18596 (DeltaLLM), 2404.10896 (Liguori FPGA-ANS, via ECQ's prior-art list).

**7. Smallest prototype to prove/disprove the idea** (see `prototype/README.md` "Stack-1" + new `prototype/stageC_legolm.py`):
- **P0 (1 day, off-the-shelf):** llm-compressor `quantization_2of4_sparse_w4a16` on Llama-3.1-8B → vLLM → record VRAM/tok-s/PPL; compare vs plain W4A16 and BF16. *Proves the demonstrated stack on our hardware.*
- **P1 (½ day, CPU-only):** run `stageB_entropy_probe.py` on the P0 outputs AND on an absmax-quantized copy AND an FP8-E5M2 copy. *Tests the reconciliation table: expect ~1.1× (group-wise), ~2×+ (absmax), ~1.4× (FP8-E5M2). If confirmed, the entropy-layer target format is chosen by data, not faith.*
- **P2 (1 week, data-free, single GPU):** `stageC_legolm.py` — LegoLM replication (K=64 global codebook, p99 sidecar, boundary protection) on Mistral-7B/Llama-3.1-8B → PPL + size + **feed its index stream to stageB probe** (answers "does sharing output entropy-code well?"). Gate: ≤1% PPL @ ≥3×.
- **P3 (2–4 wks, only if P1+P2 pass):** Triton fused LUT-GEMV reading entropy-coded indices resident (blueprints: SqueezeLLM LUT kernel, EOQ's kernel variants, ISCA tile-ANS design). Measure vs Marlin baseline. *This is the novel artifact; kill if <0.9× Marlin speed.*
