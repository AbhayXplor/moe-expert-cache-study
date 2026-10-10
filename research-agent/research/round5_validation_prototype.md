# Round 5 — From Findings to Buildable Reality: Validation Report & Prototype Plan
### Skeptical audit of Round-4 candidates · verified 2026-10-07 · A–F deliverable structure

Legend: ✅ verified hands-on this round (fetched repo/paper/HF API) · ◐ consistent secondhand evidence · ⚠ claim not independently verified.

---

## 1. Method-by-method validation: what ACTUALLY works today

### 1.1 DFloat11 — ✅ REAL AND USABLE (the strongest validated artifact)
- **What works today:** `pip install dfloat11[cuda12]`; `DFloat11Model.from_pretrained("DFloat11/<model>-DF11")` HF-style API; pre-compressed lossless models hosted at HF org `DFloat11` (Llama-3.x family incl. Qwen3, Gemma 3, Phi-4 per README; plus diffusion models FLUX.1/Wan2.1); fused CUDA Huffman-decompression kernels (v0.2.0: kernels 20–40% faster); safetensors storage; only the compressed model downloads (not the base).
- **Inference directly from compressed form?** YES for the resident-GPU path: weights live in VRAM Huffman-coded, decode fused into per-layer GEMV. Output is bit-for-bit identical by construction (lossless codec).
- **Actual numbers:** ~30% size cut (16→~11 bits). Decode (bandwidth-bound): neutral-to-faster. Prefill: slight overhead (⚠ exact % not published in README; paper claims "efficient GPU inference" — no independent third-party benchmark found).
- **Limitations:** its own runtime only — NOT integrated into vLLM/llama.cpp/transformers mainline (no merged PR found); CUDA 12 + NVIDIA only; LLM coverage = what the DFloat11 HF org has uploaded or what you compress yourself (compression code released Aug 2025, `examples/compress_flux1` shows the path); `cpu_offload=True` mode exists but streams from RAM (violates our isolation goal — don't use).
- **Activity:** 12 commits, last Nov 2025 (CUDA kernel add). Small but functional. Verdict: **usable today for a validation baseline; not production-engine material.**

### 1.2 DeltaLLM (cross-layer sharing + low-rank deltas) — ⚠ PAPER-ONLY in practice
- Paper (arXiv:2501.18596) claims "we release the resultant models (DeltaLLAMA, DeltaPHI)". **HF API search finds neither** (2026-10-07: `search=DeltaLLAMA` → 1 unrelated community model; `search=DeltaPHI` → 0 results). **No code repo located.**
- What the paper credibly shows: inter-block weight sharing + LoRA-class deltas trained on ~30–40M tokens; 12% param cut @ ~90% perf; 24% cut beating fine-tuned-recovery SliceGPT at 12%.
- Verdict: **method is reproducible from the description** (share block N's weights into block N+1, fit low-rank deltas — a single-GPU fine-tuning job), but there is nothing to clone. Research-only. Any prototype here is a reimplementation, not an integration.

### 1.3 Basis Sharing (arXiv:2410.03765) & Cross-layer Attention Sharing (TACL 2026) — ⚠ no public code found
- Concept validated in print (per-layer weights ≈ linear combination of shared bases; attention sharing has extra pitfalls). No repos surfaced in repeated searches. Research-only.

### 1.4 LegoLM (global codebook + outlier sidecar) — ⚠ no official code; BUT trivially reimplementable
- Single-author 8-page preprint (arXiv:2608.08652, Aug 2026); no GitHub located. Claims (+0.03% PPL @ 4.41× on Mistral-7B; 5.12× within noise at K=64/p=99) are **plausible but unreplicated** — treat as hypothesis.
- Why it matters anyway: the algorithm is ~200 lines (global k-means over weights → 4–6-bit indices; keep top ~1% |w| verbatim in a sidecar; skip first/last blocks; scalar-block normalization). The failure-mode analysis (outlier dominance: +1,134,279% PPL without protection) is the real contribution and is independently consistent with SpQR's findings.
- Verdict: **prime reimplementation candidate** (Stage-C alt track); data-free, no kernels needed for a first quality/memory measurement (dequant-at-load), Triton LUT kernel later for speed.

### 1.5 Entropy/ANS compression (ISCA 2026, arXiv:2606.15789) — ✅ methodology verified, ⚠ CODE NOT PUBLIC, and the headline needs deflating
Verified from the paper itself (Section II-B, fetched):
- **BF16:** 4–5 bits of the 16 are redundant → **~1.5× lossless** (consistent with DFloat11's measured ~30% ✓ — two independent groups agree).
- **Raw INT4/FP4 tensors:** heavily skewed symbol distributions, entropy ratios "reaching 6–10×" — but these are *raw uniform-quantized* tensors, not deployed formats.
- **Deployed group-quantized formats (AWQ-4bit, SmoothQuant-8bit): only ~1.1–1.3× above their entropy bound.** ← **This is the number that matters for "entropy-code an already-quantized model," and it's 10–30%, not 5–10×.**
- Intro's "up to 27% of stored bits redundant on average" and "up to 5× in large models" coexist awkwardly with the above; the defensible, format-specific claims are the three bullets above. **Skeptical reading: expect ~1.2–1.3× additional lossless savings on a deployed AWQ/GPTQ int4 model, ~1.4–1.5× on BF16.**
- Systems results (SGLang, multi-GPU): Qwen-14B batch 47→75 (+1.2× throughput), Mixtral-176B 20→95 (+1.6×), up to 11× vs NeuZip/DFloat11 *in their serving benchmark* (their measurement, ⚠ not independently reproduced; the 11× is vs those tools' naive layer-decompress-store-compute pattern, which Table I/Fig 3 of the same paper criticize — fair design point, aggressive marketing number).
- Key engineering insight verified: sequential entropy streams are not tile-addressable; their fix = per-tile ANS streams decoded in shared memory exactly when the GEMM microkernel consumes them (no global-memory materialization). rANS/tANS/FSE chosen for >99% Shannon efficiency + byte-level streaming.
- **Code: none public** (no artifact repo found; ISCA'26 artifact release possible later). Verdict: **research-only today; the highest-value rebuild target, and its entropy methodology is cheap to replicate ourselves** (→ `prototype/stageB_entropy_probe.py`, this round).

### 1.6 VPTQ / codebook VQ — ✅ real code, ⚠ engine support is second-tier
- `microsoft/VPTQ` ✅ active repo; EMNLP 2024; vLLM integration merged (per repo docs); 2-bit-class VQ with second-order optimization + residual codebooks; kernels included.
- Reality check: the Jan-2026 vLLM quantization benchmark guide (H200) covers AWQ/GPTQ/Marlin/BitBLAS/GGUF/BnB — VPTQ is not in the mainstream bench set; vLLM's Dec-2025 RFC #30136 is *deprecating legacy formats* toward llm-compressor/modelopt/quark/torchao. AQLM: vLLM support since 2024 (issue #3439 ✅) but同样 niche. Verdict: **works, but expect to own the integration; the durable production path for codebook-style compression in 2026 is EXL3 (QTIP-trellis) in ExLlamaV3** ✅ Round-2.

### 1.7 NeuZip — ✅ official repo, research-grade
- `BorealisAI/neuzip` ✅. Entropy-based dynamic compression: exponent/mantissa streams separated and compressed; decompressed layer-by-layer on the fly. Lossless for training, lossy variant for inference. Transformers-based research code — it demonstrates compressed-resident execution (and pays the layer-sync cost the ISCA paper criticizes). Not a serving engine. Verdict: **useful reference implementation for on-the-fly decompression plumbing; not a dependency.**

### 1.8 Qronos / error-diffusion rounding — ⚠ paper-only; ✅ but a production cousin exists
- Qronos (arXiv:2505.11695, ICLR-track, 26 citations): sequential rounding that propagates past rounding error into future rounding decisions; training-free; gains at w2–w4. **No official code repo found.**
- **Production alternative: `intel/auto-round`** ✅ — active Intel-maintained PTQ toolkit doing learned sign-rounding (same "smarter rounding" family), exporting to GPTQ/AWQ/llm-compressor formats consumable by vLLM. This is the *usable* version of the idea.

### 1.9 Structured sparsity — ✅ the most production-ready stage
- One-shot SparseGPT/Wanda → 2:4; official **llm-compressor** recipes incl. 2:4+FP8 stacking ✅ (vLLM docs); Marlin 2:4 kernels in vLLM. Caveat: 2:4 halves weight bytes only with compressed-sparse storage (metadata+50% values); kernel ecosystem supports it; quality cost small but non-zero at ≤8B scale.

### 1.10 SqueezeLLM — ✅ the historical proof of the whole thesis, ❌ dead code
- `SqueezeAILab/SqueezeLLM` ✅: k-means non-uniform 3/4-bit + sparse outlier sidecar (0.05%/0.45%) + entropy-efficient packing + **LUT-based CUDA GEMV kernels** (paper: "3/4-bit CUDA LUT-based kernels for matrix-vector multiplication") — i.e., **inference ran directly from the packed/entropy-coded representation** in 2023. Was merged into vLLM (Oct 2023, README links the old file) and has since been **removed from current vLLM** (absent from the 2026 supported-format list; RFC #30136 deprecates legacy formats). Repo dormant since Mar 2024; models are LLaMA-1/2-era. Their README claim "Vicuna-3bit in 6 GB, +2% MMLU vs FP16" ⚠ is self-reported on 2023 models.
- Why it still matters: it demonstrates end-to-end that **quantize + entropy-pack + sidecar + fused kernel** shipped and served real models. The 2026 gap is that nobody re-did this on modern stacks (Marlin-class kernels, GPTQModel/llm-compressor formats, current models). **That gap is our prototype.**

---

## 2. Combination analysis — what actually composes, what conflicts

| Combination | Status | Evidence |
|---|---|---|
| Quantization → entropy coding of the quantized stream | **Real but modest: ~1.1–1.3× on deployed AWQ/SQ8 formats** (ISCA'26 §II-B measured) | SqueezeLLM proved the mechanics; ISCA'26 measured the ceiling; DFloat11 proves it for BF16 (1.5×) |
| Sparsity (2:4) → quantization (FP8/int4) | ✅ shipping | llm-compressor official recipe + vLLM Marlin kernels |
| Pruning → light healing (deltas/LoRA) | ✅ demonstrated, artifacts missing | DeltaLLM paper; "deeper layers"+QLoRA healing ◐ |
| Sharing → quantization | ⚠ unknown / plausible tension | shared bases concentrate information → residual deltas may be *less* compressible per-bit but far fewer; nobody has measured. LegoLM's codebook IS a form of quantization (global VQ) — it composes with outlier sidecars by construction |
| Codebook VQ → entropy-coded indices | partially done | AQLM/VPTQ pack indices; a full ANS pass over indices is unexplored ⚠ |
| Entropy coding ↔ speed | the crux | Sequential codecs (zstd/Huffman whole-layer, ZipNN, NeuZip) put decode on the critical path (layer-sync stalls — ISCA'26 Fig 3 critique). **Tile-fused decode (Marlin-style for quant, ANS-tiles for entropy) is the only pattern that keeps inference fast.** Fused dequant cost today: ≈0–10% (Marlin ≥FP16 decode; llama.cpp i-quants −5% tg ✅ Round-1) |
| Pruning → sparsity → quant → entropy (full stack) | ❌ nobody ships it; partial stacks ship | This is the whitespace |

**Decompression compute budget rule (verified pattern):** if decompression happens *inside* the GEMM/GEMV inner loop from shared memory (Marlin, i-quants, ISCA'26 ANS tiles), cost ≈ 0–10%; if it materializes decompressed weights in global memory first (ZipNN, NeuZip layer-decompress), you pay bandwidth twice + sync barriers.

---

## 3. The "missing piece" audit (Q4): quantized model + strong lossless compression + efficient inference from compressed form

- **Closest working precedent:** SqueezeLLM (2023) — dead stack, 3/4-bit non-uniform + packing + LUT kernels.
- **Closest live measurements:** ISCA'26 ANS-on-SGLang (numbers above, no code) — and its honest per-format ceilings (1.1–1.3× on AWQ/SQ8; up to 6–10× only on raw INT4 tensors, which nobody deploys raw).
- **Adjacent live code:** DFloat11 (BF16 only), ZipNN (storage/transfer only, 1.2–1.7 GB/s — too slow for on-the-fly), GGUF i-quants (codebook-lattice but no entropy stage), EXL3 (trellis-coded — rate-distortion optimal *quantization*, not an entropy layer over someone else's quantization).
- **Verdict:** the missing piece is real but its honest prize is **~1.2–1.3× additional lossless reduction on deployed int4 models** (e.g., 4.0 GB → ~3.1–3.3 GB), plus faster loading/distribution — NOT 5–10×. It becomes compelling when (a) stacked on everything else, (b) the codec is tile-fused so speed is neutral, (c) applied where skew is highest (per-tensor adaptive codecs; raw-int4-style distributions; scales/zeros metadata).
- **Kill-criterion discipline:** measure before building kernels. `prototype/stageB_entropy_probe.py` (this round) computes exactly the ISCA'26 quantity (per-tensor symbol entropy of stored representations + Huffman bound + optional zstd) on ANY safetensors model, CPU-only, in minutes. **If measured headroom < ~15% on your target quantized format, the entropy-layer direction is dead for that format and should be abandoned before any kernel work.**

---

## A. What is actually real and usable today
1. **DFloat11** — lossless ~30%, pip-installable, fused CUDA kernels, own runtime (not vLLM/llama.cpp). Best validated lossless artifact. ✅
2. **llm-compressor stack** — AWQ/GPTQ W4A16, 2:4 sparsity, FP8, sparsity+FP8 stacking; Marlin-class kernels in vLLM; the production spine. ✅
3. **Intel AutoRound** — active "smarter rounding" PTQ (Qronos-family idea, productionized), exports to vLLM-consumable formats. ✅
4. **EXL3 / ExLlamaV3 (QTIP trellis)** — codebook/trellis compression with real kernels on consumer GPUs. ✅ Round-2
5. **NeuZip, BitStack** — compressed-resident research runtimes (fit >VRAM models; speed cost). ✅ repos, research-grade.
6. **SqueezeLLM** — complete historical reference implementation of quantize+pack+LUT-kernel (read the code, don't deploy it). ✅ repo, dormant.

## B. Promising but research-only (no usable code today)
1. **ANS tile-fused entropy coding (ISCA'26)** — the design is the prize; artifact not public; per-format ceilings now known (1.1–1.3× deployed quant; 1.5× BF16).
2. **DeltaLLM** — models claimed released are not discoverable; method reimplementable (~30–40M-token healing runs).
3. **LegoLM** — no code; algorithm fully specified; claims unreplicated (+0.03% PPL @ 4.4–5.1× would be remarkable — verify before believing).
4. **Basis Sharing / Cross-layer Attention Sharing / MoBE** — papers only.
5. **Qronos** — paper only (use AutoRound instead).
6. **CompactifAI tensor networks** — commercial; thin community tooling.

## C. Best combination we could realistically build
**Track 1 (production-safe, all off-the-shelf):** llm-compressor W4A16 (AWQ/GPTQ + AutoRound-style rounding) → optional 2:4 where quality allows → Marlin runtime. ~4×(×2), quality ≈ baseline at ≥7B, speed ≥ FP16 decode. Zero novel code.
**Track 2 (the differentiator, staged):** Track 1 output **+ measured entropy layer**: (i) run `stageB_entropy_probe.py` on the quantized checkpoint (go/no-go gate ≥1.15×); (ii) storage-side ANS/Huffman packing of qweights+scales+zeros with per-tensor codebooks (pure Python/zstandard first — wins loading, disk, and *cold* memory); (iii) only if (ii) shows ≥1.2×: build the tile-fused decode kernel (Triton GEMV for decode-only path first — SqueezeLLM's LUT kernel and the ISCA tile-ANS design are the two blueprints).
**Track 3 (sharing, parallel bet):** LegoLM reimplementation (global k-means + p99 sidecar + boundary protection) — data-free, one GPU, days of work; if it replicates even half the claimed ratio at <1% PPL cost, its indices become the input to Track 2's entropy layer (codebook indices are maximally skew-friendly).

## D. Exact repos to start from
| Purpose | Repo | State |
|---|---|---|
| Lossless BF16 baseline + kernel patterns | `LeanModels/DFloat11` | ✅ pip, CUDA12, active-ish |
| Quantization spine (W4A16/2:4/FP8) | `vllm-project/llm-compressor` (+ `neuralmagic/compressed-tensors`) | ✅ production |
| Better rounding | `intel/auto-round` | ✅ active |
| Codebook VQ | `microsoft/VPTQ` | ✅ active-ish, vLLM-merged |
| Entropy-packed quant reference (kernels!) | `SqueezeAILab/SqueezeLLM` (`squeezellm/` CUDA LUT kernels, packing code PR #59) | dormant — read, don't run |
| On-the-fly decompression plumbing | `BorealisAI/neuzip` | research |
| Any-size dial | `xinghaow99/BitStack` | research |
| Low-rank (data-free) | `AIoT-MLSys-Lab/SVD-LLM` | ✅ |
| Layer pruning toolkit | `PrunaAI/pruna` (ShortGPT track), `microsoft/SliceGPT` | ✅ |
| Index of everything | `HuangOwen/Awesome-LLM-Compression` | living |
| Consumer-engine codebook precedent | `turboderp-org/exllamav3` (QTIP-based EXL3) | ✅ |

## E. First prototype to build and test (concrete, staged, with kill gates)
See `prototype/README.md` + the two ready scripts in `prototype/`:
- **Stage A (½ day, any CUDA-12 GPU):** DFloat11 validation harness (`stageA_dfloat11_validate.py`) — measures compressed size, peak VRAM, tok/s vs base BF16, greedy-identity check. Establishes our measurement methodology + confirms the lossless-speed claim on OUR hardware.
- **Stage B (½ day, CPU-only, decisive):** `stageB_entropy_probe.py` on (1) a BF16 model, (2) an AWQ-int4 model, (3) a GPTQ-int4 model — replicates ISCA'26 §II-B entropy numbers per-tensor with Huffman + zstd bounds. **Gate: proceed to kernels only if measured ratio ≥1.2× on the quantized format; ≥1.4× expected on BF16.**
- **Stage C (1–2 weeks):** if B passes — storage-side ANS packer for GPTQ/AWQ safetensors (per-tensor codebooks; zstandard/FSE via `pip install zstandard` first, custom rANS later) + load-time unpack → measure load time, disk, and VRAM-parity inference; then Triton decode-only GEMV reading packed nibbles directly (blueprint: SqueezeLLM LUT kernel + ISCA tile addressing).
- **Stage C-alt (parallel, 1 week):** LegoLM replication — global k-means codebook (K=64), p99 outlier sidecar, boundary-layer protection on Mistral-7B/Llama-3.1-8B; measure PPL delta (lm-eval-harness) + compressed size; no kernels needed for the quality claim.

## F. What would make this genuinely different from existing quantization tools
1. **Format-agnostic entropy stage:** every shipping quantizer stops at the bit-packing boundary; the measured 1.1–1.3× residual redundancy on deployed formats is pure waste today. A codec layer under GPTQ/AWQ/EXL3/GGUF containers (not a new quantizer) is orthogonal to — and stackable with — everyone else's work.
2. **Tile-addressable streams:** the concrete systems insight from ISCA'26 (per-tile ANS, shared-memory reconstruction, no global materialization) is what separates "compression that costs speed" (ZipNN/NeuZip pattern) from "compression that's free" (Marlin pattern). Nobody has published an open tile-fused entropy kernel — that artifact alone would be novel and useful.
3. **Outlier-sidecar as a standard container field:** LegoLM+SpQR convergent evidence says the 1% tail carries the function; a container format {entropy-coded bulk indices + verbatim sidecar + per-tensor codebooks} with reference kernels would beat uniform formats at equal bits — and is exactly what no current GGUF/EXL3/compressed-tensors schema formalizes.
4. **Honest, measured claims:** the field's headline numbers (5–10× lossless, 4.4× @ +0.03% PPL) don't survive contact with their own per-format tables (1.1–1.3× on deployed quant; unreplicated single-author claims). A tool whose README is a reproducible measurement harness (our Stage-B probe) instead of a benchmark screenshot is itself a differentiator.

---
### Skepticism ledger (claims that did NOT hold up)
- "2–10× / up to 10× lossless" (ANS paper abstract) → per-format reality: **1.5× BF16, 1.1–1.3× deployed AWQ/SQ8**; 6–10× only for raw INT4/FP4 tensors nobody deploys unquantized-metadata-free.
- "11× throughput vs NeuZip/DFloat11" → their serving-setup comparison against layer-decompress baselines; not a like-for-like codec speed; unreplicated.
- DeltaLLM "we release the models" → **not findable on HF Hub** (2026-10-07).
- LegoLM 4.41×@+0.03% PPL → plausible mechanism, **zero independent replication, no code**; treat as hypothesis until Stage C-alt measures it.
- SqueezeLLM "+2% MMLU vs FP16" → self-reported, 2023-era models; the format/kernel mechanics are the useful part, not the quality claims.
- VPTQ/AQLM "production-ready" → merged in vLLM historically, but absent from 2026 mainstream benchmark sets and vLLM is consolidating formats (RFC #30136); own-your-integration risk.
