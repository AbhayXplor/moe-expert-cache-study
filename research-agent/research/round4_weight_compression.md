# Round 4 — Making Pretrained Weights Smaller: Redundancy, Lossless Compression, Combinations, Runtime Decompression
### Deep-research dossier · verified 2026-10-07 · hardware-agnostic (no VRAM/model-size tailoring)

Scope note: KV-cache, speculative decoding, and driver isolation are covered in Rounds 1–3 and appear here only for comparison. Confidence markers: ✅ verified this round (fetched/searched) · ◐ well-known, not re-verified this round · ⚠ unverified/uncertain.

---

## 1. Weight redundancy — is the same information stored repeatedly? YES, at four levels

### 1.1 Layer level: whole layers are partially redundant
- **ShortGPT** (arXiv:2403.03853, ACL 2025 Findings) ✅ — introduces **Block Influence (BI)**: cosine distance between a layer's input and output hidden states. Low-BI layers barely transform anything → delete them outright. Post-training, **no retraining**; works on Llama-2/3, Mistral, etc. Removing a meaningful fraction of layers costs little on standard benchmarks; aggressive rates need light recovery. Being integrated into toolkits (PrunaAI issue #418 ✅).
- **LaCo** (layer collapse) ◐ — collapses a layer into its neighbor instead of deleting; one of the baselines DeltaLLM beats (per DeltaLLM abstract ✅).
- **SliceGPT** (arXiv:2401.15024, Microsoft) ◐ — exploits *computational invariance*: apply orthogonal transforms, then slice off ~10–25% of rows/cols of every weight matrix; needs light fine-tuning to recover. Code: `microsoft/SliceGPT` ◐.
- **"The Unreasonable Ineffectiveness of the Deeper Layers"** (arXiv:2403.17887, Gromov et al.) ◐ — drop up to ~45% of layers (BI-style scoring) + **QLoRA "healing"** on a few B tokens: quality largely restored on a single GPU.
- **LoRP** (Locality-aware Redundancy Pruning) ✅ — one-shot, training-free, "recovery-free" depth pruning with baselines reproduced (`daniel-eai/LoRP-Locality-Aware-Redundancy-Pruning`).
- Takeaway: **depth redundancy is real and cheap to exploit** (hours on one GPU), but standalone quality loss grows fast beyond ~15–25%; it's best as the *first* stage feeding a repair step (see DeltaLLM) or combined with width methods.

### 1.2 Cross-layer level: adjacent/related layers store near-duplicates — store one + deltas
- **DeltaLLM** (arXiv:2501.18596, Microsoft authors) ✅ — restructures a pretrained model so **successive transformer blocks share weights**, each block keeping a small **low-rank difference (delta) matrix**. Training the deltas needs only **~30–40M tokens** (single-GPU-hours, not pretraining infra). Released models **DeltaLLAMA / DeltaPHI**: 12% parameter reduction retaining ~90% performance; DeltaPhi-2.9B at **24% reduction** matches a fine-tuned-recovery SlicedPhi-3.3B at only 12% reduction. Beats JointDrop, LaCo, ShortGPT, SliceGPT at equal removal. This is the cleanest published proof that *inter-layer deltas* compress real LLMs.
- **Basis Sharing / Cross-Layer Parameter Sharing** (arXiv:2410.03765) ✅ — represent every layer's weight matrix as a **linear combination of a shared set of basis matrices** (a learned dictionary across depth). Needs calibration/light fitting; code not located ⚠.
- **Cross-layer Attention Sharing** (TACL 2026) ✅ — systematic study of sharing *attention* weights across layers of pretrained models; identifies why naive sharing fails and how to fix it (attention is harder to share than MLP).
- **MoBE** (arXiv:2606.03465, 2026) ✅ — for MoE models: factor each expert's up/gate matrix as `W = A·B` sharing bases **across experts** (cross-expert redundancy). Same principle, MoE flavor.
- Takeaway: **the model already contains near-duplicate structure across depth**; the winning pattern is "shared thing + small per-layer correction," with correction trained on tens of millions of tokens (not billions).

### 1.3 Weight-value level: global dictionaries / codebooks
- **LegoLM** (arXiv:2608.08652, Aug 2026) ✅ — *structured global weight sharing*. Diagnoses exactly why naive global codebooks fail (**distributional mismatch** across heterogeneous layers + **outlier dominance**: ~1/K of weights sit beyond the outermost centroid → catastrophic error, +1,134,279% PPL on Mistral-7B at K=128 full replacement) and fixes it **data-free**: (1) scalar-block encoding, (2) **percentile-selective replacement** — keep the top ~1% outliers *verbatim* in a sidecar, (3) boundary-layer protection (first/last blocks untouched). Result: **+0.03% PPL at 4.41× compression on Mistral-7B** (beats PTQ-8bit in both ratio and quality); ~noise-level accuracy at **5.12×** (K=64, p=99%). Single-author 8-page paper; official code not located ⚠ — but the algorithm is fully specified and reimplementable in days (global k-means + outlier sidecar).
- **Vector quantization family = learned dictionaries with production kernels**: **AQLM** (additive multi-codebook, near-optimal 2-bit at large scale; vLLM support ✅ via issue #3439), **QTIP** (trellis-coded quantization — information-theoretically near-optimal rate-distortion; the basis of **EXL3** in ExLlamaV3 ✅ Round-2), **VPTQ** (`microsoft/VPTQ` ✅, EMNLP 2024: second-order-optimized residual VQ, extreme low-bit, vLLM integration), **GPTVQ** (Google) ◐. These are "codebook compression that already works in real engines."
- **SpQR-style outlier separation** (arXiv:2306.03078) ◐ — near-lossless 4-bit by storing the ~1% salient weights in a sparse high-precision sidecar. LegoLM's p99 sidecar is the same idea inside a global-codebook scheme — convergent evidence that **outlier sidecars are the enabling trick** for aggressive sharing/quantization.

### 1.4 Matrix-structure level: low rank and tensor networks
- **SVD-LLM** (`AIoT-MLSys-Lab/SVD-LLM` ✅; V2 arXiv:2503.12340 ✅) — whitening-aware truncated SVD, **data-free**, layer-adaptive rank; compresses LLaMA-7B at 20% ratio with perplexity+efficiency harness included.
- **ASVD** ✅ (community report: 10–20% compression "without sacrificing reasoning") — activation-aware SVD: the distribution that matters is *after* activations, so factor `W·diag(act)`.
- **CompactifAI** (arXiv:2401.14109, Multiverse Computing) ✅ — **quantum-inspired tensor networks**: collapse whole layer stacks into tensor-train-like decompositions; claims up to ~96–99% parameter cuts at moderate (tunable, size-dependent) accuracy loss, with optional fine-tuning recovery. The most radical "structure ≠ weights" approach; productionized commercially, community tooling thin ⚠.
- Takeaway: low-rank alone gives modest, model-dependent savings (10–30%) data-free; it shines as a **booster** inside other pipelines (DeltaLLM's deltas, SVD of quantization residuals, ASVD before quantization).

---

## 2. Lossless / near-lossless compression — the entropy story (NEW in 2025–2026)

The headline discovery of this round: **stored bitwidth massively overstates information content, even after quantization.**

- **"Approaching Shannon Bound with Lossless LLM Weight Compression"** (arXiv:2606.15789, **ISCA 2026**; NUS + ETH Zürich — Tan, Chen, Alonso, Wong, He) ✅
  - Entropy study across **1.5B→405B** models and formats **bf16 → int4 → AWQ/SQ8**: effective entropy is **2–10× lower** than stored bitwidth ⇒ *up to 10× lossless footprint reduction is theoretically available without changing a single weight value* — and **already-quantized weights still contain that residual redundancy**.
  - Method: **tile-level, on-the-fly ANS (Asymmetric Numeral Systems) decompression aligned with GEMM tiling** — the decoder consumes weights exactly in the order the matrix-multiply tiles need them, so compressed-resident execution costs ~nothing.
  - Results: bit-rates **within 0.01–0.1 bits of the Shannon limit**; integrated into **SGLang** (multi-GPU): Qwen-14B max batch 47→75 (+1.2× throughput); Mixtral-176B batch 20→95 (4.8×) with up to **+1.6× throughput**; **up to 11× throughput vs NeuZip and DFloat11** in serving comparisons.
  - Code: not located ⚠ (SGLang integration described; watch for artifact release).
- **DFloat11** (arXiv:2504.11651, NeurIPS 2025; `LeanModels/DFloat11` ✅) — the *usable today* lossless option: **Huffman-codes the BF16 exponent field** (~16→~11 bits, **~30% smaller, bit-for-bit identical outputs**). `pip install dfloat11[cuda12]`, HF-style API `DFloat11Model.from_pretrained("DFloat11/<model>-DF11")`, safetensors, fused **CUDA decompression kernels** (20–40% kernel speedup in v0.2.0), supports Llama-family LLMs incl. **Qwen3, Gemma 3, Phi-4** + diffusion models (FLUX.1, Wan2.1). Decode-phase: bandwidth-bound ⇒ **faster or neutral**; prefill: small overhead. Optional `cpu_offload=True` (block-streaming — uses RAM; against our isolation goal but proves the compressed-resident machinery).
- **NeuZip** (arXiv:2410.20650, `borealisai/neuzip` ✅) — dynamic compression *during* training/inference: keeps weights compressed in memory, decompresses layer-by-layer on the fly (lossless for training; a lossy inference variant). Enables fitting "impossible" sizes; slower than resident-weight inference.
- **BitStack** (arXiv:2410.23918, `xinghaow99/BitStack` ✅) — training-free decomposition into tiny sortable blocks; **megabyte-granularity** size↔quality dial; least-important blocks can live outside GPU memory. Same family as NeuZip: compressed-resident + streaming.
- **ZipNN** (arXiv:2411.05239, `zipnn/zipnn` ✅ Round-3) — lossless byte-stream codec for checkpoints (~30–50% on BF16), integrated with HF/safetensors/vLLM **storage** paths; too slow (1.2–1.7 GB/s) for on-the-fly weight decompression — it's a download/disk tool.
- **Near-lossless rounding upgrades (cheap wins on top of any PTQ):** **Qronos** (arXiv:2505.11695 ✅) — training-free sequential rounding that propagates each weight's rounding error into subsequent weights' decisions ("correcting the past by shaping the future"); SOTA-level PTQ accuracy at w2–w4. Drop-in improvement to GPTQ/AWQ pipelines.
- **Boundary cases (honest limits):** 1-bit class (**BiLLM** arXiv:2402.04291 ◐; **BTC-LLM** sub-1-bit, arXiv:2506.12040 ✅) exists but quality collapses below ~30B scale — the PTQ frontier for real quality is ~2.5–3 bpw (AQLM/QTIP-class, big models) and ~3.5–4.5 bpw (≤14B); lossless frontier is ~11 bits today (DFloat11), with the ISCA'26 entropy study implying **~5–8 bits lossless should be reachable** — that's the open engineering gap.

---

## 3. Combinations — do the savings stack? (Q3)

**Yes, and the 2026 evidence says the biggest unclaimed win is stacking *entropy coding on top of quantization*.**

Documented, working stacks today:
1. **2:4 sparsity + FP8 weights** — official `llm-compressor` recipe (vLLM docs ✅): SparseGPT/Wanda one-shot pruning to 2:4 + FP8 quantization; runs on vLLM's compressed-tensors Marlin kernels. ≈2× (sparse format) × 2× (FP8 vs BF16) on the weight footprint, small quality cost.
2. **W4A16 quantization + Marlin/Machete kernels** — 4× smaller, decode speed ≥ FP16 (bandwidth-bound) ✅ Round-1/2 context.
3. **Quantized weights + residual entropy coding** — the ISCA'26 paper *measured* AWQ/int4 weights and still found 2–10× entropy headroom, then recovered part of it with ANS (batch 47→75 even at those sizes). **This layer is not yet in any consumer engine ⇒ buildable whitespace.**
4. **Pruning → light repair → quantization** — Minitron-style recipe (arXiv:2408.11796 ◐: prune depth+width, distill-heal with <1% of pretraining compute, then PTQ) is the industry pattern; the *cheap* variant is ShortGPT/SliceGPT pruning + DeltaLLM-style 30–40M-token healing + GPTQ/AWQ.
5. **Global codebook + outlier sidecar + (optionally) entropy-coded indices** — LegoLM's p99 sidecar is SpQR's trick generalized to sharing; AQLM/VPTQ already entropy-pack indices partially; a full ANS pass over indices is unexplored ⚠ (research opportunity).
6. **Low-rank + quantized factors** — SVD-LLM factors can themselves be quantized (2 bytes → <1 per element); nobody ships this yet ⚠.

**Budget sketch (7B-class, quality-anchored):** BF16 14 GB → W4A16 AWQ ~4 GB (4×) → ANS/Huffman on top ~3.1–3.4 GB (est. 1.2–1.3× more, lossless) → + 2:4 sparsity where kernels allow ~2 GB-class. **Realistic total 5–7× with <1–3% quality loss**, vs DFloat11-only lossless path 14→9.8 GB (1.45×). Every step except "ANS over quantized weights in a consumer engine" is off-the-shelf.

---

## 4. Inference-time decompression — what does staying compressed actually cost? (Q4)

| Runtime | Decompression point | Measured cost |
|---|---|---|
| **Marlin/Machete** (vLLM W4A16/W8A16, 2:4) ◐✅ | fused dequant inside GEMM, near-ideal memory pipelining | decode **≥ FP16 speed** (bandwidth-bound); prefill ~parity at batch≤16 |
| **DFloat11 CUDA kernels** ✅ | Huffman decode fused into GEMV per layer | decode faster-or-neutral (fewer bytes moved); prefill slightly slower; kernels 20–40% faster in v0.2.0 |
| **ANS-tile decompression (ISCA'26)** ✅ | ANS decode aligned to GEMM tiles, on-the-fly | **throughput +20–60%** in SGLang (bigger batches fit) — decompression is *free* by design |
| **llama.cpp i-quants** ✅ Round-1 | codebook/lattice decode per block in CUDA kernels | tg ≈ **−5%** vs k-quants (Unsloth table: 85.7 vs 90.4 tok/s); pp ≈ +3% |
| **ExLlamaV3 EXL3 (QTIP trellis)** ✅ Round-2 | trellis decode in custom kernels | "runs about as fast as anything" (community); ~20% smaller than GGUF at equal quality |
| **NeuZip / BitStack streaming** ✅ | per-layer decompress from compressed resident copy | slower than resident (decompress compute + possible host traffic); enables *fitting*, not *speed* |
| **DeepSparse (CPU, 90–95% sparse)** ◐ | sparse-aware kernels skip zero compute | practical CPU speeds at extreme sparsity; GPU sparse kernels mostly limited to 2:4 |
| **Apple "LLM in a flash"** (arXiv:2312.11514) ◐ | windowing + row-col bundling from SSD | runs >RAM models from flash at usable tok/s; no public code |

**Rule of thumb:** if decompression is *fused into the GEMM/GEMV data path* (Marlin, DFloat11, ANS-tiles, i-quants), the cost is ≈0–10% and often *negative* (fewer bytes over the bus). If decompression is a *separate pass or streams from host/flash* (NeuZip, BitStack, offload modes), you pay real time — acceptable only to fit otherwise-impossible models.

---

## 5. Practicality matrix (the requested checklist)

| Method | Existing pretrained model? | Retraining? | Calibration/fit cost | Memory saved | Quality | Speed | Working code | Indie-reproducible? |
|---|---|---|---|---|---|---|---|---|
| AWQ/GPTQ W4 + Marlin (llm-compressor) | ✅ | ❌ | ~1 GPU-hr, 512 calib samples | ~4× | ≈lossless ≥7B | ≥FP16 decode | ✅ `vllm-project/llm-compressor` | ✅✅ easiest |
| 2:4 sparsity (+FP8) SparseGPT/Wanda | ✅ | ❌ | one-shot, minutes–hours | 2× (×2 with FP8) | small | kernel-dependent (vLLM ok) | ✅ llm-compressor recipes | ✅ |
| **DFloat11 lossless** | ✅ | ❌ | minutes (Huffman fit per model) | 1.45× | **bit-identical** | decode ≥; prefill −slight | ✅ pip + HF org | ✅✅ |
| **ANS Shannon-bound (ISCA'26)** | ✅ (any format incl. int4/AWQ) | ❌ | entropy fit | up to theoretical 2–10× | bit-identical | +20–60% tput (SGLang) | ⚠ not public yet | ⚠ paper fully specifies; rebuild = real project |
| ZipNN lossless | ✅ | ❌ | trivial | 1.3–1.5× (storage) | bit-identical | storage-only (slow codec) | ✅ `zipnn/zipnn` | ✅ |
| VPTQ / AQLM / QTIP(EXL3) 2–3-bit VQ | ✅ | ❌ | hours, GPU + calib data | 5–8× | near-lossless at ≥30B; degrades <14B | vLLM/EXL3 kernels ok | ✅ `microsoft/VPTQ`, vLLM AQLM, turboderp EXL3 | ✅ |
| **LegoLM global codebook + outlier sidecar** | ✅ | ❌ (data-free!) | k-means fit, minutes–hours | **4.4–5.1× @ +0.03% PPL (7B)** | near-lossless claim | dequant = lookup (i-quant-class est.) | ⚠ no official code; fully specified | ✅ reimplementable |
| **DeltaLLM cross-layer share + LoRA deltas** | ✅ | **light** (~30–40M tok, single-GPU hrs) | delta training | 12–24% params | ~90% perf retained | fewer bytes ⇒ decode neutral/+ | ⚠ models released (DeltaLLAMA/DeltaPHI); code repo not located | ◐ moderate |
| ShortGPT / LaCo / SliceGPT layer pruning | ✅ | ❌ (repair optional) | BI scoring, minutes | 10–25% | grows with rate; repair helps | + (less depth) | ✅ ShortGPT repo, `microsoft/SliceGPT`, PrunaAI | ✅ |
| SVD-LLM / ASVD low-rank | ✅ | ❌ (data-free/activation stats) | SVD per layer, minutes | 10–30% typical | moderate | low-rank GEMMs can help decode | ✅ `AIoT-MLSys-Lab/SVD-LLM` | ✅ |
| NeuZip / BitStack compressed-resident | ✅ | ❌ | block decomposition | arbitrary (dial) | dial | slower (streaming) | ✅ both repos | ✅ |
| CompactifAI tensor networks | ✅ | optional heal | TN decomposition | extreme (claims 96–99%) | size-dependent, significant | decompose-on-load/GEMM changes | ⚠ thin community code | ⚠ harder |
| EfficientQAT / LLM-QAT | ✅ | **moderate** (blockwise QAT) | 8×A100-class days (Tier1/2) | 4× at W4A8/A4 | better than PTQ at low bit | kernels exist | ✅ `OpenGVLab/EfficientQAT` | ◐ needs real GPUs |
| Minitron/Sheared-Llama prune+distill | ✅ | **heavy** (B-token distill) | cluster-scale | 2–40× | good (distilled) | smaller=faster | partial | ❌ not indie |

---

## A. Five strongest existing approaches (buildable today)

1. **W4A16 activation-aware PTQ with fused kernels** (AWQ/GPTQ via `vllm-project/llm-compressor`, Marlin/Machete runtime; GGUF Q4_K/EXL2 equivalents) — 4×, ≈lossless ≥7B, decode ≥FP16. The proven floor everything else stacks on.
2. **DFloat11 lossless entropy compression** — 30% smaller with *bit-identical* outputs, pip-installable, HF-hosted models, fused CUDA kernels (NeurIPS'25). The strongest no-risk win available today.
3. **Vector-quantization codebooks (VPTQ / AQLM / QTIP→EXL3)** — the "dictionary" answer with production kernels: 5–8× at 2–3 bpw; near-lossless for ≥30B, use ≥3 bpw below that.
4. **2:4 structured sparsity (+FP8/int stacking) via llm-compressor** — one-shot SparseGPT/Wanda, documented vLLM recipes, real 2×(×2) memory with kernel support.
5. **Layer-redundancy pruning with light healing** (ShortGPT BI / SliceGPT / LaCo, optionally DeltaLLM-style 30–40M-token delta repair) — 10–25% off the top, single-GPU budget, PrunaAI packages several.

## B. Five less obvious / experimental approaches worth investigating

1. **Shannon-bound ANS weight compression** (arXiv:2606.15789, ISCA 2026) — entropy study says *quantized* weights still hide 2–10× redundancy; tile-aligned ANS decode fused with GEMM made compression throughput-*positive* in SGLang (up to 11× vs DFloat11/NeuZip). No public code yet ⇒ the single highest-value artifact to watch/rebuild.
2. **LegoLM global codebook + p99 outlier sidecar + boundary-layer protection** (arXiv:2608.08652) — data-free 4.4–5.1× at +0.03% PPL (Mistral-7B), beating PTQ-8bit on both axes; no official code but fully specified — a weekend reimplementation for an indie dev.
3. **Cross-layer sharing with low-rank deltas** — DeltaLLM (12–24% off, 30–40M-token healing, models released), Basis Sharing (arXiv:2410.03765), cross-layer attention sharing (TACL 2026), MoBE for MoE experts (arXiv:2606.03465). The "store the basis, transmit the diff" family.
4. **Compressed-resident execution engines** — NeuZip (layer-wise on-the-fly decompression, official repo) and BitStack (megabyte-granularity size dial): run models that don't "fit," trading speed for footprint; also the natural host for lossless codecs on tiny GPUs.
5. **Tensor-network decomposition (CompactifAI)** — quantum-inspired multilinear compression claiming 96–99% parameter cuts with tunable accuracy; radical, commercially deployed, academically under-replicated ⇒ high-risk/high-insight.

## C. The most promising combination (concrete pipeline)

**"Prune → Share → Quantize → Entropy-code", with outlier sidecars at every lossy step:**

1. **Depth/width trim (optional, one-shot):** ShortGPT/SliceGPT at a conservative 10–15% + DeltaLLM-style delta healing (~30–40M tokens, single GPU overnight) → *~1.15–1.3×, quality restored to ~baseline*.
2. **Structure pass:** 2:4 sparsity where kernels exist (llm-compressor), and/or ASVD/SVD-LLM low-rank on non-sparse-friendly matrices → *~1.2–2× more*.
3. **Value pass:** activation-aware scalar/vector quantization to 4 bpw (AWQ/VPTQ) with **Qronos-style error-diffusion rounding** and **SpQR/LegoLM-style p99 outlier sidecar in high precision** → *~4×, near-lossless*.
4. **Residual entropy pass (lossless, stacks on everything):** DFloat11-class Huffman today; **ANS tile-fused decode (ISCA'26 pattern) as the upgrade** — the entropy study proves steps 2–3 leave 2–10× additional *lossless* redundancy on the table → *~1.2–1.5× more (est.), zero quality cost*.
5. **Runtime:** everything decompresses *inside* the GEMM/GEMV data path (Marlin-class fused dequant; ANS tiles), so speed is neutral-to-positive.

Arithmetic: 1.2 × 1.5 × 4 × 1.3 ≈ **~9× total** vs BF16 at single-digit-% quality cost — versus ~4× from quantization alone today. Steps 1–3 and 5 exist as code now; step 4's ANS tier is the whitespace.

## D. Genuinely new directions (not "another quantization method")

1. **Entropy coding as a universal second stage.** The ISCA'26 result reframes compression: *quantization decides what information to keep; entropy coding decides what the kept information costs to store.* Because even int4/AWQ weights sit 2–10× above their entropy, a general "codec layer" under any format (BF16, FP8, int4, codebook indices, sparsity bitmaps) is a new axis — and tile-fused ANS makes it speed-*positive*. Nobody ships this in consumer engines yet.
2. **Self-LoRA: the model as basis + adapters.** DeltaLLM/LaCo/Basis-Sharing/MoBE converge on representing a pretrained net as *shared structures + small trained corrections*. Generalized: automatically discover sharing (across layers, experts, even heads), then heal with 10⁷-token LoRA-class fits. This is "fine-tuning as compression" — a middle road between zero-training PTQ and full distillation that current toolkits don't occupy.
3. **Outlier-aware hybrid representations as a first-class format.** LegoLM (+1,134,279% → +15% purely by protecting ~1% of weights) and SpQR both show the distribution's tail carries the function. A standard container format of {codebook/quantized bulk + sparse verbatim tail + entropy-coded indices} — with kernel support — would beat any uniform scheme at every budget.
4. **Cross-model deduplication via superposition/task vectors.** Classic result: many models can coexist in one parameter set (Cheung et al., "Superposition of many models into one" ✅); the 2025–26 revival (local task vectors, thin task-delta arithmetic — indexed in Awesome-LLM-Compression ✅, specific IDs not re-verified ⚠) stores *one base model + ~1% deltas per task/finetune*. For anyone hosting many finetunes, this compresses the *fleet*, not the model — orthogonal to everything above.
5. **Compressed-resident engines.** NeuZip/BitStack/DFloat11-offload prove the engine can treat "decompressed weights" as a transient, per-layer artifact rather than a resident state. Combined with direction 1, the endgame is: *VRAM requirement = compressed size; decode speed = compressed-stream bandwidth* — i.e., smaller models become faster models, dissolving the size/speed trade-off instead of negotiating it.

---

## Link index (verified ✅ this round unless marked)
Papers: 2606.15789 (ANS/Shannon, ISCA'26) · 2608.08652 (LegoLM) · 2501.18596 (DeltaLLM) · 2410.03765 (Basis Sharing) · 2606.03465 (MoBE) · TACL.a.616 (cross-layer attention sharing) · 2403.03853 (ShortGPT) · 2401.15024 (SliceGPT ◐) · 2403.17887 (deeper layers ◐) · 2407.05102 (LaCo ◐) · 2503.12340 (SVD-LLM V2) · 2504.11651 (DFloat11) · 2410.20650 (NeuZip) · 2410.23918 (BitStack) · 2411.05239 (ZipNN) · 2505.11695 (Qronos) · 2409.17066 (VPTQ) · 2401.06118 (AQLM ◐) · 2404.09936 (QTIP ◐) · 2306.03078 (SpQR ◐) · 2301.00774 (SparseGPT ◐) · 2306.11695 (Wanda ◐) · 2408.11796 (Minitron ◐) · 2310.06694 (Sheared Llama ◐) · 2401.14109 (CompactifAI) · 2407.11062 (EfficientQAT) · 2506.12040 (BTC-LLM) · 2312.11514 (LLM-in-a-flash ◐).
Repos: `LeanModels/DFloat11` · `borealisai/neuzip` · `xinghaow99/BitStack` · `zipnn/zipnn` · `microsoft/VPTQ` · `vllm-project/llm-compressor` (+ vLLM 2:4/FP8 recipe docs) · `AIoT-MLSys-Lab/SVD-LLM` · `OpenGVLab/EfficientQAT` · `HuangOwen/Awesome-LLM-Compression` (living index) · `PrunaAI/pruna` (toolkit incl. ShortGPT track) · `daniel-eai/LoRP-...` · `turboderp-org/exllamav3` (QTIP-based EXL3) · `IST-DASLab/marlin` ◐ · `microsoft/SliceGPT` ◐.
