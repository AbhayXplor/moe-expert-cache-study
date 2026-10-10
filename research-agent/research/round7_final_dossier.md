# Round 7 FINAL DOSSIER — Compressed-Resident LLM Weights: Code-Level Audit & Build Decision
### Evidence tags: [P]=PROVEN (public code inspected/reproducible) · [C]=CLAIMED (repo/paper claim, not independently verified) · [S]=SPECULATIVE
### All repo inspections performed 2026-10-07 on actual source files, not abstracts.

---

## 1. BEST EXISTING REPOS (Task 1, GitHub-first)

### 1.1 LeanModels/DFloat11 — lossless Huffman, compressed-resident, INSPECTED
- GitHub: https://github.com/LeanModels/DFloat11 · Paper: arXiv:2504.11651 (NeurIPS'25) · Last meaningful update: Nov 2025 ("Add CUDA kernel"); 12 commits total
- Models: HF org `DFloat11` (Llama-3.x incl. Qwen3/Gemma3/Phi-4 per README, FLUX.1, Wan2.1) · Runtime: own (PyTorch hooks + CuPy-loaded PTX), CUDA 12, `pip install dfloat11[cuda12]`
- Method [P, code-verified]: per-tensor Huffman coding of BF16 **exponent bytes**; sign+mantissa (8 bits) stored raw. Resident buffers: `encoded_exponent` (bitstream), `sign_mantissa`, `luts` (multi-level decode LUT), `output_positions`, `gaps` (per-thread entry offsets into the bitstream — this is the random-access mechanism).
- **Compressed stays in VRAM? YES** [P]: `dfloat11.py` keeps those buffers as module buffers on-device; `offloaded_tensor_names=('encoded_exponent','sign_mantissa')` only under `cpu_offload=True` (R3 mode, avoid).
- **Full tensor reconstructed before GEMM? YES — per layer, into a REUSED scratch** [P]: `get_hook()` returns a forward **pre-hook** `decode_hook` that launches `decode.cu` (warp-cooperative bitstream decode: register buffer → 4-level LUT lookup → block prefix-sum via shared-memory `accumulators` → write BF16) into a **single shared BF16 scratch** from `TensorManager.allocate_bfloat16()` (sized to the largest layer, reused every layer), then assigns `module.weight = reconstructed.view(...)` and calls stock `F.linear`.
- **Fused with matmul? NO** [P] — decompress-store-compute. HBM traffic per decode step ≈ 11 bits read + 16 bits write + 16 bits read ≈ 2.7× a plain BF16 GEMV's weight traffic. The ISCA'26 paper explicitly criticizes this pattern ("ZipNN decompresses an entire layer before launching its GEMM... layer-level synchronization barrier") and reports up to 11× throughput vs DFloat11/NeuZip *in batched serving* [C, their benchmark].
- VRAM (derived from code, to be measured in E2): resident ≈ ~11 bits/weight + one largest-layer BF16 scratch (Llama-8B: ≈10.3 GB + ~120 MB vs 15 GB BF16) [P-structure/C-number].
- Speed: "decode neutral-to-faster, prefill slower" [C — README/paper; mechanism analysis above predicts decode CANNOT beat BF16 on pure bandwidth; plausible explanation of their wins = load time, batch headroom, and disk. Measure before believing.]
- Quality: bit-identical by construction [P — lossless codec + greedy-identity checkable].
- Run today? YES (pip + HF models). Install difficulty: LOW. Main limitation: BF16-only codec; unfused decode; own minimal runtime.
- **VERDICT: R1-resident, decompress-store-compute. Best lossless reference implementation + easiest codebase to hack (one 9.5KB .cu, one 28KB .py).**

### 1.2 xinghaow99/BitStack — SVD-block "any-size" compression with FUSED Triton kernels, INSPECTED README
- GitHub: https://github.com/xinghaow99/BitStack · Paper: arXiv:2410.23918, **ICLR 2025** · Updates: Triton fused kernels Nov 2024 ("3–10× speedup over original implementation", `--fused_level`), Mistral/Qwen support Jan 2025
- Models [P]: **pre-decomposed checkpoints on HF `BitStack/`**: Llama-2 7B/13B/70B, Llama-3(.1) 8B/70B(+Instruct), Llama-3.2 1B/3B, Mistral-7B-v0.3, Qwen2.5 0.5B→72B. Colab demo with memory slider.
- Method [P-README]: iterative SVD-style decomposition into small blocks (`--niter 16 --k 16` singular vectors), importance-scored, sorted, "stacked" so a `--max_memory_MB` budget selects how much stays resident; fused Triton kernels compute from the decomposed representation.
- Resident compressed? YES within budget (fused inference from factors) [P-kernels-exist]; beyond budget it transfers from CPU/disk → **R1/R3 hybrid with an explicit megabyte dial** — the only repo that makes the residency tradeoff a first-class runtime parameter.
- Quality [C-paper]: "maintains or surpassing GPTQ/AWQ" at equal budgets; PPL + lm-eval harnesses included (`--do_eval --lm_eval`) [P-scripts].
- Run today? YES (conda + `pip install -e .`; decomposition needs calibration samples, pre-decomposed models skip it). Install: LOW-MED. Limitation: research runtime (HF-style, not vLLM); fused kernel maturity unclear [C].
- **VERDICT: the closest existing thing to "sharing/redundancy + compressed-resident execution" with real models and fused kernels. Prime candidate for the sharing layer (Task 5's only shipped runtime).**

### 1.3 HanGuo97/flute (FLUTE) — fused LUT-GEMM engine for non-uniform quantization
- GitHub: https://github.com/hanguo97/flute · Paper: arXiv:2402.10896-line (FLUTE, MLSys-track) · Coverage: inference.net (Mar 2026): "2–4× faster GEMM kernels, 1.5–2× end-to-end throughput with lookup-table quantization" [C]
- Method [P-repo-exists]: offline preprocessing of LUT-quantized weights (AWQ/GPTQ-style non-uniform + outlier vectors) into a GPU-friendly layout; CUDA kernels perform **table lookup inside the matrix multiplication** (no full dequant materialization) — i.e., true fused decode+GEMM for codebook formats.
- Resident compressed? YES (packed codes + LUTs) [P-design]; reconstructs full tensor? NO — lookup happens in-kernel [P-design]; details/维护状态 need verification ⚠ (check last commits + Ada support before adopting).
- **VERDICT: the kernel abstraction our target stack needs — a codebook/LUT GEMM that could host entropy-decoded indices and shared codebooks. Candidate BASE for the novel build (Task 3/8).**

### 1.4 SqueezeAILab/SqueezeLLM — historical full-stack (quant+sidecar+LUT-GEMV kernels)
- GitHub: https://github.com/SqueezeAILab/SqueezeLLM · ICML 2024 · **dormant since Mar 2024** [P], was merged into vLLM then removed (absent from current format list; RFC #30136) [P-round5]
- Method [P-paper+repo-structure]: k-means non-uniform 3/4-bit + **dense-and-sparse** (0.05–0.45% outliers in CSR-style sidecar) + packing code (`squeezellm/` package, "Dense-and-sparse Quantization packing code" PR #59) + **"3/4-bit CUDA LUT-based kernels for matrix-vector multiplication"** [P-paper-text] — decode fused into GEMV for the decode phase.
- Models [P]: HF `squeeze-ai-lab/sq-{llama,llama-2,mistral,vicuna,opt}-{7b..65b}-w{3,4}-s{0,5,45}` (2023-era).
- Resident compressed? YES (packed codes + sparse sidecar) [P]; full reconstruction? NO for GEMV path (LUT in-kernel) [P-paper].
- Run today? Technically yes on old stacks (CUDA 11-era build, python 3.9); practically: read-only reference. Install: HIGH (bit-rot).
- **VERDICT: the blueprint repo — proves quant+entropy-ish-packing+sidecar+fused-kernel shipped and served real 7B–65B models in 6 GB-class VRAM [C-their-claim]. Mine `squeezellm/` for kernel + packing patterns; do not deploy.**

### 1.5 caiovicentino/eoq-quantization — INSPECTED; critical path has a TODO
- GitHub: https://github.com/caiovicentino/eoq-quantization · 30 commits, Mar–Sep 2026 · solo dev, LLM-co-authored ⚠
- Method [P-code]: absmax/PolarQuant quantization + **rANS at rest** (`.eoq` format); `core/eoq_linear.py` docstring: "Disk: .eoq rANS (~287MB Q4) / RAM: INT4 packed (~350MB) / Compute: dequantize on-the-fly during matmul". **BUT `load_eoq_to_quantized_model()` contains: `# TODO: Load .eoq data directly into QuantizedLinear buffers. For now, this just quantizes from the pretrained weights`** [P] — the direct compressed→resident path is NOT implemented in this file; the working path loads FP32 from HF then quantizes+packs (transient full-precision at load).
- Real assets [P]: `benchmarks/memory_profiler.py` (42KB), `speed_benchmark.py`, `measure_perplexity.py`, `core/bit_packing.py`, `core/fast_dequant.py`, `kernels/` (16 CUDA variants per commit msg [C]), llama.cpp integration patches + C rANS v2 (114 MB/s [C]), 10 HF models `caiovicentino1/*-EOQ-*` [P-API-verified R6].
- Resident? INT4-packed (QuantizedLinear) — R1 at the int4 level, i.e., **same VRAM class as GPTQ**; rANS layer = R2 (disk/load only).
- Claims audit: "Q5 11–41% smaller than GGUF Q4_K_M, PPL<1.0 delta, zero inference overhead/identical tok/s to FP16" [C — the tok/s claim is inconsistent with their own commit log ("GPU dequant 3.5s, 45.8 tok/s") and with any dequant-in-PyTorch path; treat all speed claims as unverified until re-run].
- **VERDICT: useful scaffolding (rANS codec, bit-packing, benchmark harnesses) with an unfinished critical path. Fork-and-audit candidate, NOT a base.**

### 1.6 BorealisAI/neuzip — INSPECTED README; pip-installable dynamic compression
- GitHub: https://github.com/borealisai/neuzip · Paper arXiv:2410.20650 · `pip install neuzip` (source dist, needs NVCC) [P]
- API [P]: `manager = neuzip.Manager(); model = manager.convert(model)` — converts any PyTorch model to dynamically-compressed form; "used the same way while consuming less memory."
- Method: exponent/mantissa split + entropy coding, **layer-wise on-the-fly decompression during forward** (lossless training variant; lossy inference variant trimming mantissa bits) [P-paper].
- Resident compressed? YES; reconstruction? per-layer, transient (same decompress-store-compute family as DFloat11, finer-grained) [P-design]; fused? NO.
- **VERDICT: R1-resident research runtime; the cleanest open example of "compressed-resident with per-layer reconstruction" plumbing (module-conversion pattern worth stealing). Speed cost expected (unfused + ISCA critique applies).**

### 1.7 drxddy/ecq — measurement code only
- [P] `entropy_analysis.py`, `rans_codec.py`, `validate_compression.py` + real-model nibble-entropy numbers (Qwen2.5-0.5B/1.5B, GPT-2, SmolLM: 1.12–1.54 bits → 3.27–3.58× over 4-bit). **No inference kernel** — Metal kernel is a design doc (`docs/metal_kernel_design.md`); the tok/s table (42.9→140) is a projection [P-README-structure]. Small models + naive quant recipe (max skew). **VERDICT: borrow the analysis code; ignore the speed table.**

### 1.8 Production quantization toolchain (Layer-2 baseline, no deep inspection needed — verified across rounds)
- `vllm-project/llm-compressor` + `neuralmagic/compressed-tensors` [P]: W4A16/W8A8/FP8/**2:4+FP8 and 2:4+W4A16 official recipes** (docs.vllm.ai examples) → resident packed formats, Marlin/Machete fused dequant-GEMM in vLLM = **R1, production**.
- `ModelCloud/GPTQModel` [P]: maintained GPTQ/AWQ quantizer+runtime (successor ecosystem to AutoGPTQ).
- `intel/auto-round` [P]: better-rounding PTQ exporting to the above formats.
- `microsoft/VPTQ` [P]: VQ codebooks + CUDA kernels + vLLM integration (second-tier support; R1).
- llama.cpp GGUF k/i-quants & ExLlamaV3 EXL3 [P-rounds-1/2]: the consumer-engine R1 incumbents (i-quants = lattice/codebook-resident with in-kernel decode, ~5% tg cost).

---

## 2. WHAT EACH ACTUALLY DOES + 3. WHICH REALLY REDUCE INFERENCE VRAM

| Repo | Resident representation (from code) | Class | VRAM during inference (weights) | Fused decode? | Real-model evidence |
|---|---|---|---|---|---|
| DFloat11 | Huffman exponent stream + sign/mantissa bytes + LUTs; **one-layer BF16 scratch** | **R1** | ≈11 bits/w + largest-layer scratch | ✗ (pre-hook, store-compute) | HF models, pip [P] |
| BitStack | sorted SVD blocks within `--max_memory_MB`; rest streams | **R1/R3 dial** | user-set budget | ✓ Triton `--fused_level` [P-exists/C-perf] | 12+ HF checkpoints incl. 70B [P] |
| FLUTE | packed non-uniform codes + LUTs + outlier vecs | **R1** | ~4-bit class + tables | ✓ in-GEMM lookup [P-design/C-perf] | AWQ/GPTQ models [P] |
| SqueezeLLM | packed 3/4-bit codes + CSR sparse sidecar | **R1** | 3–4 bits/w + sidecar | ✓ LUT-GEMV [P-paper] | 7B–65B HF models [P], 2023 |
| EOQ | INT4-packed (QuantizedLinear); rANS on disk only (direct path TODO) | **R1(int4)+R2(rANS)** | ≈GPTQ-class | partial (dequant-on-matmul; kernel maturity ⚠) | 10 HF models [P], benchmarks [C] |
| NeuZip | compressed exponent/mantissa per module; per-layer transient decompress | **R1** | compressed + layer scratch | ✗ | pip, paper experiments [P/C] |
| ECQ | none (no runtime) | — | — | ✗ (design doc) | analysis numbers [P] |
| llm-compressor→vLLM (Marlin) | packed int4/int8/FP8/2:4 | **R1** | 4–8 bits/w (+2:4 ≈ half of that on values) | ✓ | production [P] |
| VPTQ | VQ codebooks + packed indices | **R1** | ~2-bit class + tables | ✓ kernels [P] | HF models [P] |
| llama.cpp i-quants / EXL3 | lattice/trellis codes resident | **R1** | 2–4.5 bits/w | ✓ | massive community evidence [P] |
| ZipNN / QStore | — (storage formats) | **R2** | **NO VRAM WIN** | n/a | [P] |
| NeuZip-offload / DFloat11-offload / llama.cpp -ngl<all | — | **R3** | reduced w/ host traffic | n/a | [P] |

**Answer to the core question:** genuine compressed-resident inference is REAL and shipping in three families: (a) packed-quant + fused dequant-GEMM (Marlin/llama.cpp/EXL3/FLUTE/SqueezeLLM/VPTQ) [P]; (b) lossless-codec resident with unfused per-layer decode (DFloat11/NeuZip) [P, with speed caveat]; (c) factorized-resident with a memory dial (BitStack) [P]. What does NOT exist: entropy-coded streams decoded *inside* the GEMM (ISCA'26 = private; ECQ = unbuilt; EOQ = TODO) [P-absence across all inspections].

---

## 4. BEST REPO TO BUILD ON (Task 3 ranking)

Ranked on: real VRAM reduction · speed · quality · code availability · modifiability · combinability.

1. **DFloat11** — VRAM ✓(R1, ~1.45×); speed: unfused (the weakness = the experiment); quality: lossless ✓; code: tiny & clean ✓✓ (2 files + 1 kernel); modifiability: **best in class** (hook-based, PTX-loaded kernel, no engine fork); combinability: codec is format-agnostic in principle (exponent Huffman generalizes to any symbol stream).
2. **FLUTE** — VRAM ✓(R1 fused); speed: fused LUT-GEMM (2–4× GEMM claims [C]); quality: hosts AWQ/GPTQ-class quant; code: real CUDA kernels; modifiability: medium (CUDA/C++); combinability: **the LUT-GEMM abstraction is exactly where entropy-decoded and shared-codebook indices belong**.
3. **BitStack** — VRAM ✓✓(only repo with an explicit MB budget); speed: fused Triton (3–10× vs their baseline [C]); quality: ICLR'25 paper + lm-eval scripts; code: research-grade but complete with 12+ real checkpoints; combinability: its factorized blocks could carry quantized factors (nobody does).

**BEST STARTING REPO: DFloat11** — because the first experiments are *measurement* experiments (is compressed-resident decode actually speed-neutral? what does an entropy layer cost when fused vs unfused?), and DFloat11 is the smallest complete R1 codebase where every byte of the resident representation and every kernel launch is visible in ~900 lines. **FLUTE is the phase-2 base** (when a fused kernel for a new format is needed). BitStack is the sharing-layer testbed. (If the goal were shipping today instead of researching: llm-compressor+vLLM.)

---

## 5. BEST CURRENT METHOD (Task: what to run TODAY for max VRAM savings at high quality/speed)
**W4A16 (AWQ/GPTQ/AutoRound) or 2:4+FP8 via llm-compressor → vLLM Marlin** [P]: 4–8× VRAM cut, ≈FP16 decode speed, small characterized quality cost, official recipes, zero novel code. Add DFloat11 only if bit-exactness matters more than VRAM (1.45× lossless). This is the proven floor our research stack must beat.

## 6. BEST STACK WE CAN ACTUALLY TEST (Tasks 4+9 combined)
`BitStack/LegoLM-class redundancy layer → W4A16-class quantization (llm-compressor) → entropy layer ONLY where measured headroom exists (stageB probe: FP8-E5M2 ~1.4× [P-community-measured]; absmax-int4 2.9–3.6× [P-ECQ-small-models/C-generalization]; group-wise AWQ-int4 1.1–1.3× [P-ISCA]) → compressed-resident runtime (DFloat11-pattern hooks for validation; FLUTE/SqueezeLLM-pattern fused kernel for the final artifact)`.
Combination status: quant+sparsity [P-shipping]; quant+entropy [P-partial: EntroLLM edge-hardware, SqueezeLLM dormant, EOQ TODO]; sharing+resident-fused [P-BitStack only, without quantized factors]; sharing+quant [NONE — gap]; entropy-decode fused in GEMM [NONE public — the gap].

## 7. WHAT HAS ALREADY BEEN DONE (PROVEN)
- Packed-quant compressed-resident inference with fused dequant (Marlin, llama.cpp quants, EXL3, VPTQ, FLUTE, SqueezeLLM-in-2023) — the entire "Method B" of your problem statement, for quantization.
- Lossless compressed-resident (DFloat11, NeuZip) — real VRAM cut, unfused decode.
- Sparsity×quantization stacking with kernels (llm-compressor 2:4 recipes).
- Factorized-resident with MB budget + fused Triton (BitStack, real 70B checkpoints).
- Quant+entropy with real speed wins on edge hardware (EntroLLM, Jetson, +32–147% [C-paper, peer-reviewed]).
- Entropy headroom measurements on real quantized models (ISCA per-format table [P-paper]; ECQ small-model nibble entropy [P-repo]; FP8-E5M2 1.43× [P-issue-thread measurement]).

## 8. WHAT HAS NOT BEEN DONE (verified absence, not quick-search absence)
1. **Entropy-decoding INSIDE the GEMM** for LLM weights, in public code (ISCA'26 tile-ANS: private; ECQ: design doc; EOQ: TODO; SqueezeLLM: fixed-width packing, not variable-length). ← the kernel gap
2. **Cross-layer shared codebooks + quantization + fused runtime** (LegoLM/DeltaLLM/Basis-Sharing: no code, no models on HF [P-verified R5/R6]; BitStack shares nothing across layers; VPTQ codebooks are per-tensor).
3. **Entropy layer on group-wise-quantized production formats with a fused kernel** (headroom exists per ISCA 1.1–1.3×; nobody ships it).
4. **A measured study of sharing×quantization error interaction** (shared-base error reuse) — zero results anywhere.

## 9. THE REAL GAP (Task 7)
> A **unified compressed-resident format** `{cross-layer shared codebook + per-tensor packed indices + entropy-coded index/scale streams + verbatim outlier sidecar}` **with one fused decode-in-GEMM kernel**, benchmarked against Marlin. Every component exists separately in public code (SqueezeLLM sidecar+LUT-GEMV, FLUTE fused LUT-GEMM, DFloat11 Huffman streams+random-access `gaps` mechanism, BitStack budget loader, LegoLM recipe on paper). Nobody has assembled them. That assembly is a 4–8 week engineering project, not a research breakthrough — and each stage has a public blueprint.

## 10. EXACT FIRST EXPERIMENT (Task 9 — kill-fast ordering)
**E1 (½ day):** Baseline: vLLM + `llm-compressor` W4A16 on Llama-3.1-8B → record peak VRAM (torch/nvidia-smi), tok/s (fixed prompt, batch 1 + 8), wikitext PPL.
**E2 (½ day):** DFloat11 `DFloat11/Llama-3.1-8B*` via `prototype/stageA_dfloat11_validate.py` → same 3 metrics + greedy-identity vs BF16. **Kill/learn gate: if decode tok/s < 0.7× BF16, the unfused pattern is confirmed as the bottleneck ⇒ fusion (FLUTE path) is mandatory, not optional.**
**E3 (½ day, CPU):** `prototype/stageB_entropy_probe.py` on: E1's AWQ output, an FP8-E5M2 conversion, an absmax-int4 conversion → pick the entropy-layer target format by measured headroom (gate ≥1.15×).
**E4 (1 day):** BitStack fused inference on `BitStack/BitStack-Llama-3.1-8B` with `--max_memory_MB` ∈ {4000, 5000, 6000} + `--fused_level` on → PPL-vs-MB curve + tok/s. This is the only off-the-shelf measurement of "factorized compressed-resident" quality/speed that exists.
**E5 (1 week):** `prototype/stageC_legolm.py --measure` + full replication on Llama-3.1-8B → PPL gate (≤1% at ≥3×) + feed indices to stageB probe (Layer-3-on-shared-codes question). Kill LegoLM track if gate fails.
**E6 (2–4 weeks, only if E2/E3 gates pass):** Fork **DFloat11**, replace the exponent-Huffman payload with E3's winning quantized format (e.g., FP8-E5M2 or absmax-int4 codes), keeping its `gaps`-based random access + scratch-decode; measure. Then port the decode INTO a FLUTE/SqueezeLLM-style LUT-GEMV (fused) and re-measure. **Success = VRAM < Marlin-W4 at equal PPL and tok/s ≥ 0.9× Marlin.**

## 11. EXACT DOWNLOADS
```
git clone https://github.com/LeanModels/DFloat11            # base repo (read dfloat11/dfloat11.py, decode.cu first)
git clone https://github.com/xinghaow99/BitStack            # sharing-layer testbed + pre-decomposed models via HF
git clone https://github.com/hanguo97/flute                 # fused LUT-GEMM reference (phase 2 base)
git clone https://github.com/SqueezeAILab/SqueezeLLM        # kernel/packing blueprints: squeezellm/ package
git clone https://github.com/caiovicentino/eoq-quantization # rANS codec + benchmark harnesses (audit before trusting)
git clone https://github.com/drxddy/ecq                     # entropy_analysis.py / rans_codec.py (measurement code)
git clone https://github.com/BorealisAI/neuzip              # module-conversion plumbing reference
pip install llm-compressor compressed-tensors vllm dfloat11[cuda12] neuzip
hf download BitStack/BitStack-Llama-3.1-8B --local-dir models/bitstack-llama31-8b
# + one W4A16 AWQ checkpoint (e.g. hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4) and its BF16 original
```
Files to read in order: `DFloat11/dfloat11/dfloat11.py` (`get_hook`, `TensorManager`, `load_and_replace_tensors`) → `DFloat11/dfloat11/decode.cu` (LUT decode + `gaps` random access + shared-mem prefix sum) → `SqueezeLLM/squeezellm/` (packing + `cuda_kernel/` LUT-GEMV) → `BitStack/bitstack/main.py` + Triton kernels dir → `flute/` kernel launcher + offline preprocessing → `eoq-quantization/core/bit_packing.py`, `benchmarks/memory_profiler.py`.

## 12. GO / MODIFY / ABANDON VERDICT
- **GO — compressed-resident quantized inference** (already true in production: Marlin/llama.cpp/EXL3/VPTQ). No research risk; adopt as baseline.
- **GO (MODIFY) — the entropy-layer-on-quantized-weights direction**, conditionally: E3 headroom measurement decides the target format (FP8-E5M2 ~1.4× and absmax-class are worth it; group-wise int4 at 1.1–1.3× is marginal and needs the fused kernel to be net-positive). Build path = fork DFloat11 (E6) → port decode into FLUTE-style fused GEMV. Budget: 4–8 weeks. Kill gates pre-registered in E2/E3/E6.
- **GO (SMALL BET) — sharing layer** via BitStack (shipped) + LegoLM replication (stageC); the sharing×quantization interaction is unmeasured by anyone — cheap to measure (E5), expensive to assume.
- **ABANDON — expecting ZipNN/QStore/EOQ-rANS to cut inference VRAM** (R2: disk-only wins; EOQ's direct path is a TODO). **ABANDON — 2–10× lossless headlines** for deployed formats (per-format evidence says 1.1–1.5×; higher numbers belong to naive-quant recipes whose quality you'd lose elsewhere). **ABANDON — DeltaLLM/Basis-Sharing/MoBE as dependencies** (no artifacts; methods only).

### Skepticism ledger additions (this round)
- DFloat11 "efficient/faster inference" [C] vs code-verified decompress-store-compute with ~2.7× weight traffic per decode pass [P-structure] → E2 will settle it; do not quote their speedup without measuring.
- EOQ "zero inference overhead / identical tok/s to FP16" [C] contradicted by its own TODO and commit-log timings [P].
- ECQ "140 tok/s" is a projection; kernel unbuilt [P].
- BitStack "3–10× faster" is vs *their own unfused baseline*, not vs Marlin/GPTQ runtimes [P-README wording].
- FLUTE perf numbers come from a vendor-adjacent article [C]; verify commits/activity and Ada support before adoption.
- EntroLLM's 65%-over-uint4 uses *their* tensor-level quantizer (weaker baseline recipe), not AWQ/GPTQ group-wise [P-paper-text].
