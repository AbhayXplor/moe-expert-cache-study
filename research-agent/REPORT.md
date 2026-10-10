# Eliminating the 8GB VRAM Wall: 12–16B LLMs at 30+ tok/s with Zero System-RAM Fallback

**Target hardware:** NVIDIA RTX 4060 Laptop GPU — 8 GiB (8192 MiB) GDDR6, 128-bit @ 16 Gbps → **256 GB/s peak bandwidth**, Ada Lovelace (sm_89), 3072 CUDA cores.
**Problem statement:** 12–16B models spill layers into Shared System RAM over PCIe → 4–5 tok/s (a >90% throughput loss, independently measured in *Silicon Showdown*, arXiv:2605.00519).
**Goal:** ≥30 tok/s sustained generation at long context, **100% GPU-resident**, no host-RAM dependency.

**Research window:** 2022 → October 2026 (arXiv + GitHub, current as of 2026-10-07).

---

## Executive Summary

Three independent 2024–2026 developments converge to make this solvable on 8 GiB:

1. **Quantization-Aware Training (QAT) went mainstream for open weights.** Google DeepMind released the **Gemma 4 family with official QAT Q4_0 checkpoints** (technical report arXiv:2607.02770; GGUF at `google/gemma-4-12B-it-qat-q4_0-gguf`), preserving near-BF16 quality at ~72% memory reduction. Unsloth's "Dynamic v3 / smart 4-bit recovery" GGUF (`UD-Q4_K_XL`, **6.72 GB / 6405 MiB**) lifts Gemma 4 12B QAT top-1 logit agreement from 74.1% (naive Q4_0 conversion) to **88.8%** while being 260 MB smaller.

2. **Architecture-level KV state management.** Gemma 4 12B uses **hybrid attention (5:1 sliding-window:global)** — 40 of 48 layers keep only a 1024-token window; 8 global layers use **unified K/V + p-RoPE**. Result: the KV cache at **128K context is ≈1.2 GB** (~6× smaller than a full-attention model), and ≈180–350 MiB at 32K with q8_0 KV quantization. Long context stops being the memory killer.

3. **Native Multi-Token Prediction (MTP) speculative decoding, merged into llama.cpp (PR #23398).** Google trained QAT-compatible MTP drafter heads for Gemma 4 (Unsloth GGUFs: **Q4_0 = 242 MiB**, Q8_0 = 444 MiB). Verified lossless (target-verified tokens only), delivering **1.4–2.2× decode speedup** in llama.cpp.

**The physics check (roofline):** decode is memory-bandwidth-bound. Streaming 6.72 GB of weights per token at 256 GB/s peak = **38 tok/s theoretical ceiling** at 100% efficiency; real-world llama.cpp on Ada achieves ~55–80% → **~21–30 tok/s baseline** (community-measured: 21 tok/s on RTX 4060 without MTP; 40 tok/s with QAT 12B on Windows/CUDA). **MTP is therefore the mechanism that clears the 30 tok/s bar with margin**: 1.5–2.2× on a ~22–26 tok/s baseline → **~35–55 tok/s expected**, with only +242 MiB VRAM for the drafter and negligible extra bandwidth per accepted token.

**Final recommendation (Plan A):**

| Item | Choice | Why |
|---|---|---|
| Model | **Gemma 4 12B QAT** (`unsloth/gemma-4-12b-it-qat-GGUF`, `UD-Q4_K_XL`) | Only 12–16B model with official QAT + tiny hybrid KV + MTP drafter; 6405 MiB weights |
| Engine | **llama.cpp `llama-server`** (CUDA, built from source, ≥ b9551-class) | Mature sm_89 kernels, MTP merged, full flag control |
| Spec decode | `-md MTP/mtp-gemma-4-12B-it-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2` | Lossless 1.5–2.2×; drafter = 242 MiB |
| KV cache | `--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0` | −47% KV memory, ~0% decode cost at ≤32K (q4_0 avoided: −37% decode at long ctx) |
| Context | 32 768 default (16K safe / 64K max) | Hybrid attention keeps KV ≤ ~330 MiB |
| RAM-fallback kill | `-ngl 99 --no-mmap --cache-ram 0` + **NVIDIA "Prefer No Sysmem Fallback"** (Windows) | Hard-fail instead of silent PCIe spill |
| Sampling | `--temp 1.0 --top-p 0.95 --top-k 64` (Google's official Gemma 4 QAT settings) | Preserves trained behavior / exact syntax |

**VRAM budget (balanced profile, nvidia-smi MiB):** weights 6405 + MTP 242 + KV@32K q8_0 ≈180–350 + compute buffers ≈200–300 + CUDA context ≈250–350 → **≈7.3–7.6 GiB of 8 GiB**. Zero shared-memory use, verified by the included monitoring script.

Full details: Phase 1 in `research/phase1_arxiv.md`, Phase 2 in `research/phase2_github.md`, runnable artifacts in `deploy/`.

---

# Phase 1 — Academic Research (arXiv, 2022→2026)

## 1.1 Taxonomy of runtime-memory reduction

The footprint of a serving LLM is: **F = W (weights) + KV(ctx) + A (activations/compute buffers) + C (CUDA context/fragmentation)**. Papers attack each term:

### A. Static weight compression (attacks W)

| Paper | arXiv | Core idea | Relevance / trade-off for code+reasoning fidelity |
|---|---|---|---|
| GPTQ | 2210.17323 | Layer-wise PTQ via approximate second-order (Hessian) error compensation | Foundational 4-bit; error accumulates in small dense models at ≤3-bit |
| AWQ (MLSys'24) | 2306.00978 | Protect ~1% salient weight channels identified by **activation** magnitudes; scale-equivalent transformation, no backprop | Best-in-class 4-bit retention of reasoning/coding vs GPTQ; basis of most "activation-aware" pipelines |
| OmniQuant | 2308.13137 | Learnable weight clipping + learnable equivalent transformation; enables W4A4/W3A16 | Shows PTQ can be trained cheaply; W4A4 still risky for exact syntax |
| OmniQuant successors / rotation methods (QuaRot, SpinQuant family) | 2404.00456 etc. | Hadamard rotations suppress outlier dimensions before low-bit quantization | The same "rotate then quantize" principle powers TurboQuant (below) |
| **QAT — Gemma 4 Technical Report** | **2607.02770** | **Quantization-aware training**: model is fine-tuned *with* the Q4_0 lattice in the loop; ~72% memory cut at near-BF16 quality; QAT drafter models for MTP included | The decisive 2026 result: 4-bit is no longer "lossy PTQ" for this family — it is the trained precision. Coding/agentic capability is explicitly preserved (native function-calling, thinking modes) |
| Unsloth Dynamic v3 (industry report + KLD methodology) | — (docs + 9 TB public artifacts) | Per-tensor-type mixed precision: `attn_*` kept high-precision (most KLD-sensitive), `ffn_up/gate` fine at 3-bit, `ffn_down` sensitive; **smart 4-bit recovery** re-aligns llama.cpp Q4_0 blocks to the BF16-QAT lattice (24.8% → 99.96% byte-exactness) | Empirical Pareto frontier on 99.9% KL-divergence: Gemma 4 12B UD-Q4_K_XL = 88.76% top-1 vs BF16-QAT (naive Q4_0 = 74.08%). Honest caveat: 12B is the hardest family member to recover (31B reaches 96.7%) |
| BitNet b1.58 / ternary LLMs | 2402.17764, 2504.12285 | Native 1.58-bit training | Frontier; no 12–16B open dense ternary in range as of 2026 — watchlist only |

**Verdict for W:** For an 8 GiB card, the winning move in 2026 is not "find a better PTQ algorithm" but **use a model whose 4-bit form was trained, then apply sensitivity-aware mixed precision (Unsloth UD-Q4_K_XL)** → 6405 MiB with measured high logit agreement. PTQ 4-bit (AWQ-style) remains the fallback for non-QAT models (Qwen3-14B etc.), at ~3.5–4.5 bpw.

### B. Dynamic runtime memory — KV cache compression (attacks KV)

| Paper | arXiv | Method | Savings | Fidelity risk for exact syntax / long-horizon agents |
|---|---|---|---|---|
| StreamingLLM (attention sinks) | 2309.17453 | Keep first tokens + sliding window; enables "infinite" stream | window-bounded KV | Loses mid-context facts — unacceptable for code across files |
| H2O | 2306.14048 | Evict low-cumulative-attention tokens | up to 5× KV | Eviction is **irreversible**; risks dropping exact identifiers |
| SnapKV | 2404.14469 | Pre-fill-time clustering/pooling of KV per head | 3–8× | Better than token-level eviction; still lossy on retrieval tasks |
| PyramidKV | 2406.02069 | Layer-adaptive budget (more KV in lower layers) | matches full KV at **12% of cache** | Pooling smooths precise token info — risky for code |
| ChunkKV | 2502.00299 | Semantic-chunk-level eviction (keeps chunk continuity) | 2–3× | Reduces mid-sentence corruption vs token-level methods |
| KIVI | 2402.02750 | **2-bit KV quantization** (per-channel K, per-token V) | 2.6× memory, up to 4× batch | Quantization (not eviction): recoverable precision; small degradation |
| QServe / QoQ (MLSys'25) | 2405.04532 | **W4A8KV4** system co-design; progressive quantization + dequant overhead elimination | 3.5× GPU memory reduction, 1.2–3× throughput | Shows KV4 is system-viable; A8 activations protect quality |
| **TurboQuant** | **2504.19874** | Online vector quantization: random rotation → concentrated Beta-distributed coordinates → near-optimal distortion at **<3 bits**, no calibration, per-vector | KV at ~2.5–3 bit with ≈0 accuracy loss claimed | Strongest theoretical KV result; active llama.cpp integration effort (discussion #20969, fork `TheTom/llama-cpp-turboquant`) — experimental as of 2026 |
| TriAxialKV | 2605.17170 | Mixed-precision per-token triaxial tags (recency/modality/semantic role), custom kernels, agentic workloads | extreme low-precision KV | 2026 SOTA direction; serving-system-grade engineering, not yet consumer-runnable |
| UltraQuant | 2606.20474 | 4-bit KV anchored to FP8 deployment, context-heavy agents | cuts KV percentile memory sharply | Confirms 4-bit KV is production-plausible in 2026 |
| Sensitivity-aware layerwise mixed-precision KV | 2502.04420 | Per-layer bit allocation by sensitivity | balanced | Same principle as Unsloth's weight allocation, applied to KV |
| FP8 KV productionization | vLLM blog 2026-04-22 | FP8 KV+attention quantization now default-ready (early naive FP8 collapsed accuracy 91%→13%; solved via scaling recipes) | 2× KV vs FP16 | Practical anchor: **FP8/q8_0-class KV is the safe default; q4_0-class is the aggressive tier** |

**Empirical tuning data (llama.cpp, DGX Spark benchmark, 2026):** vs f16 KV — `q8_0` = **−47% KV memory with ~0% generation-speed cost at ≤24K context**; `q4_0` = −72% memory but **−11.9% decode at 24K and −36.8% at 110K** (per-token dequantization overhead). → **Use q8_0 KV.** On Gemma 4 12B the hybrid architecture already shrinks KV 6×, so q8_0 is all you need; q4_0 would trade speed you need for memory you no longer lack.

### C. Architectural state management (attacks KV by design)

- **Interleaved sliding-window attention (SWA)** — Gemma 2/3/4 lineage (llama.cpp issue #12637 → SWA support merged 2025, "80% less KV VRAM"). Gemma 4 12B: 40/48 layers window=1024 → KV is O(window·L_local + ctx·L_global) instead of O(ctx·L).
- **Unified K/V + p-RoPE in global layers** (Gemma 4, arXiv:2607.02770) — further cuts the global-layer KV term; 128K context ≈ 1.2 GB KV (community architecture analyses).
- **GQA/MQA** (standard in all 2024+ models) and **cross-layer KV sharing / MLA** (DeepSeek-V2 2401.02669; YOCO-style 2402.18096) — the datacenter-side of the same idea.
- **Per-Layer Embeddings (PLE)** (Gemma 4 E2B/E4B) — parameters that don't participate in per-token bandwidth.

### D. Decode acceleration under bandwidth ceilings (attacks tok/s directly)

| Paper | arXiv | Mechanism | Notes |
|---|---|---|---|
| Speculative decoding | 2211.17192 / 2302.01318 | Draft-then-verify; **output distribution provably identical** | Lossless — the key property for exact code syntax |
| EAGLE / EAGLE-3 | 2401.15077 / 2503.01840 | Feature-level autoregressive drafting, trained heads | 2–6× in datacenter settings; needs per-model head training |
| **Native MTP heads** (Gemma 4, Qwen3.5/3.6) | model tech reports; llama.cpp PR #23398 | Model ships **trained multi-token predictor**; llama.cpp `--spec-type draft-mtp` | Gemma 4 QAT: drafter must be same-precision QAT checkpoint (Google compatibility note); Unsloth publishes Q4_0/Q8_0/BF16 MTP GGUFs; **1.4–2.2× measured on GGUF**, most on high-latency consumer GPUs |
| Bandwidth roofline study | 2507.14397 | Decode = memory-bound limit study | Formalizes why ≤7 GB weights is the hard requirement for 30 tok/s at 256 GB/s |

### E. System-level evidence for this exact hardware class

- **Silicon Showdown (arXiv:2605.00519, May 2026):** on discrete consumer GPUs, 70B-class models force the choice between *"Q2 quantization that degrades intelligence"* and *"PCIe CPU offloading that cuts throughput >90%"* — precisely the user's 4–5 tok/s symptom. Conclusion adopted here: **never split; shrink instead** (QAT weights + hybrid-attention KV + speculative decode).
- **Efficient Inference for Edge LLMs (TsingSci&Tech 2026 / arXiv surveys):** weak-GPU/strong-CPU devices benefit from speculative parallelism and static scheduling — consistent with MTP choice.

## 1.2 Fidelity trade-off synthesis (technical reasoning & exact syntax)

Ranked by risk to code correctness:

1. **Safest:** FP8/q8_0 KV quantization (≈0 measurable decode loss ≤24K), MTP/speculative decoding (provably lossless), QAT 4-bit weights (trained precision), hybrid SWA architecture (model-level, benchmarked by the vendor).
2. **Moderate:** PTQ 4-bit with activation-aware scaling + imatrix + sensitivity-based mixed precision (AWQ/Unsloth Dynamic) — 88–98% top-1 logit agreement depending on model size.
3. **Riskier:** ≤3-bit PTQ weight quant (IQ3/Q3 class) on non-QAT models — perplexity rises; exact identifier recall degrades; avoid for system-level script generation.
4. **Riskiest for code:** KV **eviction/pooling** (H2O/SnapKV/PyramidKV) — irreversible information loss can corrupt precise token recall (variable names, flags, paths). Great for chat summarization; **not recommended** for a coding/agent workload on 8 GiB where q8_0 + hybrid attention already suffices.

---

# Phase 2 — Open-Source Codebase Audit (GitHub, current 2026-10)

## 2.1 Engine comparison for a single-user 8 GiB Ada laptop GPU

| Engine | Verdict | Evidence |
|---|---|---|
| **llama.cpp / llama-server** | **PRIMARY.** Complete toolchain: QAT GGUF, FA + q8_0/q4_0 KV, SWA-aware KV allocation, MTP (`--spec-type draft-mtp`, PR #23398 merged), `-ngl all`, `--no-mmap`, `--cache-ram 0`, `-hf` auto-download | Community: 40 tok/s Gemma 4 12B UD-Q4_K_XL @ 64K ctx, `-fa on -ctk q4_0 -ctv q4_0` (Windows/CUDA, issue #24266 report); 120 tok/s QAT+MTP on a 12 GB card (r/LocalLLaMA); 21 tok/s baseline RTX 4060 no-MTP |
| **ExLlamaV3 + tabbyAPI** | **SECONDARY.** EXL3 trellis-coded quantization; Gemma 4 support landed Sept 2026; 2–8-bit cache modes; often fastest raw decode on single consumer GPU | turboderp-org/exllamav3; v1.0.0 perf release. Caveat: no Unsloth-style QAT-lattice recovery for EXL3 at the time of audit; MTP path less battle-tested than llama.cpp |
| **Unsloth (GGUF producer + Studio)** | **CRITICAL DEPENDENCY.** Dynamic v3 QAT recovery (UD-Q4_K_XL), MTP GGUFs bundled in-repo (`MTP/` folder), published KLD artifacts (9 TB) | unsloth.ai docs; `unsloth/gemma-4-12b-it-qat-GGUF` file tree verified via HF API |
| **vLLM** | Not recommended here. Gemma 4 QAT ships `w4a16` compressed-tensors for vLLM and FP8 KV is production-ready (2026-04 blog), but CUDA-graph pools, activation slabs, and fragmentation make 8 GiB single-stream marginal; strength is batched multi-user serving | google/gemma-4-12B-it-qat-q4_0-gguf model card; vllm.ai blog |
| **SGLang** | Same class as vLLM — RadixAttention/paged KV built for throughput servers; overkill and VRAM-hungry for one 8 GiB GPU | repo audit |
| **TensorRT-LLM** | "Backend dichotomy" (Silicon Showdown): NVFP4 wins on Blackwell, but consumer Ada 8 GiB path = heavy engine-build friction, kernel tuning, no MTP-QAT ecosystem parity | arXiv:2605.00519 |
| **ik_llama.cpp / experimental forks** | Watchlist: speculative-ngram and KLD-focused tooling; not required for Plan A | community forks |
| **TurboQuant fork (`TheTom/llama-cpp-turboquant`)** | Frontier KV path (turbo3/turbo4, block_size=128, CUDA fixes merged) — adopt only if you later need 128K+ context and outgrow q8_0 KV | llama.cpp discussion #20969 |

## 2.2 Memory-management mechanisms found in the wild

- **llama.cpp:** unified KV with per-layer SWA awareness; `--cache-type-k/v` (f16/bf16/q8_0/q4_0); `--cache-ram 0` disables the host-RAM prompt cache; `--no-mmap` avoids host page-cache-resident weights (prevents lazy paging stalls); `--no-warmup` avoids a startup allocation spike; `-ngl 99/all` pins every tensor to GPU. **No silent CPU fallback once everything fits** — the failure mode becomes a hard CUDA OOM (desired: fail loudly, never spill).
- **Windows-specific silent killer:** WDDM **"CUDA – Sysmem Fallback Policy"** lets CUDA allocations transparently spill to shared system RAM — this is exactly how users end up at 4–5 tok/s without any error. Must be set to **"Prefer No Sysmem Fallback"** (NVIDIA Control Panel → Manage 3D Settings), converting silent spill into a loud OOM.
- **ExLlamaV3:** `cache_mode` (FP8/Q8/Q6/Q4) for KV, `max_seq_len`/`max_batch_size` preallocation (fragmentation-free by design: one slab, no paging).
- **Display overhead:** on laptops, if the dGPU drives the display (MUX/direct mode), Windows compositing consumes 300–500 MiB of the 8 GiB. Keep the iGPU driving displays (hybrid/Optimus) or budget for it.
- **Fragmentation hygiene:** single-slot serving (`--parallel 1`), fixed context, `--no-mmap`, avoid multiple concurrent model loads.

## 2.3 Model-selection audit (12–16B class, must fit 6.3–6.8 GB to leave room for KV+MTP+buffers)

| Model (2026 catalog) | Quant | Weights | KV architecture | Fits 8 GiB + 30 tok/s? |
|---|---|---|---|---|
| **Gemma 4 12B Unified (11.95B)** QAT | **UD-Q4_K_XL** | **6405 MiB** | Hybrid 5:1 SWA + unified KV → 128K≈1.2GB | **YES — the only in-range model with QAT + MTP + tiny KV** |
| Gemma 4 12B QAT | Google official q4_0 GGUF | ~6.98 GB | same | Yes, but naive-conversion accuracy below Unsloth recovery (74% vs 89% top-1) |
| Gemma 3 12B QAT | Q4_0/UD | ~6.6–7.0 GB | Hybrid SWA (Gemma-3 style) | Yes; superseded by Gemma 4 (coding/agentic upgrades, MTP) |
| Qwen3-14B | UD-IQ4_XS ≈ 8.0 GB / IQ3_XXS ≈ 6.2 GB | 4-bit doesn't fit; 3-bit fits | Full attention, 40 layers → KV@32K q8_0 ≈ 640 MiB | Marginal: only at 3-bit (fidelity risk for exact syntax) — Plan B |
| Ministral-12B (Mistral) | Q4_K_M ≈ 7.4 GB | too tight with 32K KV | SWA 8:1 helps | Marginal; no QAT, no MTP — Plan B− |
| Qwen3.5-9B | UD-Q4_K_XL | ~5.5 GB | hybrid (SSM/attention mix) | Comfortable but **out of the 12–16B mandate** — Plan C (max speed) |
| Gemma 4 26B-A4B MoE | UD-Q4_K_XL | 14.2 GB | MoE | Requires expert offload to RAM — **violates the zero-RAM constraint**; rejected |
| Nemotron/Qwen 16B-class | Q4 | ≥9 GB | full attention | Rejected at 4-bit; ≤3-bit violates fidelity requirement |

---

# Phase 3 — Implementation Specification (RTX 4060 Laptop, 8 GiB)

## 3.0 Decision record

- **Weights:** Gemma 4 12B QAT `UD-Q4_K_XL` (Unsloth smart-recovery). 6405 MiB.
- **Speed:** MTP drafter Q4_0 (242 MiB), `--spec-draft-n-max 2` (tune 1–4). Lossless; converts a ~22–26 tok/s bandwidth-bound baseline into ~35–55 tok/s.
- **KV:** flash-attn on, `q8_0` K and V (−47% vs f16, ≈0 decode cost ≤32K). 32 768 ctx default.
- **Zero-RAM enforcement:** `-ngl 99` + `--no-mmap` + `--cache-ram 0` + `GGML_CUDA_NO_PINNED=1` + NVIDIA *Prefer No Sysmem Fallback* (Windows) + verification script proving shared-memory usage = 0 MiB.
- **Sampling (Google official for QAT):** temp 1.0, top-p 0.95, top-k 64; thinking mode on with a bounded budget for coding tasks.

## 3.1 VRAM budget (MiB, nvidia-smi units)

| Component | safe (16K) | balanced (32K) | max (64K) |
|---|---|---|---|
| Weights UD-Q4_K_XL | 6405 | 6405 | 6405 |
| MTP drafter Q4_0 | 242 | 242 | 242 |
| KV cache q8_0 (hybrid attn, est.) | ~100–180 | ~180–350 | ~330–660 |
| Compute buffers (`--n-batch`) | ~150 (nb=256) | ~250 (nb=512) | ~250 (nb=512) |
| CUDA context | ~250–350 | ~250–350 | ~250–350 |
| **Total (est.)** | **~7.0–7.3 GiB** | **~7.2–7.5 GiB** | **~7.4–7.8 GiB** |
| Headroom vs 8192 MiB | ≥0.9 GiB | ≥0.6 GiB | thin — Linux/iGPU-display only |

Windows users whose dGPU drives the display: subtract 300–500 MiB → use `safe` or `balanced`. If you ever hit CUDA OOM, that is the enforcement working (no silent spill): drop one FIT_MODE.

## 3.2 Step-by-step deployment

### Step 1 — Build llama.cpp with CUDA (sm_89)

Linux / WSL2:
```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake git curl pciutils
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89          # Ada laptop; omit to build all
cmake --build llama.cpp/build --config Release -j \
      --target llama-cli llama-server llama-bench
# binaries in llama.cpp/build/bin/
```
Windows: use the official CUDA release binaries (llama.cpp Releases → `cudart-llama-bin-win-cuda-*.zip`) **built after the MTP merge (≥ June 2026, b9551-class or newer)**, or build with Visual Studio + CUDA Toolkit 12.x/13.x using `-DGGML_CUDA=ON`. Verify MTP support: `llama-server --help | grep spec-type` must list `draft-mtp`.

### Step 2 — Enforce GPU-only execution (one-time host config)

- **Windows (mandatory):** NVIDIA Control Panel → Manage 3D Settings → Global → **CUDA – Sysmem Fallback Policy → "Prefer No Sysmem Fallback"**. This is the switch that turns your 4–5 tok/s silent RAM spill into an honest CUDA OOM.
- **Windows (recommended):** keep displays on the iGPU (hybrid graphics mode, not dGPU-direct/MUX) to reclaim 300–500 MiB; set `llama-server.exe` GPU preference = High Performance in Settings → Display → Graphics.
- **Linux:** ensure no desktop compositor on the dGPU (headless or iGPU display). Nothing else needed — llama.cpp with `-ngl 99` and a model that fits never touches host RAM for weights.
- Close browsers/GPU apps before serving (WDDM shares the 8 GiB).

### Step 3 — Download model + MTP drafter (6.7 GB total)

```bash
pip install -U "huggingface_hub[cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download unsloth/gemma-4-12b-it-qat-GGUF \
    gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
    MTP/mtp-gemma-4-12B-it-Q4_0.gguf \
    --local-dir ~/models/gemma-4-12b-qat
# Optional Q8_0 drafter (+200 MiB VRAM, slightly better acceptance):
#   MTP/mtp-gemma-4-12B-it-Q8_0.gguf
# Optional multimodal (skip to save VRAM — text-only serving):
#   mmproj-BF16.gguf
```
(Alternative source: `google/gemma-4-12B-it-qat-q4_0-gguf` — official, but Unsloth's recovery quant is smaller and benchmarks better.)

### Step 4 — Serve (the production command)

```bash
CUDA_VISIBLE_DEVICES=0 GGML_CUDA_NO_PINNED=1 \
llama-server \
  --model ~/models/gemma-4-12b-qat/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  -md ~/models/gemma-4-12b-qat/MTP/mtp-gemma-4-12B-it-Q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 \
  --flash-attn on \
  --ctx-size 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --n-batch 512 --ubatch-size 512 \
  --parallel 1 \
  --no-mmap --cache-ram 0 --no-warmup \
  --jinja \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --host 127.0.0.1 --port 8080
```
Use the wrapped scripts instead: `deploy/serve_gemma4_12b.sh` (Linux/WSL) or `deploy/serve_gemma4_12b.ps1` (Windows) — they add preflight checks, auto-download, FIT_MODE profiles, and VRAM logging.

### Step 5 — Validate (proof of zero-RAM operation + 30 tok/s)

Run `deploy/benchmark_validate.sh` (or the manual sequence):
1. `llama-bench` pp512/tg128 with and without MTP → record tg delta (expect 1.4–2.2×).
2. Start server, fire a long-context generation, and watch:
   - `nvidia-smi --query-gpu=memory.used --format=csv -l 1` → stays ≤ ~7.6 GiB;
   - **Windows:** Task Manager → Performance → GPU → *Shared GPU memory* delta = **0 MB** during generation (the hard proof);
   - `llama-server` stderr: all 48 layers on GPU0, KV buffer size as budgeted, `n_ctx` per slot as set.
3. Code-fidelity smoke test: prompt for a bash function, pipe output through `bash -n` (syntax check) — included in the script.

### Step 6 — Tuning matrix

| Symptom | Action |
|---|---|
| tg < 30 tok/s | Confirm `--spec-type draft-mtp` active; sweep `--spec-draft-n-max` ∈ {1,2,3,4} (hardware-dependent, do not assume 2); try Q8_0 drafter; ensure FA on |
| CUDA OOM at startup | Drop FIT_MODE (64K→32K→16K); Q4_0 drafter (already default); `--n-batch 256`; check display/MUX VRAM tax |
| Speed tanks mid-conversation | Never add `--spec-type ngram*` flags alongside MTP (known conflict, llama.cpp issue #24266 — 40→4 tok/s regression); `--cache-ram 0` already set |
| Prompt processing slow at 32K+ | Normal for 4060 Laptop (~200–400 tok/s pp); chunk long documents; keep `--ubatch-size 512` |
| Want 128K context | KV@128K q8_0 ≈ 0.6–1.2 GiB → won't fit with everything else; options: RAG/chunking, or evaluate TurboQuant fork (experimental) |
| Quality regression vs BF16 | You're on UD-Q4_K_XL (88.8% top-1); do **not** "upgrade" to higher-bit requants of the QAT checkpoint (Unsloth: >Q4_K_XL degrades, not improves); keep official sampling params |

## 3.3 Fallback ladder (if constraints shift)

- **Plan B — insist on 14B+ parameters:** Qwen3-14B `UD-IQ3_XXS` (~6.2 GB) + full-attention KV q8_0 @16K (~640 MiB) → fits, ~25–32 tok/s, but 3-bit PTQ: measurable syntax-fidelity risk. Only if parameter count outweighs precision.
- **Plan B′ — no-MTP Gemma 4 12B:** drop `-md/--spec-type` flags → 6.5 GiB footprint, ~21–28 tok/s (below target; MTP is what buys the margin).
- **Plan C — maximum speed, accept 9B:** Qwen3.5-9B UD-Q4_K_XL + MTP → ~60–90 tok/s, out of the 12–16B mandate.
- **Plan D — different engine:** ExLlamaV3/tabbyAPI with EXL3 (Gemma 4 support since Sept 2026), `cache_mode=FP8/Q8`, single-slab preallocation. Re-test MTP equivalence before switching.
- **Anti-patterns rejected (documented so they are never re-tried):** layer splitting with `-ngl < all` (the original 4–5 tok/s disease), MoE expert CPU-offload (`--n-cpu-moe`) — violates zero-RAM, KV eviction stacks (H2O/SnapKV) for code workloads, Windows Sysmem Fallback left enabled, GGUF q4_0 KV at long context (−37% decode).

## 3.4 Expected performance envelope (honest estimates)

| Metric | Estimate | Basis |
|---|---|---|
| Decode, no MTP | 21–26 tok/s | 6.72 GB/token ÷ (256 GB/s × 55–80% eff.); community 21 tok/s RTX 4060 |
| **Decode, MTP n=2 (default)** | **~35–55 tok/s** | 1.4–2.2× (Unsloth measured); 40 tok/s community report on 12B QAT Windows/CUDA |
| Prompt processing | ~200–450 tok/s | Ada SM89 + FA; batch 512 |
| Long-context decode penalty @32K | <5% | hybrid KV adds only ~0.2 GB/token-pass |
| VRAM | 7.2–7.6 GiB steady | budget table; verify with monitor script |
| System RAM used by inference | ~0 (process only) | `--no-mmap --cache-ram 0 GGML_CUDA_NO_PINNED=1` + no-sysmem-fallback |

---

## Source index (primary)

arXiv: 2607.02770 (Gemma 4/QAT tech report) · 2605.00519 (Silicon Showdown) · 2504.19874 (TurboQuant) · 2605.17170 (TriAxialKV) · 2606.20474 (UltraQuant) · 2405.04532 (QServe) · 2402.02750 (KIVI) · 2306.00978 (AWQ) · 2210.17323 (GPTQ) · 2308.13137 (OmniQuant) · 2406.02069 (PyramidKV) · 2404.14469 (SnapKV) · 2306.14048 (H2O) · 2502.00299 (ChunkKV) · 2309.17453 (StreamingLLM) · 2211.17192 & 2302.01318 (speculative decoding) · 2401.15077 (EAGLE) · 2507.14397 (bandwidth limit study) · 2502.04420 (mixed-precision KV).
GitHub/HF/docs: ggml-org/llama.cpp (PR #23398 MTP; issue #24266 spec-type conflict; discussion #20969 TurboQuant; docs/speculative.md) · turboderp-org/exllamav3 · google/gemma-4-12B-it-qat-q4_0-gguf · unsloth/gemma-4-12b-it-qat-GGUF (+ MTP/ tree) · unsloth.ai docs (Dynamic 3.0 GGUFs, Gemma 4 QAT analysis, MTP guide) · vllm.ai FP8 KV-cache blog (2026-04-22) · Memoriant/dgx-spark-kv-cache-benchmark (q8_0/q4_0 decode-cost data) · TheTom/llama-cpp-turboquant (experimental fork).
