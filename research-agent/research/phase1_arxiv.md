# Phase 1 — arXiv Deep-Dive Notes (2022 → 2026-10)

Method: broad keyword sweeps (weight quantization, KV-cache quantization/eviction, architectural state management, speculative decoding, edge/consumer inference systems), cross-referenced against citations and 2026 follow-up work. Every entry lists the memory term it attacks (W = weights, KV = cache, A = activations, S = speed), savings, and the fidelity trade-off for **technical reasoning / exact code syntax**.

---

## A. Static weight compression (W)

### A1. GPTQ — arXiv:2210.17323 (2022)
- One-shot PTQ to 3–4 bit via approximate second-order information (layer-wise OBQ with lazy batch updates + Cholesky reformulation).
- Made 4-bit 100B+ models loadable; error grows sharply below 4-bit on dense models.
- Fidelity: acceptable at 4-bit for general text; coding benchmarks show measurable drops at 3-bit.

### A2. AWQ — arXiv:2306.00978 (MLSys 2024)
- Observation: ~0.1–1% of weight channels are *salient*, identified by **activation** (not weight) magnitude; protect them via per-channel scaling — mathematically equivalent transform, no mixed-precision storage, no backprop.
- Hardware-efficient INT4/INT3 W4A16; the "activation-aware" principle now appears in nearly every serious quantizer.
- Fidelity: consistently better than GPTQ at 4-bit on reasoning/code (community benchmarks and Microsoft's INT4 guide concur).

### A3. OmniQuant — arXiv:2308.13137 (2023)
- Learnable weight clipping (LWC) + learnable equivalent transformation (LET); bridges PTQ→QAT cheaply; enables W4A4/W3A16 configs.
- Fidelity: W4A4 still risky for exact syntax; W4A16 solid.

### A4. Rotation-based outlier suppression (QuaRot arXiv:2404.00456, SpinQuant lineage)
- Hadamard/random rotations make activations & weights outlier-free → enables 4-bit (even W4A4) with small loss.
- Same mathematical core as TurboQuant's random rotation (B2).

### A5. QAT — Gemma 4 Technical Report, arXiv:2607.02770 (2026) ★ decisive for this project
- Google DeepMind trained the entire Gemma 4 family (E2B, E4B, **12B**, 26B-A4B, 31B) **with quantization in the loop**; ships Q4_0 checkpoints (unquantized, GGUF, mobile wNa8o8, vLLM w4a16 compressed-tensors) plus **QAT MTP drafter ("assistant") checkpoints**.
- ~72% memory reduction at near-BF16 quality. Compatibility rule: **drafter must be QAT checkpoint at same precision as target**.
- Fidelity: this is trained 4-bit, not post-hoc compression — the strongest available guarantee for code syntax at this footprint.

### A6. Unsloth Dynamic v3 / smart 4-bit recovery (2025–2026, industry; 9 TB public KLD artifacts)
- Mixed precision **by tensor type**, chosen on KL-divergence Pareto frontiers: `attn_*` tensors most KLD-sensitive → kept higher precision; `ffn_up/gate` tolerate 3-bit; `ffn_down` intermediate; embeddings over-quantized is wasteful.
- **QAT lattice recovery:** naive BF16-QAT→llama.cpp Q4_0 conversion matches the trained lattice only 24.77% byte-exact (F16 vs BF16 scale mismatch). Unsloth's recovery reaches **99.96% byte-exactness**.
- Measured (Gemma 4 12B): UD-Q4_K_XL = 6.72 GB, top-1 agreement 88.76%, mean KLD 0.133 — vs naive Q4_0 = 6.98 GB, top-1 74.08%, KLD 0.507. (For scale: 31B recovers to 96.67%; 12B is the hardest member — honest caveat.)
- Also: imatrix calibration helps low-bit KLD materially (−5–10% inference speed cost); I-quants (IQ2/IQ3) run 5–10% slower than K-quants at equal size.

### A7. Extreme-compression frontier (watchlist)
- BitNet b1.58 (arXiv:2402.17764) and BitNet b1.58 2B4T (arXiv:2504.12285): native ternary {-1,0,1} training — matches FP16 at ~7× fewer bits/energy. No 12–16B open dense ternary model yet in range; Unsloth publishes 1-bit-class GGUFs retaining ~77% accuracy (Qwen3.8 generation).
- AQLM/QuIP#-style vector/lattice coding → productized as **EXL3 trellis quantization** in ExLlamaV3 (Phase 2).

---

## B. Dynamic runtime memory — KV cache (KV)

### B1. Quantization family (reversible-precision; safe tier)
- **KIVI** — arXiv:2402.02750: 2-bit KV, per-channel for K, per-token for V; 2.6× memory ↓, larger feasible batch. Small quality loss.
- **QServe/QoQ** — arXiv:2405.04532 (MLSys'25): W4A8KV4 co-design; group dequant on CUDA cores, KV4 re-quantization fusion; 1.2–3× throughput vs TensorRT-LLM/llama.cpp of its day; 3.5× GPU memory ↓. Proved KV4 is *systems*-viable.
- **TurboQuant** — arXiv:2504.19874 (2025): online VQ with near-optimal distortion rate. Random rotation → coordinates follow concentrated Beta law → per-vector scaling achieves **<3-bit KV at ≈0 accuracy loss, no calibration, streaming-friendly**. Google Research blog (2026) popularized it; llama.cpp discussion #20969 tracks integration; working fork `TheTom/llama-cpp-turboquant` (turbo3/turbo4, block_size=128 → 5.12× KV compression, CUDA OOB-write fix merged). Status 2026-10: experimental.
- **TriAxialKV** — arXiv:2605.17170 (2026): mixed-precision KV assigning each token a triaxial tag (temporal recency / modality / semantic role) with per-tag calibrated sensitivity; end-to-end serving system with custom kernels; targets **agentic** long-horizon workloads. Confirms the field's direction: precision allocated by token importance, not uniform.
- **UltraQuant** — arXiv:2606.20474 (2026): 4-bit KV caching for context-heavy agents; FP8 KV as deployment anchor; cuts percentile memory on long multi-turn agent traces.
- **Sensitivity-aware layerwise mixed-precision KV** — arXiv:2502.04420 (2025): per-layer bit allocation.
- **FP8 KV productionization** — vLLM engineering blog 2026-04-22: early naive FP8 KV collapsed accuracy (91%→13% on their eval); with proper scaling recipes FP8 KV is now the recommended default tier for long context. Lesson: **q8_0/FP8 = safe default; q4_0/2-bit = aggressive tier needing recipe care**.

### B2. Eviction / pooling family (irreversible; risky tier for code)
- StreamingLLM — arXiv:2309.17453: attention-sink + window → unbounded streams, bounded KV. Loses mid-context facts.
- H2O — arXiv:2306.14048: heavy-hitter eviction by cumulative attention.
- SnapKV — arXiv:2404.14469: observation-window clustering → per-head KV pooling at prefill.
- PyramidKV — arXiv:2406.02069: pyramid layer budgets; matches full-cache quality at **12% of cache** on needle benchmarks.
- ChunkKV — arXiv:2502.00299 (2025): chunk-level (semantic-unit) eviction; fixes mid-token corruption of token-level methods.
- **Trade-off verdict:** eviction methods are benchmarked on retrieval/needle tasks, not byte-exact recall; for system-level script generation an evicted variable name = broken code. Use only if quantization + architecture no longer suffice (they do here).

### B3. Architectural state management (KV by design) ★ the Gemma 4 advantage
- **Interleaved SWA** (Gemma 2/3/4; Mistral lineage): most layers attend to a short window → their KV is window-sized forever. llama.cpp merged SWA-aware KV allocation in 2025 ("80% less KV VRAM" for Gemma 3, issue #12637 thread).
- **Gemma 4 12B specifics** (arXiv:2607.02770 + architecture analyses): 48 layers = 40 local (window 1024) + 8 global; global layers use **unified Keys/Values** and **p-RoPE** (proportional RoPE for long context). Community measurement: **KV ≈ 1.2 GB at 128K context** — vs a full-attention 12B model which would need ~10 GB+.
- GQA/MQA (universal), cross-layer KV sharing (YOCO arXiv:2402.18096), Multi-head Latent Attention (DeepSeek-V2 arXiv:2401.02669) — same direction at datacenter scale.
- PLE (per-layer embeddings, Gemma 4 E2B/E4B): parameter mass that adds no per-token bandwidth.

---

## C. Decode acceleration under bandwidth ceilings (S)

### C1. Speculative decoding — arXiv:2211.17192 (Leviathan et al.) / arXiv:2302.01318 (Chen et al.)
- Draft γ tokens cheaply, verify in one target pass, accept by rejection sampling → **output distribution mathematically identical** to the target model. The only "free" speedup: zero fidelity cost for exact syntax.

### C2. EAGLE / EAGLE-3 — arXiv:2401.15077 / arXiv:2503.01840
- Feature-level drafting with trained heads; 2–6× on datacenter GPUs; per-model head training required. NVIDIA/vLLM/SGLang ship production paths.

### C3. Native MTP heads — Gemma 4 (Google), Qwen3.5/3.6 (Alibaba) ★ adopted
- Models trained with multi-token-prediction objectives ship drafter weights; llama.cpp PR **#23398** (merged ~June 2026) exposes `--spec-type draft-mtp`, `--spec-draft-model/-md`, `--spec-draft-n-max`.
- Unsloth measured GGUF MTP: **1.4–2.2× generation speedup**, gains largest on memory-bandwidth-limited devices (exactly the 4060 Laptop regime). Drafter memory: Gemma 4 12B MTP Q4_0 = 242 MiB, Q8_0 = 444 MiB.
- Known pitfall (llama.cpp issue #24266): mixing `--spec-type ngram*` with MTP-era builds regressed 40→4 tok/s; resolved by removing ngram spec flags. Also a Vulkan `--spec-default` regression on b9549. → **Pin explicit spec flags; validate with llama-bench after any engine update.**

### C4. Roofline justification — arXiv:2507.14397 (2025)
- Autoregressive decode is memory-bandwidth-bound; compute utilization stays low regardless of FLOPs. For 256 GB/s (4060 Laptop): tok/s ceiling ≈ 256 / GB-of-weights-per-pass × efficiency. 6.72 GB → 38 tok/s ceiling at 100%; ~22–30 realistic at 55–80% efficiency → **MTP required to guarantee 30+**; it amortizes the weight-streaming pass over ~1.5–2.2 accepted tokens.

---

## D. Systems papers characterizing this exact failure mode

- **Silicon Showdown — arXiv:2605.00519 (May 2026):** empirical study of consumer LLM inference. Quantifies the "VRAM Wall": CPU offloading = **>90% throughput loss** vs full-GPU (matches the reported 4–5 tok/s vs 30+ target); Q2-style compression to fit = intelligence degradation; documents the TensorRT-LLM NVFP4 "backend dichotomy" on Blackwell and 23× energy-efficiency advantage of Apple UMA. Direct academic validation of the strategy: *fit entirely in VRAM with trained 4-bit + architectural KV economy + lossless speculation*.
- **Efficient Inference for Edge LLMs (Tsinghua Sci&Tech 2026):** edge devices with weak GPUs benefit from speculative parallelism, static memory planning, single-slab allocation (mirrors ExLlamaV3 design).
- **DGX Spark KV benchmark data (llama.cpp discussion #20969, corrected numbers, build 8399):** f16→q8_0 KV: −360 MiB (−47%) with **no measurable decode cost at 6–24K**; f16→q4_0: −72% memory but decode −11.9% @24K, **−36.8% @110K** (per-token dequant overhead). → q8_0 is the right KV tier for 32K working context; q4_0 only when KV-bound, not speed-bound.

## Cross-reference matrix (what each discovery is worth on 8 GiB / 256 GB/s)

| Discovery | Memory saved on this build | Speed effect | Code-fidelity risk | Adopted? |
|---|---|---|---|---|
| QAT weights (A5) | 23.8→6.98 GB | none | ≈0 (trained) | ✅ via A6 recovery |
| Unsloth UD-Q4_K_XL (A6) | 6.98→6.72 GB | none | low (88.8% top-1) | ✅ primary |
| Hybrid SWA + unified KV (B3) | KV@128K ~10GB→1.2GB | neutral+ | none (vendor-benchmarked) | ✅ inherent to model |
| q8_0 KV quant (B1) | KV −47% | ~0% @≤32K | negligible | ✅ |
| q4_0 KV | KV −72% | −12…−37% long ctx | small | ❌ (speed-bound, not KV-bound) |
| TurboQuant <3-bit KV (B1) | KV −75%+ | TBD | ≈0 claimed | ⏸ experimental fork; only if 128K needed |
| KV eviction H2O/SnapKV/Pyramid (B2) | KV −60…88% | +small | **high for exact syntax** | ❌ |
| MTP spec decode (C3) | +242 MiB | **×1.4–2.2** | **zero (lossless verify)** | ✅ the 30 tok/s guarantor |
| EAGLE-3 (C2) | head weights | ×2+ | zero | ⏸ (no trained head needed — MTP exists) |
| Layer-split / MoE-offload to RAM | "fits" anything | **−90%** (the disease) | none | ❌ hard-banned by spec |
