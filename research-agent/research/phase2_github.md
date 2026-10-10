# Phase 2 — GitHub / Open-Source Codebase Audit (state as of 2026-10-07)

Scope: working implementations of Phase 1 theory, engine memory-management internals, and active developer fights against the 8 GiB ceiling (issues, PRs, discussions, forks).

---

## 1. llama.cpp (ggml-org/llama.cpp) — PRIMARY ENGINE ✅

**Why it wins for this hardware:** per-layer GPU offload control, SWA-aware unified KV cache, KV quantization (f16/bf16/q8_0/q4_0), MTP speculative decoding merged, single-user efficiency on consumer CUDA, GGUF ecosystem (QAT + Unsloth recovery quants + MTP drafters all published as GGUF).

Key mechanisms audited:
- `-ngl 99` / `-ngl all`: every tensor (incl. output norm) to GPU. With a model that fits, no host-RAM path exists at decode time.
- `--flash-attn on`: required for quantized KV; Ada-tuned FA kernels.
- `--cache-type-k q8_0 --cache-type-v q8_0`: −47% KV vs f16; measured ≈0% decode cost ≤24K ctx (DGX Spark data, discussion #20969). `q4_0` rejected: −11.9% @24K, −36.8% @110K decode.
- `--no-mmap`: weights fully read into VRAM at load; no page-cache-resident tensor paging (removes a hidden host-RAM dependency and first-token stalls).
- `--cache-ram 0`: disables the host-RAM prompt cache (2026 builds) — aligns with zero-RAM mandate.
- `--no-warmup`: skips startup warmup pass (avoids transient allocation spike on 8 GiB).
- `--parallel 1`: single slot → single KV slab → no fragmentation from concurrent slot caches.
- **MTP (PR #23398, merged ~June 2026):** `-md <drafter.gguf> --spec-type draft-mtp --spec-draft-n-max N`. Verified command shape from field reports:
  `llama-server -m gemma-4-12B-it-qat-UD-Q4_K_XL.gguf -md mtp-…-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 -ngl all -fa on -ctk q8_0 -ctv q8_0 -c 40960 --no-mmap --cache-ram 0 -np 1`
  Docs: `docs/speculative.md` (spec-type taxonomy incl. ngram-map, draft-mtp).
- **Issue #24266 (June 2026) — critical pitfall:** combining `--spec-type ngram-mod` with post-MTP builds dropped 40→4 tok/s; removing ngram spec restored 40 tok/s. Also a `--spec-default` Vulkan regression on b9549 (Intel Mac). Lesson baked into deploy scripts: explicit spec flags only, re-bench after engine updates.
- Field performance data points: **40 tok/s** Gemma 4 12B UD-Q4_K_XL @64K ctx, Windows CUDA (`-fa on -ctk q4_0 -ctv q4_0 -c 64000`); **120 tok/s** Gemma 4 12B QAT+MTP on a 12 GB card (r/LocalLLaMA, mtp-bench.py); **21 tok/s** baseline (no MTP) RTX 4060 + CUDA 13.2 build.
- **Discussion #20969 — TurboQuant integration thread:** Google's <3-bit online KV quantization (arXiv:2504.19874); 189 upvotes; working fork `TheTom/llama-cpp-turboquant` (branch `feature/turboquant-kv-cache`): CUDA path with block_size=128 (compression 4.57×→5.12×), HIP support, turbo4 prefill. Status: experimental — tracked as the future path to 128K+ context inside 8 GiB.
- **Issue #12637 → SWA support (2025):** interleaved sliding-window KV allocation for Gemma-family = the implementation behind the "1.2 GB @128K" KV economy.

## 2. Unsloth (huggingface.co/unsloth + unsloth.ai docs) — CRITICAL MODEL PRODUCER ✅

- Repo audited via HF API (`unsloth/gemma-4-12b-it-qat-GGUF`, file tree verified 2026-10):
  - `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` — **6,716,356,800 B (6405 MiB)** — the only quant they publish for QAT models, because *higher-bit requants of QAT checkpoints measurably degrade* (their KLD work).
  - `MTP/mtp-gemma-4-12B-it-Q4_0.gguf` — **253,708,800 B (242 MiB)**; `MTP/…-Q8_0.gguf` — 465 MB; `MTP/…-BF16.gguf`/`F16` — 862 MB; root `mtp-gemma-4-12B-it.gguf` (Q4_0 duplicate).
  - `mmproj-BF16/F16/F32.gguf` — 175–210 MB (vision/audio projector; **skip for text-only to save VRAM**).
- Dynamic v3 methodology: per-tensor-type bit allocation on KLD Pareto frontiers; 150+ KLD benchmarks, 9 TB artifacts public; imatrix data refresh (Mar 2026) improving chat/coding/long-context/tool-calling; tool-calling chat-template fix (affects all quant uploaders).
- QAT recovery analysis: naive Q4_0 conversion of BF16-QAT lattice = 24.77% byte-exact; Unsloth recovery = 99.96%; Gemma 4 12B: top-1 74.08%→**88.76%**, mean KLD 0.507→0.133, size 6.98→6.72 GB.
- MTP guide: llama.cpp invocation (`--spec-type draft-mtp --spec-draft-n-max 2..4`), +~2 GB RAM/VRAM headroom warning (their generic figure; actual Gemma-4-12B drafter is only 242–465 MiB + draft KV), Unsloth Studio auto-tuning.
- Hardware requirement tables: Gemma 4 **12B QAT = 7 GB total** (matches our 8 GiB budget); Qwen3.5-9B = the sub-range alternative.

## 3. Google official releases ✅ (fallback model source)

- `google/gemma-4-12B-it-qat-q4_0-gguf` (788K downloads, verified via HF API): official QAT GGUF; repo total 11.9 GB (includes multiple precision artifacts); Apache-2.0; card confirms drafter-compatibility rule (QAT assistant must match target precision) and vLLM `w4a16` compressed-tensors siblings.
- Gemma 4 architecture (from card): 12B Unified = 11.95B params, 48 layers, SWA 1024, **256K context**, unified K/V + p-RoPE global layers, encoder-free multimodality, native system role, function calling, thinking modes.

## 4. ExLlamaV3 / tabbyAPI (turboderp-org/exllamav3) — SECONDARY ENGINE ⚠️

- EXL3: trellis-coded quantization (AQ/Lloyd-Max lineage from Phase 1), 2–8-bit cache modes (KV quantization), dynamic batching, single-slab preallocation (`max_seq_len`, `max_batch_size`) → **zero fragmentation** memory model, ideal for fixed 8 GiB envelopes.
- Gemma 4 support shipped ~Sept 2026 ("started off last month with the release of gemma 4 support" — v1.0.0-era updates); DFlash attention support followed.
- Community: EXL3 ≈ fastest raw decode on single consumer GPUs; 3 bpw class quants ~20% smaller than GGUF equivalents at similar quality (5090 comparison, atomic.chat guide).
- Why not primary: (a) no Unsloth-grade QAT-lattice recovery published for EXL3 Gemma-4-12B at audit time; (b) MTP-drafter path younger than llama.cpp's; (c) GGUF ecosystem has the verified 8 GiB field reports. Kept as Plan D.

## 5. vLLM — NOT RECOMMENDED for this envelope ❌

- 2026 state: FP8 KV-cache + attention quantization production-ready (official blog 2026-04-22; earlier naive FP8 collapsed accuracy 91%→13% — fixed by scaling recipes); Gemma 4 QAT `w4a16` compressed-tensors supported; benchmarked all 4-bit weight methods (finding: GPTQ-without-Marlin slower than FP16; AWQ oddly slow in some versions).
- Why rejected at 8 GiB: designed for batched throughput — CUDA-graph pools, per-step activation slabs, paged-KV block granularity, and weight+KV preallocation (`gpu_memory_utilization`) leave little slack on 8 GiB; single-user latency optimizations (MTP-QAT GGUF path) live in llama.cpp. Documented so the choice is explicit.

## 6. SGLang / TensorRT-LLM — REJECTED for this envelope ❌

- SGLang: RadixAttention prefix caching + continuous batching = server-side tech; same 8 GiB slack problem as vLLM.
- TensorRT-LLM: per Silicon Showdown (arXiv:2605.00519), its consumer win (NVFP4, 1.6× over BF16) is Blackwell-gated; on Ada 8 GiB you inherit engine-build friction without the format advantage; no Gemma-4-QAT+MTP parity.

## 7. Forks, PRs, discussions actively fighting the 8 GiB wall (watchlist)

| Thread/repo | What it fights | Status |
|---|---|---|
| llama.cpp discussion #20969 + `TheTom/llama-cpp-turboquant` | KV <3-bit via TurboQuant (Google) | Active fork; CUDA fixes merged upstream into fork; experimental |
| llama.cpp PR #23398 / issue #24266 | MTP support + spec-type conflicts | Merged; conflict pitfalls documented; re-bench after updates |
| llama.cpp issue #12637 + SWA merge | Gemma-family KV bloat | Merged (2025); foundation of 1.2GB@128K |
| `Memoriant/dgx-spark-kv-cache-benchmark` | q8_0/q4_0 vs f16 KV decode-cost data | Public methodology; source of our q8_0 choice |
| Unsloth Qwen3.5/3.6 MTP GGUF uploads | MTP beyond Gemma (9B in-range-adjacent) | Available (`unsloth/Qwen3.6-27B-MTP-GGUF` etc.; 27B too big for 8 GiB, 9B too small for mandate) |
| LM Studio (`lmstudio-community/gemma-4-12B-it-QAT-GGUF`) | GUI distribution of QAT GGUFs | Convenient experimentation; llama.cpp engine underneath — production spec targets llama-server directly for flag control |
| ik_llama.cpp and KLD-oriented forks | Better spec-decode + quant evaluation tooling | Not required for Plan A |

## 8. Host/OS memory-management findings (the silent killers)

1. **Windows WDDM "CUDA – Sysmem Fallback Policy" (default: enabled)** — CUDA allocations that exceed VRAM *silently* spill to Shared System RAM over PCIe. This is the exact mechanism producing the user's 4–5 tok/s with no error message. Setting **"Prefer No Sysmem Fallback"** (NVIDIA Control Panel → Manage 3D Settings, global or per-process for `llama-server.exe`) converts spill into a loud CUDA OOM — the enforcement layer the task demands.
2. **Display VRAM tax:** if the laptop dGPU drives displays (MUX/direct mode), compositing consumes 300–500 MiB of the 8 GiB. Hybrid/iGPU-display mode reclaims it.
3. **Pinned host buffers:** ggml CUDA allocates pinned host memory for async H2D (prompt processing). `GGML_CUDA_NO_PINNED=1` avoids large pinned allocations (slight pp cost; removes a host-RAM footprint).
4. **Page-cache-resident weights:** mmap'd GGUFs keep tensor pages in host RAM and can fault mid-generation; `--no-mmap` forces full VRAM residency at load (slower start, deterministic runtime).
5. **Host prompt cache:** 2026 llama.cpp keeps a RAM-backed prompt cache for slot reuse; `--cache-ram 0` disables it.

## 9. Reproduction snapshot (versions to pin)

- llama.cpp: any build **after** PR #23398 merge (June 2026; ≥ b9551-class) with `draft-mtp` in `--spec-type` help. CUDA arch sm_89.
- Model: `unsloth/gemma-4-12b-it-qat-GGUF` @ commit `sha 29d0977…` era (files byte-sizes verified 2026-10-07 via HF tree API).
- Driver: ≥ 576-series (CUDA 12.8+) or CUDA 13.x on Windows; Linux 570+.
- Verification tooling: `llama-bench`, `nvidia-smi`, Windows Task Manager shared-GPU-memory counter, `bash -n` syntax smoke test (in `deploy/benchmark_validate.sh`).
