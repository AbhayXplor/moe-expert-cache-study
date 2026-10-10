# PLAYBOOK — Unified Compilation & Script Execution Plan
### Testing lossless context pruning, verified layer-skipping, and hard VRAM isolation on an 8 GiB consumer GPU

Companion dossiers: `research/mandate1_context_pruning.md` · `research/mandate2_sparsity_layerskip.md` · `research/mandate3_hardware_isolation.md`.
Baseline deployment from Round 1: `REPORT.md` + `deploy/serve_gemma4_12b.sh|.ps1` (Gemma 4 12B QAT UD-Q4_K_XL + MTP drafter, 32K ctx, q8_0 KV).

---

## Stage 0 — Compilation (one engine build, three test branches)

### 0.1 Primary: llama.cpp master (CUDA, sm_89) — required for full `--spec-type` taxonomy
```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake git curl pciutils python3-pip
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
      -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build llama.cpp/build --config Release -j \
      --target llama-cli llama-server llama-bench
```
Verify the build exposes everything the mandates need:
```bash
./llama.cpp/build/bin/llama-server --help | grep -E "spec-type|cache-type-k|cache-ram|spec-draft-ngl|spec-synth"
# expect: draft-mtp / draft-eagle3 / draft-dflash / draft-dspark / ngram-* types,
#         --spec-synth-rates|--spec-synth-len, --spec-draft-ngl, --cache-ram
```
Windows: official CUDA release zip dated after PR #23398 (≥ June 2026, b9551-class+), or VS2022 + CUDA 12.x/13.x build with the same cmake flags.

### 0.2 Experimental branch: TurboQuant KV fork (Mandate 1, T2+ tier — optional)
```bash
git clone https://github.com/TheTom/llama-cpp-turboquant -b feature/turboquant-kv-cache
cmake llama-cpp-turboquant -B tq/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build tq/build --config Release -j --target llama-cli llama-server
# use EXACTLY as primary, replacing cache types with turbo3/turbo4 per fork docs;
# block_size=128 path => 5.12x KV compression (discussion #20969)
```
Only needed if you later require 64K–128K context inside 8 GiB. Do not mix binaries with Stage-2 results tables.

### 0.3 Models & assets (once)
```bash
pip install -U "huggingface_hub[cli]" hf_transfer
hf download unsloth/gemma-4-12b-it-qat-GGUF \
    gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
    MTP/mtp-gemma-4-12B-it-Q4_0.gguf MTP/mtp-gemma-4-12B-it-Q8_0.gguf \
    --local-dir ~/models/gemma-4-12b-qat
```

---

## Stage 1 — Hardware isolation enforcement (Mandate 3) — do FIRST, test everything under it

| OS | Action | Script |
|---|---|---|
| **Windows** | Driver ≥546.01; NVCP → Manage 3D Settings → (optionally a per-program profile for `llama-server.exe`) → **CUDA – Sysmem Fallback Policy → "Prefer No Sysmem Fallback"**; GPU preference High-Performance; displays on iGPU | `deploy/isolate_and_verify.ps1` (checks driver, sets GPU-preference registry, monitors `GPU Process Memory\Shared Usage` counter, PASS/FAIL) |
| **WSL2** | Same host-side Windows setting (WSL CUDA traverses host WDDM) | run the .ps1 monitor on the host while stressing from WSL |
| **Linux native** | Strict by default (`cudaMalloc` OOMs, never spills; no UVM in ggml weight/KV paths) | `deploy/isolate_and_verify.sh` (nvidia-smi + RSS + engine-log audit) |

**Negative test (proves the enforcement is live):** with fallback disabled, launch an intentionally impossible config (128K ctx + Q8_0 drafter). Correct outcome: **loud CUDA OOM at load**. Wrong outcome: server starts and crawls at 4–5 tok/s ⇒ your fallback policy did not take effect — fix before trusting any benchmark.

---

## Stage 2 — Test matrix (`deploy/test_mandates.sh`)

Run order matters: T5 monitoring wraps T1–T4 so every speed number is also an isolation proof.

| ID | Mandate | Test | Method | Pass criteria / expected |
|---|---|---|---|---|
| **T1** | 2 | Speculative stack A/B | `llama-cli` greedy 256-token gen, four arms: `--spec-type none` · `ngram-mod` · `draft-mtp` · `draft-mtp,ngram-mod` (MTP Q4_0 drafter, `--spec-draft-ngl all`, `-ctkd/-ctvd q8_0`) | mtp ≥ 1.4× none; combo ≥ mtp on code-repetition prompts; acceptance rate printed from stderr; **outputs identical across arms** (greedy + verification ⇒ lossless) |
| **T2** | 1 | KV-tier fidelity & speed | Same long technical prompt (embedded identifiers, paths, flags), greedy `--temp 0 --seed 42`, arms: `f16` / `q8_0` / `q4_0` KV; diff token streams vs f16 ground truth | q8_0 divergence ≈ 0 (few tokens over 256); q4_0 divergence materially higher; q8_0 VRAM −47% vs f16, decode ≈0% slower ≤32K (matches DGX-Spark dataset) |
| **T3** | 1+2 | Long-context scaling (SWA linearity) | `llama-bench -pg 8192,128 -pg 16384,128 -pg 32768,128` (plus 65536 if profile allows) | tg degradation 8K→32K < ~10% (hybrid attention); KV MiB in log grows ~linearly only in the 8 global layers' share |
| **T4** | 2 | Throughput model calibration | Synthetic acceptance: `--spec-synth-len L` for L∈{1.5, 2.0, 2.5, 3.0} (no real drafter quality needed) | measured tok/s ≈ baseline × f(L) per §2.4 math; validates the acceptance→speed transfer function before trusting T1 deltas |
| **T5** | 3 | Isolation under load | Wrap T1–T4 with monitors: Windows `Shared Usage` counter / Linux nvidia-smi + RSS + engine-log buffer audit | Δshared == 0 B (Win); dedicated ≤ ~7.7 GiB; RSS flat after load; log shows CUDA0-only model/KV buffers |

Each test prints a machine-readable `RESULT` line; the harness ends with a summary table and an overall PASS/FAIL.

### Manual spot-checks (if you don't use the harness)
```bash
# T1 arm: MTP + ngram-mod mix (code-repetition workload)
./llama.cpp/build/bin/llama-cli -m ~/models/gemma-4-12b-qat/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  -md ~/models/gemma-4-12b-qat/MTP/mtp-gemma-4-12B-it-Q4_0.gguf \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --spec-draft-ngl all \
  -ngl 99 -fa on -c 32768 -ctk q8_0 -ctv q8_0 --no-mmap --cache-ram 0 \
  --temp 0 --seed 42 -no-cnv --no-display-prompt -n 256 -p "$(cat prompts/tech_hist.txt)"
```

---

## Stage 3 — Decision rules (what the results mean)

| Observation | Action |
|---|---|
| T1: `draft-mtp,ngram-mod` ≥ mtp alone on your workload | Ship the mixed spec stack in production (comma-separated `--spec-type`; draftless has precedence — verify no regression like issue #24266's ngram conflict on YOUR build) |
| T1: acceptance mean length < 1.8 at n-max=2 | Sweep `--spec-draft-n-max 1..4` and/or Q8_0 drafter (`MTP/…-Q8_0.gguf`, +202 MiB) |
| T2: q8_0 divergence > ~1% of tokens on your prompt corpus | Move to f16 KV if VRAM allows (safe profile), else evaluate TurboQuant fork (T2+ tier, bounded distortion) before considering any eviction method |
| T2: q4_0 needed for ctx ≥64K | Only with TurboQuant-class kernels (q4_0 stock costs −37% decode at 110K); otherwise prefer lower ctx + VeriCache-style architecture (below) |
| T3: >10% tg degradation 8K→32K | Check FA actually enabled (log), KV types applied, and that no CPU buffer lines appear (isolation leak) |
| Context demand outgrows VRAM *and* losslessness is non-negotiable | Adopt **VeriCache pattern** (compressed KV drafts in VRAM + full KV verified from host/NVMe, overlapped with decode — arXiv:2605.17613, LMCache ecosystem). This deliberately re-admits host RAM as a *cold verification store* (not a compute path) — the documented, paper-backed exception to zero-RAM |
| Any T5 failure | Stop; benchmark numbers are meaningless under spill. Re-run Stage 1 |

---

## Stage 4 — Production integration

Promote winning arms into the Round-1 launcher (already patched): `deploy/serve_gemma4_12b.sh|.ps1` now include `--spec-draft-ngl all`, drafter KV quantization (`-ctkd/-ctvd q8_0`), `--metrics`, and the isolation env block. Validation: re-run `deploy/benchmark_validate.sh` + `isolate_and_verify.*` after any engine/model change (pin commit hash; re-bench per issue #24266 lesson).

**Recommended production line (post-playbook, balanced profile):**
```bash
CUDA_VISIBLE_DEVICES=0 GGML_CUDA_NO_PINNED=1 \
llama-server \
  --model ~/models/gemma-4-12b-qat/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  -md ~/models/gemma-4-12b-qat/MTP/mtp-gemma-4-12B-it-Q4_0.gguf \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --spec-draft-ngl all \
  -ctkd q8_0 -ctvd q8_0 \
  -ngl 99 --flash-attn on --ctx-size 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --n-batch 512 --ubatch-size 512 --parallel 1 \
  --no-mmap --cache-ram 0 --no-warmup --jinja --metrics \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --host 127.0.0.1 --port 8080
```

## Appendix — pins & links
- llama.cpp master ≥ post-#25173 (DSpark) for full spec taxonomy; `docs/speculative.md` = flag source of truth.
- Forks/branches: `TheTom/llama-cpp-turboquant@feature/turboquant-kv-cache` (experimental KV).
- Papers: 2605.17613 (VeriCache) · 2605.01708 (SplitZip, repo `Intelligent-Microsystems-Lab/SplitZip`) · 2504.19874 (TurboQuant) · 2404.02258 (MoD) · 2404.16710 (LayerSkip, repo `facebookresearch/LayerSkip`) · 2510.13876 (What Layers When) · 2605.05222 (adaptive depth) · 2211.17192/2302.01318 (speculative sampling proofs).
- Threads: llama.cpp #10787 (LayerSkip feasibility) · #20969 (TurboQuant) · #24266 (spec mixing pitfall) · NVIDIA KB 5490 + forum "cudaMalloc with sysmem fallback" (isolation).
