# Mandate 2 — Runtime Compute Sparsity & Layer Skipping (research dossier, 2026-10)

**Goal:** dynamically bypass network depth / skip internal layers per token complexity, reducing the runtime data-reading footprint — with working kernels, PRs, or forks.

## 2.1 Academic lineage (the proofs)

| Work | arXiv / venue | Mechanism | Guarantee & numbers |
|---|---|---|---|
| **Mixture-of-Depths (MoD)** | 2404.02258 (DeepMind, 2024) | Per-block **router** selects top-k tokens to receive attention+MLP; the rest pass through the residual. Hard capacity k ⇒ **enforced total compute budget** | Matches iso-FLOPs baselines; **up to 50% faster wall-clock stepping**; converts "dynamic compute" into predictable static-shaped batches (padding to k). Repo: `google-deepmind/mixture-of-depths` |
| **LayerSkip** | 2404.16710 (Meta, ACL 2024) | Train with layer dropout + **early-exit losses**; at inference: exit early on easy tokens, and use the first *E* layers as a **self-draft**, verified by the full model | Self-speculative ⇒ **lossless** (rejection-sampling verification). 1.4–2× decode on Llama-class. Repo: `facebookresearch/LayerSkip` (inference integrated into HF transformers ecosystem) |
| CALM | 2207.07061 (Google) | Confidence-calibrated early exit per token | Lossy unless verified; the calibration theorem (exit when softmax margin > threshold with bounded error) is the ancestor of all halting gates |
| SkipDecode | 2307.07262 | Skip later layers for low-entropy continuation positions | Training-free; lossy |
| AdaSkip | 2403.05327 | Adaptive per-layer classifiers decide skip vs exit | Training-light |
| **What Layers When** | 2510.13876 (ICLR 2026) | **Residual gates** learned during training to skip layers token-wise | Modern re-formulation; shows skip patterns must be trained, not bolted on |
| **Adaptive Computation Depth / Token-Selective Attention** | 2605.05222 (2026) | Lightweight 2-layer MLP gate on residual updates between consecutive blocks → continuous **halting probability**, end-to-end trainable | The current arXiv state of per-token depth routing |

**Consensus theorem-level takeaway:** unverified depth skipping is *lossy* (changes the output distribution); **verified** depth skipping (draft with fewer layers / a head, verify with full depth) is *lossless* by the speculative-sampling proof (2211.17192). Every production system in 2026 chose the verified form.

## 2.2 Production implementations in llama.cpp (audited `docs/speculative.md`, master @ 2026-10)

llama.cpp's `--spec-type` taxonomy — each accepted draft token **skips one full-depth (48-layer) decode pass** of the target model, which is exactly "runtime layer bypass" in its verified form:

| spec-type | What runs | VRAM cost | Lossless? | Notes for 8 GiB build |
|---|---|---|---|---|
| `draft-mtp` | **Trained MTP heads from the main model** (Gemma 4, Qwen3.5/3.6) | 242–465 MiB (Gemma 4 12B Q4_0/Q8_0 drafter) | ✅ verified | **Primary.** Closest production analog to LayerSkip self-speculation: the head is a shallow continuation of the same network |
| `draft-eagle3` | 1-layer transformer reading target hidden states (EAGLE-3) | ~100–500 MiB | ✅ verified | Supported heads incl. `RedHatAI/gemma-4-31B…eagle3`, `AngelSlim/Qwen3-{4B,8B,14B…}_eagle3` (conversion: `convert_hf_to_gguf.py --target-model-dir`) |
| `draft-dflash` | Block-diffusion draft — emits a whole block per pass (`z-lab/Qwen3-4B-DFlash`) | small | ✅ verified | `--spec-draft-n-max` clamped to trained block size; PR #22105 |
| `draft-dspark` | DFlash + semi-autoregressive **Markov head** (`deepseek-ai/dspark_qwen3_4b_block7`) | small | ✅ verified | Qwen3 backbone only; **Gemma 4 support planned**; PR #25173 |
| `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` | Draftless pattern lookup in context history | ~0 (CPU tables) | ✅ verified | Docs explicitly recommend for **code refactoring/iteration** workloads |
| `ngram-mod` | Rolling LCG hash pool, **~16 MB constant memory**, shared across slots, variable draft length | 16 MiB | ✅ verified | Best for repeated blocks (llama.vim, reasoning echo); tunables `--spec-ngram-mod-n-match/n-min/n-max` |
| Mixing | `--spec-type draft-mtp,ngram-mod` (comma-separated; draftless has precedence when combined with draft-model types) | sum | ✅ | Enables MTP + code-repetition speculation simultaneously |

Supporting machinery: `--spec-draft-n-max/-n-min`, `--spec-draft-p-split`, `--spec-draft-ngl all` (keep drafter 100% in VRAM — isolation), `-ctkd/-ctvd` (drafter KV quantization), `--spec-synth-rates P0,P1,…` / `--spec-synth-len L` (**synthetic acceptance** — calibrate the throughput model without any drafter by solving `p + p² + … + p^K = L−1`), `--metrics` (acceptance-rate counters).

**Not merged:** true LayerSkip early-exit inside ggml — discussion **#10787** ("requires getting one intermediate layer's output + some KV cache changes") is the canonical feasibility thread; community PR interest continues (r/LocalLLaMA "Add self-speculative decoding (no draft model required)"). Watchlist item; do not plan around it.

## 2.3 Static/bought-in sparsity (MoE) — and why it fails the 8 GiB zero-RAM constraint

- MoE routes tokens through k-of-N experts (Qwen3.5-35B-A3B: 3B active; Gemma 4 26B-A4B: 3.8B active) — the commercial form of "skip most weights per token."
- 26B-A4B UD-Q4_K_XL = **14.2 GB** > 8 GiB ⇒ requires `--n-cpu-moe` expert offload to system RAM ⇒ violates the isolation mandate (and reintroduces the PCIe disease; Silicon Showdown arXiv:2605.00519 measures >90% loss). **Rejected**, documented as anti-pattern.
- Same arithmetic kills gpt-oss-20b (MXFP4 ≈ 12–13 GB).

## 2.4 Bandwidth math: why verified skipping is THE lever on 256 GB/s

Decode cost per token ≈ bytes(weights touched) / bandwidth. Gemma 4 12B UD-Q4_K_XL: 6.72 GB/pass ⇒ ≤38 tok/s at 100% efficiency, ~22–26 realistic. Each draft token accepted via MTP **amortizes the full-depth weight read over 1+α tokens**: with acceptance giving mean accepted length L (incl. target token), effective tok/s ≈ baseline × L / (1 + draft_cost/full_cost). For L≈2–3 and a 242 MiB head (≈3.6% of target bytes), expected **1.5–2.2×** (Unsloth measured range) ⇒ 35–55 tok/s. `--spec-synth-len` lets you verify this model analytically before trusting live runs (playbook T4).

**SWA as layer-work skipping:** in Gemma 4, 40/48 layers never read beyond 1024 tokens of KV regardless of context length — per-layer KV read footprint is O(window) not O(ctx). At 32K context this removes ~90% of attention memory traffic vs a full-attention 12B model; measured long-context decode penalty <5% (playbook T3 verifies).

## 2.5 Repos & threads index

- `facebookresearch/LayerSkip` — early-exit + self-speculation reference implementation
- `google-deepmind/mixture-of-depths` — MoD reference (JAX)
- `ggml-org/llama.cpp` — `docs/speculative.md` (spec-type table), PR #23398 (MTP), #22105 (DFlash), #25173 (DSpark), #18039 (EAGLE-3 model list), discussion #10787 (LayerSkip feasibility), issue #24266 (spec-type mixing regression history)
- `z-lab` DFlash checkpoints; `deepseek-ai/dspark_*`; `AngelSlim/*_eagle3`; `RedHatAI/gemma-4-*-speculator.eagle3`
- `vllm-project/speculators` — portable draft-head export format (DSpark/EAGLE3 interop)
