# deploy/ — Quickstart: Gemma 4 12B QAT on RTX 4060 Laptop (8 GiB), GPU-only, 30+ tok/s

Companion to `../REPORT.md` (research + rationale). Three files:

| File | Purpose |
|---|---|
| `serve_gemma4_12b.sh` | Production server launcher (Linux / WSL2) — now with `--spec-draft-ngl all`, drafter KV q8_0, `--metrics`, optional `SPEC_MIX=1` (draft-mtp,ngram-mod) |
| `serve_gemma4_12b.ps1` | Production server launcher (Windows PowerShell) — same additions |
| `benchmark_validate.sh` | Round-1 validation: throughput + zero-RAM-fallback proof + code-fidelity smoke test |
| `test_mandates.sh` | Round-2 A/B harness: **T1** speculative-stack arms · **T2** KV-tier fidelity/divergence metric · **T3** long-context SWA scaling · **T4** synthetic-acceptance calibration · **T5** isolation monitor |
| `isolate_and_verify.sh` | Linux/WSL2 hard-isolation audit (VRAM peak + RSS + engine-log buffer audit + negative-test guide) |
| `isolate_and_verify.ps1` | Windows hard-isolation audit (driver gate, GPU-preference pin, `GPU Process Memory` Shared/Dedicated Usage counters under live load, PASS/FAIL) |
| `set_sysmem_fallback.ps1` | **Round 3:** programmatic driver enforcement of "Prefer No Sysmem Fallback" (SettingID 283962569/0x10ECECC9, Dword 1) via auto-downloaded nvidiaProfileInspector + `.nip` XML — no Control Panel clicking; global or per-app profile |
| `splitzip_smoke.sh` | **Round 3:** clone + venv + bitwise round-trip + throughput validation of the SplitZip lossless KV codec on your GPU (Linux/WSL2) |

Round-2 dossiers + execution plan: `../PLAYBOOK.md`, `../research/mandate{1,2,3}_*.md`. Round-3 deep dive (SplitZip/VeriCache implementation status, lossless-KV landscape, driver automation mechanics): `../research/round3_splitzip_vericache_status.md`.

## TL;DR command (after building llama.cpp with CUDA and downloading the model)

```bash
llama-server \
  -m gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  -md MTP/mtp-gemma-4-12B-it-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 --flash-attn on -c 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --n-batch 512 --ubatch-size 512 --parallel 1 \
  --no-mmap --cache-ram 0 --no-warmup --jinja \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --host 127.0.0.1 --port 8080
```

Every flag maps to a constraint: `-ngl 99` (no layer split → no PCIe spill), `--no-mmap` + `--cache-ram 0` + `GGML_CUDA_NO_PINNED=1` (no host-RAM residency of any kind), `q8_0` KV (−47% KV memory at ~0 decode cost ≤32K), `draft-mtp` (lossless 1.4–2.2× that clears the 30 tok/s bar on 256 GB/s).

## Profile matrix

| FIT_MODE | Context | Est. VRAM | Use when |
|---|---|---|---|
| `safe` | 16 384 | ~7.0–7.3 GiB | dGPU drives displays; other GPU apps open |
| `balanced` (default) | 32 768 | ~7.2–7.5 GiB | normal operation |
| `max` | 65 536 | ~7.4–7.8 GiB | Linux / iGPU-display, nothing else on GPU |

If you get CUDA OOM: that's the enforcement working (Sysmem Fallback disabled = loud failure instead of silent 4 tok/s). Step down one profile.

## Windows one-time setup (mandatory)

1. **NVIDIA Control Panel → Manage 3D Settings → CUDA – Sysmem Fallback Policy → "Prefer No Sysmem Fallback"** — this single setting is what *completely disables* the Shared-System-RAM fallback path.
2. Keep displays on the iGPU (hybrid graphics, not dGPU-direct/MUX) → reclaims 300–500 MiB.
3. Settings → System → Display → Graphics → add `llama-server.exe` → High Performance.
4. Verify during generation: Task Manager → Performance → GPU → **Shared GPU memory must not move**.

## WSL2 notes

- WSL2 CUDA works out of the box with recent Windows drivers; use the Linux build steps.
- Reserve VRAM sanity: run `nvidia-smi` inside WSL — it reports the same 8192 MiB.
- Keep the Windows-side NVIDIA Sysmem Fallback setting (it governs WSL CUDA too).

## Validation (do this once per engine/model update)

```bash
./benchmark_validate.sh              # full: residency + bench + live 30 tok/s check + bash -n fidelity test
./benchmark_validate.sh --bench-only # engine benchmark without a running server
```

Targets: decode ≥30 tok/s with MTP (expect ~35–55), prompt-processing ≥200 tok/s, `nvidia-smi` steady ≤ ~7.6 GiB, shared-memory delta = 0, generated script passes `bash -n`.

## Tuning cheatsheet

- `SPEC_DRAFT_N_MAX=1..4` — sweep; keep the fastest (hardware-dependent; 2 is only a starting point).
- `MTP_QUANT=Q8_0` — +202 MiB VRAM, marginally better draft acceptance.
- `ENABLE_THINKING=false` — lower latency for non-reasoning chat; keep `true` for coding/system tasks (bounded reasoning improves exactness).
- `ENABLE_MTP=0` — only for A/B measurement; baseline decode (~21–26 tok/s) is below target on this card, which is precisely why MTP ships enabled.
- Never pass `--spec-type ngram*` together with MTP-era builds (llama.cpp issue #24266: 40→4 tok/s regression).
- Multimodal (`mmproj-*.gguf`): +175 MiB — text-only serving omits it by design.

## Plan B / C / D (documented in REPORT.md §3.3)

- **B — strict 14B:** Qwen3-14B `UD-IQ3_XXS` (~6.2 GB) @16K, q8_0 KV, no MTP head of equal quality → fits, ~25–32 tok/s, 3-bit fidelity risk.
- **C — max speed:** Qwen3.5-9B UD-Q4_K_XL + MTP (~60–90 tok/s) — below the 12B mandate.
- **D — other engine:** ExLlamaV3/tabbyAPI, EXL3, `cache_mode=Q8/FP8`, single-slab preallocation (Gemma 4 supported since Sept 2026).
