# Mandate 3 — System-Level Hardware Isolation (research dossier, 2026-10)

**Goal:** engine implementations + driver-level overrides that *completely disable* shared-system-memory fallback routing, forcing all allocation natively into the GPU frame buffer — and prove it.

## 3.1 The mechanism of the disease (Windows/WDDM)

- Under WDDM, when CUDA allocations exceed dedicated VRAM, the driver **silently** migrates pages to "Shared GPU Memory" (= up to 50% of system RAM, reachable only over PCIe). No error, no log line — just a 10–30× throughput collapse (llmconfigurator.com troubleshooting guide, 2026; dev.to llama.cpp shared-VRAM analysis).
- **The kill switch:** driver **≥ 546.01** (Windows) added *"CUDA – Sysmem Fallback Policy"* (NVIDIA KB 5490). Set to **"Prefer No Sysmem Fallback"** → CUDA allocations that don't fit **fail with OOM instead of spilling**. Community adoption is widespread exactly for local-LLM/Stable-Diffusion workloads (oobabooga/textgen discussion #4484 "Nvidia finally fixes the VRAM manager"; r/LocalLLaMA PSA for driver 564.01; NVIDIA developer forum "cudaMalloc with sysmem fallback" — *"The only way to prevent this is to change the sysmem fallback in the Windows driver"*).
- Scope: NVCP setting applies **globally or per-program profile** (create a profile for `llama-server.exe` to avoid affecting games that rely on fallback). Automation route: **NVIDIA Profile Inspector** (open-source, `orbmu2k/nvidiaProfileInspector`) exposes the same setting ("CUDA - Sysmem Fallback Policy") for scripted profile edits.
- **Linux native:** `cudaMalloc` is already strict — VRAM exhaustion returns `cudaErrorMemoryAllocation`, no silent spill. Oversubscription only exists if the app explicitly uses Unified Memory/HMM page migration (`cudaMallocManaged`) — ggml/llama.cpp does **not** for weights or KV. ⇒ Linux native = isolated by default.
- **WSL2 caveat:** WSL CUDA goes through the Windows host WDDM stack (`/dev/dxg`) ⇒ the **host's** Sysmem Fallback Policy governs. Must be set on the Windows side; verify with the monitor (host Task Manager shows the vmmem/WSL process shared-usage).

## 3.2 Engine-level allocation controls (force on-chip residency)

### llama.cpp (primary engine)
| Control | Effect |
|---|---|
| `-ngl 99` / `-ngl all` | every tensor (incl. output norm) to GPU; no layer split exists at runtime |
| `--spec-draft-ngl all` | drafter (MTP/EAGLE-3/n-gram none needed) fully in VRAM — don't let the draft model default-offload |
| `--no-mmap` | weights read fully into VRAM at load; no host page-cache-resident tensors, no mid-generation page faults |
| `--cache-ram 0` | disables the host-RAM prompt cache (2026 builds) |
| `GGML_CUDA_NO_PINNED=1` | avoids large pinned host buffers (H2D staging) — removes host-RAM footprint at slight prompt-processing cost |
| `--parallel 1` + fixed `--ctx-size` | single KV slab ⇒ no multi-slot fragmentation |
| `--no-warmup` | avoids a transient allocation spike at startup on tight cards |
| Never set `--n-cpu-moe` / `-ot .*cpu` | explicit host-placement overrides — banned by mandate |

### ExLlamaV2/V3
- Single-slab **preallocation**: `max_seq_len`, `max_batch_size`, `cache_mode` (Q8/FP8/…) reserve one contiguous VRAM region at load; no paging, no host residency, fragmentation-free by design. If it loads, it is isolated. (`turboderp-org/exllamav3`)

### vLLM / PyTorch-class engines
- `gpu_memory_utilization` = hard VRAM budget; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` fixes the classic "free VRAM exists but fragmented OOM" failure (PyTorch allocator advisory; netray/codegenes guides) — fragmentation is what pushes naive configs into spill. Not needed for llama.cpp (static slabs), mandatory for PyTorch engines on 8 GiB.

### Display/compositor tax (laptops)
- dGPU-driven displays (MUX/direct mode) consume 300–500 MiB of the 8 GiB inside the same frame buffer. Hybrid/iGPU-display mode reclaims it. Not a *fallback* issue but part of the native-allocation budget. Windows: also set `llama-server.exe` → High Performance GPU preference (`HKCU\Software\Microsoft\DirectX\UserGpuPreferences`, `GpuPreference=2;`).

## 3.3 Verification instrumentation (the proof layer)

1. **Windows shared-usage counter (authoritative):** performance counter set `GPU Process Memory` → instances `pid_<PID>_luid_…` → counters **`Dedicated Usage`** and **`Shared Usage`** (bytes). Assert `ΔShared == 0` across a full generation. PowerShell: `Get-Counter '\GPU Process Memory(*)\Shared Usage'`. Task Manager → Details → add "Dedicated GPU memory" column (per-process) as a GUI cross-check (stackoverflow #73476947).
2. **Linux:** `nvidia-smi --query-gpu=memory.used --format=csv` (dedicated only; the driver does not spill natively) + watch process **RSS** (`/proc/<pid>/status`) for host-side growth; llama.cpp load log must show `CUDA0 model buffer`/`CUDA0 KV buffer` accounting ≈ weights+KV with no meaningful `CPU model buffer`.
3. **Engine logs:** llama.cpp prints per-device buffer allocations at load; any CPU-resident tensor beyond a few MiB of norm weights = violation. llama-server `--metrics` exposes KV usage + speculative acceptance for continuous telemetry.
4. **Failure semantics check (one-time):** with fallback disabled, deliberately load an oversized config (e.g. `FIT_MODE=max` + Q8_0 drafter + 128K ctx) — the correct behavior is a **loud CUDA OOM at load**, never 4–5 tok/s operation. This negative test proves the isolation layer is active.

## 3.4 Driver/version matrix

| Platform | Spill risk | Enforcement |
|---|---|---|
| Windows native (WDDM) | **High (silent)** | Driver ≥546.01 + NVCP/Profile-Inspector "Prefer No Sysmem Fallback" + monitor |
| WSL2 | High (inherits host WDDM) | Same host-side setting; verify via host Task Manager/vmmem |
| Linux native | None (strict cudaMalloc) | Nothing to disable; verify via nvidia-smi + RSS + engine logs |
| macOS | n/a (UMA — no dedicated/shared split) | — |
