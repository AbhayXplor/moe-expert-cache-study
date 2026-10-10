# Round 3 Deep-Dive: SplitZip & VeriCache — Practical Implementation Status
### For prototype design · verified 2026-10-07 · plain-English structural report

**One-line answers:**
1. **SplitZip** — code IS public (`Intelligent-Microsystems-Lab/SplitZip`, NeurIPS 2026, MIT): a standalone PyTorch/**Triton** codec (~4 Python files). The SGLang+Mooncake serving integration used in the paper is **NOT released**; zero integrations exist in llama.cpp/vLLM/ExLlama today.
2. **VeriCache** — **no public repo exists** (GitHub search: 0 hits; no LMCache PR/issue mentions). It is "explicitly built on top of LMCache" (UChicago + Tensormesh Inc.). The cloneable substrate is `github.com/LMCache/LMCache` (Apache-style OSS, extremely active — 2,442 commits, commits signed by VeriCache co-authors @tensormesh.ai, prefetch controller + multi-tier DRAM/Device-DAX plumbing already in tree).
3. **Lossless guess-and-check for KV** — VeriCache is the **first and only** published framework of its kind (their claim, and our survey agrees). The mathematical machinery (draft-then-verify rejection sampling) is productionized for *tokens* (llama.cpp MTP/EAGLE-3/DFlash/DSpark), and bitwise-lossless *codecs* exist (SplitZip, DFloat11, ZipNN), but nobody else publicly verifies a *compressed KV state* against a full KV state at inference time.
4. **Windows driver enforcement** — fully scriptable: setting **"CUDA Sysmem Fallback Policy", SettingID 283962569 (0x10ECECC9), DWORD value 1 = "Prefer No Sysmem Fallback" (0 = driver default/allow spill)**, applied via NVAPI-DRS. Proven community pattern: generate an nvidiaProfileInspector `.nip` XML and import it with `inspector.exe <file.nip>` (working example: FR33THY's Disable-TAA script embeds exactly this setting). Ready-made script: `deploy/set_sysmem_fallback.ps1`.

---

## §1 SplitZip — exact state of the code

**Repo:** `github.com/Intelligent-Microsystems-Lab/SplitZip` · University of Notre Dame (Intelligent Microsystems Lab: Yipin Guo, Prof. Siddharth Joshi) · **NeurIPS 2026** · MIT license · released 2026-06-05, last commit 2026-09-28 · 11 commits (research-artifact scale).

### 1.1 What the repo actually contains (audited file-by-file)
| File | Role |
|---|---|
| `codec_gpu.py` | The entire GPU codec — **Triton kernels + Python wrapper class `ChunkLocalSplitZipGPU`** |
| `bench_codec_throughput.py` | Codec-path throughput benchmark; verifies bitwise round-trip before reporting ratios (synthetic mode or saved `.pt` KV tensors) |
| `analyze_bf16_exponent_entropy_qwen32.py` | Paper's exponent-entropy study (WikiText-2 × Qwen3-32B) |
| `requirements.txt`, `figs/`, `LICENSE`, `README.md` | Python 3.12 + PyTorch/Triton stack; no C++/CUDA build needed |

### 1.2 How the compression works (plain English + kernel map)
BF16 numbers are `1 sign + 8 exponent + 7 mantissa` bits. KV activations cluster on a handful of exponent values (magnitudes live in a narrow range), so the exponent byte is highly compressible while sign+mantissa must stay exact:
- **Calibration** (`codec.calibrate(x)` → coverage): histogram the 8-bit exponent field over a calibration set; the **top-16 exponents** become a codebook; build a 256-entry LUT `exponent → 4-bit code` (miss ⇒ escape marker 16).
- **Encode** (kernels `_enc_4bit_quad64[_count_chunk|_count2_chunks]`, `_enc_4bit`): process 4×BF16 packed in a `uint64` lane; emit (a) an **exponent plane** — 4 codes packed per `uint16` (4 bits/value), (b) a **sign+mantissa plane** — `(sign<<7)|mantissa7` stored verbatim (8 bits/value), (c) **escape metadata** — `_count_escapes_chunk` counts LUT misses per 1024-element chunk; `_write_escapes_chunk[_split4]` compacts `(chunk_id, local_pos:uint16, exp:uint8)` with `tl.cumsum`-rank scatter (the "sparse escape stream"; prefix-sum of counts done host-side).
- **Decode** (`_dec_4bit*`, `_decode_exp_*`, `_merge_exp_sm_*`, `_fix_escape_exponents_local_linear`): LUT-expand codes → exponent plane, scatter escaped exponents back to exact positions, recombine `raw = (sign<<15)|(exp<<7)|mantissa7`. Bitwise identity asserted via `torch.equal(x.view(torch.int16), decoded.view(torch.int16))`.
- **Compression math:** 16 bits → 4 (exp code) + 8 (sign/mantissa) = 12 bits + amortized escapes ⇒ **≈1.31× on BF16**; applied on FP8-E5M2 KV: additional **up to 1.14×**. Speed (H200, real activations): **613.3 GB/s encode / 2181.8 GB/s decode** — decode is faster than HBM can serve it, i.e. free on the read path. Beats nvCOMP Bitcomp (341/148), LZ4 (13/137), DFloat11 decode (468), Falcon, ZipNN (1.2/1.7), ZipServ (–/1261).
- **API caveat (from README):** the public codec is **stateful** — encoded blobs must be decoded by the *same calibrated codec state*; calibrate on a separate calibration set, not the benchmark tensor.

### 1.3 Integration status — who has actually shipped it?
- **Exercised in an out-of-tree SGLang prefill-decode-disaggregation path using Mooncake as the KV transfer engine — and that integration code is NOT in the public release** (README: "We plan to document, publish, or upstream that integration path separately").
- **llama.cpp: none. vLLM: none. ExLlama(V3): none.** No PRs/issues referencing SplitZip found in any of the three as of 2026-10-07.
- README FAQ explicitly frames it as pluggable into "other KV connectors (e.g., vLLM, NVIDIA Dynamo/NIXL, LMCache)" — it's a standalone module, integration is on you.
- Models tested: Qwen3-32B, Qwen3-30B-A3B (MoE), Llama-3-8B, Phi-2. **FP8 KV path exists** (E5M2, 1.14×).

### 1.4 Limitations & integration strategy for our prototype
- **Scope:** SplitZip compresses *transfer/storage bytes*, not attention compute — it does not shrink what attention kernels read (decode must see raw BF16/FP8). Its home is the **KV-connector path**: PD-disaggregation, KV offload/restore, cross-machine prefix-cache shipping. On a single 8 GiB laptop its natural use is: (a) compressing KV archived to NVMe/RAM for session restore, (b) the transfer leg inside a VeriCache-style design.
- **Hardware:** Triton ⇒ runs on Ada sm_89 (Linux/WSL2; native Windows Triton is the constraint — use WSL2). Numbers were measured on H200; on a 4060 Laptop expect proportionally lower absolute GB/s but the same "faster than PCIe" property (PCIe Gen4 x8 laptop ≈ 16 GB/s ≪ codec speed), which is all that matters for the transfer use case.
- **Prototype hook:** clone → `pip install -r requirements.txt` → run `deploy/splitzip_smoke.sh` (bitwise round-trip + throughput on your GPU) → then wrap `codec.encode/decode` around any host↔device KV movement (e.g., llama.cpp KV save/load tensors, or a Python vLLM connector). Verify with the smoke test's `torch.equal` assertion before trusting it.

---

## §2 VeriCache — exact state of the code & the async-swap machinery

**Paper:** arXiv:2605.17613 (May 2026) — Jiayi Yao, Samuel Shen, Kuntai Du, Shaoting Feng, Dongjoo Seo, Rui Zhang, Yuyang Huang, Yuhan Liu, Shan Lu, Junchen Jiang. Affiliations: **University of Chicago + Tensormesh Inc.** (Jiang/Du/Shen's company) + Samsung Semiconductor + Microsoft Research.

### 2.1 Is there a repo to clone?
- **No standalone "VeriCache" repository exists** (GitHub repo search `vericache` → 0 results; LMCache issue/PR search `vericache` → 0 results, checked 2026-10-07).
- alphaXiv's paper summary states it directly: **"VeriCache is explicitly built on top of LMCache to handle the swapping of the full KV cache from CPU/storage to the GPU for its verification step."**
- **What you CAN clone today: `github.com/LMCache/LMCache`** (default branch `dev`) — the KV-cache layer underneath: multi-tier storage (L1 DRAM **and Device-DAX** per Oct-2026 commits), a **prefetch controller** (Sept 2026 refactor series), CUDA `csrc/` kernels (incl. FP8 e4m3/e5m2 Re-RoPE KV handling), gRPC transport, vLLM + SGLang connectors with nightly CI, docker/wheel builds (x86 + arm64). VeriCache co-author **Samuel Shen (slshen@tensormesh.ai) commits directly to LMCache** — i.e., the swapping substrate is public and under active development by the same team; the verification scheduler itself is not (yet) public. Practical posture: build on LMCache, watch for upstreaming / Tensormesh releases.

### 2.2 How the asynchronous RAM↔VRAM swapping works (paper §4–§5, plain English)
**Deployment shape:** the *compressed* KV cache lives permanently in GPU HBM and serves all drafting; the *full* KV cache lives in **CPU RAM** (long-context decode case) or **storage** (remote prefix-cache case). Verification = temporarily stream the full KV back to the GPU, run one forward pass over the x drafted tokens, accept/reject by the standard speculative-sampling rule ⇒ output **identical to full-KV greedy decoding** (their lossless definition: identical at temperature 0 modulo hardware nondeterminism).

**Why the swap can be hidden (the two design principles):**
- **P1 — Cross-resource staggering.** Drafting is HBM-bandwidth-bound; verification consumes *different* resources (PCIe/network bandwidth + GPU compute). So instead of lock-step draft-all→verify-all, the runtime verifies only ~B/x of the B concurrent requests per iteration, overlapping one request's KV_full transfer with other requests' drafting. Iteration time: `T_iter = max( (M + B·KV_full·(c + 1/x)) / BW_hbm , B·KV_full / (x·BW_inter) )`. Worked example (Mistral-24B, RTX PRO 6000, PCIe5 64 GB/s, B=10, KV_full=4 GB, c=0.25, x=30): each verify's ~80 ms PCIe transfer hides inside draft iterations; lock-step instead serializes 40 GB (~800 ms) and spikes HBM to 90 GB.
- **P2 — High acceptance amortizes verification.** The drafter *is the target model* running on compressed KV (weights and dominant attention patterns intact), so acceptance γ stays **>0.8 even at draft length 30** (measured acceptance lengths ~19 for Qwen-32B, ~23 for Llama-70B at 4× compaction — vs ~3–10 for small draft models). Ideal speedup peaks ~3.7× around draft length 15; end-to-end up to 4× with identical outputs. It **composes with MTP/EAGLE**: small-model drafter proposes → verify against compressed KV → periodic full-KV recheck; VeriCache+EAGLE ideal speedup 4.35× vs 3.50×/1.78× alone.

**The runtime scheduler (the "async" engine, §5):**
- Maintains a **W-iteration lookahead with two reservation rings**: a **BW ring** `T[i]` (interconnect time already booked in window i; link is serialized; a KV_full reload of size ℓ_r = KV_full/(BW·T_iter) occupies S_r = ⌈ℓ_r⌉ consecutive windows) and an **HBM ring** `B[i]` (in-flight KV staged for upcoming verifies), constrained by `M + KV_resident + B[i] ≤ HBM` for every window.
- **Admission (Algorithm 1):** on arrival *and after every verify*, the scheduler books request r's next verify iteration d_r: try the ideal slot x (20–50), then x±1, x±2…, clamped to [S_r, W−1]; first candidate fitting **both rings** gets the reservation; otherwise the request waits. This is what makes transfers land "just in time" without ever stalling the decode loop or blowing HBM.
- A **uniform compressor interface** lets any token-dropping or quantization method (KVzip, KIVI, TurboQuant-class, …) serve as the drafting cache.

### 2.3 Limitations & integration strategy
- **It re-admits system RAM — deliberately.** KV_full lives in host memory; that violates a *strict* zero-RAM mandate. Its trade only pays when (a) context is too big for VRAM even compressed, or (b) you serve many concurrent requests (staggering needs a batch to interleave; B=1 still pays KV_full/BW_inter per x tokens — on a laptop's PCIe4 x8 ≈16 GB/s with, say, 4 GB KV_full and x=30, that's ~250 ms of transfer per ~30 tokens ≈ tolerable-but-not-free).
- **It's a multi-user server design** (vLLM/SGLang + LMCache plumbing). No llama.cpp path exists; porting the scheduler into llama.cpp's single-slot engine is a research project, not an integration.
- **Motivation data you should steal for the prototype pitch:** under KVzip 4× compression of Qwen3-Coder-30B, F1 stays >75% while **code-format accuracy falls to near zero and function-call accuracy <10%** (SWE-bench Lite / ComplexFuncBench) — soft metrics mask functional breakage; verification is the only known cure.
- **Prototype path:** (1) clone LMCache, run its vLLM connector on a ≥16 GiB dev box to feel the swap/prefetch machinery; (2) prototype the *single-user degenerate case* — compressed KV in VRAM, KV_full in pinned host RAM, verify every x tokens — the math above gives you the exact break-even (x·T_decode vs KV_full/BW_inter); (3) track Tensormesh/LMCache for the verification-scheduler upstreaming.

---

## §3 Alternative "lossless guess-and-check" competitors — full landscape

Nobody else does KV-state draft-and-verify publicly (VeriCache's "first framework" claim survives our sweep). Adjacent mechanisms worth knowing:

| Project | Mechanism | Lossless? | Code | Relevance to prototype |
|---|---|---|---|---|
| **llama.cpp spec stack** (`draft-mtp`, `draft-eagle3`, `draft-dflash`, `draft-dspark`, `ngram-*`) | draft-then-verify at the **token** level (same rejection-sampling math as VeriCache, applied to weights-based drafts) | ✅ verified-lossless | `ggml-org/llama.cpp` `docs/speculative.md` | The production-ready half of the idea; already in our Round-1/2 stack |
| **TurboQuant** (+ `TheTom/llama-cpp-turboquant` fork) | random-rotation scalar quantization, **bounded distortion**, <3-bit KV, no verification | ❌ bounded-lossy | fork branch `feature/turboquant-kv-cache`; llama.cpp discussion #20969 | The best *compressed-draft* candidate to pair with a VeriCache-style verifier |
| **SplitZip** | bitwise codec (exponent codebook + escape stream) | ✅ **bitwise** | `Intelligent-Microsystems-Lab/SplitZip` | Transfer/storage layer; composes under everything |
| **DFloat11** (NeurIPS'25, arXiv:2504.11651) | dynamic-length float — Huffman-coded exponents, lossless **weight** compression (~30% smaller, 468 GB/s decode) | ✅ bitwise | `github.com/LeanModels/DFloat11` | Same lineage as SplitZip applied to weights; SplitZip table shows it's ~4.7× slower decode |
| **ZipNN** (arXiv:2411.05239) | lossless model-file compression (byte-stream, exponent-aware); HF Transformers RFC #34737; supports safetensors/vLLM | ✅ bitwise | `github.com/zipnn/zipnn` | Storage/download layer (1.2–1.7 GB/s — offline only, not KV-path) |
| **FastDMS** (shisa-ai, May 2026) | *trained* Dynamic Memory Sparsification: compact FP8 KV layout that **reclaims evicted slots**; 5–8× allocator-visible compression, 1.5–2× decode vs vLLM; nano-vllM-based engine + trainer, PyPI | ❌ lossy (trained sparsification) | `github.com/shisa-ai/FastDMS` | Best-in-class *lossy* engine-side compaction; candidate "compressor" behind a VeriCache-style verifier |
| **TANGRAM** (arXiv:2606.06302, Aug 2026) | serving substrate making **non-uniform** (per-layer/head) KV compression practical for multi-turn on vLLM | ❌ lossy (compressor-agnostic) | `github.com/aiha-lab/TANGRAM` | The plumbing layer non-uniform compressors need; pairs conceptually with verification |
| **llama.cpp paged-KV prototype** (discussion #21961) | PagedAttention-style KV + continuous batching inside llama.cpp | n/a (allocation) | design-feedback thread, working prototype | Fragmentation hygiene for our engine; watch for merge |
| KVzip / FastKVzip / DuoAttention / MiniCache / KeepKV | token-dropping or merging with importance scoring | ❌ (KeepKV/MiniCache: near-lossless merges) | various (see Round-2 dossier) | Drafting-cache candidates; VeriCache paper uses KVzip as its compressor exemplar |

**Design conclusion:** the emerging stack is *compressor (FastDMS/TurboQuant/KVzip-class) + bitwise codec for movement (SplitZip) + verification scheduler (VeriCache pattern) + token-level speculation (MTP/EAGLE)*. Only the last layer is fully productionized on consumer engines today; layers 1–3 are 2026 research code you can clone except the VeriCache scheduler itself.

---

## §4 Windows driver enforcement — exact programmatic mechanics

**The setting:** "CUDA – Sysmem Fallback Policy" lives in NVIDIA's **Driver Settings database (DRS)** — binary blobs at `C:\ProgramData\NVIDIA Corporation\Drs\nvdrsdb0.bin` / `nvdrsdb1.bin`, editable only through **NVAPI DRS** (never hex-edit the blobs). Requires driver **≥ 546.01** (NVIDIA KB 5490).

**Exact identifiers (verified against a working community automation script):**
- `SettingID = 283962569` (decimal) = **`0x10ECECC9`** — "CUDA Sysmem Fallback Policy", `ValueType = Dword`
- `SettingValue = 1` → **Prefer No Sysmem Fallback** (spill disabled ⇒ loud CUDA OOM)
- `SettingValue = 0` → Driver Default (fallback allowed — the silent 4–5 tok/s spill mode)
- Source of truth: FR33THY's `Disable TAA With DLSS.ps1` embeds this exact `<ProfileSetting>` block in an nvidiaProfileInspector profile and applies it headlessly: `Start-Process -wait inspector.exe -ArgumentList "config.nip"`.

**Three automation routes (ranked):**
1. **`.nip` XML + nvidiaProfileInspector CLI import (recommended; zero C code).** Build the UTF-16 XML (`<ArrayOfProfile><Profile><ProfileName>Base Profile</ProfileName>` for global, or a named profile with `<Executeables>llama-server.exe</Executeables>` for per-process scoping), then import via inspector.exe's file-argument mode. **Implemented in `deploy/set_sysmem_fallback.ps1`** (auto-downloads `Orbmu2k/nvidiaProfileInspector` latest release, unblocks it, writes the profile, imports, verifies). Restart of the target app required; no reboot.
2. **Direct NVAPI DRS calls** (for a dependency-free service): `NvAPI_Initialize → NvAPI_DRS_CreateSession → NvAPI_DRS_LoadSettings → NvAPI_DRS_FindProfileByName/CreateProfile (+ SetProfileInfo to attach the executable) → NvAPI_DRS_SetSetting(0x10ECECC9, 1) → NvAPI_DRS_SaveSettings → DeleteSession`, via C# `DllImport("nvapi64.dll")` (function ordinals from the NVAPI headers) or Python `ctypes`. This is exactly what inspector.exe wraps.
3. **Raw registry / DRS blob editing** — fragile, undocumented layout, breaks on driver updates: **do not**.

**Scope, WSL2, and verification:**
- Per-app profile beats global for shared machines (games sometimes *want* fallback); our script supports both.
- **WSL2 has no separate switch** — CUDA in WSL rides the host WDDM stack, so the host-side policy governs; verify from the host.
- Verification protocol (both in `isolate_and_verify.ps1`): (a) counter delta — `Get-Counter '\GPU Process Memory(*)\Shared Usage'` must not move during a 384-token generation; (b) **negative test** — launch an intentionally oversized config; correct behavior is `cudaMalloc failed: out of memory`, incorrect behavior is a slow-but-running server (policy not applied).
- Persistence: DRS settings survive driver upgrades; re-verify after major driver branches anyway (one negative test).

---

## Prototype integration blueprint (synthesis)

```
[8 GiB GPU — everything hot]                 [Host RAM / NVMe — cold, verified]
 Gemma4-12B QAT weights (UD-Q4_K_XL)          KV_full archive (SplitZip-compressed,
 q8_0 hybrid KV (32–64K)         ◄── verify ──   bitwise-exact on restore)
 MTP drafter (242 MiB) ── draft ──► tokens     LMCache-style prefetch if batched
        │        ▲
        └─ token-level speculation (production, today)
           KV-level speculation (VeriCache pattern — build on LMCache, watch upstream)
```
Phase 1 (now, zero new deps): Round-1/2 stack + `set_sysmem_fallback.ps1` + harness T1–T5.
Phase 2 (weeks): SplitZip smoke test on the laptop GPU (WSL2); wire codec into a KV save/restore utility for session persistence within the zero-RAM-decode guarantee (decode never touches host RAM; only cold archives do).
Phase 3 (research bet): single-user VeriCache degenerate prototype on LMCache + vLLM (needs ≥16 GiB dev box), compressor = TurboQuant-fork or KVzip-class; measure x/γ break-even at laptop PCIe bandwidth.

### Repository index (all verified live 2026-10-07)
- `github.com/Intelligent-Microsystems-Lab/SplitZip` (MIT, NeurIPS 2026) — codec + benchmarks
- `github.com/LMCache/LMCache` (dev branch) — swap/prefetch substrate for VeriCache
- `github.com/aiha-lab/TANGRAM` + `aiha-lab.github.io/tangram-page` (arXiv:2606.06302)
- `github.com/shisa-ai/FastDMS` (nano-vllm-based; `shisa-ai/Llama-3.2-1B-DMS-8x` on HF)
- `github.com/LeanModels/DFloat11` (arXiv:2504.11651) · `github.com/zipnn/zipnn` (arXiv:2411.05239)
- `github.com/ggml-org/llama.cpp` — `docs/speculative.md`, discussion #21961 (paged KV), #20969 (TurboQuant)
- `github.com/TheTom/llama-cpp-turboquant` @ `feature/turboquant-kv-cache` (experimental)
- `github.com/Orbmu2k/nvidiaProfileInspector` (DRS automation vehicle) · NVIDIA KB 5490 (policy docs)
- Setting reference in the wild: FR33THY `Disable-Anti-Aliasing/New Easy Method/Disable TAA With DLSS.ps1` (SettingID 283962569, value 1)
