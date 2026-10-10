# Literature review, pass 1 — compressed-resident LLM inference (Oct 2026)

Status: **reading only, nothing reproduced yet.** Every number below that is
labelled "ours" was measured on this machine; everything else is a claim from
the cited paper that has not yet been verified by us.

## Headline

The exact problem we identified from our own benchmarks — *lossless compressed
weights can be resident in VRAM, but the shipped implementations lose because
they decode into a global-memory scratch buffer* — **has already been diagnosed,
solved, published, and open-sourced by two independent 2026 groups.** One of
them (ZipServ, ASPLOS '26) reports being **faster than cuBLAS** while 30% smaller
and bit-exact, and explicitly targets our hardware class.

## The two key papers

### ZipServ — ASPLOS '26 (arXiv 2603.17435, Mar 2026, HKUST-GZ)
`https://github.com/HPMLL/ZipServ_ASPLOS26` (Apache-2.0)

Diagnosis, quoted: existing lossless inference is slow because of (1) *"variable-
length bitstreams produced by traditional entropy codecs break SIMT parallelism"*
and (2) *"decoupled pipelines lead to redundant memory traffic"* — they call the
DF11-style workflow **"decompress-store-compute"**. That is precisely what our
bench measured as the 2.20× byte blow-up (read streams → write bf16 scratch →
read scratch back).

Their two fixes map one-to-one onto the two things our data pointed at:

| our finding | their mechanism |
|---|---|
| Huffman is variable-length → branchy, divergent | **TCA-TBE**: fixed-length, bitmap-based exponent format, *constant-time branch-free decoding* |
| scratch round-trip = 5.65 GB/token of waste | **ZipGEMM**: decode on-the-fly *directly into Tensor Core registers*; no global buffer |

Claims: **up to 30% size reduction, up to 2.21× kernel speedup over cuBLAS**,
1.22× average end-to-end over vLLM, **up to 5.53× over DFloat11**. Evaluated on
**RTX 4090, L40S, RTX 5090**.

Repo requirements: CUDA ≥ 11.8, **compute capability ≥ 8.0** (ours is 8.9 ✓),
and the README explicitly states it is *"optimized for GDDR memory-based GPUs
(RTX 4090, L40S, RTX 5090, RTX 6000 Pro) where memory bandwidth is the primary
bottleneck. Performance gains may vary on HBM-based datacenter GPUs."*
**Our 4060 Laptop is GDDR and we measured it to be bandwidth-bound — this is the
target profile.** Build is plain `make` + a `test_mm M K N SplitK` benchmark, so
the kernel-level claim is checkable without any model integration.

### Approaching Shannon Bound with Lossless LLM Weight Compression
(arXiv 2606.15789, Jun 2026 — NUS / HUST / ETH: Tan, Chen, Alonso, Wong, He)

Entropy study across 1.5B–405B and bf16 / int4 / AWQ / SQ8. Key numbers:
* **"up to 27% of the stored bits are redundant"** in bf16.
* bf16 effective entropy ≈ 10.6 bits/weight ⇒ **~1.51× lossless ratio**.
* Uses **ANS** (not Huffman) because it is the only entropy coder that is
  simultaneously near-Shannon, **tile-addressable (random access)**, and
  GPU-parallelisable.
* Same system diagnosis as us: *"decompress-store-compute… requires additional
  memory capacity to hold the decompressed weights, generates redundant
  global-memory traffic"*; their fix is to decode **into shared memory** as the
  GEMM microkernel consumes each tile, with no intermediate global traffic.
* **The strategically important claim:** lossless is *"orthogonal to existing
  lossy compression techniques and can be applied on top of well-tuned quantized
  models"* — including int4/AWQ. In quantized formats the headroom is *larger*
  than in bf16 (they claim up to 5×).
* Results are batch-throughput shaped (SGLang; Qwen-14B batch 47→75;
  Mixtral-176B batch 20→95; up to 11× over DF11/NeuZip), **not batch-1 latency**.

## Our measurements independently reproduce their headline numbers

This is the most important cross-check in this memo. Five for five:

| quantity | ours (measured) | theirs (published) |
|---|---|---|
| bf16 exponent-field entropy | **2.6436 bits** | 2.57–2.74 bits |
| theoretical lossless floor | **10.58 bits/weight** → 1.51× | "16/10.6" → 1.51× |
| overall bf16 redundancy | **1.356× overall, 1.472× projections** | "up to 27% of stored bits" |
| why DF11 is slow | read streams + write 16-bit scratch + re-read = **2.20× the bytes** | "decompress-store-compute", "redundant memory traffic" |
| what the fix must be | **branch-free fixed-rate code + no materialisation** | TCA-TBE + ZipGEMM; ANS + shared-memory decode |

We derived the right conclusion from scratch, and landed on the same numbers as
two peer-reviewed 2026 papers. Our testbed is therefore trustworthy for judging
these methods — which matters, because we will need to judge more of them.

## Other 2026 work worth tracking

* **SplitZip** (arXiv 2605.01708, May 2026) — *"Ultra Fast Lossless KV
  Compression"*: 613 GB/s compression, 2182 GB/s decompression throughput. The
  KV-cache thread.
* **Invariant Bit Packing (IBP)** — Kamath & Peter (UW), SIGARCH blog Jul 2026.
  Data-type agnostic, does runtime compression *and* decompression so it covers
  the **KV cache**; 24% faster than FlexGen/InfiniGen by decompressing
  **on-the-fly while transferring from CPU** — i.e. the overlapped-offload
  property that keeps PCIe streaming from stalling. Directly relevant to the
  "if RAM is used it must not cost speed" constraint.
* **FLUTE** (Google, Mar 2026) — lookup-table quantization, 2–4× faster GEMM
  kernels, 1.5–2× end-to-end throughput. Weights stored as indices into a LUT =
  compressed-resident with fused lookup.
* **Marlin** (IST-DASLab) / **Machete** — the production fused FP16×INT4 kernels
  behind GPTQ/AWQ serving. The known-good 4-bit baseline path.
* **LiquidGEMM** (ACM 2025) — W4A8 fused GEMM kernel.
* **GLOWQ** (ICLR 2026) — group-shared low-rank approximation on quantized LLMs;
  caches one shared right factor per sharing group. Ties the sharing axis to the
  quantization axis.
* **"Accelerating Block Low-Rank Foundation Model Inference"** (ACM 2026) —
  stores `W = VU` and computes without materializing `W`. The low-rank branch's
  cleanest statement of our thesis.
* **TileFuse** (arXiv 2606.11357, Jun 2026) — same fusion principle on AMD XDNA2
  NPUs: *"weights remain in compressed INT4/INT8 format until they reach the
  compute core, where unpacking, dequantization, and matrix computation are
  performed within the same kernel flow. By avoiding separate BF16 weight
  materialization…"*
* **Unweight** (Cloudflare, 2026) — lossless MLP weight compression for dense
  inference; reported **15–22%** weight reduction, bit-exact.
* **ZipNN** (IBM), **LMC**, **NeuZip**, **DietGPU** — earlier lossless codecs for
  checkpoints/training/comm. NeuZip and DietGPU target *training* traffic.
* **NVIDIA B200** ships a hardware decompression engine (LZ4/Snappy/Deflate) that
  decompresses PCIe-transferred data **without using SMs** — a pointer to the
  architecture direction; not available on consumer parts.
* **"Quantized Kernels: Why a 4-Bit Model Is Not Automatically Fast"** (Jun 2026)
  — the Thread-B primer.
* **Hybrid JIT–CUDA Graph Optimization** (arXiv 2604.23467, Apr 2026) — launch
  overhead reduction; relevant to our 1644-launches-per-step wall.

## Reframing for the actual goal: a 9–10B dense model on 8 GB

The constraints are: dense (no MoE), VRAM-resident, ≥4-bit quality, no 1-bit
speed collapse, and speed must survive.

The literature says the winning combination is:

1. **Start from a 4-bit (or AWQ/GPTQ) model** — the accepted baseline; a 10B at
   ~4.25 bits ≈ 5.3 GB, which fits.
2. **Layer lossless compression on top of the quantized codes.** This is the
   genuinely untapped part for a consumer target. Our own E4b measured **21.9%
   for uniform int4** (3.1244 bits of entropy in a 4-bit code) and the NUS paper
   reports more headroom in quantized formats than in bf16. Lossless-on-quantized
   is *bit-exact with respect to the 4-bit model*, so it adds **zero** quality
   loss on top of what 4-bit already costs.
3. **Fuse the decode into the GEMM** so the extra compression becomes a speedup
   rather than a slowdown (ZipServe: fixed-length code + decode into registers).

That is: ~3.2 effective bits/weight on a 10B ⇒ ~4 GB resident, bit-exact vs the
4-bit model, and (per the fusion papers) potentially *faster* than the
uncompressed 4-bit path.

## Two things this literature does NOT solve for us

1. **Launch overhead.** Our measured wall of 1644 kernel launches/step
   (~160 ms/token of CPU enqueue) caps throughput at ~6 tok/s regardless of how
   small the weights are. ZipGEMM is a kernel-level win and does not address it.
   Note llama.cpp reportedly does 40+ tok/s on 7–8B Q4 on an 8 GB card, so most
   of that gap is *our runtime*, not the hardware.
2. **Batch-1 decode is GEMV, not GEMM.** ZipServ's headline 2.21× is a GEMM
   framing; whether it holds for batch-1 GEMV shapes on a 24-SM, 75 W laptop
   part is unverified.

## Toolchain reality on this box

**Resolved — checked, not assumed.** The Windows CUDA install is
*compiler-less*: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\`
contains `bin/ extras/ include/ lib/` but **no `nvcc.exe` anywhere in Program
Files**, and there is **no MSVC `cl.exe`** either. WSL Ubuntu is present and
running but is bare — **no `gcc`, `make`, or `nvcc`**. So both native build routes
(ZipServ's `source Init.sh && make`) would require installing a compiler and a
CUDA toolkit, i.e. administrator access inside Windows or inside WSL.

| | status |
|---|---|
| `nvcc` | **absent** everywhere searched (Windows toolkit dir is runtime-only; WSL has none) |
| `cl.exe` (MSVC) | **absent** |
| `g++` / `make` | absent in Git Bash; absent in WSL |
| WSL Ubuntu | installed and running, but bare |
| cupy 14.2.0 | works for launching **prebuilt PTX** (`cp.RawModule(path=...)`) — this is how DF11's decode kernel and our microbenchmarks run — but `RawKernel` from CUDA C **source fails**: `CUDA_ERROR_INVALID_IMAGE` (NVRTC path) |
| **Triton 3.8.0** | **installed and verified working** — a real kernel compiled and ran on the 4060 with max error 0.0, no admin needed |

**Consequence: route (c) — reimplement the fused decompression-GEMM ourselves in
Triton — is the available path, and it needs no privileges at all.** Launching
ZipServ's own `make`-based C++ benchmark would require an elevated install of a
host compiler plus the CUDA toolkit (in Windows or in WSL), so it is deferred
rather than assumed. Triton is a good fit anyway: there is a directly relevant
2026 reference implementation of a fused W4A16 dequant-GEMM in Triton
("Beating FP16 with 4-bit Weights: A Portable W4A16 GEMM", Jul 2026), so the
portable-kernel design is already documented.

## Next actions, in priority order

1. Verify the kernel-level claim on the 4060 for **both GEMM and batch-1 GEMV
   shapes**. Triton is the path we can actually take (see toolchain above);
   running ZipServ's own `make` benchmark is blocked on an elevated compiler
   install.
2. If it holds, reproduce the fused kernel against our existing baselines
   (bf16 3.20 GiB / 28.72 ms, DF11 2.36 GiB / 66.68 ms, NF4 1.24 GiB / 25.65 ms
   per token) so all numbers are apples-to-apples.
3. Test lossless-on-4-bit: apply their code to a quantized Qwen3-1.7B and check
   the measured saving against our predicted ~3.12 bits/weight.
4. Attack the launch-overhead wall separately (CUDA graphs or a leaner runtime).
