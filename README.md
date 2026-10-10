# MoE expert-weight caching on consumer hardware

A controlled study of llama.cpp's expert-weight cache (`--moe-cache-mib`) on a
single 8 GB laptop, plus the benchmark instrument used to measure it.

Every number in this repository was produced on the machine described below and
is reproducible from the raw artifacts in [`experiments/runs/`](experiments/runs).
No number here is quoted from another source without saying so.

**Status:** complete as a community report. Not yet publishable as a paper. The
gap and the exact experiments that would close it are in
[`experiments/PAPER_PACK.md`](experiments/PAPER_PACK.md) section 11.

---

## The result in one paragraph

Mainline llama.cpp now ships a VRAM cache that keeps frequently used MoE expert
weights on the GPU instead of re-reading them from host memory every token.
Published reports of it are uniformly positive (+26%, +84%, roughly 2x). On this
machine it is **strongly negative on one model and strongly positive on another**,
and the sign is predicted by one quantity: **pool size divided by total expert
weight bytes**. Below roughly one third, the cache is a large regression. Above
roughly one third, it is a large win, and it beats static layer placement while
using less VRAM. The two models tested bracket the break-even from either side.

| model | expert bytes | pool | coverage | decode, no pool | decode, with pool | change |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (256 experts) | 19.569 GB | 2048 MiB | 11.0% | 22.02 tok/s | **8.03** | **-64%** |
| Qwen3.6-35B-A3B | 19.569 GB | 4096 MiB | 21.9% | 22.29 tok/s | **11.91** | **-46%** |
| REAP-48 (133 experts) | 7.552 GB | 3072 MiB | 42.7% | 12.68 tok/s | **25.67** | **+104%** |
| REAP-48 | 7.552 GB | 4096 MiB | 56.9% | 12.56 tok/s | **33.52** | **+167%** |
| REAP-48 | 7.552 GB | 5120 MiB | 71.1% | 12.56 tok/s | **39.65** | **+216%** |

All rows are interleaved A/B/A/B in a single session. See
[`experiments/EXPERT_CACHE_AND_REAP.md`](experiments/EXPERT_CACHE_AND_REAP.md)
for the per-run tables.

### The rule

**Enable the expert cache only when the pool can hold about one third or more of
the model's expert weight bytes.**

Two independent community results point at the same region: an offline LFU
simulation needed 88 of 256 experts per layer (34%) resident to reach a 98.8%
hit rate, and the upstream RFC states that a pool the size of top-k is the worst
case (all misses, so you pay the bookkeeping and gain nothing).

Structural consequence: for this 35B model on an 8 GB card, one third of its
expert bytes is about 6.5 GB, on top of the roughly 3.2 GB the model already
needs. **No pool size makes the full model faster on this machine.** The
technique needs a model whose experts are narrow enough per layer.

### A second finding: the cache beats static placement

Pool of 4096 MiB on REAP-48 ran at **32.82 / 33.62 tok/s using 6615 MiB of VRAM**,
while pinning 26 layers of experts statically (`--n-cpu-moe 14`) ran at
**27.45 / 29.06 tok/s using 7303 MiB**. The adaptive pool wins by about 17% while
using 688 MiB less VRAM. Combining the two is worse than either alone, because
they compete for the same memory.

---

## The other result: a 35B model on 8 GB, at usable speed

Separate from the cache study, this repository also records how the same 35B
model is run day to day. Measured on the locally staged build, 32K context:

| quantity | value |
|---|---|
| decode, cool machine | 34.6 tok/s |
| decode, warm | 23 to 26 tok/s |
| VRAM used | 3203 MiB of 8188 |
| peak system RAM | 46 GiB of 63.7 |
| model file | 22.13 GB (256 experts per layer, top-8, 40 layers) |

Two speedups were found and verified:

1. **Prompt batch size: 211 to 590 tok/s (2.8x), output byte-identical.** The
   micro-batch must clear `GGML_OP_OFFLOAD_MIN_BATCH`, which is
   `32 * total_experts / active_experts` (1024 tokens for a 256-expert top-8 MoE).
   Below that, prompt processing never engages the GPU.
2. **MTP (multi-token prediction): +18 to +20% on generation**, with 73 to 75% of
   drafted tokens accepted. Not bit-exact.

A third lever was tested and rejected:

3. **ngram speculative decoding is a loss here.** On the best case for it
   (repetitive structured output, thinking disabled), `ngram-simple` and
   `ngram-mod` measured 13.50 to 14.35 tok/s against a 26.50 tok/s baseline, a
   **46 to 49% regression**, at 11 to 36% draft acceptance. `draft-simple` never
   drafted at all. The rule this yields: on an expert-offloaded MoE, speculation
   only pays above roughly 70% acceptance, which only MTP reaches.

---

## Hardware and environment

| item | value |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop, 8188 MiB, sm_89, 24 SMs |
| Driver | 566.14 (CUDA 12.7), WDDM |
| PCIe, measured | **2.80 GB/s actual**, negotiated Gen3 x4 while the chip advertises `pcie.link.gen.max=4` and `width.max=8`. **Unexplained, and it gates every weight-streaming technique.** |
| VRAM bandwidth, measured | about 240 GB/s |
| RAM | 2 x 32 GB DDR5-4800, 63.71 GiB total, 25 to 28 GiB baseline in use |
| OS | Windows, no Developer Mode, no C or C++ compiler, no CUDA toolkit |
| Session drift | **10 to 20%** between sessions on identical configurations |

The drift figure is the reason every comparison in this repository is interleaved
A/B/A/B within a single session. Absolute numbers carried across sessions are
plus or minus 20%. A cross-build "42% regression" was reported early in this work
and then disproved by a back-to-back control. It is kept in the artifacts as a
record of that mistake.

---

## Repository layout

| path | contents |
|---|---|
| [`experiments/PAPER_PACK.md`](experiments/PAPER_PACK.md) | Self-contained record written so a paper can be drafted without re-running anything: claims, protocol, anatomy, all measured tables, mechanism arithmetic, related work with the gap identified, threats to validity, and the experiments that would make it paper-grade. |
| [`experiments/EXPERT_CACHE_AND_REAP.md`](experiments/EXPERT_CACHE_AND_REAP.md) | The user-facing write-up of the cache study: full cache and placement tables, the byte-fraction rule, the quality verdict, caveats, reproduction commands. |
| [`experiments/MOE_35B_RESULT.md`](experiments/MOE_35B_RESULT.md) | The 35B-on-8-GB record: speeds, configuration table, the two verified wins, and research pointers. |
| [`experiments/INDEX.md`](experiments/INDEX.md) | Master index. Start here for orientation. |
| [`experiments/FINDINGS.md`](experiments/FINDINGS.md), [`RESULTS.md`](experiments/RESULTS.md) | Earlier findings and the raw measurements behind them. |
| [`experiments/runs/`](experiments/runs) | 36 raw run artifacts (JSON). Every table in the write-ups is generated from these. |
| [`experiments/PAPER_DATA.txt`](experiments/PAPER_DATA.txt) | Machine-generated dump of model anatomy, coverage, and the link-bandwidth arithmetic. |
| [`experiments/quality_reap_vs_full.txt`](experiments/quality_reap_vs_full.txt) | Full generated answers for the pruned versus full model comparison. |
| [`scripts/`](scripts) | The instruments. See below. |
| [`research/litreview_pass1.md`](research/litreview_pass1.md) | First literature pass. |
| [`research-agent/`](research-agent) | Earlier investigation track (DFloat11 compression, entropy probes, deployment scripts). Historical; its results are in that directory. |

### Instruments

| script | what it does |
|---|---|
| `moe_bench_35b.py` | The harness. Launches `llama-server` across named configurations with a **RAM watchdog** (polls system memory, kills the server past a ceiling so a bad configuration cannot take the machine down), measures decode and prompt processing, records draft acceptance, and writes provenance (server binary, model path, flags, context, VRAM, peak RAM, output hashes) into every result entry. Flags: `--server`, `--model`, `--configs`, `--prompts`, `--extra-body`. |
| `moe_cache_summary.py` | Regenerates every cache, placement, and quality table from the raw run JSONs, so quoted numbers can be checked instead of trusted. |
| `verify_doc_numbers.py` | Re-reads each number quoted in `EXPERT_CACHE_AND_REAP.md` from the artifact it came from and fails on mismatch. Currently 25/25, plus 5/5 pack ratios and 7/7 coverage figures. Run it after editing that document. |
| `paper_data.py` | Derives the paper-pack numbers from the GGUF tensor index and metadata: expert bytes, per-expert size, bytes per token, coverage, plus link-bandwidth arithmetic. |
| `gguf_meta.py` | Model anatomy: KV cost per token, bytes by tensor group, fit table. |
| `moe_imatrix_skew.py` | Per-expert popularity from an imatrix file, with the coverage curve and what a cache would buy. |
| `offload_bandwidth.py` | The machine cost model: VRAM read, RAM read, PCIe copy, and how well they overlap. |
| `planner.py` | Offload, KV, and context planner, calibrated to 0.9% mean error on measured runs. |
| `moe_routing_probe.py` | Routing skew for a transformers-loadable MoE. |
| `ollama_bench.py`, `ollama_clean.py` | Ollama benchmarking, and the mandatory VRAM drain that must precede it. |

---

## Reproducing

The environment expects a Python virtual environment at `.venv` with `torch`,
`transformers`, and `huggingface_hub`. Set `HF_HOME` to a local directory and
`HF_HUB_OFFLINE=1` if you already have the weights.

```bash
# every derived number in the paper pack (anatomy, coverage, arithmetic)
HF_HOME=$PWD/.hf_home HF_HUB_OFFLINE=1 .venv/Scripts/python.exe scripts/paper_data.py

# the consolidated sweep tables, regenerated from experiments/runs/
.venv/Scripts/python.exe scripts/moe_cache_summary.py

# verify the numbers quoted in the write-up against their source artifacts
.venv/Scripts/python.exe scripts/verify_doc_numbers.py

# one interleaved A/B of the cache on the full model
.venv/Scripts/python.exe scripts/moe_bench_35b.py \
  --server .runtime/official/cuda/llama-server.exe \
  --model models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --configs cpu-moe cache-2048 cpu-moe cache-2048 --reps 2 \
  --n-predict 192 --tag cache_ab2 --num-ctx 32768 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

Note on `--extra-body`: this model is a reasoning model, and with thinking enabled
it will spend an entire short generation inside its reasoning block, producing an
empty answer. That silently invalidated one pass of the speculative-decoding
comparison in this repository before it was caught. Disable thinking when the
measurement depends on the model producing particular output.

### Model files

Weights are not committed (they total 64 GB). Download them to these paths:

| path | source | notes |
|---|---|---|
| `models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `unsloth/Qwen3.6-35B-A3B-GGUF` | Main testbed. 256 experts per layer, top-8, 40 layers. Every baseline number was measured on this exact file. |
| `models/Qwen3.6-35B-A3B-GGUF/imatrix_unsloth.gguf_file` | same repository | Per-expert selection counts, used for the skew analysis. |
| `models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | Same model with the multi-token prediction head. Required for the MTP result. |
| `models/Qwen3.6-35B-A3B-REAP-48-v2-GGUF/Qwen3.6-35B-A3B-REAP-48-v2.gguf` | `crucible-labs/Qwen3.6-35B-A3B-REAP-48-v2-GGUF` | Router-pruned to 133 experts, 19.17 B parameters, 3.93 bits per weight. Community upload; the only model on which the cache wins here. |
| `models/OLMoE-1B-7B-0924-Instruct/` | `allenai/OLMoE-1B-7B-0924-Instruct` | 64-expert comparison testbed. |

### llama.cpp builds

| path | build | why it matters |
|---|---|---|
| `.runtime/official/cuda/llama-server.exe` | official prebuilt **b11537**, CUDA 12.4 | Has `--moe-cache-mib`, `--spec-type`, and `--load-mode`. **Required for the cache results.** |
| `.runtime/llamaserver/llama-server.exe` | Ollama-derived 0.5.0-dev (commit `6f767fe96`) | Known-good GPU reference. MTP works. Has no MoE cache flags, which is itself recorded as a result (`experiments/runs/moe_bench_cache_ab.json`). |

---

## What is measured, and what is only inferred

Stated plainly, because the distinction matters:

**Measured.** Decode and prefill speeds, VRAM and RAM usage, draft acceptance
counts, output hashes, model anatomy from the GGUF tensor index, PCIe and VRAM
bandwidth, cache size to speed on both models, pool versus static placement.

**Inferred, not measured.**

- **Hit rate.** The binary exposes no cache metric. Coverage is computed from pool
  size divided by expert bytes; the miss fraction `f` is back-solved from speed
  plus arithmetic. There is no counter behind it.
- **The mechanism.** The arithmetic is consistent with the cache miss cost being
  one synchronous host-to-device upload over the measured 2.80 GB/s link, and it
  explains both the negative result here and the positive results reported on
  machines with a faster link. It is a hypothesis, not a profile. There is no
  per-layer timing and no upload byte counter.
- **The break-even location.** Bracketed between 21.9% and 42.7% coverage. Not
  located.

**Also important:** the cache changes model output. A cache hit computes the
expert on the GPU and a miss computes it on the CPU, giving different accumulation
order and occasional argmax flips on near-ties. All three test prompts diverged
from the no-cache arm at temperature 0 with a fixed seed. Leave `--moe-cache-mib 0`
if you need byte-reproducible output.

---

## Threats to validity

Full list in [`experiments/PAPER_PACK.md`](experiments/PAPER_PACK.md) section 10.
The ones that most constrain the claims:

- **One machine, one GPU, one OS.** Nothing here establishes behavior on Linux, on
  a larger card, or with genuinely asynchronous uploads.
- **Session drift of 10 to 20%.** Interleaving handles this within a session;
  cross-session absolute numbers are soft.
- **The model comparison is confounded.** The two models differ in coverage (133
  versus 256 experts) *and* in quantization (IQ3_S versus a Q4_K/Q5_K/Q8_0 mix),
  so part of the pruned model's advantage could be cheaper CPU-side dequantization
  rather than coverage.
- **Quality evidence is 3 prompts, 1 sample per model.** Evidence, not a
  benchmark.
- **One llama.cpp build** for the cache tests, with an undocumented eviction policy.

---

## Open items

These are known, unresolved, and listed here so they are not mistaken for
settled questions:

1. The cache hit rate is unobservable with the current binary.
2. The mechanism is inferred rather than profiled.
3. The measured PCIe speed is 5 to 9 times below what the chip reports it can do.
   Resolving this decides whether any weight-streaming technique can apply to
   this machine at all.
4. `--load-mode none` has not been tested against the current mmap setup, despite
   the server printing a warning about CPU tensor overrides with mmap on every
   load.
5. The MTP draft length was fixed at 2 early and never swept.

---

## Citing

If this work is useful, cite the repository and the specific document you used,
since the documents have different scopes:

> MoE expert-weight caching on consumer hardware: a controlled study of
> llama.cpp `--moe-cache-mib` on an 8 GB laptop, with a break-even rule.
> https://github.com/AbhayXplor/moe-expert-cache-study

---

## License

MIT. See [LICENSE](LICENSE).
