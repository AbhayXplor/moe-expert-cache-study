# Project record — index of everything measured

This is the master index for the "run a bigger model than the card allows"
work. The goal: **the largest model this laptop runs at usable speed with
quality intact**, plus whatever genuine speedups can be found on the way.

Read the documents in this order. Each one is self-contained; the first is the
most recent and the most user-facing.

## Documents

| file | what it is | status |
|---|---|---|
| [EXPERT_CACHE_AND_REAP.md](EXPERT_CACHE_AND_REAP.md) | The expert cache (`--moe-cache-mib`): −64% on the full 35B, **+216% (3.2x) on REAP-48**, the "pool must hold ~1/3 of expert bytes" rule, and the REAP-48 quality verdict. **Start here for the cache.** | current |
| [PAPER_PACK.md](PAPER_PACK.md) | **Everything needed to write a paper later**: the candidate claims with strengths, the protocol, exact model anatomy, every measured table, the mechanism arithmetic, related work with what is *missing* from it, 9 threats to validity, and the exact experiments that would make it publishable. Written so nothing has to be re-run. | current, unused for now |
| [MOE_35B_RESULT.md](MOE_35B_RESULT.md) | The 35B MoE result: speeds, the full configuration table, the two practical wins, and the research pointers. **Start here for the 35B.** | current |
| [FINDINGS.md](FINDINGS.md) | Earlier findings: the machine cost model, the fit ladder, what the "three wastes" really amount to, dead ends. | needs the MoE session folded in |
| [RESULTS.md](RESULTS.md) | Raw earlier results with the measurements behind FINDINGS.md. | historical |
| [REPORT_ollama_and_wastes.md](REPORT_ollama_and_wastes.md) | The Ollama offload-policy discovery and the compression analysis. | historical |
| [../research/litreview_pass1.md](../research/litreview_pass1.md) | First literature pass: MoE offloading, expert caching, compression papers. | historical |

## Scripts that produced the numbers

| script | what it measures |
|---|---|
| [../scripts/moe_bench_35b.py](../scripts/moe_bench_35b.py) | Launches llama-server across configurations with a RAM watchdog, measures decode and prompt processing, records draft acceptance. Takes `--server` to benchmark a specific binary and records the binary + model path in every result. |
| [../scripts/moe_cache_summary.py](../scripts/moe_cache_summary.py) | Regenerates every expert-cache/placement/REAP table from the raw run JSONs, so the quoted numbers can be checked instead of trusted. |
| [../scripts/paper_data.py](../scripts/paper_data.py) | Derives the paper-pack numbers straight from the GGUF tensor index and metadata (expert bytes, per-expert size, bytes/token, coverage) plus the link-bandwidth arithmetic. Output dumped to [PAPER_DATA.txt](PAPER_DATA.txt). |
| [../scripts/verify_doc_numbers.py](../scripts/verify_doc_numbers.py) | Re-checks each number quoted in EXPERT_CACHE_AND_REAP.md against the raw JSON it came from (currently 25/25). Run it after editing that document. |
| [../scripts/moe_imatrix_skew.py](../scripts/moe_imatrix_skew.py) | Per-expert popularity from a llama.cpp imatrix file; the coverage curve and what a cache would buy. |
| [../scripts/offload_bandwidth.py](../scripts/offload_bandwidth.py) | The machine cost model: VRAM read, RAM read, PCIe copy, and how well they overlap. |
| [../scripts/planner.py](../scripts/planner.py) | Offload/KV/context planner, calibrated to 0.9% mean error on measured runs. |
| [../scripts/moe_routing_probe.py](../scripts/moe_routing_probe.py) | Routing skew for a transformers-loadable MoE (used on OLMoE). |
| [../scripts/gguf_meta.py](../scripts/gguf_meta.py) | Model anatomy: KV cost per token, bytes by tensor group, fit table. |
| [../scripts/ollama_bench.py](../scripts/ollama_bench.py) / [ollama_clean.py](../scripts/ollama_clean.py) | Ollama benchmarking, and the mandatory VRAM drain before it. |

### Cleanup log (10 Oct 2026): 13 GB reclaimed

Deleted as provably redundant or dead: the HF-cache copy of OLMoE (byte-identical
to `models/`, 4.7 GB), `DFloat11/Qwen3-4B-DF11` (5.4 GB, a model the DF11 scripts
deliberately avoided using), `experiments/df11_qwen3_1.7b/` (2.4 GB, regenerable),
and the three llama.cpp installer zips (646 MB, already unzipped). Kept: all four
models, the imatrix, both server builds, every `runs/` JSON.

**Side effect to know about:** six scripts are DF11-era and reference the deleted
`experiments/df11_qwen3_1.7b/` output — `entropy_probe.py` in code(line 92), the
rest only in usage docstrings (`compress_df11.py`, `compress_df11_par.py`,
`finish_df11.py`, `bench_infer.py`, `bench_decode_kernel.py`). Their recorded
results in `runs/` are unaffected. To revive any of them, rebuild the input first:
`compress_df11_par.py --base Qwen/Qwen3-1.7B --out experiments/df11_qwen3_1.7b`.
The Qwen3-1.7B base model itself was kept in `.hf_home`.

## Raw artifacts (`runs/`)

| file | contents |
|---|---|
| `moe_bench_moe35b_sweep.json` | The offload-placement sweep: 7 configurations, all experts in RAM through `--n-cpu-moe 28`. |
| `moe_bench_ab_control.json` | The interleaved A/B/A/B control that proved the placement effect was thermal drift. |
| `moe_bench_threads.json` | Thread-count sweep, also interleaved. |
| `moe_bench_longprefill.json` | The `-ub 512` vs `-ub 2048` prompt-processing comparison. |
| `moe_bench_mtp_ab.json` | MTP vs no MTP, interleaved, with draft-acceptance counts and output hashes. |
| `moe_bench_mtp4096.json` | The recommended configuration, verified running. |
| `moe_bench_q35b_nothink.json`, `moe_bench_q9b_nothink.json` | The quality comparison, both models, identical prompts. |
| `moe_bench_cache_ab.json` | The first cache attempt, on the **staged Ollama-derived binary — which does not have the flag**; kept because it proves that (`error: invalid argument: --moe-cache-mib`). |
| `moe_bench_cache_ab2.json` | The 35B cache A/B on official b11537: 22.2 → 8.0 tok/s, interleaved A/B/A/B. **The negative result.** |
| `moe_bench_reap.json` … `moe_bench_reap5.json` | REAP-48 placement and pool sweeps: the 12.6 → 39.6 tok/s curve, the 4096–4608 knee, pool vs static placement, the interleaved 5120 confirmation. |
| `moe_bench_qualreap.json`, `moe_bench_qualfull.json` | REAP-48 vs full 35B on the same 3 prompts, 400 tokens, plus pool-vs-no-pool output comparison (not bit-exact). |
| `moe_bench_probe.json`, `moe_bench_oldbin.json` | `-fa off` vs `-fa on`, `--perf`, and the back-to-back old-build-vs-b11537 control that disproved the suspected 40% build regression. |
| `moe_bench_spec_skim.json` | `ngram-simple-4`, `ngram-mod-8`, `ngram-cache`, `draft-simple` vs no speculation, on the prose + repetitive prompts. **Confounded**: the model's thinking consumed all 200 tokens (`answer_chars=0`), so the repetitive prompt never produced the JSON the drafter needs. Its numbers show cost, not payoff. |
| `moe_bench_spec_repeat.json` | **The valid speculative-decoding comparison.** Thinking disabled (`answer_chars=740` confirms real JSON output), 300 tokens, repetitive prompt: `spec-none` 26.50 tok/s vs `ngram-simple-4` 14.25, `ngram-simple-12` 14.35, `ngram-mod-8` 13.50. **−46 to −49%, even on the task ngram drafting is built for.** `draft-simple` never drafted at all (0/0 in both runs). |
| `moe_imatrix_skew_qwen36_35b_a3b.json` | Per-layer expert popularity for 256 experts, with the budget curve. |
| `moe_routing_probe_longdoc.json`, `moe_routing_probe_mixed.json` | OLMoE routing skew (64 experts). |
| `moe_routing_probe_BUGGY_POOLED_SUPERSEDED.json` | The first, wrong version of that probe (pooled all layers into one histogram). Kept deliberately. |
| `offload_bandwidth.json` | The three prices per byte and the overlap test. |
| `llamaserver.json`, `spec_divergence.json`, `fused_decode_gemv.json`, `compress_df11_qwen3_1.7b.json`, `entropy_probe.json` | Earlier: llama-server configs, speculative divergence, the fused kernel, DF11 compression, quantization entropy. |

### Speculative decoding beyond MTP (11 Oct 2026): a decisive negative

MTP is the **only** speculative decoder worth enabling on this machine.
On the *ideal* case for self-lookup drafting (highly repetitive structured
JSON, thinking off, 300 tokens) every `ngram-*` variant is **~2x slower** than
plain decoding:

| spec type | decode | draft acceptance |
|---|---|---|
| `none` (baseline) | **26.50 tok/s** | — |
| `ngram-simple` n=4 | 14.25 (−46%) | 20% |
| `ngram-simple` n=12 | 14.35 (−46%) | 11% |
| `ngram-mod` m=8 | 13.50 (−49%) | 36% |
| `draft-simple` | ~20–25 (−20%) | never drafted (0/0) |
| **`draft-mtp`** (recorded earlier) | **+18–20%** | **73–75%** |

**Why:** ngram drafting proposes continuations by matching the current n-gram
against text already in the context. Our JSON values differ every row, so
acceptance lands at 11–36%. On a normal GPU that might still break even — but
here every verification step pays the full CPU-side expert offload cost, so
~⅔ of verifications are wasted work *added on top* of an already CPU-bound
decode. MTP wins because its drafts come from a trained head and hit 73–75%.

**Rule:** on an expert-offloaded MoE, speculation only pays above roughly 70%
acceptance. Anything below is a net loss, and the loss is brutal because the
baseline is already bandwidth/compute constrained.

Harness changes that made this measurable: `moe_bench_35b.py` gained
`--prompts` (run one prompt in isolation — ngram tests are meaningless on
novel prose), a `repeat` prompt (repetitive JSON, the only shape a self-lookup
drafter can fire on), and the `spec-none` / `ngram-*` / `draft-simple` configs.
Flag names verified against b11537 `--help` before use.

## Models on disk

**Do not delete any of these to "free up space."** `models/` totals 64 GB, but the
drive has **420 GB free (55% used)** — there is no pressure, and every row below is
load-bearing for a recorded result. If space is ever genuinely needed, delete
`experiments/logs/` (3.3 MB, regenerable) first; models are the last resort.

| path | size | why | what breaks if deleted |
|---|---|---|---|
| `models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22.13 GB | The main testbed: 256 experts/layer, top-8, 40 layers. | **Every baseline number in `MOE_35B_RESULT.md` and the cache regression in `EXPERT_CACHE_AND_REAP.md`** — this exact file is what they were measured on. Re-download only. |
| `models/Qwen3.6-35B-A3B-GGUF/imatrix_unsloth.gguf_file` | 192 MB | Per-expert selection counts; the skew analysis. | The Gini/coverage skew numbers and the popularity curve. |
| `models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22.66 GB | Same model with the multi-token-prediction head. **This is the one to run.** | The MTP win (+18–20%) — the fastest working configuration. Not derivable from the non-MTP file. |
| `models/Qwen3.6-35B-A3B-REAP-48-v2-GGUF/Qwen3.6-35B-A3B-REAP-48-v2.gguf` | 9.43 GB | Router-pruned 48%: **19.17 B params, 133 experts, 3.93 bpw**. | The **only** model the expert cache actually wins on (12.6 → 39.6 tok/s), so the entire positive half of the ~⅓-of-expert-bytes rule, plus the REAP-vs-full quality comparison. |
| `models/OLMoE-1B-7B-0924-Instruct/` | 13.84 GB | 64-expert MoE, the small testbed. | The 64-expert routing-skew comparison and `moe_routing_probe.py`'s default target. |
| `.hf_home/hub/models--Qwen--Qwen3-1.7B` | 3.8 GB | Base model for the DF11 rebuild path. | The documented DF11 revival command (`compress_df11_par.py --base Qwen/Qwen3-1.7B`). |

### Storage audit (11 Oct 2026): nothing to reclaim

A second cleanup pass found the workspace already clean: `models/` 64 GB (all
accounted for above), `.runtime/` 2.4 GB (both llama-server builds, both needed),
`experiments/logs/` 3.3 MB, no stray archives, no leftover installer zips. The
only junk was `scripts/__pycache__`, removed. **Disk is not a constraint here —
reproducibility is.** Keep the models.

## Binaries (in `.runtime/`)

| path | what it is |
|---|---|
| `.runtime/llamaserver/llama-server.exe` | Staged Ollama-derived build (0.5.0-dev, commit 6f767fe96). Known-good GPU reference, MTP works, **no MoE cache flags**. |
| `.runtime/official/cuda/llama-server.exe` | Official llama.cpp **b11537** prebuilt (CUDA 12.4). Has `--moe-cache-mib`, loads `qwen35moe`. **Required for the cache tests.** |
| `.runtime/official/` | The same b11537 CPU build, unzipped flat. |

## The result in one paragraph

A 35B Mixture-of-Experts model runs on this 8 GB laptop at **34.6 tok/s on a
cool machine and 23–26 tok/s warm**, using only **3.2 GB of VRAM** and peaking at
**46 GB of 64 GB RAM**. Two real speedups were found and verified: raising the
prompt batch to 4096 gives **2.8x on prompt processing** (211 → 590 tok/s, output
byte-identical), and enabling MTP gives **+18–20% on generation** (73–75% of its
drafted tokens accepted, though not bit-exact).

**Update — the expert cache changes the picture.** Mainline llama.cpp now has
`--moe-cache-mib N`, a VRAM pool for hot experts. On the **full 35B it is a
regression (−64%, 22.2 → 8.0 tok/s)** because 8 GB cannot hold a useful fraction
of 19.6 GB of expert weights — but on the pruned **REAP-48 model (19.17 B, 9.4 GB,
133 experts) it gives 12.6 → 39.6 tok/s (+216%, 3.2x)**, and it beats static layer
placement while using 1.4 GB less VRAM. The rule that predicts both: **only
enable it when the pool can hold ~⅓ or more of the expert bytes** (11–22% on the
35B = loss, 43–71% on REAP = win). See
[EXPERT_CACHE_AND_REAP.md](EXPERT_CACHE_AND_REAP.md). REAP-48 is measurably
weaker in quality than the full 35B, so it is a speed option, not a replacement.
The earlier "every weight-placement idea measured as no change" conclusion is
still right **for the 35B on this card**: it is not memory-bandwidth-bound (it
moves ~110 GB/s against ~240 available, while the 9B it replaces runs at ~227
GB/s and *is* saturated), and its limit is CPU-side expert math — which is
exactly what a big-enough expert pool removes on REAP, and what a too-small pool
merely adds overhead to. The remaining unbuilt lever is still the one nobody has
packaged: splitting experts at the bit level (RapidMoE) so they can be computed
on the card without being stored there.
