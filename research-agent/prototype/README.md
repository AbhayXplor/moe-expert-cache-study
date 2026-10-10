# prototype/ — Round-5 build-and-test plan (weights compression stack)

Companion to `../research/round5_validation_prototype.md`. Everything here is
designed to be run by one developer on consumer NVIDIA hardware, with explicit
GO/NO-GO gates so we never build kernels on faith.

## The one-sentence strategy
> Take an off-the-shelf quantized model (AWQ/GPTQ int4), MEASURE its residual
> lossless redundancy (Stage B), and only if the measurement pays (≥1.15–1.2×)
> build the entropy layer — storage-side first (free wins), tile-fused kernel
> second (the novel artifact). In parallel, replicate LegoLM's data-free global
> codebook + outlier sidecar (Stage C-alt) as the redundancy-removal bet.

## Stage A — DFloat11 validation (½ day, 1× CUDA-12 GPU)
```bash
pip install "dfloat11[cuda12]" torch transformers accelerate huggingface_hub
python stageA_dfloat11_validate.py \
    --model DFloat11/<pick-from-hf-org>-DF11 --base <bf16-original-id> \
    --gen-tokens 128 --identity-check --json stageA_report.json
```
Proves on our hardware: ~1.4× size cut, bit-identical greedy output, decode
tok/s delta vs BF16, peak VRAM. Also validates the measurement harness itself.
**Gate:** if identity-check diverges or speed regresses >10% on decode, the
"lossless entropy layer is speed-safe" premise needs re-examination before B/C.

## Stage B — entropy probe (½ day, CPU-only, DECISIVE)
```bash
pip install numpy safetensors torch huggingface_hub   # torch optional (bf16)
python stageB_entropy_probe.py --selftest             # verify probe math
python stageB_entropy_probe.py --model <bf16-model>            --json b_bf16.json
python stageB_entropy_probe.py --model <AWQ-int4-model>        --json b_awq.json
python stageB_entropy_probe.py --model <GPTQ-int4-model>       --json b_gptq.json
python stageB_entropy_probe.py --model <GGUF-source-safetensors-or-any> --verbose
```
Replicates ISCA'26 §II-B per-tensor entropy on our actual candidate models:
nibble-symbol entropy of packed int4 (`qweight`/`qzeros`), exponent-field
entropy of BF16/FP16 and of fp16 `scales`, Huffman achievable bits, optional
real-zstd ratio (`pip install zstandard`).
**Gate (from the validation report):**
- AWQ/GPTQ nibble ratio ≥ 1.15× → GO to Stage C (storage-side packer)
- ratio ≥ 1.25× → GO to Stage C2 (tile-fused decode kernel worth the effort)
- < 1.15× → STOP: entropy layer is dead for that format; pivot fully to C-alt
- sanity: BF16 models should land ~1.4–1.5× (DFloat11 agreement check)
Expectation set by the ISCA paper's own per-format table: deployed AWQ/SQ8
≈1.1–1.3×; raw int4 nibbles can be much higher — measure, don't trust.

## Stage C — storage-side entropy packer (1 week, no kernels)
Build `packq`: per-tensor codebooks (exponent Huffman / nibble FSE via
`zstandard` or `constriction` pip), repack AWQ/GPTQ safetensors into
`*.packed.safetensors` + manifest; loader unpacks to standard tensors at load.
Wins immediately: disk, download, checkpoint-transfer; VRAM parity after load
(this stage does NOT claim runtime-compressed inference — that's C2).
Blueprints: ZipNN's format layout; DFloat11's per-tensor Huffman tables.

## Stage C2 — tile-fused decode-only kernel (2–4 weeks, the novel artifact)
Triton GEMV (decode path first, batch=1) that consumes packed streams directly:
per-tile bitstream offsets (ISCA'26 tile-addressability insight), decode
nibbles/exponents in registers/shared memory, dequant via LUT (SqueezeLLM
kernel pattern), no global-memory materialization.
Compare: tok/s vs Marlin int4 baseline, peak VRAM (compressed-resident),
identical greedy outputs. **This is where the publishable/ownable IP is.**

## Stage C-alt — LegoLM replication (1 week, data-free, parallel bet)
`legolm.py`: global k-means (K=64) over all weights except first/last blocks;
scalar-block normalization; keep top-1% |w| verbatim in a COO sidecar; store
indices at 6 bits.
Measure: PPL delta (lm-evaluation-harness, wikitext + lambada), compressed size,
dequant-at-load speed. **Gate:** claim requires ≤1% PPL delta at ≥3× — the
paper says +0.03% at 4.41×; treat as unverified until we measure it.
If it passes, feed its 6-bit index stream into Stage C's entropy packer
(codebook indices are maximally skew-friendly — the combination nobody ships).

## Stack-1 plan (Round 6): proving/disproving multi-layer stacking
### P0 — demonstrated stack on our hardware (1 day, off-the-shelf)
llm-compressor official recipe `quantization_2of4_sparse_w4a16` on Llama-3.1-8B
→ vLLM serve → record peak VRAM / tok-s / wikitext PPL vs plain W4A16 vs BF16.
### P1 — entropy-headroom probe (½ day, CPU-only, decisive)
`stageB_entropy_probe.py` on: P0 outputs (group-wise int4 → expect ~1.1x),
an absmax-quantized copy (→ expect ~2x+, ECQ/EntroLLM regime), an FP8-E5M2
per-channel copy (→ expect ~1.4x, DFloat11 issue #15 data). The winning
entropy-layer target format is chosen by THESE numbers, not by paper headlines.
### P2 — sharing layer replication (1 week, data-free)
`stageC_legolm.py --measure <model>` then full replication + lm-eval PPL gate.
FIRST SANDBOX RESULT (synthetic): sidecar caught 89.6% of injected outliers,
cosine 0.9995, but only 2.56x vs BF16 at K=64 (paper claims 4.41x — difference
is format engineering: sidecar position coding + real-world skew), and index
stream headroom was 1.03x => Layer-3 on codebook indices looks WEAK. Real-model
run decides; kill Layer-3-on-indices if headroom < 1.15x on real weights.
### P3 — fused kernel (2-4 weeks, only if P1+P2 pass gates)
Triton LUT-GEMV reading entropy-coded/packed resident weights. Blueprints:
SqueezeLLM LUT kernels, EOQ's 16 CUDA kernel variants, ISCA'26 tile-ANS.
Kill if < 0.9x Marlin tok/s at equal quality.
### Audit-before-trust list
`caiovicentino/eoq-quantization` (closest existing full attempt; solo dev,
LLM-co-authored, ~0 adoption — re-run its benchmarks before building on it);
`drxddy/ecq` (borrow `entropy_analysis.py`/`rans_codec.py`; its tok/s table is
a PROJECTION — the Metal kernel is in "Next Steps", not built).

## Kill criteria summary
| Stage | Kill if |
|---|---|
| A | identity diverges or decode −>10% |
| B | quantized-format headroom <1.15× |
| C | packed load slower than plain load by >2× |
| C2 | kernel <0.9× Marlin tok/s at equal quality |
| C-alt | >1% PPL delta or <3× ratio |

## What we are NOT doing (scope discipline, per Round-5 findings)
- No retraining/distillation (DeltaLLM-style healing is a later, optional add-on
  only if pruning stages show promise — its artifacts are not public anyway).
- No 1–2-bit extreme quantization (quality collapse below ~30B scale).
- No MoE/offload/KV work (Rounds 1–3 territory; explicitly out of scope).
- No trust in unreproduced headline numbers (see skepticism ledger in
  ../research/round5_validation_prototype.md).

## Round-7 final experiment ladder (supersedes Stack-1 where they overlap)
E1 vLLM+AWQ baseline → E2 DFloat11 (stageA script; tests the unfused-decode speed hypothesis)
→ E3 entropy headroom probe (stageB script; picks the target format) → E4 BitStack fused
PPL-vs-MB curve → E5 LegoLM replication (stageC script) → E6 fork DFloat11, swap payload to
E3's winning format, then port decode into a FLUTE/SqueezeLLM-style fused GEMV.
Full detail + kill gates + file-reading order: ../research/round7_final_dossier.md §10–11.
