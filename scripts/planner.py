#!/usr/bin/env python
"""
planner.py -- pick the fastest configuration a card can actually sustain.

Why this exists: runtimes choose GPU offload by *policy*, not by arithmetic.
Measured on this laptop, Ollama put 76% of a model's layers on the GPU and ran
23.43 tok/s; forcing 100% ran 40.63 tok/s -- same model, same context, same
quality. The planner computes what the runtime only guesses:

  * byte anatomy of the model (weights, per-layer bytes, embedding/output tail)
  * KV cost per token, hybrid-aware (only full-attention layers carry a cache)
  * the largest context that keeps EVERYTHING resident (no CPU spill)
  * predicted decode speed from constants measured on this machine
  * the best (layers-on-GPU, context, KV type) combination for a goal

Two different "context" numbers, and mixing them is the classic error:
  * ctx_alloc -- the buffer the runtime allocates. Costs VRAM. Must fit.
  * ctx_filled -- tokens actually in the prompt. Costs TIME. Does not affect fit.
A 64K config runs at nearly full speed with an empty prompt and slows down as
the prompt fills; both numbers are needed to predict anything.

Calibration constants (all measured on this machine, see experiments/):
  R_W   228 GB/s  weights read rate, from the flat decode term
                  (5.618 GB / 24.61 ms = 228 GB/s, and it stays flat as
                  context grows: 24.49 ms at 0 tokens, 24.92 ms at 60K)
  R_KV  258 GB/s  KV read rate, from the measured slope
                  (1.270e-4 ms per token of context at 32 KiB/token)
  R_CPU  56 GB/s  CPU-resident layer rate, from the measured 76%-GPU run
                  (1.348 GB of layers on CPU cost 24.0 ms/token)
  BW_ACH 240 GB/s achievable bandwidth (bench_bw.py), for sanity checks

Usage:
  python planner.py --ollama qwen3.5:9b
  python planner.py /path/to/model.gguf
  python planner.py --ollama qwen3.5:9b --ctx 32768 --min-tok-s 30
  python planner.py --frontier            # how far can this card go?
  python planner.py --frontier --bpw 4.0
  python planner.py --calibrate           # predictions vs our measured runs
"""
import argparse
import json
import os
import sys

GIB = 1 << 30
MIB = 1 << 20

# ---- this machine's measured limits (do not "improve" these by guessing) ----
BUDGET_GIB = 7.70     # proven: a 100%-GPU run peaked at 7878 of 8188 MiB
OVERHEAD_GIB = 0.35   # CUDA context, compute buffers, display, staging
R_W = 228.0           # GB/s, weights
R_KV = 258.0          # GB/s, KV cache reads
# CPU-resident weights run sequentially after the GPU layers, so the cost is
# additive. Fitted to ONE measured partial-offload run (25/32 layers on GPU ->
# 23.43 tok/s, i.e. 42.68 ms/token, of which 20.60 ms is the GPU part). CPU
# layers are sync/latency-bound rather than purely bandwidth-bound, so treat
# partial-offload predictions as indicative to ~20%, not exact like the
# full-GPU ones.
R_CPU = 41.7          # GB/s, CPU-resident weights

BW_ACH = 240.0

# bytes per element, by KV cache type
KV_TYPES = {"f16": 2.0, "q8_0": 34.0 / 32.0, "q4_0": 18.0 / 32.0}
# Quantized KV is SLOWER per token, not faster: it saves memory, and pays for it
# with a more expensive attention kernel. These multiply the token time.
#   f16   1.00  reference
#   q8_0  1.15  measured 40.49 -> 36.40 tok/s empty (1.11) and 34.34 -> 31.48
#              at 32K filled (1.18). One flat multiplier cannot fit both, so
#              1.15 is the compromise and q8_0 speed lands within ~5%.
#   q4_0  1.25  NOT MEASURED -- extrapolated. Do not trust until benchmarked.
KV_SPEED_PENALTY = {"f16": 1.0, "q8_0": 1.15, "q4_0": 1.25}
UNMEASURED_KV = {"q4_0"}


def find_ollama_blobs(model):
    """Locate Ollama's own blob files for a model name (via its manifest)."""
    root = os.path.expanduser("~/.ollama/models")
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    man = os.path.join(root, "manifests", "registry.ollama.ai", "library", name, tag)
    if not os.path.exists(man):
        return []
    layers = json.load(open(man, encoding="utf-8"))["layers"]
    out = []
    for l in layers:
        p = os.path.join(root, "blobs", l["digest"].replace(":", "-"))
        if os.path.exists(p):
            out.append((p, l.get("mediaType", "")))
    return out


def pick_model_blob(model):
    """Ollama keeps an optional vision projector beside the model; only the
    model blob carries the architecture metadata the planner reads. Picking the
    wrong one silently plans a 879 MiB CLIP projector instead of a 9B model."""
    blobs = find_ollama_blobs(model)
    for p, mt in blobs:
        if "model" in mt and "projector" not in mt:
            return p
    return blobs[0][0] if blobs else None


def anatomy(path):
    """Read a GGUF and return the numbers the planner needs.

    Layer bytes are summed per `blk.<i>.` group so partial offload can be
    modelled: llama.cpp splits by layer, not by tensor.
    """
    from gguf import GGUFReader
    r = GGUFReader(path)
    f = r.fields
    if "general.architecture" not in f:
        return None
    arch = f["general.architecture"].contents()

    def g(k):
        return f[f"{arch}.{k}"].contents() if f"{arch}.{k}" in f else None

    n_layers = g("block_count")
    kv_heads = g("attention.head_count_kv")
    head_dim = g("attention.key_length")
    hidden = g("embedding_length")
    interval = g("full_attention_interval")
    n_attn = n_layers
    if isinstance(interval, int) and interval > 1:
        n_attn = (n_layers + interval - 1) // interval

    layer_bytes = {}
    tail_bytes = 0        # embedding / output / norms: always resident
    total = 0
    for t in r.tensors:
        nb = t.data.nbytes
        total += nb
        if t.name.startswith("blk."):
            i = int(t.name.split(".")[1])
            layer_bytes[i] = layer_bytes.get(i, 0) + nb
        else:
            tail_bytes += nb
    per_layer = sum(layer_bytes.values()) / max(len(layer_bytes), 1)

    return dict(
        path=path, arch=arch, n_layers=n_layers, n_kv_layers=n_attn,
        kv_heads=kv_heads, head_dim=head_dim, hidden=hidden,
        hybrid=(n_attn != n_layers), interval=interval,
        total_bytes=total, tail_bytes=tail_bytes,
        per_layer_bytes=per_layer, n_layer_tensors=len(layer_bytes),
        file_type=f["general.file_type"].contents() if "general.file_type" in f else None,
    )


def kv_per_token(a, kv_type):
    """Bytes of KV cache added per token of context."""
    if not (a["n_kv_layers"] and a["kv_heads"] and a["head_dim"]):
        return None
    return 2 * a["n_kv_layers"] * a["kv_heads"] * a["head_dim"] * KV_TYPES[kv_type]


def predict_ms(a, n_gpu, ctx_filled, kv_type):
    """Predicted decode time per token, in ms. Everything is additive because
    llama.cpp runs GPU-resident layers and CPU-resident layers in sequence."""
    gpu_bytes = a["tail_bytes"] + n_gpu * a["per_layer_bytes"]
    cpu_bytes = (a["n_layers"] - n_gpu) * a["per_layer_bytes"]
    kv_bytes_per_tok = kv_per_token(a, kv_type) or 0.0
    # the R_* constants are in GB/s; bytes / (GB/s * 1e6) is milliseconds
    gpu_ms = gpu_bytes / (R_W * 1e6)
    cpu_ms = cpu_bytes / (R_CPU * 1e6)
    kv_ms = ctx_filled * kv_bytes_per_tok / (R_KV * 1e6)
    return (gpu_ms + cpu_ms + kv_ms) * KV_SPEED_PENALTY[kv_type]


def weights_ms(a, n_gpu):
    """Just the weight-read term, which is flat in context and sets the floor."""
    return (a["tail_bytes"] + n_gpu * a["per_layer_bytes"]) / (R_W * 1e6)


def vram_gib(a, n_gpu, ctx_alloc, kv_type):
    return ((a["tail_bytes"] + n_gpu * a["per_layer_bytes"]
             + (kv_per_token(a, kv_type) or 0.0) * ctx_alloc) / GIB + OVERHEAD_GIB)


def max_context(a, n_gpu, kv_type, budget=BUDGET_GIB):
    """Largest context that still keeps everything resident."""
    room = budget - OVERHEAD_GIB - (a["tail_bytes"] + n_gpu * a["per_layer_bytes"]) / GIB
    per_tok = kv_per_token(a, kv_type)
    if per_tok is None or room <= 0:
        return 0
    return int(room * GIB / per_tok)


def best_plan(a, ctx_want, min_tok_s=0.0, kv_types=("f16", "q8_0"),
              filled_ratio=1.0, budget=BUDGET_GIB):
    """Search (layers-on-GPU, KV type) for the fastest config that holds
    ctx_want tokens. `filled_ratio` is how full the context is during the
    speed measurement: 0 = empty prompt, 1 = fully filled."""
    out = []
    for kv_type in kv_types:
        for n_gpu in range(a["n_layers"], -1, -1):
            need = vram_gib(a, n_gpu, ctx_want, kv_type)
            if need > budget:
                continue
            ms = predict_ms(a, n_gpu, ctx_want * filled_ratio, kv_type)
            out.append(dict(kv_type=kv_type, n_gpu=n_gpu, vram=need,
                            tok_s=1000.0 / ms, ms=ms,
                            max_ctx=max_context(a, n_gpu, kv_type)))
            break   # most layers that fit is the fastest for this KV type
    out = [c for c in out if c["tok_s"] >= min_tok_s]
    out.sort(key=lambda c: -c["tok_s"])
    return out


def report(a, args):
    name = os.path.basename(a["path"])
    weights_gib = a["total_bytes"] / GIB
    print(f"\n{'='*74}")
    print(f"MODEL  {name}")
    print(f"       arch={a['arch']}  layers={a['n_layers']}"
          + (f"  HYBRID: {a['n_kv_layers']}/{a['n_layers']} layers carry KV"
             f" (interval {a['interval']})" if a["hybrid"] else "  (dense attention)"))
    print(f"       kv_heads={a['kv_heads']}  head_dim={a['head_dim']}  hidden={a['hidden']}"
          f"  file_type={a['file_type']}")
    print(f"\nWEIGHTS  {weights_gib:.3f} GiB total"
          f"  ({a['tail_bytes']/GIB:.3f} embedding+output tail,"
          f"  {a['per_layer_bytes']*a['n_layers']/GIB:.3f} across {a['n_layers']} layers"
          f" = {a['per_layer_bytes']/MIB:.1f} MiB/layer)")
    if a["n_layers"]:
        print(f"BUDGET   {BUDGET_GIB:.2f} GiB usable, {OVERHEAD_GIB:.2f} overhead"
              f"  ->  {(BUDGET_GIB-OVERHEAD_GIB)*GIB/a['per_layer_bytes']:.0f} layers of room"
              f" (+ tail) if no KV at all")

    print(f"\nKV PER TOKEN")
    for kt in args.kv_types:
        b = kv_per_token(a, kt)
        if b:
            print(f"  {kt:5s} {b/1024:8.2f} KiB   max context at 100% GPU:"
                  f" {max_context(a, a['n_layers'], kt)/1024:.0f}K")
    print(f"  (for reference, a {a['n_layers']}-layer dense model with the same"
          f" geometry would be {2*a['n_layers']*a['kv_heads']*a['head_dim']*2/1024:.1f} KiB/token)")

    full = a["n_layers"]
    kt = args.kv_type
    print(f"\nPREDICTED DECODE SPEED  (100% GPU, {kt} KV, weights {weights_gib:.2f} GiB)")
    print(          f"  {'tokens filled':>14s} {'ms/token':>9s} {'tok/s':>8s}   (weights term"
          f" {weights_ms(a, full):.2f} ms)")
    for filled in (0, 8000, 16000, 32000, 64000, 128000):
        ms = predict_ms(a, full, filled, kt)
        print(f"  {filled:>14d} {ms:9.2f} {1000/ms:8.2f}")

    plans = best_plan(a, args.ctx, args.min_tok_s, args.kv_types,
                      filled_ratio=args.filled_ratio)
    print(f"\nBEST PLAN for {args.ctx/1024:.0f}K context,"
          f" measured at {args.filled_ratio:.0%} filled")
    if not plans:
        print("  nothing fits this budget at that context.")
        return
    print(f"  {'kv':>5s} {'layers on GPU':>14s} {'VRAM':>7s} {'max ctx':>8s}"
          f" {'ms/token':>9s} {'tok/s':>7s}")
    for p in plans[:4]:
        print(f"  {p['kv_type']:>5s} {p['n_gpu']:>14d} {p['vram']:6.2f}G"
              f" {p['max_ctx']/1024:7.0f}K {p['ms']:9.2f} {p['tok_s']:7.2f}")
    win = plans[0]
    if args.default_layers is not None:
        d_ms = predict_ms(a, args.default_layers, args.ctx * args.filled_ratio, kt)
        d_tps = 1000 / d_ms
        print(f"\n  vs RUNTIME DEFAULT ({args.default_layers} layers on GPU):"
              f" {d_tps:.2f} tok/s  ->  planner is {win['tok_s']/d_tps:.2f}x faster"
              f"  (+{100*(win['tok_s']/d_tps-1):.0f}%)")

    print(f"\nEXACT COMMANDS")
    print(f"  # Ollama (Modelfile):")
    print(f"  FROM {args.ollama or name}")
    print(f"  PARAMETER num_gpu {win['n_gpu']}")
    print(f"  PARAMETER num_ctx {1 << (args.ctx-1).bit_length()}")
    print(f"  # then: ollama create <newname> -f Modelfile; ollama run <newname>")
    print(f"  # llama-server:")
    print(f"  llama-server -m <blob> -ngl {win['n_gpu']}"
          f" -c {1 << (args.ctx-1).bit_length()} -ctk {win['kv_type']}"
          f" -ctv {win['kv_type']} -fa on")


# ---- how far can this card go? ----
# name, params B, layers, kv heads, head dim, full-attn interval (0 = dense)
CANDIDATES = [
    ("9B dense", 9.0, 32, 8, 128, 0),
    ("12B dense", 12.0, 36, 8, 128, 0),
    ("12B hybrid", 12.0, 36, 8, 128, 4),
    ("16B dense (the one that broke)", 16.0, 36, 8, 128, 0),
    ("16B hybrid", 16.0, 40, 8, 128, 4),
    ("27B dense", 27.0, 48, 8, 128, 0),
    ("27B hybrid", 27.0, 48, 12, 128, 4),
    ("32B dense", 32.0, 64, 8, 128, 0),
    ("32B hybrid", 32.0, 64, 8, 128, 4),
]


def frontier(bpw, tail_frac, kv_type, ctx_show=32768, filled=0):
    print(f"\nHOW FAR CAN {BUDGET_GIB:.2f} GiB GO?   weights at {bpw:.2f} bpw,"
          f" {kv_type} KV, tail assumed {tail_frac:.1%} of weights")
    print(f"\n{'model':32s} {'weights':>8s} {'KV/tok':>8s} {'max ctx':>8s}"
          f" {'fit@32K':>8s} {'tok/s':>7s}  verdict")
    print("-" * 92)
    for name, params, layers, kvh, hd, interval in CANDIDATES:
        total = int(params * 1e9 * bpw / 8)
        tail = int(total * tail_frac)
        per_layer = (total - tail) / layers
        n_attn = layers if interval <= 1 else (layers + interval - 1) // interval
        kv_tok = 2 * n_attn * kvh * hd * KV_TYPES[kv_type]
        a = dict(path=name, arch="hypo", n_layers=layers, n_kv_layers=n_attn,
                 kv_heads=kvh, head_dim=hd, hidden=0, hybrid=n_attn != layers,
                 interval=interval, total_bytes=total, tail_bytes=tail,
                 per_layer_bytes=per_layer, n_layer_tensors=layers, file_type=None)
        w_gib = total / GIB
        mctx = max_context(a, layers, kv_type)
        need32 = vram_gib(a, layers, ctx_show, kv_type)
        if need32 <= BUDGET_GIB:
            ms = predict_ms(a, layers, filled, kv_type)
            tps = f"{1000/ms:.1f}"
            fit = "OK"
            verdict = "fits fully on GPU" + ("" if n_attn == layers else " (hybrid KV)")
        else:
            tps = "-"
            fit = f"+{need32-BUDGET_GIB:.2f}"
            verdict = "CPU spill / smaller quant needed"
        print(f"{name:32s} {w_gib:7.2f}G {kv_tok/1024:7.1f}K {mctx/1024:7.0f}K"
              f" {fit:>8s} {tps:>7s}  {verdict}")
    print("\n  KV/tok assumes f16-GQA geometry; context is the max that keeps"
          " weights+KV+overhead resident.")
    print("  MoE caveat: total params set the VRAM bill, ACTIVE params set the"
          " speed. llama.cpp only avoids that with --n-cpu-moe, which puts experts"
          " in system RAM (~56 GB/s here).")


# ---- calibration: do the predictions match what we measured? ----
MEASURED_9B = [
    # (tokens filled, measured tok/s, layers on GPU, ctx_alloc, kv_type)
    (0, 40.83, 32, 32768, "f16"),
    (8000, 38.82, 32, 32768, "f16"),
    (16000, 37.57, 32, 32768, "f16"),
    (32000, 34.81, 32, 32768, "f16"),
    (60000, 31.07, 32, 65536, "f16"),
    (0, 40.35, 32, 65536, "f16"),
    (0, 23.43, 25, 32768, "f16"),   # the runtime's own default policy: 76% on GPU
    (0, 36.40, 32, 65536, "q8_0"),
    (32768, 31.48, 32, 65536, "q8_0"),
]


def calibrate(ollama_name):
    blob = pick_model_blob(ollama_name)
    a = anatomy(blob) if blob else None
    if a is None:
        print(f"no GGUF metadata for {ollama_name}")
        return
    n_layers_full = a["n_layers"]
    # Ollama reports num_gpu over layers+1 (it can also offload the output
    # tensor); scale the observed count onto our layer model.
    print(f"\nCALIBRATION against measured runs of {ollama_name}")
    print(f"model: {n_layers_full} layers, {a['per_layer_bytes']/MIB:.1f} MiB/layer,"
          f" tail {a['tail_bytes']/MIB:.0f} MiB, weights {a['total_bytes']/GIB:.3f} GiB\n")
    print(f"  {'filled':>7s} {'n_gpu':>6s} {'kv':>5s} {'measured':>9s}"
          f" {'predicted':>10s} {'error':>8s}")
    errs = []
    for filled, meas, n_gpu, ctx_alloc, kv_type in MEASURED_9B:
        ng = min(n_gpu, n_layers_full)
        ms = predict_ms(a, ng, filled, kv_type)
        pred = 1000 / ms
        err = (pred - meas) / meas
        errs.append(abs(err))
        print(f"  {filled:>7d} {ng:>6d} {kv_type:>5s} {meas:9.2f} {pred:10.2f} {err:+7.1%}")
    print(f"\n  mean absolute error: {sum(errs)/len(errs):.1%}"
          f"   worst: {max(errs):.1%}")
    print("\n  (R_W/R_KV/R_CPU were fitted to these points; the useful check is"
          " whether one set of constants explains all of them, including the"
          " 76%-GPU default.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="a .gguf file")
    ap.add_argument("--ollama", help="model name, e.g. qwen3.5:9b")
    ap.add_argument("--ctx", type=int, default=32768, help="context to plan for")
    ap.add_argument("--kv-type", default="f16", choices=list(KV_TYPES))
    ap.add_argument("--kv-types", nargs="+", default=["f16", "q8_0"],
                    choices=list(KV_TYPES))
    ap.add_argument("--min-tok-s", type=float, default=0.0)
    ap.add_argument("--filled-ratio", type=float, default=1.0,
                    help="how full the context is during the speed run")
    ap.add_argument("--default-layers", type=int, default=None,
                    help="layers the runtime chose by itself, to show the gap")
    ap.add_argument("--frontier", action="store_true", help="how far can this card go")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--bpw", type=float, default=4.5)
    ap.add_argument("--tail-frac", type=float, default=0.06)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.ollama or "qwen3.5:9b")
        return
    if args.frontier:
        frontier(args.bpw, args.tail_frac, args.kv_type)
        return

    if args.ollama:
        paths = [b for b in [pick_model_blob(args.ollama)] if b]
    elif args.path:
        paths = [args.path]
    else:
        ap.error("need a .gguf path or --ollama NAME")

    for p in paths:
        a = anatomy(p)
        if a is None:
            continue
        if args.json:
            plans = best_plan(a, args.ctx, args.min_tok_s, args.kv_types,
                              args.filled_ratio)
            print(json.dumps(dict(model=os.path.basename(p), arch=a["arch"],
                                 layers=a["n_layers"], kv_layers=a["n_kv_layers"],
                                 weights_gib=a["total_bytes"] / GIB,
                                 kv_per_token={k: kv_per_token(a, k)
                                               for k in args.kv_types},
                                 max_ctx={k: max_context(a, a["n_layers"], k)
                                          for k in args.kv_types},
                                 plans=plans), indent=2))
            continue
        report(a, args)


if __name__ == "__main__":
    main()
