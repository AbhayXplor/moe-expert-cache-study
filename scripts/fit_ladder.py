#!/usr/bin/env python
"""
fit_ladder.py -- what fits in 8 GB, and why.

The interesting column is not the weights, it is the KV cache.  A classic
(dense, GQA) transformer spends 2 * layers * kv_heads * head_dim * 2 bytes
per token, and that grows linearly with context.  The new hybrid/SSM models
(qwen3.5) only pay for the minority of layers that do attention, so the same
card holds far more context.

All sizes are GiB.  The budget is what we *measured* as usable on this laptop
(RTX 4060 Laptop, 8188 MiB, driver reported 8188 total): a 100%-GPU run at
64K context peaked at 7878 MiB, so 7.70 GiB is proven reachable.  Fixed
overhead (CUDA context, compute buffers, output/embedding staging) measured
~0.30 GiB for the 9B; we round to 0.35 GiB for bigger models.
"""
import argparse

MIB = 1 << 20
GIB = 1 << 30


def gib(nbytes):
    return nbytes / GIB


def kv_per_token(layers, kv_heads, head_dim, bytes_per_elem=2):
    return 2 * layers * kv_heads * head_dim * bytes_per_elem


# name, params(B), layers, kv_heads, head_dim, note
MODELS = [
    ("qwen3.5:9b (hybrid, 8/32 attn)", 8.95, 8, 4, 256,
     "measured: 32 KiB/token"),
    ("9B classic dense", 9.0, 32, 8, 128, "typical 7-9B layout"),
    ("12B classic dense", 12.0, 36, 8, 128, "typical 12-14B layout"),
    ("16B classic dense  <-- your target", 16.0, 36, 8, 128,
     "assumed layout; tune below"),
    ("32B classic dense", 32.0, 64, 8, 128, "reference"),
]

BUDGET_GIB = 7.70
OVERHEAD_GIB = 0.35


def row(label, params_b, bpw, kv_tok, ctx, budget=BUDGET_GIB):
    weights = gib(int(params_b * 1e9 * bpw / 8))
    kv = gib(kv_tok * ctx)
    total = weights + kv + OVERHEAD_GIB
    return dict(label=label, bpw=bpw, ctx=ctx, weights=weights, kv=kv,
                total=total, ok=total <= budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-bytes", type=float, default=2.0,
                    help="2 = fp16 KV, 1 = q8_0 KV")
    ap.add_argument("--ctx", type=int, default=16384)
    args = ap.parse_args()

    print(f"\nbudget = {BUDGET_GIB:.2f} GiB usable, overhead {OVERHEAD_GIB:.2f} GiB"
          f", KV bytes/elem = {args.kv_bytes:.0f}\n")
    print(f"{'model':38s} {'params':>7s} {'KV/tok':>8s} {'4.5bpw':>8s}"
          f" {'KV@32K':>8s} {'total':>8s}  verdict")
    print("-" * 92)
    for name, params, layers, kvh, hd, note in MODELS:
        kv_tok = kv_per_token(layers, kvh, hd, args.kv_bytes)
        w = gib(int(params * 1e9 * 4.5 / 8))
        kv32 = gib(kv_tok * 32768)
        total = w + kv32 + OVERHEAD_GIB
        verdict = "FITS" if total <= BUDGET_GIB else f"NO (+{total-BUDGET_GIB:.2f})"
        print(f"{name:38s} {params:6.2f}B {kv_tok/1024:7.1f}K {w:7.2f}G"
              f" {kv32:7.2f}G {total:7.2f}G  {verdict}")
    print()

    # What bit width is needed for a given model+context?
    print(f"how low must bits/weight go to fit (context {args.ctx//1024}K,"
          f" KV {args.kv_bytes:.0f} B/elem)?")
    print(f"{'model':38s} {'now':>6s} {'need':>6s} {'gain':>7s}  comment")
    print("-" * 92)
    for name, params, layers, kvh, hd, note in MODELS:
        kv_tok = kv_per_token(layers, kvh, hd, args.kv_bytes)
        kv = gib(kv_tok * args.ctx)
        room = BUDGET_GIB - kv - OVERHEAD_GIB
        need = room * 8 / params if room > 0 else float("nan")
        cur = 4.5
        gain = (cur / need) if need > 0 else float("nan")
        if need <= 0:
            comment = "even 0 bits/weight does not fit"
        elif need >= 8:
            comment = "fits at fp16 weights"
        elif need >= cur:
            comment = "fits as-is at 4.5 bpw"
        else:
            comment = f"{gain:.2f}x smaller weights required"
        print(f"{name:38s} {cur:5.1f} {need:6.2f} {gain:6.2f}x  {comment}")
    print()

    # The three waste levers, applied on top of each other.
    print("levers, stacked (16B dense, 16K ctx):")
    params, layers, kvh, hd = 16.0, 36, 8, 128
    kv_tok = kv_per_token(layers, kvh, hd, args.kv_bytes)
    for label, bpw, ctx in [
            ("today, 4.5 bpw, fp16 KV", 4.5, 16384),
            ("+ q8_0 KV cache", 4.5, 16384),
            ("+ entropy-coded weights (-21.9%)", 4.5 * 0.781, 16384),
            ("+ entropy-coded, q8_0 KV", 4.5 * 0.781, 16384),
            ("+ everything, but context is the wall", 4.5 * 0.781, 32768)]:
        kv_bytes = 1.0 if "q8_0" in label else args.kv_bytes
        kvt = kv_per_token(layers, kvh, hd, kv_bytes)
        r = row(label, params, bpw, kvt, ctx)
        verdict = "FITS" if r["ok"] else f"NO (+{r['total']-BUDGET_GIB:.2f})"
        print(f"  {label:38s} weights {r['weights']:5.2f}G"
              f"  KV {r['kv']:5.2f}G  total {r['total']:5.2f}G  {verdict}")


if __name__ == "__main__":
    main()
