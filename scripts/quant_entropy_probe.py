#!/usr/bin/env python
"""
quant_entropy_probe.py -- does entropy coding pay off AFTER quantization?

E3 established that DF11's lossless representation is already at its floor:
H(sign+mantissa) = 7.97 of 8 stored bits, so raw bf16 mantissas are
incompressible noise. That kills "better entropy coding of bf16".

But the interesting combination is the other order: quantize first, then
entropy-code the resulting integer codes. Uniform (non-quantile) quantization
produces a peaked distribution -- most codes near zero, a Laplace/Gaussian-ish
histogram -- so the codes carry less entropy than their bit width.

This script measures, per projection weight matrix, with per-row (per-output-
channel) affine quantization:

  * int4 / int6 / int8 uniform codes: stored bits vs entropy of the codes
  * the same for a Gaussian/Laplace-optimal quantizer (sqrt-scaling)
  * the packing overhead-free size and the entropy-coded size

so we can put a number on "4-bit + entropy coding" vs "lossless DF11's
10.66 bits/weight".

Usage:
  python quant_entropy_probe.py --base Qwen/Qwen3-1.7B --layers 0 1 2 3
"""
import argparse, json, os

import numpy as np
import torch


def H(codes):
    h = np.bincount(codes.reshape(-1), minlength=codes.max() + 1).astype(np.float64)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def quantize(W, bits, kind="uniform"):
    """Per-row affine quantization; returns integer codes in [0, 2**bits)."""
    q = 2 ** bits - 1
    lo = W.min(dim=1, keepdim=True).values
    hi = W.max(dim=1, keepdim=True).values
    if kind == "sqrt":                      # ~Laplace-optimal spacing
        sc = torch.sqrt(hi - lo).clamp_min(1e-12)
        s = (hi - lo) / sc
    else:
        s = hi - lo
    s = s.clamp_min(1e-12)
    codes = ((W - lo) / s * q).round().clamp(0, q).to(torch.int32)
    return codes.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("cdf", "scripts/compress_df11_par.py")
    cdf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cdf)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    src = args.base if os.path.isdir(args.base) else snapshot_download(
        args.base, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
    weight_map = json.load(open(os.path.join(src, "model.safetensors.index.json"),
                                encoding="utf-8"))["weight_map"]
    proj = sorted({cdf.PROJ_RE.match(n).group(2) for n in weight_map if cdf.PROJ_RE.match(n)})

    acc = {}
    n_tot = 0
    for b in args.layers:
        block = f"model.layers.{b}"
        shard = weight_map[f"{block}.{proj[0]}.weight"]
        sd = load_file(os.path.join(src, shard))
        for p in proj:
            W = sd[f"{block}.{p}.weight"].float()
            for bits in (4, 6, 8):
                for kind in ("uniform", "sqrt"):
                    c = quantize(W, bits, kind)
                    mse = None
                    e = H(c)
                    k = f"{kind}-int{bits}"
                    n = c.size
                    a = acc.setdefault(k, {"bits_sum": 0.0, "n": 0, "code_sum": 0.0})
                    a["bits_sum"] += e * n
                    a["code_sum"] += n
                    a["n"] += n
            n_tot += W.numel()
            del W
        del sd
        print(f"  layer {b} done ({n_tot/1e6:.1f} M weights)", flush=True)

    print(f"\nweights analysed: {n_tot/1e6:.1f} M\n")
    print(f"{'scheme':14s} {'stored bits':>11s} {'H(codes)':>9s} {'saving':>7s} "
          f"{'GB for 1.74B':>13s}")
    print("-" * 60)
    for bits in (4, 6, 8):
        for kind in ("uniform", "sqrt"):
            k = f"{kind}-int{bits}"
            if k not in acc:
                continue
            a = acc[k]
            h = a["bits_sum"] / a["n"]
            print(f"{k:14s} {bits:11d} {h:9.4f} {(1-h/bits)*100:6.1f}% "
                  f"{n_tot*0 + 1.72e9*h/8/1e9:12.3f}")
    print("\nfor comparison:")
    print(f"  bf16 unchanged               16.0000     -        {1.72e9*2/1e9:6.3f} GB")
    print(f"  DF11 lossless (measured)     10.66 bits  (its floor: 10.58)  "
          f"{1.72e9*10.663/8/1e9:6.3f} GB on the projections")


if __name__ == "__main__":
    main()
