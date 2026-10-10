#!/usr/bin/env python
"""
moe_imatrix_skew.py -- expert popularity for a big MoE, straight from imatrix.

Why this exists
---------------
The open question is whether a *hot-expert cache* is worth building: park the
experts that fire most often in VRAM and leave the rest in RAM.  That is a
property of the model's router, not of the graphics card, so it can be answered
without writing a single kernel: a llama.cpp imatrix file records, per layer,
how many times each expert was selected across a large calibration corpus.

What it reports
---------------
Per layer and pooled: Gini, the coverage curve (what share of all expert
traffic a cache holding x% of the experts actually captures), and how many
experts are needed to reach 80% of traffic.  The practical summary line is:
for a given VRAM budget, what fraction of RAM expert reads could be avoided.

Usage
-----
  python moe_imatrix_skew.py <imatrix.gguf> [--expert-gb 19.569] [--vram-gb 2.0]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from gguf import GGUFReader

# Reference point measured earlier with moe_routing_probe.py (64-expert OLMoE,
# top-8): the "20% of experts do 80% of the work" claim was false there, and
# the real curve was 20% -> ~44-49% of traffic, ~55% of experts needed for 80%.
OLMOE_REF = {"gini": (0.380, 0.427), "top20": (0.443, 0.489),
             "for80": (0.53, 0.56), "experts": 64}

FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def coverage(counts, frac):
    """Share of traffic captured by the top `frac` of experts."""
    c = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    k = max(1, int(round(frac * c.size)))
    return float(c[:k].sum() / c.sum())


def experts_for(counts, target):
    c = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    total = c.sum()
    return float(np.searchsorted(np.cumsum(c) / total, target) + 1) / c.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imatrix")
    ap.add_argument("--expert-gb", type=float, default=19.569,
                    help="total bytes of expert tensors in the model")
    ap.add_argument("--vram-gb", type=float, default=2.0,
                    help="VRAM that could be given to a cache")
    ap.add_argument("--tag", default="qwen36_35b_a3b")
    args = ap.parse_args()

    r = GGUFReader(args.imatrix)
    per_layer = {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_gate_exps\.weight\.counts$", t.name)
        if not m:
            continue
        layer = int(m.group(1))
        vals = np.asarray(t.data, dtype=np.float64).ravel()
        per_layer[layer] = vals

    if not per_layer:
        sys.exit("no per-expert 'counts' tensors found in this imatrix")

    n_experts = len(next(iter(per_layer.values())))
    layers = sorted(per_layer)
    print(f"imatrix   : {os.path.basename(args.imatrix)}")
    print(f"layers    : {len(layers)}   experts/layer: {n_experts}")

    # Sanity: gate/up/down counts for the same layer should agree; if they do
    # not, the file is reporting something other than per-expert selection.
    mism = []
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_(up|down)_exps\.weight\.counts$", t.name)
        if m:
            other = np.asarray(t.data, dtype=np.float64).ravel()
            ref = per_layer[int(m.group(1))]
            if other.shape == ref.shape and not np.allclose(other, ref):
                mism.append(t.name)
    if mism:
        print(f"WARNING: {len(mism)} layers disagree between gate/up/down "
              f"counts, e.g. {mism[:2]}")
    else:
        print("sanity    : gate/up/down per-expert counts agree on every layer")

    rows = []
    print(f"\n{'layer':>5} {'gini':>6} " +
          " ".join(f"{int(f*100):>4d}%" for f in FRACTIONS) +
          f" {'for80':>6}")
    for L in layers:
        c = per_layer[L]
        cov = [coverage(c, f) for f in FRACTIONS]
        e80 = experts_for(c, 0.80)
        rows.append({"layer": L, "gini": gini(c), "coverage": cov,
                     "for80": e80, "total_hits": float(c.sum())})
        print(f"{L:5d} {gini(c):6.3f} " +
              " ".join(f"{v*100:5.1f}" for v in cov) + f" {e80*100:6.1f}")

    g = np.array([r_["gini"] for r_ in rows])
    e80 = np.array([r_["for80"] for r_ in rows])
    cov = np.array([r_["coverage"] for r_ in rows])
    pool = np.concatenate([per_layer[L] / per_layer[L].sum() for L in layers])

    print("\n=== pooled across layers ===")
    print(f"  gini                     {gini(pool):.3f}"
          f"   (per-layer mean {g.mean():.3f}, "
          f"range {g.min():.3f}-{g.max():.3f})")
    for i, f in enumerate(FRACTIONS):
        if f in (0.05, 0.10, 0.20, 0.50):
            print(f"  top {f*100:4.0f}% of experts capture "
                  f"{cov[:, i].mean()*100:5.1f}% of traffic")
    print(f"  experts needed for 80%   {e80.mean()*100:.1f}%"
          f"  (range {e80.min()*100:.0f}-{e80.max()*100:.0f}%)")

    print("\n=== what a cache would buy (per-layer static, top-N pinned) ===")
    print(f"{'cache %':>8} {'experts':>8} {'VRAM GB':>9} {'traffic %':>10}"
          f"{'RAM reads cut':>14}")
    budget = []
    for f in FRACTIONS:
        n_exp = max(1, int(round(f * n_experts)))
        gb = args.expert_gb * f
        cap = cov[:, FRACTIONS.index(f)].mean() if f in FRACTIONS else 0.0
        budget.append({"cache_frac": f, "experts": n_exp, "vram_gb": round(gb, 3),
                       "traffic_captured": round(cap, 4)})
        print(f"{f*100:7.0f}% {n_exp:8d} {gb:9.2f} {cap*100:9.1f}% {cap*100:13.1f}%")

    fit = [b for b in budget if b["vram_gb"] <= args.vram_gb]
    best = fit[-1] if fit else None
    print(f"\n=== verdict for a {args.vram_gb:.1f} GB cache ===")
    if best:
        print(f"  hold top {best['experts']} of {n_experts} experts per layer "
              f"({best['cache_frac']*100:.0f}%, {best['vram_gb']:.2f} GB)")
        print(f"  captures {best['traffic_captured']*100:.1f}% of all expert "
              f"traffic  ->  that much RAM reading removed")
        # Ratio of traffic captured to bytes spent, vs the naive uniform case.
        print(f"  efficiency vs random (cache/random = "
              f"{best['traffic_captured']/best['cache_frac']:.2f}x)")
    else:
        print("  budget too small to hold even 1% of experts")

    print(f"\n  reference (OLMoE, {OLMOE_REF['experts']} experts/layer): "
          f"gini {OLMOE_REF['gini']}, top-20% -> {OLMOE_REF['top20']}, "
          f"80% needs {OLMOE_REF['for80']}")

    out = os.path.join("experiments", "runs", f"moe_imatrix_skew_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"imatrix": args.imatrix, "n_experts": n_experts,
                   "layers": rows, "pooled_gini": gini(pool),
                   "mean_gini": float(g.mean()),
                   "mean_for80": float(e80.mean()),
                   "budget_curve": budget,
                   "vram_gb": args.vram_gb,
                   "olmoE_reference": OLMOE_REF}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
