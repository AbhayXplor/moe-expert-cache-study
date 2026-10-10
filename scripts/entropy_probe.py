#!/usr/bin/env python
"""
entropy_probe.py -- how much VRAM headroom is left in DF11's representation?

DF11 splits every bf16 weight into two bytes and treats them very differently
(verified in dfloat11_utils.encode_weights):

    exponent_8bits = (W >> 7) & 0xFF   -> sign + top 7 exponent bits, HUFFMAN coded
    other_8bits    = (W >> 8) & 0x80 | W & 0x7F
                                       -> 7 mantissa bits + exponent LSB, STORED RAW

Measured standing costs in the built models:

    1.7B: exponent 468,613,348 B / 1,409,286,144 w = 2.660 bits/w
          other    1,409,286,144 B / 1,409,286,144 w = 8.000 bits/w   <-- 62% of bytes
    4B  : exponent 1,217,171,760 B / 3,633,300,224 w = 2.680 bits/w

So the sign+mantissa byte is emitted uncoded. This script measures the empirical
entropy of these bytes straight from the ORIGINAL bf16 weights (no 1e9-symbol
Python Huffman decode needed, since the split is a pure bit-field extraction),
and asks: what would a stronger entropy layer save, and at what point does it
stop being free?

Note the exponent codec is built from marginal frequencies only (one Huffman
table per layer, decomposed into ~6 LUT subtables purely for fast two-level
decoding) -- it is memoryless, so its cost should sit close to H(exponent).

Usage:
  python entropy_probe.py --base Qwen/Qwen3-1.7B --layers 0 1 2 3
"""
import argparse, json, os

import numpy as np
import torch


def H(hist):
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--proj", nargs="+", default=None,
                    help="projection names in the compressor's order")
    ap.add_argument("--zstd", action="store_true", help="also try zstd on the raw byte stream")
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
    proj = args.proj or sorted({cdf.PROJ_RE.match(n).group(2) for n in weight_map if cdf.PROJ_RE.match(n)})
    print(f"projections in coding order: {proj}", flush=True)

    hist_exp = np.zeros(256, dtype=np.int64)
    hist_oth = np.zeros(256, dtype=np.int64)
    hist_joint = np.zeros(256 * 256, dtype=np.int64)      # (exp, other)
    hist_pair = np.zeros(256 * 256, dtype=np.int64)       # (exp[i-1], exp[i])
    n_total = 0
    bits_exp_total = 0
    zstd_samples = []

    for b in args.layers:
        block = f"model.layers.{b}"
        shard = weight_map[f"{block}.{proj[0]}.weight"]
        ws = [load_file(os.path.join(src, shard))[f"{block}.{p}.weight"] for p in proj]
        W = torch.cat([w.reshape(-1) for w in ws]).view(torch.int16).numpy().astype(np.uint16)
        del ws
        e = ((W >> 7) & 0xFF).astype(np.uint8)
        o = (((W >> 8) & 0x80) | (W & 0x7F)).astype(np.uint8)
        del W

        hist_exp += np.bincount(e, minlength=256)
        hist_oth += np.bincount(o, minlength=256)
        hist_joint += np.bincount((e.astype(np.int32) << 8) | o, minlength=65536)
        hist_pair += np.bincount((e[:-1].astype(np.int32) << 8) | e[1:], minlength=65536)
        n_total += e.size

        # what actually got written to disk for this block
        f = os.path.join("experiments/df11_qwen3_1.7b",
                         block.replace(".", "_") + ".safetensors")
        if os.path.exists(f):
            bits_exp_total += os.path.getsize(f)  # upper bound; includes the bf16 remainder
        if args.zstd and len(zstd_samples) < 2:
            zstd_samples.append((e.tobytes(), o.tobytes(), e.size))
        del e, o

    def cond(pair_hist, n_ctx):
        """H(Y|X) from a joint histogram."""
        j = pair_hist.reshape(n_ctx, -1)
        tot = j.sum(axis=1, keepdims=True)
        out = 0.0
        for r in range(n_ctx):
            if tot[r, 0]:
                out += tot[r, 0] / n_total * H(j[r])
        return out

    h_exp = H(hist_exp)
    h_oth = H(hist_oth)
    h_joint = H(hist_joint)
    h_oth_given_exp = cond(hist_joint, 256)          # H(other | exp)
    h_exp_given_prev = cond(hist_pair, 256)          # H(exp_i | exp_{i-1})
    # for the pairwise term the weight count is n_total - 1 per block; close enough
    n_pair = n_total - len(args.layers)

    print(f"\nweights analysed: {n_total/1e6:.1f} M  ({len(args.layers)} layers x {len(proj)} projections)")
    print(f"\n  H(exponent byte)                    {h_exp:6.4f} bits   (stored: 2.663 bits, Huffman)")
    print(f"  H(exponent | prev exponent)         {h_exp_given_prev:6.4f} bits"
          f"   -> context saves {h_exp - h_exp_given_prev:+.4f}")
    print(f"  H(sign+mantissa byte)               {h_oth:6.4f} bits   (stored: 8.000 bits raw)")
    print(f"  H(sign+mantissa | exponent)         {h_oth_given_exp:6.4f} bits"
          f"   -> conditioning saves {h_oth - h_oth_given_exp:+.4f}")
    print(f"  H(exp, other) jointly               {h_joint:6.4f} bits")

    cur = 2.663 + 8.000
    best_pairs = h_exp_given_prev + cond(hist_joint, 256)      # order-1 exponent + exp-conditioned other
    print(f"\n  current standing cost                {cur:6.4f} bits/weight")
    print(f"  order-1 achievable (static, lossless){best_pairs:6.4f} bits/weight"
          f"   = {(1-best_pairs/cur)*100:5.1f}% smaller")
    cpart_cur = 1.409286144 + 0.468613348
    print(f"\n  compressed part on disk: {cpart_cur:.4f} GB")
    print(f"  at the achievable rate : {cpart_cur*(best_pairs/cur):.4f} GB"
          f"  (saves {(cpart_cur*(1-best_pairs/cur))*1000:.0f} MB on the 1.7B)")

    if zstd_samples:
        import zstandard as zstd
        c = zstd.ZstdCompressor(level=19)
        print(flush=True)
        for e, o, n in zstd_samples:
            ze = len(c.compress(e))
            zo = len(c.compress(o))
            print(f"  zstd-19 on one layer's streams: exponent {8*ze/n:.3f} bits/w "
                  f"(vs 2.663 stored), other {8*zo/n:.3f} bits/w (vs 8.000 stored)", flush=True)


if __name__ == "__main__":
    main()
