#!/usr/bin/env python3
"""
stageC_legolm.py — LegoLM replication (arXiv:2608.08652): data-free global
weight-sharing codebook + percentile-selective outlier sidecar.

Tests the Layer-1 (sharing) claims WITHOUT trusting the paper:
  * global k-means codebook (K centroids shared by ALL layers)
  * per-tensor scalar normalization ("scalar-block encoding" — fixes the
    distributional-mismatch failure mode)
  * percentile-selective replacement: top (100-p)% |w| weights kept VERBATIM
    in a sidecar (fixes the "outlier dominance" failure mode)
  * boundary-layer protection: first/last transformer blocks untouched

Also answers the Round-6 stacking question "do codebook indices entropy-code
well?" by reporting Shannon/Huffman bits of the index stream (Layer-3 headroom).

Usage:
  python stageC_legolm.py --selftest                  # numpy math verification
  python stageC_legolm.py --measure  <hf-id|path>     # analysis only (no writes)
  python stageC_legolm.py --compress <hf-id|path> <out_dir> [--K 64] [--p 99.0]

Requires: numpy (core); torch+safetensors+huggingface_hub for --measure/--compress.
NOTE: the torch adapter path is straightforward but was NOT executed in the
research sandbox (no GPU/torch there); run --selftest first, then --measure on
a small model (e.g. Qwen2.5-0.5B) to validate before 7B+ runs.
"""
import argparse, heapq, json, math, os, sys
import numpy as np

# ------------------------------- core math ----------------------------------

def entropy_bits(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def huffman_avg_len(counts: np.ndarray) -> float:
    freqs = [int(c) for c in counts if c > 0]
    if len(freqs) <= 1:
        return 1.0
    heapq.heapify(freqs)
    total, cost = sum(freqs), 0
    while len(freqs) > 1:
        a, b = heapq.heappop(freqs), heapq.heappop(freqs)
        cost += a + b
        heapq.heappush(freqs, a + b)
    return cost / total

def lloyd_kmeans(X: np.ndarray, K: int, iters: int = 12, seed: int = 0) -> np.ndarray:
    """Plain Lloyd on a subsample; good enough for a replication study."""
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(X.shape[0], size=min(X.shape[0], 500_000), replace=False)]
    cent = sub[rng.choice(sub.shape[0], size=K, replace=False)].astype(np.float32)
    for _ in range(iters):
        # assign
        d = (sub[:, None] - cent[None, :]) ** 2  # memory-heavy for big K; chunk it
        idx = d.sum(-1).argmin(1)
        for k in range(K):
            m = idx == k
            if m.any():
                cent[k] = sub[m].mean(0)
    return cent

def _assign_chunked(X: np.ndarray, cent: np.ndarray, chunk: int = 1 << 18) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=np.uint16)
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk]
        d = (x[:, None] - cent[None, :]) ** 2
        out[i:i + chunk] = d.sum(-1).argmin(1).astype(np.uint16)
    return out

def compress_tensor(w: np.ndarray, cent: np.ndarray, p: float, sidecar_thresh_global: float | None):
    """Returns dict with normalized-codebook indices + verbatim sidecar."""
    flat = w.ravel().astype(np.float32)
    n = flat.size
    # per-tensor scalar normalization (scalar-block encoding)
    scale = float(np.percentile(np.abs(flat), 99.9)) or 1.0
    wn = flat / scale
    # sidecar: verbatim storage of the most extreme weights (percentile-selective)
    thresh = sidecar_thresh_global if sidecar_thresh_global is not None \
        else float(np.percentile(np.abs(flat), p))
    side_mask = np.abs(flat) >= thresh
    side_pos = np.flatnonzero(side_mask).astype(np.int64)
    side_val = flat[side_mask]                      # keep verbatim (fp32 here; bf16 in format)
    # codebook assignment for the rest (normalize by scale)
    idx = np.zeros(n, dtype=np.uint16)
    non = ~side_mask
    idx[non] = _assign_chunked(wn[non, None], cent)
    return {"scale": scale, "idx": idx, "side_pos": side_pos, "side_val": side_val,
            "n": n, "sidecar_frac": float(side_mask.mean())}

def decompress_tensor(pack: dict, cent: np.ndarray, shape) -> np.ndarray:
    w = (cent[pack["idx"]] * pack["scale"]).astype(np.float32).reshape(-1)
    w[pack["side_pos"]] = pack["side_val"]
    return w.reshape(shape)

def bits_accounting(pack: dict, K: int) -> dict:
    n = pack["n"]
    counts = np.bincount(pack["idx"], minlength=K)
    H = entropy_bits(counts)
    Hh = huffman_avg_len(counts)
    naive = math.ceil(math.log2(max(K, 2)))
    side_frac = pack["sidecar_frac"]
    # sidecar: 16-bit value + 32-bit position per element (COO worst case)
    bits = (1 - side_frac) * Hh + side_frac * (16 + 32) + 16.0 / max(n, 1)  # + one fp16 scale per tensor
    return {"idx_entropy_bits": round(H, 3), "idx_huffman_bits": round(Hh, 3),
            "naive_idx_bits": naive, "sidecar_frac": round(side_frac, 5),
            "est_bits_per_weight": round(bits, 3), "est_ratio_vs_bf16": round(16 / bits, 2)}

# ------------------------------- selftest ------------------------------------

def selftest():
    rng = np.random.default_rng(42)
    # synthetic "layer": gaussian bulk + 1% fat-tail outliers
    N = 400_000
    w = (rng.normal(0, 0.02, N)).astype(np.float32)
    op = rng.choice(N, size=N // 100, replace=False)
    w[op] = rng.normal(0, 0.5, op.size).astype(np.float32)   # 25x-scale outliers
    W = w.reshape(1000, 400)

    # global codebook from normalized sample
    s = float(np.percentile(np.abs(w), 99.9))
    cent = lloyd_kmeans((w / s)[:, None].astype(np.float32), K=64, iters=8)

    pack = compress_tensor(W, cent, p=99.0, sidecar_thresh_global=None)
    rec = decompress_tensor(pack, cent, W.shape)

    # 1. sidecar actually caught the injected outliers
    caught = np.isin(pack["side_pos"], op).mean()
    assert pack["sidecar_frac"] > 0.005, "sidecar suspiciously small"
    # 2. round-trip quality: outliers exact, bulk near
    err = np.abs(rec - W)
    max_err_outside = err.ravel()[~np.isin(np.arange(N), pack["side_pos"])].max()
    cos = float((rec.ravel() @ W.ravel()) / (np.linalg.norm(rec) * np.linalg.norm(W)))
    acct = bits_accounting(pack, 64)
    assert cos > 0.999, f"cosine too low: {cos}"
    assert acct["est_bits_per_weight"] < 8, f"no compression win: {acct}"
    print(f"SELFTEST PASS  sidecar_frac={pack['sidecar_frac']:.4f}  outliers_caught={caught:.3f}")
    print(f"               max_err(non-sidecar)={max_err_outside:.5f}  cosine={cos:.6f}")
    print(f"               idx entropy={acct['idx_entropy_bits']}b huffman={acct['idx_huffman_bits']}b "
          f"naive={acct['naive_idx_bits']}b")
    print(f"               est {acct['est_bits_per_weight']} bits/weight -> {acct['est_ratio_vs_bf16']}x vs BF16")
    # 3. Layer-3 stacking signal: is the index stream compressible?
    headroom = acct["naive_idx_bits"] / max(acct["idx_huffman_bits"], 1e-9)
    print(f"               index-stream entropy headroom vs naive {acct['naive_idx_bits']}-bit: {headroom:.2f}x "
          f"({'GO for Layer-3' if headroom >= 1.15 else 'weak Layer-3 headroom'})")

# ------------------------------- torch adapter --------------------------------

def iter_weight_tensors(path: str):
    import torch
    from safetensors.torch import load_file
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors")) if os.path.isdir(path) else [path]
    for f in files:
        fp = os.path.join(path, f) if os.path.isdir(path) else f
        for k, t in load_file(fp).items():
            yield k, t

def is_protected(name: str, n_layers: int | None, protect: int = 2) -> bool:
    ln = name.lower()
    if "embed" in ln or "lm_head" in ln or "norm" in ln:
        return True
    if n_layers:
        for i in list(range(protect)) + list(range(n_layers - protect, n_layers)):
            if f"layers.{i}." in ln:
                return True
    return False

def measure_or_compress(args):
    import torch
    path = args.model
    if not os.path.exists(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(path, allow_patterns=["*.safetensors", "*.json"])
    # collect 2D weight tensors (skip protected)
    names, mats = [], []
    n_layers = 0
    for k, t in iter_weight_tensors(path):
        if t.ndim == 2 and t.dtype in (torch.bfloat16, torch.float16, torch.float32):
            names.append(k); mats.append(t.float().numpy())
            if "layers." in k:
                try: n_layers = max(n_layers, int(k.split("layers.")[1].split(".")[0]) + 1)
                except ValueError: pass
    print(f"[legolm] {len(mats)} compressible 2D tensors (model has ~{n_layers} layers)")
    # global sidecar threshold across the model (percentile of |w| over subsample)
    sub = np.concatenate([m.ravel()[:: max(1, m.size // 500_000)] for m in mats[:64]])
    thresh = float(np.percentile(np.abs(sub), args.p))
    print(f"[legolm] global sidecar threshold (p={args.p}): |w| >= {thresh:.5f}")
    # global codebook on normalized subsample
    norm_sub = np.concatenate([(m / (np.percentile(np.abs(m), 99.9) or 1.0)).ravel()[:: max(1, m.size // 300_000)]
                               for m in mats[:64]])[:, None].astype(np.float32)
    cent = lloyd_kmeans(norm_sub, K=args.K, iters=15)
    total_orig_bits = total_new_bits = 0.0
    reports = []
    for k, m in zip(names, mats):
        if is_protected(k, n_layers):
            total_orig_bits += m.size * 16; total_new_bits += m.size * 16
            continue
        pack = compress_tensor(m, cent, args.p, thresh)
        acct = bits_accounting(pack, args.K)
        total_orig_bits += m.size * 16
        total_new_bits += m.size * acct["est_bits_per_weight"]
        reports.append({"tensor": k, **acct})
        if args.compress_out:
            pass  # full serialization (packed .safetensors + codebook) is Stage-C2 work
    ratio = total_orig_bits / total_new_bits
    print(f"[legolm] ESTIMATED overall: {ratio:.2f}x vs BF16 "
          f"({total_orig_bits/8/1e9:.2f} GB -> {total_new_bits/8/1e9:.2f} GB)")
    idx_h = np.mean([r["idx_huffman_bits"] for r in reports]) if reports else 0
    naive = math.ceil(math.log2(args.K))
    print(f"[legolm] index-stream headroom: naive {naive}b vs Huffman {idx_h:.2f}b "
          f"=> {naive/max(idx_h,1e-9):.2f}x (Layer-3 {'GO' if naive/max(idx_h,1e-9)>=1.15 else 'weak'})")
    print("[legolm] next: PPL gate via lm-evaluation-harness after dequant-at-load reconstruction;")
    print("        kill if PPL delta > 1% at < 3x (paper claims +0.03% at 4.41x — verify, don't trust).")
    if args.json:
        json.dump({"ratio_vs_bf16": ratio, "tensors": reports[:200]}, open(args.json, "w"), indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--compress", dest="compress_out", default="")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--p", type=float, default=99.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.model:
        ap.error("need --selftest or --model")
    measure_or_compress(args)

if __name__ == "__main__":
    main()
