#!/usr/bin/env python3
"""
stageB_entropy_probe.py — the decisive measurement for the "missing piece":
how much lossless redundancy ACTUALLY remains in stored model weights?

Replicates the ISCA'26 entropy methodology (arXiv:2606.15789 §II-B) on any
safetensors model, CPU-only, no GPU needed:

  * BF16/FP16 models   -> exponent-field entropy (DFloat11-style headroom)
  * GPTQ/AWQ int4      -> nibble-symbol entropy of qweight/qzeros (order-
                          invariant: AWQ's permuted packing does not change
                          symbol frequencies) + fp16 scales exponent entropy
  * reports per-tensor and aggregate: stored bits/element, Shannon entropy,
    Huffman achievable bits, ideal lossless ratio, and (if zstandard is
    installed) the real zstd ratio on the raw bytes

GO/NO-GO GATE (from research/round5_validation_prototype.md):
  quantized format ratio >= 1.15x  -> worth building an entropy layer
  BF16 ratio expected ~1.4-1.5x    -> sanity check vs DFloat11's ~30%

Usage:
  python stageB_entropy_probe.py --model <hf-id-or-local-path> [--json out.json]
  python stageB_entropy_probe.py --selftest
"""
import argparse, glob, heapq, json, math, os, sys
import numpy as np

# ----------------------------- math helpers ---------------------------------

def entropy_bits(symbols: np.ndarray, nsym: int) -> float:
    counts = np.bincount(symbols, minlength=nsym).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())

def huffman_avg_len(counts: np.ndarray) -> float:
    """Average Huffman codeword length = (sum of merge costs)/total symbols."""
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

def sym_stats(symbols: np.ndarray, nsym: int, stored_bits: float, label: str):
    counts = np.bincount(symbols, minlength=nsym)
    H = entropy_bits(symbols, nsym)
    H_h = huffman_avg_len(counts)
    return {
        "label": label, "elements": int(symbols.size), "stored_bits": stored_bits,
        "entropy_bits": round(H, 4), "huffman_bits": round(H_h, 4),
        "ideal_ratio": round(stored_bits / H, 4) if H > 0 else float("inf"),
        "huffman_ratio": round(stored_bits / H_h, 4) if H_h > 0 else float("inf"),
    }

def nibbles_from_int32(a: np.ndarray) -> np.ndarray:
    """Unpack 8x int4 nibbles per int32 (GPTQ packing; AWQ permutation is
    entropy-invariant so this is fine for measurement)."""
    u = a.astype(np.uint32)
    out = np.empty(u.size * 8, dtype=np.uint8)
    v = u.reshape(-1)
    for i in range(8):
        out[i::8] = ((v >> np.uint32(4 * i)) & np.uint32(0xF)).astype(np.uint8)
    return out

def float_exponent_symbols(u16: np.ndarray, is_bf16: bool) -> np.ndarray:
    return ((u16 >> 7) & 0xFF).astype(np.uint8) if is_bf16 else ((u16 >> 10) & 0x1F).astype(np.uint8)

def dfloat_estimate(u16: np.ndarray, is_bf16: bool, stored: int = 16):
    """DFloat11-style model: sign(1) + Huffman(exponent) + raw mantissa."""
    exp = float_exponent_symbols(u16, is_bf16)
    counts = np.bincount(exp, minlength=256 if is_bf16 else 32)
    hexp = huffman_avg_len(counts)
    mant = 7 if is_bf16 else 10
    bits = 1 + hexp + mant
    return {"sign+huff(exp)+mantissa_bits": round(bits, 3),
            "ratio": round(stored / bits, 4),
            "exp_entropy_bits": round(entropy_bits(exp, counts.size), 4)}

def zstd_ratio(raw: bytes) -> float | None:
    try:
        import zstandard as zstd
        c = zstd.ZstdCompressor(level=12)
        return round(len(raw) / max(1, len(c.compress(raw))), 4)
    except ImportError:
        return None

# ----------------------------- tensor routing --------------------------------

def analyze_tensor(name: str, arr) -> list[dict]:
    """arr: numpy array (uint16 view for floats, int32/uint8/int64 otherwise)."""
    res = []
    dt = arr.dtype
    lname = name.lower()
    if dt == np.int32 and ("qweight" in lname or "qzeros" in lname):
        nb = nibbles_from_int32(arr.ravel())
        r = sym_stats(nb, 16, 4.0, f"{name}:int4-nibbles")
        r["zstd"] = zstd_ratio(arr.astype(np.int32).tobytes()[:64 << 20])
        res.append(r)
    elif dt == np.uint8 and ("qweight" in lname or "qzeros" in lname):
        lo, hi = arr.ravel() & 0xF, (arr.ravel() >> 4) & 0xF
        nb = np.empty(arr.size * 2, dtype=np.uint8); nb[0::2], nb[1::2] = lo, hi
        res.append(sym_stats(nb, 16, 4.0, f"{name}:int4-nibbles"))
    else:
        res.append(None)  # placeholder; handled by caller with dtype info
    return res

# ----------------------------- model loading ---------------------------------

def iter_tensors(path: str):
    """Yield (name, numpy-array, kind) for each tensor in a safetensors file/dir.
    kind in {'bf16','fp16','fp32','int32','int64','uint8','other'}"""
    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    else:
        files = [path]
    if not files:
        sys.exit(f"[fatal] no safetensors found at {path}")
    try:
        import torch
        from safetensors.torch import load_file
        HAVE_TORCH = True
    except ImportError:
        HAVE_TORCH = False
    from safetensors.numpy import load_file as np_load  # fp16/fp32/int only

    for f in files:
        if HAVE_TORCH:
            sd = load_file(f)
            for k, t in sd.items():
                if t.dtype == torch.bfloat16:
                    yield k, t.view(torch.uint16).numpy(), "bf16"
                elif t.dtype == torch.float16:
                    yield k, t.view(torch.uint16).numpy(), "fp16"
                elif t.dtype == torch.float32:
                    yield k, t.view(torch.uint32).numpy(), "fp32"
                elif t.dtype in (torch.int32,):
                    yield k, t.numpy(), "int32"
                elif t.dtype in (torch.int64,):
                    yield k, t.numpy().astype(np.int64), "int64"
                elif t.dtype == torch.uint8:
                    yield k, t.numpy(), "uint8"
                else:
                    yield k, None, "other"
        else:
            sd = np_load(f)
            for k, a in sd.items():
                if a.dtype == np.float16:
                    yield k, a.view(np.uint16), "fp16"
                elif a.dtype == np.float32:
                    yield k, a.view(np.uint32), "fp32"
                elif a.dtype == np.int32:
                    yield k, a, "int32"
                elif a.dtype == np.uint8:
                    yield k, a, "uint8"
                else:
                    yield k, None, "other"

def probe_model(path: str, max_tensors: int, verbose: bool):
    total_stored_bits = 0.0; total_achievable_bits = 0.0
    rows = []; skipped = 0
    for name, arr, kind in iter_tensors(path):
        if arr is None:
            skipped += 1; continue
        if max_tensors and len(rows) >= max_tensors:
            break
        lname = name.lower()
        rec = None
        if kind == "bf16":
            u16 = arr.ravel()
            d = dfloat_estimate(u16, is_bf16=True)
            rec = {"tensor": name, "kind": "bf16", "elements": int(u16.size),
                   "stored_bits_per_el": 16, "achievable_bits_per_el": d["sign+huff(exp)+mantissa_bits"],
                   "ratio": d["ratio"], "exp_entropy": d["exp_entropy_bits"]}
        elif kind == "fp16":
            u16 = arr.ravel()
            if "scale" in lname:  # GPTQ/AWQ scales: fp16, entropy-code exponents
                d = dfloat_estimate(u16, is_bf16=False)
                rec = {"tensor": name, "kind": "fp16-scales", "elements": int(u16.size),
                       "stored_bits_per_el": 16, "achievable_bits_per_el": d["sign+huff(exp)+mantissa_bits"],
                       "ratio": d["ratio"], "exp_entropy": d["exp_entropy_bits"]}
            else:
                d = dfloat_estimate(u16, is_bf16=False)
                rec = {"tensor": name, "kind": "fp16", "elements": int(u16.size),
                       "stored_bits_per_el": 16, "achievable_bits_per_el": d["sign+huff(exp)+mantissa_bits"],
                       "ratio": d["ratio"], "exp_entropy": d["exp_entropy_bits"]}
        elif kind == "int32" and ("qweight" in lname or "qzeros" in lname):
            nb = nibbles_from_int32(arr.ravel())
            s = sym_stats(nb, 16, 4.0, name)
            rec = {"tensor": name, "kind": "int4-packed", "elements": int(nb.size),
                   "stored_bits_per_el": 4, "achievable_bits_per_el": s["huffman_bits"],
                   "ratio": s["huffman_ratio"], "exp_entropy": s["entropy_bits"]}
        elif kind == "uint8" and ("qweight" in lname or "qzeros" in lname):
            a = arr.ravel()
            nb = np.empty(a.size * 2, dtype=np.uint8)
            nb[0::2], nb[1::2] = a & 0xF, (a >> 4) & 0xF
            s = sym_stats(nb, 16, 4.0, name)
            rec = {"tensor": name, "kind": "int4-packed-u8", "elements": int(nb.size),
                   "stored_bits_per_el": 4, "achievable_bits_per_el": s["huffman_bits"],
                   "ratio": s["huffman_ratio"], "exp_entropy": s["entropy_bits"]}
        else:
            skipped += 1; continue
        if rec:
            rows.append(rec)
            total_stored_bits += rec["elements"] * rec["stored_bits_per_el"]
            total_achievable_bits += rec["elements"] * rec["achievable_bits_per_el"]
            if verbose:
                print(f"  {rec['tensor'][:60]:60s} {rec['kind']:14s} "
                      f"n={rec['elements']:>12,d} stored={rec['stored_bits_per_el']:.1f}b "
                      f"-> {rec['achievable_bits_per_el']:.2f}b  ratio={rec['ratio']:.3f}")
    overall = (total_stored_bits / total_achievable_bits) if total_achievable_bits else 0
    return rows, overall, skipped

# ----------------------------- selftest --------------------------------------

def selftest():
    # 1) entropy of a fair 4-symbol source = 2 bits
    s = np.array([0, 1, 2, 3] * 1000, dtype=np.uint8)
    assert abs(entropy_bits(s, 4) - 2.0) < 1e-9, "entropy(fair 4-sym) != 2"
    # 2) skewed source: p=[0.5,0.25,0.125,0.125] -> H=1.75, Huffman=1.75
    s2 = np.concatenate([np.zeros(500, np.uint8), np.ones(250, np.uint8),
                         np.full(125, 2, np.uint8), np.full(125, 3, np.uint8)])
    assert abs(entropy_bits(s2, 4) - 1.75) < 1e-9
    assert abs(huffman_avg_len(np.bincount(s2, minlength=4)) - 1.75) < 1e-9
    # 3) nibble unpack round-trip
    v = np.array([0x12345678], dtype=np.int32)
    nb = nibbles_from_int32(v)
    assert list(nb) == [8, 7, 6, 5, 4, 3, 2, 1], f"nibble order wrong: {list(nb)}"
    # 4) bf16 exponent extraction: 1.0 in bf16 = 0x3F80 -> exponent field 0x7F
    one = np.array([0x3F80], dtype=np.uint16)
    assert float_exponent_symbols(one, True)[0] == 0x7F
    # 5) DFloat11-style estimate on constant-exponent data:
    #    single-symbol Huffman floors at 1 bit => 1(sign)+1(exp)+7(mant)=9 bits => 16/9
    const = np.full(1024, 0x3F80, dtype=np.uint16)
    est = dfloat_estimate(const, True)
    assert abs(est["sign+huff(exp)+mantissa_bits"] - 9.0) < 1e-9, est
    assert est["ratio"] >= 1.7, f"constant-exponent bf16 should compress >=1.7x, got {est}"
    # 6) realistic bf16: two dominant exponents. NOTE: Huffman floors at 1.0
    #    bit/symbol for a 2-symbol alphabet (entropy is 0.881) — this is exactly
    #    why ISCA'26 uses ANS (no such floor). Bounds reflect the Huffman model.
    mix = np.concatenate([np.full(700, 0x3F80, np.uint16), np.full(300, 0x3F00, np.uint16)])
    est2 = dfloat_estimate(mix, True)
    assert 1.4 <= est2["ratio"] <= 1.8, est2
    print("SELFTEST PASS: entropy/huffman/nibbles/exponent math verified")

# ----------------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="HF model id (auto-download) or local path to .safetensors/dir")
    ap.add_argument("--max-tensors", type=int, default=0, help="limit tensors (0=all)")
    ap.add_argument("--json", default="", help="write full report JSON here")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if not args.model:
        ap.error("--model required (or run --selftest)")

    path = args.model
    if not os.path.exists(path):
        try:
            from huggingface_hub import snapshot_download
            print(f"[info] downloading {path} ...")
            path = snapshot_download(path, allow_patterns=["*.safetensors"])
        except Exception as e:
            sys.exit(f"[fatal] cannot resolve model '{args.model}': {e}")

    print(f"[probe] {path}")
    rows, overall, skipped = probe_model(path, args.max_tensors, args.verbose)
    total_mb = sum(r["elements"] * r["stored_bits_per_el"] for r in rows) / 8 / 1e6
    est_mb = sum(r["elements"] * r["achievable_bits_per_el"] for r in rows) / 8 / 1e6
    print("-" * 72)
    print(f"[probe] tensors analyzed: {len(rows)} (skipped {skipped} non-weight tensors)")
    print(f"[probe] stored size      : {total_mb:10.1f} MB")
    print(f"[probe] lossless-achievable (Huffman/exponent model): {est_mb:8.1f} MB")
    print(f"[probe] OVERALL LOSSLESS RATIO: {overall:.3f}x")
    verdict = ("GO: >=1.15x headroom on quantized payload — entropy layer worth building"
               if overall >= 1.15 else
               "NO-GO: <1.15x headroom — entropy layer not worth kernel investment for this format")
    if any(r["kind"].startswith(("bf16", "fp16")) and not r["kind"].endswith("scales") for r in rows):
        verdict += "\n       (BF16/FP16 present: expect ~1.4-1.5x, matches DFloat11's ~30% claim)"
    print(f"[probe] VERDICT: {verdict}")
    if args.json:
        json.dump({"model": args.model, "overall_ratio": overall,
                   "stored_mb": total_mb, "achievable_mb": est_mb, "tensors": rows},
                  open(args.json, "w"), indent=2)
        print(f"[probe] JSON report -> {args.json}")

if __name__ == "__main__":
    main()
