#!/usr/bin/env python
"""
fused_decode_gemv.py -- does decoding weights *inside* the kernel keep the
speed win that compression promises?

The thesis.  Our storage format is 1.36x smaller and bit-exact, but 2.32x
slower, because the decode writes a bf16 scratch array that the GEMM then reads
back.  For a token on the 1.7B the seven weight projections per layer are 100 MB
bf16, so the unfused path moves 0.70 GB (compressed) + 2.82 GB (scratch write) +
2.82 GB (scratch read) = 6.34 GB, while the fused path moves 0.70 GB.  That
difference is the whole argument for fusing.

Measurement discipline (learned the hard way here):

  * This card has 32 MB of L2.  Re-running a single 8-25 MB matrix measures L2,
    not DRAM, and reports impossible bandwidths (>480 GB/s on a ~240 GB/s part).
    So per-projection timings evict L2 with a 128 MB flush before every timed
    call, and the headline number is a whole-token pass over all 28 layers
    (2.82 GB of distinct weights), which cannot fit in cache.
  * CUDA graphs: a real runtime replays a graph of all 196 kernels (Ollama does
    exactly this), so the token measurement is a captured graph, which removes
    launch overhead from every variant equally.

Variants, all computing the same y = x @ W^T on the same real weights:

  bf16      cuBLAS matvec over the original bf16 weights     2.00 B/elem read
  unfused   dequant -> bf16 scratch -> cuBLAS matvec         0.52 + 2 + 2 B/elem
  fused     one Triton kernel: unpack codes in registers, MAC 0.52 B/elem read
  floor     read the packed payload only, discard              0.52 B/elem read

Usage:
  python scripts/fused_decode_gemv.py --layers 4 --token
"""
import argparse
import json
import os

import torch
import triton
import triton.language as tl

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PARENT = {p: ("self_attn" if p.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
              else "mlp") for p in PROJ}
GROUP = 128
FLUSH_MIB = 128            # > 4x this card's 32 MB L2


# ----------------------------------------------------------------- kernels
@triton.jit
def _fused_int4_matvec(codes_ptr, scales_ptr, x_ptr, y_ptr,
                       K: tl.constexpr, G: tl.constexpr,
                       BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """y[n] = sum_k (code[n, k] - 8) * scale[n, k // G] * x[k]"""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        packed = tl.load(codes_ptr + offs_n[:, None] * (K // 2)
                         + (k0 // 2) + tl.arange(0, BLOCK_K // 2)[None, :])
        lo = (packed & 0x0F).to(tl.float32) - 8.0
        hi = ((packed >> 4) & 0x0F).to(tl.float32) - 8.0
        codes = tl.interleave(lo, hi)                      # [BLOCK_N, BLOCK_K]
        scale = tl.load(scales_ptr + offs_n[:, None] * (K // G)
                        + (k0 // G) + offs_k[None, :] // G)
        xv = tl.load(x_ptr + k0 + offs_k)
        acc += codes * scale * xv[None, :]
    tl.store(y_ptr + offs_n, tl.sum(acc, axis=1))


@triton.jit
def _dequant_int4_to_bf16(codes_ptr, scales_ptr, out_ptr,
                          K: tl.constexpr, G: tl.constexpr,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """The unfused step: materialise bf16 weights (this is the scratch write)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        packed = tl.load(codes_ptr + offs_n[:, None] * (K // 2)
                         + (k0 // 2) + tl.arange(0, BLOCK_K // 2)[None, :])
        lo = (packed & 0x0F).to(tl.float32) - 8.0
        hi = ((packed >> 4) & 0x0F).to(tl.float32) - 8.0
        codes = tl.interleave(lo, hi)
        scale = tl.load(scales_ptr + offs_n[:, None] * (K // G)
                        + (k0 // G) + offs_k[None, :] // G)
        tl.store(out_ptr + offs_n[:, None] * K + k0 + offs_k[None, :],
                 (codes * scale).to(tl.bfloat16))


@triton.jit
def _read_only(codes_ptr, scales_ptr, out_ptr,
               K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Traffic floor: touch exactly the payload bytes, compute nothing useful."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    half: tl.constexpr = BLOCK_K // 2
    s = tl.zeros((BLOCK_N, half), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        packed = tl.load(codes_ptr + offs_n[:, None] * (K // 2)
                         + (k0 // 2) + tl.arange(0, half)[None, :])
        s += (packed & 0x0F).to(tl.float32)
    tl.store(out_ptr + offs_n, tl.sum(s, axis=1))


# ----------------------------------------------------------------- helpers
def quantise_int4(w):
    """Per-row, group-wise symmetric int4. Returns (packed, scales, w_deq)."""
    n, k = w.shape
    assert k % GROUP == 0
    wg = w.reshape(n, k // GROUP, GROUP).float()
    scale = (wg.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)
    q = torch.round(wg / scale).clamp(-8, 7).to(torch.int16).reshape(n, k)
    scale = scale.squeeze(-1).to(torch.float16)
    codes = (q + 8).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    w_deq = q.float() * scale.float().repeat_interleave(GROUP, dim=1)
    return packed, scale, w_deq


def timeit_flushed(fn, flush, flush_out, iters, warmup=5):
    """ms per call, with L2 evicted before each timed call.

    The eviction is a large *read* (sum), not a write: zeroing 128 MB leaves
    dirty lines in L2 that get written back to DRAM during the timed kernel and
    steal its bandwidth.  Reads evict without creating writeback traffic.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    total = 0.0
    for _ in range(iters):
        torch.sum(flush, dim=0, out=flush_out)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)
    return total / iters


def graph_timeit(ops, iters, warmup=3):
    """Capture `ops` (a list of callables) as one graph and time replay."""
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for fn in ops:
                fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for fn in ops:
            fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, g


def build_layer(w, dev, block_n):
    """All buffers and callables for one projection weight."""
    n, k = w.shape
    packed, scales, w_deq = quantise_int4(w)
    t = {"n": n, "k": k, "name": None}
    t["packed"] = packed.to(dev)
    t["scales"] = scales.to(dev)
    t["w16"] = w.to(dev)
    t["w_deq"] = w_deq.to(dev)
    t["scratch"] = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
    t["y_bf"] = torch.empty(1, n, dtype=torch.bfloat16, device=dev)
    t["y_out"] = torch.empty(n, dtype=torch.float32, device=dev)
    t["dummy"] = torch.empty(n, dtype=torch.float32, device=dev)
    t["grid"] = ((n + block_n - 1) // block_n,)
    t["x2"] = torch.empty(1, k, dtype=torch.bfloat16, device=dev)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--layers", type=int, default=4,
                    help="layers used for per-projection statistics")
    ap.add_argument("--token", action="store_true",
                    help="also measure a whole 28-layer token pass")
    # Tuned on this card: narrow row blocks with a wide K block minimise the
    # unpack overhead.  Measured per-token fused times (ms):
    #   32/128 8.19 | 16/512 6.63 | 8/1024 5.80 | 4/2048 5.71 | 8/2048 7.20
    # with bf16 a flat 13.3 and the pure-read floor a flat 3.1-3.2.
    ap.add_argument("--block-n", type=int, default=8)
    ap.add_argument("--block-k", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="experiments/runs/fused_decode_gemv.json")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    hf_home = os.environ.get("HF_HOME", ".hf_home")
    snap = args.snapshot or next(
        os.path.join(hf_home, "hub", "models--Qwen--Qwen3-1.7B", "snapshots", d)
        for d in os.listdir(os.path.join(hf_home, "hub",
                                         "models--Qwen--Qwen3-1.7B", "snapshots")))
    from safetensors.torch import load_file
    dev = torch.device("cuda:0")
    block_n, block_k = args.block_n, args.block_k
    print(f"triton {triton.__version__}  device {torch.cuda.get_device_name(0)}"
          f"  block_n={block_n} block_k={block_k}  layers={args.layers}\n")

    all_w = {}
    for f in sorted(f for f in os.listdir(snap) if f.endswith(".safetensors")):
        all_w.update(load_file(os.path.join(snap, f)))
    print(f"loaded {len(all_w)} tensors\n")

    # shaped so the eviction read can use a preallocated out= reduction
    flush = torch.empty((FLUSH_MIB * 1024 * 1024 // 4 // 1024, 1024),
                        dtype=torch.float32, device=dev)
    flush_out = torch.empty(1024, dtype=torch.float32, device=dev)
    rows, packed_bytes_bf16, packed_bytes_q = {}, {}, {}
    print(f"per-projection, L2 flushed before every timed call "
          f"(iters={args.iters})")
    for layer in range(args.layers):
        for proj in PROJ:
            name = f"model.layers.{layer}.{PARENT[proj]}.{proj}.weight"
            w = all_w[name]
            n, k = w.shape
            t = build_layer(w, dev, block_n)
            xg = (torch.randn(k, dtype=torch.bfloat16) * 0.05).to(dev)
            t["x2"].copy_(xg.view(1, -1))

            y_ref = (xg.float() @ t["w_deq"].float().t()).float()
            y16 = (xg.float() @ t["w16"].float().t()).float()

            def run_bf16():
                torch.matmul(t["x2"], t["w16"].t(), out=t["y_bf"])

            def run_unfused():
                _dequant_int4_to_bf16[t["grid"]](
                    t["packed"], t["scales"], t["scratch"], K=k, G=GROUP,
                    BLOCK_N=block_n, BLOCK_K=block_k)
                torch.matmul(t["x2"], t["scratch"].t(), out=t["y_bf"])

            def run_fused():
                _fused_int4_matvec[t["grid"]](
                    t["packed"], t["scales"], t["x2"].view(-1), t["y_out"],
                    K=k, G=GROUP, BLOCK_N=block_n, BLOCK_K=block_k)

            def run_floor():
                _read_only[t["grid"]](t["packed"], t["scales"], t["dummy"],
                                      K=k, BLOCK_N=block_n, BLOCK_K=block_k)

            run_fused()
            run_bf16()
            run_unfused()
            torch.cuda.synchronize()
            rel_fused = (t["y_out"] - y_ref).abs().max().item() \
                / y_ref.abs().max().item()
            rel_bf16 = (t["y_bf"].view(-1).float() - y16).abs().max().item() \
                / y16.abs().max().item()
            rel_unfused = (t["y_bf"].view(-1).float()
                           - (t["x2"].float() @ t["scratch"].float().t())
                           .view(-1)).abs().max().item() \
                / max(y_ref.abs().max().item(), 1e-6)
            qerr = (y_ref - y16).abs().max().item() / y16.abs().max().item()

            ms = {}
            for kind, fn in (("bf16", run_bf16), ("unfused", run_unfused),
                             ("fused", run_fused), ("floor", run_floor)):
                ms[kind] = timeit_flushed(fn, flush, flush_out, args.iters)

            elems = n * k
            byts = {"bf16": elems * 2,
                    "unfused": elems * 0.5 + elems * 2 + elems * 2,
                    "fused": elems * 0.5,
                    "floor": elems * 0.5}
            gbps = {kk: byts[kk] / 1e9 / (ms[kk] / 1e3) for kk in ms}
            rows[(layer, proj)] = {"layer": layer, "proj": proj, "n": n, "k": k,
                                   "ms": ms, "bytes": byts, "gbps": gbps,
                                   "rel_fused": rel_fused, "rel_bf16": rel_bf16,
                                   "rel_unfused": rel_unfused, "quant_err": qerr}
            print(f"  L{layer} {proj:10s} {n:5d}x{k:<5d} "
                  f"bf16 {ms['bf16']*1000:6.1f}us  fused {ms['fused']*1000:6.1f}us"
                  f"  unfused {ms['unfused']*1000:6.1f}us"
                  f"  floor {ms['floor']*1000:6.1f}us"
                  f"  [{gbps['bf16']:5.0f}/{gbps['fused']:4.0f} GB/s]"
                  f"  err fused {rel_fused:.1e} bf16 {rel_bf16:.1e}", flush=True)
            del t, w, xg, y_ref, y16
            torch.cuda.empty_cache()

    # aggregate over layer 0's seven projections -> per-token, 28 layers
    layer0 = [r for (l, _), r in rows.items() if l == 0]
    agg = {kk: sum(r["ms"][kk] for r in layer0) for kk in
           ("bf16", "fused", "unfused", "floor")}
    gb = {kk: sum(r["bytes"][kk] for r in layer0) / 1e9 for kk in agg}
    print("\nper layer, 7 projections (L2-flushed, launch overhead included):")
    for kk in ("bf16", "fused", "unfused", "floor"):
        print(f"  {kk:8s} {agg[kk]:7.3f} ms/layer  {agg[kk]*28:8.2f} ms/token(28L)"
              f"  {gb[kk]:5.3f} GB read  {gb[kk]/(agg[kk]/1e3):6.1f} GB/s")
    print(f"    fused/bf16 {agg['bf16']/agg['fused']:.2f}x   "
          f"fused/unfused {agg['unfused']/agg['fused']:.2f}x   "
          f"fused/floor {agg['fused']/agg['floor']:.2f}x")

    result = {"config": vars(args), "per_projection": {
        f"L{l}.{p}": v for (l, p), v in rows.items()},
        "per_layer_ms": agg, "per_token_ms_28L": {k: v * 28 for k, v in agg.items()},
        "per_layer_gb": gb,
        "max_rel_fused": max(v["rel_fused"] for v in rows.values()),
        "max_rel_bf16": max(v["rel_bf16"] for v in rows.values())}

    if args.token:
        print("\nwhole token: 28 layers x 7 projections, one CUDA graph")
        n_layers = 28
        dev = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        # Only what the three variants actually need: the fp32 dequantised
        # reference (w_deq) and per-projection scratch buffers are NOT kept --
        # keeping them resident over 196 projections overflowed VRAM and made
        # Windows page to system RAM, which silently produced 7 GB/s numbers.
        token_valid = True
        built, scratch_by_shape = [], {}
        for layer in range(n_layers):
            for proj in PROJ:
                name = f"model.layers.{layer}.{PARENT[proj]}.{proj}.weight"
                w = all_w[name]
                n, k = w.shape
                packed, scales, _ = quantise_int4(w)
                t = {"proj": proj, "n": n, "k": k,
                     "packed": packed.to(dev), "scales": scales.to(dev),
                     "w16": w.to(dev),
                     "y_bf": torch.empty(1, n, dtype=torch.bfloat16, device=dev),
                     "y_out": torch.empty(n, dtype=torch.float32, device=dev),
                     "dummy": torch.empty(n, dtype=torch.float32, device=dev),
                     "grid": ((n + block_n - 1) // block_n,),
                     "x2": torch.empty(1, k, dtype=torch.bfloat16, device=dev)}
                t["x2"].copy_(torch.randn(k, dtype=torch.bfloat16, device=dev)
                              .mul(0.05).view(1, -1))
                if (n, k) not in scratch_by_shape:
                    scratch_by_shape[(n, k)] = torch.empty(
                        (n, k), dtype=torch.bfloat16, device=dev)
                t["scratch"] = scratch_by_shape[(n, k)]
                built.append(t)
                del w, packed, scales
        total_bf16 = sum(t["n"] * t["k"] * 2 for t in built)
        total_q = sum(t["n"] * t["k"] * 0.5 for t in built)
        peak = torch.cuda.max_memory_allocated() / (1 << 30)
        print(f"  resident: {len(built)} projections, "
              f"{total_bf16/1e9:.2f} GB bf16 / {total_q/1e9:.2f} GB packed"
              f"  + {len(scratch_by_shape)} shared scratch buffers"
              f"  |  peak VRAM {peak:.2f} GiB")
        if peak > 7.0:
            # Above ~7 GiB the card is oversubscribed, Windows pages to system
            # RAM, and the timings become meaningless-but-plausible (we have
            # measured 401 ms/token this way).  Promptly refuse to report them.
            print("  !! peak VRAM above 7 GiB: timings would be "
                  "paging-contaminated, so the token section is skipped")
            result["token_ms"] = {"invalid": f"peak VRAM {peak:.2f} GiB"}
            token_valid = False

        def mk(kind):
            if kind == "bf16":
                return [lambda t=t: torch.matmul(t["x2"], t["w16"].t(),
                                                 out=t["y_bf"]) for t in built]
            if kind == "unfused":
                def fn(t):
                    _dequant_int4_to_bf16[t["grid"]](
                        t["packed"], t["scales"], t["scratch"], K=t["k"],
                        G=GROUP, BLOCK_N=block_n, BLOCK_K=block_k)
                    torch.matmul(t["x2"], t["scratch"].t(), out=t["y_bf"])
                return [lambda t=t: fn(t) for t in built]
            if kind == "fused":
                def fn(t):
                    _fused_int4_matvec[t["grid"]](
                        t["packed"], t["scales"], t["x2"].view(-1),
                        t["y_out"], K=t["k"], G=GROUP, BLOCK_N=block_n,
                        BLOCK_K=block_k)
                return [lambda t=t: fn(t) for t in built]
            def fn(t):
                _read_only[t["grid"]](t["packed"], t["scales"], t["dummy"],
                                      K=t["k"], BLOCK_N=block_n,
                                      BLOCK_K=block_k)
            return [lambda t=t: fn(t) for t in built]

        tok = {}
        if token_valid:
            for kind in ("bf16", "fused", "unfused", "floor"):
                ms, _ = graph_timeit(mk(kind), iters=max(5, args.iters // 3))
                tok[kind] = ms
                read_gb = (total_q if kind in ("fused", "floor")
                           else total_bf16) / 1e9
                print(f"  {kind:8s} {ms:7.2f} ms/token   "
                      f"({read_gb:.2f} GB / {ms/1e3:.3f} s = "
                      f"{read_gb/(ms/1e3):5.1f} GB/s)", flush=True)
            print(f"    fused/bf16 {tok['bf16']/tok['fused']:.2f}x   "
                  f"fused/unfused {tok['unfused']/tok['fused']:.2f}x")
            result["token_ms"] = tok

    worst = max(v["rel_fused"] for v in rows.values())
    print(f"\ncorrectness: fused vs fp32 reference, worst relative error "
          f"{worst:.2e} over {len(rows)} projections")
    print(f"sanity     : bf16 path vs fp32 reference "
          f"{max(v['rel_bf16'] for v in rows.values()):.2e} (bf16 rounding)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
