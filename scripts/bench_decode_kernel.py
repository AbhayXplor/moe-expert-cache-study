#!/usr/bin/env python
"""
bench_decode_kernel.py -- measure the DFloat11 `decode` kernel in isolation.

Why: inside the model the profiler attributes 49.5 ms/token to `decode` for
Qwen3-1.7B and 74.6 ms/token for Qwen3-4B, but the effective bandwidth implied
by those two numbers is wildly different (96 vs 164 GB/s). Profiler attribution
over 28-36 sequential launches is easy to get wrong, so measure the kernel
directly on real layer buffers, back to back, with CUDA events.

For each layer file we time:
  * the real decode kernel at its production (grid, block, shared_mem)
  * the same kernel with grid scaled x2 / x4 (more blocks in flight)
  * the pure-traffic floor: read sign_mantissa u8 -> write bf16 of the same size

and report GB/s so traffic cost and entropy cost can be separated.

Usage:
  python bench_decode_kernel.py --dir experiments/df11_qwen3_1.7b --layers 0 1 2 --iters 30
  python bench_decode_kernel.py --dir <4B snapshot> --layers 0 1 --iters 30
"""
import argparse, glob, math, os, time

import torch


def find_layer_files(d):
    out = {}
    for f in glob.glob(os.path.join(d, "model_layers_*.safetensors")):
        idx = int(os.path.basename(f).rsplit("_", 1)[1].split(".")[0])
        out[idx] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    import cupy as cp
    import dfloat11.dfloat11 as D
    from safetensors.torch import load_file

    dev = torch.device("cuda:0")
    files = find_layer_files(args.dir)
    print(f"dir {args.dir}: {len(files)} layer files "
          f"(using {args.layers})", flush=True)

    def timed(fn, iters, warmup):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters          # ms

    for li in args.layers:
        sd = load_file(files[li])
        base = f"model.layers.{li}"
        enc = sd[f"{base}.encoded_exponent"].to(dev)
        oth = sd[f"{base}.sign_mantissa"].to(dev)
        lut = sd[f"{base}.luts"].to(dev)
        opos = sd[f"{base}.output_positions"].to(dev)
        gaps = sd[f"{base}.gaps"].to(dev)
        del sd

        n_el, n_bytes, n_luts = oth.numel(), enc.numel(), lut.shape[0]
        out = torch.empty(n_el, dtype=torch.bfloat16, device=dev)
        opos_np = opos.view(torch.uint32).cpu().numpy()
        smem = 512 * 4 + 4 + int((opos_np[1:] - opos_np[:-1]).max()) * 2
        g1 = int(math.ceil(n_bytes / (512 * 8)))

        read_gb = (n_bytes + n_el + gaps.numel() + opos.numel()) / 1e9
        write_gb = n_el * 2 / 1e9
        traffic = read_gb + write_gb

        def launch(grid):
            with cp.cuda.Device(0):
                D._decode(grid=(grid,), block=(512,), shared_mem=smem,
                          args=[lut.data_ptr(), enc.data_ptr(), oth.data_ptr(),
                                opos.data_ptr(), gaps.data_ptr(), out.data_ptr(),
                                n_luts, n_bytes, n_el])

        ms1 = timed(lambda: launch(g1), args.iters, args.warmup)
        # NOTE: the kernel is NOT safe to over-launch. Its output_positions/gaps
        # arrays are sized for exactly ceil(n_bytes/(block*bytes_per_thread))
        # blocks, and a larger grid writes out of bounds (verified: CUDA_ERROR_
        # ILLEGAL_ADDRESS). So only the production grid is measurable here.

        # pure traffic floor on the same buffers: read sign_mantissa, write bf16
        mini = timed(lambda: out.copy_(oth), args.iters, args.warmup)
        floor_traffic = n_el * 3 / 1e9                 # read 1B + write 2B per elem

        print(f"\nlayer {li}: {n_el/1e6:.1f}M weights @8bit  n_bytes={n_bytes/1e6:.2f}M  "
              f"luts={n_luts}  smem={smem/1024:.1f} KB  grid={g1}", flush=True)
        print(f"  decode  grid x1   {ms1:8.3f} ms   {traffic/ms1*1000:7.1f} GB/s "
              f"(traffic {traffic:.3f} GB)", flush=True)
        print(f"  traffic floor     {mini:8.3f} ms   {floor_traffic/mini*1000:7.1f} GB/s "
              f"(read {n_el/1e9:.3f} + write {n_el*2/1e9:.3f} GB)", flush=True)
        print(f"  => entropy-decode cost above its own traffic scale: "
              f"{ms1 - mini * (traffic/floor_traffic):.3f} ms/layer", flush=True)

        del enc, oth, lut, opos, gaps, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
