#!/usr/bin/env python
"""
bench_bw.py -- calibrate the bandwidth model for this GPU.

The whole project rests on a traffic model:

  bf16 GEMM per token :  read 16 bits/weight
  DF11 unfused        :  read 11 bits/weight + WRITE 16 bits/weight + READ 16 bits/weight

so DF11 should move ~2.2x the bytes. This script measures what each elementary
traffic pattern actually costs on this machine, so we can ask the decisive
question: is DF11's `decode` kernel time explained by its TRAFFIC (in which case
fusing decode into the GEMV is the win), or is the Huffman bit-unpacking itself
the cost (in which case the entropy code must get cheaper or go)?

Sizes are taken from the real Qwen3-1.7B DF11 model (1.409e9 compressed weight
elements, 0.469 GB of exponent bitstream, 3.44 GB bf16 weight volume).

Run in two memory phases so it fits in 8 GB of VRAM.
"""
import time

import torch

DEV = "cuda:0"
N_W = 1_409_286_144           # compressed weight elements (sign_mantissa numel)
N_ENC = 468_613_348           # encoded_exponent bytes
N_BF16 = 1_720_450_000        # bf16 parameters in Qwen3-1.7B (3.44 GB)


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def rep(label, gb, fn, iters=20):
    t = timeit(fn, iters=iters)
    print(f"{label:46s} {t*1000:8.3f} ms  {gb:6.3f} GB  {gb/t:7.1f} GB/s", flush=True)
    return t


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"device {torch.cuda.get_device_name(0)}  L2 {p.L2_cache_size/1e6:.1f} MB  "
          f"{p.multi_processor_count} SMs", flush=True)

    # ---------------------------------------------------------------- phase 1
    print("\n--- phase 1: bf16 weight pass + copy ---", flush=True)
    srcbf = torch.randn(N_BF16, dtype=torch.bfloat16, device=DEV)   # 3.441 GB
    dstbf = torch.empty_like(srcbf)                                 # 3.441 GB
    rep("A  read 3.441 GB bf16 (bf16 weight pass)", 3.441,
        lambda: torch.sum(srcbf, dtype=torch.float32))
    rep("D  copy 3.441 GB bf16 (read+write 6.88 GB)", 6.882,
        lambda: dstbf.copy_(srcbf))
    rep("D2 pure write 3.441 GB bf16", 3.441, lambda: dstbf.zero_())
    del srcbf, dstbf
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- phase 2
    print("\n--- phase 2: DF11 stream patterns ---", flush=True)
    u8w = torch.randint(0, 256, (N_W,), dtype=torch.uint8, device=DEV)   # 1.409 GB
    u8e = torch.randint(0, 256, (N_ENC,), dtype=torch.uint8, device=DEV)  # 0.469 GB
    outbf = torch.empty(N_W, dtype=torch.bfloat16, device=DEV)            # 2.818 GB
    # NOTE: sum() on uint8 promotes to int64 and becomes ALU-bound (~0.5 GB/s),
    # which measures nothing. max() keeps the dtype and stays memory-bound.
    rep("B  read 1.409 GB u8 (sign_mantissa, max)", 1.409, lambda: u8w.max())
    rep("E  read 0.469 GB u8 (encoded_exponent, max)", 0.469, lambda: u8e.max())
    # copy_ with differing dtypes casts element-wise: no intermediate allocation
    rep("C  expand u8->bf16 (read 1.409, write 2.818)", 4.227,
        lambda: outbf.copy_(u8w))

    def full():
        outbf.copy_(u8w)
        u8e.max()

    # traffic is read 1.878 GB, write 2.818 GB  ->  4.696 GB total
    rep("F  decode traffic floor (read 1.878, write 2.818)", 4.696, full)
    def fused_read():
        u8w.max()
        u8e.max()

    rep("G  fused-input read 1.878 GB (both streams)", 1.878, fused_read)

    print(flush=True)
    print("measured (from bench_infer profiler, Qwen3-1.7B, per token):", flush=True)
    print("  bf16  gemv pass           28.72 ms   read 3.44 GB + activations", flush=True)
    print("  DF11  `decode` kernel     49.53 ms   read 1.92 GB, write 2.82 GB", flush=True)
    print("  DF11  total               66.68 ms", flush=True)
    print(flush=True)
    print("=> decode kernel excess over its own traffic floor =", flush=True)
    print("   49.53 - 25.85 = 23.68 ms/token of entropy-decode (non-traffic) work", flush=True)


if __name__ == "__main__":
    main()
