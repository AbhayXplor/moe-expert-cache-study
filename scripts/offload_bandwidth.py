#!/usr/bin/env python
"""
offload_bandwidth.py -- what does RAM offload actually cost on THIS machine?

The offload question reduces to three prices per byte:

  1. read a weight out of VRAM                    (fast)
  2. read a weight out of host RAM, on the CPU    (medium)
  3. copy a weight, RAM -> VRAM over PCIe         (slow)

If (3) is far worse than (2), then *streaming* weights to the GPU is the wrong
move and *computing them where they sit* is the right one -- which is the
opposite of what the word "offload" suggests.

It also tests whether a PCIe copy can run at the same time as GPU compute,
which is the premise of every prefetching offloader.

MEASUREMENT HAZARDS (both found the hard way, both produced wrong numbers first):

  * This card parks its PCIe link at Gen1 x4 and its memory clock at 405 MHz
    when idle. Timing a copy straight after a quiet period measures the *ramp*.
    The first version of this script reported 3.8 GB/s for the link. Every
    number here is taken after a time-based ramp with the link held awake, and
    the link state is sampled under sustained load.
  * Pinned allocations above ~256 MiB hang on this box. The harness caps them.

The minimum over repeats is reported: it is the least-contended sample.

Run:  ../.venv/Scripts/python.exe offload_bandwidth.py
"""
import json
import os
import subprocess
import sys
import time

import torch

MIB = 1 << 20
REPEAT = 5
RAMP_SECONDS = 2.0
MAX_PINNED_MIB = 256


def say(*a):
    print(*a, flush=True)


def gbps(nbytes, seconds):
    return nbytes / seconds / 1e9


def link_state():
    """PCIe link generation/width and clocks, straight from the driver."""
    fields = ("pcie.link.gen.current,pcie.link.width.current,"
              "clocks.current.sm,clocks.current.memory,power.draw")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - diagnostics must never kill the run
        return f"<link_state failed: {exc}>"
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 5:
        return out
    return (f"pcie=gen{parts[0]} x{parts[1]}  sm={parts[2]}  "
            f"mem={parts[3]}  power={parts[4]}")


def ramp(fn, seconds=RAMP_SECONDS):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        fn()
    torch.cuda.synchronize()


def timed(fn, repeat=REPEAT, ramp_seconds=RAMP_SECONDS):
    """Ramped, synchronised, minimum-of-N seconds per call."""
    if ramp_seconds:
        ramp(fn, ramp_seconds)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def cpu_read_rate(nbytes):
    """Multi-threaded streaming read of host RAM (sum reduction)."""
    f = torch.empty(nbytes // 4, dtype=torch.float32)
    out = torch.empty((), dtype=torch.float32)
    torch.sum(f, dim=0, out=out)  # warm
    best = float("inf")
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        torch.sum(f, dim=0, out=out)
        best = min(best, time.perf_counter() - t0)
    del f, out
    return best


def cpu_add_rate(nbytes):
    """c = a + b. Traffic is 3 halves, so it is a cleaner bandwidth probe."""
    a = torch.empty(nbytes // 4, dtype=torch.float32).fill_(1.0)
    b = torch.empty(nbytes // 4, dtype=torch.float32).fill_(2.0)
    c = torch.empty(nbytes // 4, dtype=torch.float32)
    torch.add(a, b, out=c)
    best = float("inf")
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        torch.add(a, b, out=c)
        best = min(best, time.perf_counter() - t0)
    del a, b, c
    return best


def main():
    if not torch.cuda.is_available():
        say("no CUDA device")
        return 1
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    say(f"device: {props.name}  ({props.total_memory / MIB:.0f} MiB)")
    say(f"link at rest: {link_state()}")
    say(f"torch cpu threads: {torch.get_num_threads()}  "
        f"cpus: {os.cpu_count()}")
    say(f"ramp: {RAMP_SECONDS}s before every timed measurement")

    results = {}

    # ------------------------------------------------------------ VRAM read
    say("\n[1] VRAM read (float32 sum over 512 MiB) ...")
    n = 512 * MIB
    f = torch.empty(n // 4, dtype=torch.float32, device=dev)
    out = torch.empty((), dtype=torch.float32, device=dev)
    t = timed(lambda: torch.sum(f, dim=0, out=out))
    results["vram_read"] = {"bytes": n, "sec": t, "gbps": gbps(n, t)}
    say(f"    {gbps(n, t):.1f} GB/s   link: {link_state()}")
    del f, out
    torch.cuda.empty_cache()

    # ------------------------------------------------------------- CPU read
    say("[2] CPU read (float32 sum over 512 MiB) ...")
    t = cpu_read_rate(512 * MIB)
    results["cpu_read"] = {"bytes": 512 * MIB, "sec": t,
                           "gbps": gbps(512 * MIB, t)}
    say(f"    {gbps(512 * MIB, t):.1f} GB/s")

    say("[3] CPU add  c = a + b  (512 MiB each; traffic = 1.5 GiB) ...")
    t = cpu_add_rate(512 * MIB)
    traffic = int(512 * MIB * 1.5)
    results["cpu_add_traffic"] = {"bytes": traffic, "sec": t,
                                  "gbps": gbps(traffic, t)}
    say(f"    {gbps(traffic, t):.1f} GB/s of actual traffic")

    # --------------------------------------------------------------- PCIe
    for mib in (64, 128, 256):
        label = f"pcie_h2d_{mib}MiB"
        say(f"[4] {label} (pinned, non_blocking) ...")
        pinned = torch.empty(mib * MIB, dtype=torch.uint8, pin_memory=True)
        dst = torch.empty(mib * MIB, dtype=torch.uint8, device=dev)
        t = timed(lambda: dst.copy_(pinned, non_blocking=True))
        results[label] = {"bytes": mib * MIB, "sec": t,
                          "gbps": gbps(mib * MIB, t)}
        say(f"    {gbps(mib * MIB, t):.2f} GB/s   link: {link_state()}")
        del pinned, dst
        torch.cuda.empty_cache()

    # ------------------------------------------- sustained link: does it ramp?
    say("[5] sustained-link probe (20 x 64 MiB pinned copies per sample) ...")
    pinned = torch.empty(64 * MIB, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(64 * MIB, dtype=torch.uint8, device=dev)
    samples = []
    for _ in range(4):
        for _ in range(20):
            dst.copy_(pinned, non_blocking=True)
        torch.cuda.synchronize()
        samples.append(link_state())
    results["sustained_link_samples"] = samples
    for s in samples:
        say(f"    {s}")

    # ------------------------------------------------------------- overlap
    say("[6] overlap test (PCIe copy on one stream, GPU kernel on another) ...")
    copy_bytes = 64 * MIB
    kern_bytes = 1 << 30
    kf = torch.empty(kern_bytes // 4, dtype=torch.float32, device=dev)
    kout = torch.empty((), dtype=torch.float32, device=dev)

    t_copy = timed(lambda: dst.copy_(pinned, non_blocking=True), ramp_seconds=1.0)
    t_kern = timed(lambda: torch.sum(kf, dim=0, out=kout), ramp_seconds=1.0)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    def both():
        with torch.cuda.stream(s1):
            dst.copy_(pinned, non_blocking=True)
        with torch.cuda.stream(s2):
            torch.sum(kf, dim=0, out=kout)

    t_both = timed(both, ramp_seconds=1.0)
    results["overlap"] = {
        "copy_bytes": copy_bytes, "kernel_bytes": kern_bytes,
        "copy_only_ms": t_copy * 1e3, "kernel_only_ms": t_kern * 1e3,
        "both_ms": t_both * 1e3,
        "sum_if_serial_ms": (t_copy + t_kern) * 1e3,
        "max_if_hidden_ms": max(t_copy, t_kern) * 1e3,
        "copy_hidden_fraction": ((t_copy + t_kern) - t_both) / t_copy,
    }
    del kf, kout, pinned, dst
    torch.cuda.empty_cache()

    # -------------------------------------------------------------- report
    ref = results["vram_read"]["gbps"]
    say("")
    say(f"{'path':30s} {'GB/s':>8s} {'ms/MB':>9s} {'vs VRAM':>9s}")
    say("-" * 60)
    for key in ("vram_read", "cpu_read", "pcie_h2d_256MiB",
                "pcie_h2d_128MiB", "pcie_h2d_64MiB"):
        r = results[key]
        ms_per_mb = r["sec"] / (r["bytes"] / MIB) * 1e3
        say(f"{key:30s} {r['gbps']:8.2f} {ms_per_mb:9.4f} "
            f"{ref / r['gbps']:8.1f}x")

    ov = results["overlap"]
    say("")
    say("prefetch test (can a PCIe copy hide behind GPU compute?)")
    say(f"  copy alone      {ov['copy_only_ms']:7.2f} ms "
        f"({ov['copy_bytes'] / MIB:.0f} MiB)")
    say(f"  kernel alone    {ov['kernel_only_ms']:7.2f} ms "
        f"({ov['kernel_bytes'] / MIB:.0f} MiB read)")
    say(f"  both together   {ov['both_ms']:7.2f} ms")
    say(f"  serial would be {ov['sum_if_serial_ms']:7.2f} ms; "
        f"hidden would be {ov['max_if_hidden_ms']:7.2f} ms")
    say(f"  copy hidden fraction: {ov['copy_hidden_fraction'] * 100:.0f}%")

    out = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "experiments", "runs", "offload_bandwidth.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "device": props.name,
            "total_mib": props.total_memory / MIB,
            "torch_cpu_threads": torch.get_num_threads(),
            "ramp_seconds": RAMP_SECONDS,
            "link_at_rest": link_state(),
            "results": results,
            "per_byte_multiple_vs_vram": {
                k: ref / results[k]["gbps"]
                for k in ("cpu_read", "pcie_h2d_256MiB",
                          "pcie_h2d_128MiB", "pcie_h2d_64MiB")
            },
        }, fh, indent=2)
    say(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
