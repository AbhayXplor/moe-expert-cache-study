#!/usr/bin/env python
"""
ollama_bench.py -- measure an Ollama model the way it actually runs.

Reports, for one configuration:
  * real decode tok/s and prefill tok/s, straight from Ollama's own counters
    (eval_count / eval_duration -- these are measured server-side, so they are
    not polluted by our HTTP overhead)
  * the GPU/CPU offload split from /api/ps (size vs size_vram), which is the
    number that actually predicts speed
  * VRAM used, sampled while generation is in flight

This exists because "30-40 tok/s feels fine" is not a measurement, and because
a partially-offloaded model reports itself as working normally.

Usage:
  python ollama_bench.py --model qwen3.5:9b --num-ctx 32768
  python ollama_bench.py --model qwen3.5:9b --num-ctx 8192 --num-predict 256
"""
import argparse
import json
import subprocess
import threading
import time
import urllib.request

import os
HOST = os.environ.get("OLLAMA_BENCH_HOST", "http://localhost:11434")


def api(path, payload=None, timeout=1800):
    url = HOST + path
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class VramSampler(threading.Thread):
    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval, self.samples, self._stop = interval, [], threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                if out:
                    used, util = (x.strip() for x in out.splitlines()[0].split(",")[:2])
                    self.samples.append((int(used), int(util)))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


PROMPT = (
    "Explain in detail how the KV cache grows with context length, why that "
    "limits the largest model that fits in VRAM, and what techniques reduce it. "
    "Be thorough and technical."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--num-gpu", type=int, default=None,
                    help="force this many layers onto the GPU (Ollama default is automatic)")
    ap.add_argument("--fill", type=int, default=0,
                    help="pad the prompt with filler until the context holds about "
                         "this many tokens (measures decode at a FILLED context)")
    args = ap.parse_args()

    label = args.label or f"ctx={args.num_ctx}"
    print(f"\n{'='*74}\n{args.model}  {label}", flush=True)

    s = VramSampler()
    s.start()
    idle = s.samples[-1][0] if s.samples else None

    opts = {"num_ctx": args.num_ctx, "num_predict": args.num_predict,
            "temperature": 0.2}
    if args.num_gpu is not None:
        opts["num_gpu"] = args.num_gpu

    prompt = PROMPT
    if args.fill:
        # ~4 chars/token; keep the real question at the end so the answer is
        # still generated after a genuinely long context.
        filler = "The quick brown fox jumps over the lazy dog. " * 20
        reps = max(1, int(args.fill * 4 / len(filler)))
        prompt = "Read the following filler text.\n" + (filler * reps) + "\n\n" + PROMPT

    t0 = time.perf_counter()
    r = api("/api/generate", {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
        "options": opts,
    })
    wall = time.perf_counter() - t0
    s.stop()
    s.join(timeout=3)

    ec, ed = r.get("eval_count", 0), r.get("eval_duration", 1)
    pc, pd = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 1)
    print(f"  filled      {pc:5d} tok of real context")
    print(f"  decode      {ec:5d} tok in {ed/1e9:7.3f} s  =  {ec/(ed/1e9):6.2f} tok/s")
    print(f"  prefill     {pc:5d} tok in {pd/1e9:7.3f} s  =  {pc/(pd/1e9):7.1f} tok/s")
    if pc > 1 and ed:
        kv_kib = 32.0  # measured: f16 K+V = 32 KiB/token for this model
        ms = ed / 1e9 * 1000 / max(ec, 1)
        kv_ms = pc * kv_kib * 1024 / 240e9 * 1000
        print(f"  per-token   {ms:6.2f} ms total; ~{ms-kv_ms:6.2f} ms weights + "
              f"~{kv_ms:5.2f} ms KV read at this context")
    print(f"  load        {r.get('load_duration',0)/1e9:7.3f} s")
    print(f"  wall        {wall:7.3f} s")

    # what is actually resident, and where
    ps = api("/api/ps")
    for m in ps.get("models", []):
        if m["name"] != args.model:
            continue
        size, vram = m.get("size", 0), m.get("size_vram", 0)
        pct = 100.0 * vram / size if size else 0
        print(f"  ctx loaded  {m.get('context_length')}")
        print(f"  size        {size/1e9:7.3f} GB   vram {vram/1e9:7.3f} GB "
              f"->  {pct:.1f}% on GPU, {100-pct:.1f}% on CPU")
        eff = 5.618e9 if "9b" in args.model else None
        if eff and ec and ed:
            print(f"  implied weight bandwidth: "
                  f"{eff*ec/(ed/1e9)/1e9:6.1f} GB/s  "
                  f"(a fully-resident dense read would be ~240 GB/s here)")

    if s.samples:
        used = [x[0] for x in s.samples]
        util = [x[1] for x in s.samples]
        print(f"  VRAM        idle {idle} MiB -> peak {max(used)} MiB "
              f"of 8188 MiB   (GPU util mean {sum(util)/len(util):.0f}%)")


if __name__ == "__main__":
    main()
