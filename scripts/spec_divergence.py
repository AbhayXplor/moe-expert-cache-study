#!/usr/bin/env python
"""
spec_divergence.py -- is speculative decoding *bit*-exact?

Theory says yes: the draft/verify rule accepts a draft token only when it
equals what the target model would have produced, so the output distribution
is unchanged.  Practice can differ, because verification runs in a batch and
a batched GEMM sums in a different order than a single-token GEMM.  Logits
then differ in the low bits, and a greedy decode sitting on a near-tie can
flip one token -- after which the whole continuation shifts.

This script measures exactly that: it runs the same prompt, greedily, twice
per configuration (to prove each path is itself deterministic) and reports
the first character where the configurations disagree, with context.

Usage:
  python spec_divergence.py --configs baseline ngram-n12 --prompt repetitive
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llamaserver_bench import (  # noqa: E402
    CONFIGS, PROMPTS, SERVER, WORKSPACE, completion, wait_health)


def run_config(name, prompt_name, port, num_ctx, ngl, n_predict, repeats):
    cmd = [SERVER, "--model", os.path.expanduser(
        r"~\.ollama\models\blobs"
        r"\sha256-02d45dc1cf451ba2475ac33b301c2dd8f985abe4c182ce04a1f2f5bf0260278d"),
        "--port", str(port), "--host", "127.0.0.1", "--no-webui", "--offline",
        "-c", str(num_ctx), "-ngl", str(ngl), "-np", "1",
        "-b", "512", "-ub", "512", "--flash-attn", "auto",
        "--no-jinja", "--chat-template", "chatml"] + CONFIGS[name]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    logpath = os.path.join(WORKSPACE, "experiments", "logs",
                           f"specdiv_{name}.log")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logf = open(logpath, "wb")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        if not wait_health(port, time.time() + 240):
            raise RuntimeError(f"{name}: server never became healthy")
        out = []
        for _ in range(repeats):
            out.append(completion(port, PROMPTS[prompt_name], n_predict))
    finally:
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:
            pass
        time.sleep(1)
        proc.kill()
        proc.wait(timeout=30)
        logf.close()
        time.sleep(4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["baseline", "ngram-n12"])
    ap.add_argument("--prompt", default="repetitive", choices=list(PROMPTS))
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--ngl", type=int, default=33)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()

    results = {}
    for name in args.configs:
        runs = run_config(name, args.prompt, args.port, args.num_ctx,
                          args.ngl, args.n_predict, args.repeats)
        results[name] = runs
        hashes = [hashlib.sha256(r["content"].encode()).hexdigest()[:12]
                  for r in runs]
        tps = [r.get("timings", {}).get("predicted_per_second") for r in runs]
        acc = [f'{r.get("timings",{}).get("draft_n_accepted",0)}/'
               f'{r.get("timings",{}).get("draft_n",0)}' for r in runs]
        same = len(set(hashes)) == 1
        print(f"{name:12s} hashes {hashes}  self_consistent={same}"
              f"  tok/s {[round(t,2) if t else None for t in tps]}"
              f"  drafts {acc}", flush=True)

    base_name = args.configs[0]
    base = results[base_name][-1]["content"]
    print(f"\nfirst difference vs {base_name} (prompt={args.prompt}):")
    for name in args.configs[1:]:
        text = results[name][-1]["content"]
        i = next((k for k in range(min(len(base), len(text)))
                  if base[k] != text[k]), min(len(base), len(text)))
        if i == len(base) == len(text):
            print(f"  {name:12s} IDENTICAL")
        else:
            print(f"  {name:12s} diverges at char {i} / {len(text)}")
            print(f"      base: ...{base[max(0,i-70):i]!r}[{base[i:i+70]!r}]")
            print(f"      spec: ...{text[max(0,i-70):i]!r}[{text[i:i+70]!r}]")
    out = os.path.join(WORKSPACE, "experiments", "runs", "spec_divergence.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({k: [r["content"] for r in v] for k, v in results.items()},
                  fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
