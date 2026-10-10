"""Derive the paper-pack numbers from the artifacts, instead of typing them by hand.

Three parts:
  1. model anatomy read from the GGUF tensor index.  Note the GGUF layout: all
     experts of one layer live in ONE stacked tensor, so the file has only
     3 x n_layers expert tensors, not one per expert.  The real expert count
     comes from metadata (qwen35moe.expert_count), so per-expert sizes and the
     bytes-read-per-token figure need both numbers.
  2. the coverage table (pool bytes / expert bytes) against measured decode.
  3. the mechanism arithmetic: if a miss costs one synchronous upload, how much
     time does that predict, and what miss fraction would the measured speed
     imply?

Usage:  .venv/Scripts/python.exe scripts/paper_data.py
"""
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader  # noqa: E402

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(WORKSPACE, "experiments", "runs")

MODELS = [
    ("Qwen3.6-35B-A3B (full)",
     "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    ("Qwen3.6-35B-A3B-REAP-48",
     "models/Qwen3.6-35B-A3B-REAP-48-v2-GGUF/Qwen3.6-35B-A3B-REAP-48-v2.gguf"),
]

# config name in the run JSONs -> pool size in MiB
POOLS = {"cache-2048": 2048, "cache-3072": 3072, "cache-4096": 4096,
         "cache-4608": 4608, "cache-4864": 4864, "cache-5120": 5120}
POOL_TAGS = ("cache_ab2", "reap", "reap2", "reap3", "reap4", "reap5")

PCIE_BYTES_PER_S = 2.80e9     # measured on this machine (Gen3 x4 negotiated)


def _meta(r, key, default=None):
    for field in r.fields.values():
        if field.name == key:
            return field.contents()
    return default


def anatomy():
    print("=" * 78)
    print("1. MODEL ANATOMY (from the GGUF tensor index + metadata)")
    print("=" * 78)
    out = {}
    for label, rel in MODELS:
        path = os.path.join(WORKSPACE, rel)
        if not os.path.exists(path):
            print(f"\n{label}: MISSING {path}")
            continue
        r = GGUFReader(path)
        total = sum(t.n_bytes for t in r.tensors)
        exps = [t for t in r.tensors if "exps" in t.name]
        expert = sum(t.n_bytes for t in exps)
        layers = sorted({int(p) for t in exps for p in t.name.split(".")
                         if p.isdigit()})
        n_layers = len(layers)
        n_expert = int(_meta(r, "qwen35moe.expert_count", 0))
        n_used = int(_meta(r, "qwen35moe.expert_used_count", 0))
        per_expert = expert / (n_layers * n_expert)
        per_layer = expert / n_layers
        per_token = expert * n_used / n_expert
        out[label] = dict(total=total, expert=expert, n_expert=n_expert,
                          n_used=n_used, n_layers=n_layers,
                          per_expert=per_expert, per_layer=per_layer,
                          per_token=per_token,
                          n_exps_tensors=len(exps))
        print(f"\n{label}")
        print(f"  file                 {total/1024**3:8.3f} GiB ({total:,} B)")
        print(f"  expert weights       {expert/1024**3:8.3f} GiB "
              f"({expert:,} B) = {100*expert/total:.1f}% of file, "
              f"{len(exps)} stacked tensors")
        print(f"  experts per layer    {n_expert} (top-{n_used} used), "
              f"{n_layers} layers")
        print(f"  per expert x layer   {per_expert/1024**2:8.3f} MiB")
        print(f"  per layer, all       {per_layer/1024**2:8.3f} MiB")
        print(f"  expert bytes/token   {per_token/1e6:8.1f} MB"
              f"  (top-{n_used} of {n_expert})")
        if n_layers * n_expert:
            print(f"  expert 'slots'       {n_layers*n_expert}"
                  f"  (layer x expert combinations)")
    return out


def coverage(anat):
    print()
    print("=" * 78)
    print("2. COVERAGE = pool bytes / expert bytes, vs measured decode")
    print("=" * 78)
    for label, a in anat.items():
        is_reap = "REAP" in label
        rows = []
        for tag in POOL_TAGS:
            p = os.path.join(RUNS, f"moe_bench_{tag}.json")
            if not os.path.exists(p):
                continue
            for e in json.load(open(p, encoding="utf-8")):
                cfg = e.get("config", "")
                if cfg not in POOLS:
                    continue
                # keep the two models apart: the full model never ran with a
                # pool, REAP only ran with one (so match on the model field)
                if ("REAP" in (e.get("model") or "")) != is_reap:
                    continue
                v = [x["decode_tps"] for rep in e.get("prompts", [])
                     for k, x in rep.items()
                     if "decode_tps" in x and k != "longprefill"]
                if v:
                    rows.append((POOLS[cfg], st.mean(v), tag))
        if not rows:
            continue
        print(f"\n{label}   expert weights {a['expert']/1e9:.3f} GB")
        print(f"  {'pool MiB':>8} {'pool GB':>8} {'coverage':>9} "
              f"{'decode':>7}  source")
        for pool, mean, tag in sorted(rows):
            gb = pool * 1024 ** 2 / 1e9
            print(f"  {pool:>8} {gb:>8.2f} {100*gb/(a['expert']/1e9):>8.1f}% "
                  f"{mean:>7.2f}  {tag}")
        # also show the no-pool baseline for the same model, if recorded
        for tag in ("cache_ab2", "reap", "reap2", "reap3"):
            p = os.path.join(RUNS, f"moe_bench_{tag}.json")
            if not os.path.exists(p):
                continue
            for e in json.load(open(p, encoding="utf-8")):
                if e.get("config") != "cpu-moe":
                    continue
                if ("REAP" in (e.get("model") or "")) != is_reap:
                    continue
                v = [x["decode_tps"] for rep in e.get("prompts", [])
                     for k, x in rep.items()
                     if "decode_tps" in x and k != "longprefill"]
                if v:
                    print(f"  {'none':>8} {0:>8.2f} {0:>8.1f}% "
                          f"{st.mean(v):>7.2f}  {tag} (baseline)")


def pcie_check(anat):
    print()
    print("=" * 78)
    print("3. MECHANISM CROSS-CHECK -- hypothesis, arithmetic only")
    print("=" * 78)
    print(f"Assume a cache miss costs one synchronous host->device upload of")
    print(f"that expert's weights over the measured {PCIE_BYTES_PER_S/1e9:.2f} GB/s PCIe.")
    print("time/token = (expert bytes/token) * f / link, f = missing fraction.")
    for label, a in anat.items():
        per_token = a["per_token"]
        print(f"\n{label}   expert bytes/token = {per_token/1e6:.1f} MB")
        print(f"  {'f':>5} {'ms/token':>9} {'predicted':>10}   {'implied by':>18}")
        for f in (1.0, 0.55, 0.37, 0.10):
            ms = per_token * f / PCIE_BYTES_PER_S * 1e3
            print(f"  {f:>5.2f} {ms:>9.1f} {1/(ms/1e3):>9.2f} tok/s")
        for measured, note in ((8.0, "pool 2048, full model"),
                               (11.9, "pool 4096, full model")):
            if "REAP" in label:
                continue
            ms = 1e3 / measured
            implied = ms / 1e3 * PCIE_BYTES_PER_S / per_token
            print(f"  measured {measured:5.1f} tok/s ({note}) implies "
                  f"f = {implied:.2f}")


def measured():
    print()
    print("=" * 78)
    print("4. MEASURED CONFIGURATIONS (as recorded in runs/)")
    print("=" * 78)
    for tag in ("cache_ab", "cache_ab2", "reap", "reap2", "reap3", "reap4",
                "reap5", "probe", "oldbin", "qualreap", "qualfull"):
        p = os.path.join(RUNS, f"moe_bench_{tag}.json")
        if not os.path.exists(p):
            continue
        print(f"\n[{tag}]")
        for e in json.load(open(p, encoding="utf-8")):
            v = [x["decode_tps"] for rep in e.get("prompts", [])
                 for k, x in rep.items()
                 if "decode_tps" in x and k != "longprefill"]
            pf = [x["prefill_tps"] for rep in e.get("prompts", [])
                  for k, x in rep.items() if k == "longprefill"
                  and "prefill_tps" in x]
            srv = os.path.basename(os.path.dirname(e.get("server") or ""))
            mdl = os.path.basename(e.get("model") or "")
            if not v:
                print(f"  {e.get('config'):18s} no timing (err={e.get('error')})")
                continue
            print(f"  {e.get('config'):18s} {st.mean(v):6.2f} tok/s "
                  f"[{min(v):5.1f}-{max(v):5.1f}]  prefill "
                  f"{(st.mean(pf) if pf else 0):7.0f}  vram {e.get('vram_used_mib')}"
                  f"  ram {e.get('ram_peak_gib')}  {srv}  {mdl}")


if __name__ == "__main__":
    a = anatomy()
    if a:
        coverage(a)
        pcie_check(a)
    measured()
