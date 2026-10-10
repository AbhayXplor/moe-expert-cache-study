"""Consolidate every expert-cache / placement run into one table.

Reads the run JSONs written by moe_bench_35b.py and prints the decode and
prefill numbers per configuration, so the numbers quoted in the markdown
deliverables can be regenerated from the raw artifacts instead of trusted.

Usage:  .venv/Scripts/python.exe scripts/moe_cache_summary.py
"""
import json
import os
import statistics as st

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(WORKSPACE, "experiments", "runs")

# (tag, human label).  Order matters: it is the order they were measured in.
RUN_SETS = [
    ("cache_ab", "full 35B, ollama-derived binary (old, no cache flag)"),
    ("cache_ab2", "full 35B, official b11537 (cpu-moe vs cache-2048/4096)"),
    ("reap", "REAP-48, placement sweep 1"),
    ("reap2", "REAP-48, placement sweep 2"),
    ("reap3", "REAP-48, pool sweep"),
    ("reap4", "REAP-48, pool sweep (upper)"),
    ("reap5", "REAP-48, pool sweep confirmation (interleaved)"),
    ("probe", "full 35B, official b11537: -fa on vs off, and --perf"),
    ("oldbin", "full 35B, staged Ollama-derived build, same session as probe"),
    ("qualreap", "REAP-48 quality: pool vs no pool, 400 tokens"),
    ("qualfull", "full 35B quality, 400 tokens, same conditions"),
]

DECODE_PROMPTS = ("prose", "code", "security")


def summarise(entry):
    per = {}
    for rep in entry.get("prompts", []):
        for pname, v in rep.items():
            if "decode_tps" in v:
                per.setdefault(pname, []).append(v["decode_tps"])
    decode = [v for p in DECODE_PROMPTS for v in per.get(p, [])]
    prefill = [v["prefill_tps"] for rep in entry.get("prompts", [])
               for p in ("longprefill",)
               for v in (rep.get(p),) if v and "prefill_tps" in v]
    return decode, prefill


def main():
    for tag, label in RUN_SETS:
        path = os.path.join(RUNS, f"moe_bench_{tag}.json")
        if not os.path.exists(path):
            print(f"\n### {tag} ({label}) -- MISSING {path}")
            continue
        with open(path, encoding="utf-8") as fh:
            entries = json.load(fh)
        print(f"\n### {tag}  --  {label}")
        print(f"{'config':20s} {'VRAM':>5} {'free':>5} {'decode3':>8} "
              f"{'min':>6} {'max':>6} {'prefill':>8} {'RAM':>6}")
        for e in entries:
            decode, prefill = summarise(e)
            if not decode:
                print(f"{e['config']:20s}  no decode tokens "
                      f"(error={e.get('error')} killed={e.get('killed')})")
                continue
            vram = e.get("vram_used_mib") or 0
            print(f"{e['config']:20s} {vram:>5} {8188 - vram:>5} "
                  f"{st.mean(decode):>8.2f} {min(decode):>6.1f} "
                  f"{max(decode):>6.1f} "
                  f"{(st.mean(prefill) if prefill else 0):>8.1f} "
                  f"{e.get('ram_peak_gib', 0):>6.2f}")


if __name__ == "__main__":
    main()
