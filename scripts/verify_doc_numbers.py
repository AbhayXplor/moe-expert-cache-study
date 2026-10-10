"""Check the numbers quoted in the write-ups against the raw run JSONs.

Covers EXPERT_CACHE_AND_REAP.md (raw per-config figures) and PAPER_PACK.md
(derived figures: ratios versus baseline, expert-byte totals, coverage).
"""
import json, os, statistics as st, sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(WORKSPACE, 'experiments', 'runs')


def runs(tag):
    return json.load(open(os.path.join(RUNS, f'moe_bench_{tag}.json'),
                          encoding='utf-8'))

def decode(path, nth=0):
    e = runs_fn(path)[nth]
    v = [v['decode_tps'] for rep in e['prompts'] for k, v in rep.items()
         if 'decode_tps' in v and k != 'longprefill']
    return st.mean(v), e.get('vram_used_mib')

runs_fn = lambda t: runs(t)

# (doc claim, tag, entry index, expected mean decode, expected vram)
checks = [
    ("35B cpu-moe #1",   'cache_ab2', 0, 22.02, 3187),
    ("35B cache-2048 #1",'cache_ab2', 1,  8.03, 5237),
    ("35B cpu-moe #2",   'cache_ab2', 2, 22.16, 3187),
    ("35B cache-2048 #2",'cache_ab2', 3,  7.97, 5237),
    ("35B cpu-moe #3",   'cache_ab2', 4, 22.29, 3187),
    ("35B cache-4096",   'cache_ab2', 5, 11.91, 7283),
    ("REAP cache-3072",  'reap',     5, 30.44, 5593),
    ("REAP cache-4096 r2","reap2",   5, 34.11, 6615),
    ("REAP ncmoe-14 r2","reap2",    4, 31.49, 7303),
    ("REAP cache-4608 a",'reap3',    4, 41.40, 7129),
    ("REAP cache-4608 b",'reap4',    0, 36.10, 7129),
    ("REAP cache-4864 a",'reap4',    1, 37.62, 7383),
    ("REAP cache-5120x", 'reap4',    4, 49.14, 7639),
    ("REAP cache-5120 #1",'reap5',   0, 39.28, 7639),
    ("REAP cache-5120 #2",'reap5',   2, 39.40, 7639),
    ("REAP cache-5120 #3",'reap5',   4, 40.26, 7639),
    ("REAP cache-4864 b",'reap5',    1, 37.80, 7383),
    ("REAP qual 4608",   'qualreap', 0, 45.57, 7129),
    ("35B probe cpu-moe a",'probe',  0, 26.30, 3187),
    ("35B probe fa-off", 'probe',    1, 28.87, 4033),
    ("35B probe cpu-moe b",'probe',  2, 26.38, 3187),
    ("35B probe perf",   'probe',    3, 25.21, 3187),
    ("35B oldbin a",     'oldbin',   0, 28.57, 3203),
    ("35B oldbin b",     'oldbin',   1, 26.38, 3203),
    ("35B oldbin fa-off",'oldbin',   2, 23.99, 4049),
]
bad = 0
for label, tag, idx, exp_d, exp_v in checks:
    got_d, got_v = decode(tag, idx)
    ok_d = abs(got_d - exp_d) < 0.02
    ok_v = got_v == exp_v
    if not (ok_d and ok_v):
        bad += 1
        print(f"MISMATCH {label}: doc={exp_d}/{exp_v} raw={got_d:.2f}/{got_v}")
print(f"{len(checks)-bad}/{len(checks)} doc numbers match the raw artifacts")

# --- derived figures quoted in PAPER_PACK.md -------------------------------
# Means over independent server loads, not over prompts: that is what the docs
# quote, and mixing the two is exactly the kind of error this script exists for.
def mean_of(tag, idx):
    e = runs(tag)[idx]
    v = [v['decode_tps'] for rep in e['prompts'] for k, v in rep.items()
         if 'decode_tps' in v and k != 'longprefill']
    return st.mean(v)

base35 = st.mean([mean_of('cache_ab2', i) for i in (0, 2, 4)])
base_reap = st.mean([mean_of('reap', i) for i in (0, 2)])
ratios = [
    ("PAPER_PACK 35B/pool-2048", st.mean([mean_of('cache_ab2', 1),
                                          mean_of('cache_ab2', 3)]) / base35, 0.361),
    ("PAPER_PACK 35B/pool-4096", mean_of('cache_ab2', 5) / base35, 0.538),
    ("PAPER_PACK REAP/pool-3072", st.mean([mean_of('reap2', 1),
                                           mean_of('reap2', 3)]) / base_reap, 2.044),
    ("PAPER_PACK REAP/pool-4096", st.mean([mean_of('reap2', 5),
                                           mean_of('reap3', 0),
                                           mean_of('reap3', 2)]) / base_reap, 2.669),
    ("PAPER_PACK REAP/pool-5120", st.mean([mean_of('reap5', 0),
                                           mean_of('reap5', 2),
                                           mean_of('reap5', 4)]) / base_reap, 3.157),
]
bad_ratio = 0
for label, got, exp in ratios:
    if abs(got - exp) > 0.002:
        bad_ratio += 1
        print(f"MISMATCH {label}: doc={exp:.3f} raw={got:.3f}")
print(f"{len(ratios)-bad_ratio}/{len(ratios)} pack ratios match the raw artifacts "
      f"(35B baseline {base35:.2f} tok/s, REAP baseline {base_reap:.2f} tok/s)")

# Coverage percentages: expert bytes must come from the GGUF index, not a
# constant copied out of the doc.
EXPERT_BYTES = {
    "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf":
        (19_568_525_312, [(2048, 11.0), (4096, 21.9)]),
    "models/Qwen3.6-35B-A3B-REAP-48-v2-GGUF/Qwen3.6-35B-A3B-REAP-48-v2.gguf":
        (7_551_574_016, [(3072, 42.7), (4096, 56.9), (5120, 71.1)]),
}
bad_cov = 0
cov_checks = 0
for rel, (exp_bytes, pools) in EXPERT_BYTES.items():
    path = os.path.join(WORKSPACE, rel)
    if not os.path.exists(path):
        print(f"SKIP coverage for {os.path.basename(rel)} (model not on disk)")
        continue
    from gguf import GGUFReader
    real = sum(t.n_bytes for t in GGUFReader(path).tensors
               if "exps" in t.name)
    cov_checks += 1
    if real != exp_bytes:
        bad_cov += 1
        print(f"MISMATCH expert bytes {os.path.basename(rel)}: "
              f"doc={exp_bytes} raw={real}")
    for pool, exp_pct in pools:
        cov_checks += 1
        got_pct = 100 * (pool * 1024 ** 2) / real
        if abs(got_pct - exp_pct) > 0.15:
            bad_cov += 1
            print(f"MISMATCH coverage pool {pool}: doc={exp_pct}% "
                  f"raw={got_pct:.2f}%")
print(f"{cov_checks-bad_cov}/{cov_checks} coverage/expert-byte figures match")

sys.exit(1 if (bad or bad_ratio or bad_cov) else 0)
