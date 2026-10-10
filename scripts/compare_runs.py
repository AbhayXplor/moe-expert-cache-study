#!/usr/bin/env python
"""
compare_runs.py -- compare two bench_infer.py JSON reports for token-identity
and logit agreement (the lossless claim check).

Usage:
  python compare_runs.py --a experiments/runs/e0_bf16_1.7b.json \
                         --b experiments/runs/e2a_df11_1.7b.json
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    A = json.load(open(args.a, encoding="utf-8"))
    B = json.load(open(args.b, encoding="utf-8"))
    ta, tb = A["bench"]["tokens"], B["bench"]["tokens"]
    n = min(len(ta), len(tb))

    first_div = next((i for i in range(n) if ta[i] != tb[i]), -1)
    matched = sum(1 for i in range(n) if ta[i] == tb[i])

    la, lb = A["bench"]["logits"], B["bench"]["logits"]
    max_logit_diff = 0.0
    for i in range(n):
        max_logit_diff = max(max_logit_diff, abs(la[i][0] - lb[i][0]))

    out = {
        "a": args.a, "b": args.b,
        "tokens_compared": n,
        "tokens_identical": matched,
        "identity_match": first_div == -1 and len(ta) == len(tb),
        "first_divergence_index": first_div,
        "max_abs_top1_logit_diff": max_logit_diff,
        "vram": {
            "a_resident_gib": A.get("resident_total_gib"),
            "b_resident_gib": B.get("resident_total_gib"),
            "a_peak_alloc_gib": A.get("torch_peak_allocated_gib"),
            "b_peak_alloc_gib": B.get("torch_peak_allocated_gib"),
            "a_peak_smi_mib": (A.get("vram_sampler") or {}).get("max_used_mib"),
            "b_peak_smi_mib": (B.get("vram_sampler") or {}).get("max_used_mib"),
        },
        "speed": {
            "a_decode_tps": A["bench"]["decode_tps"],
            "b_decode_tps": B["bench"]["decode_tps"],
            "speed_ratio_b_over_a": round(B["bench"]["decode_tps"] / A["bench"]["decode_tps"], 3)
            if A["bench"]["decode_tps"] else None,
            "a_ttft_s": A["bench"]["ttft_s"],
            "b_ttft_s": B["bench"]["ttft_s"],
        },
        "resident_classes_a": A.get("resident_bytes_by_class"),
        "resident_classes_b": B.get("resident_bytes_by_class"),
        "cpu_offload_bytes_a": A.get("bytes_offloaded_to_cpu"),
        "cpu_offload_bytes_b": B.get("bytes_offloaded_to_cpu"),
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
