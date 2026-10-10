#!/usr/bin/env python
"""
moe_routing_probe.py -- is MoE expert routing actually lopsided enough to exploit?

The whole "keep hot experts in VRAM" idea rests on one factual claim: that a
small fraction of experts serves a large fraction of tokens. If routing were
uniform, every expert would be equally likely and caching would be worthless.
If routing is heavily skewed, a few megabytes of VRAM can replace gigabytes of
RAM traffic.

This script answers that question directly and cheaply:

  1. Run a realistic workload through a real MoE (OLMoE-1B-7B: 64 experts x 16
     layers, top-8) and record WHICH experts fire, per layer, per token.
  2. Measure the concentration: what fraction of picks land on the top X% of
     experts.
  3. Simulate a hot-expert VRAM cache two ways, on a held-out trace:
       * LRU  -- adaptive, learns which experts are hot online
       * static top-K -- chosen once from a warm-up trace, never evicted
     and report the hit rate plus the share of expert reads avoided.
  4. Report how big the cache actually has to be, in megabytes, at int4.

Point (4) is the number that matters: a hit rate is only interesting if the
cache that achieves it fits in the VRAM you have.

Nothing here needs a compiler, a GPU kernel, or offload machinery. It is pure
measurement. If the skew is not there, the whole direction dies here, cheaply.

Run:  ../.venv/Scripts/python.exe moe_routing_probe.py
      ../.venv/Scripts/python.exe moe_routing_probe.py --limit 6 --tokens 384
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"

# Prefer the complete local copy.  The hub cache only ever held one of the
# three shards (the local_dir download put the real files under models/), so
# resolving the repo id offline cannot work anyway, and the duplicate cache
# copy was deleted to reclaim 4.7 GB.  Change the default so the probe keeps
# loading without needing a re-download.
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_OLMOE = os.path.join(WORKSPACE, "models", "OLMoE-1B-7B-0924-Instruct")
if os.path.isdir(_LOCAL_OLMOE):
    MODEL_ID = _LOCAL_OLMOE

# A deliberately mixed workload: the point is to see whether skew survives
# across very different text, or whether it is an artifact of one prompt.
WORKLOAD = [
    ("code",
     "Write a Python function that parses a pcap file and extracts every unique "
     "destination IP address, then resolves them asynchronously without blocking "
     "the event loop. Include error handling for truncated captures."),
    ("security",
     "Explain how a Kerberoasting attack works against an Active Directory "
     "environment, what artefacts it leaves in Windows event logs, and how a "
     "defender would detect it in a SIEM using only native Windows telemetry."),
    ("reasoning",
     "A train leaves station A at 60 km/h. Another leaves station B, 300 km away, "
     "at 90 km/h heading toward A. A bird flies between them at 120 km/h until "
     "they meet. How far does the bird travel? Explain your reasoning step by step."),
    ("prose",
     "The lighthouse had not been lit for thirty years. Nevertheless, on the "
     "night the storm arrived, every sailor in the harbour swore they saw it "
     "burning, steady and white against the black water."),
    ("bugreport",
     "My build fails with 'undefined reference to symbol pthread_create'. "
     "The link line includes -lpthread and it worked yesterday. What changed "
     "and how do I diagnose which object file still needs it?"),
    ("json",
     'Return a JSON object describing a network incident with fields: '
     'severity, hosts (array), timeline (array of {ts, event}), and remediation.'),
]

PAD_TEXT = (
    "Context: the following log entries were collected from the target host "
    "during the engagement window. Each line is a timestamped syscall record. "
)


def build_workload(tokens_per_prompt, tokenizer, limit):
    """Repeat each prompt's text until it reaches roughly the requested length."""
    prompts = []
    for name, text in WORKLOAD[:limit]:
        body = text
        while len(tokenizer(body)["input_ids"]) < tokens_per_prompt:
            body = body + "\n\n" + text
        ids = tokenizer(body, return_tensors="pt", truncation=True,
                        max_length=tokens_per_prompt)["input_ids"]
        prompts.append((name, ids))
    return prompts


def build_longdoc(tokens, tokenizer, limit):
    """One long single-domain sequence, split in half.

    This is the honest cache test. With the mixed workload, the warm-up uses
    different *topics* from the test half, so a low static hit rate could just
    mean 'code experts differ from prose experts' rather than 'routing is
    uniform'. Here warm-up and test come from the same distribution, so a
    failure cannot be blamed on a topic change.
    """
    text = "\n\n".join(t for _, t in WORKLOAD[:limit])
    while len(tokenizer(text)["input_ids"]) < tokens * 2:
        text = text + "\n\n" + text
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=tokens * 2)["input_ids"]
    half = ids.shape[1] // 2
    return [("longdoc-warm", ids[:, :half]), ("longdoc-test", ids[:, half:])]


class RouterTracer:
    """Records which experts each layer's router picks, for every token."""

    def __init__(self, model, top_k):
        self.top_k = top_k
        self.picks = {}      # layer_idx -> list of LongTensor [n_tokens, top_k]
        self.handles = []
        layers = _moe_layers(model)
        for i, gate in enumerate(layers):
            self.handles.append(
                gate.register_forward_hook(self._make(i)))

    def _make(self, layer_idx):
        def hook(module, args, output):
            logits = output[0] if isinstance(output, tuple) else output
            k = min(self.top_k, logits.shape[-1])
            idx = torch.topk(logits.detach().float(), k, dim=-1).indices
            self.picks.setdefault(layer_idx, []).append(idx.cpu())
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stack(self):
        """layer_idx -> LongTensor [n_tokens, top_k]"""
        return {k: torch.cat(v, dim=0) for k, v in sorted(self.picks.items())}


def _layer_no(name):
    """Pull the layer index out of a module path like model.layers.7.mlp.gate"""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return 0


def _moe_layers(model):
    """Find the router ('gate') module of every sparse MoE block, in order."""
    named = [(name, _layer_no(name)) for name, _ in model.named_modules()
             if name.endswith(".mlp.gate")]
    named = [(n, i) for n, i in named if i not in (0, 1) or n.count(".") >= 4]
    return [model.get_submodule(n) for n, _ in
            sorted(named, key=lambda t: t[1])]


def concentration(picks, n_experts):
    """Lorenz-style summary: what share of picks the top-N% of experts take."""
    counts = torch.bincount(picks.flatten(), minlength=n_experts).float()
    total = counts.sum()
    if total == 0:
        return {}
    ordered, _ = torch.sort(counts, descending=True)
    cum = torch.cumsum(ordered, 0) / total
    out = {}
    for frac in (0.05, 0.10, 0.20, 0.50):
        n = max(1, int(round(frac * n_experts)))
        out[f"top_{int(frac * 100)}pct_experts_share"] = float(cum[n - 1])
    # How many experts are needed to cover 50/80/90/95% of picks?
    for target in (0.5, 0.8, 0.9, 0.95):
        n = int((cum < target).sum().item()) + 1
        out[f"experts_for_{int(target * 100)}pct_coverage"] = n
        out[f"experts_for_{int(target * 100)}pct_coverage_frac"] = n / n_experts
    # Gini coefficient of the count distribution.
    s, _ = torch.sort(counts)
    n = s.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    gini = float((2 * (idx * s).sum()) / (n * s.sum()) - (n + 1) / n)
    out["gini"] = gini
    out["n_experts"] = n_experts
    out["total_picks"] = int(total)
    out["distinct_experts_used"] = int((counts > 0).sum())
    return out


def aggregate_skew(by_layer, n_experts):
    """Per-layer concentration, summarised across layers.

    Pooling picks across layers is WRONG and flatters a uniform story: every
    layer owns its own expert set, so binning 'expert 7' from sixteen
    unrelated routers together averages sixteen different distributions and
    manufactures near-uniformity. Measured on OLMoE, pooling reports Gini
    0.106 while the layers themselves sit at 0.24-0.45 -- a 2-4x understatement.
    """
    per = {l: concentration(by_layer[l], n_experts) for l in sorted(by_layer)}
    keys = ("gini", "top_5pct_experts_share", "top_10pct_experts_share",
            "top_20pct_experts_share", "experts_for_50pct_coverage_frac",
            "experts_for_80pct_coverage_frac", "experts_for_90pct_coverage_frac",
            "experts_for_95pct_coverage_frac")
    agg = {}
    for k in keys:
        vals = [per[l][k] for l in per]
        agg[k] = {"mean": sum(vals) / len(vals),
                  "min": min(vals), "max": max(vals)}
    agg["per_layer"] = {str(l): per[l] for l in per}
    return agg


def simulate_static(warm, test, n_experts, cache_fracs):
    """Pin each layer's most-used experts; measure on held-out text.

    Per layer, because a cache slot holds 'expert 7 of layer 3' and cannot be
    shared with 'expert 7 of layer 9'.
    """
    total = sum(v.numel() for v in test.values())
    rows = []
    for frac in cache_fracs:
        k = max(1, int(round(frac * n_experts)))
        hits = 0
        for l, picks in test.items():
            counts = torch.bincount(warm[l].flatten(),
                                    minlength=n_experts).float()
            keep = torch.argsort(counts, descending=True)[:k]
            mask = torch.zeros(n_experts, dtype=torch.bool)
            mask[keep] = True
            hits += int(mask[picks.flatten()].sum())
        rows.append({"cache_frac": frac, "slots_per_layer": k,
                     "hit_rate": hits / total})
    return rows


def simulate_lru(test, n_experts, cache_fracs):
    """Independent LRU cache per layer, driven by the real pick sequence.

    Per token the router picks several experts at once; they are all read in
    the same step, so eviction only happens between tokens, never between
    experts used together.
    """
    total = sum(v.numel() for v in test.values())
    rows = []
    for frac in cache_fracs:
        cap = max(1, int(round(frac * n_experts)))
        hits = 0
        for l, picks in test.items():
            cache = []      # LRU order, least-recently-used first
            for t in range(picks.shape[0]):
                for e in picks[t].tolist():
                    if e in cache:
                        hits += 1
                        cache.remove(e)
                        cache.append(e)
                    else:
                        cache.append(e)
                while len(cache) > cap:
                    cache.pop(0)
        rows.append({"cache_frac": frac, "slots_per_layer": cap,
                     "hit_rate": hits / total, "policy": "lru"})
    return rows


def load_model(model_id, mode):
    """Load so it fits in 8 GiB. 13.8 GiB of bf16 does not, so quantise.

    4-bit is the default because this probe studies *routing*, not answer
    quality -- and routing decisions survive quantisation far better than
    generated text does. Falls back to a CPU/GPU split if bitsandbytes is
    unavailable, which is slow but still correct.
    """
    if mode in ("4bit", "auto"):
        try:
            from transformers import BitsAndBytesConfig
            qc = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
            m = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=qc, device_map="cuda:0")
            print("  loaded in 4-bit (nf4) on GPU", flush=True)
            return m
        except Exception as exc:  # noqa: BLE001 - fall back, never fail here
            print(f"  4-bit load unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to split bf16", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto")
    print("  loaded bf16 with a GPU/CPU split (slow, but routing is identical)",
          flush=True)
    return m


def expert_bytes(model, n_experts, bpw):
    """Bytes per expert at a given bits-per-weight, from the real tensor shapes."""
    per_layer = []
    for name, mod in model.named_modules():
        if name.endswith(".mlp.experts"):
            # Total params of all experts in this layer.
            total = sum(math.prod(p.shape) for p in mod.parameters())
            per_layer.append(total / n_experts)
    if not per_layer:
        return None, None
    avg = sum(per_layer) / len(per_layer)
    return avg * bpw / 8, per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--limit", type=int, default=len(WORKLOAD),
                    help="how many workload prompts to use")
    ap.add_argument("--tokens", type=int, default=320,
                    help="approx tokens per prompt")
    ap.add_argument("--bpw", type=float, default=4.5,
                    help="bits per weight, for sizing the cache in MB")
    ap.add_argument("--load", default="4bit", choices=("4bit", "bf16"),
                    help="how to fit a 13.8 GiB model in 8 GiB of VRAM")
    ap.add_argument("--workload", default="mixed", choices=("mixed", "longdoc"),
                    help="mixed = varied topics; longdoc = one domain split in half")
    ap.add_argument("--trace-only", action="store_true",
                    help="record and dump the raw trace, then stop")
    args = ap.parse_args()

    print(f"loading {args.model} ...", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.load)
    model.eval()
    placed = {}
    for p in model.parameters():
        placed[str(p.device)] = placed.get(str(p.device), 0) + p.numel()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s; "
          f"vram {torch.cuda.memory_allocated() / (1 << 30):.2f} GiB; "
          f"weight placement {placed}", flush=True)

    cfg = model.config
    n_experts = int(getattr(cfg, "num_experts", 0) or 64)
    top_k = int(getattr(cfg, "num_experts_per_tok", 0) or 8)
    n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 16)
    print(f"  experts/layer={n_experts}  top_k={top_k}  layers={n_layers}",
          flush=True)

    if args.workload == "longdoc":
        prompts = build_longdoc(args.tokens, tokenizer, args.limit)
    else:
        prompts = build_workload(args.tokens, tokenizer, args.limit)
    print(f"  workload: {args.workload}, {len(prompts)} prompts", flush=True)

    # Split the workload: the first half warms the cache, the second half is the
    # honest held-out test. A cache tuned and measured on the same text would
    # report a hit rate that does not survive contact with new input.
    warm, test = prompts[:max(1, len(prompts) // 2)], prompts[len(prompts) // 2:]
    if not test:
        test = prompts

    def trace(subset):
        tracer = RouterTracer(model, top_k)
        per_prompt = {}
        with torch.no_grad():
            for name, ids in subset:
                ids = ids.to("cuda:0")
                tracer.picks.clear()
                model(input_ids=ids)
                per_prompt[name] = tracer.stack()
        tracer.close()
        return per_prompt

    print("tracing warm-up half ...", flush=True)
    warm_trace = trace(warm)
    print("tracing held-out half ...", flush=True)
    test_trace = trace(test)

    warm_by_layer = {l: torch.cat([p[l] for p in warm_trace.values()], 0)
                     for l in next(iter(warm_trace.values()))}
    test_by_layer = {l: torch.cat([p[l] for p in test_trace.values()], 0)
                     for l in next(iter(test_trace.values()))}

    n_test_tokens = next(iter(test_by_layer.values())).shape[0]
    print(f"  test tokens={n_test_tokens}  layers={len(test_by_layer)}",
          flush=True)

    skew = aggregate_skew(test_by_layer, n_experts)
    g = skew["gini"]
    print(f"\nPER-LAYER ROUTING SKEW (mean across {len(test_by_layer)} layers)",
          flush=True)
    print(f"  Gini: {g['mean']:.3f}  (layer range {g['min']:.3f}-{g['max']:.3f})",
          flush=True)
    for pct in (5, 10, 20):
        s = skew[f"top_{pct}pct_experts_share"]["mean"]
        print(f"  top {pct:2d}% of experts take {s * 100:5.1f}% of picks",
              flush=True)
    for cov in (50, 80, 90, 95):
        f = skew[f"experts_for_{cov}pct_coverage_frac"]["mean"]
        print(f"  experts needed for {cov}% of picks: {f * 100:5.1f}% of them",
              flush=True)

    report = {
        "model": args.model,
        "workload": args.workload,
        "n_experts": n_experts,
        "top_k": top_k,
        "n_layers": n_layers,
        "test_tokens": int(n_test_tokens),
        "skew": skew,
    }

    cache_fracs = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
    print("\nPin the most-used experts (per layer), measured on held-out text:",
          flush=True)
    static_rows = simulate_static(warm_by_layer, test_by_layer, n_experts,
                                  cache_fracs)
    for r in static_rows:
        print(f"  cache {r['cache_frac'] * 100:5.1f}% "
              f"({r['slots_per_layer']:3d} slots/layer)  "
              f"hit {r['hit_rate'] * 100:5.1f}%", flush=True)
    report["static_topk"] = static_rows

    print("\nKeep the most-recently-used experts (per layer):", flush=True)
    lru_rows = simulate_lru(test_by_layer, n_experts, cache_fracs)
    for r in lru_rows:
        print(f"  cache {r['cache_frac'] * 100:5.1f}% "
              f"({r['slots_per_layer']:3d} slots/layer)  "
              f"hit {r['hit_rate'] * 100:5.1f}%", flush=True)
    report["lru"] = lru_rows

    per_expert, _ = expert_bytes(model, n_experts, args.bpw)
    if per_expert:
        kb = per_expert / 1024
        report["expert_bytes"] = per_expert
        print(f"\nexpert size at {args.bpw} bpw: {kb:.2f} KiB "
              f"({per_expert:.0f} bytes)", flush=True)
        print(f"full expert set per layer: "
              f"{n_experts * kb / 1024:.2f} MiB; "
              f"all {n_layers} layers: "
              f"{n_experts * n_layers * kb / 1024 / 1024:.2f} GiB", flush=True)
        print("\n  cache budget -> VRAM needed (all layers):", flush=True)
        for r in static_rows:
            need = r["slots_per_layer"] * n_layers * per_expert / (1 << 20)
            print(f"    {r['cache_frac'] * 100:5.1f}% hit {r['hit_rate'] * 100:5.1f}%"
                  f"   {need:8.1f} MiB", flush=True)

    if args.trace_only:
        torch.save({str(l): test_by_layer[l] for l in test_by_layer},
                   "olmoe_route_trace.pt")
        print("saved olmoe_route_trace.pt", flush=True)

    out = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "runs",
        f"moe_routing_probe_{args.workload}.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
