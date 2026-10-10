#!/usr/bin/env python
"""
gguf_meta.py -- inspect a GGUF model and plan the memory budget for this GPU.

Uses the official `gguf` library rather than a hand-rolled header parser (an
earlier hand-rolled version silently misaligned its metadata scan; the official
reader is correct, so there is no reason to reimplement it).

Reports, for a model blob:
  * exact architecture and the resulting KV-cache cost per token
  * where the bytes actually live, by ggml type and by tensor group
    (embedding / LM head / attention / FFN), because that is what determines
    whether a compression or re-quantisation idea can pay off
  * a fit table over context lengths x KV cache types for this card

Usage:
  python gguf_meta.py <model.gguf> [--vram-gib 8.0] [--other-gib 0.3]
  python gguf_meta.py --ollama qwen3.5:9b
"""
import argparse
import glob
import os
import sys

VRAM_TOTAL_GIB = 8188 / 1024.0        # RTX 4060 Laptop, measured


def find_ollama_blobs(model):
    """Locate Ollama's blob files for a model name."""
    root = os.path.expanduser("~/.ollama/models")
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    man = os.path.join(root, "manifests", "registry.ollama.ai", "library", name, tag)
    if os.path.exists(man):
        import json
        layers = json.load(open(man, encoding="utf-8"))["layers"]
        out = []
        for l in layers:
            p = os.path.join(root, "blobs", l["digest"].replace(":", "-"))
            if os.path.exists(p):
                out.append((p, l.get("mediaType", "")))
        return out
    return [(p, "") for p in sorted(glob.glob(os.path.join(root, "blobs", "*")),
                                     key=os.path.getsize, reverse=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="path to a .gguf file")
    ap.add_argument("--ollama", help="model name, e.g. qwen3.5:9b")
    ap.add_argument("--vram-gib", type=float, default=VRAM_TOTAL_GIB)
    ap.add_argument("--other-gib", type=float, default=0.30,
                    help="CUDA context / display / other resident overhead")
    args = ap.parse_args()

    from gguf import GGUFReader

    if args.ollama:
        cands = find_ollama_blobs(args.ollama)
    elif args.path:
        cands = [(args.path, "")]
    else:
        ap.error("need a path or --ollama")

    for path, mt in cands:
        if mt and "model" not in mt:
            continue
        try:
            r = GGUFReader(path)
        except Exception as e:
            print(f"skip {os.path.basename(path)}: {type(e).__name__}")
            continue
        f = r.fields
        arch = f["general.architecture"].contents() if "general.architecture" in f else "?"
        print(f"\n{'='*72}\n{os.path.basename(path)}  ({os.path.getsize(path)/1e9:.3f} GB)"
              f"  arch={arch}")
        if "general.architecture" not in f:
            print("  no GGUF metadata (not a model blob?)")
            continue

        def g(k):
            return f[f"{arch}.{k}"].contents() if f"{arch}.{k}" in f else None

        L, kvh, hd = g("block_count"), g("attention.head_count_kv"), g("attention.key_length")
        hid = g("embedding_length")
        print(f"  layers={L}  kv_heads={kvh}  head_dim={hd}  hidden={hid}  "
              f"file_type={f['general.file_type'].contents() if 'general.file_type' in f else '?'}")

        # Hybrid architectures (SSM / linear attention interleaved with full
        # attention) only keep a KV cache on the full-attention layers, so the
        # per-token KV cost is a FRACTION of what the layer count implies.
        # qwen35 uses `full_attention_interval` to say "1 in every N layers is
        # full attention".  Getting this wrong makes the fit table wildly
        # pessimistic -- it was, before this was added.
        interval = g("full_attention_interval")
        n_attn = L
        if isinstance(interval, int) and interval > 1:
            n_attn = (L + interval - 1) // interval
        ssm = {k.split(".ssm.", 1)[1]: v for k, v in f.items() if ".ssm." in k}
        if ssm:
            ssm = {k: v.contents() if hasattr(v, "contents") else v
                   for k, v in ssm.items()}
            print(f"  HYBRID: full attention every {interval} layers "
                  f"-> only {n_attn}/{L} layers carry a KV cache")
            print(f"  SSM layers: {L - n_attn} x (state {ssm.get('state_size')}, "
                  f"inner {ssm.get('inner_size')}, groups {ssm.get('group_count')}, "
                  f"conv {ssm.get('conv_kernel')})  = CONSTANT memory, no growth")
        n_kv_layers = n_attn

        # ---------------- where the bytes live ----------------
        tot_ne = tot_b = 0
        by_type, by_group = {}, {}
        for t in r.tensors:
            ne = 1
            for d in t.shape:
                ne *= int(d)
            nb = t.data.nbytes
            tot_ne += ne
            tot_b += nb
            e = by_type.setdefault(t.tensor_type.name, [0, 0])
            e[0] += ne
            e[1] += nb
            if "token_embd" in t.name:
                grp = "embedding"
            elif "output.weight" in t.name:
                grp = "lm_head"
            elif "attn" in t.name:
                grp = "attention"
            elif "ffn" in t.name:
                grp = "ffn"
            else:
                grp = "other"
            e = by_group.setdefault(grp, [0, 0])
            e[0] += ne
            e[1] += nb

        print(f"  TOTAL {tot_ne/1e9:.3f} B params in {tot_b/1e9:.3f} GB "
              f"= {8*tot_b/tot_ne:.3f} bpw average")
        print("\n  --- by ggml type ---")
        for k, (ne, b) in sorted(by_type.items(), key=lambda x: -x[1][1]):
            print(f"    {k:10s} {ne/1e6:9.2f} M elems  {b/1e9:6.3f} GB  {8*b/ne:6.3f} bpw")
        print("\n  --- by tensor group (this is where compression can pay) ---")
        for k, (ne, b) in sorted(by_group.items(), key=lambda x: -x[1][1]):
            print(f"    {k:10s} {ne/1e6:9.2f} M elems  {b/1e9:6.3f} GB  "
                  f"{100*b/tot_b:5.1f}% of model")

        # ---------------- fit table ----------------
        if not (L and kvh and hd):
            print("\n  (cannot derive KV geometry; skipping fit table)")
            continue
        per_tok_f16 = 2 * n_kv_layers * kvh * hd * 2   # bytes, K and V, fp16
        print(f"\n  KV cache = {per_tok_f16/1024:.1f} KiB per token "
              f"(2 x {n_kv_layers} KV layers x {kvh} kv_heads x {hd} dim x 2 B)"
              + (f"  [a {L}-layer pure transformer would be "
                 f"{2*L*kvh*hd*2/1024:.1f} KiB/token]" if n_attn < L else ""))
        weights_gib = tot_b / 2**30
        # SSM layers hold a fixed-size state (no growth with context); count it
        # in the fixed cost so the fit table stays honest for hybrid models.
        ssm_mib = 0.0
        if ssm and n_attn < L:
            try:
                ssm_mib = ((L - n_attn) * int(ssm["inner_size"])
                           * int(ssm["state_size"]) * 4 / 2**20)
            except Exception:
                ssm_mib = 0.0
        fixed = weights_gib + args.other_gib + ssm_mib / 1024.0
        # bytes per element: fp16 = 2, q8_0 = 34/32, q4_0 = 18/32
        kv_scale = {"fp16": 1.0, "q8_0": 34/32/2, "q4_0": 18/32/2}
        print(f"\n  weights {weights_gib:.2f} GiB + SSM state {ssm_mib/1024:.2f} GiB "
              f"+ overhead {args.other_gib:.2f} GiB "
              f"= {fixed:.2f} GiB fixed, of {args.vram_gib:.2f} GiB total")
        print(f"\n  {'ctx':>6s} " + "".join(f"{k:>11s}" for k in kv_scale))
        for ctx in (4096, 8192, 16384, 32768, 65536, 131072):
            row = f"  {ctx//1024:4d}K "
            for k, sc in kv_scale.items():
                need = fixed + per_tok_f16 * ctx * sc / 2**30
                mark = "OK" if need <= args.vram_gib else "NO"
                row += f"{need:8.2f}{mark:>3s}"
            print(row)
        biggest = max((t for t in r.tensors), key=lambda t: t.data.nbytes)
        print(f"\n  largest single tensor: {biggest.name} "
              f"({biggest.tensor_type.name}, {biggest.data.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
