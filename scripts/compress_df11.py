#!/usr/bin/env python
"""
compress_df11.py -- build a DFloat11-compressed copy of ANY small HF model with
the released DFloat11 compression code (no reliance on their prebuilt HF repos).

Usage:
  python compress_df11.py --base Qwen/Qwen3-1.7B --out experiments/df11_qwen3_1.7b
"""
import argparse, contextlib, io, json, os, re, shutil, time

import torch
import torch.nn as nn


def quiet_dfloat11():
    """The released compressor prints a 256-line Huffman table per module and a
    dict per LUT prefix. Over 196 modules that dominates the runtime."""
    import dahuffman
    import dfloat11.dfloat11_utils as U
    import dfloat11.dfloat11 as D

    dahuffman.HuffmanCodec.print_code_table = lambda self, *a, **k: None
    orig_get_luts = D.get_luts  # the name compress_model actually calls
    orig_get_32bit = D.get_32bit_codec

    def get_luts(table):
        with contextlib.redirect_stdout(io.StringIO()):
            return orig_get_luts(table)

    def get_32bit_codec(counter):
        with contextlib.redirect_stdout(io.StringIO()):
            return orig_get_32bit(counter)

    D.get_luts = get_luts
    D.get_32bit_codec = get_32bit_codec
    U.get_luts = get_luts


def discover_pattern_dict(model):
    """Find the transformer blocks and the linear projections inside them."""
    block_re = re.compile(r"^(.*\.layers\.\d+)$")
    blocks = {}
    for name, mod in model.named_modules():
        m = block_re.match(name)
        if m:
            blocks[m.group(1)] = mod
    prefixes = {name.rsplit(".layers.", 1)[0] for name in blocks}

    prefix = sorted(prefixes)[0]
    lin_sets = []
    for bname in sorted(blocks, key=lambda n: int(n.rsplit(".", 1)[-1])):
        lin = set()
        for cname, cmod in blocks[bname].named_children():
            if isinstance(cmod, nn.Linear):
                lin.add(cname)
            elif isinstance(cmod, nn.Module):
                for gcname, gcmod in cmod.named_children():
                    if isinstance(gcmod, nn.Linear):
                        lin.add(f"{cname}.{gcname}")
        lin_sets.append(lin)
    common = sorted(set.intersection(*lin_sets)) if lin_sets else []
    if not common:
        raise SystemExit(f"no common linear projections found; per-layer sets: {lin_sets[:2]}")
    pattern = f"{re.escape(prefix)}\\.layers\\.\\d+"
    return {pattern: common}, prefix, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-correctness-check", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep existing output dir")
    args = ap.parse_args()

    if os.path.isdir(args.out) and not args.keep:
        print(f"[compress] removing existing {args.out}")
        shutil.rmtree(args.out)

    quiet_dfloat11()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dfloat11.dfloat11 import compress_model

    t0 = time.perf_counter()
    print(f"[compress] loading {args.base} (bf16, cpu)")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model.eval()

    pattern_dict, prefix, common = discover_pattern_dict(model)
    print(f"[compress] pattern_dict = {json.dumps(pattern_dict)}")

    os.makedirs(args.out, exist_ok=True)
    t1 = time.perf_counter()
    compress_model(
        model,
        pattern_dict,
        args.out,
        save_single_file=False,
        check_correctness=not args.no_correctness_check,
    )
    print(f"[compress] compress_model() took {time.perf_counter()-t1:.1f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out)
    try:
        from transformers import GenerationConfig
        GenerationConfig.from_pretrained(args.base).save_pretrained(args.out)
    except Exception as e:
        print(f"[compress] generation config not saved: {e}")

    # verify the written config carries dfloat11_config
    cfg = json.load(open(os.path.join(args.out, "config.json"), encoding="utf-8"))
    assert "dfloat11_config" in cfg, "dfloat11_config missing from written config.json"

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out) if f.endswith(".safetensors"))
    print(f"[compress] done in {time.perf_counter()-t0:.1f}s  "
          f"safetensors bytes = {total/1e9:.3f} GB ({total/1024**3:.3f} GiB)")


if __name__ == "__main__":
    main()
