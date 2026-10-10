#!/usr/bin/env python
"""
compress_df11_par.py -- same output format as dfloat11.compress_model, but one
worker process per transformer block (their version does ~50M weights per layer
through a pure-Python Huffman encoder, single-threaded: ~50-95s per layer).

Writes, matching what DFloat11Model.from_pretrained expects:
  <out>/config.json                     (+ dfloat11_config)
  <out>/model_layers_<i>.safetensors    (luts / encoded_exponent / sign_mantissa /
                                         output_positions / gaps / split_positions)
  <out>/model.safetensors               (all non-compressed weights, bf16)
  <out>/<tokenizer files>, generation_config.json

Then verifies every layer by decoding it with the DFloat11 CUDA kernel and
comparing against the original bf16 weights (round-trip exactness).

Usage:
  python compress_df11_par.py --base Qwen/Qwen3-1.7B --out experiments/df11_qwen3_1.7b --workers 8
"""
import argparse, json, os, re, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

BYTES_PER_THREAD = 8
THREADS_PER_BLOCK = 512

# DFloat11's own pattern_dict convention: the 7 attention/MLP projections per block.
# Everything else stays bf16 and is written out uncompressed (asserted in main()).
PROJ_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$")


# --------------------------------------------------------------- worker side
def _shard(src, weight_map, name):
    return os.path.join(src, weight_map[name])


def _encode_block(job):
    """Runs in a worker process: encode one transformer block, write its file."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    block = job["block"]
    weight_names, out_dir = job["weight_names"], job["out_dir"]

    def shard_of(name):
        return _shard(job["src"], job["weight_map"], name)

    import contextlib, io
    import dahuffman
    import dfloat11.dfloat11_utils as U
    import dfloat11.dfloat11 as D

    dahuffman.HuffmanCodec.print_code_table = lambda self, *a, **k: None
    _orig_luts, _orig_32 = D.get_luts, D.get_32bit_codec
    D.get_luts = lambda t: (lambda b: (contextlib.redirect_stdout(b), _orig_luts(t))[1])(io.StringIO())
    D.get_32bit_codec = lambda c: (lambda b: (contextlib.redirect_stdout(b), _orig_32(c))[1])(io.StringIO())

    t0 = time.perf_counter()
    weights = []
    for wn in weight_names:
        with safe_open(shard_of(wn), framework="pt") as f:
            weights.append(f.get_tensor(wn).flatten().to(torch.bfloat16).clone())
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _codec, counter = U.get_codec(torch.cat(weights))
    codec, _c, table = D.get_32bit_codec(counter)
    luts = D.get_luts(table)
    encoded, other_8bits, output_positions, gaps, split_positions = U.encode_weights(
        weights, codec, BYTES_PER_THREAD, THREADS_PER_BLOCK
    )
    enc_s = time.perf_counter() - t0

    state = {
        f"{block}.luts": luts,
        f"{block}.encoded_exponent": encoded,
        f"{block}.sign_mantissa": other_8bits,
        f"{block}.output_positions": output_positions.view(torch.uint8),
        f"{block}.gaps": gaps,
        f"{block}.split_positions": split_positions,
    }
    fname = os.path.join(out_dir, block.replace(".", "_") + ".safetensors")
    save_file(state, fname)

    n_el = sum(w.numel() for w in weights)
    stored = (encoded.numel() + other_8bits.numel()
              + output_positions.numel() * 4 + gaps.numel())
    return {
        "block": block,
        "file": os.path.basename(fname),
        "elements": int(n_el),
        "stored_bytes": int(stored),
        "bf16_bytes": int(n_el * 2),
        "ratio": round(n_el * 2 / stored, 4),
        "load_s": round(load_s, 2),
        "encode_s": round(enc_s, 2),
    }


# --------------------------------------------------------------- main process
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--skip-encode", action="store_true",
                    help="reuse layer files already in --out and only write the "
                         "uncompressed weights / config, then verify")
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    src = args.base if os.path.isdir(args.base) else snapshot_download(
        args.base, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
    print(f"[par] source: {src}", flush=True)

    index_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = json.load(open(index_path, encoding="utf-8"))["weight_map"]
    else:
        single = [f for f in os.listdir(src) if f.endswith(".safetensors") and "index" not in f]
        with safe_open(os.path.join(src, single[0]), framework="pt") as f:
            weight_map = {k: single[0] for k in f.keys()}

    def shard_of(name):
        return os.path.join(src, weight_map[name])

    cfg_json = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))

    all_names = list(weight_map)
    compressed_names = [n for n in all_names if PROJ_RE.match(n)]
    blocks = sorted({f"model.layers.{PROJ_RE.match(n).group(1)}" for n in compressed_names},
                    key=lambda b: int(b.rsplit(".", 1)[1]))
    common = sorted({PROJ_RE.match(n).group(2) for n in compressed_names})
    if not compressed_names:
        sys.exit("[par] no compressible projections matched -- extend PROJ_RE for this architecture")
    # every block must carry the same projection set, and the concatenation order
    # used by encode_weights must match the order the loader rebuilds it from config
    for b in blocks:
        got = sorted(n[len(b) + 1:-len(".weight")] for n in compressed_names
                     if n.startswith(b + ".") and n.endswith(".weight"))
        if got != common:
            sys.exit(f"[par] projection set differs in {b}: {got} != {common}")
    print(f"[par] {len(blocks)} blocks x {len(common)} projections: {common}", flush=True)

    if args.skip_encode:
        os.makedirs(args.out, exist_ok=True)
    else:
        if os.path.isdir(args.out) and os.listdir(args.out):
            print(f"[par] clearing {args.out}")
            shutil.rmtree(args.out)
        os.makedirs(args.out, exist_ok=True)
    # tokenizer + generation config
    AutoTokenizer.from_pretrained(src).save_pretrained(args.out)
    for f in ("generation_config.json",):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.out, f))

    jobs = [{"idx": i, "block": b, "weight_names": [f"{b}.{p}.weight" for p in common],
             "src": src, "weight_map": weight_map, "out_dir": args.out}
            for i, b in enumerate(blocks)]

    results = []
    if args.skip_encode:
        missing = [b for b in blocks
                   if not os.path.exists(os.path.join(args.out, b.replace(".", "_") + ".safetensors"))]
        if missing:
            sys.exit(f"[par] --skip-encode but layer files missing: {missing}")
        print(f"[par] --skip-encode: reusing {len(blocks)} existing layer files", flush=True)
    else:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_encode_block, j): j["block"] for j in jobs}
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                print(f"[par] {r['block']:20s} ratio {r['ratio']:.3f}x "
                      f"(encode {r['encode_s']:5.1f}s)", flush=True)
        print(f"[par] {len(results)} blocks compressed in {time.perf_counter()-t0:.1f}s", flush=True)

    # non-compressed weights -> model.safetensors
    compressed_prefixes = tuple(f"{b}.{p}.weight" for b in blocks for p in common)
    compressed_set = set(compressed_names)
    rest = [n for n in all_names if n not in compressed_set]
    accounted = compressed_set | set(rest)
    if accounted != set(all_names):
        sys.exit(f"[par] INTERNAL: {len(set(all_names) - accounted)} source tensors unaccounted for")
    print(f"[par] coverage: {len(compressed_names)} compressed + {len(rest)} stored bf16 "
          f"= {len(all_names)} source tensors", flush=True)
    rest_state = {}
    for name in rest:
        with safe_open(shard_of(name), framework="pt") as f:
            rest_state[name] = f.get_tensor(name)
    # a tied lm_head is an alias of embed_tokens: named_parameters() reports only
    # one of the two, so DFloat11's loader cannot look up 'lm_head.weight'.
    if (cfg_json.get("tie_word_embeddings") and "lm_head.weight" in rest_state
            and torch.equal(rest_state["lm_head.weight"], rest_state["model.embed_tokens.weight"])):
        del rest_state["lm_head.weight"]
        rest = [n for n in rest if n != "lm_head.weight"]
        print("[par] dropped duplicate tied lm_head.weight", flush=True)
    save_file(rest_state, os.path.join(args.out, "model.safetensors"))
    print(f"[par] model.safetensors: {len(rest)} uncompressed tensors "
          f"({sum(t.numel() for t in rest_state.values())*2/1e9:.3f} GB bf16)", flush=True)

    dfloat11_config = {
        "version": "0.5.0",
        "threads_per_block": [THREADS_PER_BLOCK],
        "bytes_per_thread": BYTES_PER_THREAD,
        "pattern_dict": {r"model\.layers\.\d+": list(common)},
    }
    cfg_path = os.path.join(args.out, "config.json")
    cfg_json["dfloat11_config"] = dfloat11_config
    json.dump(cfg_json, open(cfg_path, "w", encoding="utf-8"), indent=2)

    if results:
        total_stored = sum(r["stored_bytes"] for r in results)
        total_bf16 = sum(r["bf16_bytes"] for r in results)
    else:  # resumed run: recompute from what is on disk
        total_stored = sum(os.path.getsize(os.path.join(args.out, b.replace(".", "_") + ".safetensors"))
                           for b in blocks)
        total_bf16 = len(compressed_names) * 0  # filled from source below
        for n in compressed_names:
            with safe_open(shard_of(n), framework="pt") as f:
                total_bf16 += f.get_slice(n).get_shape()[0] * f.get_slice(n).get_shape()[1] * 2
    rest_bytes = sum(t.numel() * t.element_size() for t in rest_state.values())
    print(f"[par] compressed part: {total_bf16/1e9:.3f} GB bf16 -> {total_stored/1e9:.3f} GB "
          f"({total_bf16/total_stored:.3f}x)", flush=True)
    print(f"[par] whole model     : {(total_bf16+rest_bytes)/1e9:.3f} GB bf16 -> "
          f"{(total_stored+rest_bytes)/1e9:.3f} GB ({(total_bf16+rest_bytes)/(total_stored+rest_bytes):.3f}x)",
          flush=True)

    if not args.no_verify:
        verify(args.out, shard_of, blocks, common, weight_map=weight_map, _src_dir=src)

    os.makedirs(os.path.join("experiments", "runs"), exist_ok=True)
    json.dump({"base": args.base, "results": sorted(results, key=lambda r: r["block"]),
               "compressed_ratio": round(total_bf16 / total_stored, 4)},
              open(os.path.join("experiments", "runs",
                                f"compress_{os.path.basename(args.out)}.json"), "w"), indent=2)


def verify(out_dir, shard_of, blocks, common, weight_map=None, _src_dir=None):
    """Decode every compressed block with the DFloat11 kernel, compare to bf16."""
    import math
    import torch
    import cupy as cp
    from safetensors.torch import load_file
    import dfloat11.dfloat11 as D

    dev = torch.device("cuda:0")
    # read the original weights one shard at a time (per-tensor safe_open handles
    # crash this process on Windows after ~40 opens)
    wanted = {f"{b}.{p}.weight" for b in blocks for p in common}
    ref_weights = {}
    for shard in sorted(set(weight_map[n] for n in wanted)):
        example = next(n for n in wanted if weight_map[n] == shard)
        sd = load_file(_shard(_src_dir, weight_map, example))
        ref_weights.update({k: v for k, v in sd.items() if k in wanted})
        del sd
    print(f"[verify] loaded {len(ref_weights)} original weight tensors for comparison", flush=True)
    ok, bad = 0, []
    for b in blocks:
        fname = os.path.join(out_dir, b.replace(".", "_") + ".safetensors")
        sd = load_file(fname)
        enc = sd[f"{b}.encoded_exponent"].to(dev)
        oth = sd[f"{b}.sign_mantissa"].to(dev)
        lut = sd[f"{b}.luts"].to(dev)
        opos = sd[f"{b}.output_positions"].to(dev)
        gaps = sd[f"{b}.gaps"].to(dev)
        n_el, n_bytes, n_luts = oth.numel(), enc.numel(), lut.shape[0]
        out = torch.empty(n_el, dtype=torch.bfloat16, device=dev)
        opos_np = opos.view(torch.uint32).cpu().numpy()
        smem = THREADS_PER_BLOCK * 4 + 4 + int((opos_np[1:] - opos_np[:-1]).max()) * 2
        grid = (int(math.ceil(n_bytes / (THREADS_PER_BLOCK * BYTES_PER_THREAD))),)
        with cp.cuda.Device(0):
            D._decode(grid=grid, block=(THREADS_PER_BLOCK,), shared_mem=smem,
                      args=[lut.data_ptr(), enc.data_ptr(), oth.data_ptr(),
                            opos.data_ptr(), gaps.data_ptr(), out.data_ptr(),
                            n_luts, n_bytes, n_el])
        torch.cuda.synchronize()
        orig = [ref_weights[f"{b}.{p}.weight"].flatten().to(torch.bfloat16) for p in common]
        ref = torch.cat(orig)
        same = bool(torch.equal(ref, out.cpu()))
        ok += same
        if not same:
            bad.append(b)
        print(f"[verify] {b:20s} {'EXACT' if same else 'MISMATCH'}", flush=True)
        del enc, oth, lut, opos, gaps, out
    print(f"[verify] {ok}/{len(blocks)} blocks bit-exact"
          + (f"  FAILED: {bad}" if bad else ""), flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
