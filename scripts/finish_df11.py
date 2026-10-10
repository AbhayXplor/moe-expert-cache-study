#!/usr/bin/env python
"""
finish_df11.py -- complete a partially built DF11 model directory.

The parallel compressor writes the 28 compressed layer files first, then the
uncompressed remainder + config. If that second half is interrupted (or the
output directory was recreated), this finishes the job without re-encoding:

  1. writes <out>/model.safetensors with every non-compressed weight (bf16)
  2. writes <out>/config.json including dfloat11_config
  3. verifies every compressed layer against the original bf16 weights

Usage:
  python finish_df11.py --base Qwen/Qwen3-1.7B --out experiments/df11_qwen3_1.7b
"""
import argparse, importlib.util, json, os, sys

spec = importlib.util.spec_from_file_location("cdf", "scripts/compress_df11_par.py")
cdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    import psutil
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    vm = psutil.virtual_memory()
    print(f"[finish] RAM free {vm.available/1e9:.1f} GB of {vm.total/1e9:.1f} GB", flush=True)

    src = args.base if os.path.isdir(args.base) else snapshot_download(
        args.base, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
    index_path = os.path.join(src, "model.safetensors.index.json")
    weight_map = json.load(open(index_path, encoding="utf-8"))["weight_map"]

    def shard_of(name):
        return os.path.join(src, weight_map[name])

    all_names = list(weight_map)
    compressed_names = [n for n in all_names if cdf.PROJ_RE.match(n)]
    blocks = sorted({f"model.layers.{cdf.PROJ_RE.match(n).group(1)}" for n in compressed_names},
                    key=lambda b: int(b.rsplit(".", 1)[1]))
    common = sorted({cdf.PROJ_RE.match(n).group(2) for n in compressed_names})
    rest = [n for n in all_names if n not in set(compressed_names)]
    print(f"[finish] {len(compressed_names)} compressed / {len(rest)} stored bf16", flush=True)

    # NOTE: opening a safetensors handle per tensor silently crashes this process
    # after ~40 reads on Windows (safetensors 0.8.0), so read one shard at a time.
    from safetensors.torch import load_file
    compressed_set = set(compressed_names)
    rest_state = {}
    for shard in sorted(set(weight_map[n] for n in rest)):
        sd = load_file(shard_of(next(n for n in rest if weight_map[n] == shard)))
        rest_state.update({k: v for k, v in sd.items() if k in set(rest) and k not in compressed_set})
        del sd
        print(f"[finish] read {shard}: {len(rest_state)} tensors kept", flush=True)
    print(f"[finish] loaded {len(rest_state)} tensors from source", flush=True)

    cfg_json = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    if (cfg_json.get("tie_word_embeddings") and "lm_head.weight" in rest_state
            and torch.equal(rest_state["lm_head.weight"], rest_state["model.embed_tokens.weight"])):
        del rest_state["lm_head.weight"]
        print("[finish] dropped duplicate tied lm_head.weight", flush=True)

    out_model = os.path.join(args.out, "model.safetensors")
    save_file(rest_state, out_model)
    print(f"[finish] wrote {out_model} "
          f"({sum(t.numel()*t.element_size() for t in rest_state.values())/1e9:.3f} GB)", flush=True)

    cfg_json["dfloat11_config"] = {
        "version": "0.5.0",
        "threads_per_block": [cdf.THREADS_PER_BLOCK],
        "bytes_per_thread": cdf.BYTES_PER_THREAD,
        "pattern_dict": {r"model\.layers\.\d+": list(common)},
    }
    json.dump(cfg_json, open(os.path.join(args.out, "config.json"), "w", encoding="utf-8"), indent=2)
    print("[finish] wrote config.json with dfloat11_config", flush=True)

    stored = sum(os.path.getsize(os.path.join(args.out, b.replace(".", "_") + ".safetensors"))
                 for b in blocks)
    rest_bytes = sum(t.numel() * t.element_size() for t in rest_state.values())
    bf16_total = rest_bytes
    src_sizes = {}
    for shard in sorted(set(weight_map[n] for n in compressed_names)):
        sd = load_file(shard_of(next(n for n in compressed_names if weight_map[n] == shard)))
        src_sizes.update({k: v.numel() * 2 for k, v in sd.items() if k in set(compressed_names)})
        del sd
    bf16_total += sum(src_sizes.values())
    print(f"[finish] model: {bf16_total/1e9:.3f} GB bf16 -> {(stored+rest_bytes)/1e9:.3f} GB DF11 "
          f"({bf16_total/(stored+rest_bytes):.3f}x overall)", flush=True)

    if not args.no_verify:
        del rest_state
        cdf.verify(args.out, shard_of, blocks, common, weight_map=weight_map, _src_dir=src)


if __name__ == "__main__":
    main()
