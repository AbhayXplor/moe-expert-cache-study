#!/usr/bin/env python3
"""
stageA_dfloat11_validate.py — validate the DFloat11 lossless claim on YOUR GPU
before building anything on top of it.

Measures, for a DFloat11-compressed model vs its BF16 original:
  1. compressed footprint on disk (safetensors bytes) vs base model
  2. peak GPU memory during load + generation (torch.cuda.max_memory_allocated)
  3. decode throughput (tokens/s) on a fixed prompt
  4. OPTIONAL greedy-identity check: same prompt, temp=0, both models ->
     token streams must match exactly (lossless claim verification)

Prereqs: CUDA-12 NVIDIA GPU,  pip install "dfloat11[cuda12]" torch transformers accelerate

Usage:
  python stageA_dfloat11_validate.py --model DFloat11/<NAME>-DF11 \
      [--base <hf-id-of-BF16-original>] [--gen-tokens 128] [--identity-check]

Notes:
  - Model ids live under the HF org 'DFloat11' (e.g. Llama-3.x/Qwen3/Gemma3/Phi4
    DF11 variants). Pick one that has a BF16 sibling for --base.
  - DFloat11Model mirrors HF from_pretrained usage; if the API drifts, check
    github.com/LeanModels/DFloat11 (inference.py example).
  - Do NOT enable cpu_offload here: this test measures the GPU-resident path.
"""
import argparse, json, time, sys

def human(nbytes): return f"{nbytes/1e9:.2f} GB"

def repo_bytes(repo_id: str) -> int:
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
        return sum((s.size or 0) for s in info.siblings
                   if s.rfilename.endswith((".safetensors", ".bin")))
    except Exception as e:
        print(f"[warn] cannot stat {repo_id}: {e}")
        return -1

def load_dfloat11(model_id: str):
    import torch
    from dfloat11 import DFloat11Model
    t0 = time.time()
    model = DFloat11Model.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                          device_map="cuda:0")
    tok = model.load_tokenizer()
    print(f"[load] DFloat11 model loaded in {time.time()-t0:.1f}s")
    return model, tok

def bench_generation(model, tok, prompt: str, gen_tokens: int):
    import torch
    torch.cuda.reset_peak_memory_stats()
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=gen_tokens, do_sample=False,
                             temperature=None, top_p=None)
    dt = time.time() - t0
    n = out.shape[-1] - inputs["input_ids"].shape[-1]
    peak = torch.cuda.max_memory_allocated()
    print(f"[bench] generated {n} tokens in {dt:.2f}s -> {n/dt:.1f} tok/s | peak GPU alloc {human(peak)}")
    return out[0].tolist(), n / dt, peak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="DFloat11 HF id, e.g. DFloat11/<name>-DF11")
    ap.add_argument("--base", default="", help="BF16 original HF id for size + identity comparison")
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--identity-check", action="store_true",
                    help="also load base model and compare greedy token streams")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    prompt = ("Explain precisely how a POSIX shell parses a pipeline with "
              "command substitution and file-descriptor redirection, citing exact syntax: ")

    report = {"model": args.model}
    c_size = repo_bytes(args.model)
    report["compressed_bytes"] = c_size
    if args.base:
        b_size = repo_bytes(args.base)
        report["base_bytes"] = b_size
        if c_size > 0 and b_size > 0:
            report["size_ratio"] = round(b_size / c_size, 3)
            print(f"[size] base {human(b_size)} -> DF11 {human(c_size)}  ratio {report['size_ratio']}x "
                  f"(expected ~1.4x)")

    model, tok = load_dfloat11(args.model)
    toks_df11, tps, peak = bench_generation(model, tok, prompt, args.gen_tokens)
    report.update({"tok_per_s": round(tps, 2), "peak_gpu_bytes": peak})

    if args.identity_check:
        if not args.base:
            sys.exit("--identity-check requires --base")
        import torch, transformers
        del model
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        print(f"[base] loading {args.base} (BF16) ...")
        bmodel = transformers.AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map="cuda:0")
        btok = transformers.AutoTokenizer.from_pretrained(args.base)
        toks_base, tps_b, peak_b = bench_generation(bmodel, btok, prompt, args.gen_tokens)
        report["base_tok_per_s"] = round(tps_b, 2)
        n = min(len(toks_df11), len(toks_base))
        first_diff = next((i for i in range(n) if toks_df11[i] != toks_base[i]), -1)
        report["identity"] = {"match": first_diff == -1 and len(toks_df11) == len(toks_base),
                              "first_divergence_index": first_diff, "compared": n}
        print(f"[identity] greedy streams {'IDENTICAL' if report['identity']['match'] else 'DIVERGED at index %d' % first_diff}"
              f" ({n} tokens compared)")
        if not report["identity"]["match"]:
            print("[identity] NOTE: DFloat11 claims bit-identical outputs. Divergence here means "
                  "either different sampling defaults between runtimes or a real codec issue — "
                  "re-test with both models on the same transformers version before concluding.")

    print(json.dumps(report, indent=2))
    if args.json:
        json.dump(report, open(args.json, "w"), indent=2)
        print(f"[done] report -> {args.json}")

if __name__ == "__main__":
    main()
