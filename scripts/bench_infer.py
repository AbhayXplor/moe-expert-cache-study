#!/usr/bin/env python
"""
bench_infer.py -- one-shot inference measurement for a single model/config.

Runs ONE model in ONE process (so VRAM numbers are not contaminated by other
models), measures:
  * resident bytes by tensor class (bf16 weights / DF11 compressed streams / ...)
  * torch allocator stats after load, and peak during decode
  * system-wide dedicated VRAM from nvidia-smi (sampled in a thread)
  * prefill (TTFT) latency and per-step decode latency (manual greedy loop)
  * the greedy token stream + top-1 logit per step (for identity comparison)

Usage:
  python bench_infer.py --kind bf16 --model Qwen/Qwen3-1.7B --out experiments/runs/e0.json
  python bench_infer.py --kind df11 --model experiments/df11_qwen3_1.7b --out experiments/runs/e2a.json
"""
import argparse, json, os, subprocess, sys, threading, time

import torch


# ---------------------------------------------------------------- VRAM sampler
class NvSampler(threading.Thread):
    """Poll nvidia-smi for system-wide dedicated VRAM + utilization."""

    def __init__(self, interval=0.2, index=0):
        super().__init__(daemon=True)
        self.interval, self.index = interval, index
        self.samples = []           # (t, used_mib, util_pct)
        self._stop = threading.Event()
        self.t0 = None

    def run(self):
        self.t0 = time.perf_counter()
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.index}",
                     "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()
                if out:
                    used, util = (x.strip() for x in out[0].split(",")[:2])
                    self.samples.append((time.perf_counter() - self.t0, int(used), int(util)))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def summary(self):
        if not self.samples:
            return {}
        used = [s[1] for s in self.samples]
        util = [s[2] for s in self.samples]
        return {
            "n_samples": len(used),
            "min_used_mib": min(used),
            "max_used_mib": max(used),
            "util_max_pct": max(util),
            "util_mean_pct_while_active": round(sum(util) / len(util), 1),
        }


# --------------------------------------------------------- resident accounting
def classify(name, tensor):
    n = name.lower()
    for key in ("encoded_exponent", "sign_mantissa", "output_positions", "gaps", "luts"):
        if key in n:
            return f"df11:{key}"
    # 4-bit quantized weights: packed uint8, plus fp32/bf16 absmax+quant_map
    if tensor.dim() > 1 and tensor.dtype in (torch.uint8, torch.int8):
        return "int4:packed"
    if tensor.dtype == torch.float32 and tensor.numel() < 20_000_000:
        return "quant:scales"
    if tensor.dtype == torch.bfloat16 and tensor.dim() > 1:
        return "bf16:matrix"
    return f"other:{str(tensor.dtype).replace('torch.', '')}"


def resident_accounting(model):
    """Bytes resident on GPU, grouped by tensor class. Proves what is in VRAM."""
    groups = {}
    for name, p in model.named_parameters():
        if p.device.type != "cuda":
            continue
        k = classify(name, p.data)
        groups[k] = groups.get(k, 0) + p.numel() * p.element_size()
    for name, b in model.named_buffers():
        if b.device.type != "cuda":
            continue
        k = classify(name, b)
        groups[k] = groups.get(k, 0) + b.numel() * b.element_size()
    off_cpu = 0
    for _, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.device.type != "cuda":
            off_cpu += t.numel() * t.element_size()
    return groups, off_cpu


def n_params_effective(model):
    """Count of logical weight elements (compressed streams counted as their
    logical element count, i.e. encoded_exponent/sign_mantissa numel)."""
    total = 0
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        n = name.lower()
        if "encoded_exponent" in n or "sign_mantissa" in n:
            continue  # counted via sign_mantissa below
        total += t.numel()
    # add logical params inside compressed layers
    enc = sum(t.numel() for n, t in list(model.named_parameters()) + list(model.named_buffers())
              if "sign_mantissa" in n.lower())
    return total + enc


# --------------------------------------------------------------------- loading
def load_model(kind, model_id, device="cuda:0"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    t0 = time.perf_counter()

    if kind == "bf16":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model = model.to(device)
    elif kind == "df11":
        from dfloat11 import DFloat11Model
        model = DFloat11Model.from_pretrained(model_id, device=device)
    elif kind == "nf4":
        # the lossy reference point: 4-bit NF4, double quantization, bf16 compute
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=qc, dtype=torch.bfloat16,
            device_map={"": device},
        )
    else:
        raise SystemExit(f"unknown kind {kind}")

    model.eval()
    load_s = time.perf_counter() - t0
    return model, tok, load_s


# ------------------------------------------------------------------ bench loop
@torch.no_grad()
def run_bench(model, tok, prompt, gen_tokens, device="cuda:0", warmup=3, limit_ctx=None):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    if limit_ctx:
        ids = ids[:, -limit_ctx:]
    torch.cuda.synchronize()

    # prefill
    t0 = time.perf_counter()
    out = model(ids, use_cache=True)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    past, logits = out.past_key_values, out.logits[:, -1, :]
    del out

    steps = []          # (token_id, top1_logit, sum_top5, seconds)
    for i in range(gen_tokens):
        tok_id = int(torch.argmax(logits, dim=-1))
        top5 = torch.topk(logits.float(), 5, dim=-1).values[0]
        rec = [tok_id, float(top5[0]), float(top5.sum())]

        t0 = time.perf_counter()
        out = model(torch.tensor([[tok_id]], device=device), past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        rec.append(time.perf_counter() - t0)

        past, logits = out.past_key_values, out.logits[:, -1, :]
        del out
        steps.append(rec)

    times = [s[3] for s in steps]
    usable = times[warmup:] if len(times) > warmup else times
    decode_tps = len(usable) / sum(usable) if usable else 0.0
    return {
        "prompt_tokens": int(ids.shape[1]),
        "ttft_s": round(ttft, 4),
        "prefill_tps": round(ids.shape[1] / ttft, 1) if ttft > 0 else None,
        "decode_tps": round(decode_tps, 2),
        "decode_ms_per_tok": round(1000.0 / decode_tps, 3) if decode_tps else None,
        "step_ms": [round(t * 1000, 2) for t in times],
        "tokens": [s[0] for s in steps],
        "logits": [[s[1], round(s[2], 4)] for s in steps],
        "text": tok.decode([s[0] for s in steps]),
    }


@torch.no_grad()
def gpu_time_per_token(model, tok, prompt, steps=5, device="cuda:0"):
    """Sum of CUDA kernel durations per decode step (GPU busy time).

    Necessary on this box because wall time is dominated by kernel-launch
    overhead, so tok/s says nothing about the weight format's bandwidth cost.
    """
    from torch.profiler import profile, ProfilerActivity

    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model(ids, use_cache=True)
    torch.cuda.synchronize()
    past, logits = out.past_key_values, out.logits[:, -1, :]
    del out

    nid = torch.argmax(logits, dim=-1, keepdim=True)
    for _ in range(3):  # warm
        out = model(nid, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        past, logits = out.past_key_values, out.logits[:, -1, :]
        nid = torch.argmax(logits, dim=-1, keepdim=True)
        del out

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            out = model(nid, past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
            nid = torch.argmax(logits, dim=-1, keepdim=True)
            del out
        torch.cuda.synchronize()

    ka = prof.key_averages()
    total_us = sum(e.self_device_time_total for e in ka)
    detail = sorted(((e.key[:48], round(e.self_device_time_total / steps / 1000, 3), e.count // steps)
                     for e in ka if e.self_device_time_total > 0),
                    key=lambda x: -x[1])[:8]
    return {
        "gpu_ms_per_token": round(total_us / steps / 1000, 3),
        "gpu_tps_upper_bound": round(steps / (total_us / 1e6), 2) if total_us else None,
        "gpu_kernel_ms_top": detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["bf16", "df11", "nf4"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--gpu-profile-steps", type=int, default=5)
    args = ap.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = open(args.prompt_file, encoding="utf-8").read()
    if not prompt:
        prompt = (
            "The following is a description of a memory-bandwidth-bound inference system. "
            "Decode-phase token generation reads every model weight once per token, so the "
            "achievable tokens per second is the memory bandwidth divided by the number of "
            "bytes read per token. Explain, in precise technical terms, what happens to "
            "throughput when weights are stored in a losslessly compressed form and "
            "reconstructed just before each matrix multiplication."
        )

    log = {"kind": args.kind, "model": args.model, "gen_tokens": args.gen_tokens}
    dev_index = int(args.device.split(":")[-1]) if ":" in args.device else 0

    sampler = NvSampler(index=dev_index)
    sampler.start()
    time.sleep(1.0)
    idle = sampler.samples[-1][1] if sampler.samples else None
    log["vram_idle_mib"] = idle
    log["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda}

    model, tok, load_s = load_model(args.kind, args.model, args.device)
    log["load_seconds"] = round(load_s, 2)

    torch.cuda.synchronize()
    log["resident_bytes_by_class"] = resident_accounting(model)[0]
    log["bytes_offloaded_to_cpu"] = resident_accounting(model)[1]
    log["resident_total_gib"] = round(sum(log["resident_bytes_by_class"].values()) / 1024 ** 3, 4)
    log["logical_weight_elements"] = n_params_effective(model)

    log["torch_after_load"] = {
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024 ** 3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024 ** 3, 3),
    }
    log["vram_after_load_mib"] = sampler.samples[-1][1] if sampler.samples else None

    torch.cuda.reset_peak_memory_stats()
    if not args.skip_generate:
        bench = run_bench(model, tok, prompt, args.gen_tokens, args.device, args.warmup)
        log["bench"] = bench
        if args.gpu_profile_steps:
            log["gpu"] = gpu_time_per_token(model, tok, prompt, args.gpu_profile_steps, args.device)

    log["torch_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
    log["torch_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    sampler.stop()
    sampler.join(timeout=5)
    log["vram_sampler"] = sampler.summary()
    if idle is not None and sampler.summary():
        log["vram_peak_delta_vs_idle_mib"] = sampler.summary()["max_used_mib"] - idle

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[bench_infer] wrote {args.out}")
    print(json.dumps({k: v for k, v in log.items()
                      if k not in ("bench", "resident_bytes_by_class")}, indent=2))
    if "bench" in log:
        print(f"[bench_infer] decode_tps={log['bench']['decode_tps']} "
              f"ttft={log['bench']['ttft_s']}s resident={log['resident_total_gib']} GiB")
        if "gpu" in log:
            print(f"[bench_infer] gpu_ms_per_token={log['gpu']['gpu_ms_per_token']} "
                  f"(gpu-bound upper bound {log['gpu']['gpu_tps_upper_bound']} tok/s)")


if __name__ == "__main__":
    main()
