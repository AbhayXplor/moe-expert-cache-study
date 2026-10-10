#!/usr/bin/env python
"""
moe_bench_35b.py -- stopwatch a large MoE on an 8 GB card, safely.

Why this exists
---------------
Every 30-40 tok/s figure for a 35B-A3B model on a laptop 8 GB card is a
community report or arithmetic.  This measures it on *this* machine, and it
sweeps how much of the expert weight lives in VRAM vs system RAM, because that
is the only knob that matters for a MoE that does not fit.

Memory safety (the machine has 64 GB and is already ~25 GB in use)
------------------------------------------------------------------
The model is memory-mapped, which is llama.cpp's default: the bulk of the
weights stay as evictable page cache, never process heap.  This script never
passes --no-mmap and never passes --mlock, because either one turns a 22 GB
model into 22 GB of committed RAM.  A watchdog thread samples system RAM every
2 s and hard-kills the server if in-use memory crosses a ceiling, so a bad
configuration cannot take the laptop down.

Usage
-----
  python moe_bench_35b.py --model models/.../X.gguf \
      --configs cpu-moe ncmoe-36 ncmoe-32 --reps 2
"""
import argparse
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(WORKSPACE, ".runtime", "llamaserver", "llama-server.exe")
# Official upstream prebuilt (no compiler needed).  As of build b11537 this one
# carries the mainline MoE expert cache, --moe-cache-mib, which the Ollama-derived
# binary does not have at all.
SERVER_OFFICIAL = os.path.join(WORKSPACE, ".runtime", "official", "cuda",
                               "llama-server.exe")

# --- kill switches -------------------------------------------------------
RAM_CEILING_GB = 52.0      # of 63.7 GB total; leaves ~11 GB of headroom
RAM_POLL_S = 2.0


class _MemStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def ram_gb():
    """(used, total) system RAM in GiB, from the OS, no subprocess."""
    m = _MemStatus()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    gib = 1024.0 ** 3
    return (m.ullTotalPhys - m.ullAvailPhys) / gib, m.ullTotalPhys / gib


def gpu_mib():
    """Free/total VRAM in MiB, or (None, None) if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        used, total = out.splitlines()[0].split(",")
        return int(used), int(total)
    except Exception:
        return None, None


class Watchdog(threading.Thread):
    """Kill the server before the machine runs out of memory."""

    def __init__(self, proc, ceiling_gb):
        super().__init__(daemon=True)
        self.proc = proc
        self.ceiling = ceiling_gb
        self.peak = 0.0
        self.tripped = False
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            used, _ = ram_gb()
            self.peak = max(self.peak, used)
            if used > self.ceiling:
                self.tripped = True
                print(f"\n  !! RAM {used:.1f} GB > ceiling "
                      f"{self.ceiling:.0f} GB -- killing server", flush=True)
                try:
                    self.proc.kill()
                except Exception:
                    pass
                return
            self._stop.wait(RAM_POLL_S)

    def stop(self):
        self._stop.set()


# Prompt set: one prose-heavy, one code, one in the user's actual domain.
PROMPTS = {
    "prose": (
        "Explain why reading weights from system RAM is roughly 80 times more "
        "expensive per byte than reading them from GPU memory, and what that "
        "implies for choosing where a mixture-of-experts model's weights should "
        "live. Be concrete and thorough."
    ),
    "code": (
        "Write a Python function that takes a list of (timestamp, latency_ms) "
        "samples and returns the p50, p95, p99 and a rolling 60-second average. "
        "Include type hints and a docstring. Then explain any correctness trap."
    ),
    "security": (
        "You have a Linux host you are authorised to test. A service on TCP "
        "8443 is running an unknown TLS stack and returns a certificate with "
        "an unrecognised issuer. Lay out a step-by-step, safe methodology to "
        "fingerprint the service, identify the daemon behind it, and report "
        "findings, naming the specific tools and the reasoning at each step."
    ),
    # Structured, highly repetitive output.  This is the only shape where a
    # self-lookup drafter (ngram-*) can fire at all: it drafts by finding what
    # followed the current n-gram earlier in the context.  Novel prose gives it
    # nothing to look up, so a null result on the other three prompts is the
    # expected outcome, not a broken test.
    "repeat": (
        "Output a JSON array of 25 objects. Every object must have exactly "
        "the same six keys in the same order: id, name, category, status, "
        "latency_ms, retries. Use plausible placeholder values. No prose, "
        "just the JSON array."
    ),
    # ~8k tokens of context.  Prompt processing on an expert-offloaded MoE is
    # the expensive half of a long-conversation turn, and short prompts hide it
    # behind fixed per-request overhead, so it gets its own measurement.
    "longprefill": (
        "Context material:\n" +
        "The system reads expert weights from system memory and the transfer "
        "cost dominates the design. " * 300 +
        "\n\nIn one sentence, state what dominates the cost."
    ),
}


def wait_health(port, deadline):
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


# Extra request-body fields (e.g. chat_template_kwargs to switch the model's
# thinking mode off).  Set from --extra-body.
CHAT_EXTRA = {}


def chat(port, prompt, n_predict, timeout=3600):
    """One chat completion.

    Returns (content, reasoning, timings, wall).  A reasoning model emits its
    scratchpad separately: llama-server puts it in `reasoning_content` and
    leaves `content` empty until thinking ends.  Reporting only `content`
    silently produces empty answers, so both are captured.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "seed": 0, "max_tokens": n_predict,
        "cache_prompt": False, "stream": False,
    }
    payload.update(CHAT_EXTRA)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    wall = time.perf_counter() - t0
    msg = (out.get("choices") or [{}])[0].get("message", {}) or {}
    return (msg.get("content") or "", msg.get("reasoning_content") or "",
            out.get("timings", {}), wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2,
                    help="passes over the prompt set; the last one is reported "
                         "separately because disk/page-cache warm-up inflates "
                         "the first")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--ram-ceiling", type=float, default=RAM_CEILING_GB)
    ap.add_argument("--tag", default="qwen36_35b_a3b")
    ap.add_argument("--server", default=None,
                    help="override the llama-server binary to launch")
    ap.add_argument("--extra-body", default=None,
                    help="JSON merged into every chat request body, e.g. "
                         "'{\"chat_template_kwargs\":{\"enable_thinking\":false}}'")
    ap.add_argument("--prompts", nargs="+", default=None,
                    help="subset of prompt names to run (default: all). Used "
                         "to point ngram-speculation tests at the repetitive "
                         "prompt, where a self-lookup drafter can fire.")
    args = ap.parse_args()

    unknown_prompts = [p for p in (args.prompts or []) if p not in PROMPTS]
    if unknown_prompts:
        sys.exit(f"unknown prompt(s): {unknown_prompts}\nknown: {list(PROMPTS)}")
    prompts = (PROMPTS if not args.prompts else
               {k: v for k, v in PROMPTS.items() if k in args.prompts})

    global CHAT_EXTRA
    if args.extra_body:
        CHAT_EXTRA = json.loads(args.extra_body)

    # name -> expert-placement arguments.
    #
    # -ot exps=CPU keeps *only* the expert tensors in RAM; attention/SSM/router/
    # head stay on the GPU, because they are read every single token and there
    # is no reason to make them slow.  That is the baseline.
    #
    # --n-cpu-moe N instead moves N layers' worth of expert weights to RAM and
    # leaves the rest resident in VRAM.  Do NOT combine the two: `-ot exps=CPU`
    # already claims *every* expert tensor, so adding --n-cpu-moe on top of it
    # pins everything to the CPU and the sweep would look flat for the wrong
    # reason.  ncmoe-40 is therefore the control: it should reproduce cpu-moe.
    CONFIGS = {
        # control: no offload arguments at all (only usable for a model that
        # fits in VRAM, e.g. the 9B, so the two models can be compared through
        # exactly the same harness)
        "none": [],
        "cpu-moe": ["-ot", "exps=CPU"],
        # thread-count variations on the expert-in-RAM path: if the CPU half of
        # the MoE is what sets the pace, this is the knob that moves it
        "t4": ["-ot", "exps=CPU", "-t", "4"],
        "t8": ["-ot", "exps=CPU", "-t", "8"],
        "t12": ["-ot", "exps=CPU", "-t", "12"],
        "t16": ["-ot", "exps=CPU", "-t", "16"],
        # Bigger batch: prompt processing cost, larger compute buffer.
        #
        # Why this matters more than it looks: llama.cpp will not hand a prompt
        # batch to the GPU until it reaches GGML_OP_OFFLOAD_MIN_BATCH, which is
        # 32 * total_experts / active_experts.  For a 256-expert top-8 MoE that
        # is 32 * 256/8 = 1024 tokens.  Below it, prompt processing runs on the
        # CPU -- which is the wrong place for it.  So the micro-batch has to
        # clear 1024 before GPU prompt processing engages at all.
        "ub2048": ["-ot", "exps=CPU", "-b", "2048", "-ub", "2048"],
        "ub4096": ["-ot", "exps=CPU", "-b", "4096", "-ub", "4096"],
        "ub8192": ["-ot", "exps=CPU", "-b", "8192", "-ub", "8192"],
        # MTP: multi-token prediction built into the model.  Towards llama.cpp
        # this is the `--spec-type` draft head (renamed from --spec-type mtp in
        # May 2026); the model file must carry the MTP head.
        "mtp": ["-ot", "exps=CPU", "-b", "2048", "-ub", "2048",
                "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        # The configuration actually recommended for daily use: the winning
        # batch size and MTP together.  Measured separately they do not
        # interact (MTP is decode-only, the batch is prefill-only), but the
        # combination is what gets handed over, so it gets exercised directly.
        "mtp4096": ["-ot", "exps=CPU", "-b", "4096", "-ub", "4096",
                    "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        # --- MoE expert cache (mainline --moe-cache-mib, needs the official build).
        #
        # This is the one lever the whole research pass pointed at: the expert
        # weights stay in host RAM, but a fixed pool of N MiB of expert slots
        # lives in VRAM and is reused between tokens instead of being copied in
        # and thrown away every step.  Two independent community reports put it
        # at 1.7-2.1x on *this* model class (Qwen3.6-35B-A3B, 8 GB card).
        #
        # Budget arithmetic before running: baseline non-expert + KV + compute
        # is ~3200 MiB of 8188, so ~4900 MiB is free, but a known trap is that
        # under ~800 MiB free at max context the WDDM spill turns decode into a
        # crawl.  So the sweep stops well short of the wall on purpose.
        # NOTE: the cache configs use the same '-ot exps=CPU' flag form as the
        # baseline, so the cache size is the ONLY difference against 'cpu-moe'.
        # Using '--cpu-moe' instead would change two things at once.
        "cache-512": ["-ot", "exps=CPU", "--moe-cache-mib", "512"],
        "cache-1024": ["-ot", "exps=CPU", "--moe-cache-mib", "1024"],
        "cache-2048": ["-ot", "exps=CPU", "--moe-cache-mib", "2048"],
        "cache-3072": ["-ot", "exps=CPU", "--moe-cache-mib", "3072"],
        "cache-4096": ["-ot", "exps=CPU", "--moe-cache-mib", "4096"],
        # Where does the pool stop paying?  On REAP-48 (133 experts, ~192 MB of
        # experts per layer) the 3072 -> 4096 step was worth +33%, so the curve
        # had not flattened yet.  This one probes just above it while keeping
        # >=1 GB of VRAM free, which is the measured rail on this card.
        "cache-4608": ["-ot", "exps=CPU", "--moe-cache-mib", "4608"],
        "cache-4864": ["-ot", "exps=CPU", "--moe-cache-mib", "4864"],
        "cache-5120": ["-ot", "exps=CPU", "--moe-cache-mib", "5120"],
        "cache-4096-ub4096": ["-ot", "exps=CPU", "--moe-cache-mib", "4096",
                              "-b", "4096", "-ub", "4096"],
        # cache + MTP: the two verified wins stacked, if they compose.
        # MTP needs the *MTP* model file, so run this one with --model
        # models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
        "cache-2048-mtp": ["-ot", "exps=CPU", "--moe-cache-mib", "2048",
                           "-b", "2048", "-ub", "2048",
                           "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        # Why is the official b11537 build ~30% slower than the staged
        # Ollama-derived build on this same model and flags?  -fa is the one
        # flag whose accepted values changed between them, so this isolates it.
        "fa-off": ["-ot", "exps=CPU", "-fa", "off"],
        # --perf makes llama.cpp print where the time actually goes per backend,
        # which is the direct evidence for the "CPU-side expert math is the
        # bottleneck" claim that has so far only been inferred from null results.
        "perf": ["-ot", "exps=CPU", "--perf"],
        # --- Other speculative-decoding families, against MTP.
        #
        # None of these needs a second model file, so they cost ~no VRAM --
        # unlike draft-eagle3 / draft-dflash / draft-dspark, which need a
        # trained draft head we do not have.  They are lookups over context
        # already in memory (ngram-*) or a built-in cheap draft.
        #
        # They can only pay when the output repeats text already present:
        # structured output, code edits, templated text.  On novel prose the
        # lookup finds nothing and acceptance is ~0 -- a result about where
        # they belong, not a broken measurement.  Defaults are n=12/m=48 (a
        # 12-token lookup almost never fires), so the ngram configs sweep
        # shorter lookups too.
        "spec-none": ["-ot", "exps=CPU", "-b", "2048", "-ub", "2048",
                      "--spec-type", "none"],
        "ngram-simple-4": ["-ot", "exps=CPU", "-b", "2048", "-ub", "2048",
                           "--spec-type", "ngram-simple",
                           "--spec-ngram-simple-size-n", "4",
                           "--spec-ngram-simple-size-m", "16"],
        "ngram-simple-12": ["-ot", "exps=CPU", "-b", "2048", "-ub", "2048",
                            "--spec-type", "ngram-simple",
                            "--spec-ngram-simple-size-n", "12",
                            "--spec-ngram-simple-size-m", "48"],
        "ngram-mapk-4": ["-ot", "exps=CPU", "-b", "2048", "-ub", "2048",
                         "--spec-type", "ngram-map-k",
                         "--spec-ngram-map-k-size-n", "4",
                         "--spec-ngram-map-k-size-m", "16"],
        "ngram-mod-8": ["-ot", "exps=CPU", "-b", "2048", "-ub", "2048",
                        "--spec-type", "ngram-mod",
                        "--spec-ngram-mod-n-match", "8",
                        "--spec-ngram-mod-n-min", "4",
                        "--spec-ngram-mod-n-max", "16"],
        "ngram-cache": ["-ot", "exps=CPU", "-b", "2048", "-ub", "2048",
                        "--spec-type", "ngram-cache"],
        "draft-simple": ["-ot", "exps=CPU", "-b", "2048", "-ub", "2048",
                         "--spec-type", "draft-simple",
                         "--spec-draft-n-max", "4"],
        "ncmoe-40": ["--n-cpu-moe", "40"],
        "ncmoe-38": ["--n-cpu-moe", "38"],
        "ncmoe-36": ["--n-cpu-moe", "36"],
        "ncmoe-34": ["--n-cpu-moe", "34"],
        "ncmoe-32": ["--n-cpu-moe", "32"],
        "ncmoe-30": ["--n-cpu-moe", "30"],
        "ncmoe-28": ["--n-cpu-moe", "28"],
        "ncmoe-24": ["--n-cpu-moe", "24"],
        # Finer steps aimed at the REAP-48 model, whose expert tensors are only
        # ~192 MB/layer (133 experts) versus ~486 MB/layer for the full model.
        # That means many more layers of experts fit in 8 GB than for the full
        # 35B, so the interesting placement region is lower down.
        "ncmoe-20": ["--n-cpu-moe", "20"],
        "ncmoe-16": ["--n-cpu-moe", "16"],
        "ncmoe-14": ["--n-cpu-moe", "14"],
        "ncmoe-12": ["--n-cpu-moe", "12"],
        "ncmoe-10": ["--n-cpu-moe", "10"],
        # Combinability test for the REAP model: static GPU placement plus a
        # cache pool on top of it.  They compete for the same VRAM, so this is
        # the question of whether the two wins add or merely overlap.
        "ncmoe-20-c1024": ["--n-cpu-moe", "20", "--moe-cache-mib", "1024"],
    }
    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown config(s): {unknown}\nknown: {list(CONFIGS)}")
    # abspath matters: Windows CreateProcess cannot start an executable given a
    # forward-slash relative path, it fails with a bare WinError 2.
    server = os.path.abspath(args.server or SERVER)
    if not os.path.exists(server):
        sys.exit(f"llama-server not found at {server}")
    print(f"llama-server: {server}", flush=True)
    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")

    used0, total0 = ram_gb()
    print(f"RAM before any load: {used0:.1f} / {total0:.1f} GiB used", flush=True)

    # Keep one bad configuration from ending the sweep: a server that dies
    # mid-request raises out of chat(), and Python would otherwise abort the
    # whole run before anything was written to disk.
    results = []
    for name in args.configs:
        extra = CONFIGS[name]
        cmd = [server, "--model", args.model,
               "--host", "127.0.0.1", "--port", str(args.port),
               "--no-webui", "--offline", "--jinja",
               "-c", str(args.num_ctx), "-np", "1",
               "-b", "2048", "-ub", str(args.ub),
               "-fa", "on", "-ngl", "999", "--metrics"] + extra
        print(f"\n{'='*76}\nconfig={name}  extra={' '.join(extra)}"
              f"  ctx={args.num_ctx}  ub={args.ub}"
              f"  body={CHAT_EXTRA or '{}'}", flush=True)

        logpath = os.path.join(WORKSPACE, "experiments", "logs",
                               f"moe_{args.tag}_{name}.log")
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
        logf = open(logpath, "wb")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env.pop("GGML_BACKEND_PATH", None)

        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                env=env,
                                creationflags=getattr(
                                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        dog = Watchdog(proc, args.ram_ceiling)
        dog.start()
        # Provenance: earlier artifacts did not record which binary or which
        # model file produced them, which made the cache A/B unverifiable after
        # the fact.  Record both, always.
        entry = {"config": name, "extra": extra, "ctx": args.num_ctx,
                 "ub": args.ub, "server": server, "model": args.model,
                 "prompts": [], "killed": False}
        try:
            if not wait_health(args.port, time.time() + 600):
                print("  server never became healthy", flush=True)
                entry["killed"] = dog.tripped
                entry["error"] = "no /health"
                results.append(entry)
                continue
            load_s = time.perf_counter() - t0
            vused, vtotal = gpu_mib()
            rused, _ = ram_gb()
            print(f"  load      {load_s:6.1f} s"
                  f"   VRAM {vused}/{vtotal} MiB   RAM used {rused:.1f} GiB",
                  flush=True)
            entry.update({"load_s": round(load_s, 2), "vram_used_mib": vused,
                          "ram_used_gib": round(rused, 2)})

            for rep in range(args.reps):
                rep_out = {}
                for pname, prompt in prompts.items():
                    try:
                        text, reasoning, t, wall = chat(args.port, prompt,
                                                        args.n_predict)
                    except Exception as exc:
                        # A configuration can kill the server (watchdog trip, or
                        # a batch big enough to exhaust memory).  That is a
                        # *result* -- record it and move to the next config
                        # rather than losing the whole sweep.
                        print(f"    rep{rep} {pname:9s} FAILED: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                        rep_out[pname] = {"error": f"{type(exc).__name__}: {exc}"}
                        break
                    tps = t.get("predicted_per_second") or 0.0
                    n = t.get("predicted_n") or 0
                    pre = t.get("prompt_per_second") or 0.0
                    if not tps and n:
                        tps = n / wall
                    # draft_* are only meaningful when a speculative decoder is
                    # active; they are the proof that MTP is drafting and that
                    # the drafts are being accepted, rather than the numbers
                    # simply looking better for another reason.
                    dn = t.get("draft_n") or 0
                    da = t.get("draft_n_accepted") or 0
                    print(f"    rep{rep} {pname:9s} decode {tps:6.2f} tok/s"
                          f"  prefill {pre:7.1f} tok/s  n={n}"
                          f"  wall {wall:5.2f}s"
                          f"  draft {da}/{dn}"
                          f"  answer_chars={len(text)}", flush=True)
                    rep_out[pname] = {
                        "decode_tps": round(tps, 3), "prefill_tps": round(pre, 1),
                        "n": n, "wall_s": round(wall, 2),
                        "sha": hashlib.sha256(text.encode()).hexdigest()[:12],
                        "text": text, "reasoning": reasoning,
                        "thinking_chars": len(reasoning),
                        "answer_chars": len(text),
                        "draft_n": dn, "draft_n_accepted": da,
                    }
                entry["prompts"].append(rep_out)
                vu, _ = gpu_mib()
                ru, _ = ram_gb()
                print(f"    rep{rep} peak-ish VRAM {vu} MiB  RAM {ru:.1f} GiB",
                      flush=True)
                entry[f"vram_mib_rep{rep}"] = vu
                entry[f"ram_gib_rep{rep}"] = round(ru, 2)
        finally:
            dog.stop()
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                pass
            time.sleep(1)
            try:
                proc.kill()
                proc.wait(timeout=30)
            except Exception:
                pass
            logf.close()
            entry["ram_peak_gib"] = round(dog.peak, 2)
            entry["watchdog_tripped"] = dog.tripped
            if dog.tripped:
                entry["killed"] = True
            # Surface the offload summary llama.cpp printed at load time.
            try:
                with open(logpath, "rb") as fh:
                    lines = fh.read().split(b"\n")
                keep = [l for l in lines if any(
                    k in l.lower() for k in
                    (b"offload", b"buffer size", b"out of memory",
                     b"failed", b"n_cpu_moe", b"cuda", b"cache", b"expert"))]
                entry["log_tail"] = [l.decode("utf-8", "replace")[:200]
                                     for l in keep[-12:]]
                for l in entry["log_tail"]:
                    print("      | " + l, flush=True)
            except Exception:
                pass
            results.append(entry)
            time.sleep(5)

    out = os.path.join(WORKSPACE, "experiments", "runs",
                       f"moe_bench_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
