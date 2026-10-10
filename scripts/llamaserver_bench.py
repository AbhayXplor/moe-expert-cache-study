#!/usr/bin/env python
"""
llamaserver_bench.py -- measure the *bundled* llama-server directly.

Why: Ollama hides most of llama.cpp's switches.  The binary it ships
(lib/ollama/llama-server.exe) can do speculative decoding, projectors, KV types
and offload control that Ollama's own CLI never exposes.  To know whether those
are worth anything on this card, we have to drive the binary ourselves.

It launches one server per configuration, waits for /health, runs each prompt
through /completion, and reports the server's own timings.

Usage:
  python llamaserver_bench.py --configs baseline ngram
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

# Ollama ships llama-server.exe in lib/ollama but the CUDA backend
# (ggml-cuda.dll + cublas) in lib/ollama\cuda_v12, and ggml's dynamic backend
# loader only scans the executable's own directory.  Run in place and it
# silently falls back to the CPU ("no usable GPU found").  We stage hardlinks
# of both directories into .runtime/llamaserver so CUDA0 is found.
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(WORKSPACE, ".runtime", "llamaserver", "llama-server.exe")
BLOB = os.path.expanduser(
    r"~\.ollama\models\blobs"
    r"\sha256-02d45dc1cf451ba2475ac33b301c2dd8f985abe4c182ce04a1f2f5bf0260278d")

# A prompt whose answer naturally quotes the input back (where n-gram draft
# should shine) and one that does not (where it should not).
PROMPTS = {
    "freeform": (
        "Explain in detail how the KV cache grows with context length and why "
        "that limits the largest model that fits in VRAM. Be thorough."
    ),
    "copy-heavy": (
        "Here is a changelog entry. Rewrite it as a bullet list, keeping the "
        "wording as close to the original as possible.\n\n"
        "Release 2.4.0 adds support for quantized KV caches, fixes a crash when "
        "the context is shifted during generation, reduces peak VRAM by 12% on "
        "long contexts, and improves prefill throughput on hybrid SSM models. "
        "It also deprecates the legacy attention kernel, which was slower on "
        "cards with less than 12 GB of VRAM, and adds an n-gram speculative "
        "decoder that requires no additional model. Users on 8 GB cards should "
        "see the largest improvement, because prefill and decode both benefit."
    ),
    # Highly structured output: the model repeats the same keys over and over,
    # which is exactly where a self-lookup draft should pay off.
    "repetitive": (
        "Output a JSON array of 25 objects. Every object must have exactly the "
        "same six keys in the same order: id, name, category, status, "
        "latency_ms, retries. Use plausible placeholder values. No prose, "
        "just the JSON array."
    ),
}

# Defaults for ngram-simple are size-n=12 (lookup length) / size-m=48 (draft
# length) / min-hits=1 -- a 12-token lookup almost never fires on novel prose.
# Shorter lookups fire more often, so sweep them.
CONFIGS = {
    "baseline": [],
    "ngram-n12": ["--spec-type", "ngram-simple"],
    "ngram-n6": ["--spec-type", "ngram-simple",
                 "--spec-ngram-simple-size-n", "6",
                 "--spec-ngram-simple-size-m", "24"],
    "ngram-n4": ["--spec-type", "ngram-simple",
                 "--spec-ngram-simple-size-n", "4",
                 "--spec-ngram-simple-size-m", "16"],
    "mapk-n4": ["--spec-type", "ngram-map-k",
                "--spec-ngram-map-k-size-n", "4",
                "--spec-ngram-map-k-size-m", "16"],
    "mod-m8": ["--spec-type", "ngram-mod",
               "--spec-ngram-mod-n-match", "8",
               "--spec-ngram-mod-n-min", "4",
               "--spec-ngram-mod-n-max", "16"],
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


def completion(port, prompt, n_predict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        # temperature 0 so every configuration must produce the *same* tokens:
        # speculative decoding may not change quality, and this proves it here.
        data=json.dumps({"prompt": prompt, "n_predict": n_predict,
                         "temperature": 0.0, "seed": 0,
                         "cache_prompt": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode())


def main():
    results = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--ngl", type=int, default=33)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--mmproj", action="store_true",
                    help="also load the vision projector (Ollama always does)")
    args = ap.parse_args()

    if not os.path.exists(SERVER):
        sys.exit(f"llama-server not found at {SERVER}")

    for name in args.configs:
        extra = CONFIGS[name]
        cmd = [SERVER, "--model", BLOB, "--port", str(args.port),
               "--host", "127.0.0.1", "--no-webui", "--offline",
               "-c", str(args.num_ctx), "-ngl", str(args.ngl),
               "-np", "1", "-b", "512", "-ub", "512",
               "--flash-attn", "auto", "--no-jinja",
               "--chat-template", "chatml"] + extra
        if args.mmproj:
            cmd += ["--mmproj", os.path.expanduser(
                r"~\.ollama\models\blobs"
                r"\sha256-f836f08f921193f4d6b6a6952dba0a6fb116759d26a757bcb395d570d73976ca")]
        print(f"\n{'='*74}\nconfig={name}  extra={extra}  ctx={args.num_ctx}"
              f"  ngl={args.ngl}  mmproj={args.mmproj}", flush=True)

        logpath = os.path.join("experiments", "logs",
                               f"llamaserver_{name}.log")
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
        logf = open(logpath, "wb")
        env = dict(os.environ)
        # Keep every run on one GPU and ignore any stale profile state.
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env.pop("GGML_BACKEND_PATH", None)
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                env=env,
                                creationflags=getattr(subprocess,
                                                      "CREATE_NEW_PROCESS_GROUP", 0))
        try:
            if not wait_health(args.port, time.time() + 240):
                print("  server did not become healthy", flush=True)
                continue
            print(f"  load      {time.perf_counter()-t0:6.2f} s", flush=True)
            for pname, prompt in PROMPTS.items():
                r = completion(args.port, prompt, args.n_predict)
                t = r.get("timings", {})
                text = r.get("content", "")
                digest = hashlib.sha256(text.encode()).hexdigest()[:12]
                print(f"  {pname:11s} decode {t.get('predicted_per_second', 0):6.2f} tok/s"
                      f"   prefill {t.get('prompt_per_second', 0):8.1f} tok/s"
                      f"   n {t.get('predicted_n')}"
                      f"   draft_n {t.get('draft_n', 0)}"
                      f"   draft_acc {t.get('draft_n_accepted', 0)}"
                      f"   sha {digest}", flush=True)
                results.append({"config": name, "prompt": pname,
                                "decode_tps": t.get("predicted_per_second"),
                                "prefill_tps": t.get("prompt_per_second"),
                                "n": t.get("predicted_n"),
                                "draft_n": t.get("draft_n"),
                                "draft_acc": t.get("draft_n_accepted"),
                                "sha": digest})
        finally:
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                pass
            time.sleep(1)
            proc.kill()
            proc.wait(timeout=30)
            logf.close()
            with open(logpath, "rb") as fh:
                blob = fh.read()
            tail = [l for l in blob.split(b"\n")
                    if b"offload" in l.lower() or b"usable GPU" in l][-4:]
            for l in tail:
                print("  " + l.decode("utf-8", "replace")[:150], flush=True)
            time.sleep(4)

    out = os.path.join(WORKSPACE, "experiments", "runs", "llamaserver.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
