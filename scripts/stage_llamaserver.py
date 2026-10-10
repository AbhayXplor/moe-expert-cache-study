#!/usr/bin/env python
"""
stage_llamaserver.py -- make Ollama's bundled llama-server usable directly.

Ollama ships `llama-server.exe` (plus libllama/ggml-base/...) in
`%LOCALAPPDATA%\\Programs\\Ollama\\lib\\ollama`, but the CUDA backend
(`ggml-cuda.dll`, cublas) in `...\\lib\\ollama\\cuda_v12`.  ggml's dynamic backend
loader only scans the executable's own directory, so running the binary in place
silently falls back to the CPU:

    warning: no usable GPU found, --gpu-layers option will be ignored

which looks exactly like a slow GPU run unless you read the log.  Hardlinks cost
no disk space and need no administrator rights, so we stage both directories into
a single runtime directory next to the executable.

Verify with:  .runtime/llamaserver/llama-server.exe --list-devices
Expected:     CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (...)
"""
import glob
import os
import sys

OLLAMA_LIB = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\lib\ollama")
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(WORKSPACE, ".runtime", "llamaserver")

SOURCES = [OLLAMA_LIB, os.path.join(OLLAMA_LIB, "cuda_v12")]


def main():
    if not os.path.isdir(OLLAMA_LIB):
        sys.exit(f"Ollama lib directory not found: {OLLAMA_LIB}")
    os.makedirs(DEST, exist_ok=True)
    made = skipped = 0
    for src_dir in SOURCES:
        for path in (sorted(glob.glob(os.path.join(src_dir, "*.dll")))
                     + sorted(glob.glob(os.path.join(src_dir, "*.exe")))):
            target = os.path.join(DEST, os.path.basename(path))
            if os.path.exists(target):
                skipped += 1
                continue
            os.link(path, target)
            made += 1
    print(f"staged {made} new, {skipped} already present, "
          f"{len(os.listdir(DEST))} files in {DEST}")

    server = os.path.join(DEST, "llama-server.exe")
    if not os.path.exists(server):
        sys.exit(f"staging failed: {server} is missing")


if __name__ == "__main__":
    main()
