"""Wait until the GPU is genuinely idle before benchmarking.

Ollama picks the GPU/CPU offload split from the free VRAM it sees AT LOAD TIME
(and can fall back to cached values when its own query times out).  So a
benchmark that starts while a previous model is still unloading measures the
unloading, not the model.  This is the control that was missing.
"""
import subprocess, sys, time

def used():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return int(o.splitlines()[0].strip())

target = int(sys.argv[1]) if len(sys.argv) > 1 else 700
deadline = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 120)
print(f"waiting for VRAM < {target} MiB (now {used()} MiB)", flush=True)
while time.time() < deadline:
    u = used()
    if u <= target:
        print(f"VRAM drained to {u} MiB", flush=True)
        break
    time.sleep(3)
else:
    print(f"WARNING: VRAM still {used()} MiB after timeout", flush=True)
